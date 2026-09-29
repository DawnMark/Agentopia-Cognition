"""Idea event stream and views (design doc 5.2 / 10, phase 3).

    cognition/idea_events.jsonl        one stream per persona
    cognition/views/ideas.json         derived, deletable, rebuildable

Same contract as the other cognition streams: phase 0 identity, week-scoped
idempotency keys, and a pure fold for the view. An idea is only ever *created*
here; anything that would change capability (a candidate methodology) is written
to the capability stream by the orchestrator, never to the idea stream.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from src.agents.cognition.idea_models import IDEA_EVENT_TYPES, Idea

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.agents.data_manager import DataManager

IDEAS_VIEW_FILENAME = "ideas.json"


class IdeaEventStore:
    """`idea_events.jsonl` reader/writer."""

    def __init__(self, dm: "DataManager") -> None:
        self.dm = dm
        self.path: Path = dm.root / "cognition" / "idea_events.jsonl"
        self._keys: Optional[Set[str]] = None

    def append(
        self,
        event_type: str,
        *,
        idempotency_key: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], bool]:
        if event_type not in IDEA_EVENT_TYPES:
            raise ValueError(f"unknown idea event type: {event_type!r}")
        if idempotency_key in self.idempotency_keys():
            return {"type": event_type, "idempotency_key": idempotency_key, "duplicate": True}, False

        record: Dict[str, Any] = {"type": event_type, "idempotency_key": idempotency_key}
        for key, value in (payload or {}).items():
            if key in ("time", "ledger_event_id", "schema_version", "idempotency_key"):
                continue
            record[key] = value
        self.dm.append_ledger_record(self.path, record, idempotency_key=idempotency_key)
        if self._keys is not None:
            self._keys.add(idempotency_key)
        return record, True

    # -- convenience wrappers ---------------------------------------------
    def skipped(
        self,
        *,
        week: str,
        reason: str,
        detail: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], bool]:
        """The pipeline ran but produced nothing this week (observability).

        Before this event the only trace of "no idea budget" was a log line, so
        a run where the engine stopped halfway through the year looked
        identical to a run where nothing was worth learning (run 09261721).
        """
        return self.append(
            "IDEA_SKIPPED",
            idempotency_key=f"IDEA_SKIPPED:{week}:{reason}",
            payload={"week": week, "reason": reason, **(detail or {})},
        )

    def created(self, idea: Idea, *, week: str):
        payload = idea.to_dict()
        payload["week"] = week
        return self.append(
            "IDEA_CREATED",
            idempotency_key=f"IDEA_CREATED:{idea.idea_id}:{week}",
            payload=payload,
        )

    def rejected(
        self,
        *,
        idea_id: str,
        week: str,
        reasons: List[str],
        candidate: Dict[str, Any],
        potential: Optional[float] = None,
        score_components: Optional[Dict[str, Any]] = None,
        feasibility: Optional[float] = None,
    ):
        """Rejections are recorded too: they are the negative half of novelty.

        Diagnostics (`potential`, `score_components`, `feasibility`) are stored
        with the rejection: without them a run where nothing is stored gives no
        way to tell "nothing was worth learning" from "the gate is broken"
        (run 09261721).
        """
        payload: Dict[str, Any] = {
            "idea_id": idea_id,
            "week": week,
            "reasons": list(reasons),
            "candidate": candidate,
        }
        if potential is not None:
            payload["potential"] = round(float(potential), 6)
        if score_components:
            payload["score_components"] = dict(score_components)
        if feasibility is not None:
            payload["feasibility"] = round(float(feasibility), 4)
        return self.append(
            "IDEA_REJECTED",
            idempotency_key=f"IDEA_REJECTED:{idea_id}:{week}",
            payload=payload,
        )

    def refined(
        self,
        *,
        idea_id: str,
        week: str,
        duplicate_of: str,
        candidate: Optional[Dict[str, Any]] = None,
    ):
        """The same insight was proposed again (KI-5).

        Week-independent key: one refinement record per (idea, original), so a
        weekly re-derivation does not grow the stream without bound.
        """
        return self.append(
            "IDEA_REFINED",
            idempotency_key=f"IDEA_REFINED:{idea_id}:{duplicate_of}",
            payload={
                "idea_id": idea_id,
                "week": week,
                "duplicate_of": duplicate_of,
                "candidate": candidate or {},
            },
        )

    def scored(self, *, idea_id: str, week: str, potential: float, components: Dict[str, Any]):
        return self.append(
            "IDEA_SCORED",
            idempotency_key=f"IDEA_SCORED:{idea_id}:{week}",
            payload={
                "idea_id": idea_id,
                "week": week,
                "potential": potential,
                "score_components": components,
            },
        )

    def converted(self, *, idea_id: str, week: str, method_id: str):
        return self.append(
            "IDEA_CONVERTED",
            idempotency_key=f"IDEA_CONVERTED:{idea_id}",
            payload={"idea_id": idea_id, "week": week, "candidate_method_id": method_id},
        )

    # -- reading -----------------------------------------------------------
    def events(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        out: List[Dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{self.path.as_posix()}: invalid idea event: {line}") from e
        return out

    def of_type(self, *event_types: str) -> List[Dict[str, Any]]:
        wanted = set(event_types)
        return [e for e in self.events() if e.get("type") in wanted]

    def idempotency_keys(self) -> Set[str]:
        if self._keys is None:
            self._keys = {
                str(e["idempotency_key"])
                for e in self.events()
                if isinstance(e.get("idempotency_key"), str)
            }
        return self._keys

    def invalidate(self) -> None:
        self._keys = None


def materialize_ideas(
    idea_events: List[Dict[str, Any]],
    capability_events: Optional[List[Dict[str, Any]]] = None,
    *,
    persona: str = "",
) -> Dict[str, Any]:
    """Fold the idea stream (plus conversions in the capability stream) into a view."""
    ideas: Dict[str, Dict[str, Any]] = {}
    rejected: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    by_status: Dict[str, int] = {}
    by_motif: Dict[str, int] = {}
    by_type: Dict[str, int] = {}
    by_event_type: Dict[str, int] = {}

    for event in idea_events:
        event_type = str(event.get("type") or "")
        by_event_type[event_type] = by_event_type.get(event_type, 0) + 1

        if event_type == "IDEA_CREATED":
            idea_id = str(event.get("idea_id") or "")
            if not idea_id:
                continue
            payload = {
                k: v
                for k, v in event.items()
                if k
                not in ("type", "idempotency_key", "week", "time", "ledger_event_id", "schema_version")
            }
            try:
                ideas[idea_id] = Idea.from_dict(payload).to_dict()
            except (KeyError, ValueError):
                continue
            ideas[idea_id]["week"] = event.get("week")
            ideas[idea_id]["converted_method_id"] = None

        elif event_type == "IDEA_SKIPPED":
            skipped.append(
                {
                    "week": event.get("week"),
                    "reason": event.get("reason"),
                    "detail": {
                        k: v
                        for k, v in event.items()
                        if k
                        not in (
                            "type",
                            "week",
                            "reason",
                            "idempotency_key",
                            "time",
                            "ledger_event_id",
                            "schema_version",
                        )
                    },
                }
            )

        elif event_type == "IDEA_REFINED":
            # The same insight re-derived (KI-5): keep the original idea, note
            # that it was proposed again, and never let it spawn a second idea.
            idea_id = str(event.get("duplicate_of") or event.get("idea_id") or "")
            if idea_id in ideas:
                refinements = ideas[idea_id].setdefault("refinements", [])
                refinements.append(
                    {
                        "week": event.get("week"),
                        "reasons": list(event.get("reasons") or []),
                    }
                )

        elif event_type == "IDEA_REJECTED":
            rejected.append(
                {
                    "idea_id": event.get("idea_id"),
                    "week": event.get("week"),
                    "reasons": list(event.get("reasons") or []),
                    "motif": (event.get("candidate") or {}).get("motif"),
                    "potential": event.get("potential"),
                    "feasibility": event.get("feasibility"),
                }
            )

        elif event_type == "IDEA_CONVERTED":
            idea_id = str(event.get("idea_id") or "")
            if idea_id in ideas:
                ideas[idea_id]["converted_method_id"] = event.get("candidate_method_id")

        elif event_type in ("IDEA_TESTED", "IDEA_ADOPTED", "IDEA_EXPIRED"):
            idea_id = str(event.get("idea_id") or "")
            if idea_id in ideas:
                ideas[idea_id]["status"] = {
                    "IDEA_TESTED": "tested",
                    "IDEA_ADOPTED": "adopted",
                    "IDEA_EXPIRED": "expired",
                }[event_type]

    # Conversions are also visible from the capability side (METHOD_PROPOSED
    # with source_type=idea_conversion), which keeps the two streams consistent
    # even if one is rebuilt alone.
    converted_from_capability: List[str] = []
    for event in capability_events or []:
        if str(event.get("type")) != "METHOD_PROPOSED":
            continue
        if str(event.get("source_type") or "") != "idea_conversion":
            continue
        source_idea = event.get("source_idea_id")
        if source_idea and str(source_idea) in ideas:
            ideas[str(source_idea)]["converted_method_id"] = event.get("method_id")
            converted_from_capability.append(str(source_idea))

    for idea in ideas.values():
        by_status[idea["status"]] = by_status.get(idea["status"], 0) + 1
        by_motif[idea.get("motif") or "unknown"] = by_motif.get(idea.get("motif") or "unknown", 0) + 1
        by_type[idea["idea_type"]] = by_type.get(idea["idea_type"], 0) + 1

    return {
        "persona": persona,
        "ideas": dict(sorted(ideas.items())),
        "rejected": rejected,
        "skipped": skipped,
        "skipped_by_reason": dict(
            sorted(
                Counter(str(row.get("reason") or "") for row in skipped).items()
            )
        ),
        "stats": {
            "ideas": len(ideas),
            "rejected": len(rejected),
            "by_status": dict(sorted(by_status.items())),
            "by_motif": dict(sorted(by_motif.items())),
            "by_type": dict(sorted(by_type.items())),
            "by_event_type": dict(sorted(by_event_type.items())),
            "converted_from_capability": sorted(set(converted_from_capability)),
        },
    }


def build_idea_view(dm: "DataManager") -> Dict[str, Any]:
    from src.agents.cognition.event_store import CapabilityEventStore
    from src.agents.cognition.idea_engine import IdeaConfig  # noqa: F401  (kept for callers)

    store = IdeaEventStore(dm)
    return materialize_ideas(
        store.events(),
        CapabilityEventStore(dm).events(),
        persona=dm.char,
    )


def write_idea_view(dm: "DataManager", view: Optional[Dict[str, Any]] = None) -> Path:
    if view is None:
        view = build_idea_view(dm)
    target_dir = dm.root / "cognition" / "views"
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / IDEAS_VIEW_FILENAME
    path.write_text(
        json.dumps(view, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def load_ideas(dm: "DataManager") -> List[Idea]:
    """Existing ideas as objects (used for the novelty check)."""
    view = build_idea_view(dm)
    out: List[Idea] = []
    for entry in view["ideas"].values():
        payload = {
            k: v
            for k, v in entry.items()
            if k not in ("week", "converted_method_id")
        }
        # Keep the conversion visible: an idea that already produced a candidate
        # methodology must not produce another one next week.
        payload["candidate_method_id"] = entry.get("converted_method_id")
        try:
            out.append(Idea.from_dict(payload))
        except (KeyError, ValueError):
            continue
    return out
