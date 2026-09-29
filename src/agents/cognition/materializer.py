"""Materialized capability views, rebuilt from capability events (doc 5.3/5.4/10).

    persona/<name>/cognition/views/capabilities.json

A view is derived and disposable: deleting it and rebuilding from
`capability_events.jsonl` must produce exactly the same file ("materialized views
must be fully rebuildable from the event stream", invariant #9). The fold is
pure — no clocks, no randomness, no LLM — so it is also a cheap consistency
check on the event stream itself.

The authoritative numbers live in `METHOD_VALUE_UPDATED` events (written by
reward_model), not in the fold: rebuilding must not re-derive values with a
different formula than the one that produced them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TYPE_CHECKING

from src.agents.cognition.models import Methodology

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.agents.data_manager import DataManager

VIEW_FILENAME = "capabilities.json"


def week_key_from_time(time_str: str) -> str:
    """`Y2020-W03-activity-D2` -> `Y2020-W03` (empty when unparsable)."""
    parts = str(time_str).split("-")
    if len(parts) >= 2 and parts[0].startswith("Y") and parts[1].startswith("W"):
        return f"{parts[0]}-{parts[1]}"
    return ""


def _ensure_method(
    methods: Dict[str, Dict[str, Any]], method_id: str
) -> Dict[str, Any]:
    if method_id not in methods:
        methods[method_id] = {
            "method_id": method_id,
            "skill_id": "",
            "version": 1,
            "title": "",
            "description": "",
            "status": "proposed",
            "source_type": "practice_reflection",
            "parent_method_id": None,
            "applicable_contexts": [],
            "contraindications": [],
            "steps": [],
            "checks": [],
            "failure_modes": [],
            "global_value": 0.0,
            "confidence": 0.0,
            "practice_count": 0,
            "success_count": 0,
            "context_values": {},
            "source_event_ids": [],
            "last_used": "",
            # view-only bookkeeping
            "selections": 0,
            "applications": 0,
            "declines": 0,
            "evidence": [],
            "last_selection": None,
            "last_declined": None,
            "skill_mapped": True,
            # Week the proposal came from: an activity in that same week must not
            # feed the method (reflection cannot reinforce itself).
            "proposed_week": "",
        }
    return methods[method_id]


def materialize(
    events: Iterable[Dict[str, Any]],
    *,
    skills: Optional[Dict[str, float]] = None,
    persona: str = "",
) -> Dict[str, Any]:
    """Fold capability events into the capability view."""
    methods: Dict[str, Dict[str, Any]] = {}
    by_status: Dict[str, int] = {}
    by_type: Dict[str, int] = {}
    last_time = ""
    count = 0
    known_skills = set((skills or {}).keys())

    for event in events:
        count += 1
        event_type = str(event.get("type") or "")
        by_type[event_type] = by_type.get(event_type, 0) + 1
        last_time = str(event.get("time") or last_time)

        method_id = event.get("method_id")
        if not method_id:
            continue
        method = _ensure_method(methods, str(method_id))
        if event.get("ledger_event_id"):
            method["source_event_ids"].append(str(event["ledger_event_id"]))

        if event_type in ("METHOD_PROPOSED", "METHOD_LEARNED", "METHOD_REFINED"):
            if event.get("title"):
                method["title"] = str(event["title"])
            if event.get("description"):
                method["description"] = str(event["description"])
            if event.get("skill_id"):
                method["skill_id"] = str(event["skill_id"])
            if event.get("source_type"):
                method["source_type"] = str(event["source_type"])
            if event.get("parent_method_id"):
                method["parent_method_id"] = str(event["parent_method_id"])
            # Phase 5 version evolution: a `METHOD_REFINED` is the same method,
            # re-derived in a later week with a better wording. The version is
            # authoritative from the event, never re-derived here.
            if event.get("version") is not None:
                try:
                    method["version"] = int(event["version"])
                except (TypeError, ValueError):
                    pass
            elif event_type == "METHOD_REFINED":
                method["version"] = int(method.get("version", 1)) + 1
            for key in ("applicable_contexts", "contraindications", "steps", "checks", "failure_modes"):
                if event.get(key):
                    method[key] = list(event[key])
            if event.get("week"):
                method["proposed_week"] = str(event["week"])
            elif event.get("time"):
                method["proposed_week"] = week_key_from_time(str(event["time"]))
            if event_type == "METHOD_LEARNED":
                method["status"] = "learned"
            if event.get("skill_mapped") is False:
                method["skill_mapped"] = False
            # Phase 4 needs a method's provenance and its structure flag: the hint
            # provider refuses to offer a structure-incomplete method (KI-6) and
            # marks the memories behind an adopted method as used (KI-10).
            if event.get("source_memory_ids"):
                method["source_memory_ids"] = list(event["source_memory_ids"])
            if event.get("source_idea_id"):
                method["source_idea_id"] = str(event["source_idea_id"])
            if event.get("source_motif"):
                method["source_motif"] = str(event["source_motif"])
            if event.get("structure_incomplete"):
                method["structure_incomplete"] = True
                method["structure_missing"] = list(event.get("structure_missing") or [])

        elif event_type == "METHOD_SELECTED":
            method["selections"] = int(method.get("selections", 0)) + 1
            method["last_used"] = str(event.get("time") or method.get("last_used") or "")
            method["last_selection"] = {
                "activity_id": event.get("activity_id"),
                "context_key": event.get("context_key"),
                "explored": bool(event.get("explored")),
                "role": event.get("role", "primary"),
            }

        elif event_type == "METHOD_APPLIED":
            method["applications"] = int(method.get("applications", 0)) + 1

        elif event_type == "METHOD_DECLINED":
            # Phase 5: the character was shown this method and did not take it
            # up. It moves no value (invariant #11) — it is the offerer's own
            # feedback about its choice of what to show.
            method["declines"] = int(method.get("declines", 0)) + 1
            method["last_declined"] = {
                "week": event.get("week"),
                "reason": event.get("reason"),
                "role": event.get("role"),
                "context_key": event.get("context_key"),
            }

        elif event_type == "METHOD_OUTCOME_OBSERVED":
            method["evidence"].append(
                {
                    "activity_id": event.get("activity_id"),
                    "time": event.get("time"),
                    "context_key": event.get("context_key"),
                    "reward": event.get("reward"),
                    "success": bool(event.get("success")),
                }
            )

        elif event_type == "METHOD_VALUE_UPDATED":
            for key in (
                "global_value",
                "confidence",
                "practice_count",
                "success_count",
                "status",
            ):
                if key in event:
                    method[key] = event[key]
            if event.get("context_values"):
                method["context_values"] = dict(event["context_values"])
            if event.get("last_used"):
                method["last_used"] = str(event["last_used"])

        elif event_type in ("METHOD_ARCHIVED", "METHOD_SUPERSEDED"):
            method["status"] = (
                "archived" if event_type == "METHOD_ARCHIVED" else "deprecated"
            )

    # Skill projection: existing state.skills, annotated with the methods that
    # belong to each skill. Skills themselves are never modified here.
    skill_view: Dict[str, Any] = {}
    for name, proficiency in (skills or {}).items():
        skill_view[name] = {"proficiency": proficiency, "methodology_ids": []}
    for method_id, method in sorted(methods.items()):
        by_status[method["status"]] = by_status.get(method["status"], 0) + 1
        skill_id = method.get("skill_id") or ""
        if skill_id in skill_view:
            skill_view[skill_id]["methodology_ids"].append(method_id)

    unmapped = sorted(
        m["method_id"]
        for m in methods.values()
        if m.get("skill_mapped") is False
        or (m.get("skill_id") and m["skill_id"] not in known_skills)
    )

    return {
        "persona": persona,
        "event_count": count,
        "last_event_time": last_time,
        "skills": skill_view,
        "methodologies": dict(sorted(methods.items())),
        "stats": {
            "by_status": dict(sorted(by_status.items())),
            "by_event_type": dict(sorted(by_type.items())),
            "methods": len(methods),
            "unmapped_skill_methods": unmapped,
        },
    }


def skills_from_state(dm: "DataManager") -> Dict[str, float]:
    """Read the simulation's skill numbers (phase 6 migrates this projection)."""
    try:
        state = dm.read_state(exclude_cur_t=False)
    except (IndexError, FileNotFoundError):
        return {}
    skills = state.get("skills") or {}
    out: Dict[str, float] = {}
    for name, value in skills.items():
        try:
            out[str(name)] = round(float(value), 3)
        except (TypeError, ValueError):
            continue
    return out


def build_capability_view(dm: "DataManager") -> Dict[str, Any]:
    """Read the event stream and build the view for one persona."""
    from src.agents.cognition.event_store import CapabilityEventStore

    store = CapabilityEventStore(dm)
    return materialize(
        store.events(), skills=skills_from_state(dm), persona=dm.char
    )


def views_dir(dm: "DataManager") -> Path:
    return dm.root / "cognition" / "views"


def write_capability_view(
    dm: "DataManager", view: Optional[Dict[str, Any]] = None
) -> Path:
    """Write the view next to the event stream (safe to delete and rebuild)."""
    if view is None:
        view = build_capability_view(dm)
    target_dir = views_dir(dm)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / VIEW_FILENAME
    path.write_text(
        json.dumps(view, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def load_methodologies(view: Dict[str, Any]) -> List[Methodology]:
    """Turn the view's methodology entries back into objects (for the policy)."""
    out: List[Methodology] = []
    for entry in (view.get("methodologies") or {}).values():
        payload = {
            k: v
            for k, v in entry.items()
            if k not in ("selections", "applications", "evidence", "last_selection", "skill_mapped")
        }
        try:
            out.append(Methodology.from_dict(payload))
        except (KeyError, ValueError):
            continue
    return out
