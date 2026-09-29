"""Memory event streams (design doc 10, draft §3.2).

Two append-only files per persona:

    cognition/memory_events.jsonl            memories and their lifecycle
    cognition/memory_relation_events.jsonl   asserted relations between them

Both reuse the phase 0 identity contract (via `DataManager.append_ledger_record`)
and the phase 1 idempotency style: every event carries the simulated `week`, and
its key is derived from the effect (not the record), so re-running a week after a
resume cannot duplicate a memory or a relation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from src.agents.cognition.memory_models import MemoryRelation

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.agents.data_manager import DataManager

MEMORY_EVENT_TYPES = (
    "MEMORY_CREATED",
    "MEMORY_REINFORCED",
    "MEMORY_MERGED",
    "MEMORY_SUPERSEDED",
    "MEMORY_CONTRADICTED",
    "MEMORY_STRENGTH_UPDATED",
    "MEMORY_ARCHIVED",
    "MEMORY_RECALLED",
    "MEMORY_RETRIEVAL_COMPARED",
    # Phase 4: a memory actually informed an adopted method (KI-10). Only real
    # use counts -- being placed in a prompt does not (invariant #6).
    "MEMORY_USED_IN_PLAN",
)

RELATION_EVENT_TYPES = ("RELATION_ASSERTED",)


class _EventStream:
    """Shared reader/writer for one cognition jsonl stream."""

    event_types: Tuple[str, ...] = ()

    def __init__(self, dm: "DataManager", filename: str) -> None:
        self.dm = dm
        self.path: Path = dm.root / "cognition" / filename
        self._keys: Optional[Set[str]] = None

    # -- writing -----------------------------------------------------------
    def append(
        self,
        event_type: str,
        *,
        idempotency_key: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], bool]:
        if self.event_types and event_type not in self.event_types:
            raise ValueError(f"unknown event type for {self.path.name}: {event_type!r}")

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
                raise ValueError(f"{self.path.as_posix()}: invalid event: {line}") from e
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


class MemoryEventStore(_EventStream):
    """`memory_events.jsonl`."""

    event_types = MEMORY_EVENT_TYPES

    def __init__(self, dm: "DataManager") -> None:
        super().__init__(dm, "memory_events.jsonl")

    # Convenience wrappers keep the idempotency key format in one place.
    def created(self, *, memory_id: str, week: str, payload: Dict[str, Any]):
        return self.append(
            "MEMORY_CREATED",
            idempotency_key=f"MEMORY_CREATED:{memory_id}:{week}",
            payload={"memory_id": memory_id, "week": week, **payload},
        )

    def reinforced(self, *, memory_id: str, week: str, source_event_ids: List[str], payload=None):
        return self.append(
            "MEMORY_REINFORCED",
            idempotency_key=f"MEMORY_REINFORCED:{memory_id}:{week}",
            payload={
                "memory_id": memory_id,
                "week": week,
                "source_event_ids": list(source_event_ids),
                **(payload or {}),
            },
        )

    def merged(self, *, memory_id: str, week: str, merged_from: List[str], payload=None):
        return self.append(
            "MEMORY_MERGED",
            idempotency_key=f"MEMORY_MERGED:{memory_id}:{week}",
            payload={
                "memory_id": memory_id,
                "week": week,
                "merged_from": list(merged_from),
                **(payload or {}),
            },
        )

    def contradicted(self, *, memory_id: str, week: str, conflicts_with: str, reason: str):
        # Directional on purpose: both memories are marked, so the pair is
        # stored as two events that each name their own conflict.
        return self.append(
            "MEMORY_CONTRADICTED",
            idempotency_key=f"MEMORY_CONTRADICTED:{memory_id}:{conflicts_with}",
            payload={
                "memory_id": memory_id,
                "conflicts_with": conflicts_with,
                "reason": reason,
                "week": week,
            },
        )

    def strength_updated(self, *, memory_id: str, week: str, strength: float, tier: str, factors: Dict):
        return self.append(
            "MEMORY_STRENGTH_UPDATED",
            idempotency_key=f"MEMORY_STRENGTH_UPDATED:{memory_id}:{week}",
            payload={
                "memory_id": memory_id,
                "week": week,
                "strength": strength,
                "tier": tier,
                "factors": factors,
            },
        )

    def archived(self, *, memory_id: str, week: str, reason: str, payload=None):
        return self.append(
            "MEMORY_ARCHIVED",
            idempotency_key=f"MEMORY_ARCHIVED:{memory_id}:{week}",
            payload={"memory_id": memory_id, "week": week, "reason": reason, **(payload or {})},
        )

    def recalled(
        self,
        *,
        memory_id: str,
        week: str,
        query_context: str,
        rank: int,
        score: float,
        tier: str,
    ):
        return self.append(
            "MEMORY_RECALLED",
            idempotency_key=f"MEMORY_RECALLED:{memory_id}:{week}:{rank}",
            payload={
                "memory_id": memory_id,
                "week": week,
                "query_context": query_context,
                "rank": rank,
                "score": round(float(score), 4),
                "tier": tier,
                # A shadow recall is never "used": nothing is injected yet.
                "used": False,
            },
        )


    def used_in_plan(self, *, memory_id: str, week: str, method_id: str, idea_id: str = ""):
        """Record that a memory informed an adopted method (phase 4, KI-10)."""
        return self.append(
            "MEMORY_USED_IN_PLAN",
            idempotency_key=f"MEMORY_USED_IN_PLAN:{memory_id}:{method_id}",
            payload={
                "memory_id": memory_id,
                "week": week,
                "method_id": method_id,
                "idea_id": idea_id,
            },
        )

    def retrieval_compared(self, *, week: str, legacy: Dict[str, Any], proposed: Dict[str, Any], overlap: Dict[str, Any], extra: Optional[Dict[str, Any]] = None):
        """Weekly shadow comparison of old vs new retrieval (design draft §7)."""
        return self.append(
            "MEMORY_RETRIEVAL_COMPARED",
            idempotency_key=f"MEMORY_RETRIEVAL_COMPARED:{week}",
            payload={
                "week": week,
                "legacy": legacy,
                "proposed": proposed,
                "overlap": overlap,
                **(extra or {}),
            },
        )


class RelationEventStore(_EventStream):
    """`memory_relation_events.jsonl`."""

    event_types = RELATION_EVENT_TYPES

    def __init__(self, dm: "DataManager") -> None:
        super().__init__(dm, "memory_relation_events.jsonl")

    def asserted(self, relation: MemoryRelation):
        payload = relation.to_dict()
        payload.pop("type", None)
        return self.append(
            "RELATION_ASSERTED",
            idempotency_key=relation.key,
            payload=payload,
        )
