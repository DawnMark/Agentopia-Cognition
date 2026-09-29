"""Append-only capability events (design doc 5.5 / 10).

The stream lives at `persona/<name>/cognition/capability_events.jsonl` and
follows exactly the same contract as the simulation ledger:

- every record carries `ledger_event_id` + `schema_version` from
  `src/world/events.py`, so capability events are comparable across runs and
  survive replay;
- writes go through `DataManager.append_ledger_record`, inheriting per-file
  locking, the time-ordering guard and a full flush;
- effects are idempotent: a re-entered stage (resume, retry) must not record the
  same selection/observation twice, and applying the same value update twice
  must not move the number twice.

Reading is deliberately forgiving: records written before a schema change (or by
an older build) are returned as-is, and unknown event types are kept so a newer
writer's events are not silently dropped by an older reader.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, TYPE_CHECKING

from src.agents.cognition.models import CAPABILITY_EVENT_TYPES

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.agents.data_manager import DataManager


class CapabilityEventStore:
    """Reader/writer for one persona's capability event stream."""

    def __init__(self, dm: "DataManager") -> None:
        self.dm = dm
        self.path: Path = dm.root / "cognition" / "capability_events.jsonl"
        self._keys: Optional[Set[str]] = None

    # -- writing -----------------------------------------------------------
    def append(
        self,
        event_type: str,
        *,
        method_id: Optional[str] = None,
        activity_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], bool]:
        """Append one capability event.

        Returns `(record, created)`. `created` is False when the idempotency key
        was already present, in which case nothing is written.
        """
        if event_type not in CAPABILITY_EVENT_TYPES:
            raise ValueError(f"unknown capability event type: {event_type!r}")

        key = idempotency_key or self._default_key(
            event_type, method_id=method_id, activity_id=activity_id, payload=payload
        )
        if key in self.idempotency_keys():
            return {"type": event_type, "idempotency_key": key, "duplicate": True}, False

        record: Dict[str, Any] = {"type": event_type, "idempotency_key": key}
        if method_id is not None:
            record["method_id"] = method_id
        if activity_id is not None:
            record["activity_id"] = activity_id
        for k, v in (payload or {}).items():
            if k in ("time", "ledger_event_id", "schema_version", "idempotency_key"):
                continue
            record[k] = v

        self.dm.append_ledger_record(self.path, record, idempotency_key=key)
        if self._keys is not None:
            self._keys.add(key)
        return record, True

    @staticmethod
    def _default_key(
        event_type: str,
        *,
        method_id: Optional[str],
        activity_id: Optional[str],
        payload: Optional[Dict[str, Any]],
    ) -> str:
        """Identity of the *effect*, not of the record.

        Tied to the simulated time so the same action in a later week is a
        different effect, and to the activity/method so a re-entered stage
        collapses onto the first write.
        """
        week = str((payload or {}).get("week") or "")
        parts = [event_type, method_id or "-", activity_id or "-", week]
        return ":".join(parts)

    # -- reading -----------------------------------------------------------
    def events(self) -> List[Dict[str, Any]]:
        """All capability events, oldest first."""
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
                raise ValueError(
                    f"{self.path.as_posix()}: invalid capability event: {line}"
                ) from e
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
        """Drop the cached key set (after external writes, e.g. a resume)."""
        self._keys = None


def count_by_type(events: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    """Small helper for logs and tests."""
    counts: Dict[str, int] = {}
    for e in events:
        counts[str(e.get("type"))] = counts.get(str(e.get("type")), 0) + 1
    return counts
