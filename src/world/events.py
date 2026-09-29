"""Event identity for the append-only ledger (phase 0).

Every record written through `DataManager._append_jsonl` carries a stable
`ledger_event_id` and a `schema_version`. The point is determinism: two runs of the
same simulation must produce the *same* ids for the same logical events, so the
event sets can be compared (and replayed) instead of relying on physical line
order, thread timing or wall-clock.

Design constraints
------------------
- `ledger_event_id` is derived from content only: `(stream, time, payload)`. It does
  not depend on append order, thread scheduling, file offsets or the run
  directory name, so a parallel run and a serial replay of it produce identical
  ids.
- The stream id is the path *relative to the run directory* (`persona/<name>/
  state.jsonl`), because the run directory name contains a timestamp and would
  otherwise make every id run-specific.
- The id is content-addressed, so appending the exact same record twice at the
  same simulated time yields the same id: those are semantically the same event,
  and a comparator counting ids still detects a run that emitted it twice.
- Records written before this change have no id; readers must tolerate them
  (`event_id` is optional, and missing means "legacy").

The field is deliberately *not* called `event_id`: `public_events.jsonl` already
uses `event_id` for a domain identifier (`public-2020-W01-1`) that is stable per
slot but independent of content, so reusing the name would either clobber that
value or make the ledger identity silently unavailable for those records.

`idempotency_key` is separate: it names an *effect* (for example applying one
activity's outcome to one persona) so that applying it twice cannot happen, no
matter how often a stage is re-entered after a resume.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

# Bumped whenever the meaning or shape of a ledger record changes.
SCHEMA_VERSION = 1

# Prefix keeps ids recognizable in raw jsonl dumps.
_EVENT_ID_PREFIX = "ev"

# Fields that are identity/annotation rather than payload: they must not feed
# the hash, or the id would change as soon as it is attached or annotated.
_NON_PAYLOAD_KEYS = frozenset(
    {
        "ledger_event_id",
        "schema_version",
        "idempotency_key",
        # Per-file ordinal used to target one record for annotation; it depends
        # on how many records the file already holds, so it is identity, not
        # payload (see DataManager.save_generation).
        "record_id",
        # Annotations added after the fact.
        "rejected",
        "reject_reason",
    }
)


def event_stream_id(path: Path | str) -> str:
    """Return the run-independent stream id for a ledger file.

    `data/<world>_<runid>/persona/<name>/state.jsonl` becomes
    `persona/<name>/state.jsonl`, so ids stay comparable across runs. Paths
    outside a run directory are used as given (posix form).
    """
    parts = Path(path).as_posix().split("/")
    if len(parts) >= 3 and parts[0] == "data":
        # data/<run-id>/<stream...>
        return "/".join(parts[2:])
    return Path(path).as_posix()


def _canonical_payload(payload: Mapping[str, Any]) -> str:
    """Canonical JSON of the payload, excluding identity/annotation fields."""
    clean = {k: v for k, v in payload.items() if k not in _NON_PAYLOAD_KEYS}
    return json.dumps(clean, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def make_event_id(
    *, stream: str, time_str: str, payload: Mapping[str, Any]
) -> str:
    """Content-addressed, order-independent id for one ledger event."""
    digest = hashlib.sha1(
        f"{SCHEMA_VERSION}\x1f{stream}\x1f{time_str}\x1f{_canonical_payload(payload)}".encode(
            "utf-8"
        )
    ).hexdigest()
    return f"{_EVENT_ID_PREFIX}-{digest[:16]}"


def attach_event_identity(
    record: Dict[str, Any],
    *,
    stream: str,
    idempotency_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Add `event_id` / `schema_version` (and optionally the key) in place.

    `record["time"]` must already be set: it is part of the identity.
    An existing `event_id` is preserved, so re-writing a record (for example to
    annotate a rejection) keeps the identity of the original event.
    """
    if "ledger_event_id" not in record:
        record["ledger_event_id"] = make_event_id(
            stream=stream, time_str=str(record.get("time", "")), payload=record
        )
    record["schema_version"] = SCHEMA_VERSION
    if idempotency_key is not None:
        record["idempotency_key"] = str(idempotency_key)
    return record


def stamped_record(record: Dict[str, Any], *, path: "Path | str") -> Dict[str, Any]:
    """Return a copy of `record` stamped with the ledger identity of `path`.

    For the few writers that own their file directly (public events, position
    application log, reward views, god generations) instead of going through
    DataManager._append_jsonl. Callers keep their own locking and formatting;
    this only guarantees that every ledger record is identifiable and therefore
    comparable across runs.
    """
    outgoing = dict(record)
    attach_event_identity(outgoing, stream=event_stream_id(path))
    return outgoing
