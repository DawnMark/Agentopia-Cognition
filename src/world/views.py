"""Materialized views over a run's event ledger (phase 0).

The ledger files under `data/<run>/` are the source of truth. A *view* is a
derived, deletable, rebuildable summary of them:

    views/event_index.json

The event index answers the phase 0 acceptance question directly: do two runs
(parallel vs serial, original vs replay, interrupted vs uninterrupted) contain
the same events? It is built purely from `ledger_event_id`s, so it does not care about
physical line order, thread scheduling or which run directory the events live
in.

Nothing here writes to the ledger: rebuilding a view must always be safe.
"""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.world.events import event_stream_id

# Derived artifacts that are not part of the ledger.
_EXCLUDED_DIR_PARTS = ("views",)
_EXCLUDED_SUFFIXES = (".tmp",)


def iter_ledger_files(run_dir: Path) -> List[Path]:
    """Return the ledger files of a run, in a stable order."""
    files: List[Path] = []
    for path in sorted(run_dir.rglob("*.jsonl")):
        if any(part in _EXCLUDED_DIR_PARTS for part in path.parts):
            continue
        if path.name.endswith(_EXCLUDED_SUFFIXES):
            continue
        files.append(path)
    return files


def _read_records(path: Path) -> Iterable[Dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path.as_posix()}: invalid ledger record: {line}") from e


def build_event_index(run_dir: Path | str) -> Dict[str, Any]:
    """Build the comparable event index of a run.

    Layout:
        {
          "run": "<run directory name>",
          "records": <int>,          # total ledger records
          "legacy_records": <int>,   # records written before event ids existed
          "streams": {
            "<stream id>": {
                "records": <int>,
                "event_ids": [<sorted, de-duplicated>],
                "duplicate_events": <int>,   # same id emitted more than once
                "first_time": str, "last_time": str,
            }, ...
          }
        }
    """
    run_dir = Path(run_dir)
    streams: Dict[str, Dict[str, Any]] = {}
    records = 0
    legacy = 0

    for path in iter_ledger_files(run_dir):
        stream = event_stream_id(path)
        ids: List[str] = []
        times: List[str] = []
        legacy_in_stream = 0
        count = 0
        for record in _read_records(path):
            records += 1
            count += 1
            time_str = str(record.get("time", ""))
            if time_str:
                times.append(time_str)
            event_id = record.get("ledger_event_id")
            if isinstance(event_id, str) and event_id:
                ids.append(event_id)
            else:
                legacy += 1
                legacy_in_stream += 1
        unique_ids = sorted(set(ids))
        streams[stream] = {
            "records": count,
            "legacy_records": legacy_in_stream,
            "event_ids": unique_ids,
            "duplicate_events": len(ids) - len(unique_ids),
            "first_time": min(times) if times else "",
            "last_time": max(times) if times else "",
        }

    return {
        "run": run_dir.name,
        "records": records,
        "legacy_records": legacy,
        "streams": streams,
    }


def filter_streams(
    index: Dict[str, Any], patterns: Sequence[str]
) -> Dict[str, Any]:
    """Return a copy of `index` without streams matching any glob pattern.

    Used to compare a run against one that did extra (additive) bookkeeping:
    the phase 1 shadow writes only to persona/*/cognition/*, so ignoring those
    streams must leave the simulation ledger identical.
    """
    if not patterns:
        return index
    kept = {
        stream: data
        for stream, data in (index.get("streams") or {}).items()
        if not any(fnmatch.fnmatch(stream, pattern) for pattern in patterns)
    }
    filtered = dict(index)
    filtered["streams"] = kept
    filtered["ignored_streams"] = sorted(
        set((index.get("streams") or {}).keys()) - set(kept.keys())
    )
    # Recompute the totals so a report never quotes the unfiltered count.
    filtered["records"] = sum(int(d.get("records", 0)) for d in kept.values())
    filtered["legacy_records"] = sum(
        int(d.get("legacy_records", 0)) for d in kept.values()
    )
    return filtered


def compare_event_index(
    index_a: Dict[str, Any], index_b: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare two indices event-by-event.

    Returns a report with per-stream differences; `identical` is True only when
    every stream carries exactly the same event ids.
    """
    streams_a = index_a.get("streams", {})
    streams_b = index_b.get("streams", {})

    only_a = sorted(set(streams_a) - set(streams_b))
    only_b = sorted(set(streams_b) - set(streams_a))

    per_stream: Dict[str, Dict[str, Any]] = {}
    count_differences: Dict[str, Dict[str, int]] = {}
    for stream in sorted(set(streams_a) & set(streams_b)):
        data_a, data_b = streams_a[stream], streams_b[stream]
        ids_a = set(data_a["event_ids"])
        ids_b = set(data_b["event_ids"])
        missing = sorted(ids_a - ids_b)
        extra = sorted(ids_b - ids_a)
        if missing or extra:
            per_stream[stream] = {"missing_in_b": missing, "extra_in_b": extra}

        # Same events but a different number of records means one run emitted an
        # event more than once (e.g. a re-executed stage after a resume), which
        # the id sets alone cannot show.
        if data_a["records"] != data_b["records"] or int(
            data_a.get("duplicate_events", 0)
        ) != int(data_b.get("duplicate_events", 0)):
            count_differences[stream] = {
                "records_a": int(data_a["records"]),
                "records_b": int(data_b["records"]),
                "duplicate_events_a": int(data_a.get("duplicate_events", 0)),
                "duplicate_events_b": int(data_b.get("duplicate_events", 0)),
            }

    identical = not (only_a or only_b or per_stream or count_differences)
    return {
        "identical": identical,
        "streams_only_in_a": only_a,
        "streams_only_in_b": only_b,
        "differences": per_stream,
        "count_differences": count_differences,
        "records_a": index_a.get("records", 0),
        "records_b": index_b.get("records", 0),
        "legacy_records_a": index_a.get("legacy_records", 0),
        "legacy_records_b": index_b.get("legacy_records", 0),
    }


def write_event_index(run_dir: Path | str, index: Optional[Dict[str, Any]] = None) -> Path:
    """Materialize the event index under `<run>/views/event_index.json`."""
    run_dir = Path(run_dir)
    if index is None:
        index = build_event_index(run_dir)
    views_dir = run_dir / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    out = views_dir / "event_index.json"
    out.write_text(
        json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out


def summarize_index(index: Dict[str, Any]) -> List[Tuple[str, int, int]]:
    """Return (stream, unique events, duplicate events) rows for reporting."""
    rows: List[Tuple[str, int, int]] = []
    for stream, data in sorted(index.get("streams", {}).items()):
        rows.append(
            (stream, len(data.get("event_ids", [])), int(data.get("duplicate_events", 0)))
        )
    return rows
