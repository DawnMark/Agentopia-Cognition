#!/usr/bin/env python3
"""Compare the event ledgers of two runs (phase 0 acceptance tool).

Answers the phase 0 question directly:

    python scripts/verify_replay.py --run-a shanghai_apartment_09252339 \\
                                    --run-b shanghai_apartment_09261012

Both arguments are run directory names under `data/` (or paths). The comparison
is by `event_id`, so it is independent of line order, thread scheduling and the
run directory name — exactly what "parallel vs serial", "original vs replay" and
"interrupted vs uninterrupted" need.

Exit code 0 means the event sets are identical, 1 means they differ, 2 means the
comparison could not run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.world.views import (  # noqa: E402
    build_event_index,
    compare_event_index,
    filter_streams,
    summarize_index,
    write_event_index,
)


def _resolve_run_dir(value: str) -> Path:
    direct = Path(value)
    if direct.is_dir():
        return direct
    under_data = Path("data") / value
    if under_data.is_dir():
        return under_data
    raise FileNotFoundError(f"run directory not found: {value}")


def _print_index(label: str, index: dict) -> None:
    print(
        f"{label}: run={index['run']} records={index['records']} "
        f"streams={len(index['streams'])} legacy_records={index['legacy_records']}"
    )
    for stream, unique_events, duplicates in summarize_index(index):
        suffix = f" (duplicates={duplicates})" if duplicates else ""
        print(f"  {stream}: {unique_events} events{suffix}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", required=True, help="reference run (dir or name under data/)")
    parser.add_argument("--run-b", required=True, help="run to compare against the reference")
    parser.add_argument(
        "--write-views",
        action="store_true",
        help="materialize data/<run>/views/event_index.json for both runs",
    )
    parser.add_argument(
        "--ignore-stream",
        action="append",
        default=[],
        metavar="GLOB",
        help=(
            "ignore streams matching this glob (repeatable), e.g. "
            "'persona/*/cognition/*' to compare only the simulation ledger"
        ),
    )
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args()

    try:
        run_a = _resolve_run_dir(args.run_a)
        run_b = _resolve_run_dir(args.run_b)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    index_a = build_event_index(run_a)
    index_b = build_event_index(run_b)
    if args.ignore_stream:
        index_a = filter_streams(index_a, args.ignore_stream)
        index_b = filter_streams(index_b, args.ignore_stream)

    if args.write_views:
        print(f"view written: {write_event_index(run_a, index_a).as_posix()}")
        print(f"view written: {write_event_index(run_b, index_b).as_posix()}")

    report = compare_event_index(index_a, index_b)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        _print_index("A", index_a)
        _print_index("B", index_b)
        if args.ignore_stream:
            print(f"ignored stream patterns: {', '.join(args.ignore_stream)}")
        print("-" * 60)
        if report["identical"]:
            print(
                f"IDENTICAL: both runs contain the same {index_a['records']} events "
                f"across {len(index_a['streams'])} streams."
            )
        else:
            print("DIFFERENT:")
            for stream in report["streams_only_in_a"]:
                print(f"  stream only in A: {stream}")
            for stream in report["streams_only_in_b"]:
                print(f"  stream only in B: {stream}")
            for stream, diff in report["differences"].items():
                print(
                    f"  {stream}: missing_in_b={len(diff['missing_in_b'])} "
                    f"extra_in_b={len(diff['extra_in_b'])}"
                )
                for event_id in diff["missing_in_b"][:5]:
                    print(f"      - only in A: {event_id}")
                for event_id in diff["extra_in_b"][:5]:
                    print(f"      + only in B: {event_id}")
            for stream, diff in report.get("count_differences", {}).items():
                print(
                    f"  {stream}: records {diff['records_a']} vs {diff['records_b']}, "
                    f"duplicates {diff['duplicate_events_a']} vs "
                    f"{diff['duplicate_events_b']}"
                )

    return 0 if report["identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
