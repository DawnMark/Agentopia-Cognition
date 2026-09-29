"""Analyze per-week wall-clock time for simulation runs.

Usage:
    python scripts/time_analysis.py <run_id> [<run_id2> ...]
    python scripts/time_analysis.py --all          # analyze all runs
    python scripts/time_analysis.py --latest 3     # analyze 3 most recent runs

Output: data/{run_id}/time_analysis.md for each run, plus stdout summary.
"""

import argparse
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Server is PDT (UTC-7), we want GMT+8 (UTC+8) => +15h
TZ_OFFSET = timedelta(hours=15)

PLAN_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*== PLAN STAGE == year=(\d+) week=(\d+)"
)
SETTLE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*== SETTLE STAGE == year=(\d+) week=(\d+)"
)


def parse_ts(ts_str: str) -> datetime:
    return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S,%f") + TZ_OFFSET


def find_log_file(run_id: str) -> Path | None:
    """Find the world.log for a run_id."""
    log_dir = Path("logs") / run_id
    if log_dir.is_dir():
        wlog = log_dir / "world.log"
        if wlog.exists():
            return wlog
    return None


def parse_world_log(log_path: Path) -> list[dict]:
    """Parse PLAN/SETTLE timestamps from world.log, return per-week rows."""
    plans: dict[str, datetime] = {}  # label -> ts (last wins for resume)
    settles: dict[str, datetime] = {}

    with open(log_path) as f:
        for line in f:
            m = PLAN_RE.match(line)
            if m:
                ts, year, week = m.groups()
                label = f"Y{year}-W{week}"
                plans[label] = parse_ts(ts)
                continue
            m = SETTLE_RE.match(line)
            if m:
                ts, year, week = m.groups()
                label = f"Y{year}-W{week}"
                settles[label] = parse_ts(ts)

    rows = []
    for label, start in plans.items():
        if label in settles:
            end = settles[label]
            mins = (end - start).total_seconds() / 60
            rows.append({
                "week": label,
                "start": start,
                "end": end,
                "minutes": mins,
            })
    rows.sort(key=lambda r: r["start"])
    return rows


def load_model_info(run_id: str) -> str:
    """Load role_model from run config."""
    config_path = Path("data") / run_id / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        return cfg.get("role_model", "unknown")
    return "unknown"


def count_agents(run_id: str) -> int:
    persona_dir = Path("data") / run_id / "persona"
    if persona_dir.is_dir():
        return len([d for d in persona_dir.iterdir() if d.is_dir()])
    return 0


def format_report(run_id: str, rows: list[dict], model: str, n_agents: int) -> str:
    """Generate markdown report."""
    lines = [
        f"# Time Analysis: {run_id}",
        "",
        f"- **Model**: `{model}`",
        f"- **Agents**: {n_agents}",
        f"- **Weeks completed**: {len(rows)}",
        "",
    ]

    if not rows:
        lines.append("No completed weeks found.")
        return "\n".join(lines)

    total = sum(r["minutes"] for r in rows)
    avg = total / len(rows)
    first_start = rows[0]["start"]
    last_end = rows[-1]["end"]

    lines += [
        f"- **Total time**: {total:.0f} min ({total/60:.1f} h)",
        f"- **Average**: {avg:.0f} min/week",
        f"- **Run period (GMT+8)**: {first_start.strftime('%m/%d %H:%M')} ~ {last_end.strftime('%m/%d %H:%M')}",
        "",
        "## Per-Week Detail",
        "",
        "| Week | Start (GMT+8) | End (GMT+8) | Duration |",
        "|------|---------------|-------------|----------|",
    ]

    for r in rows:
        s = r["start"].strftime("%m/%d %H:%M")
        e = r["end"].strftime("%m/%d %H:%M")
        m = r["minutes"]
        lines.append(f"| {r['week']} | {s} | {e} | {m:.0f}m |")

    lines += [
        "",
        f"**Total: {total:.0f}m ({total/60:.1f}h) | Avg: {avg:.0f}m/week | {len(rows)} weeks**",
    ]

    return "\n".join(lines)


def format_stdout(run_id: str, rows: list[dict], model: str, n_agents: int) -> str:
    """Format compact stdout output."""
    sep = "=" * 68
    lines = [
        sep,
        f"  {run_id} | {model} | {n_agents} agents",
        sep,
        f"  {'Week':<12}{'Start':>16}{'End':>16}{'Duration':>8}",
        f"  {'-'*54}",
    ]
    total = 0
    for r in rows:
        s = r["start"].strftime("%m/%d %H:%M")
        e = r["end"].strftime("%m/%d %H:%M")
        m = r["minutes"]
        total += m
        lines.append(f"  {r['week']:<12}{s:>16}{e:>16}{m:>6.0f}m")

    lines.append(f"  {'-'*54}")
    if rows:
        avg = total / len(rows)
        lines.append(f"  {'Total':<12}{'':>16}{'':>16}{total:>6.0f}m  (avg {avg:.0f}m x {len(rows)}w)")
    else:
        lines.append("  No completed weeks.")
    return "\n".join(lines)


def find_run_ids() -> list[str]:
    """Find all run_ids that have both data/ and logs/ directories."""
    log_dir = Path("logs")
    data_dir = Path("data")
    ids = []
    if log_dir.is_dir():
        for d in sorted(log_dir.iterdir()):
            if d.is_dir() and (data_dir / d.name).is_dir():
                ids.append(d.name)
    return ids


def analyze_run(run_id: str, write_file: bool = True) -> str:
    log_path = find_log_file(run_id)
    if not log_path:
        return f"No world.log found for {run_id}"

    rows = parse_world_log(log_path)
    model = load_model_info(run_id)
    n_agents = count_agents(run_id)

    stdout_text = format_stdout(run_id, rows, model, n_agents)

    if write_file and rows:
        report = format_report(run_id, rows, model, n_agents)
        out_path = Path("data") / run_id / "time_analysis.md"
        out_path.write_text(report, encoding="utf-8")
        stdout_text += f"\n  -> Saved to {out_path}"

    return stdout_text


def main():
    parser = argparse.ArgumentParser(description="Analyze per-week time for simulation runs")
    parser.add_argument("run_ids", nargs="*", help="Run IDs to analyze (e.g. school_03070554)")
    parser.add_argument("--all", action="store_true", help="Analyze all available runs")
    parser.add_argument("--latest", type=int, metavar="N", help="Analyze N most recent runs")
    parser.add_argument("--no-save", action="store_true", help="Only print, don't write files")
    args = parser.parse_args()

    if args.all:
        run_ids = find_run_ids()
    elif args.latest:
        run_ids = find_run_ids()[-args.latest:]
    elif args.run_ids:
        run_ids = args.run_ids
    else:
        parser.print_help()
        sys.exit(1)

    if not run_ids:
        print("No runs found.")
        sys.exit(1)

    for rid in run_ids:
        print(analyze_run(rid, write_file=not args.no_save))
        print()


if __name__ == "__main__":
    main()
