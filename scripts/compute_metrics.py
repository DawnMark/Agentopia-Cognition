#!/usr/bin/env python3
"""Compute per-agent per-week and per-agent per-year metrics from simulation data.

Usage:
    python scripts/compute_metrics.py --data-dir school_03121031
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ALL_METRICS = [
    "input_tokens", "output_tokens", "function_call_count",
    "active_contacts", "passive_contacts",
    "joint_proposed", "joint_participated", "public_participated", "solo_count",
    "total_spending_amount", "activity_consumption_count",
    "extra_earning_count", "skill_improvement_count", "total_skills", "deposit", "deposit_diff",
    # Fulfillment (per-week snapshot)
    "fulfillment_mood", "fulfillment_material", "fulfillment_social", "fulfillment_esteem",
    # Social eval (per-settle-period snapshot, spread to weeks)
    "liked_by_count", "respected_by_count", "likes_count", "respects_count",
    "subjective_n_penalties",
]

# Yearly aggregation rules
_CUMULATIVE = {
    "input_tokens", "output_tokens", "function_call_count",
    "active_contacts", "passive_contacts",
    "joint_proposed", "joint_participated", "public_participated", "solo_count",
    "total_spending_amount", "activity_consumption_count",
    "extra_earning_count", "skill_improvement_count",
    "deposit_diff",
    "subjective_n_penalties",
}
_SNAPSHOT = {
    "total_skills", "deposit",
}
# Snapshot but skip weeks where value is 0 (settle-week placeholders)
_SNAPSHOT_LAST_NONZERO = {
    "fulfillment_mood", "fulfillment_material", "fulfillment_social", "fulfillment_esteem",
    "liked_by_count", "respected_by_count", "likes_count", "respects_count",
}


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> List[dict]:
    """Read a JSONL file, returning list of dicts."""
    if not path.exists():
        return []
    results = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def parse_time(time_str: str) -> Tuple[int, int]:
    """Extract (year, week) from time string like 'Y2020-W01-xxx'."""
    m = re.match(r"Y(\d+)-W(\d+)", time_str)
    if not m:
        raise ValueError(f"Cannot parse time: {time_str}")
    return int(m.group(1)), int(m.group(2))


def week_key(year: int, week: int) -> str:
    return f"Y{year}-W{week:02d}"


def year_key(year: int) -> str:
    return f"Y{year}"


def extract_day(time_str: str) -> int | None:
    """Extract day number from 'Y2020-W01-activity-D3'."""
    m = re.search(r"-D(\d+)", time_str)
    return int(m.group(1)) if m else None


_AGENT_NAME_RE = re.compile(r"^# (.+?)'s Context$", re.MULTILINE)


def extract_agent_from_god_prompt(system_content: str) -> str | None:
    """Extract agent name from '# {name}'s Context' in god system prompt."""
    m = _AGENT_NAME_RE.search(system_content)
    return m.group(1) if m else None


def iter_agent_dirs(data_dir: Path) -> List[Tuple[str, Path]]:
    """Return sorted (agent_name, agent_dir) pairs under persona/."""
    persona_dir = data_dir / "persona"
    return [
        (d.name, d)
        for d in sorted(persona_dir.iterdir())
        if d.is_dir()
    ]


# ---------------------------------------------------------------------------
# State data (read once, used by multiple metric functions)
# ---------------------------------------------------------------------------

def _prefer_fix(agent_dir: Path, basename: str, data_dir: Path | None = None) -> Path:
    """Return *-fix.jsonl if it exists, else the original file."""
    fix = agent_dir / basename.replace(".jsonl", "-fix.jsonl")
    if fix.exists():
        return fix
    return agent_dir / basename


def load_all_state(data_dir: Path) -> Dict[str, List[dict]]:
    """Load state.jsonl for every agent. Returns {agent_name: [records]}.

    Prefers state-fix.jsonl when available (pre-0319 runs with stale-read bug).
    """
    result = {}
    for agent_name, agent_dir in iter_agent_dirs(data_dir):
        records = read_jsonl(_prefer_fix(agent_dir, "state.jsonl", data_dir))
        if records:
            result[agent_name] = records
    return result


def get_week_snapshots(state_records: List[dict]) -> Dict[str, Tuple[dict, dict]]:
    """Group state records by week, return (first_content, last_content) per week."""
    week_records: Dict[str, List[dict]] = defaultdict(list)
    for rec in state_records:
        yr, wk = parse_time(rec["time"])
        week_records[week_key(yr, wk)].append(rec)

    return {
        wk: (records[0]["content"], records[-1]["content"])
        for wk, records in week_records.items()
    }


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_system_metrics(data_dir: Path) -> Dict[str, Dict[str, dict]]:
    """input_tokens, output_tokens, function_call_count from generation/*.jsonl."""
    results: Dict[str, Dict[str, dict]] = defaultdict(
        lambda: defaultdict(lambda: {"input_tokens": 0, "output_tokens": 0, "function_call_count": 0})
    )

    for agent_name, agent_dir in iter_agent_dirs(data_dir):
        gen_dir = agent_dir / "generation"
        if not gen_dir.exists():
            continue

        for jsonl_path in sorted(gen_dir.rglob("*.jsonl"), key=lambda p: p.as_posix()):
            for record in read_jsonl(jsonl_path):
                yr, wk = parse_time(record["time"])
                wk = week_key(yr, wk)
                bucket = results[agent_name][wk]

                bucket["input_tokens"] += record["input_tokens"]
                bucket["output_tokens"] += record["output_tokens"]

                for out_msg in record["outputs"]:
                    tool_calls = out_msg.get("tool_calls")
                    if tool_calls:
                        bucket["function_call_count"] += len(tool_calls)

    return results


def compute_contact_metrics(data_dir: Path) -> Dict[str, Dict[str, dict]]:
    """active_contacts, passive_contacts from contact/sig.jsonl."""
    results: Dict[str, Dict[str, dict]] = defaultdict(
        lambda: defaultdict(lambda: {"active_contacts": 0, "passive_contacts": 0})
    )

    for agent_name, agent_dir in iter_agent_dirs(data_dir):
        for record in read_jsonl(agent_dir / "contact" / "sig.jsonl"):
            yr, wk = parse_time(record["time"])
            wk = week_key(yr, wk)
            bucket = results[agent_name][wk]

            if record["from"] == agent_name:
                bucket["active_contacts"] += 1
            if record["to"] == agent_name:
                bucket["passive_contacts"] += 1

    return results


def compute_activity_metrics(
    data_dir: Path,
    all_state: Dict[str, List[dict]],
    n_day: int,
) -> Dict[str, Dict[str, dict]]:
    """joint_proposed, joint_participated, public_participated, solo_count, spending."""
    results: Dict[str, Dict[str, dict]] = defaultdict(
        lambda: defaultdict(lambda: {
            "joint_proposed": 0, "joint_participated": 0, "public_participated": 0,
            "solo_count": 0, "total_spending_amount": 0, "activity_consumption_count": 0,
        })
    )

    for agent_name, agent_dir in iter_agent_dirs(data_dir):
        # --- Schedule ---
        occupied_days: Dict[str, set] = defaultdict(set)

        for rec in read_jsonl(agent_dir / "schedule.jsonl"):
            activity_time = rec["activity_time"]
            yr, wk = parse_time(activity_time)
            wk_key_ = week_key(yr, wk)
            day = extract_day(activity_time)
            if day is not None:
                occupied_days[wk_key_].add(day)

            rec_type = rec["type"]
            if rec_type == "joint":
                if rec["proposer"] == agent_name:
                    results[agent_name][wk_key_]["joint_proposed"] += 1
                if agent_name in rec["participants"]:
                    results[agent_name][wk_key_]["joint_participated"] += 1
            elif rec_type == "public":
                if agent_name in rec["participants"]:
                    results[agent_name][wk_key_]["public_participated"] += 1

        # --- State-derived: solo_count + spending counts ---
        state_records = all_state.get(agent_name)
        if not state_records:
            continue

        # Solo count from week snapshots
        snapshots = get_week_snapshots(state_records)
        for wk_key_ in snapshots:
            results[agent_name][wk_key_]["solo_count"] = n_day - len(occupied_days[wk_key_])

        # Spending: walk consecutive state records, detect deposit drops
        # Only count drops in -plan (living standard) and -activity- (consumption) stages.
        # Begin-stage drops are artifacts of state initialization, not real spending.
        for i in range(1, len(state_records)):
            prev_deposit = state_records[i - 1]["content"]["assets"]["deposit"]
            cur_deposit = state_records[i]["content"]["assets"]["deposit"]
            if cur_deposit >= prev_deposit:
                continue

            time_str = state_records[i]["time"]
            is_plan = "-plan" in time_str
            is_activity = "-activity-" in time_str
            if not (is_plan or is_activity):
                continue

            yr, wk = parse_time(time_str)
            wk_key_ = week_key(yr, wk)
            drop = prev_deposit - cur_deposit

            results[agent_name][wk_key_]["total_spending_amount"] += drop
            if is_activity:
                results[agent_name][wk_key_]["activity_consumption_count"] += 1

    return results


def compute_growth_metrics(
    data_dir: Path,
    all_state: Dict[str, List[dict]],
) -> Dict[str, Dict[str, dict]]:
    """extra_earning_count, skill_improvement_count, total_skills, deposit, deposit_diff."""
    results: Dict[str, Dict[str, dict]] = defaultdict(
        lambda: defaultdict(lambda: {
            "extra_earning_count": 0, "skill_improvement_count": 0,
            "total_skills": 0, "deposit": 0, "deposit_diff": 0,
        })
    )

    # --- God solo_activity ---
    god_solo_dir = data_dir / "god" / "solo_activity"
    skipped = 0
    if god_solo_dir.exists():
        for jsonl_path in sorted(god_solo_dir.rglob("*.jsonl"), key=lambda p: p.as_posix()):
            for record in read_jsonl(jsonl_path):
                sys_content = record["inputs"][0]["content"]

                # Skip consumption records (different prompt format, no delta_money/delta_skills)
                if "consumption activity" in sys_content:
                    continue

                yr, wk = parse_time(record["time"])
                wk_key_ = week_key(yr, wk)

                agent_name = extract_agent_from_god_prompt(sys_content)
                if not agent_name:
                    skipped += 1
                    continue

                # Parse output (JSON string from god.py's json.dumps)
                raw = record["outputs"][0]["content"]
                if isinstance(raw, str):
                    try:
                        data = json.loads(raw)
                    except (json.JSONDecodeError, ValueError):
                        continue
                else:
                    data = raw

                # LLM output: fields may be absent or have unexpected types
                delta_money = data.get("delta_money", 0)
                if isinstance(delta_money, (int, float)) and delta_money > 0:
                    results[agent_name][wk_key_]["extra_earning_count"] += 1

                delta_skills = data.get("delta_skills") or {}
                if isinstance(delta_skills, dict) and any(
                    isinstance(v, (int, float)) and v > 0 for v in delta_skills.values()
                ):
                    results[agent_name][wk_key_]["skill_improvement_count"] += 1

    if skipped:
        print(f"    WARNING: {skipped} god records skipped (agent name extraction failed)")

    # --- State-based: total_skills, deposit, deposit_diff ---
    # Exclude settle records: settle resets deposit/skills for the new year,
    # which would corrupt the weekly snapshot values.
    for agent_name, state_records in sorted(all_state.items()):
        non_settle = [r for r in state_records if "settle" not in r["time"]]
        snapshots = get_week_snapshots(non_settle)
        sorted_weeks = sorted(snapshots.keys())

        prev_deposit = None
        prev_year = None
        for wk_key_ in sorted_weeks:
            first_content, last_content = snapshots[wk_key_]
            skills = last_content["skills"]
            deposit = last_content["assets"]["deposit"]
            yr, _ = parse_time(wk_key_)

            results[agent_name][wk_key_]["total_skills"] = sum(skills.values())
            results[agent_name][wk_key_]["deposit"] = deposit

            # Use first_content as baseline for the first week overall or
            # the first week of a new year (settle resets deposit between years).
            if prev_deposit is None or yr != prev_year:
                init_deposit = first_content["assets"]["deposit"]
                results[agent_name][wk_key_]["deposit_diff"] = deposit - init_deposit
            else:
                results[agent_name][wk_key_]["deposit_diff"] = deposit - prev_deposit

            prev_deposit = deposit
            prev_year = yr

    return results


def compute_fulfillment_metrics(
    all_state: Dict[str, List[dict]],
) -> Dict[str, Dict[str, dict]]:
    """fulfillment_mood, fulfillment_material, fulfillment_social, fulfillment_esteem from state.

    Skips settle-stage records (auto-initialized placeholders with default 50s).
    """
    results: Dict[str, Dict[str, dict]] = defaultdict(lambda: defaultdict(dict))

    for agent_name, state_records in sorted(all_state.items()):
        # Filter out settle records before grouping
        non_settle = [r for r in state_records if "settle" not in r["time"]]
        snapshots = get_week_snapshots(non_settle)
        for wk_key_, (_, last_content) in snapshots.items():
            fulfillment = last_content["fulfillment"]
            results[agent_name][wk_key_] = {
                "fulfillment_mood": fulfillment["mood"],
                "fulfillment_material": fulfillment["material"],
                "fulfillment_social": fulfillment["social"],
                "fulfillment_esteem": fulfillment["esteem"],
            }

    return results


def compute_social_eval_metrics(data_dir: Path) -> Dict[str, Dict[str, dict]]:
    """liked_by_count, respected_by_count, likes_count, respects_count from reward.jsonl.

    - liked_by_count / respected_by_count: reuse compute_social_metrics() from reward.py
    - likes_count / respects_count: computed here (reward.py has no equivalent)
    """
    from src.world.reward import SocialRanking, SOCIAL_NEUTRAL_SCORE, compute_social_metrics

    # Load rankings per settle week: {wk_key: {agent_name: ranking_dict}}
    rankings_by_week: Dict[str, Dict[str, dict]] = defaultdict(dict)

    for agent_name, agent_dir in iter_agent_dirs(data_dir):
        for rec in read_jsonl(_prefer_fix(agent_dir, "reward.jsonl", data_dir)):
            yr, wk = parse_time(rec["time"])
            wk_key_ = week_key(yr, wk)
            rankings_by_week[wk_key_][agent_name] = rec["ranking"]

    results: Dict[str, Dict[str, dict]] = defaultdict(lambda: defaultdict(dict))

    for wk_key_ in sorted(rankings_by_week):
        agent_rankings = rankings_by_week[wk_key_]

        # Skip weeks with old ranking format (lists instead of score dicts)
        sample_ranking = next(iter(agent_rankings.values()))
        if "affection_scores" not in sample_ranking:
            continue

        # Build SocialRanking list for compute_social_metrics
        sr_list = [
            SocialRanking(
                agent_name=name,
                time=wk_key_,
                affection_scores=ranking["affection_scores"],
                respect_scores=ranking["respect_scores"],
            )
            for name, ranking in sorted(agent_rankings.items())
        ]
        social_metrics = compute_social_metrics(sr_list, wk_key_)

        for agent_name, ranking in agent_rankings.items():
            aff = ranking["affection_scores"]
            resp = ranking["respect_scores"]

            results[agent_name][wk_key_] = {
                "likes_count": sum(1 for v in aff.values() if v >= SOCIAL_NEUTRAL_SCORE),
                "respects_count": sum(1 for v in resp.values() if v >= SOCIAL_NEUTRAL_SCORE),
                "liked_by_count": social_metrics[agent_name].num_people_favor,
                "respected_by_count": social_metrics[agent_name].num_people_respect,
            }

    # n_penalties from reward.jsonl subjective field
    for agent_name, agent_dir in iter_agent_dirs(data_dir):
        for rec in read_jsonl(_prefer_fix(agent_dir, "reward.jsonl", data_dir)):
            yr, wk = parse_time(rec["time"])
            wk_key_ = week_key(yr, wk)
            if agent_name in results and wk_key_ in results[agent_name]:
                results[agent_name][wk_key_]["subjective_n_penalties"] = rec["subjective"]["n_penalties"]
            else:
                results[agent_name][wk_key_] = {
                    "subjective_n_penalties": rec["subjective"]["n_penalties"],
                }

    return results


def compute_innate_metrics(
    data_dir: Path,
    start_year: int,
) -> Dict[str, dict]:
    """Static innate metrics: weekly_income, init_deposit, init_relationship_count.

    Returns {agent_name: {metric: value}} (not per-week).
    """
    results: Dict[str, dict] = {}

    for agent_name, agent_dir in iter_agent_dirs(data_dir):
        profile_path = agent_dir / "profile" / f"year={start_year}.json"
        if not profile_path.exists():
            continue

        with profile_path.open("r", encoding="utf-8") as f:
            profile = json.load(f)

        weekly_income = profile["position"]["weekly_income"] + profile["extra_income"]
        init_deposit = profile["init_assets"]["deposit"]

        # Count initial relationship files
        chars_dir = agent_dir / "memory" / "scratchpad" / "characters"
        if chars_dir.exists():
            init_rel_count = len([p for p in chars_dir.iterdir() if p.suffix == ".jsonl"])
        else:
            init_rel_count = 0

        results[agent_name] = {
            "weekly_income": weekly_income,
            "init_deposit": init_deposit,
            "init_relationship_count": init_rel_count,
        }

    return results


# ---------------------------------------------------------------------------
# Merge + Aggregate
# ---------------------------------------------------------------------------

def merge_metrics(
    active_weeks: Dict[str, set],
    *metric_dicts: Dict[str, Dict[str, dict]],
) -> Dict[str, Dict[str, dict]]:
    """Merge metric dicts, filtering to active weeks only."""
    merged: Dict[str, Dict[str, dict]] = defaultdict(lambda: defaultdict(dict))

    for md in metric_dicts:
        for agent_name, weeks in md.items():
            allowed = active_weeks.get(agent_name, set())
            for wk_key_, metrics in weeks.items():
                if wk_key_ in allowed:
                    merged[agent_name][wk_key_].update(metrics)

    # Fill missing metrics with 0
    for agent_name in merged:
        for wk_key_ in merged[agent_name]:
            for metric in ALL_METRICS:
                merged[agent_name][wk_key_].setdefault(metric, 0)

    return dict(merged)


def find_last_complete_year(
    by_week: Dict[str, Dict[str, dict]],
    n_week: int,
) -> int | None:
    """Find the last year where all agents have exactly n_week weeks of data.

    Returns the year number, or None if no complete year exists.
    """
    # Collect all (year, week_count) per agent
    year_week_counts: Dict[int, List[int]] = defaultdict(list)
    for agent_name, weeks in by_week.items():
        agent_years: Dict[int, int] = defaultdict(int)
        for wk_key_ in weeks:
            yr, _ = parse_time(wk_key_)
            agent_years[yr] += 1
        for yr, count in agent_years.items():
            year_week_counts[yr].append(count)

    # A year is complete if ALL agents have n_week weeks
    complete_years = []
    for yr in sorted(year_week_counts.keys()):
        counts = year_week_counts[yr]
        if all(c == n_week for c in counts):
            complete_years.append(yr)

    return complete_years[-1] if complete_years else None


def aggregate_yearly(
    by_week: Dict[str, Dict[str, dict]],
    last_complete_year: int | None = None,
) -> Dict[str, Dict[str, dict]]:
    """Aggregate per-week into per-year metrics.

    If last_complete_year is set, only include years up to that year.
    """
    by_year: Dict[str, Dict[str, dict]] = {}

    for agent_name, weeks in by_week.items():
        year_weeks: Dict[str, List[str]] = defaultdict(list)
        for wk_key_ in weeks:
            yr, _ = parse_time(wk_key_)
            if last_complete_year is not None and yr > last_complete_year:
                continue
            year_weeks[year_key(yr)].append(wk_key_)

        agent_yearly = {}
        for yr_key, wk_keys in sorted(year_weeks.items()):
            sorted_wks = sorted(wk_keys)
            yearly: Dict[str, Any] = {}

            for metric in _CUMULATIVE:
                yearly[metric] = sum(weeks[wk][metric] for wk in sorted_wks)

            for metric in _SNAPSHOT:
                yearly[metric] = weeks[sorted_wks[-1]][metric]

            for metric in _SNAPSHOT_LAST_NONZERO:
                # Walk backwards to find last week with non-zero value
                val = 0
                for wk in reversed(sorted_wks):
                    v = weeks[wk][metric]
                    if v != 0:
                        val = v
                        break
                yearly[metric] = val

            agent_yearly[yr_key] = yearly
        by_year[agent_name] = agent_yearly

    return by_year


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_one(data_dir_name: str) -> None:
    """Compute and save metrics for a single data_dir."""
    data_dir = ROOT / "data" / data_dir_name
    if not data_dir.exists():
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        return

    # Prefer per-run config snapshot over global config
    run_config_path = data_dir / "config.json"
    config_path = run_config_path if run_config_path.exists() else ROOT / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    n_day = config["world"]["time"]["n_day"]
    n_week = config["world"]["time"]["n_week"]
    start_year = config["world"]["time"]["start_year"]

    print(f"Computing metrics for {data_dir_name} (n_day={n_day}, n_week={n_week})")

    # Load state once, reuse everywhere
    print("  Loading state data...")
    all_state = load_all_state(data_dir)
    active_weeks = {
        name: {week_key(*parse_time(r["time"])) for r in records}
        for name, records in all_state.items()
    }

    print("  Computing system metrics...")
    system_metrics = compute_system_metrics(data_dir)

    print("  Computing contact metrics...")
    contact_metrics = compute_contact_metrics(data_dir)

    print("  Computing activity metrics...")
    activity_metrics = compute_activity_metrics(data_dir, all_state, n_day)

    print("  Computing growth metrics...")
    growth_metrics = compute_growth_metrics(data_dir, all_state)

    print("  Computing fulfillment metrics...")
    fulfillment_metrics = compute_fulfillment_metrics(all_state)

    print("  Computing social eval metrics...")
    social_eval_metrics = compute_social_eval_metrics(data_dir)

    by_week = merge_metrics(
        active_weeks,
        system_metrics, contact_metrics, activity_metrics, growth_metrics,
        fulfillment_metrics, social_eval_metrics,
    )

    # Truncate yearly aggregation to last complete year
    last_complete = find_last_complete_year(by_week, n_week)
    if last_complete is not None:
        print(f"  Last complete year: Y{last_complete}")
    else:
        print("  WARNING: No complete year found, including all years in by_year")

    print("  Aggregating yearly...")
    by_year = aggregate_yearly(by_week, last_complete)

    print("  Computing innate metrics...")
    innate = compute_innate_metrics(data_dir, start_year)

    output_dir = ROOT / "analysis" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{data_dir_name}_metrics.json"

    output = {
        "data_dir": data_dir_name,
        "config": {"n_day": n_day, "n_week": n_week, "start_year": start_year},
        "last_complete_year": last_complete,
        "innate": innate,
        "by_week": by_week,
        "by_year": by_year,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    n_agents = len(by_week)
    n_weeks_total = sum(len(w) for w in by_week.values())
    n_years = len(set(yr for a in by_year.values() for yr in a))
    print(f"Done. {n_agents} agents, {n_weeks_total} agent-weeks, {n_years} years in by_year.")
    print(f"Output: {output_path}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute metrics from simulation data")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Data dir to process (e.g. school_03121031).",
    )
    args = parser.parse_args()

    run_one(args.data_dir)


if __name__ == "__main__":
    main()
