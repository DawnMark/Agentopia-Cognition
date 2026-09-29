#!/usr/bin/env python3
"""Build RFT (rejection sampling fine-tuning) training data from top-advantage trajectories.

It computes life-reward returns & advantages AND builds the RFT training set in one step.

Flow:
1. Calculate returns & advantages from reward history
2. Per-period select top trajectories by advantage
3. Collect generation data for selected (agent, time_range) pairs
4. Output as JSONL with messages format + companion .md report

Usage:
    python scripts/build_rft_data.py --data-dir school
    python scripts/build_rft_data.py --data-dir apartment --top 0.5

The --data-dir is a world run directory under data/ (e.g., school_06031205,
named worldname_MMDDHHMM by generate_run_id). The shipped example worlds are
`school` and `apartment`.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import get_config
from src.world.clock import TimeState
from src.world.reward import (
    calculate_advantages,
    calculate_returns,
)


# =============================================================================
# Load reward history
# =============================================================================


def load_all_reward_history(data_dir: str) -> Dict[str, List[Tuple[str, float]]]:
    """Load reward history from all agents' reward.jsonl files."""
    persona_dir = Path("data") / data_dir / "persona"
    if not persona_dir.exists():
        raise FileNotFoundError(f"Persona directory not found: {persona_dir}")

    results: Dict[str, List[Tuple[str, float]]] = {}
    for agent_dir in sorted(persona_dir.iterdir()):
        if not agent_dir.is_dir():
            continue
        reward_file = agent_dir / "reward.jsonl"
        if not reward_file.exists():
            continue

        history: List[Tuple[str, float]] = []
        with reward_file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                history.append((data["time"], data["total_score"]))

        history.sort(key=lambda x: x[0])
        if history:
            results[agent_dir.name] = history

    return results


# =============================================================================
# Detect last completed week
# =============================================================================


def detect_last_completed_week(data_dir: str) -> Tuple[int, int]:
    """Find the last week that completed the review stage.

    Scans from the latest week backwards. A week is "complete" if at least
    one agent has a "review" entry in that week's generation file.
    """
    persona_dir = Path("data") / data_dir / "persona"
    all_year_weeks: Set[Tuple[int, int]] = set()
    for agent_dir in sorted(persona_dir.iterdir()):
        gen_dir = agent_dir / "generation"
        if not gen_dir.exists():
            continue
        for year_dir in sorted(gen_dir.iterdir()):
            if not year_dir.name.startswith("year="):
                continue
            year = int(year_dir.name.split("=")[1])
            for week_file in sorted(year_dir.glob("week=*.jsonl")):
                week = int(week_file.stem.split("=")[1])
                all_year_weeks.add((year, week))

    if not all_year_weeks:
        raise FileNotFoundError(f"No generation data found in {persona_dir}")

    for year, week in sorted(all_year_weeks, reverse=True):
        for agent_dir in persona_dir.iterdir():
            gen_file = (
                agent_dir / "generation" / f"year={year}" / f"week={week}.jsonl"
            )
            if not gen_file.exists():
                continue
            with gen_file.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    if "-review" in json.loads(line)["time"]:
                        return year, week

    raise ValueError("No completed week found in generation data")


# =============================================================================
# Per-period trajectory selection
# =============================================================================


def select_top_per_period(
    advantages: Dict[str, List[Tuple[str, str, float]]],
    top_fraction: float,
) -> List[Tuple[str, str, str, float]]:
    """Select top trajectories per period, avoiding first-year bias.

    A_0 = Return_1 - 0 (virtual zero point) is naturally larger than later
    advantages. Per-period selection ensures each period contributes.
    """
    by_period: Dict[Tuple[str, str], List[Tuple[str, float]]] = defaultdict(list)
    for agent, adv_list in advantages.items():
        for start_time, end_time, adv in adv_list:
            by_period[(start_time, end_time)].append((agent, adv))

    result: List[Tuple[str, str, str, float]] = []
    for (start, end), items in sorted(by_period.items()):
        items.sort(key=lambda x: (-x[1], x[0]))
        n_select = max(1, int(len(items) * top_fraction))
        for agent, adv in items[:n_select]:
            result.append((agent, start, end, adv))

    result.sort(key=lambda x: (-x[3], x[0], x[1]))
    return result


# =============================================================================
# Generation data -> RFT format
# =============================================================================


def parse_advantage_year(start_time: str, end_time: str) -> int:
    """Extract the year covered by an advantage period."""
    return TimeState.from_string(end_time).year


def get_generation_weeks_for_year(
    data_dir: str, agent_name: str, year: int, max_year: int, max_week: int
) -> List[Path]:
    """Get generation JSONL files for an agent in a year, up to (max_year, max_week)."""
    gen_dir = (
        Path("data") / data_dir / "persona" / agent_name / "generation" / f"year={year}"
    )
    if not gen_dir.exists():
        return []

    files = []
    for f in sorted(gen_dir.glob("week=*.jsonl")):
        week = int(f.stem.split("=")[1])
        if year < max_year or (year == max_year and week <= max_week):
            files.append(f)
    return files


def load_generation_entries(filepath: Path) -> List[Dict]:
    """Load generation entries from a JSONL file, skipping rejected records."""
    entries = []
    with filepath.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("rejected"):
                continue
            entries.append(record)
    return entries


def generation_to_messages(entry: Dict) -> Dict:
    """Convert a generation entry to messages format.

    messages = inputs + outputs, preserving full multi-turn tool interactions.
    """
    return {"messages": entry["inputs"] + entry["outputs"]}


# =============================================================================
# God model sampling
# =============================================================================


def count_output_tokens(samples: List[Dict]) -> int:
    """Count total output tokens across training samples.

    Uses output_tokens from metadata if available, otherwise estimates
    from assistant message content length (chars / 3).
    """
    total = 0
    for sample in samples:
        meta = sample.get("metadata", {})
        if "output_tokens" in meta:
            total += meta["output_tokens"]
        else:
            for m in sample["messages"]:
                if m["role"] == "assistant":
                    total += len(m.get("content", "")) // 3
    return total


def load_god_entries(data_dir: str) -> Dict[str, List[Dict]]:
    """Load all god model generation entries grouped by feature.

    Returns:
        Dict[feature_name, List[entry]] where each entry has
        {time, inputs, outputs, input_tokens, output_tokens}
    """
    god_dir = Path("data") / data_dir / "god"
    if not god_dir.exists():
        return {}

    entries_by_feature: Dict[str, List[Dict]] = {}
    for feature_dir in sorted(god_dir.iterdir()):
        if not feature_dir.is_dir():
            continue
        feature = feature_dir.name
        entries = []
        for f in sorted(feature_dir.rglob("*.jsonl")):
            with f.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if record.get("rejected"):
                        continue
                    entries.append(record)
        if entries:
            entries_by_feature[feature] = entries

    return entries_by_feature


def sample_god_data(
    god_entries: Dict[str, List[Dict]],
    target_output_tokens: int,
    seed: int = 42,
) -> List[Dict]:
    """Sample god model entries to reach target output tokens.

    Strategy: include ALL entries from small features (for diversity),
    then sample from the largest feature (joint_activity) to fill remaining budget.
    """
    rng = random.Random(seed)

    # Sort features by total output tokens (ascending)
    feature_tokens: List[Tuple[str, int]] = []
    for feature, entries in god_entries.items():
        total = sum(e.get("output_tokens", 0) for e in entries)
        feature_tokens.append((feature, total))
    feature_tokens.sort(key=lambda x: x[1])

    selected: List[Dict] = []
    remaining_budget = target_output_tokens

    # Include all entries from smaller features until budget runs low
    for feature, total_tokens in feature_tokens[:-1]:  # All except largest
        entries = god_entries[feature]
        if total_tokens <= remaining_budget:
            for entry in entries:
                sample = generation_to_messages(entry)
                sample["metadata"] = {
                    "source": "god",
                    "feature": feature,
                    "time": entry["time"],
                    "output_tokens": entry.get("output_tokens", 0),
                }
                selected.append(sample)
            remaining_budget -= total_tokens
        else:
            # Even this "small" feature exceeds budget, sample from it
            rng.shuffle(entries)
            for entry in entries:
                tok = entry.get("output_tokens", 0)
                if remaining_budget <= 0:
                    break
                sample = generation_to_messages(entry)
                sample["metadata"] = {
                    "source": "god",
                    "feature": feature,
                    "time": entry["time"],
                    "output_tokens": tok,
                }
                selected.append(sample)
                remaining_budget -= tok

    # Sample from the largest feature to fill remaining budget
    if remaining_budget > 0 and feature_tokens:
        largest_feature = feature_tokens[-1][0]
        entries = list(god_entries[largest_feature])
        rng.shuffle(entries)
        for entry in entries:
            if remaining_budget <= 0:
                break
            tok = entry.get("output_tokens", 0)
            sample = generation_to_messages(entry)
            sample["metadata"] = {
                "source": "god",
                "feature": largest_feature,
                "time": entry["time"],
                "output_tokens": tok,
            }
            selected.append(sample)
            remaining_budget -= tok

    return selected


# =============================================================================
# Report generation
# =============================================================================


def generate_report(
    data_dir: str,
    last_year: int,
    last_week: int,
    top_fraction: float,
    reward_history: Dict[str, List[Tuple[str, float]]],
    advantages: Dict[str, List[Tuple[str, str, float]]],
    top_trajs: List[Tuple[str, str, str, float]],
    n_samples: int,
    output_path: str,
) -> str:
    """Generate markdown report for the RFT data."""
    lines: List[str] = []
    lines.append(f"# RFT Data Report: {data_dir}")
    lines.append(f"")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"")

    # Summary
    lines.append(f"## Summary")
    lines.append(f"")
    lines.append(f"| Key | Value |")
    lines.append(f"|-----|-------|")
    lines.append(f"| Data dir | `{data_dir}` |")
    lines.append(f"| Last completed week | Y{last_year}-W{last_week:02d} |")
    lines.append(f"| Agents | {len(reward_history)} |")
    n_periods = max(len(v) for v in advantages.values()) if advantages else 0
    lines.append(f"| Reward periods | {n_periods} |")
    lines.append(f"| Selection | top {top_fraction:.0%} per period |")
    lines.append(f"| Selected trajectories | {len(top_trajs)} |")
    lines.append(f"| Total RFT samples | {n_samples} |")
    lines.append(f"| Output | `{output_path}` |")
    lines.append(f"")

    # Reward at each settle point
    lines.append(f"## Reward Rankings by Period")
    lines.append(f"")
    rewards_by_time: Dict[str, Dict[str, float]] = defaultdict(dict)
    for agent, history in reward_history.items():
        for time_str, score in history:
            rewards_by_time[time_str][agent] = score

    for time_str in sorted(rewards_by_time.keys()):
        agents = rewards_by_time[time_str]
        ranked = sorted(agents.items(), key=lambda x: -x[1])
        mean_score = sum(agents.values()) / len(agents)
        lines.append(f"### {time_str} (mean={mean_score:.2f})")
        lines.append(f"")
        lines.append(f"| Rank | Agent | Score |")
        lines.append(f"|------|-------|-------|")
        for i, (agent, score) in enumerate(ranked):
            lines.append(f"| {i+1} | {agent} | {score:.2f} |")
        lines.append(f"")

    # Advantage per period
    lines.append(f"## Advantage per Period")
    lines.append(f"")
    selected_set = {(a, s, e) for a, s, e, _ in top_trajs}
    by_period: Dict[Tuple[str, str], List[Tuple[str, float]]] = defaultdict(list)
    for agent, adv_list in advantages.items():
        for start, end, adv in adv_list:
            by_period[(start, end)].append((agent, adv))

    for (start, end), items in sorted(by_period.items()):
        items.sort(key=lambda x: (-x[1], x[0]))
        mean_adv = sum(a for _, a in items) / len(items)
        lines.append(f"### {start} -> {end} (mean={mean_adv:.4f})")
        lines.append(f"")
        lines.append(f"| Rank | Agent | Advantage | Selected |")
        lines.append(f"|------|-------|-----------|----------|")
        for i, (agent, adv) in enumerate(items):
            sel = "Y" if (agent, start, end) in selected_set else ""
            lines.append(f"| {i+1} | {agent} | {adv:.4f} | {sel} |")
        lines.append(f"")

    # Selected trajectory details
    lines.append(f"## Selected Trajectories Detail")
    lines.append(f"")
    for agent, start, end, adv in top_trajs:
        # Reward rank at end time
        end_rewards = rewards_by_time.get(end, {})
        if end_rewards:
            rank = sorted(end_rewards.keys(), key=lambda a: -end_rewards[a])
            agent_rank = rank.index(agent) + 1
            agent_score = end_rewards[agent]
        else:
            agent_rank, agent_score = 0, 0.0

        lines.append(f"**{agent}**: {start} -> {end}")
        lines.append(f"- Advantage: {adv:.4f}")
        if end_rewards:
            lines.append(f"- Reward at {end}: {agent_score:.2f} (rank {agent_rank}/{len(end_rewards)})")
        # All advantages for this agent
        lines.append(f"- All periods: {', '.join(f'{a:.2f}' for _, _, a in advantages[agent])}")
        lines.append(f"")

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build RFT data from top-advantage trajectories"
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="World run directory name under data/ (e.g., school, apartment, "
             "or a timestamped run like school_06031205)",
    )
    parser.add_argument(
        "--top",
        type=float,
        default=None,
        help="Override top fraction of trajectories per period "
             "(default: reward.rft_top_fraction from config)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output path (default: auto-generated under rft_data/)",
    )
    parser.add_argument(
        "--n-year",
        type=int,
        default=None,
        help="Only use first N years of data (e.g., --n-year 4)",
    )
    args = parser.parse_args()

    config = get_config()
    top_fraction = args.top or config["world"]["reward"]["rft_top_fraction"]
    start_year = config["world"]["time"]["start_year"]
    n_week = config["world"]["time"]["n_week"]

    # Step 1: Detect last completed week
    last_year, last_week = detect_last_completed_week(args.data_dir)
    print(f"Last completed week: Y{last_year}-W{last_week:02d}")

    # Apply --n-year cutoff
    if args.n_year is not None:
        cutoff_year = start_year + args.n_year - 1
        cutoff_week = n_week
        if cutoff_year < last_year or (cutoff_year == last_year and cutoff_week < last_week):
            last_year, last_week = cutoff_year, cutoff_week
            print(f"  --n-year={args.n_year}: truncated to Y{last_year}-W{last_week:02d}")

    # Step 2: Load rewards & calculate advantages
    print("Loading reward history...")
    reward_history = load_all_reward_history(args.data_dir)

    # Truncate reward history to last_year
    if args.n_year is not None:
        cutoff_yw = (last_year, last_week)
        for agent in reward_history:
            reward_history[agent] = [
                (t, s) for t, s in reward_history[agent]
                if (TimeState.from_string(t).year, TimeState.from_string(t).week) <= cutoff_yw
            ]
        reward_history = {a: h for a, h in reward_history.items() if h}

    print(f"Loaded {len(reward_history)} agents")

    if not reward_history:
        print("No reward history found. Exiting.")
        return

    returns = calculate_returns(reward_history, normalize=True)
    advantages = calculate_advantages(returns)
    n_adv = sum(len(v) for v in advantages.values())
    print(f"Calculated {n_adv} advantages across {len(advantages)} agents")

    # Step 3: Select top trajectories (per-period to avoid first-year bias)
    top_trajs = select_top_per_period(advantages, top_fraction)
    print(f"Selected {len(top_trajs)} top trajectories (top {top_fraction:.0%} per period)")

    if not top_trajs:
        print("No trajectories selected. Exiting.")
        return

    adv_values = [t[3] for t in top_trajs]
    print(f"  Advantage range: [{min(adv_values):.4f}, {max(adv_values):.4f}]")
    for agent, start, end, adv in top_trajs:
        print(f"    {agent}: {start} -> {end}, advantage={adv:.4f}")

    # Step 4: Collect generation data for selected trajectories
    rft_samples = []
    for agent, start_time, end_time, advantage in top_trajs:
        year = parse_advantage_year(start_time, end_time)
        gen_files = get_generation_weeks_for_year(
            args.data_dir, agent, year, last_year, last_week
        )

        n_entries = 0
        for gen_file in gen_files:
            for entry in load_generation_entries(gen_file):
                sample = generation_to_messages(entry)
                sample["metadata"] = {
                    "agent": agent,
                    "time": entry["time"],
                    "advantage": advantage,
                    "period": {"start": start_time, "end": end_time},
                }
                rft_samples.append(sample)
                n_entries += 1

        print(f"  {agent} year={year}: {len(gen_files)} files, {n_entries} entries")

    print(f"\nTotal RFT samples: {len(rft_samples)}")

    # Step 5: Save JSONL
    output_path = args.output or (
        f"rft_data/{args.data_dir}_Y{last_year}W{last_week:02d}.jsonl"
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in rft_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Saved to {output_path}")

    # Step 6: Save companion .md report
    report = generate_report(
        data_dir=args.data_dir,
        last_year=last_year,
        last_week=last_week,
        top_fraction=top_fraction,
        reward_history=reward_history,
        advantages=advantages,
        top_trajs=top_trajs,
        n_samples=len(rft_samples),
        output_path=output_path,
    )
    md_path = output_path.replace(".jsonl", ".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Report saved to {md_path}")

    # Step 7: Sample god model data (20% of roleplay output tokens)
    print("\n--- God Model Data ---")
    god_entries = load_god_entries(args.data_dir)
    if god_entries:
        # Downsample high-volume features to 10%
        downsample_features = {"joint_activity": 0.1, "solo_activity": 0.1}
        ds_rng = random.Random(42)
        for feat, ratio in downsample_features.items():
            if feat not in god_entries:
                continue
            entries = god_entries[feat]
            n_orig = len(entries)
            n_keep = max(1, int(n_orig * ratio))
            ds_rng.shuffle(entries)
            god_entries[feat] = entries[:n_keep]
            print(f"  {feat}: downsampled {n_orig} -> {n_keep} ({ratio:.0%})")
        roleplay_output_tokens = count_output_tokens(rft_samples)
        god_target = int(roleplay_output_tokens * 0.2)
        god_total = sum(
            sum(e.get("output_tokens", 0) for e in entries)
            for entries in god_entries.values()
        )
        print(f"Roleplay output tokens: ~{roleplay_output_tokens:,}")
        print(f"God target (20%): ~{god_target:,}")
        print(f"God total available: {god_total:,}")

        god_samples = sample_god_data(god_entries, god_target)
        god_actual_tokens = sum(
            s["metadata"].get("output_tokens", 0) for s in god_samples
        )

        # Print per-feature stats
        god_feature_counts: Dict[str, Tuple[int, int]] = defaultdict(lambda: (0, 0))
        for s in god_samples:
            feat = s["metadata"]["feature"]
            tok = s["metadata"].get("output_tokens", 0)
            n, t = god_feature_counts[feat]
            god_feature_counts[feat] = (n + 1, t + tok)

        for feat in sorted(god_feature_counts.keys()):
            n, t = god_feature_counts[feat]
            total_feat = len(god_entries.get(feat, []))
            print(f"  {feat}: {n}/{total_feat} entries, {t:,} tokens")

        print(f"God samples: {len(god_samples)} entries, {god_actual_tokens:,} output tokens")

        # Save god data
        god_output_path = output_path.replace(
            f"rft_data/{args.data_dir}",
            f"rft_data/god_{args.data_dir}",
        )
        with open(god_output_path, "w", encoding="utf-8") as f:
            for sample in god_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"Saved to {god_output_path}")
    else:
        print("No god model data found.")


if __name__ == "__main__":
    main()
