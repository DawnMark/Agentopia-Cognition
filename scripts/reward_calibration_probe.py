#!/usr/bin/env python
"""Offline calibration probe for the KI-8 centred reward (phase 5 acceptance).

The shadow distribution run is supposed to answer one question before
`RewardWeights.use_baseline` is switched on: does centring produce a reward
distribution that can actually separate methods — both signs present
(`positive_reward_share` 0.5-0.8), a tail at or below -0.1, and `deprecated`
(value <= -0.25) reachable?

Run `09270135` answered **no**, and the reason is structural rather than
data-thin: the centred `quality` term has a positive floor (`0.5 + 0.5*delta`,
so it never goes below 0), the `skill_gain` term is positive-only *and* the
largest weight, `BASELINE_FULL_SCALE` is wider than the proxy's whole dynamic
range, and `time_cost` is a constant placeholder because activity records carry
no `turns`. The cheapest way to see that is to replay the same outcomes under
alternative scalings, which is what this probe does — no LLM calls, no new run.

It reads `METHOD_OUTCOME_OBSERVED` events (whose `shadow_components` carry every
baseline-independent term), rebuilds the baseline cascade in time order, and
prints the resulting distribution plus the projected `deprecated`/`validated`
transitions for each candidate.

Usage:
    .venv/Scripts/python.exe scripts/reward_calibration_probe.py --data-dir shanghai_apartment_09270135
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.cognition.reward_model import (  # noqa: E402
    BaselineTracker,
    OutcomeSignals,
    RewardWeights,
    compute_reward,
    update_value,
)


def load_outcomes(run_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(str(run_dir / "persona" / "*" / "cognition" / "capability_events.jsonl"))):
        persona = Path(path).parts[-3]
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("type") != "METHOD_OUTCOME_OBSERVED":
                continue
            if row.get("shadow_components") is None:
                continue
            row["_persona"] = persona
            rows.append(row)
    rows.sort(key=lambda r: (str(r.get("_persona")), str(r.get("time") or "")))
    return rows


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def signals_from_event(row: Dict[str, Any]) -> OutcomeSignals:
    """Rebuild the objective signals an outcome event recorded.

    Everything `compute_reward` reads is in the payload (`skill_gains`,
    `delta_vitality`, `delta_money`, `turns`, `rejections`) except the
    fulfillment deltas, which only feed the social cost term. They are read from
    `delta_fulfillment` when a run recorded it and default to none otherwise —
    measured on run `09270135`, the social cost was 0 for all 75 outcomes.
    """
    gains: Dict[str, float] = {}
    for name, delta in (row.get("skill_gains") or {}).items():
        try:
            gains[str(name)] = float(delta)
        except (TypeError, ValueError):
            continue
    fulfillment: Dict[str, int] = {}
    for name, delta in (row.get("delta_fulfillment") or {}).items():
        try:
            fulfillment[str(name)] = int(delta)
        except (TypeError, ValueError):
            continue

    def _int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    goal: Optional[float] = None
    if row.get("goal_progress") is not None:
        try:
            goal = float(row["goal_progress"])
        except (TypeError, ValueError):
            goal = None

    return OutcomeSignals(
        activity_id=str(row.get("activity_id") or ""),
        activity_type=str(row.get("activity_type") or ""),
        skill_gains=gains,
        delta_vitality=_int(row.get("delta_vitality")),
        delta_fulfillment=fulfillment,
        delta_money=_int(row.get("delta_money")),
        verification_rejections=_int(row.get("rejections")),
        turns=_int(row.get("turns")),
        goal_progress=goal,
    )


def replay(
    outcomes: List[Dict[str, Any]],
    *,
    weights: RewardWeights,
    skill_id_for: Optional[Callable[[Dict[str, Any]], str]] = None,
) -> Tuple[List[float], Dict[str, int]]:
    """Replay one candidate formula over the recorded outcomes.

    Uses the shipped `compute_reward`, so the probe cannot drift away from the
    code it is calibrating: the candidate *is* a weight configuration.
    """
    tracker = BaselineTracker()
    rewards: List[float] = []
    transitions: Dict[str, int] = {}
    state: Dict[str, Dict[str, Any]] = {}

    for row in outcomes:
        signals = signals_from_event(row)
        context_key = str(row.get("context_key") or "default")
        activity_type = str(row.get("activity_type") or "")
        baseline = tracker.baseline_for(context_key, activity_type=activity_type)
        skill_id = skill_id_for(row) if skill_id_for else ""
        components = compute_reward(
            signals,
            skill_id=skill_id,
            weights=weights,
            baseline=baseline,
            baseline_samples=tracker.samples(context_key, activity_type=activity_type),
        )
        reward = components.total
        rewards.append(reward)

        method_id = str(row.get("method_id") or "")
        current = state.setdefault(
            method_id, {"value": 0.0, "practice": 0, "success": 0, "status": "proposed"}
        )
        update = update_value(
            value=current["value"],
            practice_count=current["practice"],
            success_count=current["success"],
            reward=reward,
            current_status=current["status"],
        )
        if update.status != current["status"]:
            key = f"{current['status']}->{update.status}"
            transitions[key] = transitions.get(key, 0) + 1
        current.update(
            value=update.value,
            practice=update.practice_count,
            success=update.success_count,
            status=update.status,
        )

        tracker.observe(
            context_key,
            max(0.0, min(1.0, signals.total_positive_gain() / 6.0)),
            activity_type=activity_type,
        )

    return rewards, transitions


def summarise(name: str, rewards: List[float], transitions: Dict[str, int]) -> Dict[str, Any]:
    negative = [r for r in rewards if r < 0]
    return {
        "candidate": name,
        "outcomes": len(rewards),
        "positive_share": round(sum(1 for r in rewards if r > 0) / len(rewards), 4),
        "negative": len(negative),
        "below_minus_0_1": sum(1 for r in rewards if r <= -0.1),
        "min": round(min(rewards), 4),
        "max": round(max(rewards), 4),
        "mean": round(st.mean(rewards), 4),
        "sd": round(st.pstdev(rewards), 4),
        "deprecated_reachable": any("deprecated" in key for key in transitions),
        "transitions": dict(sorted(transitions.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--json", help="Optional JSON output path")
    args = parser.parse_args()

    run_dir = Path(args.data_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / "data" / run_dir
    if not run_dir.exists():
        raise SystemExit(f"run not found: {run_dir}")

    outcomes = load_outcomes(run_dir)
    if not outcomes:
        raise SystemExit("no METHOD_OUTCOME_OBSERVED rows carrying shadow_components")

    candidates: List[Tuple[str, Callable[[], Tuple[List[float], Dict[str, int]]]]] = [
        (
            "as-designed: use_baseline (floored quality)",
            lambda: replay(outcomes, weights=RewardWeights(use_baseline=True)),
        ),
        (
            "signed performance (centred_performance)",
            lambda: replay(
                outcomes,
                weights=RewardWeights(use_baseline=True, centred_performance=True),
            ),
        ),
        (
            "signed performance + goal_progress 0.30",
            lambda: replay(
                outcomes,
                weights=RewardWeights(
                    use_baseline=True, centred_performance=True, goal_progress_weight=0.30
                ),
            ),
        ),
        (
            "phase 1-3 (live, no baseline)",
            lambda: replay(outcomes, weights=RewardWeights()),
        ),
    ]

    results = [summarise(name, *fn()) for name, fn in candidates]

    print(f"# Reward calibration probe — {run_dir.name}")
    print(f"# {len(outcomes)} recorded outcomes with a centred score\n")
    header = f"{'candidate':44s} {'pos':>6s} {'neg':>4s} {'<=-.1':>5s} {'min':>8s} {'max':>7s} {'mean':>7s} {'sd':>6s} {'depr':>5s}"
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{row['candidate']:44s} {row['positive_share']:6.2f} {row['negative']:4d} "
            f"{row['below_minus_0_1']:5d} {row['min']:8.4f} {row['max']:7.4f} "
            f"{row['mean']:7.4f} {row['sd']:6.4f} {str(row['deprecated_reachable']):>5s}"
        )
    print("\nprojected status transitions")
    for row in results:
        print(f"  {row['candidate']}: {row['transitions']}")

    if args.json:
        target = Path(args.json)
        if not target.is_absolute():
            target = ROOT / target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {"run": run_dir.name, "outcomes": len(outcomes), "candidates": results},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {target}")


if __name__ == "__main__":
    main()
