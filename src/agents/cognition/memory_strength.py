"""Strength, decay, tiers and the protected list (design doc 6, draft §6).

All pure functions: no clock, no randomness, no LLM. Given the same event
history they produce the same numbers, which is what makes the strength view
rebuildable and the weekly settlement replayable.

Two rules from the design document are visible here:

- **being placed in a prompt is not reinforcement** — phase 2 does not inject
  anything, so only recorded evidence (source events, explicit reads) counts;
- **forgetting lowers accessibility, it never deletes history** — the tier is a
  classification, and the event stream keeps every memory forever.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from src.agents.cognition.memory_models import MEMORY_TIERS

# Tier thresholds (design draft §6.2).
HOT_THRESHOLD = 0.6
WARM_THRESHOLD = 0.3
COLD_THRESHOLD = 0.1

# Decay: strength halves every HALF_LIFE_WEEKS simulated weeks.
DEFAULT_HALF_LIFE_WEEKS = 8

# Reinforcement weights (design draft §6.1). "used in plan" stays 0 in phase 2.
W_READ = 0.15
W_USED_IN_PLAN = 0.25
W_POSITIVE_OUTCOME = 0.4
W_SOURCE_EVENTS = 0.1

CONTRADICTION_PENALTY = 0.15
GOAL_RELEVANCE_FLOOR = 0.8
GOAL_RELEVANCE_CEILING = 1.2

# Protected memories never fall below this strength and never archive.
PROTECTED_MIN_STRENGTH = 0.35
PROTECTED_EMOTION_STRONG = 0.6
PROTECTED_EMOTION_MAJOR = 0.8


def decay_factor(weeks_since_created: int, *, half_life_weeks: int = DEFAULT_HALF_LIFE_WEEKS) -> float:
    """0.5 ** (weeks / half_life), clamped for absurd inputs."""
    weeks = max(0, int(weeks_since_created))
    half_life = max(1, int(half_life_weeks))
    return float(0.5 ** (weeks / half_life))


def tier_for(strength: float) -> str:
    if strength >= HOT_THRESHOLD:
        return "hot"
    if strength >= WARM_THRESHOLD:
        return "warm"
    if strength >= COLD_THRESHOLD:
        return "cold"
    return "archived"


@dataclass
class StrengthFactors:
    """Every input that produced a strength value, for auditability."""

    salience: float
    decay: float
    read_count: int
    used_in_plan_count: int
    positive_outcome_count: int
    source_event_count: int
    goal_relevance: float
    contradiction_penalty: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "salience": round(self.salience, 4),
            "decay": round(self.decay, 4),
            "read_count": int(self.read_count),
            "used_in_plan_count": int(self.used_in_plan_count),
            "positive_outcome_count": int(self.positive_outcome_count),
            "source_event_count": int(self.source_event_count),
            "goal_relevance": round(self.goal_relevance, 4),
            "contradiction_penalty": round(self.contradiction_penalty, 4),
        }


def compute_strength(
    *,
    salience: float,
    weeks_since_created: int,
    read_count: int = 0,
    used_in_plan_count: int = 0,
    positive_outcome_count: int = 0,
    source_event_count: int = 1,
    goal_relevance: float = 1.0,
    contradictions: int = 0,
    protected: bool = False,
    half_life_weeks: int = DEFAULT_HALF_LIFE_WEEKS,
) -> tuple[float, StrengthFactors]:
    """Return (strength, factors) for one memory.

    Phase 2 note: `used_in_plan_count` is always 0 because there are no
    structured goals yet (design draft Q4). The factor is in the formula so
    phase 3 can switch it on without changing the contract.
    """
    decay = decay_factor(weeks_since_created, half_life_weeks=half_life_weeks)
    relevance = max(GOAL_RELEVANCE_FLOOR, min(GOAL_RELEVANCE_CEILING, float(goal_relevance)))
    contradiction_penalty = CONTRADICTION_PENALTY * max(0, int(contradictions))

    reinforcement = (
        1.0
        + W_READ * max(0, int(read_count))
        + W_USED_IN_PLAN * max(0, int(used_in_plan_count))
        + W_POSITIVE_OUTCOME * max(0, int(positive_outcome_count))
        + W_SOURCE_EVENTS * max(0, int(source_event_count) - 1)
    )

    strength = max(0.0, min(1.0, float(salience) * decay * reinforcement * relevance - contradiction_penalty))
    if protected:
        strength = max(strength, PROTECTED_MIN_STRENGTH)

    factors = StrengthFactors(
        salience=float(salience),
        decay=decay,
        read_count=int(read_count),
        used_in_plan_count=int(used_in_plan_count),
        positive_outcome_count=int(positive_outcome_count),
        source_event_count=int(source_event_count),
        goal_relevance=relevance,
        contradiction_penalty=contradiction_penalty,
    )
    return round(strength, 4), factors


def is_protected(
    *,
    kind: str,
    emotion: float,
    status: str,
    has_active_commitment: bool = False,
    is_identity: bool = False,
    is_current_asset: bool = False,
    is_long_term_goal: bool = False,
) -> bool:
    """Whether ordinary forgetting may not touch this memory (design doc 6).

    Inputs are structured flags only — no LLM judgement — so the rule is
    reproducible and auditable.
    """
    if is_identity or is_current_asset or has_active_commitment or is_long_term_goal:
        return True
    if kind == "relationship" and abs(float(emotion)) >= PROTECTED_EMOTION_STRONG:
        return True
    if abs(float(emotion)) >= PROTECTED_EMOTION_MAJOR:
        return True
    if kind == "goal" and status == "active":
        return True
    return False


def tier_after(previous_tier: str, new_tier: str, *, protected: bool) -> str:
    """Apply the tier transition, honouring the protected floor."""
    if protected and new_tier == "archived":
        return "cold"
    if previous_tier not in MEMORY_TIERS:
        return new_tier
    return new_tier


def weeks_between(time_a: str, time_b: str) -> int:
    """Simulated weeks between two `Y<year>-W<week>...` stamps (>= 0)."""
    a = _parse_week(time_a)
    b = _parse_week(time_b)
    if a is None or b is None:
        return 0
    weeks_a = a[0] * 100 + a[1]
    weeks_b = b[0] * 100 + b[1]
    # Weeks are numbered 0..n_week inside a year; a year is 100 units apart in
    # this encoding, which is exact enough for decay purposes and never negative.
    return max(0, abs(weeks_b - weeks_a))


def _parse_week(stamp: str) -> Any:
    parts = str(stamp).split("-")
    if len(parts) < 2:
        return None
    year_part, week_part = parts[0], parts[1]
    if not year_part.startswith("Y") or not week_part.startswith("W"):
        return None
    try:
        return int(year_part[1:]), int(week_part[1:])
    except ValueError:
        return None


def settle_strengths(
    memories: Iterable[Dict[str, Any]],
    *,
    current_time: str,
    positive_outcome_counts: Dict[str, int] | None = None,
    read_counts: Dict[str, int] | None = None,
    contradiction_counts: Dict[str, int] | None = None,
    used_in_plan_counts: Dict[str, int] | None = None,
    goal_relevance: Dict[str, float] | None = None,
    half_life_weeks: int = DEFAULT_HALF_LIFE_WEEKS,
) -> List[Dict[str, Any]]:
    """Recompute strength/tier for every memory at a point in time.

    Returns a list of `{memory_id, strength, tier, factors}` for the memories
    whose values changed — the caller turns those into events.
    """
    positive_outcome_counts = positive_outcome_counts or {}
    read_counts = read_counts or {}
    contradiction_counts = contradiction_counts or {}
    used_in_plan_counts = used_in_plan_counts or {}
    goal_relevance = goal_relevance or {}

    updates: List[Dict[str, Any]] = []
    for memory in memories:
        memory_id = str(memory.get("memory_id") or "")
        if not memory_id or memory.get("status") == "archived":
            continue
        weeks = weeks_between(str(memory.get("created_at") or ""), current_time)
        strength, factors = compute_strength(
            salience=float(memory.get("salience", 0.5)),
            weeks_since_created=weeks,
            read_count=read_counts.get(memory_id, int(memory.get("read_count", 0))),
            # KI-10: real use (an adopted method citing this memory), not mere
            # placement in a prompt (design doc invariant #6).
            used_in_plan_count=used_in_plan_counts.get(
                memory_id, int(memory.get("used_in_plan_count", 0))
            ),
            positive_outcome_count=positive_outcome_counts.get(memory_id, 0),
            source_event_count=len(memory.get("source_event_ids") or []) or 1,
            goal_relevance=goal_relevance.get(memory_id, 1.0),
            contradictions=contradiction_counts.get(memory_id, 0),
            protected=bool(memory.get("protected")),
            half_life_weeks=half_life_weeks,
        )
        tier = tier_after(
            str(memory.get("tier") or "warm"), tier_for(strength), protected=bool(memory.get("protected"))
        )
        if abs(float(memory.get("strength", -1.0)) - strength) < 1e-9 and memory.get("tier") == tier:
            continue
        updates.append(
            {
                "memory_id": memory_id,
                "strength": strength,
                "tier": tier,
                "factors": factors.to_dict(),
            }
        )
    return updates
