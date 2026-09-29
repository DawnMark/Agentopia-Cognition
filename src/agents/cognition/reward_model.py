"""Objective outcome -> method reward -> value update (design doc 7.4).

This module is the only place allowed to move a methodology's numbers, and it
only accepts **objective** signals that were already recorded in the ledger:

    skill gains assigned by the environment model
    vitality / fulfillment / money deltas from the applied outcome
    verification rejections (joint activities)
    the number of dialog turns

The agent's own reflection is deliberately *not* an input: "Idea or Reflection
cannot directly increase skill or method value" (design doc invariant #4), and
"reinforcement must have external evidence" (#2.3).

Reward components (each normalized, then combined and clipped to [-1, 1]):

    + skill gain          how much capability the practice actually produced
    + quality             absence of rejected output
    + transferability     gains that spilled over into other skills
    - time cost           session length (turns, or one slot when unknown)
    - money cost          money spent
    - vitality cost       energy consumed
    - social cost         negative social/esteem movement

Value and confidence stay separate: the first practice can already produce a
strong value, but confidence only grows with the number of practices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Learning is evidence-driven: the step size shrinks as evidence accumulates.
# (Deliberately not a function of the character's intelligence.)
ALPHA0 = 0.5
ALPHA_DECAY = 0.5

# Practices needed for confidence to reach 0.5.
CONFIDENCE_HALF_LIFE = 4.0

# Normalization scales, aligned with the delta limits in config.json so a
# "maximum" outcome from the environment model maps to 1.0.
SKILL_GAIN_FULL_SCALE = 6.0  # total skill points considered full credit
VITALITY_FULL_SCALE = 5.0
MONEY_FULL_SCALE = 200.0
FULFILLMENT_FULL_SCALE = 5.0
TURNS_FULL_SCALE = 12.0
UNKNOWN_SESSION_TIME_COST = 0.25  # a single occupied slot

# Outcome quality: how many rejections count as "no quality left".
REJECTIONS_FULL_SCALE = 3.0

# --- baseline architecture (KI-8, design: docs/methodology-reward-design.md) --
# The current reward has a positive floor: `quality` is 1.0 whenever the
# environment model rejected nothing, which alone (+0.25) exceeds every cost
# term in practice. Measured on the 1-year run: 135/135 observed outcomes were
# positive, so value only ever rose and `deprecated` was unreachable.
#
# The fix is to score *relative to what this character normally achieves in the
# same kind of situation* instead of against an absolute scale. The switch is
# deliberately inert: `use_baseline=False` reproduces today's numbers exactly,
# and phase 4 turns it on after the calibration numbers are reviewed.
BASELINE_PRIOR = 0.5          # neutral baseline before a context has samples
BASELINE_MIN_SAMPLES = 3      # below this, fall back to the wider context
BASELINE_WINDOW = 30          # how many past outcomes a context baseline uses
BASELINE_SHRINKAGE = 4.0      # prior weight when blending a thin context
# Width of the "as expected" band: |actual - baseline| / this maps to +-1.
BASELINE_FULL_SCALE = 0.5
# Phase 5 measured that 0.5 is wider than the whole dynamic range of the
# `skill_gain` proxy (observed sd 0.17 on run 09270135), so the centred quality
# term barely moves: with the design's `quality = 0.5 + 0.5*delta` mapping the
# reward kept a positive floor and never reached the design's own target
# (`positive_reward_share` 0.5-0.8, a sample at or below -0.1, `deprecated`
# reachable). The signed mode uses a band scaled to the proxy instead.
BASELINE_FULL_SCALE_SIGNED = 0.25

# Status thresholds (design doc 5.4 lifecycle).
VALIDATED_MIN_PRACTICES = 3
VALIDATED_MIN_SUCCESS_RATE = 0.6
VALIDATED_MIN_CONFIDENCE = 0.4
DEPRECATED_MAX_VALUE = -0.25
DEPRECATED_MIN_PRACTICES = 3

# `specialized` (phase 5, reward design §3.3): a method whose value in one
# situation stands this far above its pooled value is good *there*, not overall.
SPECIALIZED_CONTEXT_MARGIN = 0.2
SPECIALIZED_MIN_CONTEXT_SAMPLES = 3


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass
class RewardWeights:
    """Component weights. Exposed so a later phase can tune them per activity."""

    skill_gain: float = 0.45
    quality: float = 0.25
    transferability: float = 0.15
    time_cost: float = 0.10
    money_cost: float = 0.08
    vitality_cost: float = 0.12
    social_cost: float = 0.10
    # KI-8 knobs. Defaults keep the phase-1..3 behaviour bit-for-bit:
    # with `use_baseline=False` the quality term is the old
    # `1 - rejections/3` and no centring happens.
    use_baseline: bool = False
    quality_weight: float = 0.25
    goal_progress_weight: float = 0.0
    # Phase 5 probe (default off, design §3 stays the default): with
    # `use_baseline=True`, score *performance* against the baseline directly
    # instead of mapping it onto `quality = 0.5 + 0.5*delta`. See
    # `BASELINE_FULL_SCALE_SIGNED` for why: the design's mapping has a positive
    # floor and, combined with a positive-only `skill_gain` term, makes the value
    # monotonically rise so `deprecated` can never be reached.
    centred_performance: bool = False


@dataclass
class OutcomeSignals:
    """Objective facts about one finished activity, taken from its ledger record."""

    activity_id: str
    activity_type: str = "solo"
    skill_gains: Dict[str, float] = field(default_factory=dict)
    delta_vitality: int = 0
    delta_fulfillment: Dict[str, int] = field(default_factory=dict)
    delta_money: int = 0
    verification_rejections: int = 0
    turns: int = 0
    # Phase 5: how fully the activity satisfied the character's stated intent, as
    # reported by the world model on a 0-1 scale. `None` when the world model was
    # not asked for it (`world.cognition.goal_progress` off), in which case the
    # goal-progress term contributes nothing.
    goal_progress: Optional[float] = None

    def gain_for(self, skill_id: str) -> float:
        for name, delta in self.skill_gains.items():
            if name == skill_id:
                return float(delta)
        return 0.0

    def total_positive_gain(self) -> float:
        return sum(v for v in self.skill_gains.values() if v > 0)

    def spillover_gain(self, skill_id: str) -> float:
        """Gains in skills other than the method's own skill (transfer proxy)."""
        return sum(
            v
            for name, v in self.skill_gains.items()
            if name != skill_id and v > 0
        )


@dataclass
class RewardComponents:
    skill_gain: float
    quality: float
    transferability: float
    time_cost: float
    money_cost: float
    vitality_cost: float
    social_cost: float
    total: float
    baseline: float = 0.0
    baseline_samples: int = 0
    quality_delta: float = 0.0
    used_baseline: bool = False
    # True when the score came from the signed "better or worse than usual" term
    # rather than the floored `0.5 + 0.5*delta` quality (phase 5 probe).
    signed_performance: bool = False
    # Phase 5: the world model's goal-progress report, and that report relative to
    # the character's own usual goal progress. `None`/0 when not reported.
    goal_progress: Optional[float] = None
    goal_progress_delta: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "skill_gain": round(self.skill_gain, 4),
            "quality": round(self.quality, 4),
            "transferability": round(self.transferability, 4),
            "time_cost": round(self.time_cost, 4),
            "money_cost": round(self.money_cost, 4),
            "vitality_cost": round(self.vitality_cost, 4),
            "social_cost": round(self.social_cost, 4),
            "total": round(self.total, 4),
            "baseline": round(self.baseline, 4),
            "baseline_samples": int(self.baseline_samples),
            "quality_delta": round(self.quality_delta, 4),
            "used_baseline": bool(self.used_baseline),
            "signed_performance": bool(self.signed_performance),
            "goal_progress": (
                None if self.goal_progress is None else round(self.goal_progress, 4)
            ),
            "goal_progress_delta": round(self.goal_progress_delta, 4),
        }


def signals_from_activity_record(record: Dict[str, Any]) -> OutcomeSignals:
    """Build objective signals from an `activity.jsonl` record.

    Works for solo, joint and public records; missing fields simply stay at
    their neutral value, so an incomplete record produces a weaker (not a
    fabricated) signal.
    """
    outcome = record.get("outcome") or {}
    if not isinstance(outcome, dict):
        outcome = {}

    skill_gains: Dict[str, float] = {}
    for name, delta in (outcome.get("delta_skills") or {}).items():
        try:
            skill_gains[str(name)] = float(delta)
        except (TypeError, ValueError):
            continue

    def _int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    return OutcomeSignals(
        activity_id=str(record.get("activity_id") or ""),
        activity_type=str(record.get("type") or "solo"),
        skill_gains=skill_gains,
        delta_vitality=_int(outcome.get("delta_vitality")),
        delta_fulfillment={
            str(k): _int(v)
            for k, v in (outcome.get("delta_fulfillment") or {}).items()
        },
        delta_money=_int(outcome.get("delta_money")),
        verification_rejections=_int(record.get("verification_rejections")),
        turns=_int(record.get("turns")),
        goal_progress=_optional_float(outcome.get("goal_progress")),
    )


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class BaselineTracker:
    """Per-context expected performance (KI-8).

    Pure and offline: it is fed finished outcomes and answers "what does this
    character normally get in this kind of situation?".

    The fallback cascade is the design's §3.1 in full, and it matters more than
    it looks. Phase 5 measures it: context keys at offer time are *predicted*
    situations (`learning+solo+sufficient_time`) and at practice time they are
    observed ones, so in a six-week run almost no bucket reaches
    `BASELINE_MIN_SAMPLES`. With only an exact→"default" fallback — and nothing
    ever writing "default" — every outcome would be scored against the neutral
    prior instead of against the character's own history, which reports a
    character-wide shift rather than "better or worse than usual". So:

        exact context  ->  same activity type  ->  all of this character's
                       ->  neutral prior
    """

    def __init__(self) -> None:
        self._samples: Dict[str, List[float]] = {}
        self._by_activity: Dict[str, List[float]] = {}
        self._all: List[float] = []

    def observe(
        self,
        context_key: str,
        outcome_score: float,
        *,
        activity_type: str = "",
    ) -> None:
        score = float(outcome_score)
        bucket = self._samples.setdefault(str(context_key), [])
        bucket.append(score)
        if len(bucket) > BASELINE_WINDOW:
            del bucket[: len(bucket) - BASELINE_WINDOW]
        if activity_type:
            wider = self._by_activity.setdefault(str(activity_type), [])
            wider.append(score)
            if len(wider) > BASELINE_WINDOW:
                del wider[: len(wider) - BASELINE_WINDOW]
        self._all.append(score)
        if len(self._all) > BASELINE_WINDOW:
            del self._all[: len(self._all) - BASELINE_WINDOW]

    def samples(self, context_key: str, *, activity_type: str = "") -> int:
        """How much evidence actually backs the number `baseline_for` returns."""
        exact = len(self._samples.get(str(context_key), []))
        if exact >= BASELINE_MIN_SAMPLES:
            return exact
        wider = len(self._by_activity.get(str(activity_type), [])) if activity_type else 0
        if wider >= BASELINE_MIN_SAMPLES:
            return exact + wider
        return exact + len(self._all)

    def _bucket(self, context_key: str, *, activity_type: str = "") -> List[float]:
        exact = list(self._samples.get(str(context_key), []))
        if len(exact) >= BASELINE_MIN_SAMPLES:
            return exact
        if activity_type:
            wider = list(self._by_activity.get(str(activity_type), []))
            if len(wider) >= BASELINE_MIN_SAMPLES:
                return exact + wider
        return exact + list(self._all)

    def baseline_for(
        self,
        context_key: str,
        *,
        activity_type: str = "",
        fallback_key: str = "default",
    ) -> float:
        """The expected performance, widened until there is enough evidence.

        The widenings are folded widest-first, so each level only moves the
        estimate by its own evidence weight and the neutral prior fades as soon
        as the character has any history. That ordering is what keeps the prior
        from biasing a whole run: measured on run `09270135`, shrinking every
        level towards 0.5 (while the `skill_gain` proxy sits near 0.2) pushed the
        mean `quality_delta` to -0.28 and made the centred score look like a
        character-wide shift instead of a per-situation comparison.
        """
        levels: List[tuple[int, List[float]]] = []
        all_scores = list(self._all)
        if all_scores:
            levels.append((len(all_scores), all_scores))
        if activity_type:
            wider = list(self._by_activity.get(str(activity_type), []))
            if wider:
                levels.append((len(wider), wider))
        exact = list(self._samples.get(str(context_key), []))
        if exact:
            levels.append((len(exact), exact))
        if not levels:
            # The character has no history at all yet: neutral, not optimistic.
            return BASELINE_PRIOR

        estimate = BASELINE_PRIOR
        for count, bucket in levels:
            weight = count / (count + BASELINE_SHRINKAGE)
            estimate = (1.0 - weight) * estimate + weight * (sum(bucket) / len(bucket))
        return round(max(0.0, min(1.0, estimate)), 4)

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return {
            key: {"samples": len(values), "mean": round(sum(values) / len(values), 4)}
            for key, values in sorted(self._samples.items())
            if values
        }


def compute_reward(
    signals: OutcomeSignals,
    *,
    skill_id: str,
    weights: Optional[RewardWeights] = None,
    baseline: Optional[float] = None,
    baseline_samples: int = 0,
    goal_baseline: Optional[float] = None,
) -> RewardComponents:
    """Map objective signals to a reward in [-1, 1] for one method practice.

    `baseline` is optional and inert unless `weights.use_baseline` is set: with
    the default weights this function returns exactly the phase-1..3 numbers
    (KI-8 keeps the switch off until the design in
    `docs/methodology-reward-design.md` is approved).
    """
    w = weights or RewardWeights()

    skill_gain = _clip(signals.total_positive_gain() / SKILL_GAIN_FULL_SCALE)
    quality = _clip(1.0 - signals.verification_rejections / REJECTIONS_FULL_SCALE)
    quality_delta = 0.0
    used_baseline = False
    signed_performance = False
    if w.use_baseline:
        # Score against what this character normally achieves here: the same
        # activity is good or bad only relative to its own history.
        expected = BASELINE_PRIOR if baseline is None else float(baseline)
        observed = skill_gain
        if w.centred_performance:
            # "Better or worse than my usual" *is* the score, so a below-baseline
            # practice can push the total negative on its own. The `skill_gain`
            # weight folds into the signed term instead of sitting next to it as
            # a positive-only component.
            signed_performance = True
            quality_delta = max(
                -1.0, min(1.0, (observed - expected) / BASELINE_FULL_SCALE_SIGNED)
            )
            quality = _clip(0.5 + 0.5 * quality_delta)
        else:
            quality_delta = max(-1.0, min(1.0, (observed - expected) / BASELINE_FULL_SCALE))
            quality = _clip(0.5 + 0.5 * quality_delta)
        used_baseline = True
    transferability = _clip(signals.spillover_gain(skill_id) / SKILL_GAIN_FULL_SCALE)

    if signals.turns > 0:
        time_cost = _clip(signals.turns / TURNS_FULL_SCALE)
    else:
        time_cost = UNKNOWN_SESSION_TIME_COST

    money_cost = _clip(-min(0, signals.delta_money) / MONEY_FULL_SCALE)
    vitality_cost = _clip(-min(0, signals.delta_vitality) / VITALITY_FULL_SCALE)

    # Phase 5 goal progress: how much better or worse than usual the activity
    # satisfied the character's stated intent. Zero when the world model did not
    # report one, so the term is inert for every run that did not ask for it.
    goal_progress_delta = 0.0
    if signals.goal_progress is not None:
        expected_goal = BASELINE_PRIOR if goal_baseline is None else float(goal_baseline)
        goal_progress_delta = max(
            -1.0,
            min(1.0, (float(signals.goal_progress) - expected_goal) / BASELINE_FULL_SCALE),
        )

    social_negative = sum(
        -v
        for k, v in signals.delta_fulfillment.items()
        if k in ("social", "esteem") and v < 0
    )
    social_cost = _clip(social_negative / FULFILLMENT_FULL_SCALE)

    if signed_performance:
        # Both performance weights land on the same signed quantity: the reward
        # answers "how much better or worse than usual was this", and the
        # absolute skill gain no longer carries a positive floor.
        performance = (w.skill_gain + w.quality_weight) * quality_delta
        positive = (
            performance
            + w.transferability * transferability
            + w.goal_progress_weight * goal_progress_delta
        )
    else:
        quality_weight = w.quality_weight if used_baseline else w.quality
        positive = (
            w.skill_gain * skill_gain
            + quality_weight * quality
            + w.transferability * transferability
            + w.goal_progress_weight * max(0.0, goal_progress_delta)
        )
    negative = (
        w.time_cost * time_cost
        + w.money_cost * money_cost
        + w.vitality_cost * vitality_cost
        + w.social_cost * social_cost
    )
    total = max(-1.0, min(1.0, positive - negative))

    return RewardComponents(
        baseline=round(float(baseline if baseline is not None else 0.0), 4),
        baseline_samples=int(baseline_samples),
        quality_delta=round(quality_delta, 4),
        used_baseline=used_baseline,
        signed_performance=signed_performance,
        skill_gain=skill_gain,
        quality=quality,
        transferability=transferability,
        time_cost=time_cost,
        money_cost=money_cost,
        vitality_cost=vitality_cost,
        social_cost=social_cost,
        total=total,
        goal_progress=(
            None if signals.goal_progress is None else round(float(signals.goal_progress), 4)
        ),
        goal_progress_delta=round(goal_progress_delta, 4),
    )


@dataclass
class ValueUpdate:
    """Result of folding one practice outcome into a methodology."""

    value: float
    confidence: float
    practice_count: int
    success_count: int
    learning_rate: float
    reward: float
    success: bool
    status: str
    # Phase 5: the context this method turned out to be good *in*, when one
    # stands clearly above its pooled value (reward design §3.3).
    specialized_context: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        # `global_value` matches the methodology schema, because this dict is
        # written straight into METHOD_VALUE_UPDATED events.
        payload = {
            "global_value": round(self.value, 4),
            "confidence": round(self.confidence, 4),
            "practice_count": self.practice_count,
            "success_count": self.success_count,
            "learning_rate": round(self.learning_rate, 4),
            "reward": round(self.reward, 4),
            "success": self.success,
            "status": self.status,
        }
        if self.specialized_context is not None:
            payload["specialized_context"] = dict(self.specialized_context)
        return payload


def learning_rate(practice_count: int) -> float:
    """Step size for the Q update, shrinking as evidence accumulates."""
    return ALPHA0 / (1.0 + ALPHA_DECAY * max(0, practice_count))


def confidence_for(practice_count: int) -> float:
    """Evidence-backed confidence in [0, 1)."""
    if practice_count <= 0:
        return 0.0
    return round(practice_count / (practice_count + CONFIDENCE_HALF_LIFE), 4)


def status_for(
    *,
    practice_count: int,
    success_count: int,
    value: float,
    current_status: str,
) -> str:
    """Lifecycle transition driven purely by accumulated evidence.

    `proposed -> tested -> validated`, plus a deprecation path for methods that
    keep failing in practice. `learned` (direct teaching) and the archival
    states are left to later phases.
    """
    if current_status in ("deprecated", "archived"):
        return current_status
    if practice_count <= 0:
        return current_status if current_status in ("proposed", "learned") else "proposed"
    if practice_count >= DEPRECATED_MIN_PRACTICES and value <= DEPRECATED_MAX_VALUE:
        return "deprecated"

    success_rate = success_count / practice_count
    if (
        practice_count >= VALIDATED_MIN_PRACTICES
        and success_rate >= VALIDATED_MIN_SUCCESS_RATE
        and confidence_for(practice_count) >= VALIDATED_MIN_CONFIDENCE
    ):
        return "validated"
    return "tested"


def update_value(
    *,
    value: float,
    practice_count: int,
    success_count: int,
    reward: float,
    current_status: str = "proposed",
    context_values: Optional[Dict[str, Dict[str, float]]] = None,
    specialized_margin: float = SPECIALIZED_CONTEXT_MARGIN,
    specialized_min_samples: int = SPECIALIZED_MIN_CONTEXT_SAMPLES,
) -> ValueUpdate:
    """Fold one practice reward into a methodology's numbers.

    `Q_new = Q_old + alpha * (reward - Q_old)` with alpha from the evidence
    count, so the numbers converge instead of oscillating with every week.
    """
    alpha = learning_rate(practice_count)
    new_value = value + alpha * (reward - value)
    new_value = max(-1.0, min(1.0, new_value))

    success = reward > 0.0
    new_practice = practice_count + 1
    new_success = success_count + (1 if success else 0)

    status = status_for(
        practice_count=new_practice,
        success_count=new_success,
        value=new_value,
        current_status=current_status,
    )
    # `specialized` is not a promotion, it is a narrowing: this method is good
    # *here*, whatever it is like elsewhere. Checked here because this is the only
    # place with both the fresh value and the context breakdown.
    #
    # It deliberately does NOT require `validated` first. The reward design's §3.3
    # defines it purely as "one context stands >= margin above the pooled value",
    # and requiring an overall validation would be contradictory — a method that
    # is only good in one situation is, by construction, not good overall. Run
    # 09272220 (2 years) measured the consequence: with the `validated` gate, zero
    # methods specialised in 20 weeks, while two had a context that qualified.
    specialized: Optional[Dict[str, Any]] = None
    if status in ("tested", "validated") and context_values:
        from src.agents.cognition.method_lifecycle import strongest_context

        found = strongest_context(
            context_values,
            pooled=new_value,
            margin=specialized_margin,
            min_samples=specialized_min_samples,
        )
        if found is not None:
            specialized = found.to_dict()
            status = "specialized"

    return ValueUpdate(
        value=new_value,
        confidence=confidence_for(new_practice),
        practice_count=new_practice,
        success_count=new_success,
        learning_rate=alpha,
        reward=reward,
        success=success,
        status=status,
        specialized_context=specialized,
    )


def update_context_value(
    context_values: Dict[str, Dict[str, float]],
    *,
    context_key: str,
    reward: float,
) -> Dict[str, Dict[str, float]]:
    """Per-context Q update (design doc 5.4 `context_values`).

    A method can be good in one situation and bad in another; the overall value
    is the pooled estimate, these are the conditional ones.
    """
    entry = context_values.get(context_key) or {"value": 0.0, "count": 0}
    count = int(entry.get("count", 0))
    old = float(entry.get("value", 0.0))
    alpha = learning_rate(count)
    merged = dict(context_values)
    merged[context_key] = {
        "value": max(-1.0, min(1.0, old + alpha * (reward - old))),
        "count": count + 1,
    }
    return merged


def value_calibration_error(
    methods: List[Any],
) -> Optional[float]:
    """Mean |value - observed success rate| over evidence-backed methods.

    Part of the evaluation metrics (design doc 12): a value that does not track
    the actual success rate is worse than no value at all.
    """
    errors: List[float] = []
    for m in methods:
        if not getattr(m, "practice_count", 0):
            continue
        observed = m.success_count / m.practice_count
        errors.append(abs(float(m.global_value) - observed))
    if not errors:
        return None
    return sum(errors) / len(errors)


# --- shadow baseline instrumentation (KI-8, design §6 step 2) ----------------
#
# The switch `RewardWeights.use_baseline` is not thrown until the baseline
# distribution has been reviewed on a real run (docs/methodology-reward-design.md
# §6.1). Until then every practice outcome carries **both** numbers: the one the
# live value uses, and the one the centred formula *would* have produced. That
# way the review is a reading of the run, not a second run.


@dataclass
class ShadowBaseline:
    """One practice outcome, scored the baseline way (never used for value)."""

    reward: float
    baseline: float
    baseline_samples: int
    quality_delta: float
    components: Dict[str, float]
    goal_baseline: Optional[float] = None
    goal_progress: Optional[float] = None
    # Which formula produced `reward`. Written into the event so a later reader
    # does not have to guess from the config of the day.
    caliber: str = ""

    def to_payload(self) -> Dict[str, Any]:
        payload = {
            "shadow_reward": round(self.reward, 4),
            "shadow_baseline": round(self.baseline, 4),
            "shadow_baseline_samples": int(self.baseline_samples),
            "shadow_quality_delta": round(self.quality_delta, 4),
            "shadow_components": dict(self.components),
            "goal_progress": self.goal_progress,
            "goal_baseline": self.goal_baseline,
        }
        if self.caliber:
            payload["shadow_caliber"] = self.caliber
        return payload


def shadow_weights(
    *, centred_performance: bool = False, goal_progress_weight: float = 0.0
) -> RewardWeights:
    """The weights the centred formula would run with (design §3).

    `centred_performance=True` selects the signed variant the phase-5 shadow run
    argued for (see `BASELINE_FULL_SCALE_SIGNED`); it stays opt-in because the
    design document specifies the floored quality mapping. `goal_progress_weight`
    is the design's 0.30 goal term and stays 0 until the run's own distribution
    argues for it.
    """
    return RewardWeights(
        use_baseline=True,
        centred_performance=centred_performance,
        goal_progress_weight=max(0.0, float(goal_progress_weight)),
    )


# --- calibers -----------------------------------------------------------------
#
# Phase 5 made the caliber a *switch* rather than a hard-coded formula, because
# the shadow distribution run showed the design's floored quality mapping cannot
# produce the distribution the design asks for (KI-19). Every run now evaluates
# one caliber live and records the other one alongside it, so the comparison
# never needs a second run.

REWARD_CALIBERS = ("off", "quality", "signed")

# What a run uses unless its config says otherwise. Phase 5 measured that the
# design's floored mapping cannot produce the distribution the design asks for
# (KI-19), so the signed caliber is the default; `quality` and `off` remain one
# config value away.
DEFAULT_REWARD_CALIBER = "signed"


def weights_for_caliber(
    caliber: str, *, goal_progress_weight: float = 0.0
) -> RewardWeights:
    """Turn a caliber name into weights.

    `off`     phases 1-3 exactly: absolute scale, positive by construction.
    `quality` the design document's centred mapping (`0.5 + 0.5*delta`).
    `signed`  centred performance as the score itself, so below-baseline
              practice can push the total negative and `deprecated` is reachable.
    """
    name = str(caliber or "off").strip().lower()
    if name not in REWARD_CALIBERS:
        name = "off"
    if name == "off":
        return RewardWeights()
    return shadow_weights(
        centred_performance=(name == "signed"),
        goal_progress_weight=goal_progress_weight,
    )


def alternative_caliber(live: str) -> str:
    """The caliber worth recording *next to* the live one.

    With the live score already centred, the interesting comparison is the design
    document's floored mapping; with the live score on the old absolute scale,
    the interesting comparison is the centred one the project is moving to.
    """
    name = str(live or "off").strip().lower()
    if name == "signed":
        return "quality"
    return "signed"


def baseline_tracker_from_events(events: List[Dict[str, Any]]) -> BaselineTracker:
    """Rebuild the per-context baseline from the capability event stream.

    No new event stream and no private state: `context_key` and `activity_type`
    are already in every `METHOD_OUTCOME_OBSERVED`, and the observed performance
    is the same `skill_gain` proxy the live formula uses (design §4).
    """
    tracker = BaselineTracker()
    for event in events:
        if str(event.get("type") or "") != "METHOD_OUTCOME_OBSERVED":
            continue
        context_key = str(event.get("context_key") or "default")
        activity_type = str(event.get("activity_type") or "")
        components = event.get("components") or {}
        observed = components.get("skill_gain")
        if observed is None:
            # Older events carry only the total; the reward itself is a usable,
            # if coarser, stand-in for "what this character usually gets here".
            observed = event.get("reward")
        try:
            tracker.observe(context_key, float(observed), activity_type=activity_type)
        except (TypeError, ValueError):
            continue
    return tracker


def observed_proxy(signals: OutcomeSignals) -> float:
    """The "performance" number the baseline is built from (design §3.1)."""
    return _clip(signals.total_positive_gain() / SKILL_GAIN_FULL_SCALE)


def goal_baseline_tracker_from_events(events: List[Dict[str, Any]]) -> BaselineTracker:
    """The character's own usual goal progress, rebuilt from the same stream.

    A separate series because goal progress is a different quantity from skill
    gain: mixing them would make "as expected" mean the average of two unrelated
    scales.
    """
    tracker = BaselineTracker()
    for event in events:
        if str(event.get("type") or "") != "METHOD_OUTCOME_OBSERVED":
            continue
        reported = event.get("goal_progress")
        if reported is None:
            components = event.get("components") or {}
            reported = components.get("goal_progress")
        if reported is None:
            continue
        try:
            tracker.observe(
                str(event.get("context_key") or "default"),
                float(reported),
                activity_type=str(event.get("activity_type") or ""),
            )
        except (TypeError, ValueError):
            continue
    return tracker


def shadow_baseline_for(
    signals: OutcomeSignals,
    *,
    skill_id: str,
    context_key: str,
    tracker: Optional[BaselineTracker] = None,
    weights: Optional[RewardWeights] = None,
    activity_type: str = "",
    goal_tracker: Optional[BaselineTracker] = None,
    caliber: str = "",
) -> ShadowBaseline:
    """Score one practice the centred way, without touching any value."""
    tracker = tracker or BaselineTracker()
    activity = activity_type or signals.activity_type
    baseline = tracker.baseline_for(context_key, activity_type=activity)
    goal_baseline = None
    if goal_tracker is not None and signals.goal_progress is not None:
        goal_baseline = goal_tracker.baseline_for(context_key, activity_type=activity)
    components = compute_reward(
        signals,
        skill_id=skill_id,
        weights=weights or shadow_weights(),
        baseline=baseline,
        baseline_samples=tracker.samples(context_key, activity_type=activity),
        goal_baseline=goal_baseline,
    )
    return ShadowBaseline(
        reward=components.total,
        baseline=baseline,
        baseline_samples=int(components.baseline_samples),
        quality_delta=components.quality_delta,
        components=components.to_dict(),
        goal_baseline=goal_baseline,
        goal_progress=components.goal_progress,
        caliber=caliber,
    )


def live_reward_for(
    signals: OutcomeSignals,
    *,
    skill_id: str,
    context_key: str,
    tracker: Optional[BaselineTracker] = None,
    weights: Optional[RewardWeights] = None,
    activity_type: str = "",
    goal_tracker: Optional[BaselineTracker] = None,
) -> RewardComponents:
    """The score that actually moves a value, under whatever caliber is live.

    With a centred caliber this needs the character's own baseline, so the live
    path and the recorded path share the same tracker cascade — otherwise "live"
    would silently score against the neutral prior instead of its own history.
    """
    w = weights or RewardWeights()
    if not w.use_baseline:
        return compute_reward(signals, skill_id=skill_id, weights=w)

    tracker = tracker or BaselineTracker()
    activity = activity_type or signals.activity_type
    baseline = tracker.baseline_for(context_key, activity_type=activity)
    goal_baseline = None
    if goal_tracker is not None and signals.goal_progress is not None:
        goal_baseline = goal_tracker.baseline_for(context_key, activity_type=activity)
    return compute_reward(
        signals,
        skill_id=skill_id,
        weights=w,
        baseline=baseline,
        baseline_samples=tracker.samples(context_key, activity_type=activity),
        goal_baseline=goal_baseline,
    )
