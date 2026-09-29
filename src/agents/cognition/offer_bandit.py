"""Phase 5: the method menu as a contextual bandit (design doc 阶段 5).

Phase 4 offered a fixed shape of menu: "the best-scoring method, one exploration
slot, then fillers". Measured on run `09261617` that shape had two problems the
design document had already predicted:

1. **The exploit slot never rotated.** `自然亲和-c449f9c899` was offered in
   W06, W07, W09 and W10 in a row, because nothing in the score knew that the
   character had already been shown it and ignored it.
2. **The offering decision had no negative half.** 26 of 27 person-weeks adopted
   *something*, so "offered and not taken up" — which is the only feedback the
   offerer gets about its own choice — was not observable at all.

Phase 5 turns *what to offer* into a contextual bandit over the character's own
situation:

    score(m | signature) = exploitation(m | signature) + exploration_bonus(m)

    exploitation        a context-matched value estimate (shrunk towards the
                        method's pooled value), evidence confidence, lifecycle
                        bonus, provenance bonus (the character's own lessons and
                        ideas), minus what its own responses said: being offered
                        and ignored is negative evidence, and repeated ignores
                        decay the bonus that keeps a method in the menu
    exploration_bonus   UCB1-style `c * sqrt(ln(N+1) / (offers(m)+1))`, with `c`
                        scaled by the character's creativity/curiosity

Three properties are deliberate:

* **It is still a menu.** The bandit decides what to *show*, never what to do;
  adoption stays the character's own declaration (invariant #11), and a decline
  moves no value either.
* **Deterministic.** No randomness at all: the same situation and the same event
  history produce the same ranking, so replay and materialized views stay exact.
* **Rebuildable from events.** Offers, adoptions and declines all come from the
  capability event stream, so the bandit needs no private state to survive a
  resume.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.agents.cognition.models import CONTEXT_TAGS, ContextSignature, Methodology
from src.agents.cognition.methodology_policy import context_signature_from_activity

# --- scoring weights ---------------------------------------------------------

# Context estimates are thin in a short run, so they are shrunk towards the
# method's pooled value; this is the prior weight in that blend.
CONTEXT_SHRINKAGE = 3.0
W_CONTEXT = 1.1
W_VALUE = 1.0
W_EVIDENCE = 0.3

STATUS_BONUS = {
    "proposed": 0.0,
    "learned": 0.05,
    "tested": 0.1,
    "validated": 0.2,
    "specialized": 0.15,
}

# The character's own material (a method grown from a lesson it wrote, or from
# its own idea) stays ahead: practice is the only thing that can validate it,
# and it is the loop phase 4.5 exists for. Same magnitudes as phase 4.5.
PROVENANCE_BONUS = {
    "own_lesson_untried": 0.3,
    "own_lesson": 0.15,
    "own_idea": 0.1,
    "practice": 0.0,
}

# Being offered and ignored is the offerer's own negative feedback, but it is
# weak evidence: the character may simply have had no occasion to use it.
DECLINE_PENALTY = 0.45
# An offer that is never taken up decays towards the bottom of the menu.
STALE_OFFER_PENALTY = 0.06
STALE_OFFER_CAP = 4

# The other half of "when to keep using, when to switch": a method the character
# has used several weeks running keeps its value (nothing here touches it) but
# stops monopolising the menu's lead slot. Measured on run 09271745: each
# character adopted the same method in all five offered weeks, its score climbed
# with every practice (0.31 -> 1.29) and the menu never rotated, while the
# exploration slot it never took up earned no evidence either.
REPEAT_PENALTY = 0.15
REPEAT_CAP = 3

# Base exploration coefficient; persona scales it inside the span below.
UCB_EXPLORATION_C = 0.22
PERSONA_EXPLORATION_MIN = 0.5
PERSONA_EXPLORATION_MAX = 1.7

# A situation with less than a week of income in the bank is a "low money" one.
DEFAULT_LOW_MONEY_THRESHOLD = 300.0
LOW_VITALITY = 30.0
HIGH_VITALITY = 70.0


# --- what the character's own responses said ---------------------------------


@dataclass
class MethodStat:
    """Offer/adoption history of one method, folded from capability events."""

    method_id: str
    offers: int = 0
    adoptions: int = 0
    declines: int = 0
    practices: int = 0
    weeks_offered: int = 0
    last_offered_week: str = ""
    last_adopted_week: str = ""
    # Offers in weeks strictly after the last adoption — the streak that tells
    # the bandit "showing this again is not working".
    ignored_streak: int = 0
    # Weeks the character actually took this method up, oldest first. Used for
    # the *other* half of "when to keep using, when to switch": a method used
    # several weeks running does not need the menu's scarce lead slot any more,
    # however good its value looks.
    adopted_weeks: List[str] = field(default_factory=list)

    @property
    def adopted_at_least_once(self) -> bool:
        return self.adoptions > 0


def method_stats(events: Iterable[Dict[str, Any]]) -> Dict[str, MethodStat]:
    """Fold the capability event stream into per-method offer statistics.

    Rebuilt from events on every use: a resume must not need a private counter
    to remember what the character was already shown. Two passes, because the
    ignored streak is defined *relative to the last adoption* and that is only
    known once the whole stream has been read.
    """
    stats: Dict[str, MethodStat] = {}
    offer_weeks: Dict[str, set[str]] = {}
    weeks_per_method: Dict[str, List[str]] = {}

    def _stat(method_id: str) -> MethodStat:
        if method_id not in stats:
            stats[method_id] = MethodStat(method_id=method_id)
        return stats[method_id]

    for event in events:
        method_id = str(event.get("method_id") or "")
        if not method_id:
            continue
        event_type = str(event.get("type") or "")
        stat = _stat(method_id)
        week = str(event.get("week") or "")
        if event_type == "METHOD_HINTED":
            stat.offers += 1
            if week:
                offer_weeks.setdefault(method_id, set()).add(week)
            weeks_per_method.setdefault(method_id, []).append(week)
            if week > stat.last_offered_week:
                stat.last_offered_week = week
        elif event_type == "METHOD_DECLINED":
            stat.declines += 1
        elif event_type == "METHOD_SELECTED" and event.get("adoption"):
            stat.adoptions += 1
            if week and week not in stat.adopted_weeks:
                stat.adopted_weeks.append(week)
            if week > stat.last_adopted_week:
                stat.last_adopted_week = week
        elif event_type == "METHOD_OUTCOME_OBSERVED":
            if str(event.get("attribution") or "") == "real_adoption":
                stat.practices += 1

    for method_id, stat in stats.items():
        stat.adopted_weeks.sort()
        stat.weeks_offered = len(offer_weeks.get(method_id, ()))
        cutoff = stat.last_adopted_week
        weeks = [w for w in weeks_per_method.get(method_id, []) if w]
        if cutoff:
            stat.ignored_streak = len([w for w in weeks if w > cutoff])
        else:
            stat.ignored_streak = stat.offers

    return stats


def ignored_streak(stat: Optional[MethodStat]) -> int:
    """Weeks the method was offered *after* the last adoption (see above)."""
    return 0 if stat is None else int(stat.ignored_streak)


def consecutive_weeks_used(stat: Optional[MethodStat]) -> int:
    """How many weeks in a row, ending at the last adoption, the character used it.

    Measured on run `09271745`: every character adopted the *same* method in all
    five weeks it was offered, its value and confidence climbed with each
    practice, and so its exploit score climbed too (0.31 -> 0.84 -> 1.24 ->
    1.29). The menu could not rotate because nothing in the score knew the
    character had stopped needing to be told about it, and the exploration slot
    it never took up never earned any evidence either. This counter is that
    missing signal.
    """
    if stat is None or not stat.adopted_weeks:
        return 0
    weeks = sorted(stat.adopted_weeks)
    run = 1
    for earlier, later in zip(weeks, weeks[1:]):
        run = run + 1 if _weeks_apart(earlier, later) == 1 else 1
    return run


def _weeks_apart(earlier: str, later: str) -> int:
    """Week distance for `Y2020-W03` style keys; a large number when unparsable."""
    def _parts(value: str):
        text = str(value or "")
        if "-W" not in text:
            return None
        year, week = text.split("-W", 1)[:2]
        try:
            return int(year.lstrip("Y")), int(week[:2])
        except (TypeError, ValueError):
            return None

    left, right = _parts(earlier), _parts(later)
    if left is None or right is None:
        return 10 ** 6
    return (right[0] - left[0]) * 10 + (right[1] - left[1])


# --- context-matched value ---------------------------------------------------


@dataclass
class ContextEstimate:
    """What the method's own context values say about this situation."""

    value: float
    weight: float
    samples: int
    matched_keys: List[str] = field(default_factory=list)
    exact: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": round(self.value, 4),
            "weight": round(self.weight, 3),
            "samples": int(self.samples),
            "keys": list(self.matched_keys),
            "exact": bool(self.exact),
        }


def _tags_of_key(key: str) -> List[str]:
    return [t for t in str(key).split("+") if t and t != "default"]


def _canonical_key(key: str) -> str:
    """Tag order must not decide whether two contexts are the same situation."""
    return "+".join(sorted(_tags_of_key(key)))


def context_estimate(
    method: Methodology,
    signature: ContextSignature,
    *,
    shrinkage: float = CONTEXT_SHRINKAGE,
) -> Optional[ContextEstimate]:
    """Shrunk estimate of the method's value in `signature`'s situations.

    Contexts are matched by tag overlap rather than by exact key: the offer is
    made before the week's activities exist, so the week's signature is a
    *prediction* of the contexts the method will actually be practised in. Thin
    evidence is pulled back towards the method's pooled value, which is what
    keeps a 6-week run from making decisions on single samples.
    """
    want = set(signature.context_tags)
    if not method.context_values:
        return None

    weight_sum = 0.0
    weighted_value = 0.0
    samples = 0
    keys: List[str] = []
    exact = False

    for key, entry in sorted(method.context_values.items()):
        key_tags = set(_tags_of_key(key))
        entry_value = float((entry or {}).get("value", 0.0))
        entry_count = int((entry or {}).get("count", 0))
        if entry_count <= 0:
            continue

        if not want or not key_tags:
            overlap = 0.0
        else:
            shared = key_tags & want
            if not shared:
                continue
            overlap = len(shared) / max(1, len(key_tags | want))
        if overlap <= 0:
            continue

        if key and _canonical_key(key) == _canonical_key(signature.context_key()):
            overlap = max(overlap, 1.0)
            exact = True

        weight = overlap * entry_count
        weight_sum += weight
        weighted_value += weight * entry_value
        samples += entry_count
        keys.append(key)

    if weight_sum <= 0:
        return None

    observed = weighted_value / weight_sum
    shrink = weight_sum / (weight_sum + max(0.0, shrinkage))
    value = shrink * observed + (1.0 - shrink) * float(method.global_value)
    return ContextEstimate(
        value=max(-1.0, min(1.0, value)),
        weight=weight_sum,
        samples=samples,
        matched_keys=keys,
        exact=exact,
    )


# --- provenance --------------------------------------------------------------


def origin_of(method: Any) -> str:
    """Where an offered method came from, from the character's point of view.

    `own_lesson` — the method was grown from a lesson the character wrote in its
    own notes; `own_idea` — from an idea of its own; `practice` — extracted from
    a finished week by the cognition layer.
    """
    if str(getattr(method, "source_motif", "") or "") == "lesson_application":
        return "own_lesson"
    if str(getattr(method, "source_type", "") or "") == "idea_conversion":
        return "own_idea"
    return "practice"


def practice_priority(method: Any, *, practised: int) -> float:
    """Extra ordering weight for the menu (never written into `score`).

    Kept as a separate function because the lesson bridge (phase 4.5) and the
    bandit (phase 5) must give the character's own material the same advantage:
    an untried lesson method outranks a tried one.
    """
    if origin_of(method) != "own_lesson":
        return 0.0
    return PROVENANCE_BONUS["own_lesson_untried" if practised == 0 else "own_lesson"]


def provenance_bonus(method: Any) -> float:
    origin = origin_of(method)
    if origin == "own_lesson":
        return practice_priority(method, practised=int(getattr(method, "practice_count", 0) or 0))
    if origin == "own_idea":
        return PROVENANCE_BONUS["own_idea"]
    return PROVENANCE_BONUS["practice"]


# --- exploration -------------------------------------------------------------


def exploration_coefficient(traits: Optional[Dict[str, Any]] = None) -> float:
    """Persona-controlled exploration strength (design doc 阶段 5).

    creativity/curiosity widen the search for unproven methods; the coefficient
    is bounded so a "creative" character still mostly reuses what works.
    """
    traits = traits or {}
    values: List[float] = []
    for key in ("creativity", "curiosity"):
        try:
            raw = traits.get(key)
            if raw is not None:
                values.append(float(raw))
        except (TypeError, ValueError):
            continue
    if not values:
        return UCB_EXPLORATION_C
    mean = sum(values) / len(values)  # 0-100 persona scale
    scaled = 0.5 + (mean - 50.0) / 100.0  # 0.0 .. 1.0
    factor = PERSONA_EXPLORATION_MIN + scaled * (
        PERSONA_EXPLORATION_MAX - PERSONA_EXPLORATION_MIN
    )
    return round(UCB_EXPLORATION_C * factor, 4)


# --- the offering decision ---------------------------------------------------


@dataclass
class OfferScore:
    """One candidate's menu score, with every part kept for the audit."""

    method: Methodology
    score: float
    role: str
    context: Optional[ContextEstimate]
    stat: Optional[MethodStat]
    parts: Dict[str, float] = field(default_factory=dict)

    @property
    def ignored_streak(self) -> int:
        return ignored_streak(self.stat)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_id": self.method.method_id,
            "title": self.method.title,
            "skill_id": self.method.skill_id,
            "status": self.method.status,
            "role": self.role,
            "score": round(self.score, 4),
            "context_key": (self.context.matched_keys[0] if self.context and self.context.matched_keys else ""),
            "context_estimate": (self.context.to_dict() if self.context else None),
            "offers": int(self.stat.offers) if self.stat else 0,
            "adoptions": int(self.stat.adoptions) if self.stat else 0,
            "declines": int(self.stat.declines) if self.stat else 0,
            "ignored_streak": self.ignored_streak,
            "consecutive_weeks_used": consecutive_weeks_used(self.stat),
            "parts": {k: round(v, 4) for k, v in self.parts.items()},
        }


def score_offer(
    method: Methodology,
    signature: ContextSignature,
    *,
    stat: Optional[MethodStat] = None,
    total_offers: int = 0,
    exploration_c: float = UCB_EXPLORATION_C,
    contraindicated: bool = False,
) -> OfferScore:
    """Score one candidate for *being shown* in this situation."""
    context = context_estimate(method, signature)
    parts: Dict[str, float] = {}

    parts["context"] = W_CONTEXT * (context.value if context else 0.0)
    parts["value"] = W_VALUE * float(method.global_value)
    parts["evidence"] = W_EVIDENCE * float(method.confidence)
    parts["status"] = STATUS_BONUS.get(method.status, 0.0)
    parts["provenance"] = provenance_bonus(method)

    offers = int(stat.offers) if stat else 0
    declines = int(stat.declines) if stat else 0
    if offers > 0:
        parts["decline"] = -DECLINE_PENALTY * min(1.0, declines / offers)

    streak = ignored_streak(stat)
    parts["stale"] = -STALE_OFFER_PENALTY * min(STALE_OFFER_CAP, streak)

    # Already in use: the first week is free, every week after it costs the lead
    # slot (never the method's value).
    used_in_a_row = consecutive_weeks_used(stat)
    if used_in_a_row > 1:
        parts["repeat"] = -REPEAT_PENALTY * min(REPEAT_CAP, used_in_a_row - 1)

    parts["explore"] = exploration_c * math.sqrt(
        math.log(max(2, total_offers + 2)) / (offers + 1)
    )

    if contraindicated:
        parts["contraindication"] = -0.4

    score = sum(parts.values())
    return OfferScore(
        method=method,
        score=score,
        role="filler",
        context=context,
        stat=stat,
        parts=parts,
    )


def is_unproven(method: Methodology) -> bool:
    """No practice evidence yet: the only thing exploration can offer."""
    return int(getattr(method, "practice_count", 0) or 0) <= 0


def rank_offers(
    methods: Sequence[Methodology],
    signature: ContextSignature,
    *,
    stats: Optional[Dict[str, MethodStat]] = None,
    traits: Optional[Dict[str, Any]] = None,
    top_k: int = 3,
    exploration_c: Optional[float] = None,
    contraindication_tags: Sequence[str] = (),
) -> List[OfferScore]:
    """Rank the menu: exploitation, plus a UCB bonus for the untried.

    Deterministic: ties break on `method_id`, so replaying the same events in
    the same situation yields the same menu (and the same `METHOD_HINTED`).
    """
    stats = stats or {}
    coefficient = (
        exploration_coefficient(traits) if exploration_c is None else float(exploration_c)
    )
    total_offers = sum(int(s.offers) for s in stats.values())
    signature_tags = set(signature.context_tags)

    scored: List[OfferScore] = []
    for method in methods:
        contraindicated = bool(signature_tags & set(method.contraindications or []))
        scored.append(
            score_offer(
                method,
                signature,
                stat=stats.get(method.method_id),
                total_offers=total_offers,
                exploration_c=coefficient,
                contraindicated=contraindicated,
            )
        )

    scored.sort(key=lambda s: (-s.score, s.method.method_id))

    # Phase 4.5's two guarantees survive the bandit, in their narrowest useful
    # form: material the character produced **itself** and has **never been
    # shown** always reaches the menu — a lesson-grown method takes the leading
    # slot (its own conclusion about its own behaviour), an idea of its own is
    # guaranteed a place in the menu even when the bandit would not have picked
    # it. Showing its own new material is not something the bandit may trade
    # away; whether it *stays* in the menu is exactly what the bandit now
    # decides (offer history, declines and per-situation value all enter the
    # score).
    def _never_shown(item: OfferScore, origin: str) -> bool:
        return (
            origin_of(item.method) == origin
            and int(item.stat.offers if item.stat else 0) == 0
        )

    menu_size = max(1, top_k)
    own_lesson = next((i for i in scored if _never_shown(i, "own_lesson")), None)
    if own_lesson is not None:
        scored.remove(own_lesson)
        scored.insert(0, own_lesson)

    own_idea = next((i for i in scored if _never_shown(i, "own_idea")), None)
    if own_idea is not None and own_idea not in scored[:menu_size]:
        scored.insert(menu_size - 1, scored.pop(scored.index(own_idea)))

    # A method earns the `explore` label only when the exploration bonus is what
    # put it in the menu: it is unproven *and* it would fall outside the menu on
    # exploitation alone. "Unproven" on its own is not enough — in a young run
    # almost every method is unproven, and labelling all of them `explore` makes
    # the reading meaningless (run 09271745 recorded 21 of 45 offers that way).
    merit_rank = {
        item.method.method_id: rank
        for rank, item in enumerate(
            sorted(
                scored,
                key=lambda s: (
                    -(s.score - s.parts.get("explore", 0.0)),
                    s.method.method_id,
                ),
            )
        )
    }

    # The bandit decides the *order*; the best-scoring method keeps the leading
    # slot so "keep using what works" stays visible.
    if scored:
        scored[0].role = "exploit"
        for item in scored[1:menu_size]:
            outside_on_merit = merit_rank.get(item.method.method_id, 0) >= menu_size
            if is_unproven(item.method) and outside_on_merit:
                item.role = "explore"

    return scored[:menu_size]


# --- the situation the menu is offered in ------------------------------------


def tags_from_records(
    records: Sequence[Dict[str, Any]], *, limit: int = 2
) -> List[str]:
    """The situation tags of a week, predicted from what the character did last.

    Objective only: the tags come from the ledger shape of past activities
    (`solo`/`joint`/`public`, `learning`, `work`, ...), never from free text, so
    the controlled vocabulary cannot drift.
    """
    counts: Dict[str, int] = {}
    for record in records:
        signature = context_signature_from_activity(record, skill_id="")
        for tag in signature.context_tags:
            counts[tag] = counts.get(tag, 0) + 1
    if not counts:
        return []
    ordered = sorted(
        counts.items(),
        key=lambda kv: (
            -kv[1],
            CONTEXT_TAGS.index(kv[0]) if kv[0] in CONTEXT_TAGS else len(CONTEXT_TAGS),
        ),
    )
    return [tag for tag, _ in ordered[:limit]]


def week_signature(
    *,
    skills: Sequence[str] = (),
    state: Optional[Dict[str, Any]] = None,
    goal_memories: Sequence[Dict[str, Any]] = (),
    recent_records: Sequence[Dict[str, Any]] = (),
    low_money_threshold: float = DEFAULT_LOW_MONEY_THRESHOLD,
    max_goals: int = 2,
    max_skills: int = 6,
) -> ContextSignature:
    """Build the week's situation signature (phase 4 review §3 #5).

    Phase 4 scored every offer against "the character's skills and nothing
    else", so `ContextSignature` could not tell a rich, rested week from a broke,
    exhausted one and the menu never moved with the situation. The signature
    below is still fully objective — character state plus last week's ledger
    shape plus its own stated goals — and it stays inside the controlled tag
    vocabulary.
    """
    tags: List[str] = []
    constraints: Dict[str, float] = {}

    state = state or {}
    skills_state = state.get("skills") or {}
    ordered_skills = [str(s) for s in skills if s] or sorted(
        str(name) for name in skills_state
    )

    try:
        vitality = float(state.get("vitality") or 0.0)
    except (TypeError, ValueError):
        vitality = 0.0
    if vitality < LOW_VITALITY:
        tags.append("time_pressure")
        constraints["time_pressure"] = round(min(1.0, (LOW_VITALITY - vitality) / LOW_VITALITY), 3)
    elif vitality >= HIGH_VITALITY:
        tags.append("sufficient_time")
        constraints["time_pressure"] = 0.0

    try:
        deposit = float(((state.get("assets") or {}).get("deposit")) or 0.0)
    except (TypeError, ValueError):
        deposit = 0.0
    if low_money_threshold > 0 and deposit < low_money_threshold:
        tags.append("low_money")
        constraints["money_limit"] = round(
            min(1.0, (low_money_threshold - deposit) / low_money_threshold), 3
        )

    tags.extend(tags_from_records(recent_records))

    goals: List[str] = []
    for memory in goal_memories:
        content = str(memory.get("content") or "").strip()
        if content and content not in goals:
            goals.append(content)
        if len(goals) >= max_goals:
            break

    social_records = [
        r
        for r in recent_records
        if "high_social_risk" in context_signature_from_activity(r, skill_id="").context_tags
    ]
    if recent_records:
        constraints["social_risk"] = round(len(social_records) / len(recent_records), 3)

    return ContextSignature(
        skill_ids=ordered_skills[:max_skills],
        context_tags=tags,
        goal=" / ".join(goals)[:160],
        constraints=constraints,
    )
