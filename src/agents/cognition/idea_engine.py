"""Idea engine: relation motifs -> gated candidates (design doc 8.6 / 8.7 / 8.8).

Pipeline, all of it shadow-only:

    memories + relation edges
      -> rule motifs (no LLM): the six motifs of design doc 8.6
      -> phrase + propose a test for the top few (one LLM call per week)
      -> five gating checks (grounding, novelty, feasibility, testability, safety)
      -> IdeaPotential score with named penalties
      -> at most N ideas per week reach the store

Division of labour, following the design document: the language model only
*phrases* a candidate and states what it would require; every accept/reject
decision is made here from structured data (memories, world state, existing
ideas and methods). Personality shapes the candidate range and the filtering,
never the underlying scores.

An idea that survives may become a *candidate methodology* (phase 1 store, value
0), which is the only path by which it can eventually influence capability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from src.agents.cognition.idea_models import (
    CANDIDATE_CONFIDENCE_CEILING,
    MAX_IDEAS_PER_WEEK,
    MIN_IDEA_POTENTIAL,
    Idea,
    IdeaCandidate,
    IdeaPotential,
    idea_id_for,
)
from src.agents.cognition.memory_models import (
    MemoryItem,
    _affinity,
    token_set,
    jaccard,
)
from src.utils import get_logger

IDEA_LOGGER = get_logger("cognition", quiet=True)

# How many candidates get phrased by the model in one week.
MAX_CANDIDATES_FOR_PHRASING = 4
# No single motif may take more than this share of the phrasing budget, so one
# noisy rule cannot starve the others (measured failure: 4/4 slots taken by
# `goal_obstacle_resource` every week of run 09261721).
MAX_CANDIDATES_PER_MOTIF = 2
# Lesson-driven candidates: how many lessons are considered per week, and the
# relation score attached to them (they carry no edge score of their own).
MAX_LESSON_MOTIF_CANDIDATES = 3
LESSON_MOTIF_SCORE = 0.7
# Ranking nudge so a lesson candidate competes with pattern motifs that carry
# twice as many sources (measured: it otherwise always lost the 4 phrasing slots).
LESSON_MOTIF_PRIOR_BOOST = 1.3
# Similarity above which an idea counts as a rewrite of something that exists.
NOVELTY_REJECT_THRESHOLD = 0.6
# Semantic duplication (KI-5): the same insight re-proposed with a different
# partner memory is not a new idea. Judged on the *evidence set* plus the text,
# because the wording changes every week while the insight does not.
SOURCE_OVERLAP_DUPLICATE = 0.3
TEST_PLAN_DUPLICATE_SIMILARITY = 0.45

# A requirement named through a "soft" entity (mentioned by the character
# but not a canonical world entity): allowed, slightly less certain (KI-7).
SOFT_REQUIREMENT_FACTOR = 0.9
# Evidence confidence below which sources count as "all low confidence".
LOW_CONFIDENCE_THRESHOLD = 0.4
# How many penalty reasons a candidate may collect before it is dropped.
MAX_PENALTIES = 2

PENALTY_HALVING = 0.5


@dataclass
class IdeaConfig:
    """`world.cognition` idea settings, inert by default."""

    enabled: bool = False
    max_ideas_per_week: int = MAX_IDEAS_PER_WEEK
    max_extract_calls_per_week: int = 1
    # Budget for the idea → method ("methodisation") call (KI-6).
    max_conversion_calls_per_week: int = 1
    # Optional hard floor: below this vitality the weekly budget is 0. Default 0
    # means "a tired character still gets one idea" (see weekly_idea_budget).
    min_vitality: float = 0.0
    min_potential: float = MIN_IDEA_POTENTIAL

    @staticmethod
    def from_world_config(world_cfg: Dict[str, Any]) -> "IdeaConfig":
        section = (world_cfg or {}).get("cognition") or {}

        def _int(key: str, default: int) -> int:
            try:
                return int(section.get(key, default))
            except (TypeError, ValueError):
                return default

        def _float(key: str, default: float) -> float:
            try:
                return float(section.get(key, default))
            except (TypeError, ValueError):
                return default

        return IdeaConfig(
            enabled=bool(section.get("idea_engine", False)),
            max_ideas_per_week=max(0, _int("idea_max_per_week", MAX_IDEAS_PER_WEEK)),
            max_extract_calls_per_week=max(
                0, _int("idea_max_extract_calls_per_week", 1)
            ),
            max_conversion_calls_per_week=max(
                0, _int("idea_max_conversion_calls_per_week", 1)
            ),
            min_vitality=max(0.0, _float("idea_min_vitality", 0.0)),
            min_potential=max(0.0, _float("idea_min_potential", MIN_IDEA_POTENTIAL)),
        )


@dataclass
class IdeaWorldState:
    """Objective facts the feasibility check is allowed to use."""

    skills: Set[str] = field(default_factory=set)
    entities: Set[str] = field(default_factory=set)
    locations: Set[str] = field(default_factory=set)
    deposit: float = 0.0
    vitality: float = 100.0
    scheduled_activity_times: Set[str] = field(default_factory=set)
    # The persona's alias book (KI-2): resolves "日和" → 吉日和, "菜市场" →
    # Vegetable_Market, and knows which unlisted names the character has
    # actually mentioned (soft entities).
    impressions: Any = None
    observed: Set[str] = field(default_factory=set)

    def resolve_entity(self, name: Any) -> Tuple[Optional[str], str]:
        """`(canonical, how)`; `how` is exact | alias | containment | observed."""
        raw = str(name or "").strip()
        if not raw:
            return None, ""
        if raw in self.entities:
            return raw, "exact"
        if self.impressions is not None:
            found, how = self.impressions.resolve(raw)
            if found:
                return found, how or "alias"
        if raw in self.observed:
            return raw, "observed"
        return None, ""

    def missing_entities(self, names: Iterable[str]) -> List[str]:
        """Names that resolve to nothing the character can actually reach."""
        return [str(n) for n in names if not self.resolve_entity(n)[0]]

    def soft_requirements(self, names: Iterable[str]) -> List[str]:
        """Requirements met only by a name the character has mentioned (KI-7).

        These no longer hard-fail the idea; they lower its feasibility.
        """
        out: List[str] = []
        for name in names:
            _canonical, how = self.resolve_entity(name)
            if how == "observed":
                out.append(str(name))
        return out

    def missing_skills(self, names: Iterable[str]) -> List[str]:
        return [str(n) for n in names if str(n) not in self.skills]


def world_state_from(dm, *, persona_skills: Optional[Set[str]] = None) -> IdeaWorldState:
    """Collect the facts used for feasibility from the live world."""
    from src.agents.cognition.consolidator import Consolidator

    skills = set(persona_skills or ())
    entities: Set[str] = set()
    locations: Set[str] = set()
    deposit = 0.0
    vitality = 100.0

    try:
        state = dm.read_state(exclude_cur_t=False)
        if not skills:
            skills = {str(k) for k in (state.get("skills") or {}).keys()}
        deposit = float((state.get("assets") or {}).get("deposit") or 0.0)
        vitality = float(state.get("vitality") or 0.0)
        for item in (state.get("assets") or {}).get("possessions") or []:
            name = item.get("name") if isinstance(item, dict) else item
            if name:
                entities.add(str(name))
    except (IndexError, FileNotFoundError, AttributeError):
        pass

    # Reuse the consolidator's world vocabulary (personas, contacts, places).
    try:
        helper = Consolidator(
            dm=dm, clock=dm.clock, agent_name=dm.char, model=""
        )
        entities |= helper.known_entities()
    except Exception:  # pragma: no cover - defensive
        pass

    try:
        store = getattr(dm, "location_store", None)
        if store is not None:
            for key, location in list(getattr(store, "public", {}).items()) + list(
                getattr(store, "private", {}).items()
            ):
                locations.add(str(key))
                display = getattr(location, "display_name", None) or (
                    location.get("display_name") if isinstance(location, dict) else None
                )
                if display:
                    locations.add(str(display))
        entities |= locations
    except Exception:  # pragma: no cover - defensive
        pass

    impressions = None
    observed: Set[str] = set()
    try:
        from src.agents.cognition.impressions import ensure_impressions

        clock = getattr(dm, "clock", None)
        if clock is not None:
            t = clock.get_time()
            _keeper, impressions = ensure_impressions(
                dm, clock, week=f"Y{t.year}-W{t.week:02d}"
            )
            observed = set(impressions.soft_entities)
            entities |= impressions.known()
    except Exception:  # pragma: no cover - feasibility must never break a run
        pass

    return IdeaWorldState(
        skills=skills,
        entities=entities,
        locations=locations,
        deposit=deposit,
        vitality=vitality,
        impressions=impressions,
        observed=observed,
    )


# ---------------------------------------------------------------------------
# Motif detection (rules only)
# ---------------------------------------------------------------------------
def _by_id(memories: Sequence[MemoryItem]) -> Dict[str, MemoryItem]:
    return {m.memory_id: m for m in memories}


def _shared_goal(a: MemoryItem, b: MemoryItem) -> str:
    shared = set(a.goal_ids) & set(b.goal_ids)
    if shared:
        return sorted(shared)[0]
    return (a.goal_ids or b.goal_ids or [""])[0]


def _resource_meets_obstacle(a: MemoryItem, b: MemoryItem) -> Tuple[List[str], List[str]]:
    """Return (obstacles, resources) for a problem-resource pair.

    Only the *exact* string intersection used to be returned. Since the relation
    rule now matches by graded affinity (KI-3), every containment / bigram /
    topical match produced a candidate whose obstacle and resource lists were
    empty — `_complementarity` then returned 0, IdeaPotential was exactly 0, and
    the candidate was rejected as `below_min_potential` **while still consuming
    one of the four phrasing slots** (measured on run 09261721: 17 of 17
    rejections for 萧亦岚 were exactly this).

    The fix is to hand over the texts the memories actually carry, with the side
    that answers better deciding the direction.
    """
    a_answered_by_b = _affinity(a.obstacles, b.resources)
    b_answered_by_a = _affinity(b.obstacles, a.resources)
    if a_answered_by_b <= 0 and b_answered_by_a <= 0:
        # No lexical affinity at all: the caller only reaches here because the
        # relation rule found a weak, same-topic problem/resource pair (KI-3).
        # Hand over the shape the memories have, and let `_complementarity`
        # score it as a weak match (0.5) instead of as a certain failure.
        if a.obstacles and b.resources:
            return list(a.obstacles), list(b.resources)
        if b.obstacles and a.resources:
            return list(b.obstacles), list(a.resources)
        return [], []
    if a_answered_by_b >= b_answered_by_a:
        return list(a.obstacles), list(b.resources)
    return list(b.obstacles), list(a.resources)


def find_motif_candidates(
    memories: Sequence[MemoryItem],
    edges: Sequence[Dict[str, Any]],
    *,
    traits: Optional[Dict[str, float]] = None,
    max_candidates: int = MAX_CANDIDATES_FOR_PHRASING,
) -> List[IdeaCandidate]:
    """Rule-based motif detection over the phase 2 relation graph."""
    traits = traits or {}
    index = _by_id(memories)
    creativity = traits.get("creativity")
    curiosity = traits.get("curiosity")

    # Creativity widens the analogy distance the persona is willing to bridge:
    # distance is "how far apart the two practices are" (1 - relation score),
    # and the threshold grows with creativity (creativity 50 -> 55, 90 -> 75).
    analogy_threshold = 30.0 + 0.5 * float(creativity) if creativity is not None else 55.0
    allow_cold = curiosity is not None and float(curiosity) >= 50.0

    def usable(memory: MemoryItem) -> bool:
        if memory.status == "superseded":
            return False
        if memory.tier in ("cold", "archived") and not allow_cold:
            return False
        return True

    candidates: List[IdeaCandidate] = []
    seen: Set[Tuple[str, Tuple[str, ...]]] = set()

    def add(candidate: IdeaCandidate) -> None:
        ids = tuple(sorted(candidate.memory_ids))
        key = (candidate.motif, ids)
        if key in seen or len(ids) < 2:
            return
        seen.add(key)
        candidates.append(candidate)

    for edge in edges:
        types = set(edge.get("types") or [])
        a = index.get(str(edge.get("source")))
        b = index.get(str(edge.get("target")))
        if a is None or b is None or not usable(a) or not usable(b):
            continue
        score = float(edge.get("score") or 0.0)
        shared_focus = str(edge.get("shared_focus") or "")

        if "problem_resource" in types or "precondition" in types:
            obstacles, resources = _resource_meets_obstacle(a, b)
            if obstacles and resources:
                add(
                    IdeaCandidate(
                        motif="goal_obstacle_resource",
                    memory_ids=[a.memory_id, b.memory_id],
                    relationship_types=sorted(types),
                    goal=_shared_goal(a, b),
                    obstacles=obstacles,
                    resources=resources,
                    shared_focus=shared_focus,
                    score=score,
                    hint="一个障碍遇上了一个可用的资源",
                )
            )

        if "repeated_pattern" in types:
            family = [
                m.memory_id
                for m in memories
                if m.outcome_polarity == a.outcome_polarity
                and set(m.topics) & set(a.topics)
                and (allow_cold or m.tier not in ("cold", "archived"))
            ][:4]
            add(
                IdeaCandidate(
                    motif="repeated_pattern",
                    memory_ids=sorted(set([a.memory_id, b.memory_id] + family))[:4],
                    relationship_types=sorted(types),
                    goal=_shared_goal(a, b),
                    shared_focus=shared_focus,
                    score=score,
                    hint="同样的条件反复导致同样的结果",
                )
            )

        if "contradiction" in types:
            add(
                IdeaCandidate(
                    motif="contradiction",
                    memory_ids=[a.memory_id, b.memory_id],
                    relationship_types=sorted(types),
                    goal=_shared_goal(a, b),
                    shared_focus=shared_focus,
                    score=score,
                    hint="同一件事有时成功有时失败，差异可能来自情境",
                )
            )

        if "method_transfer" in types:
            different_skills = set(a.skill_ids) != set(b.skill_ids)
            analogy_distance = (1.0 - score) * 100.0
            if different_skills and analogy_distance <= analogy_threshold:
                add(
                    IdeaCandidate(
                        motif="cross_domain_analogy",
                        memory_ids=[a.memory_id, b.memory_id],
                        relationship_types=sorted(types),
                        goal=_shared_goal(a, b),
                        shared_focus=shared_focus,
                        score=score,
                        hint="另一个领域的做法可能迁移过来",
                    )
                )

        if "gap" in types:
            add(
                IdeaCandidate(
                    motif="causal_gap",
                    memory_ids=[a.memory_id, b.memory_id],
                    relationship_types=sorted(types),
                    goal=_shared_goal(a, b),
                    obstacles=sorted(set(a.obstacles) | set(b.obstacles)),
                    shared_focus=shared_focus,
                    score=score,
                    hint="目标两侧都卡住、中间缺少一步",
                )
            )

    # unused resource: a resource memory whose goal has not progressed
    for resource_memory in memories:
        if not usable(resource_memory) or not resource_memory.resources:
            continue
        for goal_memory in memories:
            if goal_memory.memory_id == resource_memory.memory_id or not usable(goal_memory):
                continue
            shared = set(resource_memory.goal_ids) & set(goal_memory.goal_ids)
            if not shared or goal_memory.outcome_polarity == "positive":
                continue
            if set(resource_memory.resources) & set(goal_memory.obstacles):
                continue  # that is problem_resource, already covered
            add(
                IdeaCandidate(
                    motif="unused_resource",
                    memory_ids=[resource_memory.memory_id, goal_memory.memory_id],
                    relationship_types=["same_goal"],
                    goal=sorted(shared)[0],
                    resources=list(resource_memory.resources),
                    obstacles=list(goal_memory.obstacles),
                    shared_focus=sorted(shared)[0],
                    score=0.5,
                    hint="手上有资源，但目标还没有推进",
                )
            )

    # -- lesson_application (phase 4.5) -------------------------------------
    # The character's own notes contain lessons it wrote for itself ("next time,
    # do X"). Those are its most personal, most actionable material, and pairing
    # one with a goal or an experience of its own is exactly the "hypothesis to
    # test" the idea layer exists for. This runs next to — not instead of — the
    # edge rules above: the random-combination motifs keep working as before.
    lesson_memories = [
        m
        for m in memories
        if usable(m) and str(getattr(m, "origin", "")) == "scratchpad_lesson"
    ]
    non_lesson = [m for m in memories if usable(m) and m not in lesson_memories]
    for lesson in lesson_memories[:MAX_LESSON_MOTIF_CANDIDATES]:
        partner = _lesson_partner(lesson, non_lesson)
        if partner is None:
            # No goal or experience to attach it to yet: the lesson still pairs
            # with another lesson, so an early-week character is not silent.
            partner = _lesson_partner(lesson, [m for m in lesson_memories if m is not lesson])
        if partner is None:
            continue
        shared_goal = _shared_goal(lesson, partner)
        if not shared_goal and partner.kind == "goal":
            shared_goal = partner.content
        add(
            IdeaCandidate(
                motif="lesson_application",
                memory_ids=[lesson.memory_id, partner.memory_id],
                relationship_types=["lesson"],
                goal=shared_goal,
                obstacles=list(lesson.obstacles) or [lesson.content],
                resources=list(partner.resources) or list(lesson.resources),
                shared_focus=shared_goal or (lesson.entities[:1] or [""])[0],
                score=LESSON_MOTIF_SCORE,
                hint=f"角色自己写下的教训：{lesson.content}",
            )
        )

    # Rank by *expected* potential (everything the score can know before the
    # model phrases it) rather than by the edge score: a high-scoring edge whose
    # candidate can never pass the gate must not take a slot from a candidate
    # that can.
    index = _by_id(memories)
    candidates.sort(
        key=lambda c: (
            -_candidate_prior(c, index, traits),
            -c.score,
            c.motif,
            tuple(c.memory_ids),
        )
    )
    selected: List[IdeaCandidate] = []
    per_motif: Dict[str, int] = {}
    for candidate in candidates:
        if len(selected) >= max_candidates:
            break
        used = per_motif.get(candidate.motif, 0)
        if used >= MAX_CANDIDATES_PER_MOTIF:
            continue
        per_motif[candidate.motif] = used + 1
        selected.append(candidate)
    # The character's own lesson keeps one slot. Losing the ranking to a pattern
    # motif is the normal case (fewer sources), and "what I concluded never gets
    # tried" is exactly the failure this phase removes.
    if not any(c.motif == "lesson_application" for c in selected):
        reserve = next((c for c in candidates if c.motif == "lesson_application"), None)
        if reserve is not None and _candidate_prior(reserve, index, traits) > 0:
            if len(selected) >= max_candidates and selected:
                selected = selected[:-1]
            selected.append(reserve)

    if selected and len(selected) < max_candidates:
        # Fill the remaining slots with the best leftovers — still respecting the
        # per-motif share, so "one loud motif" cannot take the whole budget.
        for candidate in candidates:
            if len(selected) >= max_candidates:
                break
            if candidate in selected:
                continue
            if per_motif.get(candidate.motif, 0) >= MAX_CANDIDATES_PER_MOTIF:
                continue
            per_motif[candidate.motif] = per_motif.get(candidate.motif, 0) + 1
            selected.append(candidate)
    if not selected:
        # Only one motif fired and the cap swallowed everything: phrase its best
        # candidates rather than skipping the week entirely.
        selected = list(candidates[:max_candidates])
    return selected


def _lesson_partner(lesson, candidates: Sequence[MemoryItem]) -> Optional[MemoryItem]:
    """The memory a lesson should be attached to (deterministic best match).

    Preference order: a goal it speaks to, then an experience sharing an entity
    or topic with it, then anything strong. Ties break on strength, entity
    overlap and memory id, so the same week always picks the same partner.
    """
    scored: List[Tuple[Tuple[int, int, float, str], MemoryItem]] = []
    for memory in candidates:
        if memory.memory_id == lesson.memory_id or memory.status == "superseded":
            continue
        shared_topics = len(set(lesson.topics) & set(memory.topics))
        shared_entities = len(set(lesson.entities) & set(memory.entities))
        shared_skills = len(set(lesson.skill_ids) & set(memory.skill_ids))
        overlap = shared_topics + shared_entities + shared_skills
        is_goal = 1 if memory.kind == "goal" else 0
        if not is_goal and overlap == 0:
            continue  # nothing in common: not a partner, just a coincidence
        scored.append(
            (
                (is_goal, overlap, float(memory.strength or 0.0), memory.memory_id),
                memory,
            )
        )
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _candidate_prior(
    candidate: IdeaCandidate,
    index: Dict[str, MemoryItem],
    traits: Dict[str, float],
) -> float:
    """Expected IdeaPotential before the model writes the wording.

    The phrasing step supplies `actionability` (≈1.0 when a test plan exists)
    and `novelty` (measured against what already exists); every other factor of
    `IdeaPotential` is knowable here. Ranking on this proxy is what keeps the
    four slots for candidates that can actually pass the gate.
    """
    sources = [index[mid] for mid in candidate.memory_ids if mid in index]
    if not sources:
        return 0.0
    solid = [m for m in sources if m.is_evidence_backed]
    groundedness = min(1.0, 0.25 * len(solid))
    if len(solid) < 2:
        return 0.0
    complementarity = _complementarity(candidate, index)
    if complementarity <= 0:
        return 0.0
    evidence_confidence = sum(m.confidence for m in solid) / len(solid)
    goal_relevance = 1.0 if candidate.goal else 0.5
    score = (
        groundedness
        * complementarity
        * goal_relevance
        * _persona_fit(candidate, traits=traits)
        * evidence_confidence
    )
    if candidate.motif == "lesson_application":
        # A lesson pairs with one other memory, so it carries fewer sources than
        # a pattern motif and would lose the prior ranking on groundedness alone.
        # The character's own conclusion is the material this phase exists for,
        # so it gets a modest edge — not a free pass: it still has to clear the
        # same gate, and its evidence confidence stays a belief's.
        score *= LESSON_MOTIF_PRIOR_BOOST
    return score


# ---------------------------------------------------------------------------
# Scoring and the five checks
# ---------------------------------------------------------------------------
def _complementarity(candidate: IdeaCandidate, memories: Dict[str, MemoryItem]) -> float:
    if candidate.motif == "goal_obstacle_resource":
        obstacles = [o for o in candidate.obstacles if str(o).strip()]
        resources = [r for r in candidate.resources if str(r).strip()]
        if not obstacles or not resources:
            return 0.0
        # Graded affinity (KI-3): exact > containment > shared bigrams, so the
        # motif is reachable on free text instead of requiring equal strings.
        answered = sum(1 for o in obstacles if _affinity([o], resources) > 0)
        if answered == 0:
            # One side states the problem, the other states a resource: the
            # motif is still worth phrasing, at half credit.
            return 0.5
        return min(1.0, answered / len(obstacles))
    if candidate.motif == "repeated_pattern":
        return min(1.0, len(candidate.memory_ids) / 5.0)
    if candidate.motif == "contradiction":
        return 1.0
    if candidate.motif == "cross_domain_analogy":
        return 0.5
    if candidate.motif == "causal_gap":
        return 0.6
    if candidate.motif == "lesson_application":
        # A lesson the character wrote itself, attached to something it wants:
        # the two sides are complementary by construction (that is why it is the
        # most actionable material it has), but it is still only a hypothesis.
        return 0.7
    if candidate.motif == "unused_resource":
        return 0.5
    return 0.3


def _novelty(
    content: str,
    *,
    existing_ideas: Sequence[Idea],
    existing_methods: Sequence[Any],
) -> Tuple[float, float]:
    """Return (novelty, worst similarity against what already exists)."""
    tokens = token_set(content)
    worst = 0.0
    for idea in existing_ideas:
        worst = max(worst, jaccard(tokens, token_set(idea.content)))
    for method in existing_methods:
        text = f"{getattr(method, 'title', '')} {getattr(method, 'description', '')}"
        worst = max(worst, jaccard(tokens, token_set(text)))
    return max(0.0, 1.0 - worst), worst


def _source_overlap(left: Sequence[str], right: Sequence[str]) -> float:
    """Jaccard of two evidence sets (empty sets never count as duplicates)."""
    a, b = {str(v) for v in left if str(v)}, {str(v) for v in right if str(v)}
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _semantic_duplicate(
    *,
    candidate: IdeaCandidate,
    content: str,
    test_plan: str,
    existing_ideas: Sequence[Idea],
    existing_methods: Sequence[Any],
) -> Optional[str]:
    """The id an existing idea repeats, or None (KI-5).

    Lexical novelty cannot see this: the phrasing model rewrites the same
    insight every week, and `idea_id` is keyed on the evidence set, so a new
    partner memory produces a "new" idea whose *content* is the old one. This
    looks at the evidence set and the test plan as well, and only fires when
    both the evidence overlaps and the text is close — a genuinely new partner
    with a genuinely new plan still gets through.

    Root cause of the repetition this fixes (measured on the 1-year run): the
    motif rules fan out from one hub memory (1 negative lesson × N positive
    memories ⇒ N contradiction ideas), and `_complementarity` gives
    `contradiction` the highest score of all motifs, so the same insight is
    re-derived every week.
    """
    tokens = token_set(content)
    plan_tokens = token_set(test_plan)
    for idea in existing_ideas:
        overlap = _source_overlap(candidate.memory_ids, idea.source_memory_ids)
        if overlap < SOURCE_OVERLAP_DUPLICATE:
            continue
        content_similarity = jaccard(tokens, token_set(idea.content))
        plan_similarity = (
            jaccard(plan_tokens, token_set(idea.test_plan)) if test_plan and idea.test_plan else 0.0
        )
        if content_similarity >= NOVELTY_REJECT_THRESHOLD or plan_similarity >= TEST_PLAN_DUPLICATE_SIMILARITY:
            return idea.idea_id
    for method in existing_methods:
        if str(getattr(method, "source_type", "")) != "idea_conversion":
            continue
        overlap = _source_overlap(candidate.memory_ids, getattr(method, "source_memory_ids", []) or [])
        if overlap < SOURCE_OVERLAP_DUPLICATE:
            continue
        text = f"{getattr(method, 'title', '')} {getattr(method, 'description', '')}"
        if jaccard(tokens, token_set(text)) >= NOVELTY_REJECT_THRESHOLD:
            return str(getattr(method, "source_idea_id", "") or getattr(method, "method_id", ""))
    return None


def _persona_fit(
    candidate: IdeaCandidate, *, traits: Dict[str, float]
) -> float:
    """How well the motif suits this persona (design doc 8.8), bounded [0.5, 1]."""
    fit = 0.8
    creativity = traits.get("creativity")
    if candidate.motif == "cross_domain_analogy" and creativity is not None:
        fit = 0.5 + 0.5 * min(1.0, float(creativity) / 100.0)
    curiosity = traits.get("curiosity")
    if candidate.motif == "unused_resource" and curiosity is not None:
        fit = 0.5 + 0.5 * min(1.0, float(curiosity) / 100.0)
    return max(0.5, min(1.0, fit))


@dataclass
class IdeaDecision:
    """Outcome of the five checks plus the score."""

    accepted: bool
    reasons: List[str]
    potential: IdeaPotential
    feasibility: float
    worst_similarity: float = 0.0
    # Set when the candidate restates an insight that already exists (KI-5);
    # the caller records IDEA_REFINED instead of a second idea.
    duplicate_of: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "feasibility": round(self.feasibility, 4),
            "worst_similarity": round(self.worst_similarity, 4),
            "duplicate_of": self.duplicate_of,
            "potential": self.potential.to_dict(),
        }


def evaluate_candidate(
    candidate: IdeaCandidate,
    *,
    content: str,
    test_plan: str,
    requires: Optional[Dict[str, Any]] = None,
    memories: Sequence[MemoryItem],
    existing_ideas: Sequence[Idea],
    existing_methods: Sequence[Any],
    world_state: IdeaWorldState,
    traits: Optional[Dict[str, float]] = None,
    min_potential: float = MIN_IDEA_POTENTIAL,
) -> IdeaDecision:
    """Apply the five gating checks and compute IdeaPotential (design doc 8.7)."""
    traits = traits or {}
    requires = requires or {}
    index = _by_id(memories)
    sources = [index[i] for i in candidate.memory_ids if i in index]
    reasons: List[str] = []

    # -- check 1: grounding (2-4 memories, each with evidence) --------------
    solid = [m for m in sources if m.is_evidence_backed]
    groundedness = min(1.0, 0.25 * len(solid))
    if len(solid) < 2:
        reasons.append("not_grounded")
    if len(solid) > 4:
        sources = solid[:4]

    # -- check 2: novelty ---------------------------------------------------
    novelty, worst_similarity = _novelty(
        content, existing_ideas=existing_ideas, existing_methods=existing_methods
    )
    if worst_similarity >= NOVELTY_REJECT_THRESHOLD:
        reasons.append("duplicates_existing_idea_or_method")
    duplicate_of = _semantic_duplicate(
        candidate=candidate,
        content=content,
        test_plan=test_plan,
        existing_ideas=existing_ideas,
        existing_methods=existing_methods,
    )
    if duplicate_of:
        # Re-proposing an insight already in the store is a *refinement*, not a
        # new idea: reject it here so the caller can record IDEA_REFINED.
        reasons.append("duplicates_existing_idea_semantics")

    # -- check 3: feasibility ----------------------------------------------
    required_entities = requires.get("entities") or []
    missing_entities = world_state.missing_entities(required_entities)
    soft_entities = world_state.soft_requirements(required_entities)
    missing_skills = world_state.missing_skills(requires.get("skills") or [])
    try:
        money_needed = float(requires.get("money") or 0)
    except (TypeError, ValueError):
        money_needed = 0.0
    money_short = money_needed > world_state.deposit

    feasibility = 1.0
    if missing_entities or missing_skills:
        reasons.append("depends_on_missing_entity")
        feasibility = 0.2
    elif soft_entities:
        # The character referred to something they have mentioned but that is
        # not (yet) a world entity: allowed, a little less certain (KI-7).
        feasibility = min(feasibility, SOFT_REQUIREMENT_FACTOR)
    if money_short:
        reasons.append("conflicts_with_world_state")
        feasibility = min(feasibility, 0.3)

    # -- check 4: testability ----------------------------------------------
    if not test_plan.strip():
        reasons.append("not_testable")

    # -- check 5: safety / consistency -------------------------------------
    evidence_confidence = (
        sum(m.confidence for m in solid) / len(solid) if solid else 0.0
    )
    if evidence_confidence < LOW_CONFIDENCE_THRESHOLD:
        reasons.append("sources_low_confidence")
    if requires.get("asserts_fact"):
        reasons.append("conflicts_with_world_state")

    # -- penalties that only lower the score -------------------------------
    penalties: List[str] = []
    if novelty < 0.25:
        penalties.append("duplicates_existing_plan")

    # Checks that must pass for the idea to enter the store at all: grounding,
    # novelty, feasibility and testability (design doc 8.7). Everything else
    # only lowers the score.
    # Checks that must pass for the idea to enter the store at all (design doc
    # 8.7 checks 1-4). "conflicts_with_world_state" is a feasibility failure
    # (the character cannot afford it), not merely a score penalty.
    hard_gate_failures = {
        "not_grounded",
        "duplicates_existing_idea_or_method",
        "depends_on_missing_entity",
        "conflicts_with_world_state",
        "not_testable",
    }
    gating_failures = [r for r in reasons if r in hard_gate_failures]
    soft_reasons = [r for r in reasons if r not in hard_gate_failures]
    all_penalties = penalties + soft_reasons

    potential = IdeaPotential(
        groundedness=groundedness,
        complementarity=_complementarity(candidate, index),
        novelty=novelty,
        actionability=0.4 + 0.6 * feasibility,
        goal_relevance=1.0 if candidate.goal else 0.5,
        persona_fit=_persona_fit(candidate, traits=traits),
        evidence_confidence=evidence_confidence,
        penalty_factor=PENALTY_HALVING ** len(all_penalties),
        penalties=all_penalties,
    )

    threshold = min_potential
    intelligence = traits.get("intelligence")
    if intelligence is not None:
        # Filtering quality only, bounded to +-15% (design doc 8.8: persona
        # shapes filtering, it never changes the underlying scores).
        threshold = min_potential * (1.15 - 0.3 * min(1.0, float(intelligence) / 100.0))

    accepted = (
        not gating_failures
        and len(all_penalties) <= MAX_PENALTIES
        and potential.potential >= threshold
    )
    reasons = gating_failures + all_penalties
    if potential.potential < threshold and "below_min_potential" not in reasons:
        # Always name the scoring floor explicitly; otherwise a rejection can be
        # recorded with a reason list that hides *why* nothing was stored
        # (`sources_low_confidence` alone, for instance).
        reasons.append("below_min_potential")
    return IdeaDecision(
        accepted=accepted,
        reasons=reasons,
        potential=potential,
        feasibility=feasibility,
        worst_similarity=worst_similarity,
        duplicate_of=duplicate_of or "",
    )


def weekly_idea_budget(
    config: IdeaConfig,
    *,
    vitality: float,
    traits: Optional[Dict[str, float]] = None,
    min_vitality: Optional[float] = None,
) -> int:
    """How many ideas this character can afford this week (design doc 8.8).

    vitality/stress set the weekly budget; low confidence makes a persona less
    willing to try something new. Bounded by `idea_max_per_week`.

    Two corrections from run 09261721 (2026-09-26):

    * **0 is not "no cognition".** The original rule (`vitality < 20 -> 0`)
      assumed vitality fluctuates. In this world vitality starts at 70 and only
      drains (no rest mechanic), so every long run ends with the whole cast at
      0 — and the idea engine silently produced nothing for the back half of the
      year. A drained character now thinks once a week; a hard stop requires an
      explicit `idea_min_vitality` floor in the config.
    * **The caller must pass the week's own vitality** (plan time), not the
      post-week low: see `RoleAgent._week_start_vitality`.
    """
    budget = config.max_ideas_per_week
    floor = config.min_vitality if min_vitality is None else float(min_vitality)
    if floor > 0 and float(vitality) < floor:
        return 0
    if float(vitality) < 40:
        # Tired: at most one idea this week. Never zero for a living character.
        budget = min(budget, 1)
    confidence = (traits or {}).get("confidence")
    if confidence is not None and float(confidence) < 30:
        budget = min(budget, 1)
    return max(0, budget)


def build_idea(
    *,
    candidate: IdeaCandidate,
    phrased: Dict[str, Any],
    persona: str,
    created_at: str,
    expires_at: str,
    decision: IdeaDecision,
) -> Idea:
    """Assemble the stored Idea from the motif, the phrasing and the decision."""
    content = str(phrased.get("content") or "").strip()
    idea = Idea(
        idea_id=idea_id_for(
            persona=persona, motif=candidate.motif, source_memory_ids=candidate.memory_ids
        ),
        content=content[:300],
        idea_type=str(phrased.get("idea_type") or "hypothesis"),
        motif=candidate.motif,
        source_memory_ids=list(candidate.memory_ids),
        relationship_types=list(candidate.relationship_types),
        related_goal_ids=[candidate.goal] if candidate.goal else [],
        related_skill_ids=list(phrased.get("skills") or [])[:4],
        novelty=decision.potential.novelty,
        feasibility=decision.feasibility,
        groundedness=decision.potential.groundedness,
        confidence=min(
            CANDIDATE_CONFIDENCE_CEILING,
            max(0.0, float(phrased.get("confidence") or 0.3)),
        ),
        status="candidate",
        created_at=created_at,
        expires_at=expires_at,
        test_plan=str(phrased.get("test_plan") or "")[:300],
        potential=decision.potential.potential,
        score_components=decision.potential.to_dict(),
        persona=persona,
    )
    return idea
