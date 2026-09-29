"""Memory relation graph: inverted-index recall + rule verdicts (doc 8, draft §5).

Phase 2 is deliberately rule-only (no embeddings, no LLM judge): every relation
that is asserted must be explainable by a rule a reviewer can check. Recall is
index-based, so building relations stays O(memories x candidates) instead of
O(n^2).

The ten relation types of design doc 8.1 are all decidable from structured
fields:

    same_entity        entity overlap
    same_goal          goal overlap
    causal             same entity+topic, polarity flips, close in time
    temporal           close in time *and* something else shared (time alone
                       never proves relevance, per doc 8.1)
    repeated_pattern   the same (topic, polarity) family has >= 3 members
    contradiction      same entity/topic with opposite polarity
    problem_resource   one memory's obstacles meet another's resources
    method_transfer    different skills, but overlapping topics
    precondition       resources meet obstacles *and* a shared goal
    gap                shared goal, both blocked, nobody supplies the resource
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from src.agents.cognition.memory_models import (
    _affinity,
    shared_anchor,
    MemoryItem,
    MemoryRelation,
    RELATION_TYPES,
    jaccard,
    token_set,
)
from src.agents.cognition.memory_strength import weeks_between

MAX_CANDIDATES_PER_SEED = 20
DEFAULT_EXPLORATION_CANDIDATES = 2
DEFAULT_MAX_RELATIONS_PER_WEEK = 10
NEAR_DUPLICATE_JACCARD = 0.85
REPEATED_PATTERN_MIN_FAMILY = 3

# A pair that states a problem on one side and a resource on the other within
# the same topic, without any lexical overlap (KI-3).
WEAK_PROBLEM_RESOURCE = 0.3


@dataclass
class RelationWeights:
    """Weights of the explainable RelationScore (design draft 5.2).

    Initial values only: the design document requires tuning them against
    historical runs rather than hard-coding a personality difference.
    """

    semantic: float = 0.2
    entity: float = 0.2
    goal: float = 0.15
    skill: float = 0.1
    time: float = 0.1
    problem_resource: float = 0.15
    analogy: float = 0.05
    contradiction: float = 0.1
    # A near-duplicate must score below the assertion threshold: "highly similar
    # but merely repeated" carries no relation worth recording (design doc 8.1).
    # With 0.25 a pure duplicate (semantic 1.0 + entity 1.0 + time) still scored
    # 0.25, so the penalty is deliberately larger than the components it cancels.
    redundancy: float = 0.5


def _overlap(a: Iterable[str], b: Iterable[str]) -> Set[str]:
    return set(a) & set(b)


class InvertedIndex:
    """Cheap candidate recall over structured fields (design draft §5.1)."""

    def __init__(self, memories: Sequence[MemoryItem]) -> None:
        self.memories = list(memories)
        self.by_entity: Dict[str, List[str]] = {}
        self.by_topic: Dict[str, List[str]] = {}
        self.by_skill: Dict[str, List[str]] = {}
        self.by_goal: Dict[str, List[str]] = {}
        self.by_method: Dict[str, List[str]] = {}
        self.by_obstacle: Dict[str, List[str]] = {}
        self.by_resource: Dict[str, List[str]] = {}
        self.families: Dict[str, List[str]] = {}
        self._by_id: Dict[str, MemoryItem] = {}

        for memory in self.memories:
            self._by_id[memory.memory_id] = memory
            self._index(self.by_entity, memory.entities, memory.memory_id)
            self._index(self.by_topic, memory.topics, memory.memory_id)
            self._index(self.by_skill, memory.skill_ids, memory.memory_id)
            self._index(self.by_goal, memory.goal_ids, memory.memory_id)
            self._index(self.by_method, memory.method_ids, memory.memory_id)
            self._index(self.by_obstacle, memory.obstacles, memory.memory_id)
            self._index(self.by_resource, memory.resources, memory.memory_id)
            for topic in memory.topics:
                self.families.setdefault(f"{topic}|{memory.outcome_polarity}", []).append(
                    memory.memory_id
                )

    @staticmethod
    def _index(bucket: Dict[str, List[str]], keys: Iterable[str], memory_id: str) -> None:
        for key in keys:
            bucket.setdefault(str(key), []).append(memory_id)

    @staticmethod
    def _family(memory: MemoryItem) -> str:
        """Repeated-pattern family key.

        Kept for callers that still key on a memory; the *counting* now happens
        per (topic, polarity) — see `shared_families`. Keying on the complete
        topic set (the phase-2 rule) produced families of size 1 for every
        memory, so `repeated_pattern` could never reach the 3-member threshold
        (KI-3).
        """
        topics = ",".join(sorted(memory.topics))
        return f"{topics}|{memory.outcome_polarity}"

    def family_sizes(self, memory: MemoryItem) -> Dict[str, int]:
        """`{(topic, polarity): member count}` for the memory's own topics."""
        out: Dict[str, int] = {}
        for topic in memory.topics:
            key = f"{topic}|{memory.outcome_polarity}"
            out[key] = len(self.families.get(key, []))
        return out

    def get(self, memory_id: str) -> Optional[MemoryItem]:
        return self._by_id.get(memory_id)

    def candidates(
        self,
        seed: MemoryItem,
        *,
        limit: int = MAX_CANDIDATES_PER_SEED,
        exploration: int = DEFAULT_EXPLORATION_CANDIDATES,
        rng: Optional[random.Random] = None,
    ) -> List[MemoryItem]:
        """Memories worth comparing with `seed`: index hits plus a little noise."""
        found: List[str] = []
        seen: Set[str] = set()

        def add(ids: Iterable[str]) -> None:
            for memory_id in ids:
                if memory_id == seed.memory_id or memory_id in seen:
                    continue
                seen.add(memory_id)
                found.append(memory_id)
                if len(found) >= limit:
                    return

        for bucket, keys in (
            (self.by_entity, seed.entities),
            (self.by_goal, seed.goal_ids),
            (self.by_obstacle, seed.obstacles),
            (self.by_resource, seed.resources),
            (self.by_topic, seed.topics),
            (self.by_skill, seed.skill_ids),
            (self.by_method, seed.method_ids),
        ):
            for key in keys:
                add(bucket.get(str(key), []))
            if len(found) >= limit:
                break

        if rng is not None and exploration > 0 and len(found) < limit:
            pool = [m.memory_id for m in self.memories if m.memory_id not in seen and m.memory_id != seed.memory_id]
            if pool:
                picks = rng.sample(pool, min(exploration, len(pool)))
                add(picks)

        return [self._by_id[i] for i in found[:limit]]


def _temporal_closeness(a: MemoryItem, b: MemoryItem) -> float:
    weeks = weeks_between(a.created_at, b.created_at)
    if weeks <= 1:
        return 1.0
    if weeks <= 3:
        return 0.5
    return 0.0


def _problem_resource_overlap(a: MemoryItem, b: MemoryItem) -> float:
    """How much of one memory's obstacles the other's resources answer.

    Equality alone (the phase-2 rule) never fires on free text, so this now
    grades the match: exact equality > containment > shared bigrams (see
    `memory_models._affinity`). A pair that states an obstacle on one side and a
    resource on the other *within the same topic* counts as a weak match (0.3),
    which is what makes `goal_obstacle_resource` reachable on real data (KI-3);
    weak matches stay well below lexical ones so the ranking still prefers
    genuine matches.
    """
    needs = (1 if a.obstacles else 0) + (1 if b.obstacles else 0)
    if needs == 0:
        return 0.0
    hits = 0.0
    if a.obstacles and b.resources:
        hits += _affinity(a.obstacles, b.resources)
    if b.obstacles and a.resources:
        hits += _affinity(b.obstacles, a.resources)
    score = min(1.0, hits / needs)
    if score == 0.0 and set(a.topics) & set(b.topics):
        problems = bool(a.obstacles) and bool(b.resources)
        mirrored = bool(b.obstacles) and bool(a.resources)
        if problems or mirrored:
            score = WEAK_PROBLEM_RESOURCE
    return score


def relation_types_between(
    a: MemoryItem,
    b: MemoryItem,
    *,
    index: Optional[InvertedIndex] = None,
) -> List[str]:
    """All rule-decidable relation types holding between two memories."""
    if a.memory_id == b.memory_id:
        return []

    types: List[str] = []
    shared_entities = _overlap(a.entities, b.entities)
    shared_topics = _overlap(a.topics, b.topics)
    shared_goals = _overlap(a.goal_ids, b.goal_ids)
    shared_skills = _overlap(a.skill_ids, b.skill_ids)
    closes = _temporal_closeness(a, b)
    pr_overlap = _problem_resource_overlap(a, b)

    if shared_entities:
        types.append("same_entity")
    if shared_goals:
        types.append("same_goal")
    if pr_overlap > 0:
        types.append("problem_resource")
    if shared_goals and pr_overlap > 0:
        types.append("precondition")

    opposite = {a.outcome_polarity, b.outcome_polarity} == {"positive", "negative"}
    anchors = shared_anchor(a, b)
    # KI-4: a contradiction is "the same thing turned out differently", so it
    # needs a concrete anchor. Sharing only a coarse topic is not enough.
    if opposite and anchors:
        types.append("contradiction")

    if shared_skills and not shared_entities and shared_topics:
        # Same practice area, different people/places: analogy candidate.
        pass
    if shared_topics and set(a.skill_ids) != set(b.skill_ids) and (a.skill_ids or b.skill_ids):
        types.append("method_transfer")

    if (
        shared_entities
        and shared_topics
        and opposite
        and closes > 0
        and a.created_at != b.created_at
    ):
        types.append("causal")

    # Time alone never proves relevance (design doc 8.1): it only joins a pair
    # that already shares something.
    if closes > 0 and (shared_entities or shared_topics or shared_goals):
        types.append("temporal")

    if index is not None and a.topics and b.topics:
        shared_topics = set(a.topics) & set(b.topics)
        if shared_topics and a.outcome_polarity == b.outcome_polarity:
            family_sizes = index.family_sizes(a)
            for topic in shared_topics:
                if family_sizes.get(f"{topic}|{a.outcome_polarity}", 0) >= REPEATED_PATTERN_MIN_FAMILY:
                    types.append("repeated_pattern")
                    break
    # `gap`: both sides are stuck on their own obstacle and neither side's
    # resources answer the other's (KI-3 dropped the stricter "neither may hold
    # any resource at all" clause, which almost never held).
    if shared_goals and a.obstacles and b.obstacles:
        answers = _affinity(a.obstacles, b.resources) + _affinity(b.obstacles, a.resources)
        if answers == 0.0:
            types.append("gap")

    # Deterministic order, closed vocabulary only.
    return [t for t in RELATION_TYPES if t in set(types)]


def relation_score(
    a: MemoryItem,
    b: MemoryItem,
    types: Sequence[str],
    *,
    weights: Optional[RelationWeights] = None,
) -> Tuple[float, Dict[str, float]]:
    """Explainable RelationScore plus its components (design draft 5.2)."""
    w = weights or RelationWeights()
    semantic = jaccard(token_set(a.content), token_set(b.content))
    entity = jaccard(a.entities, b.entities)
    goal = jaccard(a.goal_ids, b.goal_ids)
    skill = jaccard(set(a.skill_ids) | set(a.method_ids), set(b.skill_ids) | set(b.method_ids))
    time_component = _temporal_closeness(a, b)
    pr = _problem_resource_overlap(a, b)
    analogy = 0.5 if (set(a.topics) & set(b.topics) and set(a.skill_ids) != set(b.skill_ids)) else 0.0
    contradiction = 1.0 if "contradiction" in types else 0.0
    redundancy = 1.0 if semantic >= NEAR_DUPLICATE_JACCARD else 0.0

    components = {
        "semantic": semantic,
        "entity": entity,
        "goal": goal,
        "skill": skill,
        "time": time_component,
        "problem_resource": pr,
        "analogy": analogy,
        "contradiction": contradiction,
        "redundancy": redundancy,
    }
    score = (
        w.semantic * semantic
        + w.entity * entity
        + w.goal * goal
        + w.skill * skill
        + w.time * time_component
        + w.problem_resource * pr
        + w.analogy * analogy
        + w.contradiction * contradiction
        - w.redundancy * redundancy
    )
    return max(0.0, min(1.0, score)), {k: round(v, 4) for k, v in components.items()}


def build_relations(
    memories: Sequence[MemoryItem],
    *,
    week: str,
    rng: Optional[random.Random] = None,
    weights: Optional[RelationWeights] = None,
    max_relations: int = DEFAULT_MAX_RELATIONS_PER_WEEK,
    min_score: float = 0.15,
) -> List[MemoryRelation]:
    """Assert relations for one week, capped and de-duplicated.

    Returns at most `max_relations` relations, highest score first, with ties
    broken by ids so the result does not depend on iteration order.
    """
    index = InvertedIndex(memories)
    seen: Set[str] = set()
    relations: List[MemoryRelation] = []

    for seed in memories:
        for other in index.candidates(seed, rng=rng):
            types = relation_types_between(seed, other, index=index)
            if not types:
                continue
            score, components = relation_score(seed, other, types, weights=weights)
            if score < min_score:
                continue
            relation = MemoryRelation(
                source_memory_id=seed.memory_id,
                target_memory_id=other.memory_id,
                relationship_types=types,
                score=score,
                method="rule",
                shared_focus=", ".join(
                    sorted(
                        (_overlap(seed.topics, other.topics))
                        or (_overlap(seed.entities, other.entities))
                        or (_overlap(seed.goal_ids, other.goal_ids))
                    )
                ),
                explanation=_explain(seed, other, types, components),
                week=week,
            )
            if relation.key in seen:
                continue
            seen.add(relation.key)
            relations.append(relation)

    relations.sort(key=lambda r: (-r.score, r.source_memory_id, r.target_memory_id))
    return relations[:max_relations]


def _explain(
    a: MemoryItem, b: MemoryItem, types: Sequence[str], components: Dict[str, float]
) -> str:
    """Human-readable justification, kept short enough to store per event."""
    parts = []
    if "same_entity" in types:
        parts.append(f"shared entities: {', '.join(sorted(_overlap(a.entities, b.entities)))}")
    if "same_goal" in types:
        parts.append(f"shared goal: {', '.join(sorted(_overlap(a.goal_ids, b.goal_ids)))}")
    if "problem_resource" in types or "precondition" in types:
        parts.append(
            "obstacles/resources match: "
            + ", ".join(
                sorted(_overlap(a.obstacles, b.resources) | _overlap(b.obstacles, a.resources))
            )
        )
    if "contradiction" in types:
        parts.append(f"opposite outcomes ({a.outcome_polarity} vs {b.outcome_polarity})")
    if "repeated_pattern" in types:
        parts.append("same (topic, polarity) family repeats")
    if "gap" in types:
        parts.append("shared goal blocked on both sides with no resource")
    if not parts:
        parts.append(f"score components {components}")
    return "; ".join(parts)
