"""Idea contracts (design doc 5.2 / 8.7, phase 3).

An Idea is a **candidate hypothesis** assembled from memories, never a fact:

- it must cite 2-4 memories that exist and carry evidence (grounding);
- it starts as `candidate` with a low confidence, and only a later practice can
  move it to `tested` / `adopted` (phases 4-5);
- it can create a *candidate methodology* (`METHOD_PROPOSED`, value 0), which is
  the only way an idea may influence the capability system — an idea can never
  raise a value or a skill by itself (design doc invariant #4).

The five checks of design doc 8.7 that gate entry into the idea store are
implemented in `idea_engine.py`; this module owns the vocabulary and the score
arithmetic.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

IDEA_TYPES = (
    "hypothesis",
    "opportunity",
    "method_candidate",
    "goal_adjustment",
)

IDEA_STATUSES = (
    "candidate",
    "tested",
    "adopted",
    "rejected",
    "expired",
)

IDEA_EVENT_TYPES = (
    "IDEA_SKIPPED",
    "IDEA_CREATED",
    "IDEA_SCORED",
    "IDEA_REFINED",
    "IDEA_REJECTED",
    "IDEA_EXPIRED",
    "IDEA_TESTED",
    "IDEA_ADOPTED",
    "IDEA_CONVERTED",
)

# Relation motifs of design doc 8.6.
IDEA_MOTIFS = (
    "goal_obstacle_resource",
    "repeated_pattern",
    "contradiction",
    "cross_domain_analogy",
    "unused_resource",
    "causal_gap",
    # Phase 4.5: an idea built from a lesson the character wrote for itself
    # ("next time, do X"), paired with a goal or an experience of its own.
    "lesson_application",
)

# Penalty reasons of design doc 8.7.
PENALTY_REASONS = (
    "duplicates_existing_plan",
    "depends_on_missing_entity",
    "conflicts_with_world_state",
    "not_testable",
    "sources_low_confidence",
    "duplicates_existing_idea_or_method",
)

# A candidate must clear this to be stored.
MIN_IDEA_POTENTIAL = 0.05
# Product-of-factors scoring makes values small; the threshold is deliberately
# low and the components are what reviewers should read.
MAX_IDEAS_PER_WEEK = 2

# Confidence ceiling for a freshly generated idea: it is a hypothesis.
CANDIDATE_CONFIDENCE_CEILING = 0.5


def idea_id_for(*, persona: str, motif: str, source_memory_ids: List[str]) -> str:
    """Stable id: the same memory combination yields the same idea.

    That is what makes the novelty check meaningful across weeks — an idea
    rediscovered from the same evidence is the same idea, not a new one.
    """
    payload = "|".join([persona, motif, ",".join(sorted(source_memory_ids))])
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return f"idea-{motif[:12]}-{digest[:10]}"


def method_id_for_idea(idea_id: str, skill_id: str) -> str:
    """Stable methodology id for an idea that became a candidate method.

    Derived from the idea, so re-converting the same idea in a later week maps
    to the same methodology instead of minting a duplicate.
    """
    digest = hashlib.sha1(f"{idea_id}{skill_id}".encode("utf-8")).hexdigest()
    return f"method-idea-{digest[:10]}"


def _clip01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


@dataclass
class IdeaPotential:
    """The scored components of design doc 8.7.

    `potential` is the product of the positive factors; `penalties` are the
    named reasons that shrank it; both are stored so a reviewer can see *why* an
    idea was kept or dropped.
    """

    groundedness: float = 0.0
    complementarity: float = 0.0
    novelty: float = 0.0
    actionability: float = 0.0
    goal_relevance: float = 0.0
    persona_fit: float = 0.0
    evidence_confidence: float = 0.0
    penalty_factor: float = 1.0
    penalties: List[str] = field(default_factory=list)

    @property
    def raw_product(self) -> float:
        return (
            self.groundedness
            * self.complementarity
            * self.novelty
            * self.actionability
            * self.goal_relevance
            * self.persona_fit
            * self.evidence_confidence
        )

    @property
    def potential(self) -> float:
        return max(0.0, min(1.0, self.raw_product * self.penalty_factor))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "groundedness": round(self.groundedness, 4),
            "complementarity": round(self.complementarity, 4),
            "novelty": round(self.novelty, 4),
            "actionability": round(self.actionability, 4),
            "goal_relevance": round(self.goal_relevance, 4),
            "persona_fit": round(self.persona_fit, 4),
            "evidence_confidence": round(self.evidence_confidence, 4),
            "penalty_factor": round(self.penalty_factor, 4),
            "penalties": list(self.penalties),
            "raw_product": round(self.raw_product, 6),
            "potential": round(self.potential, 6),
        }


@dataclass
class Idea:
    """One candidate idea (design doc 5.2)."""

    idea_id: str
    content: str
    idea_type: str = "hypothesis"
    motif: str = ""
    source_memory_ids: List[str] = field(default_factory=list)
    relationship_types: List[str] = field(default_factory=list)
    related_goal_ids: List[str] = field(default_factory=list)
    related_skill_ids: List[str] = field(default_factory=list)
    candidate_method_id: Optional[str] = None
    novelty: float = 0.0
    feasibility: float = 0.0
    groundedness: float = 0.0
    confidence: float = 0.0
    status: str = "candidate"
    expires_at: str = ""
    created_at: str = ""
    # Phase 3 extras: how it would be tested, and the score breakdown.
    test_plan: str = ""
    potential: float = 0.0
    score_components: Dict[str, Any] = field(default_factory=dict)
    persona: str = ""

    def __post_init__(self) -> None:
        if self.idea_type not in IDEA_TYPES:
            raise ValueError(f"unknown idea type: {self.idea_type!r}")
        if self.status not in IDEA_STATUSES:
            raise ValueError(f"unknown idea status: {self.status!r}")
        if self.motif and self.motif not in IDEA_MOTIFS:
            raise ValueError(f"unknown idea motif: {self.motif!r}")
        self.content = str(self.content).strip()
        self.test_plan = str(self.test_plan).strip()
        self.confidence = _clip01(self.confidence)
        self.novelty = _clip01(self.novelty)
        self.feasibility = _clip01(self.feasibility)
        self.groundedness = _clip01(self.groundedness)
        # A hypothesis may not sound certain (design doc 8.7, check 5).
        if self.status == "candidate":
            self.confidence = min(self.confidence, CANDIDATE_CONFIDENCE_CEILING)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "idea_id": self.idea_id,
            "content": self.content,
            "idea_type": self.idea_type,
            "motif": self.motif,
            "source_memory_ids": list(self.source_memory_ids),
            "relationship_types": list(self.relationship_types),
            "related_goal_ids": list(self.related_goal_ids),
            "related_skill_ids": list(self.related_skill_ids),
            "candidate_method_id": self.candidate_method_id,
            "novelty": round(self.novelty, 4),
            "feasibility": round(self.feasibility, 4),
            "groundedness": round(self.groundedness, 4),
            "confidence": round(self.confidence, 4),
            "status": self.status,
            "expires_at": self.expires_at,
            "created_at": self.created_at,
            "test_plan": self.test_plan,
            "potential": round(self.potential, 6),
            "score_components": dict(self.score_components),
            "persona": self.persona,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Idea":
        return Idea(
            idea_id=str(d["idea_id"]),
            content=str(d.get("content", "")),
            idea_type=str(d.get("idea_type", "hypothesis")),
            motif=str(d.get("motif", "")),
            source_memory_ids=list(d.get("source_memory_ids") or []),
            relationship_types=list(d.get("relationship_types") or []),
            related_goal_ids=list(d.get("related_goal_ids") or []),
            related_skill_ids=list(d.get("related_skill_ids") or []),
            candidate_method_id=d.get("candidate_method_id"),
            novelty=float(d.get("novelty", 0.0)),
            feasibility=float(d.get("feasibility", 0.0)),
            groundedness=float(d.get("groundedness", 0.0)),
            confidence=float(d.get("confidence", 0.0)),
            status=str(d.get("status", "candidate")),
            expires_at=str(d.get("expires_at", "")),
            created_at=str(d.get("created_at", "")),
            test_plan=str(d.get("test_plan", "")),
            potential=float(d.get("potential", 0.0)),
            score_components=dict(d.get("score_components") or {}),
            persona=str(d.get("persona", "")),
        )


@dataclass
class IdeaCandidate:
    """A motif instance found in the relation graph, before phrasing.

    Candidates are produced by rules (deterministic); only the top few are sent
    to the language model for phrasing, which keeps the weekly cost at one call.
    """

    motif: str
    memory_ids: List[str]
    relationship_types: List[str] = field(default_factory=list)
    goal: str = ""
    obstacles: List[str] = field(default_factory=list)
    resources: List[str] = field(default_factory=list)
    shared_focus: str = ""
    score: float = 0.0
    hint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "motif": self.motif,
            "memory_ids": list(self.memory_ids),
            "relationship_types": list(self.relationship_types),
            "goal": self.goal,
            "obstacles": list(self.obstacles),
            "resources": list(self.resources),
            "shared_focus": self.shared_focus,
            "score": round(self.score, 4),
        }
