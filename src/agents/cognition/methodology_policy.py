"""Contextual method selection, in shadow mode (design doc 7.2 / 7.3).

Phase 1 records **what the policy would have chosen**; it never changes what the
character actually does. The policy itself is deliberately simple and fully
deterministic given its inputs, so a replay produces the same selections:

    score = w_value * global_value
          + w_context * context_value(context_key)      (when that context has evidence)
          + w_evidence * confidence
          + status_bonus                                (validated > tested > candidate)
          - contraindication_penalty

Exploration: with a small probability the policy picks a `proposed`/`learned`
candidate instead of the current best. The design document lets persona shape the
exploration rate (creativity/curiosity); that only affects which *shadow*
selection is recorded, so it stays safe in phase 1.

Selection is at most one primary plus at most one supporting methodology (design
doc 7.3: more than that makes attribution impossible).
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.agents.cognition.models import ContextSignature, Methodology

# Scoring weights.
W_VALUE = 1.0
W_CONTEXT = 0.8
W_EVIDENCE = 0.3

STATUS_BONUS = {
    "proposed": 0.0,
    "learned": 0.05,
    "tested": 0.1,
    "validated": 0.2,
    "specialized": 0.15,
}

# Candidate statuses that exploration may pick.
EXPLORABLE_STATUSES = ("proposed", "learned")
CONTRADICTION_PENALTY = 0.4

BASE_EXPLORATION_RATE = 0.15


def exploration_rate_for(
    *,
    creativity: Optional[float] = None,
    curiosity: Optional[float] = None,
    base: float = BASE_EXPLORATION_RATE,
) -> float:
    """Persona-shaped exploration willingness (0.05 .. 0.5), shadow only."""
    traits = [t for t in (creativity, curiosity) if t is not None]
    if not traits:
        return base
    mean = sum(float(t) for t in traits) / len(traits)  # 0-100 scale
    scaled = 0.5 + (mean - 50.0) / 100.0  # 0.0 .. 1.0
    return round(max(0.05, min(0.5, base * (0.5 + scaled))), 4)


def deterministic_rng(*parts: Any) -> random.Random:
    """RNG seeded by content, so replay reproduces the same exploration draw."""
    payload = "|".join(str(p) for p in parts)
    seed = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return random.Random(int(seed, 16))


def context_signature_from_activity(
    record: Dict[str, Any], *, skill_id: str
) -> ContextSignature:
    """Build the problem signature of one activity from its ledger record.

    Tags come from objective record fields only (activity type, whether skills
    were gained, money moved), never from free text, so the controlled
    vocabulary cannot drift.
    """
    tags: List[str] = []
    activity_type = str(record.get("type") or "")
    if activity_type:
        tags.append(activity_type)

    outcome = record.get("outcome") or {}
    if not isinstance(outcome, dict):
        outcome = {}
    gains = outcome.get("delta_skills") or {}
    if gains:
        tags.append("learning")
    if isinstance(gains, dict) and len(gains) > 1:
        tags.append("complex_structure")
    if int(outcome.get("delta_money") or 0) != 0:
        tags.append("work" if int(outcome.get("delta_money") or 0) > 0 else "consumption")
    negative_social = sum(
        -int(v)
        for k, v in (outcome.get("delta_fulfillment") or {}).items()
        if k in ("social", "esteem") and int(v) < 0
    )
    if negative_social > 0:
        tags.append("high_social_risk")
    if int(record.get("turns") or 0) > 8:
        tags.append("long_form")

    return ContextSignature(
        skill_ids=[skill_id] if skill_id else [],
        context_tags=tags,
        goal=str(record.get("content") or "")[:120],
        constraints={},
    )


def _contraindicated(method: Methodology, signature: ContextSignature) -> bool:
    tags = set(signature.context_tags)
    return bool(tags & set(method.contraindications))


@dataclass
class ScoredMethod:
    method: Methodology
    score: float
    context_key: str
    context_value: Optional[float]
    contraindicated: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_id": self.method.method_id,
            "skill_id": self.method.skill_id,
            "status": self.method.status,
            "score": round(self.score, 4),
            "global_value": self.method.global_value,
            "context_key": self.context_key,
            "context_value": self.context_value,
            "contraindicated": self.contraindicated,
        }


@dataclass
class Selection:
    """What the policy would have done for one activity."""

    primary: Optional[ScoredMethod] = None
    supporting: Optional[ScoredMethod] = None
    explored: bool = False
    context_key: str = "default"
    candidates: List[ScoredMethod] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return self.primary is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_method_id": self.primary.method.method_id if self.primary else None,
            "supporting_method_id": (
                self.supporting.method.method_id if self.supporting else None
            ),
            "explored": self.explored,
            "context_key": self.context_key,
            "candidates": [c.to_dict() for c in self.candidates[:5]],
            "n_candidates": len(self.candidates),
        }


def score_methods(
    methods: Iterable[Methodology],
    signature: ContextSignature,
) -> List[ScoredMethod]:
    """Score every candidate; deterministic order (score desc, method_id asc)."""
    context_key = signature.context_key()
    scored: List[ScoredMethod] = []
    for method in methods:
        context_value = method.context_value(context_key)
        score = W_VALUE * float(method.global_value)
        if context_value is not None:
            score += W_CONTEXT * context_value
        score += W_EVIDENCE * float(method.confidence)
        score += STATUS_BONUS.get(method.status, 0.0)

        contraindicated = _contraindicated(method, signature)
        if contraindicated:
            score -= CONTRADICTION_PENALTY

        scored.append(
            ScoredMethod(
                method=method,
                score=score,
                context_key=context_key,
                context_value=context_value,
                contraindicated=contraindicated,
            )
        )
    scored.sort(key=lambda s: (-s.score, s.method.method_id))
    return scored


def select_methods(
    methods: Sequence[Methodology],
    signature: ContextSignature,
    *,
    exploration_rate: float = BASE_EXPLORATION_RATE,
    rng: Optional[random.Random] = None,
    allow_supporting: bool = True,
) -> Selection:
    """Pick at most one primary (plus one supporting) methodology."""
    context_key = signature.context_key()
    scored = score_methods(methods, signature)
    if not scored:
        return Selection(context_key=context_key, candidates=[])

    rng = rng or deterministic_rng("selection", context_key)
    explored = False
    viable = [s for s in scored if not s.contraindicated]

    primary = viable[0] if viable else scored[0]
    explorable = [s for s in viable if s.method.status in EXPLORABLE_STATUSES]
    if explorable and rng.random() < exploration_rate:
        # KI-9: exploration must actually explore. Picking `explorable[0]` meant
        # the highest-scoring untested method was "explored" over and over while
        # the rest never received any evidence at all; a draw from the whole
        # explorable set (still seeded per activity, so replay stays exact) lets
        # a character's own history decide which methods get tried, without any
        # artificial quota.
        primary = explorable[rng.randrange(len(explorable))]
        explored = True

    supporting: Optional[ScoredMethod] = None
    if allow_supporting:
        for candidate in viable:
            if candidate.method.method_id == primary.method.method_id:
                continue
            # A supporting method must come from a different skill, otherwise it
            # is just a near-duplicate of the primary.
            if candidate.method.skill_id != primary.method.skill_id:
                supporting = candidate
                break

    return Selection(
        primary=primary,
        supporting=supporting,
        explored=explored,
        context_key=context_key,
        candidates=scored,
    )
