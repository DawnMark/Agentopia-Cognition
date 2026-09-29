"""Data contracts for the capability system (design doc 5.3 / 5.4 / 5.5 / 7.2).

These are the shapes written to `persona/<name>/cognition/capability_events.jsonl`
and folded into the materialized views. Two rules from the design document are
enforced here rather than by convention:

1. **A methodology is not a fact.** A methodology starts as `proposed` with
   `global_value = 0.0` and `confidence = 0.0`; only observed practice outcomes
   may move value (see reward_model.py).
2. **Value and confidence are separate.** `global_value` is what the practice
   results say; `confidence` is how much evidence backs it. One success gives a
   high value but still low confidence.

The controlled vocabularies below are deliberately closed sets: the design
document requires controlled tags so the space cannot grow without bound.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# --- controlled vocabularies -------------------------------------------------

# Lifecycle (design doc 5.4)
METHOD_STATUSES = (
    "proposed",
    "learned",
    "tested",
    "validated",
    "specialized",
    "deprecated",
    "archived",
)

# How a methodology came into existence (design doc 7.1)
METHOD_SOURCE_TYPES = (
    "practice_reflection",  # extracted from a finished week / activity outcome
    "direct_learning",  # course, reading, someone teaching
    "idea_conversion",  # phase 3
    "refinement",  # new version of an existing method
    "system_seed",  # bootstrap from existing skill names
)

# Capability events (design doc 5.5)
CAPABILITY_EVENT_TYPES = (
    "METHOD_PROPOSED",
    # Phase 4: a method was offered to the character in a prompt. Being offered
    # is not reinforcement (design doc invariant #6), so it carries no value.
    "METHOD_HINTED",
    # Phase 5: the menu was offered and the character did not take this method
    # up. A decline moves no value either (same invariant), but it is the half
    # of the offering decision phase 4 could not observe at all: measured on run
    # 09261617, 26 of 27 person-weeks adopted something, so "offered and
    # ignored" was invisible and the offer policy had no negative signal.
    "METHOD_DECLINED",
    "METHOD_LEARNED",
    "METHOD_REFINED",
    "METHOD_SELECTED",
    "METHOD_APPLIED",
    "METHOD_OUTCOME_OBSERVED",
    "METHOD_VALUE_UPDATED",
    "METHOD_SUPERSEDED",
    "METHOD_ARCHIVED",
    # Phase 6: the world layer capped a skill gain because the character's
    # *practised* capability on that skill is already high (see
    # `capability_gain.py`). Recorded so "the rule fired" is evidence, not an
    # inference from the numbers.
    "CAPABILITY_GAIN_CAPPED",
)

# Context tags the extractor/selector may use (design doc 7.2: controlled list).
CONTEXT_TAGS = (
    "long_form",
    "short_task",
    "complex_structure",
    "high_complexity",
    "routine",
    "novel",
    "solo",
    "joint",
    "public",
    "physical",
    "social",
    "creative",
    "analytical",
    "learning",
    "work",
    "consumption",
    "sufficient_time",
    "time_pressure",
    "low_money",
    "high_social_risk",
)


def _known_tags(values: List[str]) -> List[str]:
    """Keep only controlled tags, order-preserving and de-duplicated."""
    seen: Dict[str, None] = {}
    for v in values:
        tag = str(v).strip()
        if tag and tag in CONTEXT_TAGS:
            seen.setdefault(tag, None)
    return list(seen)


def _clean_str_list(values: Any, *, limit: int = 12) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    out: List[str] = []
    for v in values:
        s = str(v).strip()
        if s and s not in out:
            out.append(s)
        if len(out) >= limit:
            break
    return out


@dataclass
class ContextSignature:
    """The problem a methodology is supposed to solve (design doc 7.2)."""

    skill_ids: List[str] = field(default_factory=list)
    context_tags: List[str] = field(default_factory=list)
    goal: str = ""
    constraints: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.skill_ids = _clean_str_list(self.skill_ids)
        self.context_tags = _known_tags(self.context_tags)
        self.goal = str(self.goal or "").strip()
        cleaned: Dict[str, float] = {}
        for key in ("time_pressure", "money_limit", "social_risk"):
            if key in (self.constraints or {}):
                try:
                    cleaned[key] = round(float(self.constraints[key]), 3)
                except (TypeError, ValueError):
                    continue
        self.constraints = cleaned

    def context_key(self) -> str:
        """Stable key for per-context value bookkeeping.

        Tags are sorted so two signatures built in a different order share one
        bucket; an empty signature is its own bucket ("default").
        """
        if not self.context_tags:
            return "default"
        return "+".join(sorted(self.context_tags))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_ids": list(self.skill_ids),
            "context_tags": list(self.context_tags),
            "goal": self.goal,
            "constraints": dict(self.constraints),
        }

    @staticmethod
    def from_dict(d: Optional[Dict[str, Any]]) -> "ContextSignature":
        d = d or {}
        return ContextSignature(
            skill_ids=list(d.get("skill_ids") or []),
            context_tags=list(d.get("context_tags") or []),
            goal=str(d.get("goal") or ""),
            constraints=dict(d.get("constraints") or {}),
        )


@dataclass
class Methodology:
    """A candidate or validated way of solving a class of problems (doc 5.4)."""

    method_id: str
    skill_id: str
    version: int = 1
    title: str = ""
    description: str = ""
    status: str = "proposed"
    source_type: str = "practice_reflection"
    parent_method_id: Optional[str] = None
    applicable_contexts: List[str] = field(default_factory=list)
    contraindications: List[str] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)
    checks: List[str] = field(default_factory=list)
    failure_modes: List[str] = field(default_factory=list)
    # Evidence-backed numbers. They start at zero: a proposal is not evidence.
    global_value: float = 0.0
    confidence: float = 0.0
    practice_count: int = 0
    success_count: int = 0
    context_values: Dict[str, Dict[str, float]] = field(default_factory=dict)
    source_event_ids: List[str] = field(default_factory=list)
    # Provenance kept for phase 4: an adopted method can name the memories that
    # produced it, which is what makes "used in plan" observable (KI-10).
    source_memory_ids: List[str] = field(default_factory=list)
    source_idea_id: Optional[str] = None
    # The motif of the idea this method came from. `lesson_application` means the
    # character wrote the underlying lesson itself, which earns it practice
    # priority in the phase 4 menu (phase 4.5 lesson bridge).
    source_motif: Optional[str] = None
    last_used: str = ""

    def __post_init__(self) -> None:
        if self.status not in METHOD_STATUSES:
            raise ValueError(f"unknown methodology status: {self.status!r}")
        if self.source_type not in METHOD_SOURCE_TYPES:
            raise ValueError(f"unknown methodology source_type: {self.source_type!r}")
        self.applicable_contexts = _known_tags(self.applicable_contexts)
        self.contraindications = _clean_str_list(self.contraindications)
        self.steps = _clean_str_list(self.steps)
        self.checks = _clean_str_list(self.checks)
        self.failure_modes = _clean_str_list(self.failure_modes)
        self.global_value = round(float(self.global_value), 4)
        self.confidence = round(float(self.confidence), 4)

    # -- derived helpers ---------------------------------------------------
    @property
    def is_evidence_backed(self) -> bool:
        return self.practice_count > 0

    def context_value(self, context_key: str) -> Optional[float]:
        entry = self.context_values.get(context_key)
        return None if entry is None else float(entry.get("value", 0.0))

    def context_count(self, context_key: str) -> int:
        entry = self.context_values.get(context_key)
        return 0 if entry is None else int(entry.get("count", 0))

    # -- serialization -----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_id": self.method_id,
            "skill_id": self.skill_id,
            "version": int(self.version),
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "source_type": self.source_type,
            "parent_method_id": self.parent_method_id,
            "applicable_contexts": list(self.applicable_contexts),
            "contraindications": list(self.contraindications),
            "steps": list(self.steps),
            "checks": list(self.checks),
            "failure_modes": list(self.failure_modes),
            "global_value": self.global_value,
            "confidence": self.confidence,
            "practice_count": int(self.practice_count),
            "success_count": int(self.success_count),
            "context_values": {
                k: {"value": round(float(v.get("value", 0.0)), 4),
                    "count": int(v.get("count", 0))}
                for k, v in self.context_values.items()
            },
            "source_event_ids": list(self.source_event_ids),
            "source_memory_ids": list(self.source_memory_ids),
            "source_idea_id": self.source_idea_id,
            "source_motif": self.source_motif,
            "last_used": self.last_used,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Methodology":
        return Methodology(
            method_id=str(d["method_id"]),
            skill_id=str(d.get("skill_id", "")),
            version=int(d.get("version", 1)),
            title=str(d.get("title", "")),
            description=str(d.get("description", "")),
            status=str(d.get("status", "proposed")),
            source_type=str(d.get("source_type", "practice_reflection")),
            parent_method_id=d.get("parent_method_id"),
            applicable_contexts=list(d.get("applicable_contexts") or []),
            contraindications=list(d.get("contraindications") or []),
            steps=list(d.get("steps") or []),
            checks=list(d.get("checks") or []),
            failure_modes=list(d.get("failure_modes") or []),
            global_value=float(d.get("global_value", 0.0)),
            confidence=float(d.get("confidence", 0.0)),
            practice_count=int(d.get("practice_count", 0)),
            success_count=int(d.get("success_count", 0)),
            context_values=dict(d.get("context_values") or {}),
            source_event_ids=list(d.get("source_event_ids") or []),
            source_memory_ids=list(d.get("source_memory_ids") or []),
            source_idea_id=d.get("source_idea_id"),
            source_motif=d.get("source_motif"),
            last_used=str(d.get("last_used", "")),
        )


@dataclass
class SkillProjection:
    """Existing `state.skills` as seen by the capability system (doc 5.3).

    Phase 1 reads skills, never writes them: `state.jsonl` stays the source of
    truth until phase 6 migrates the projection.
    """

    skill_id: str
    name: str
    proficiency: float
    aliases: List[str] = field(default_factory=list)
    methodology_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "proficiency": round(float(self.proficiency), 3),
            "aliases": list(self.aliases),
            "methodology_ids": list(self.methodology_ids),
        }


def method_id_for(skill_id: str, title: str, *, version: int = 1) -> str:
    """Stable, human-readable method id.

    Derived from the skill and the title so re-extracting the same method in a
    later week maps to the same identity (and can be refined rather than
    duplicated), while staying independent of run ordering.
    """
    import hashlib

    digest = hashlib.sha1(f"{skill_id}\x1f{title.strip()}".encode("utf-8")).hexdigest()
    suffix = "" if version == 1 else f".v{version}"
    return f"method-{skill_id}-{digest[:10]}{suffix}"
