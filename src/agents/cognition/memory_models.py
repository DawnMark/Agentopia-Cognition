"""Memory data contracts (design doc 5.1 / 8.1 / 8.2, phase 2 design draft §3).

A memory is a *derived* record: it is created from ledger events, it must point
back at them, and it can be superseded, contradicted or archived — but never
rewritten. Three rules are enforced here rather than trusted to the extractor:

1. every memory cites at least one existing ledger event;
2. entities are restricted to the vocabulary the world already contains
   (personas, places, possessions); anything else is dropped and counted, so a
   model cannot invent people or places into memory;
3. skills must be skills the character actually had at the time.

The controlled vocabularies (kinds, topics, relation types, statuses) are closed
sets: the design document requires them so the space cannot grow without bound.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# --- controlled vocabularies (design draft 3.3) ------------------------------

MEMORY_KINDS = (
    "episodic",
    "semantic",
    "relationship",
    "goal",
    "belief",
    "lesson",
)

MEMORY_STATUSES = ("active", "superseded", "contradicted", "archived")

OUTCOME_POLARITIES = ("positive", "negative", "mixed", "neutral")

MEMORY_TOPICS = (
    "work",
    "study",
    "health",
    "exercise",
    "food",
    "money",
    "housing",
    "family",
    "friendship",
    "romance",
    "conflict",
    "cooperation",
    "planning",
    "creation",
    "performance",
    "learning",
    "routine",
    "travel",
    "consumption",
    "self_care",
    "mistake",
    "recovery",
    "opportunity",
    "risk",
)

# Design doc 8.1, all ten implemented as decidable rules in relation_graph.py.
RELATION_TYPES = (
    "same_entity",
    "same_goal",
    "causal",
    "temporal",
    "repeated_pattern",
    "contradiction",
    "problem_resource",
    "method_transfer",
    "precondition",
    "gap",
)

MEMORY_TIERS = ("hot", "warm", "cold", "archived")

CONTENT_MAX_LEN = 200
MAX_ENTITIES = 6
MAX_SOURCE_EVENTS = 6


def _clean_list(values: Any, *, allowed: Optional[Sequence[str]] = None, limit: int = 8) -> List[str]:
    """Normalize a string list, optionally restricted to a controlled set."""
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple, set)):
        return []
    out: List[str] = []
    allowed_set = set(allowed) if allowed else None
    for raw in values:
        text = str(raw).strip()
        if not text:
            continue
        if allowed_set is not None and text not in allowed_set:
            continue
        if text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _tokens(text: str) -> List[str]:
    """Cheap tokenizer good enough for overlap/dedup (CJK kept per character)."""
    lowered = str(text).lower()
    latin = re.findall(r"[a-z0-9]+", lowered)
    cjk = re.findall(r"[\u4e00-\u9fff]", lowered)
    return latin + cjk


def token_set(text: str) -> Set[str]:
    return set(_tokens(text))


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    set_a, set_b = set(a), set(b)
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def semantic_key(*, kind: str, topics: Sequence[str], entities: Sequence[str]) -> str:
    """Normalized identity of a memory, independent of wording and order."""
    payload = "|".join(
        [
            kind,
            ",".join(sorted(str(t) for t in topics)),
            ",".join(sorted(str(e) for e in entities)),
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def memory_id_for(*, persona: str, kind: str, key: str, disambiguator: str = "") -> str:
    """Content-addressed memory id.

    The semantic key names the *anchor* (kind + topics + entities), so two
    contradictory claims about the same anchor would collide. A deterministic
    `disambiguator` (polarity + content hash) keeps them as two memories that a
    contradiction relation can then link.
    """
    payload = f"{persona}\x1f{kind}\x1f{key}\x1f{disambiguator}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return f"mem-{kind}-{digest[:10]}"


@dataclass
class MemoryItem:
    """One structured memory (design doc 5.1)."""

    memory_id: str
    kind: str
    content: str
    persona: str = ""
    semantic_key: str = ""
    entities: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    goal_ids: List[str] = field(default_factory=list)
    skill_ids: List[str] = field(default_factory=list)
    method_ids: List[str] = field(default_factory=list)
    source_event_ids: List[str] = field(default_factory=list)
    confidence: float = 0.5
    salience: float = 0.5
    strength: float = 0.5
    last_recalled_at: str = ""
    successful_recall_count: int = 0
    status: str = "active"
    protected: bool = False
    created_at: str = ""
    # Structured extras used by the relation graph (design doc 8.2).
    outcome_polarity: str = "neutral"
    resources: List[str] = field(default_factory=list)
    obstacles: List[str] = field(default_factory=list)
    emotion: float = 0.0
    tier: str = "warm"
    read_count: int = 0
    dropped_entities: List[str] = field(default_factory=list)
    # Names the extractor used that could not be resolved to a world entity.
    # They are kept (KI-2) instead of being dropped: they are still what the
    # character said, and a later learned alias can promote them.
    soft_entities: List[str] = field(default_factory=list)
    # Where the memory came from. Empty/"extraction" for the weekly extraction;
    # "scratchpad_lesson" for a line the character wrote in its own notes
    # (phase 4.5 lesson bridge). Kept on the item so rules can tell "I concluded
    # this myself" from "this was extracted for me".
    origin: str = ""

    def __post_init__(self) -> None:
        if self.kind not in MEMORY_KINDS:
            raise ValueError(f"unknown memory kind: {self.kind!r}")
        if self.status not in MEMORY_STATUSES:
            raise ValueError(f"unknown memory status: {self.status!r}")
        if self.outcome_polarity not in OUTCOME_POLARITIES:
            raise ValueError(f"unknown outcome polarity: {self.outcome_polarity!r}")
        self.content = str(self.content).strip()[:CONTENT_MAX_LEN]
        self.topics = _clean_list(self.topics, allowed=MEMORY_TOPICS)
        self.entities = _clean_list(self.entities, limit=MAX_ENTITIES)
        self.goal_ids = _clean_list(self.goal_ids)
        self.skill_ids = _clean_list(self.skill_ids)
        self.method_ids = _clean_list(self.method_ids)
        self.source_event_ids = _clean_list(self.source_event_ids, limit=MAX_SOURCE_EVENTS)
        self.resources = _clean_list(self.resources)
        self.obstacles = _clean_list(self.obstacles)
        self.soft_entities = _clean_list(self.soft_entities, limit=MAX_ENTITIES)
        self.confidence = _clip01(self.confidence)
        self.salience = _clip01(self.salience)
        self.strength = _clip01(self.strength)
        self.emotion = max(-1.0, min(1.0, float(self.emotion)))
        if self.tier not in MEMORY_TIERS:
            self.tier = "warm"

    @property
    def is_evidence_backed(self) -> bool:
        return bool(self.source_event_ids)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "kind": self.kind,
            "content": self.content,
            "persona": self.persona,
            "semantic_key": self.semantic_key,
            "entities": list(self.entities),
            "topics": list(self.topics),
            "goal_ids": list(self.goal_ids),
            "skill_ids": list(self.skill_ids),
            "method_ids": list(self.method_ids),
            "source_event_ids": list(self.source_event_ids),
            "confidence": round(self.confidence, 4),
            "salience": round(self.salience, 4),
            "strength": round(self.strength, 4),
            "last_recalled_at": self.last_recalled_at,
            "successful_recall_count": int(self.successful_recall_count),
            "status": self.status,
            "protected": bool(self.protected),
            "created_at": self.created_at,
            "outcome_polarity": self.outcome_polarity,
            "resources": list(self.resources),
            "obstacles": list(self.obstacles),
            "emotion": round(self.emotion, 4),
            "tier": self.tier,
            "read_count": int(self.read_count),
            "dropped_entities": list(self.dropped_entities),
            "soft_entities": list(self.soft_entities),
            "origin": self.origin,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "MemoryItem":
        return MemoryItem(
            memory_id=str(d["memory_id"]),
            kind=str(d["kind"]),
            content=str(d.get("content", "")),
            persona=str(d.get("persona", "")),
            semantic_key=str(d.get("semantic_key", "")),
            entities=list(d.get("entities") or []),
            topics=list(d.get("topics") or []),
            goal_ids=list(d.get("goal_ids") or []),
            skill_ids=list(d.get("skill_ids") or []),
            method_ids=list(d.get("method_ids") or []),
            source_event_ids=list(d.get("source_event_ids") or []),
            confidence=float(d.get("confidence", 0.5)),
            salience=float(d.get("salience", 0.5)),
            strength=float(d.get("strength", 0.5)),
            last_recalled_at=str(d.get("last_recalled_at", "")),
            successful_recall_count=int(d.get("successful_recall_count", 0)),
            status=str(d.get("status", "active")),
            protected=bool(d.get("protected", False)),
            created_at=str(d.get("created_at", "")),
            outcome_polarity=str(d.get("outcome_polarity", "neutral")),
            resources=list(d.get("resources") or []),
            obstacles=list(d.get("obstacles") or []),
            emotion=float(d.get("emotion", 0.0)),
            tier=str(d.get("tier", "warm")),
            read_count=int(d.get("read_count", 0)),
            dropped_entities=list(d.get("dropped_entities") or []),
            soft_entities=list(d.get("soft_entities") or []),
            origin=str(d.get("origin", "")),
        )


def _clip01(value: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


@dataclass
class MemoryRelation:
    """One asserted relation between two memories (design draft 5.3)."""

    source_memory_id: str
    target_memory_id: str
    relationship_types: List[str]
    score: float
    method: str = "rule"  # "rule" | "llm"
    shared_focus: str = ""
    explanation: str = ""
    week: str = ""
    time: str = ""

    def __post_init__(self) -> None:
        self.relationship_types = [
            t for t in _clean_list(self.relationship_types) if t in RELATION_TYPES
        ]
        if not self.relationship_types:
            raise ValueError("a relation needs at least one known relationship type")
        if self.method not in ("rule", "llm"):
            raise ValueError(f"unknown relation method: {self.method!r}")

    @property
    def key(self) -> str:
        """Undirected identity: a relation is the same whichever way round."""
        a, b = sorted([self.source_memory_id, self.target_memory_id])
        types = ",".join(sorted(self.relationship_types))
        return f"RELATION_ASSERTED:{a}:{b}:{types}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "RELATION_ASSERTED",
            "source_memory_id": self.source_memory_id,
            "target_memory_id": self.target_memory_id,
            "relationship_types": list(self.relationship_types),
            "score": round(float(self.score), 4),
            "method": self.method,
            "shared_focus": self.shared_focus,
            "explanation": self.explanation,
            "week": self.week,
        }


@dataclass
class ValidationOutcome:
    """Result of validating one extracted memory against the ledger."""

    ok: bool
    reason: str = ""
    item: Optional[MemoryItem] = None


def shared_anchor(a: "MemoryItem", b: "MemoryItem") -> List[str]:
    """Which *concrete* anchors two memories share (KI-4).

    A contradiction means "the same thing turned out differently", so it needs
    an anchor: the same person/place/thing, the same goal, or one memory's
    obstacle being answered by the other's resource. Sharing a coarse topic
    (`learning`, `self_care`, ...) is *not* an anchor — that was the rule that
    turned a single negative lesson into a hub contradicting every positive
    memory in the same topic.
    """
    kinds: List[str] = []
    if set(a.entities) & set(b.entities):
        kinds.append("entity")
    if set(a.goal_ids) & set(b.goal_ids):
        kinds.append("goal")
    if _affinity(a.obstacles, b.resources) > 0 or _affinity(b.obstacles, a.resources) > 0:
        kinds.append("obstacle_resource")
    return kinds


def _affinity(needs: Iterable[str], supplies: Iterable[str]) -> float:
    """Lexical affinity between "what is missing" and "what is available".

    Exact equality was the phase-2 rule and it never fired on free text
    ("承重墙打孔" vs "金属膨胀螺丝"). This keeps equality as the strongest
    signal but also accepts containment and shared bigrams, which is what makes
    `problem_resource` reachable at all (KI-3).
    """
    best = 0.0
    need_list = [str(n).strip() for n in needs if str(n).strip()]
    supply_list = [str(s).strip() for s in supplies if str(s).strip()]
    for need in need_list:
        for supply in supply_list:
            if need == supply:
                best = max(best, 1.0)
                continue
            if len(need) >= 2 and len(supply) >= 2 and (need in supply or supply in need):
                best = max(best, 0.8)
                continue
            best = max(best, jaccard(token_set(need), token_set(supply)) * 0.6)
    return best


def validate_memory(
    item: MemoryItem,
    *,
    known_event_ids: Set[str],
    known_skills: Set[str],
    known_entities: Set[str],
) -> ValidationOutcome:
    """Enforce the three rules of design draft 3.1.

    Unknown entities are not fatal: they are removed from the memory and
    reported in `dropped_entities`, because dropping a whole memory for one
    hallucinated shop name would lose otherwise usable evidence.
    """
    if not item.content:
        return ValidationOutcome(False, "empty content")

    sources = [s for s in item.source_event_ids if s in known_event_ids]
    if not sources:
        return ValidationOutcome(False, "no known source event")
    item.source_event_ids = sources

    unknown = [e for e in item.entities if e not in known_entities]
    if unknown:
        item.dropped_entities = list(unknown)
        item.entities = [e for e in item.entities if e in known_entities]
        # A memory whose only anchor was invented is not trustworthy.
        item.confidence = round(item.confidence * 0.8, 4)

    unknown_skills = [s for s in item.skill_ids if s not in known_skills]
    if unknown_skills:
        item.skill_ids = [s for s in item.skill_ids if s in known_skills]

    return ValidationOutcome(True, "", item)
