"""Shadow retrieval comparison: old retrieval vs the memory-based one (draft §7).

Phase 2 changes nothing about the prompt. What it does is answer, with data,
whether the structured memory layer would retrieve better than what the agents
already get:

    legacy    the current context: scratchpad listing ordered by recency, plus
              the recent weekly-diary window;
    proposed  memories ranked by relevance x strength x tier.

Both sides are measured for item count and estimated tokens, the overlap is
recorded, and the memories that *would* have been offered are written as
`MEMORY_RECALLED` events with `used = false` — because nothing is injected, a
recall can never be mistaken for reinforcement (design doc invariant #6).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.agents.cognition.memory_models import MemoryItem, token_set
from src.agents.cognition.memory_store import MemoryEventStore
from src.utils import num_tokens_from_string

DEFAULT_TOP_K = 8
TIER_BONUS = {"hot": 1.2, "warm": 1.0, "cold": 0.9, "archived": 0.85}
W_RELEVANCE = 0.7
W_STRENGTH = 0.3
ARCHIVED_RELEVANCE_FLOOR = 0.5


@dataclass
class RetrievalConfig:
    top_k: int = DEFAULT_TOP_K

    @staticmethod
    def from_world_config(world_cfg: Dict[str, Any]) -> "RetrievalConfig":
        section = (world_cfg or {}).get("cognition") or {}
        try:
            top_k = int(section.get("memory_top_k", DEFAULT_TOP_K))
        except (TypeError, ValueError):
            top_k = DEFAULT_TOP_K
        return RetrievalConfig(top_k=max(1, top_k))


def _activity_query_tokens(dm) -> Tuple[set, str]:
    """Phase 2 stand-in for a real query: this week's activity text.

    A structured query needs structured goals, which arrive in phase 3; until
    then the honest query is "what happened this week".
    """
    path = dm.root / "activity.jsonl"
    if not path.exists():
        return set(), ""
    texts: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        texts.append(str(record.get("content") or ""))
        outcome = record.get("outcome") or {}
        if isinstance(outcome, dict):
            texts.append(str(outcome.get("outcome") or ""))
    joined = " ".join(texts)
    return token_set(joined), joined


def legacy_retrieval(dm, *, character_limit: int = 50, other_limit: int = 10) -> Dict[str, Any]:
    """What the current prompt already contains (design draft §7)."""
    try:
        listing = dm.list_scratchpads(character_limit=character_limit, other_limit=other_limit)
    except Exception:
        listing = ""
    summaries_text = ""
    try:
        rows = dm.read_weekly_summaries(n_weeks=2)
        summaries_text = "\n".join(str(r.get("content") or "") for r in rows)
    except Exception:
        pass

    text = f"{listing}\n{summaries_text}".strip()
    # Count the entries the listing exposes (one "- " bullet per scratchpad).
    n_items = sum(1 for line in listing.splitlines() if line.strip().startswith("-"))
    return {
        "n_items": n_items,
        "est_tokens": num_tokens_from_string(text) if text else 0,
        "chars": len(text),
    }


def score_memories(
    memories: Sequence[MemoryItem], *, query_tokens: set, top_k: int
) -> List[Tuple[MemoryItem, float, float]]:
    """Rank memories by relevance x strength x tier. Returns (item, score, relevance)."""
    scored: List[Tuple[MemoryItem, float, float]] = []
    for memory in memories:
        if memory.status == "superseded":
            continue
        relevance = 0.0
        if query_tokens:
            overlap = query_tokens & token_set(memory.content)
            relevance = min(1.0, len(overlap) / max(1, min(len(query_tokens), 20)))
            if memory.entities and set(memory.entities) & query_tokens:
                relevance = min(1.0, relevance + 0.2)
        if memory.tier == "archived" and relevance < ARCHIVED_RELEVANCE_FLOOR:
            continue
        score = (W_RELEVANCE * relevance + W_STRENGTH * float(memory.strength)) * TIER_BONUS.get(
            memory.tier, 1.0
        )
        scored.append((memory, round(score, 4), round(relevance, 4)))
    scored.sort(key=lambda item: (-item[1], item[0].memory_id))
    return scored[:top_k]


class MemoryRetriever:
    """Weekly shadow comparison for one character."""

    def __init__(self, *, dm, clock, config: Optional[RetrievalConfig] = None) -> None:
        self.dm = dm
        self.clock = clock
        self.config = config or RetrievalConfig()
        self.store = MemoryEventStore(dm)

    def compare_week(self, *, memories: Sequence[MemoryItem]) -> Dict[str, Any]:
        """Record one week's comparison; returns the snapshot that was stored."""
        week = self._week_key()
        query_tokens, query_text = _activity_query_tokens(self.dm)
        legacy = legacy_retrieval(self.dm)

        ranked = score_memories(
            memories, query_tokens=query_tokens, top_k=self.config.top_k
        )
        proposed_text = "\n".join(m.content for m, _, _ in ranked)
        proposed = {
            "n_items": len(ranked),
            "est_tokens": num_tokens_from_string(proposed_text) if proposed_text else 0,
            "query_chars": len(query_text),
            "top_k": self.config.top_k,
        }

        legacy_names = set()  # the legacy side exposes text, not memory ids
        overlap = {
            "shared": 0,  # no id-level overlap is observable in phase 2
            "legacy_only": legacy["n_items"],
            "proposed_only": len(ranked),
            "legacy_names_available": bool(legacy_names),
        }
        cold_reactivated = sum(1 for m, _, _ in ranked if m.tier in ("cold", "archived"))

        for rank, (memory, score, _relevance) in enumerate(ranked, start=1):
            self.store.recalled(
                memory_id=memory.memory_id,
                week=week,
                query_context="weekly_activities",
                rank=rank,
                score=score,
                tier=memory.tier,
            )

        self.store.retrieval_compared(
            week=week,
            legacy=legacy,
            proposed=proposed,
            overlap=overlap,
            extra={"cold_reactivated": cold_reactivated},
        )

        return {
            "week": week,
            "legacy": legacy,
            "proposed": proposed,
            "overlap": overlap,
            "cold_reactivated": cold_reactivated,
        }

    def _week_key(self) -> str:
        t = self.clock.get_time()
        return f"Y{t.year}-W{t.week:02d}"
