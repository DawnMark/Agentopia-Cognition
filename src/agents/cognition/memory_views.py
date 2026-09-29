"""Materialized memory views, rebuilt from the memory event streams (draft §8).

Three derived files per persona:

    cognition/views/memories.json          memories + strength/tier/status
    cognition/views/memory_graph.json      relation nodes and edges
    cognition/views/retrieval_shadow.json  weekly old-vs-new retrieval comparison

The fold is pure — no clock, no randomness, no LLM — and authoritative numbers
(strength, tier) come from the recorded `MEMORY_STRENGTH_UPDATED` events rather
than being recomputed here, so rebuilding a view cannot disagree with the run
that produced it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.agents.data_manager import DataManager

MEMORIES_VIEW_FILENAME = "memories.json"
MEMORY_GRAPH_VIEW_FILENAME = "memory_graph.json"
RETRIEVAL_SHADOW_VIEW_FILENAME = "retrieval_shadow.json"

# Fields copied straight from a MEMORY_CREATED payload onto the view entry.
_MEMORY_PAYLOAD_FIELDS = (
    "kind",
    "content",
    "persona",
    "semantic_key",
    "entities",
    "topics",
    "goal_ids",
    "skill_ids",
    "method_ids",
    "confidence",
    "salience",
    "outcome_polarity",
    "resources",
    "obstacles",
    "emotion",
    "protected",
    "dropped_entities",
    "origin",
    "provenance",
)


def _ensure_memory(memories: Dict[str, Dict[str, Any]], memory_id: str) -> Dict[str, Any]:
    if memory_id not in memories:
        memories[memory_id] = {
            "memory_id": memory_id,
            "kind": "episodic",
            "content": "",
            "persona": "",
            "semantic_key": "",
            "entities": [],
            "topics": [],
            "goal_ids": [],
            "skill_ids": [],
            "method_ids": [],
            "source_event_ids": [],
            "confidence": 0.0,
            "salience": 0.0,
            "strength": 0.0,
            "last_recalled_at": "",
            "successful_recall_count": 0,
            "status": "active",
            "protected": False,
            "created_at": "",
            "outcome_polarity": "neutral",
            "resources": [],
            "obstacles": [],
            "emotion": 0.0,
            "tier": "warm",
            "read_count": 0,
            "dropped_entities": [],
            "origin": "",
            # Where a lesson came from, e.g. "scratchpad:general.jsonl#ev-123"
            # (phase 4.5). Empty for extracted memories.
            "provenance": "",
            # view-only bookkeeping
            "reinforcements": 0,
            "merged_from": [],
            "conflicts_with": [],
            "recalls": [],
            "strength_factors": {},
            # Phase 4 / KI-10: how often this memory actually informed an adopted
            # method (the strength formula already has this factor).
            "used_in_plan_count": 0,
            "used_in_plan_by": [],
        }
    return memories[memory_id]


def materialize_memories(
    events: Iterable[Dict[str, Any]], *, persona: str = ""
) -> Dict[str, Any]:
    """Fold `memory_events.jsonl` into the memory view."""
    memories: Dict[str, Dict[str, Any]] = {}
    by_kind: Dict[str, int] = {}
    by_tier: Dict[str, int] = {}
    by_status: Dict[str, int] = {}
    by_type: Dict[str, int] = {}
    count = 0
    last_time = ""

    for event in events:
        count += 1
        event_type = str(event.get("type") or "")
        by_type[event_type] = by_type.get(event_type, 0) + 1
        last_time = str(event.get("time") or last_time)
        memory_id = str(event.get("memory_id") or "")
        if event_type == "MEMORY_RETRIEVAL_COMPARED" or not memory_id:
            continue

        memory = _ensure_memory(memories, memory_id)

        if event_type == "MEMORY_CREATED":
            for field in _MEMORY_PAYLOAD_FIELDS:
                if field in event:
                    memory[field] = event[field]
            memory["created_at"] = str(event.get("created_at") or event.get("time") or "")
            memory["strength"] = float(event.get("strength", memory["salience"]))
            memory["tier"] = str(event.get("tier", memory.get("tier", "warm")))
            memory["source_event_ids"] = list(event.get("source_event_ids") or [])

        elif event_type == "MEMORY_REINFORCED":
            memory["reinforcements"] = int(memory.get("reinforcements", 0)) + 1
            for source in event.get("source_event_ids") or []:
                if source not in memory["source_event_ids"]:
                    memory["source_event_ids"].append(source)

        elif event_type == "MEMORY_MERGED":
            memory["merged_from"] = list(
                dict.fromkeys(
                    list(memory.get("merged_from") or []) + list(event.get("merged_from") or [])
                )
            )
            for source in event.get("source_event_ids") or []:
                if source not in memory["source_event_ids"]:
                    memory["source_event_ids"].append(source)
            if event.get("content") and not memory.get("content"):
                memory["content"] = event["content"]

        elif event_type == "MEMORY_CONTRADICTED":
            other = str(event.get("conflicts_with") or "")
            if other and other not in memory["conflicts_with"]:
                memory["conflicts_with"].append(other)
            memory["status"] = "contradicted"

        elif event_type == "MEMORY_STRENGTH_UPDATED":
            if "strength" in event:
                memory["strength"] = float(event["strength"])
            if "tier" in event:
                memory["tier"] = str(event["tier"])
            if event.get("factors"):
                memory["strength_factors"] = dict(event["factors"])

        elif event_type == "MEMORY_ARCHIVED":
            memory["status"] = "archived"
            memory["tier"] = "archived"

        elif event_type == "MEMORY_USED_IN_PLAN":
            memory["used_in_plan_count"] = int(memory.get("used_in_plan_count", 0)) + 1
            used_by = str(event.get("method_id") or "")
            if used_by and used_by not in memory["used_in_plan_by"]:
                memory["used_in_plan_by"].append(used_by)

        elif event_type == "MEMORY_RECALLED":
            memory["read_count"] = int(memory.get("read_count", 0)) + 1
            memory["last_recalled_at"] = str(event.get("time") or "")
            memory["recalls"].append(
                {
                    "week": event.get("week"),
                    "rank": event.get("rank"),
                    "score": event.get("score"),
                    "used": bool(event.get("used")),
                }
            )

    for memory in memories.values():
        by_kind[memory["kind"]] = by_kind.get(memory["kind"], 0) + 1
        by_tier[memory["tier"]] = by_tier.get(memory["tier"], 0) + 1
        by_status[memory["status"]] = by_status.get(memory["status"], 0) + 1

    return {
        "persona": persona,
        "event_count": count,
        "last_event_time": last_time,
        "memories": dict(sorted(memories.items())),
        "stats": {
            "memories": len(memories),
            "by_kind": dict(sorted(by_kind.items())),
            "by_tier": dict(sorted(by_tier.items())),
            "by_status": dict(sorted(by_status.items())),
            "by_event_type": dict(sorted(by_type.items())),
        },
    }


def materialize_relations(
    events: Iterable[Dict[str, Any]],
    *,
    memories: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Fold `memory_relation_events.jsonl` into the graph view."""
    edges: List[Dict[str, Any]] = []
    by_type: Dict[str, int] = {}
    nodes: Dict[str, Dict[str, Any]] = {}
    for memory_id, memory in (memories or {}).items():
        nodes[memory_id] = {
            "memory_id": memory_id,
            "kind": memory.get("kind"),
            "tier": memory.get("tier"),
            "topics": memory.get("topics"),
            "entities": memory.get("entities"),
        }

    for event in events:
        if str(event.get("type")) != "RELATION_ASSERTED":
            continue
        types = list(event.get("relationship_types") or [])
        for relation_type in types:
            by_type[relation_type] = by_type.get(relation_type, 0) + 1
        edges.append(
            {
                "source": event.get("source_memory_id"),
                "target": event.get("target_memory_id"),
                "types": types,
                "score": event.get("score"),
                "method": event.get("method", "rule"),
                "shared_focus": event.get("shared_focus", ""),
                "explanation": event.get("explanation", ""),
                "week": event.get("week", ""),
            }
        )

    edges.sort(
        key=lambda e: (-float(e.get("score") or 0.0), str(e["source"]), str(e["target"]))
    )
    return {
        "nodes": dict(sorted(nodes.items())),
        "edges": edges,
        "stats": {
            "nodes": len(nodes),
            "edges": len(edges),
            "by_type": dict(sorted(by_type.items())),
        },
    }


def materialize_retrieval_shadow(events: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Fold the weekly retrieval comparisons (design draft §7)."""
    weeks: List[Dict[str, Any]] = []
    recalls: Dict[str, List[Dict[str, Any]]] = {}
    for event in events:
        event_type = str(event.get("type") or "")
        week = str(event.get("week") or "")
        if event_type == "MEMORY_RETRIEVAL_COMPARED":
            weeks.append(
                {
                    "week": week,
                    "legacy": event.get("legacy") or {},
                    "proposed": event.get("proposed") or {},
                    "overlap": event.get("overlap") or {},
                    "cold_reactivated": event.get("cold_reactivated", 0),
                }
            )
        elif event_type == "MEMORY_RECALLED":
            recalls.setdefault(week, []).append(
                {
                    "memory_id": event.get("memory_id"),
                    "rank": event.get("rank"),
                    "score": event.get("score"),
                    "tier": event.get("tier"),
                }
            )
    weeks.sort(key=lambda w: str(w["week"]))
    for entry in weeks:
        entry["recalled"] = sorted(
            recalls.get(str(entry["week"]), []), key=lambda r: (r.get("rank") or 0)
        )
    return {"weeks": weeks, "stats": {"weeks": len(weeks)}}


def build_memory_views(dm: "DataManager") -> Dict[str, Dict[str, Any]]:
    """Build all three phase 2 views from the event streams."""
    from src.agents.cognition.memory_store import MemoryEventStore, RelationEventStore

    memory_events = MemoryEventStore(dm).events()
    relation_events = RelationEventStore(dm).events()
    memories = materialize_memories(memory_events, persona=dm.char)
    return {
        "memories": memories,
        "graph": materialize_relations(relation_events, memories=memories["memories"]),
        "retrieval": materialize_retrieval_shadow(memory_events),
    }


def views_dir(dm: "DataManager") -> Path:
    return dm.root / "cognition" / "views"


def write_memory_views(
    dm: "DataManager", views: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, Path]:
    """Write the three views; returns `{name: path}`."""
    views = views or build_memory_views(dm)
    target_dir = views_dir(dm)
    target_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    for key, filename in (
        ("memories", MEMORIES_VIEW_FILENAME),
        ("graph", MEMORY_GRAPH_VIEW_FILENAME),
        ("retrieval", RETRIEVAL_SHADOW_VIEW_FILENAME),
    ):
        path = target_dir / filename
        path.write_text(
            json.dumps(views[key], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        paths[key] = path
    return paths


_VIEW_ONLY_KEYS = (
    "reinforcements",
    "used_in_plan_count",
    "used_in_plan_by",
    "merged_from",
    "conflicts_with",
    "recalls",
    "strength_factors",
)


def load_memories(dm: "DataManager") -> List[Any]:
    """Memory items from the view, as objects (for the relation graph)."""
    from src.agents.cognition.memory_models import MemoryItem

    view = build_memory_views(dm)["memories"]
    out: List[MemoryItem] = []
    for entry in view["memories"].values():
        payload = {k: v for k, v in entry.items() if k not in _VIEW_ONLY_KEYS}
        try:
            out.append(MemoryItem.from_dict(payload))
        except (KeyError, ValueError):
            continue
    return out
