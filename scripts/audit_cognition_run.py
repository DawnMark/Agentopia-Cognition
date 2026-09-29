#!/usr/bin/env python3
"""Audit one Agentopia cognition run without calling an LLM.

The report is evidence for the phase 1-3 acceptance runs. It covers four
questions:

1. **contract**  — event identity, idempotency, source traceability, view
   rebuildability, ledger integrity, diary/generation integrity;
2. **effect**    — did the cognition layers actually produce memories,
   relations, ideas and methods, and how do the shadow numbers move;
3. **quality**   — are the ideas and methodologies usable (grounded, concrete,
   structured, non-duplicated) rather than a restatement of the week;
4. **growth**    — is there a memory -> idea -> method -> practice -> value
   chain that can plausibly help the character, and where does it break.

Everything here is deterministic and offline. Qualitative judgement still
needs a human read of the samples the report includes.

Usage::

    python scripts/audit_cognition_run.py --data-dir shanghai_apartment_09261237 \
        --output art/acceptance/audit_09261237.json --markdown art/acceptance/audit_09261237.md
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.cognition.idea_engine import (  # noqa: E402
    NOVELTY_REJECT_THRESHOLD,
    SOURCE_OVERLAP_DUPLICATE,
    TEST_PLAN_DUPLICATE_SIMILARITY,
)

# ---------------------------------------------------------------- helpers ---


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{no}: invalid JSON: {exc}") from exc
        if isinstance(value, dict):
            rows.append(value)
    return rows


def latest_state(persona_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(persona_dir / "state.jsonl")
    return dict((rows[-1].get("content") or {}) if rows else {})


def first_state(persona_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(persona_dir / "state.jsonl")
    return dict((rows[0].get("content") or {}) if rows else {})


def vitality_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Weekly-start vitality: median, share below 20, share pinned at 100.

    Design doc 12 lists this as the world-health reading that decides
    `vitality_recovery.daily_recovery` (decision 0.13.4: "suggest running a full
    year first"). It needs a long run, which is what a 2-year run provides.
    "Week start" is the first state row of each simulated week.
    """
    first_of_week: dict[str, float] = {}
    for row in rows:
        week = week_of(row.get("time")) or ""
        if not week:
            continue
        content = row.get("content") or {}
        try:
            value = float(content.get("vitality"))
        except (TypeError, ValueError):
            continue
        first_of_week.setdefault(week, value)
    values = [first_of_week[week] for week in sorted(first_of_week)]
    if not values:
        return {"weeks": 0}
    ordered = sorted(values)
    mid = len(ordered) // 2
    median = (
        ordered[mid] if len(ordered) % 2 else round((ordered[mid - 1] + ordered[mid]) / 2, 1)
    )
    return {
        "weeks": len(values),
        "median": median,
        "min": min(values),
        "max": max(values),
        "share_below_20": round(sum(1 for v in values if v < 20) / len(values), 4),
        "share_at_100": round(sum(1 for v in values if v >= 100) / len(values), 4),
        "series": values,
    }



def counts(rows: Iterable[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field) or "") for row in rows).items()))


def week_of(value: Any) -> str:
    match = re.search(r"Y\d+-W\d+", str(value or ""))
    return match.group(0) if match else ""


def round4(value: Any) -> float | None:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def mean(values: Iterable[float]) -> float | None:
    items = [float(v) for v in values]
    if not items:
        return None
    return round(sum(items) / len(items), 4)


def text_tokens(value: Any) -> set[str]:
    text = re.sub(r"\s+", "", str(value or "").lower())
    if len(text) < 2:
        return {text} if text else set()
    return {text[i : i + 2] for i in range(len(text) - 1)}


def jaccard(a: Any, b: Any) -> float:
    left, right = text_tokens(a), text_tokens(b)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def duplicate_pairs(
    items: list[tuple[str, str]], threshold: float = 0.6
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, (left_id, left) in enumerate(items):
        for right_id, right in items[i + 1 :]:
            score = jaccard(left, right)
            if score >= threshold:
                out.append(
                    {"left": left_id, "right": right_id, "similarity": round(score, 4)}
                )
    return out


def near_duplicate_clusters(
    items: list[tuple[str, str]], threshold: float = 0.35
) -> list[list[str]]:
    """Union-find clusters of lexically near-identical texts.

    The threshold is deliberately below the engine's own novelty gate (0.6):
    the point is to catch "same insight, reworded", which is exactly what slips
    past a 0.6 lexical gate.
    """
    parent: dict[str, str] = {item_id: item_id for item_id, _ in items}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, (left_id, left) in enumerate(items):
        for right_id, right in items[i + 1 :]:
            if jaccard(left, right) >= threshold:
                union(left_id, right_id)
    grouped: dict[str, list[str]] = defaultdict(list)
    for item_id, _ in items:
        grouped[find(item_id)].append(item_id)
    return [sorted(members) for members in grouped.values() if len(members) > 1]


ASCII_ID = re.compile(r"[A-Za-z][A-Za-z0-9_]{3,}")
CONCRETE = re.compile(r"\d|本周|下周|今天|明天|D[1-5]|每天|每周|一次|两次")


def ascii_id_leaks(texts: Iterable[Any]) -> list[str]:
    """Internal English ids (Riverside_Park) leaking into character-facing text."""
    found: set[str] = set()
    for text in texts:
        for token in ASCII_ID.findall(str(text or "")):
            if "_" in token or token.islower():
                found.add(token)
    return sorted(found)


# --------------------------------------------------------------- integrity ---


def generation_integrity(persona_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted((persona_dir / "generation").glob("**/*.jsonl")):
        rows.extend(read_jsonl(path))
    bad_final: list[str] = []
    for row in rows:
        outputs = row.get("outputs") or []
        if not outputs:
            bad_final.append(str(row.get("record_id") or row.get("time") or "unknown"))
            continue
        last = outputs[-1]
        if last.get("role") != "assistant" or not str(last.get("content") or "").strip():
            bad_final.append(str(row.get("record_id") or row.get("time") or "unknown"))
    return {
        "records": len(rows),
        "rejected": sum(bool(r.get("rejected")) for r in rows),
        "reject_reasons": dict(
            sorted(
                Counter(
                    str(r.get("reject_reason") or "")
                    for r in rows
                    if r.get("rejected")
                ).items()
            )
        ),
        "bad_final_answer_count": len(bad_final),
        "bad_final_answer_samples": bad_final[:10],
    }


def _world_aliases(run_dir: Path) -> Any:
    """The alias families that apply to a run, from the run dir or the world source.

    A run directory carries a copy of the world's data (`skill_aliases.json`
    included, from the run that started copying it). Older runs predate that copy,
    so fall back to the world source `data/<world>/`. Returning the families
    explicitly — instead of letting `build_projection` derive them from a
    stand-in object — is deliberate: that derivation reads
    `data/<dm.world>` relative to the process CWD, which for a synthetic or
    relocated run directory silently yields "no families", and a silently
    unfolded projection reads exactly like a projection with nothing to fold.
    """
    from src.agents.cognition.skills import ALIAS_FILENAME, SkillFamilies

    world = str(((read_json(run_dir / "config.json", {}) or {}).get("world") or {}).get("name") or "")
    candidates = [run_dir] + ([ROOT / "data" / world] if world else [])
    for candidate in candidates:
        if (candidate / ALIAS_FILENAME).exists():
            families = SkillFamilies.from_world_dir(candidate)
            if families.members:
                return families
    return SkillFamilies()


def ledger_integrity(run_dir: Path) -> dict[str, Any]:
    ids: set[str] = set()
    duplicates: list[str] = []
    missing_identity: list[str] = []
    invalid_jsonl: list[str] = []
    streams: dict[str, int] = {}
    total = 0
    for path in sorted(run_dir.glob("**/*.jsonl")):
        try:
            rows = read_jsonl(path)
        except ValueError as exc:
            invalid_jsonl.append(str(exc))
            continue
        rel = str(path.relative_to(run_dir)).replace("\\", "/")
        streams[rel] = len(rows)
        total += len(rows)
        for row in rows:
            event_id = row.get("ledger_event_id")
            if not event_id:
                # cognition views/other files may legitimately be plain data;
                # everything under the ledger roots must be stamped.
                if rel.startswith("persona/") or rel.startswith("god/") or rel.startswith("reward/"):
                    missing_identity.append(rel)
                continue
            event_id = str(event_id)
            if event_id in ids:
                duplicates.append(event_id)
            ids.add(event_id)
    return {
        "records": total,
        "unique_ids": len(ids),
        "duplicate_row_count": len(duplicates),
        "duplicate_id_count": len(set(duplicates)),
        "duplicate_ids": sorted(set(duplicates))[:20],
        "invalid_jsonl": invalid_jsonl,
        "missing_identity_streams": sorted(set(missing_identity)),
        "streams": dict(sorted(streams.items())),
    }


# ------------------------------------------------------------------ persona ---


def audit_persona(run_dir: Path, persona_dir: Path, ledger_ids: set[str]) -> dict[str, Any]:
    cognition = persona_dir / "cognition"
    views = cognition / "views"
    activity_rows = read_jsonl(persona_dir / "activity.jsonl")
    diary_rows = read_jsonl(persona_dir / "memory" / "weekly_diary.jsonl")
    scratch_rows = read_jsonl(persona_dir / "memory" / "scratchpad" / "general.jsonl")

    state_first, state_last = first_state(persona_dir), latest_state(persona_dir)
    skills_first = {str(k): float(v) for k, v in (state_first.get("skills") or {}).items()}
    skills_last = {str(k): float(v) for k, v in (state_last.get("skills") or {}).items()}

    result: dict[str, Any] = {
        "persona": persona_dir.name,
        "simulated": bool(activity_rows),
        "skills": {
            "count_first": len(skills_first),
            "count_last": len(skills_last),
            "gained": sorted(set(skills_last) - set(skills_first)),
            # The names themselves, so cross-character synonym fragmentation can
            # be measured (phase-5 follow-up).
            "last": sorted(skills_last),
            "delta_sum": round(
                sum(
                    skills_last[k] - skills_first.get(k, 0.0)
                    for k in skills_last
                ),
                4,
            ),
        },
        "activities": {
            "count": len(activity_rows),
            "by_type": counts(activity_rows, "type"),
            "by_week": len({week_of(r.get("time")) for r in activity_rows if week_of(r.get("time"))}),
            "activity_ids": sum(bool(r.get("activity_id")) for r in activity_rows),
        },
        "weekly_diary": {
            "count": len(diary_rows),
            "scratchpad_entries": len(scratch_rows),
        },
        "vitality": vitality_distribution(read_jsonl(persona_dir / "state.jsonl")),
        "proficiency_projection": _proficiency_projection(
            run_dir, persona_dir, skills_last
        ),
        "capability_cap": _capability_cap(run_dir, persona_dir),
        "skill_growth": _skill_growth(persona_dir),
        "memory_strength": _memory_strength(persona_dir),
    }
    if not activity_rows and not (persona_dir / "generation").exists():
        # A persona directory exists for every persona copied from the world
        # template; only the agents actually bootstrapped have data.
        result["note"] = "persona present in the run directory but never simulated"
        return result

    bad_diaries = [
        str(row.get("time") or "")
        for row in diary_rows
        if not str(row.get("content") or "").lstrip().lower().startswith("summary:")
        or "<think_brief>" in str(row.get("content") or "").lower()
    ]
    result["weekly_diary"]["bad_count"] = len(bad_diaries)
    result["weekly_diary"]["bad_weeks"] = bad_diaries
    result["generation"] = generation_integrity(persona_dir)
    fallback_rows = [
        str(row.get("time") or "")
        for row in activity_rows
        if row.get("reflection_parse_fallback")
    ]
    result["reflection_parse"] = {
        "activities": len(activity_rows),
        "fallback_count": len(fallback_rows),
        "fallback_share": (
            round(len(fallback_rows) / len(activity_rows), 4) if activity_rows else None
        ),
    }

    capability_events = read_jsonl(cognition / "capability_events.jsonl")
    memory_events = read_jsonl(cognition / "memory_events.jsonl")
    relation_events = read_jsonl(cognition / "memory_relation_events.jsonl")
    idea_events = read_jsonl(cognition / "idea_events.jsonl")
    cognition_events = capability_events + memory_events + relation_events + idea_events

    event_contract_violations = [
        str(row.get("idempotency_key") or row.get("type") or "unknown")
        for row in cognition_events
        if not row.get("ledger_event_id")
        or not row.get("schema_version")
        or not row.get("idempotency_key")
    ]
    duplicate_idempotency = [
        key
        for key, n in Counter(str(r.get("idempotency_key") or "") for r in cognition_events).items()
        if key and n > 1
    ]
    bad_events = [
        f"{row.get('type')}:{row.get('idempotency_key')}"
        for row in cognition_events
        if not row.get("ledger_event_id") or not row.get("schema_version")
    ]
    result["event_contract"] = {
        "cognition_events": len(cognition_events),
        "by_stream": {
            "capability": len(capability_events),
            "memory": len(memory_events),
            "relation": len(relation_events),
            "idea": len(idea_events),
        },
        "violations": event_contract_violations,
        "violations_without_identity": bad_events[:20],
        "duplicate_idempotency_keys": duplicate_idempotency,
    }

    # -- memory ------------------------------------------------------------
    memory_view = read_json(views / "memories.json", {"memories": {}})
    memories: dict[str, dict[str, Any]] = memory_view.get("memories") or {}
    missing_memory_sources: dict[str, list[str]] = {}
    for memory_id, memory in memories.items():
        missing = [
            str(src)
            for src in memory.get("source_event_ids") or []
            if str(src) not in ledger_ids
        ]
        if missing:
            missing_memory_sources[memory_id] = missing
    no_source_memories = [
        mid for mid, m in memories.items() if not (m.get("source_event_ids") or [])
    ]
    created = [row for row in memory_events if row.get("type") == "MEMORY_CREATED"]
    accepted_entities = sum(len(row.get("entities") or []) for row in created)
    dropped_entities = sum(len(row.get("dropped_entities") or []) for row in created)
    soft_entities = sum(len(row.get("soft_entities") or []) for row in created)
    weeks = sorted({week_of(row.get("time")) for row in created if week_of(row.get("time"))})
    per_week = Counter(week_of(row.get("time")) for row in created if week_of(row.get("time")))
    strength_rows = [row for row in memory_events if row.get("type") == "MEMORY_STRENGTH_UPDATED"]
    recall_rows = [row for row in memory_events if row.get("type") == "MEMORY_RECALLED"]
    result["memory"] = {
        "count": len(memories),
        "by_kind": dict(sorted(Counter(str(m.get("kind") or "") for m in memories.values()).items())),
        "by_status": dict(sorted(Counter(str(m.get("status") or "") for m in memories.values()).items())),
        "by_tier": dict(sorted(Counter(str(m.get("tier") or "") for m in memories.values()).items())),
        "protected": sum(bool(m.get("protected")) for m in memories.values()),
        "no_source": no_source_memories,
        "missing_sources": missing_memory_sources,
        "created_events": len(created),
        "created_per_week": dict(sorted(per_week.items())),
        "accepted_entities": accepted_entities,
        "dropped_entities": dropped_entities,
        "soft_entities": soft_entities,
        "soft_entity_rate": (
            round(soft_entities / (accepted_entities + soft_entities + dropped_entities), 4)
            if (accepted_entities + soft_entities + dropped_entities)
            else None
        ),
        "entity_drop_rate": (
            round(dropped_entities / (accepted_entities + dropped_entities), 4)
            if (accepted_entities + dropped_entities)
            else None
        ),
        "memories_without_entities": sum(
            1 for m in memories.values() if not (m.get("entities") or [])
        ),
        "with_obstacles": sum(bool(m.get("obstacles")) for m in memories.values()),
        "with_resources": sum(bool(m.get("resources")) for m in memories.values()),
        "recall_events": len(recall_rows),
        "recall_used_true": sum(bool(r.get("used")) for r in recall_rows),
        "strength_updates": len(strength_rows),
        "duplicate_pairs": duplicate_pairs(
            [(mid, str(m.get("content") or "")) for mid, m in memories.items()]
        ),
        "near_duplicate_clusters": near_duplicate_clusters(
            [(mid, str(m.get("content") or "")) for mid, m in memories.items()]
        )[:10],
        "events_by_type": counts(memory_events, "type"),
        "weeks_covered": weeks,
    }

    # -- impressions (KI-2) ------------------------------------------------
    impressions_view = read_json(persona_dir / "memory" / "impressions.json", {})
    result_impressions = {
        "exists": bool(impressions_view),
        "canonical": int((impressions_view.get("stats") or {}).get("canonical") or 0),
        "aliases": int((impressions_view.get("stats") or {}).get("aliases") or 0),
        "soft_entities": int(
            (impressions_view.get("stats") or {}).get("soft_entities") or 0
        ),
        "aliases_learned": sum(
            1
            for row in read_jsonl(persona_dir / "memory" / "impression_events.jsonl")
            if row.get("type") == "ALIAS_LEARNED"
        ),
    }

    # -- relations ---------------------------------------------------------
    graph = read_json(views / "memory_graph.json", {"edges": []})
    edges = graph.get("edges") or []
    bad_edges = [
        f"{edge.get('source')}->{edge.get('target')}"
        for edge in edges
        if str(edge.get("source") or "") not in memories
        or str(edge.get("target") or "") not in memories
    ]
    relation_types = Counter()
    for edge in edges:
        for relation_type in edge.get("types") or [edge.get("relation_type")]:
            if relation_type:
                relation_types[str(relation_type)] += 1
    asserted_events = [row for row in relation_events if row.get("type") == "RELATION_ASSERTED"]
    result["impressions"] = result_impressions

    result["relations"] = {
        "count": len(edges),
        "by_type": dict(sorted(relation_types.items())),
        "bad_references": bad_edges,
        "events": len(relation_events),
        "asserted_events": len(asserted_events),
        "edge_to_memory_ratio": round(len(edges) / len(memories), 4) if memories else None,
        "weeks": sorted(
            {week_of(row.get("time")) for row in asserted_events if week_of(row.get("time"))}
        ),
    }

    # -- retrieval shadow --------------------------------------------------
    retrieval = read_json(views / "retrieval_shadow.json", {})
    week_rows = retrieval.get("weeks") or []
    if isinstance(week_rows, dict):
        week_rows = list(week_rows.values())
    legacy_tokens = sum(int((s.get("legacy") or {}).get("est_tokens") or 0) for s in week_rows)
    proposed_tokens = sum(int((s.get("proposed") or {}).get("est_tokens") or 0) for s in week_rows)
    result["retrieval_shadow"] = {
        "snapshots": len(week_rows),
        "legacy_tokens": legacy_tokens,
        "proposed_tokens": proposed_tokens,
        "token_ratio": round(proposed_tokens / legacy_tokens, 4) if legacy_tokens else None,
        "cold_reactivated": sum(int(s.get("cold_reactivated") or 0) for s in week_rows),
        "shared_items": sum(int((s.get("overlap") or {}).get("shared") or 0) for s in week_rows),
    }

    # -- ideas -------------------------------------------------------------
    idea_view = read_json(views / "ideas.json", {"ideas": {}, "rejected": []})
    ideas: dict[str, dict[str, Any]] = idea_view.get("ideas") or {}
    rejected_ideas = idea_view.get("rejected") or []
    invalid_ideas: dict[str, list[str]] = {}
    for idea_id, idea in ideas.items():
        issues: list[str] = []
        source_ids = [str(v) for v in idea.get("source_memory_ids") or []]
        if not 2 <= len(source_ids) <= 4:
            issues.append("source_count_not_2_to_4")
        if any(mid not in memories for mid in source_ids):
            issues.append("missing_source_memory")
        if not str(idea.get("test_plan") or "").strip():
            issues.append("missing_test_plan")
        if float(idea.get("confidence") or 0) > 0.5:
            issues.append("candidate_confidence_above_0.5")
        if str(idea.get("status") or "") != "candidate":
            issues.append("unexpected_phase3_status")
        if issues:
            invalid_ideas[idea_id] = issues

    idea_source_counter: Counter[str] = Counter()
    for idea in ideas.values():
        for memory_id in idea.get("source_memory_ids") or []:
            idea_source_counter[str(memory_id)] += 1

    # Insight clusters: same motif + overlapping evidence, or very similar text.
    idea_items = list(ideas.items())
    insight_parent: dict[str, str] = {idea_id: idea_id for idea_id, _ in idea_items}

    def find_insight(x: str) -> str:
        while insight_parent[x] != x:
            insight_parent[x] = insight_parent[insight_parent[x]]
            x = insight_parent[x]
        return x

    def union_insight(a: str, b: str) -> None:
        ra, rb = find_insight(a), find_insight(b)
        if ra != rb:
            insight_parent[rb] = ra

    # A duplicate *the engine was supposed to fold*: same motif, substantial
    # evidence overlap and near-identical wording or test plan — exactly the
    # rule in `idea_engine._semantic_duplicate` (KI-5). Thresholds are imported
    # from the engine so the two cannot drift apart.
    for i, (left_id, left) in enumerate(idea_items):
        for right_id, right in idea_items[i + 1 :]:
            left_sources = {str(s) for s in left.get("source_memory_ids") or []}
            right_sources = {str(s) for s in right.get("source_memory_ids") or []}
            same_motif = str(left.get("motif")) == str(right.get("motif"))
            shared = left_sources & right_sources
            overlap = (
                len(shared) / len(left_sources | right_sources)
                if (left_sources | right_sources)
                else 0.0
            )
            if not same_motif or overlap < SOURCE_OVERLAP_DUPLICATE:
                continue
            content_similarity = jaccard(left.get("content"), right.get("content"))
            plan_similarity = jaccard(left.get("test_plan"), right.get("test_plan"))
            if (
                content_similarity >= NOVELTY_REJECT_THRESHOLD
                or plan_similarity >= TEST_PLAN_DUPLICATE_SIMILARITY
            ):
                union_insight(left_id, right_id)

    insight_groups: dict[str, list[str]] = defaultdict(list)
    for idea_id, _ in idea_items:
        insight_groups[find_insight(idea_id)].append(idea_id)
    insight_clusters = []
    for members in insight_groups.values():
        if len(members) < 2:
            continue
        converted = sorted(
            {
                str(ideas[m].get("converted_method_id"))
                for m in members
                if ideas[m].get("converted_method_id")
            }
        )
        insight_clusters.append(
            {
                "ideas": sorted(members),
                "converted_methods": converted,
                "weeks": sorted(
                    {week_of(ideas[m].get("created_at")) for m in members if week_of(ideas[m].get("created_at"))}
                ),
            }
        )
    insight_clusters.sort(key=lambda c: -len(c["ideas"]))

    # Informational: ideas that merely share *some* evidence (a hub memory can
    # legitimately feed several different hypotheses, so this is not a defect).
    hub_parent: Dict[str, str] = {idea_id: idea_id for idea_id, _ in idea_items}

    def find_hub(x: str) -> str:
        while hub_parent[x] != x:
            hub_parent[x] = hub_parent[hub_parent[x]]
            x = hub_parent[x]
        return x

    for i, (left_id, left) in enumerate(idea_items):
        for right_id, right in idea_items[i + 1 :]:
            if set(map(str, left.get("source_memory_ids") or [])) & set(
                map(str, right.get("source_memory_ids") or [])
            ):
                ra, rb = find_hub(left_id), find_hub(right_id)
                if ra != rb:
                    hub_parent[rb] = ra
    hub_groups: Dict[str, List[str]] = defaultdict(list)
    for idea_id, _ in idea_items:
        hub_groups[find_hub(idea_id)].append(idea_id)
    shared_evidence_clusters = [
        {"ideas": sorted(members), "size": len(members)}
        for members in hub_groups.values()
        if len(members) > 1
    ]
    shared_evidence_clusters.sort(key=lambda c: -c["size"])

    converted_ideas = [i for i in ideas.values() if i.get("converted_method_id")]
    goal_linked = [i for i in ideas.values() if i.get("related_goal_ids")]
    testplan_concrete = [
        i for i in ideas.values() if CONCRETE.search(str(i.get("test_plan") or ""))
    ]
    idea_texts = [str(i.get("content") or "") for i in ideas.values()] + [
        str(i.get("test_plan") or "") for i in ideas.values()
    ]
    result["ideas"] = {
        "count": len(ideas),
        "rejected": len(rejected_ideas),
        "rejected_reasons": dict(
            sorted(
                Counter(
                    str(reason)
                    for row in rejected_ideas
                    for reason in (row.get("reasons") or row.get("reason") or ["unknown"])
                ).items()
            )
        ),
        "created_events": sum(1 for row in idea_events if row.get("type") == "IDEA_CREATED"),
        "by_motif": dict(sorted(Counter(str(i.get("motif") or "") for i in ideas.values()).items())),
        "by_type": dict(sorted(Counter(str(i.get("idea_type") or "") for i in ideas.values()).items())),
        "by_week": dict(
            sorted(
                Counter(
                    week_of(row.get("created_at") or row.get("time"))
                    for row in ideas.values()
                    if week_of(row.get("created_at") or row.get("time"))
                ).items()
            )
        ),
        "converted": len(converted_ideas),
        "refined": sum(1 for row in idea_events if row.get("type") == "IDEA_REFINED"),
        "refinement_targets": dict(
            sorted(
                Counter(
                    str((row.get("duplicate_of") or row.get("idea_id") or ""))
                    for row in idea_events
                    if row.get("type") == "IDEA_REFINED"
                ).items()
            )
        ),
        "conversion_rate": round(len(converted_ideas) / len(ideas), 4) if ideas else None,
        "goal_linked": len(goal_linked),
        "goal_link_rate": round(len(goal_linked) / len(ideas), 4) if ideas else None,
        "concrete_test_plan": len(testplan_concrete),
        "potential_range": [
            min((float(i.get("potential") or 0) for i in ideas.values()), default=None),
            max((float(i.get("potential") or 0) for i in ideas.values()), default=None),
        ],
        "invalid": invalid_ideas,
        "duplicate_pairs": duplicate_pairs(
            [(idea_id, str(i.get("content") or "")) for idea_id, i in ideas.items()]
        ),
        "near_duplicate_clusters": near_duplicate_clusters(
            [(idea_id, str(i.get("content") or "")) for idea_id, i in ideas.items()]
        ),
        "insight_clusters": insight_clusters,
        "shared_evidence_clusters": shared_evidence_clusters,
        "reused_source_memories": [
            {"memory_id": mid, "ideas": n}
            for mid, n in idea_source_counter.most_common(8)
            if n > 1
        ],
        "ascii_id_leaks": ascii_id_leaks(idea_texts),
        "skipped": len(idea_view.get("skipped") or []),
        "skipped_by_reason": idea_view.get("skipped_by_reason") or {},
        "weeks_with_idea": sorted(
            {
                week_of(i.get("created_at") or i.get("week"))
                for i in ideas.values()
                if week_of(i.get("created_at") or i.get("week"))
            }
        ),
        "rejected_by_reason": dict(
            sorted(
                Counter(
                    str(reason)
                    for row in rejected_ideas
                    for reason in (row.get("reasons") or [])
                ).items()
            )
        ),
        "rejected_by_motif": dict(
            sorted(
                Counter(
                    str(row.get("motif") or "")
                    for row in rejected_ideas
                    if row.get("motif")
                ).items()
            )
        ),
        "rejected_below_min_potential_share": (
            round(
                sum(
                    1
                    for row in rejected_ideas
                    if "below_min_potential" in (row.get("reasons") or [])
                )
                / len(rejected_ideas),
                4,
            )
            if rejected_ideas
            else None
        ),
        "events_by_type": counts(idea_events, "type"),
        "samples": [
            {
                "idea_id": idea_id,
                "content": idea.get("content"),
                "test_plan": idea.get("test_plan"),
                "motif": idea.get("motif"),
                "potential": idea.get("potential"),
                "source_memory_ids": idea.get("source_memory_ids"),
                "converted_method_id": idea.get("converted_method_id"),
                "related_goal_ids": idea.get("related_goal_ids"),
            }
            for idea_id, idea in list(sorted(ideas.items()))[:20]
        ],
        "rejected_samples": [
            {
                "idea_id": row.get("idea_id"),
                "motif": row.get("motif"),
                "content": row.get("content"),
                "reasons": row.get("reasons") or row.get("reason"),
            }
            for row in rejected_ideas[:10]
        ],
    }

    # -- methodologies -----------------------------------------------------
    from src.agents.cognition.materializer import materialize

    capability_view = materialize(
        capability_events,
        skills=skills_last,
        persona=persona_dir.name,
    )
    methods: dict[str, dict[str, Any]] = capability_view.get("methodologies") or {}

    proposed = [row for row in capability_events if row.get("type") == "METHOD_PROPOSED"]
    conversions = [row for row in proposed if row.get("source_type") == "idea_conversion"]
    invalid_conversions = [
        str(row.get("method_id") or "unknown")
        for row in conversions
        if float(row.get("global_value") or 0) != 0.0
        or float(row.get("confidence") or 0) != 0.0
        or not row.get("source_idea_id")
    ]
    value_updates = [row for row in capability_events if row.get("type") == "METHOD_VALUE_UPDATED"]
    value_without_activity = [
        str(row.get("method_id") or "") for row in value_updates if not row.get("activity_id")
    ]
    unmapped_methods = [
        str(row.get("method_id") or "")
        for row in proposed
        if row.get("skill_id") and str(row.get("skill_id")) not in skills_last
    ]
    incomplete_methods = [
        {
            "method_id": str(row.get("method_id") or ""),
            "missing": list(row.get("structure_missing") or []),
            "week": row.get("week"),
        }
        for row in proposed
        if row.get("structure_incomplete")
    ]
    outcome_rows = [row for row in capability_events if row.get("type") == "METHOD_OUTCOME_OBSERVED"]
    rewards = [float(row.get("reward") or 0.0) for row in outcome_rows]
    selections = [row for row in capability_events if row.get("type") == "METHOD_SELECTED"]
    selection_counter = Counter(str(row.get("method_id") or "") for row in selections)
    explored_counter = Counter(
        str(row.get("method_id") or "") for row in selections if row.get("explored")
    )

    def structure_of(method: dict[str, Any]) -> dict[str, Any]:
        steps = method.get("steps") or []
        checks = method.get("checks") or []
        failure_modes = method.get("failure_modes") or []
        contexts = method.get("applicable_contexts") or []
        return {
            "steps": len(steps),
            "checks": len(checks),
            "failure_modes": len(failure_modes),
            "contexts": len(contexts),
            "title_chars": len(str(method.get("title") or "")),
            "complete": bool(len(steps) >= 2 and checks and failure_modes and contexts),
        }

    structures = {mid: structure_of(m) for mid, m in methods.items()}
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for method_id, method in methods.items():
        entry = dict(structures[method_id])
        entry["method_id"] = method_id
        entry["practice_count"] = int(method.get("practice_count") or 0)
        entry["value"] = round4(method.get("global_value"))
        by_source[str(method.get("source_type") or "unknown")].append(entry)

    source_summary = {
        source: {
            "methods": len(entries),
            "complete_structure": sum(bool(e["complete"]) for e in entries),
            "avg_steps": mean(e["steps"] for e in entries),
            "avg_checks": mean(e["checks"] for e in entries),
            "avg_failure_modes": mean(e["failure_modes"] for e in entries),
            "avg_contexts": mean(e["contexts"] for e in entries),
            "avg_title_chars": mean(e["title_chars"] for e in entries),
            "avg_practice_count": mean(e["practice_count"] for e in entries),
        }
        for source, entries in sorted(by_source.items())
    }

    skill_method_coverage = {
        skill: sum(1 for m in methods.values() if str(m.get("skill_id")) == skill)
        for skill in sorted(skills_last)
    }
    selected_any = {mid for mid, n in selection_counter.items() if n}
    result["methodologies"] = {
        "events_by_type": counts(capability_events, "type"),
        "methods": len(methods),
        "by_status": (capability_view.get("stats") or {}).get("by_status") or {},
        "by_source_type": dict(
            sorted(Counter(str(m.get("source_type") or "") for m in methods.values()).items())
        ),
        "idea_conversions": len(conversions),
        "invalid_conversions": invalid_conversions,
        "value_updates_without_activity": value_without_activity,
        "unmapped_methods": unmapped_methods,
        "structure": structures,
        "structure_by_source": source_summary,
        "structure_incomplete": incomplete_methods,
        "complete_structure_share": (
            round(sum(bool(s["complete"]) for s in structures.values()) / len(structures), 4)
            if structures
            else None
        ),
        "selection": {
            "events": len(selections),
            "distinct_methods_selected": len(selection_counter),
            "methods_never_selected": sorted(set(methods) - selected_any),
            "top1_method": selection_counter.most_common(1)[0][0] if selection_counter else None,
            "top1_share": (
                round(selection_counter.most_common(1)[0][1] / len(selections), 4)
                if selections
                else None
            ),
            "explored_events": sum(explored_counter.values()),
            "explored_share": (
                round(sum(explored_counter.values()) / len(selections), 4) if selections else None
            ),
            "distinct_explored_methods": len(explored_counter),
            "explored_repeat_share": (
                round(
                    1 - len(explored_counter) / sum(explored_counter.values()), 4
                )
                if explored_counter
                else None
            ),
            "by_method": dict(selection_counter.most_common()),
        },
        "outcomes": {
            "events": len(outcome_rows),
            "reward_min": round(min(rewards), 4) if rewards else None,
            "reward_max": round(max(rewards), 4) if rewards else None,
            "reward_mean": mean(rewards),
            "positive_share": (
                round(sum(1 for r in rewards if r > 0) / len(rewards), 4) if rewards else None
            ),
            "value_updates": len(value_updates),
        },
        "skill_method_coverage": skill_method_coverage,
        "skills_without_method": sorted(s for s, n in skill_method_coverage.items() if not n),
        "duplicate_pairs": duplicate_pairs(
            [(str(row.get("method_id") or ""), str(row.get("title") or "")) for row in proposed],
            threshold=0.5,
        ),
        "near_duplicate_clusters": near_duplicate_clusters(
            [
                (str(mid), f"{m.get('title') or ''} {m.get('description') or ''}")
                for mid, m in methods.items()
            ],
            threshold=0.35,
        ),
        "persisted_view_exists": (views / "capabilities.json").exists(),
        "samples": [
            {
                "method_id": method_id,
                "skill_id": method.get("skill_id"),
                "title": method.get("title"),
                "description": method.get("description"),
                "status": method.get("status"),
                "global_value": method.get("global_value"),
                "confidence": method.get("confidence"),
                "practice_count": method.get("practice_count"),
                "success_count": method.get("success_count"),
                "source_type": method.get("source_type"),
                "source_idea_id": method.get("source_idea_id"),
                "structure": structures.get(method_id),
                "steps": method.get("steps"),
                "checks": method.get("checks"),
                "failure_modes": method.get("failure_modes"),
                "applicable_contexts": method.get("applicable_contexts"),
            }
            for method_id, method in list(sorted(methods.items()))[:30]
        ],
    }

    # -- phase 4: passive method hints (offer != reinforcement) -------------
    def week_key(row: dict[str, Any]) -> str:
        return str(row.get("week") or week_of(row.get("time")))

    hint_offers = [row for row in capability_events if row.get("type") == "METHOD_HINTED"]
    hint_picks = [row for row in selections if str(row.get("source") or "") == "hint"]
    hint_applied = [
        row
        for row in capability_events
        if row.get("type") == "METHOD_APPLIED" and str(row.get("source") or "") == "hint"
    ]
    real_outcomes = [row for row in outcome_rows if row.get("attribution") == "real_adoption"]
    shadow_outcomes = [
        row for row in outcome_rows if row.get("attribution") == "shadow_counterfactual"
    ]
    real_updates = [row for row in value_updates if row.get("evidence_kind") == "real_adoption"]
    shadow_updates = [
        row for row in value_updates if row.get("evidence_kind") == "shadow_counterfactual"
    ]
    # Which slot an adopted method actually occupied: an adoption carries no
    # slot of its own, so it has to be joined to that week's offer.
    _offer_role = {
        (str(row.get("method_id") or ""), week_key(row)): str(row.get("role") or "")
        for row in hint_offers
    }
    adopted_by_role: Counter = Counter(
        _offer_role.get((str(row.get("method_id") or ""), week_key(row)), "")
        for row in hint_picks
    )
    offered_pairs = {(str(row.get("method_id") or ""), week_key(row)) for row in hint_offers}
    picked_pairs = {(str(row.get("method_id") or ""), week_key(row)) for row in hint_picks}
    applied_pairs = {(str(row.get("method_id") or ""), week_key(row)) for row in hint_applied}
    applied_ids = {str(row.get("activity_id") or "") for row in hint_applied}
    real_activities = {
        str(row.get("activity_id") or "") for row in real_outcomes if row.get("activity_id")
    }
    shadow_activities = {
        str(row.get("activity_id") or "") for row in shadow_outcomes if row.get("activity_id")
    }
    used_in_plan_rows = [row for row in memory_events if row.get("type") == "MEMORY_USED_IN_PLAN"]
    used_in_plan_methods = {str(row.get("method_id") or "") for row in used_in_plan_rows}
    block_chars = [int(row.get("block_chars") or 0) for row in hint_offers]
    result["method_hints"] = {
        "offers": {
            "events": len(hint_offers),
            "distinct_methods": len({str(row.get("method_id") or "") for row in hint_offers}),
            "weeks": len({week_key(row) for row in hint_offers}),
            "block_chars_max": max(block_chars) if block_chars else None,
            "block_chars_mean": mean(float(c) for c in block_chars),
            "roles": dict(Counter(str(row.get("role") or "") for row in hint_offers)),
        },
        "adoptions": {
            "events": len(hint_picks),
            "distinct_methods": len({str(row.get("method_id") or "") for row in hint_picks}),
            # share of offered (method, week) pairs the character chose to take up
            "adoption_rate": (
                round(len(picked_pairs & offered_pairs) / len(offered_pairs), 4)
                if offered_pairs
                else None
            ),
            "matched_by": dict(Counter(str(row.get("matched_by") or "") for row in hint_picks)),
            # adopting something that was never offered breaks the contract
            "unoffered": sorted(picked_pairs - offered_pairs),
        },
        "practice": {
            "applied": len(hint_applied),
            "real_outcomes": len(real_outcomes),
            "shadow_outcomes": len(shadow_outcomes),
            "real_share_of_adoptions": (
                round(len(real_outcomes) / len(hint_picks), 4) if hint_picks else None
            ),
            "reward_mean": mean(float(row.get("reward") or 0.0) for row in real_outcomes),
            # invariant #6: a method may only move on real practice
            "free_reinforcement": sorted(
                {
                    (str(row.get("method_id") or ""), week_key(row))
                    for row in real_updates
                    if (str(row.get("method_id") or ""), week_key(row)) not in applied_pairs
                }
            ),
            "real_updates_without_practice": [
                {
                    "method_id": str(row.get("method_id") or ""),
                    "activity_id": str(row.get("activity_id") or ""),
                }
                for row in real_updates
                if str(row.get("activity_id") or "") not in applied_ids
            ],
            # one activity must not feed both a real and a counterfactual update
            "double_counted_activities": sorted(real_activities & shadow_activities),
            "shadow_updates": len(shadow_updates),
        },
        "used_in_plan": {
            "events": len(used_in_plan_rows),
            "distinct_memories": len(
                {str(row.get("memory_id") or "") for row in used_in_plan_rows}
            ),
            "distinct_methods": len(used_in_plan_methods),
            # an adopted method carrying memories must exercise them (KI-10)
            "adoptions_missing_used_in_plan": sorted(
                {
                    str(row.get("method_id") or "")
                    for row in hint_picks
                    if (methods.get(str(row.get("method_id") or ""), {}).get("source_memory_ids") or [])
                    and str(row.get("method_id") or "") not in used_in_plan_methods
                }
            ),
        },
    }

    # -- phase 5: the character's answer to the menu (declines) -------------
    decline_rows = [row for row in capability_events if row.get("type") == "METHOD_DECLINED"]
    declined_pairs = {(str(row.get("method_id") or ""), week_key(row)) for row in decline_rows}
    menu_weeks = {week_key(row) for row in hint_offers}
    ignored_weeks = sorted(
        week
        for week in menu_weeks
        if week not in {week_key(row) for row in hint_picks}
    )
    # The exploit slot must not be occupied by the same method week after week
    # (phase4-review §3 #3: `自然亲和-c449f9c899` was offered in W06/07/09/W10).
    offers_by_week: dict[str, list[str]] = {}
    for row in sorted(hint_offers, key=lambda r: (week_key(r), str(r.get("method_id") or ""))):
        offers_by_week.setdefault(week_key(row), []).append(str(row.get("method_id") or ""))
    exploit_by_week = {
        week: [
            str(row.get("method_id") or "")
            for row in hint_offers
            if week_key(row) == week and str(row.get("role") or "") == "exploit"
        ]
        for week in offers_by_week
    }
    longest_same_lead = 0
    _current_lead, _current_run = "", 0
    for week in sorted(exploit_by_week):
        leads = exploit_by_week[week]
        lead = leads[0] if leads else ""
        if lead and lead == _current_lead:
            _current_run += 1
        else:
            _current_lead, _current_run = lead, 1 if lead else 0
        longest_same_lead = max(longest_same_lead, _current_run)
    result["method_choices"] = {
        "declines": {
            "events": len(decline_rows),
            "distinct_methods": len({str(row.get("method_id") or "") for row in decline_rows}),
            "rate_of_offers": (
                round(len(declined_pairs & offered_pairs) / len(offered_pairs), 4)
                if offered_pairs
                else None
            ),
            "reasons": dict(Counter(str(row.get("reason") or "") for row in decline_rows)),
            "by_role": dict(Counter(str(row.get("role") or "") for row in decline_rows)),
            # weeks where the menu was shown and the character took nothing up
            "weeks_with_no_adoption": len(ignored_weeks),
            "weeks_total": len(menu_weeks),
        },
        "bandit": {
            # Read off the offers themselves: the event says which selection rule
            # produced the menu, so the report does not have to trust the config.
            "enabled": any(bool(row.get("bandit")) for row in hint_offers),
            "offers_per_week": (round(len(hint_offers) / len(menu_weeks), 3) if menu_weeks else None),
            "distinct_methods_per_week": (
                round(sum(len(v) for v in offers_by_week.values()) / len(offers_by_week), 3)
                if offers_by_week
                else None
            ),
            "explore_offers": sum(1 for row in hint_offers if str(row.get("role") or "") == "explore"),
            # Adoptions per *slot*, matched to the offer for the same
            # (method, week). Reading `role` off the METHOD_SELECTED row instead
            # is always "primary" — there is no slot on an adoption — and that
            # bug made this reading report 0 through two acceptance rounds and a
            # code change built on top of it. Run 09272220's real numbers were
            # exploit 21/57, explore 12/43, filler 21/69.
            "adoptions_by_slot": {
                role: {
                    "offers": offered,
                    "adoptions": adopted_by_role.get(role, 0),
                    "rate": (
                        round(adopted_by_role.get(role, 0) / offered, 4)
                        if offered
                        else None
                    ),
                }
                for role, offered in sorted(
                    Counter(str(row.get("role") or "") for row in hint_offers).items()
                )
            },
            "explore_adoptions": adopted_by_role.get("explore", 0),
            "longest_same_exploit_method": longest_same_lead,
            "offers_with_ignored_streak": sum(
                1 for row in hint_offers if int(row.get("ignored_streak") or 0) > 0
            ),
            # The phase-4 review's complaint was that the exploit slot never
            # rotated; the reading that matters is how often the *menu* changed.
            "weeks_with_a_new_lead": len(
                [
                    week
                    for index, week in enumerate(sorted(exploit_by_week))
                    if index == 0
                    or exploit_by_week[week] != exploit_by_week[sorted(exploit_by_week)[index - 1]]
                ]
            ),
        },
        "situation": {
            "distinct_context_keys": len(
                {str(row.get("context_key") or "") for row in hint_offers}
            ),
            "context_keys": dict(
                Counter(str(row.get("context_key") or "default") for row in hint_offers)
            ),
            "tag_counts": dict(
                Counter(
                    tag
                    for row in hint_offers
                    for tag in (row.get("context_tags") or [])
                )
            ),
            "offers_with_goal": sum(1 for row in hint_offers if str(row.get("goal") or "").strip()),
        },
        "reward_baseline_shadow": _baseline_shadow_report(outcome_rows, value_updates),
        "lifecycle": _lifecycle_report(methods, capability_events),
    }

    # -- phase 4.5: the lesson bridge (the character's own notes) -----------
    lesson_memories = [
        row
        for row in memories.values()
        if str(row.get("origin") or "") == "scratchpad_lesson"
    ]
    # Independent recomputation of the diff, straight from the scratchpads, so
    # the report does not have to trust the run's own logging.
    from types import SimpleNamespace

    from src.agents.cognition.lessons import LessonConfig, collect_lesson_diff

    lesson_diff = collect_lesson_diff(SimpleNamespace(root=persona_dir), LessonConfig(enabled=True))
    lesson_derived = [
        mid
        for mid, method in methods.items()
        if str(method.get("source_motif") or "") == "lesson_application"
    ]
    lesson_derived_offers = [
        row for row in hint_offers if str(row.get("origin") or "") == "own_lesson"
    ]
    lesson_derived_picks = [
        row for row in hint_picks if str(row.get("method_id") or "") in set(lesson_derived)
    ]
    lesson_derived_practice = [
        row for row in hint_applied if str(row.get("method_id") or "") in set(lesson_derived)
    ]
    result["lessons"] = {
        "memories": len(lesson_memories),
        "by_provenance": dict(
            Counter(str(row.get("provenance") or "?") for row in lesson_memories)
        ),
        "samples": [
            {
                "memory_id": row.get("memory_id"),
                "content": row.get("content"),
                "confidence": row.get("confidence"),
                "topics": row.get("topics"),
                "provenance": row.get("provenance"),
                "week": row.get("created_at"),
            }
            for row in lesson_memories[:12]
        ],
        "scratchpad": {
            "snapshots_with_lessons": 0,
            "latest_snapshot": lesson_diff.snapshot_time,
            "latest_event_id": lesson_diff.snapshot_event_id,
            "new": [line.text for line in lesson_diff.new],
            "rewritten": [
                {"text": line.text, "was": was} for line, was in lesson_diff.rewritten
            ],
            "carried": [line.text for line in lesson_diff.carried],
            "dropped": list(lesson_diff.dropped),
        },
        "chain": {
            "methods_from_lessons": len(lesson_derived),
            "offers": len(lesson_derived_offers),
            "adoptions": len(lesson_derived_picks),
            "practices": len(lesson_derived_practice),
            "adopted_method_ids": sorted({str(r.get("method_id") or "") for r in lesson_derived_picks}),
        },
        "ideas_by_motif": dict(
            Counter(str(i.get("motif") or "") for i in (ideas or {}).values())
        ),
    }

    # -- chain: memory -> idea -> method -> activity -> value ---------------
    activity_ids = {
        str(row.get("activity_id")) for row in activity_rows if row.get("activity_id")
    }
    method_source_ok, method_source_bad = 0, []
    for method_id, method in methods.items():
        if str(method.get("source_type")) != "idea_conversion":
            continue
        if str(method.get("source_idea_id") or "") in ideas:
            method_source_ok += 1
        else:
            method_source_bad.append(method_id)
    outcome_activity_unknown = [
        str(row.get("activity_id"))
        for row in outcome_rows
        if row.get("activity_id") and str(row.get("activity_id")) not in activity_ids
    ]
    result["chain"] = {
        "memory_to_idea": {
            "ideas_with_sources": sum(
                1 for i in ideas.values() if len(i.get("source_memory_ids") or []) >= 2
            ),
            "ideas": len(ideas),
        },
        "idea_to_method": {
            "converted_ideas": len(converted_ideas),
            "methods_with_valid_source_idea": method_source_ok,
            "methods_with_missing_source_idea": method_source_bad,
        },
        "method_to_activity": {
            "selections": len(selections),
            "outcomes": len(outcome_rows),
            "outcome_activity_ids_unknown": sorted(set(outcome_activity_unknown))[:10],
        },
        "activity_to_value": {
            "value_updates": len(value_updates),
            "updates_without_activity": len(value_without_activity),
        },
    }
    return result


def red_flags(personas: list[dict[str, Any]]) -> list[str]:
    flags: list[str] = []
    for persona in personas:
        name = persona["persona"]
        if not persona.get("simulated"):
            continue
        contract = persona.get("event_contract") or {}
        if contract.get("violations"):
            flags.append(f"[{name}] {len(contract['violations'])} cognition events lack identity/idempotency")
        if (persona.get("weekly_diary") or {}).get("bad_count"):
            flags.append(f"[{name}] {persona['weekly_diary']['bad_count']} weekly diary entries are not clean")
        gen = persona.get("generation") or {}
        if gen.get("bad_final_answer_count"):
            flags.append(
                f"[{name}] {gen['bad_final_answer_count']} generation records do not end with an assistant answer"
            )
        memory = persona.get("memory") or {}
        if memory.get("entity_drop_rate") is not None and memory["entity_drop_rate"] >= 0.5:
            flags.append(
                f"[{name}] {memory['entity_drop_rate']:.0%} of extracted entities were dropped "
                f"(accepted={memory['accepted_entities']}, dropped={memory['dropped_entities']}) "
                "-> relation graph loses the entity signal"
            )
        relations = persona.get("relations") or {}
        if relations.get("count") and set(relations.get("by_type") or {}) <= {
            "temporal",
            "contradiction",
            "method_transfer",
        }:
            flags.append(
                f"[{name}] relation graph only has {sorted((relations.get('by_type') or {}))} "
                "-> idea motifs are limited to what these types can express"
            )
        ideas = persona.get("ideas") or {}
        for cluster in (ideas.get("insight_clusters") or [])[:3]:
            flags.append(
                f"[{name}] KI-5 dedup missed {len(cluster['ideas'])} near-identical ideas "
                f"({', '.join(cluster['weeks'])}) -> {len(cluster['converted_methods'])} methods"
            )
        if ideas.get("ascii_id_leaks"):
            flags.append(
                f"[{name}] internal English ids leaked into character-facing idea text: "
                f"{ideas['ascii_id_leaks'][:5]}"
            )
        methods = persona.get("methodologies") or {}
        if methods.get("structure_incomplete"):
            flags.append(
                f"[{name}] {len(methods['structure_incomplete'])} idea-converted methods are "
                "structurally incomplete (program error, see KI-6): "
                + ", ".join(
                    f"{m['method_id']}({','.join(m['missing'])})"
                    for m in methods["structure_incomplete"][:3]
                )
            )
        if (persona.get("impressions") or {}).get("exists") is False:
            flags.append(f"[{name}] no impressions.json was written (KI-2 alias book missing)")
        if (ideas.get("refined") or 0) > 0:
            flags.append(
                f"[{name}] {ideas['refined']} re-derived ideas were folded into existing ones "
                "(KI-5 semantic dedup fired)"
            )
        structure_by_source = methods.get("structure_by_source") or {}
        idea_struct = structure_by_source.get("idea_conversion")
        practice_struct = structure_by_source.get("practice_reflection")
        if idea_struct and practice_struct:
            # Only a *material* structure gap is a defect (KI-6): a converted
            # method with no checks / no failure modes / fewer than two steps.
            material_gap = (
                (idea_struct.get("avg_steps") or 0) < 2
                or (idea_struct.get("avg_checks") or 0) < 1
                or (idea_struct.get("avg_failure_modes") or 0) < 1
                or (idea_struct.get("avg_contexts") or 0) < 1
            )
            if material_gap:
                flags.append(
                    f"[{name}] idea-converted methods are structurally poorer than "
                    f"review-derived ones (steps {idea_struct['avg_steps']} vs "
                    f"{practice_struct['avg_steps']}, checks {idea_struct['avg_checks']} vs "
                    f"{practice_struct['avg_checks']}, failure_modes "
                    f"{idea_struct['avg_failure_modes']} vs {practice_struct['avg_failure_modes']})"
                )
        selection = methods.get("selection") or {}
        if selection.get("top1_share") and selection["top1_share"] >= 0.4:
            flags.append(
                f"[{name}] method selection is concentrated: top method takes "
                f"{selection['top1_share']:.0%} of {selection['events']} selections, "
                f"{len(selection.get('methods_never_selected') or [])} methods were never selected"
            )
        if selection.get("explored_repeat_share") and selection["explored_repeat_share"] >= 0.5:
            flags.append(
                f"[{name}] exploration is degenerate: {selection['explored_repeat_share']:.0%} of "
                "exploration picks re-select an already-explored method"
            )
        outcomes = methods.get("outcomes") or {}
        if outcomes.get("events") and outcomes.get("positive_share") == 1.0:
            flags.append(
                f"[{name}] every observed outcome reward is positive "
                f"({outcomes['events']} events, {outcomes['reward_min']}..{outcomes['reward_max']}) "
                "-> value only rises, the status machine saturates at validated"
            )
        if methods.get("unmapped_methods"):
            flags.append(f"[{name}] {len(methods['unmapped_methods'])} methods map to no existing skill")
        if methods.get("value_updates_without_activity"):
            flags.append(
                f"[{name}] {len(methods['value_updates_without_activity'])} value updates have no activity evidence"
            )
        hints = persona.get("method_hints") or {}
        hint_practice = hints.get("practice") or {}
        if hint_practice.get("double_counted_activities"):
            flags.append(
                f"[{name}] {len(hint_practice['double_counted_activities'])} activities produced both "
                "real and counterfactual evidence -> one practice is counted twice "
                "(phase 4 dispatcher must claim an activity first)"
            )
        if hint_practice.get("free_reinforcement"):
            flags.append(
                f"[{name}] {len(hint_practice['free_reinforcement'])} method-weeks moved without practice "
                "-> being offered was treated as reinforcement (invariant #6)"
            )
        if hint_practice.get("real_updates_without_practice"):
            flags.append(
                f"[{name}] {len(hint_practice['real_updates_without_practice'])} real value updates "
                "carry an activity that was never recorded as applied"
            )
        if (hints.get("adoptions") or {}).get("unoffered"):
            flags.append(
                f"[{name}] {len(hints['adoptions']['unoffered'])} method-weeks were adopted without being "
                f"offered: {hints['adoptions']['unoffered'][:3]}"
            )
        adoptions = (hints.get("adoptions") or {}).get("events") or 0
        real_outcomes = (hint_practice.get("real_outcomes") or 0)
        if adoptions >= 3 and real_outcomes == 0:
            flags.append(
                f"[{name}] adopted {adoptions} method-weeks and never practised any of them "
                "-> either the character ignores its own plan or the practice matcher never fires"
            )
        if (hints.get("used_in_plan") or {}).get("adoptions_missing_used_in_plan"):
            flags.append(
                f"[{name}] adopted methods carrying memories never exercised them (KI-10): "
                f"{hints['used_in_plan']['adoptions_missing_used_in_plan'][:3]}"
            )
    return flags


def audit_run(run_dir: Path) -> dict[str, Any]:
    ledger = ledger_integrity(run_dir)
    ledger_ids = set()
    for path in run_dir.glob("**/*.jsonl"):
        try:
            for row in read_jsonl(path):
                if row.get("ledger_event_id"):
                    ledger_ids.add(str(row["ledger_event_id"]))
        except ValueError:
            continue

    persona_root = run_dir / "persona"
    personas = [
        audit_persona(run_dir, path, ledger_ids)
        for path in sorted(persona_root.iterdir(), key=lambda p: p.name)
        if path.is_dir() and (path / "profile").exists()
    ]
    run_cfg = read_json(run_dir / "config.json", {})
    simulated = [p for p in personas if p.get("simulated")]
    report = {
        "run": run_dir.name,
        "path": str(run_dir),
        "config": {
            "world": (run_cfg.get("world") or {}).get("name"),
            "language": (run_cfg.get("world") or {}).get("language"),
            "years": ((run_cfg.get("world") or {}).get("time") or {}).get("n_year"),
            "weeks": ((run_cfg.get("world") or {}).get("time") or {}).get("n_week"),
            "role_model": run_cfg.get("role_model"),
            "god_model": run_cfg.get("god_model"),
            "fallback_model": run_cfg.get("fallback_model"),
            "cognition": (run_cfg.get("world") or {}).get("cognition") or {},
        },
        "ledger": {
            "records": ledger["records"],
            "unique_ids": ledger["unique_ids"],
            "invalid_jsonl": ledger["invalid_jsonl"],
            "duplicate_ids": ledger["duplicate_ids"],
            "duplicate_id_count": ledger["duplicate_id_count"],
            "duplicate_row_count": ledger["duplicate_row_count"],
            "missing_identity_streams": ledger["missing_identity_streams"],
        },
        "personas_total": len(personas),
        "personas_simulated": len(simulated),
        "personas_not_simulated": [
            p["persona"] for p in personas if not p.get("simulated")
        ],
        "skill_fragmentation": _skill_fragmentation(run_dir, personas),
        "red_flags": red_flags(personas),
        "personas": personas,
    }
    report["acceptance_gates"] = acceptance_gates(report)
    return report



# ---------------------------------------------------------------------------
# Phase 5: the KI-8 shadow baseline, read off the run instead of a second run
# ---------------------------------------------------------------------------


def _spearman(pairs: list[tuple[float, float]]) -> float | None:
    """Rank correlation; `None` when there are fewer than two usable pairs."""
    if len(pairs) < 2:
        return None

    def _ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            shared = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = shared
            i = j + 1
        return ranks

    xs = _ranks([p[0] for p in pairs])
    ys = _ranks([p[1] for p in pairs])
    n = len(pairs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0 or var_y <= 0:
        return None
    return round(cov / ((var_x * var_y) ** 0.5), 4)


def _baseline_shadow_report(
    outcome_rows: list[dict[str, Any]],
    value_updates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Read the centred (KI-8) formula off the outcomes that carry it.

    The live value still uses the phase-1..3 formula, so this block answers the
    question the design document says must be answered *before* the switch:
    would centring actually produce negative rewards, would `deprecated` become
    reachable, and would it rank the character's methods differently?

    The centred score is **recomputed from the ledger** with the current formula
    rather than read from the recorded `shadow_reward` field, for two reasons:
    the other components (costs, transfer) are baseline-independent and already
    stored, and a projection has to be re-derivable when the formula itself is
    corrected — otherwise the report would describe the code of the day the run
    happened, not the code being accepted.
    """
    from src.agents.cognition.reward_model import (
        BASELINE_FULL_SCALE,
        BaselineTracker,
        RewardWeights,
        update_value,
    )

    outcomes = [row for row in outcome_rows if row.get("shadow_components") is not None]
    if not outcomes:
        return {
            "available": False,
            "reason": (
                "no METHOD_OUTCOME_OBSERVED carried a shadow baseline score "
                "(hints and the method shadow were off, or the run predates phase 5)"
            ),
        }

    weights = RewardWeights(use_baseline=True)

    def _recompute() -> list[float]:
        tracker = BaselineTracker()
        out: list[float] = []
        for row in sorted(outcomes, key=lambda r: str(r.get("time") or "")):
            components = row.get("shadow_components") or {}
            try:
                skill_gain = float(components.get("skill_gain") or 0.0)
            except (TypeError, ValueError):
                skill_gain = 0.0
            context_key = str(row.get("context_key") or "default")
            activity_type = str(row.get("activity_type") or "")
            baseline = tracker.baseline_for(context_key, activity_type=activity_type)
            delta = max(-1.0, min(1.0, (skill_gain - baseline) / BASELINE_FULL_SCALE))
            quality = max(0.0, min(1.0, 0.5 + 0.5 * delta))
            positive = (
                weights.skill_gain * skill_gain
                + weights.quality_weight * quality
                + weights.transferability * float(components.get("transferability") or 0.0)
            )
            negative = sum(
                float(components.get(key) or 0.0) * getattr(weights, key)
                for key in ("time_cost", "money_cost", "vitality_cost", "social_cost")
            )
            out.append(max(-1.0, min(1.0, positive - negative)))
            tracker.observe(context_key, skill_gain, activity_type=activity_type)
        return out

    shadow = _recompute()
    recorded = [float(row.get("shadow_reward") or 0.0) for row in outcomes]
    live = [float(row.get("reward") or 0.0) for row in outcomes]
    negative_live = [r for r in live if r < 0]

    def _replay(pick) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
        state: dict[str, dict[str, Any]] = {}
        transitions: dict[str, int] = {}
        for row in sorted(outcomes, key=lambda r: str(r.get("time") or "")):
            method_id = str(row.get("method_id") or "")
            current = state.setdefault(
                method_id,
                {"value": 0.0, "practice": 0, "success": 0, "status": "proposed"},
            )
            update = update_value(
                value=current["value"],
                practice_count=current["practice"],
                success_count=current["success"],
                reward=pick(row),
                current_status=current["status"],
            )
            if update.status != current["status"]:
                key = f"{current['status']}->{update.status}"
                transitions[key] = transitions.get(key, 0) + 1
            current.update(
                value=update.value,
                practice=update.practice_count,
                success=update.success_count,
                status=update.status,
            )
        return state, transitions

    rewards_by_time = {
        id(row): value
        for row, value in zip(sorted(outcomes, key=lambda r: str(r.get("time") or "")), shadow)
    }
    live_state, live_transitions = _replay(lambda row: float(row.get("reward") or 0.0))
    shadow_state, shadow_transitions = _replay(lambda row: rewards_by_time[id(row)])

    recorded_values: dict[str, float] = {}
    for row in value_updates:
        if str(row.get("evidence_kind") or "") in ("real_adoption", "shadow_counterfactual"):
            recorded_values[str(row.get("method_id") or "")] = float(row.get("global_value") or 0.0)
    reproduced = sum(
        1
        for method_id, value in recorded_values.items()
        if method_id in live_state
        and abs(live_state[method_id]["value"] - value) <= 0.02
    )

    ranking_pairs = [
        (live_state[mid]["value"], shadow_state[mid]["value"])
        for mid in sorted(set(live_state) & set(shadow_state))
    ]

    # 两个口径的名字：live 用配置里的，shadow 是它的对照口径。深度判据要按刻度算。
    from src.agents.cognition.reward_model import (
        DEFAULT_REWARD_CALIBER,
        alternative_caliber,
    )

    from src.config import get_config

    _cfg = (get_config() or {}).get("world") or {}
    live_caliber = str(
        (_cfg.get("cognition") or {}).get("reward_caliber") or DEFAULT_REWARD_CALIBER
    )
    shadow_caliber = alternative_caliber(live_caliber)

    return {
        "available": True,
        "outcomes": len(outcomes),
        "live_caliber": str(live_caliber),
        "shadow_caliber": str(shadow_caliber),
        "live": {
            "positive_share": round(sum(1 for r in live if r > 0) / len(live), 4),
            "negative": len(negative_live),
            "min": round(min(live), 4),
            "max": round(max(live), 4),
            "mean": round(sum(live) / len(live), 4),
        },
        "shadow": {
            "positive_share": round(sum(1 for r in shadow if r > 0) / len(shadow), 4),
            "negative": sum(1 for r in shadow if r < 0),
            # 保留原名（老读数），但门禁按 `shadow_caliber` 的刻度算深度：
            # `quality` 的满量程只有 signed 的 1/4，拿 −0.1 去比它会把很深的一条
            # 判成"深度不足"（3 年运行就是这样误判的）。
            "below_minus_0_1": sum(1 for r in shadow if r <= -0.1),
            "values": [round(r, 4) for r in shadow],
            "min": round(min(shadow), 4),
            "max": round(max(shadow), 4),
            "mean": round(sum(shadow) / len(shadow), 4),
        },
        "recorded_shadow_mean": round(sum(recorded) / len(recorded), 4),
        "projected_status_transitions": {
            "live": dict(sorted(live_transitions.items())),
            "shadow": dict(sorted(shadow_transitions.items())),
        },
        # Which caliber can reach `deprecated` matters: with the signed caliber
        # live, the value actually moves on that one, and the recorded
        # "shadow" is the design's floored mapping kept for comparison. Reporting
        # only the shadow's answer said "unreachable" while the run's own value
        # updates showed `tested -> deprecated` twice.
        "deprecated_reachable": any("deprecated" in key for key in live_transitions),
        "shadow_deprecated_reachable": any("deprecated" in key for key in shadow_transitions),
        "live_replay_matches_recorded": {
            "methods": len(recorded_values),
            "reproduced": reproduced,
            "exact": len(recorded_values) == 0 or reproduced == len(recorded_values),
        },
        "ranking_spearman": _spearman(ranking_pairs),
    }


def _week_number(time_str: Any) -> Optional[int]:
    """`Y2021-W07-review` → 全局周序号（年*10+周）。"""
    parts = str(time_str or "").split("-")
    if len(parts) < 2 or not parts[0].startswith("Y") or not parts[1].startswith("W"):
        return None
    try:
        return int(parts[0][1:]) * 10 + int(parts[1][1:])
    except ValueError:
        return None


def _memory_strength(persona_dir: Path) -> dict[str, Any]:
    """记忆热度与归档的读数（记忆强度视图 + 事件流）。

    设计正文要的是"冷记忆重新激活率"这类指标，而 `strength` 只在视图里，
    所以这里直接从 `views/memories.json` 读最终强度，再配合事件流算再激活：
    **被归档（tier=archived）却仍被召回过的记忆**就是一次真实的再激活。
    """
    view = read_json(persona_dir / "cognition" / "views" / "memories.json", {}) or {}
    memories = view.get("memories") or {}
    if isinstance(memories, list):  # 兼容列表形态
        memories = {str(m.get("memory_id")): m for m in memories}
    if not memories:
        return {"available": False}

    strengths = [
        float(m.get("strength") or 0.0) for m in memories.values()
    ]
    buckets = {"<0.2": 0, "0.2-0.5": 0, "0.5-0.8": 0, ">=0.8": 0}
    for value in strengths:
        if value < 0.2:
            buckets["<0.2"] += 1
        elif value < 0.5:
            buckets["0.2-0.5"] += 1
        elif value < 0.8:
            buckets["0.5-0.8"] += 1
        else:
            buckets[">=0.8"] += 1

    by_tier: Dict[str, Dict[str, Any]] = {}
    for memory in memories.values():
        tier = str(memory.get("tier") or "unknown")
        entry = by_tier.setdefault(tier, {"count": 0, "strength_sum": 0.0, "recalled": 0})
        entry["count"] += 1
        entry["strength_sum"] += float(memory.get("strength") or 0.0)
        if memory.get("recalls"):
            entry["recalled"] += 1
    for entry in by_tier.values():
        entry["mean_strength"] = round(entry.pop("strength_sum") / max(1, entry["count"]), 4)

    # 再激活：被归档的记忆有没有被重新召回
    archived = [m for m in memories.values() if str(m.get("tier")) == "archived"]
    archived_recalled = [m for m in archived if m.get("recalls")]
    recall_events = sum(len(m.get("recalls") or []) for m in memories.values())
    recalled_once = [m for m in memories.values() if m.get("recalls")]
    used_true = sum(
        1
        for m in memories.values()
        for recall in (m.get("recalls") or [])
        if recall.get("used")
    )

    # 衰减：按"记忆年龄"（当前周 − 创建周）分桶看平均强度
    weeks = [
        week
        for week in (_week_number(m.get("created_at")) for m in memories.values())
        if week is not None
    ]
    last_week = max(
        [
            week
            for week in (
                _week_number(t)
                for t in [
                    *(m.get("last_recalled_at") for m in memories.values()),
                    *(m.get("created_at") for m in memories.values()),
                ]
            )
            if week is not None
        ]
        or [0]
    )
    age_buckets: Dict[str, Dict[str, Any]] = {}
    for memory in memories.values():
        created = _week_number(memory.get("created_at"))
        if created is None or not last_week:
            continue
        age = max(0, last_week - created)
        key = "0-4 周" if age < 5 else ("5-9 周" if age < 10 else ">=10 周")
        entry = age_buckets.setdefault(key, {"count": 0, "strength_sum": 0.0})
        entry["count"] += 1
        entry["strength_sum"] += float(memory.get("strength") or 0.0)
    for entry in age_buckets.values():
        entry["mean_strength"] = round(entry.pop("strength_sum") / max(1, entry["count"]), 4)

    return {
        "available": True,
        "count": len(memories),
        "strength": {
            "mean": round(sum(strengths) / len(strengths), 4),
            "median": round(statistics.median(strengths), 4),
            "min": round(min(strengths), 4),
            "max": round(max(strengths), 4),
            "buckets": buckets,
        },
        "by_tier": by_tier,
        "archived": len(archived),
        "archived_recalled": len(archived_recalled),
        "reactivation_rate": (
            round(len(archived_recalled) / len(archived), 4) if archived else None
        ),
        "recall_events": recall_events,
        "recalled_memories": len(recalled_once),
        # 两个不同的"被用上"口径，必须分开报：
        # - `recall_used_share` 是影子召回里那条记录自带的 `used` 标记。阶段 2 的实现
        #   写死了 False（"影子召回不算被使用"，见 memory_store.py），所以它恒为 0，
        #   不能拿它当"检索后实际使用率"。
        # - `used_in_plan_total` 来自 `MEMORY_USED_IN_PLAN`（方法真正被采用时给来源记忆
        #   记的一笔），这才是设计正文里那个指标现在的落点。
        "recall_used_share": round(used_true / recall_events, 4) if recall_events else None,
        "used_in_plan_total": sum(
            int(m.get("used_in_plan_count") or 0) for m in memories.values()
        ),
        "used_in_plan_memories": sum(
            1 for m in memories.values() if int(m.get("used_in_plan_count") or 0) > 0
        ),
        "reinforcements": sum(int(m.get("reinforcements") or 0) for m in memories.values()),
        "merged": sum(1 for m in memories.values() if m.get("merged_from")),
        "protected": sum(1 for m in memories.values() if m.get("protected")),
        "by_age": {key: age_buckets[key] for key in sorted(age_buckets)},
    }


def _skill_growth(persona_dir: Path) -> dict[str, Any]:
    """技能成长的**曲线**，而不是首尾两个数（阶段 6 长期观察的读数）。

    为什么要看曲线：封顶规则的作用是"练到一定程度就不太涨了"，如果只看总量，
    分不清"通胀被压住"和"角色干脆停止成长"。这里按 5 周一个桶给出：
    该桶发放的点数、当时已出现的技能个数、以及该桶里有多少条活动完全没有增益。
    """
    buckets: dict[str, dict[str, Any]] = {}
    skills_seen: set[str] = set()
    for row in read_jsonl(persona_dir / "activity.jsonl"):
        time_str = str(row.get("time") or "")
        parts = time_str.split("-")
        if len(parts) < 2 or not parts[0].startswith("Y") or not parts[1].startswith("W"):
            continue
        try:
            index = int(parts[0][1:]) * 10 + int(parts[1][1:])
        except ValueError:
            continue
        bucket = f"{(index - 1) // 5 * 5 + 1:02d}"
        gains = (row.get("outcome") or {}).get("delta_skills") or {}
        positive = {
            str(k): float(v)
            for k, v in gains.items()
            if isinstance(v, (int, float)) and float(v) > 0
        }
        skills_seen.update(positive)
        entry = buckets.setdefault(
            bucket, {"activities": 0, "points": 0.0, "activities_without_gain": 0}
        )
        entry["activities"] += 1
        entry["points"] += sum(positive.values())
        if not positive:
            entry["activities_without_gain"] += 1
        entry["distinct_skills"] = len(skills_seen)
    for entry in buckets.values():
        entry["points"] = round(entry["points"], 2)
    return {
        "per_5_weeks": {key: buckets[key] for key in sorted(buckets)},
        "skills_total": len(skills_seen),
    }


def _capability_cap(run_dir: Path, persona_dir: Path) -> dict[str, Any]:
    """按已练习能力封顶增益的读数（阶段 6 第二步，`capability_gain.py`）。

    规则如果生效却看不见，就等于没有规则：这里读 `CAPABILITY_GAIN_CAPPED` 事件，
    给出触发次数、落在哪个档、被压掉的点数和涉及的技能。
    """
    events = read_jsonl(persona_dir / "cognition" / "capability_events.jsonl")
    capped = [e for e in events if str(e.get("type") or "") == "CAPABILITY_GAIN_CAPPED"]
    config = read_json(run_dir / "config.json", {}) or {}
    cognition = ((config.get("world") or {}).get("cognition")) or {}
    by_rule: dict[str, int] = {}
    by_skill: dict[str, int] = {}
    removed = 0.0
    for event in capped:
        by_rule[str(event.get("rule"))] = by_rule.get(str(event.get("rule")), 0) + 1
        by_skill[str(event.get("skill"))] = by_skill.get(str(event.get("skill")), 0) + 1
        try:
            removed += float(event.get("delta_before") or 0.0) - float(
                event.get("delta_after") or 0.0
            )
        except (TypeError, ValueError):
            continue
    # **不变式**才是权威读数：事件只是记录，而"练到这个程度就不该再涨这么多"这条规则
    # 可以直接从账本验证——按"活动发生前"的练习量估能力（折扣取保守下界 0.5），
    # 打开时任何一条记录都不该越界。
    # 这条检查抓出过一个真错误：封顶确实在生效，但幂等键不唯一，2 年运行实际封顶
    # 41 / 11 / 12 次却只写下 1 / 4 / 1 条事件（见 known-issues KI-23）。
    from src.agents.cognition.proficiency import proficiency_from_practices

    rules = []
    if bool(cognition.get("capability_gain_cap", False)):
        from src.agents.cognition.capability_gain import cap_rules

        rules = list(cap_rules(config))
    practices: dict[str, int] = {}
    violations: list[dict[str, Any]] = []
    expected_caps = 0
    for row in read_jsonl(persona_dir / "activity.jsonl"):
        gains = (row.get("outcome") or {}).get("delta_skills") or {}
        if not isinstance(gains, dict):
            continue
        for skill, value in gains.items():
            try:
                delta = float(value)
            except (TypeError, ValueError):
                continue
            # 保守下界：折扣最差也是 0.5，所以真实能力 >= 0.5 × proficiency
            capability = proficiency_from_practices(practices.get(str(skill), 0)) * 0.5
            rule = next((r for r in rules if capability >= r.threshold), None)
            if rule is not None and delta > rule.max_gain:
                # 只有**越界**才说明"这里本该封顶"。开关关着时这一列是反事实读数
                # （本来会被压掉多少次），开关打开时它必须恒为 0——封顶后的值看起来
                # 就是一次普通的低增益，从账本上分辨不出来，所以"触发了多少次"
                # 只能读 `CAPABILITY_GAIN_CAPPED` 事件。
                if bool(cognition.get("capability_gain_cap", False)):
                    violations.append(
                        {
                            "time": str(row.get("time") or ""),
                            "skill": str(skill),
                            "delta": delta,
                            "capability_floor": round(capability, 4),
                            "rule": rule.name,
                        }
                    )
                else:
                    expected_caps += 1
            if delta > 0:
                practices[str(skill)] = practices.get(str(skill), 0) + 1

    return {
        "enabled": bool(cognition.get("capability_gain_cap", False)),
        "events": len(capped),
        "by_rule": by_rule,
        "by_skill": by_skill,
        "points_removed": round(removed, 2),
        # 开关**关着**时的反事实读数：本来会被压掉多少次。开关打开时恒为 0
        # （封顶后的记录不再越界），"触发了多少次"看 `events`。
        "would_be_capped_when_off": expected_caps,
        "invariant_violations": len(violations),
        "violations": violations[:10],
    }


def _proficiency_projection(
    run_dir: Path,
    persona_dir: Path,
    state_skills: dict[str, float],
) -> dict[str, Any]:
    """Phase 6 shadow: proficiency and effective capability derived from the ledger.

    The projection is rebuilt here rather than read from the view, so the numbers
    are the *current* code's, not the ones written the day the run happened (the
    same reasoning as the reward-caliber recomputation above).

    The comparison with `state.skills` is deliberately rank-based: `state.skills`
    carries each persona's initial endowment (100-260 points for the starting
    skills) while the projection counts only what the two years of practice
    granted, so the absolute numbers are not on the same scale and comparing them
    directly would be meaningless.

    Alias families come from `_world_aliases(run_dir)` (see its docstring): the
    first version of this block built the projection from a stand-in without a
    usable `world`, so it silently produced the *unfolded* reading on a run that
    does have alias families.
    """
    from src.agents.cognition.proficiency import build_projection

    class _DM:  # minimal stand-in: the projection only reads root + char
        def __init__(self, root: Path, char: str) -> None:
            self.root = root
            self.char = char

    try:
        projection = build_projection(
            _DM(persona_dir, persona_dir.name), families=_world_aliases(run_dir)
        )
    except Exception as exc:  # pragma: no cover - a bad ledger must not kill the audit
        return {"available": False, "reason": repr(exc)}

    derived = {
        skill: entry["proficiency"]
        for skill, entry in (projection.get("skills") or {}).items()
        if entry.get("practices")
    }
    common = [skill for skill in derived if skill in state_skills]
    spearman = None
    if len(common) >= 3:
        order_derived = sorted(common, key=lambda k: (derived[k], k))
        order_state = sorted(common, key=lambda k: (state_skills[k], k))
        rank_d = {k: i for i, k in enumerate(order_derived)}
        rank_s = {k: i for i, k in enumerate(order_state)}
        n = len(common)
        squared = sum((rank_d[k] - rank_s[k]) ** 2 for k in common)
        spearman = round(1 - (6 * squared) / (n * (n * n - 1)), 4)

    ranked = sorted(
        (projection.get("skills") or {}).items(),
        key=lambda kv: -kv[1].get("effective_capability", 0.0),
    )[:5]
    return {
        "available": True,
        "stats": projection.get("stats") or {},
        "practised_skills": len(derived),
        "compared_with_state": len(common),
        "rank_spearman": spearman,
        # 归并前后（用户决策 2026-09-27：家族归并只作用于投影统计）：不归并时
        # `skill_aliases.json` 定义过的写法各自成一族，proficiency 被切碎。两侧读数
        # 都给出，用来判断"要不要真去改 state.skills 的键"。
        "family_folding": _family_folding(run_dir, persona_dir, projection, derived),
        "top_by_capability": [
            {
                "skill_id": skill,
                "practice_units": entry.get("practice_units"),
                "proficiency": entry.get("proficiency"),
                "proficiency_points": entry.get("proficiency_points"),
                "effective_capability": entry.get("effective_capability"),
                "effective_capability_points": entry.get("effective_capability_points"),
                "state_skills": state_skills.get(skill),
                "coverage": (entry.get("factors") or {}).get("coverage"),
                "selection_accuracy": (entry.get("factors") or {}).get("selection_accuracy"),
                "verified_reliability": (entry.get("factors") or {}).get("verified_reliability"),
            }
            for skill, entry in ranked
        ],
    }


def _family_folding(
    run_dir: Path,
    persona_dir: Path,
    projection: dict[str, Any],
    folded: dict[str, float],
) -> dict[str, Any]:
    """同一本账在"归并"与"不归并"两种口径下的差别。"""
    from src.agents.cognition.proficiency import build_projection
    from src.agents.cognition.skills import SkillFamilies

    class _DM:  # minimal stand-in, same as above
        def __init__(self, root: Path, char: str) -> None:
            self.root = root
            self.char = char

    try:
        unfolded = build_projection(
            _DM(persona_dir, persona_dir.name), families=SkillFamilies()
        )
    except Exception as exc:  # pragma: no cover - defensive
        return {"available": False, "reason": repr(exc)}

    raw = {
        skill: entry["proficiency"]
        for skill, entry in (unfolded.get("skills") or {}).items()
        if entry.get("practices")
    }
    merged = (projection.get("stats") or {}).get("merged_families") or {}
    return {
        "available": True,
        "folded_families": len(folded),
        "unfolded_families": len(raw),
        "merged_family_count": len(merged),
        "merged_families": merged,
        # 每多一个写法就多一个"永远练不满"的族：这个数就是归并救回来的能力分辨率。
        "names_folded_away": sum(len(names) - 1 for names in merged.values()),
    }


def _skill_fragmentation(run_dir: Path, personas: list[dict[str, Any]]) -> dict[str, Any]:
    """Cross-character skill-name fragmentation (design §3.2, phase-5 follow-up).

    Within one character the method→skill mapping turned out to be clean: across
    two acceptance runs every method's skill was one the character already had.
    The fragmentation is *between* characters and comes from the environment
    model naming the same capability differently (`长跑` / `长跑耐力` /
    `长跑与体能`). This is a side mapping for measurement — the world supplies the
    families in `skill_aliases.json` and nothing here rewrites a skill, an event
    or a method.
    """
    from src.agents.cognition.skills import SkillFamilies, similar_names
    from src.agents.cognition.proficiency import UNMAPPED_SKILL

    families = _world_aliases(run_dir)

    per_persona: dict[str, list[str]] = {}
    sentinel_methods = 0
    for persona in personas:
        names = set()
        families_seen = (persona.get("skills") or {}).get("last") or []
        names.update(str(n).strip() for n in families_seen if str(n).strip())
        for method in ((persona.get("methodologies") or {}).get("samples") or []):
            skill = str(method.get("skill_id") or "").strip()
            if skill == UNMAPPED_SKILL:
                # `unmapped` 是方法抽取在没有技能可用时写的哨兵值，不是技能名。
                # 投影已经排除它；这里也不该把它算成一个"技能"（否则它会出现在
                # `unmapped_skills` 和相似名建议里，看起来像真的碎片）。
                sentinel_methods += 1
                continue
            if skill:
                names.add(skill)
        per_persona[persona["persona"]] = sorted(names)

    all_names = sorted({name for names in per_persona.values() for name in names})
    groups = families.fragments(all_names)
    unmapped = [name for name in all_names if name not in families.family_of]
    # 归并是否真的改变了单个角色的投影：只有**同一个角色**用过多个写法时才会。
    # 这条读数专门用来解释"归并读了半天却什么都没变"——跨角色漂移不会进入
    # 单角色的 proficiency。
    intra_persona = {
        name: families.fragments(names)
        for name, names in sorted(per_persona.items())
        if families.fragments(names)
    }
    return {
        "alias_file": bool(families.members),
        "families_defined": len(families.members),
        "skills_observed": len(all_names),
        "fragmented_families": groups,
        "fragmented_family_count": len(groups),
        "skills_per_persona": per_persona,
        "unmapped_skills": unmapped,
        "unmapped_sentinel_methods": sentinel_methods,
        "intra_persona_fragmentation": intra_persona,
        "intra_persona_fragmented_count": sum(len(v) for v in intra_persona.values()),
        # Naming a capability one way in one persona and another way in the next
        # is the one thing the design document's metric can act on, so surface a
        # concrete suggestion instead of a guess.
        "suggestions": {
            name: similar_names(name, all_names)
            for name in unmapped
            if similar_names(name, all_names)
        },
    }


def _lifecycle_report(
    methods: dict[str, dict[str, Any]],
    capability_events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Phase 5: specialised / archived / versioned methods.

    `specialized` and `archived` were declared in the model from phase 1 and had
    no writer until now, so this block is also the check that they are reachable
    at all rather than dead vocabulary.
    """
    archived = [row for row in capability_events if row.get("type") == "METHOD_ARCHIVED"]
    refined = [row for row in capability_events if row.get("type") == "METHOD_REFINED"]
    by_status = Counter(str(m.get("status") or "") for m in methods.values())
    versioned = sorted(
        (str(mid), int(m.get("version") or 1))
        for mid, m in methods.items()
        if int(m.get("version") or 1) > 1
    )
    return {
        "by_status": dict(sorted(by_status.items())),
        "archived": {
            "events": len(archived),
            "reasons": dict(Counter(str(row.get("reason") or "") for row in archived)),
            "methods": sorted({str(row.get("method_id") or "") for row in archived}),
        },
        "refined": {
            "events": len(refined),
            "methods": sorted({str(row.get("method_id") or "") for row in refined}),
        },
        "versioned_methods": dict(versioned),
        # A method that is good *here* rather than good overall (reward design §3.3)
        "specialized_methods": sorted(
            mid for mid, m in methods.items() if str(m.get("status") or "") == "specialized"
        ),
    }


# ---------------------------------------------------------------------------
# Acceptance gates (report §7.4)
#
# The authoritative inventory is built in `acceptance_gates()` below: 13 always-on
# gates (phases -1..3 plus the observation gates), 5 phase-4 gates when hints are
# on, 2 phase-4.5 gates when the lesson bridge is on, and 6 phase-5 gates
# (declines, the KI-8 shadow distribution, the lifecycle). A second, hand-written
# list used to live here and had silently drifted from the real one; the run's
# JSON report is the place to read the current inventory.
# ---------------------------------------------------------------------------

def _simulated(report: Dict[str, Any]) -> list[dict[str, Any]]:
    return [p for p in report.get("personas", []) if p.get("simulated")]


def _worst(value_pairs: list[tuple[str, float | None]], *, higher_is_better: bool):
    usable = [(name, value) for name, value in value_pairs if value is not None]
    if not usable:
        return None, None
    chosen = max(usable, key=lambda kv: kv[1]) if higher_is_better else min(usable, key=lambda kv: kv[1])
    return chosen[0], chosen[1]


def acceptance_gates(report: Dict[str, Any]) -> Dict[str, Any]:
    """Turn the acceptance report into pass/fail gates (see docs §7.4).

    Each gate records the worst simulated persona plus the raw value, so a
    failing run points at *who* failed, not just what failed.
    """
    personas = _simulated(report)
    gates: Dict[str, Any] = {}

    def add(name: str, target: str, passed: bool, persona: str | None, value: Any) -> None:
        gates[name] = {
            "target": target,
            "value": value,
            "worst_persona": persona,
            "pass": bool(passed),
        }

    name, value = _worst(
        [(p["persona"], (p.get("memory") or {}).get("entity_drop_rate")) for p in personas],
        higher_is_better=False,
    )
    add("entity_drop_rate", "< 0.4", value is not None and value < 0.4, name, value)

    motifs_with_ideas = sorted(
        {motif for p in personas for motif in ((p.get("ideas") or {}).get("by_motif") or {})}
    )
    add(
        "motifs_with_ideas",
        ">= 4",
        len(motifs_with_ideas) >= 4,
        None,
        {"count": len(motifs_with_ideas), "motifs": motifs_with_ideas},
    )

    name, value = _worst(
        [
            (p["persona"], float(len((p.get("ideas") or {}).get("near_duplicate_clusters") or [])))
            for p in personas
        ],
        higher_is_better=False,
    )
    add("duplicate_idea_clusters", "== 0", value == 0, name, value)

    def _idea_structure_complete(persona: Dict[str, Any]) -> float | None:
        methods = persona.get("methodologies") or {}
        structures = methods.get("structure") or {}
        idea_methods = [
            method_id
            for method_id in structures
            if str(((methods.get("samples") or [{}])[0].get("source_type")) or "") == "idea_conversion"
        ]
        # `by_source_type` counts are authoritative; rebuild the id list from the
        # per-method structure map using the samples when available.
        converted_total = int((methods.get("by_source_type") or {}).get("idea_conversion") or 0)
        if not converted_total:
            return 1.0
        converted_complete = 0
        for entry in methods.get("samples") or []:
            if str(entry.get("source_type")) != "idea_conversion":
                continue
            if (entry.get("structure") or {}).get("complete"):
                converted_complete += 1
        if not idea_methods and converted_complete == 0:
            return None
        return round(converted_complete / converted_total, 4)

    name, value = _worst(
        [(p["persona"], _idea_structure_complete(p)) for p in personas],
        higher_is_better=True,
    )
    add(
        "idea_method_structure_complete",
        "== 1.0",
        value is not None and value >= 1.0,
        name,
        value,
    )

    name, value = _worst(
        [
            (p["persona"], (p.get("methodologies") or {}).get("outcomes", {}).get("positive_share"))
            for p in personas
        ],
        higher_is_better=True,
    )
    add(
        "positive_reward_share",
        "0.5 - 0.8",
        value is not None and 0.5 <= value <= 0.8,
        name,
        value,
    )

    name, value = _worst(
        [
            (p["persona"], (p.get("methodologies") or {}).get("selection", {}).get("explored_repeat_share"))
            for p in personas
        ],
        higher_is_better=False,
    )
    add("explored_repeat_share", "< 0.2", value is not None and value < 0.2, name, value)

    name, value = _worst(
        [
            (
                p["persona"],
                float(len((p.get("methodologies") or {}).get("selection", {}).get("methods_never_selected") or [])),
            )
            for p in personas
        ],
        higher_is_better=False,
    )
    add("methods_never_selected", "<= 2", value is not None and value <= 2, name, value)

    # --- idea throughput (KI-15) -----------------------------------------
    weeks_in_run = int(((report.get("config") or {}).get("weeks")) or 0)
    expected_weeks = weeks_in_run if weeks_in_run > 0 else 10
    name, value = _worst(
        [
            (p["persona"], float(len((p.get("ideas") or {}).get("weeks_with_idea") or [])))
            for p in personas
        ],
        higher_is_better=True,
    )
    add(
        "idea_weeks_covered",
        f">= {max(1, int(expected_weeks * 0.7))} of {expected_weeks}",
        value is not None and value >= expected_weeks * 0.7,
        name,
        value,
    )

    no_budget_weeks = sum(
        int(((p.get("ideas") or {}).get("skipped_by_reason") or {}).get("no_budget") or 0)
        for p in personas
    )
    add(
        "idea_skipped_no_budget",
        "== 0",
        no_budget_weeks == 0,
        None,
        no_budget_weeks,
    )

    name, value = _worst(
        [
            (p["persona"], (p.get("ideas") or {}).get("rejected_below_min_potential_share"))
            for p in personas
        ],
        higher_is_better=False,
    )
    if value is None:
        # No rejections at all: nothing was wasted on an unscorable candidate.
        value = 0.0
    add("idea_below_min_potential_share", "< 0.6", value < 0.6, name, value)

    duplicate_rows = int((report.get("ledger") or {}).get("duplicate_row_count") or 0)
    add("ledger_duplicate_rows", "== 0", duplicate_rows == 0, None, duplicate_rows)

    name, value = _worst(
        [
            (
                p["persona"],
                float(len((p.get("methodologies") or {}).get("structure_incomplete") or [])),
            )
            for p in personas
        ],
        higher_is_better=False,
    )
    add("structure_incomplete_methods", "== 0", value == 0, name, value)

    name, value = _worst(
        [
            (p["persona"], (p.get("reflection_parse") or {}).get("fallback_share"))
            for p in personas
        ],
        higher_is_better=True,
    )
    add("reflection_parse_fallback", "< 0.3", value is not None and value < 0.3, name, value)

    # -- phase 4 gates (only meaningful when hints are switched on) ---------
    cognition_cfg = (report.get("config") or {}).get("cognition") or {}
    hints_on = bool(cognition_cfg.get("method_hints"))
    if hints_on:
        persona_hints = [
            (p, (p.get("method_hints") or {})) for p in personas
        ]
        offer_events = sum((h.get("offers") or {}).get("events") or 0 for _, h in persona_hints)
        add(
            "hint_offers_recorded",
            "> 0",
            offer_events > 0,
            None,
            offer_events,
        )
        # An offer the character never sees is not a passive hint, it is noise;
        # and an offer nothing ever takes up means the menu is unusable.
        name, value = _worst(
            [
                (p["persona"], (h.get("adoptions") or {}).get("adoption_rate"))
                for p, h in persona_hints
            ],
            higher_is_better=True,
        )
        add("hint_adoption_rate", "> 0", value is not None and value > 0, name, value)

        # An adoption that produced no practice is a legitimate behavioural fact
        # (the plan did not materialise) — report the rate, and fail only when the
        # loop never closes at all, which would mean the matcher is broken.
        total_adoptions = sum((h.get("adoptions") or {}).get("events") or 0 for _, h in persona_hints)
        total_real = sum(
            (h.get("practice") or {}).get("real_outcomes") or 0 for _, h in persona_hints
        )
        practised = [
            (p["persona"], (h.get("adoptions") or {}).get("events") or 0,
             (h.get("practice") or {}).get("real_outcomes") or 0)
            for p, h in persona_hints
        ]
        lowest = _worst(
            [
                (name, (real / adoptions) if adoptions else None)
                for name, adoptions, real in practised
            ],
            higher_is_better=False,
        )
        add(
            "hint_adoption_practice_rate",
            "> 0 (run aggregate)",
            total_adoptions > 0 and total_real > 0,
            lowest[0],
            round(total_real / total_adoptions, 4) if total_adoptions else None,
        )

        # Invariant #6 holds only if nothing moves without practice, no activity
        # is counted twice, and adopted methods light up their source memories.
        leaked = sum(
            len((h.get("practice") or {}).get("free_reinforcement") or [])
            + len((h.get("practice") or {}).get("real_updates_without_practice") or [])
            + len((h.get("practice") or {}).get("double_counted_activities") or [])
            + len((h.get("adoptions") or {}).get("unoffered") or [])
            + len((h.get("used_in_plan") or {}).get("adoptions_missing_used_in_plan") or [])
            for _, h in persona_hints
        )
        add("hint_evidence_leaks", "== 0", leaked == 0, None, leaked)

        max_chars = cognition_cfg.get("method_hints_max_chars")
        oversize = max(
            [
                int(((h.get("offers") or {}).get("block_chars_max") or 0))
                for _, h in persona_hints
            ]
            or [0]
        )
        add(
            "hint_block_within_budget",
            f"<= {max_chars}",
            bool(max_chars) and oversize <= int(max_chars),
            None,
            oversize,
        )

    # -- phase 4.5 gates (only meaningful when the lesson bridge is on) -----
    if bool(cognition_cfg.get("lesson_ingest")):
        lesson_total = sum(int(((p.get("lessons") or {}).get("memories") or 0)) for p in personas)
        add("lesson_memories_ingested", "> 0", lesson_total > 0, None, lesson_total)

        # The loop only counts as closed when a method grown from the
        # character's own lesson actually reached the menu and got practised;
        # offered-but-never-practised means the bridge stops one hop short.
        chain_offers = sum(
            int((((p.get("lessons") or {}).get("chain") or {}).get("offers") or 0))
            for p in personas
        )
        chain_practices = sum(
            int((((p.get("lessons") or {}).get("chain") or {}).get("practices") or 0))
            for p in personas
        )
        add(
            "lesson_loop_closed",
            "practised >= offered > 0, or nothing offered yet",
            chain_offers == 0 or chain_practices > 0,
            None,
            f"offers={chain_offers}, practices={chain_practices}",
        )

    # -- phase 5 gates (the character's answer, and the KI-8 shadow) --------
    choices = {p["persona"]: (p.get("method_choices") or {}) for p in personas}
    if hints_on:
        decline_events = sum(
            int((c.get("declines") or {}).get("events") or 0) for c in choices.values()
        )
        add(
            "declines_recorded",
            "> 0 (the ignoring half is observable)",
            decline_events > 0,
            None,
            decline_events,
        )

    shadow_reports = [
        c.get("reward_baseline_shadow") or {} for c in choices.values()
    ]
    shadow_ready = [r for r in shadow_reports if r.get("available")]
    if shadow_ready:
        total = sum(int(r.get("outcomes") or 0) for r in shadow_ready)
        negative = sum(int((r.get("shadow") or {}).get("negative") or 0) for r in shadow_ready)
        # 深度判据要按**该阴影口径自己的刻度**算。第一版拿 `below_minus_0_1`
        # 去对 `quality` 口径，而它的满量程只有 signed 的 1/4（`BASELINE_FULL_SCALE_SIGNED`
        # = 0.25），于是 3 年运行里 min = −0.053 被判成"深度不足"——其实它在 quality 刻度上
        # 已经是很深的一条（等价于 signed 的 −0.21）。见 docs/project-acceptance-3y.md。
        depth_scale = {
            "signed": -0.1,
            "quality": -0.025,
            "off": -0.1,
        }
        deep = 0
        worst_min = 0.0
        for report in shadow_ready:
            caliber = str(report.get("shadow_caliber") or "signed")
            threshold = depth_scale.get(caliber, -0.1)
            values = (report.get("shadow") or {}).get("values") or []
            deep += sum(1 for value in values if float(value) <= threshold)
            if values:
                worst_min = min(worst_min, min(float(v) for v in values))
        if not any((r.get("shadow") or {}).get("values") for r in shadow_ready):
            # 老报告没有逐条值：退回原来的读数（但对齐到该口径的阈值）
            deep = sum(
                int((r.get("shadow") or {}).get("below_minus_0_1") or 0)
                for r in shadow_ready
            )
            worst_min = min(
                float((r.get("shadow") or {}).get("min") or 0.0) for r in shadow_ready
            )
        add(
            "reward_baseline_shadow_available",
            "every simulated persona scored its outcomes the centred way",
            len(shadow_ready) == len(personas),
            None,
            f"{len(shadow_ready)}/{len(personas)}",
        )
        add(
            "reward_baseline_shadow_signs",
            "0 < negative outcomes < all outcomes",
            0 < negative < total,
            None,
            f"negative={negative}/{total}",
        )
        add(
            "reward_baseline_shadow_depth",
            ">= 1 outcome at or below -0.1 (design §5: the distribution crosses 0)",
            deep >= 1,
            None,
            {"below_minus_0_1": deep, "min": round(worst_min, 4)},
        )
        add(
            "baseline_shadow_matches_live_replay",
            "replaying the live reward reproduces the recorded values",
            all(
                (r.get("live_replay_matches_recorded") or {}).get("exact")
                for r in shadow_ready
            ),
            None,
            [
                (r.get("live_replay_matches_recorded") or {})
                for r in shadow_ready
            ],
        )

    # -- phase 5 skill-family gate -----------------------------------------
    fragmentation = report.get("skill_fragmentation") or {}
    if int(fragmentation.get("fragmented_family_count") or 0) > 0:
        add(
            "skill_alias_file_present",
            "a world with synonym fragments ships skill_aliases.json",
            bool(fragmentation.get("alias_file")),
            None,
            {
                "fragmented_families": fragmentation.get("fragmented_family_count"),
                "alias_file": bool(fragmentation.get("alias_file")),
            },
        )

    # -- phase 5 lifecycle gates (only when the lifecycle writer is on) -----
    if bool(cognition_cfg.get("method_lifecycle")):
        lifecycle_reports = [
            c.get("lifecycle") or {} for c in choices.values()
        ]
        specialised = sum(
            len(r.get("specialized_methods") or []) for r in lifecycle_reports
        )
        archived = sum(
            int((r.get("archived") or {}).get("events") or 0) for r in lifecycle_reports
        )
        refined = sum(
            int((r.get("refined") or {}).get("events") or 0) for r in lifecycle_reports
        )
        add(
            "method_lifecycle_reachable",
            "specialised / archived / refined are all reachable, not dead vocabulary",
            (specialised + archived + refined) > 0
            or all(
                int((r.get("by_status") or {}).get("validated") or 0) == 0
                for r in lifecycle_reports
            ),
            None,
            {
                "specialized": specialised,
                "archived": archived,
                "refined": refined,
            },
        )

    # Gates whose failure is a known, accepted state rather than a regression.
    gate_notes = {
        "positive_reward_share": (
            "KI-8: the baseline reward architecture is designed but still switched "
            "off (use_baseline=False), so rewards stay positive by design until "
            "phase 4 turns it on (docs/methodology-reward-design.md)."
        ),
        "capability_cap_reachable": (
            "Only when world.cognition.capability_gain_cap is on: with the cap off "
            "there is nothing to reach, so the gate is omitted rather than passed."
        ),
        "ledger_duplicate_rows": (
            "KI-11: user decision — not fixed (duplicate rows appear when the same "
            "run is started twice); tracked here for information only."
        ),
        "methods_never_selected": (
            "Only meaningful on a full year: in a short run most methods have not "
            "had a chance to be selected yet."
        ),
        "idea_weeks_covered": (
            "A short run has few weeks: the gate is scaled to the configured week "
            "count (>= 70%)."
        ),
        "motifs_with_ideas": (
            "A 10-week target: a 3-week run legitimately sees fewer motifs (and "
            "short runs also have fewer memories to relate)."
        ),
    }
    for gate_name, note in gate_notes.items():
        if gate_name in gates:
            gates[gate_name]["note"] = note

    # 阶段 6：开关打开就必须留下痕迹。声明了却一次都不触发的规则，与没实现是一回事
    # （KI-15 / 生命周期可达性是同一个道理）。
    cap_reports = [
        (persona.get("capability_cap") or {}) for persona in report.get("personas") or []
    ]
    if any(cap.get("enabled") for cap in cap_reports):
        capped_events = sum(int(cap.get("events") or 0) for cap in cap_reports)
        violations = sum(int(cap.get("invariant_violations") or 0) for cap in cap_reports)
        would_be = sum(
            int(cap.get("would_be_capped_when_off") or 0) for cap in cap_reports
        )
        # 判据是**不变式**：规则打开时，账本里不该出现任何越界的增益；
        # 同时规则必须真的触发过（"声明了却没有写入者"是 KI-15 那一类错误）。
        # 第一版门禁只看"事件数 > 0"，实际封顶 64 次只记下 6 条也照样通过（KI-23）。
        add(
            "capability_cap_reachable",
            "capability_gain_cap is on, the rule fired, and no recorded gain "
            "exceeds its cap",
            capped_events > 0 and violations == 0,
            None,
            {
                "recorded_events": capped_events,
                "violations": violations,
                "would_be_capped_when_off": would_be,
            },
        )

    failed = [g for g, spec in gates.items() if not spec["pass"]]
    return {
        "gates": gates,
        "passed": len(gates) - len(failed),
        "total": len(gates),
        "failed": failed,
    }


# ---------------------------------------------------------------- markdown ---


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value) if value else "-"
    if isinstance(value, dict):
        return ", ".join(f"{k}={v}" for k, v in value.items()) if value else "-"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    cfg = report["config"]
    lines = [
        f"# Cognition run audit · {report['run']}",
        "",
        f"- world: `{cfg['world']}` · {cfg['years']} year × {cfg['weeks']} weeks · language `{cfg['language']}`",
        f"- models: role=`{cfg['role_model']}` god=`{cfg['god_model']}` fallback=`{cfg['fallback_model']}`",
        f"- cognition: `{json.dumps(cfg['cognition'], ensure_ascii=False, sort_keys=True)}`",
        f"- personas: {report['personas_simulated']}/{report['personas_total']} simulated"
        + (
            f" (not simulated: {', '.join(report['personas_not_simulated'])})"
            if report["personas_not_simulated"]
            else ""
        ),
        f"- ledger: {report['ledger']['records']} records, {report['ledger']['unique_ids']} unique ids, "
        f"{len(report['ledger']['invalid_jsonl'])} invalid files, "
        f"{report['ledger'].get('duplicate_id_count', 0)} duplicate ids / "
        f"{report['ledger'].get('duplicate_row_count', 0)} duplicate rows",
        "",
        "## Red flags",
        "",
    ]
    flags = report.get("red_flags") or []
    lines += [f"- {flag}" for flag in flags] if flags else ["- none"]
    lines.append("")

    gate_report = report.get("acceptance_gates") or {}
    if gate_report:
        lines += [
            f"## Acceptance gates ({gate_report.get('passed')}/{gate_report.get('total')} passed)",
            "",
            "| gate | target | value | worst persona | pass |",
            "|---|---|---|---|---|",
        ]
        for gate_name, spec in (gate_report.get("gates") or {}).items():
            lines.append(
                f"| {gate_name} | {spec['target']} | {_fmt(spec['value'])} | "
                f"{spec['worst_persona'] or '-'} | {'✅' if spec['pass'] else '❌'} |"
            )
        lines.append("")

    for persona in report["personas"]:
        if not persona.get("simulated"):
            lines += [f"## {persona['persona']}", "", "_persona present but never simulated_", ""]
            continue
        memory = persona["memory"]
        ideas = persona["ideas"]
        methods = persona["methodologies"]
        selection = methods["selection"]
        outcomes = methods["outcomes"]
        lines += [
            f"## {persona['persona']}",
            "",
            f"- activities: {_fmt(persona['activities']['by_type'])} · weeks {persona['activities']['by_week']}"
            f" · activity_id {persona['activities']['activity_ids']}/{persona['activities']['count']}",
            f"- skills: {persona['skills']['count_first']} → {persona['skills']['count_last']}"
            f" (gained: {_fmt(persona['skills']['gained'])}, Δsum {persona['skills']['delta_sum']})",
            f"- weekly diary: {persona['weekly_diary']['count']} entries, "
            f"{persona['weekly_diary']['bad_count']} bad · scratchpad {persona['weekly_diary']['scratchpad_entries']}",
            f"- generation: {persona['generation']['records']} records, "
            f"{persona['generation']['rejected']} rejected, "
            f"{persona['generation']['bad_final_answer_count']} bad finals",
            f"- cognition events: {persona['event_contract']['cognition_events']} "
            f"{_fmt(persona['event_contract']['by_stream'])}",
            f"- memory: {memory['count']} items {_fmt(memory['by_kind'])} · tier {_fmt(memory['by_tier'])}"
            f" · status {_fmt(memory['by_status'])} · protected {memory['protected']}",
            f"- memory hygiene: entity drop rate {_fmt(memory['entity_drop_rate'])}"
            f" ({memory['accepted_entities']} accepted / {memory['dropped_entities']} dropped"
            f" / {memory.get('soft_entities', 0)} soft),"
            f" items without entities {memory['memories_without_entities']},"
            f" obstacles {memory['with_obstacles']}, resources {memory['with_resources']}",
            f"- recall: {memory['recall_events']} recalls, {memory['recall_used_true']} marked used-in-plan;"
            f" strength updates {memory['strength_updates']}",
            f"- relations: {persona['relations']['count']} edges {_fmt(persona['relations']['by_type'])}"
            f" (ratio {_fmt(persona['relations']['edge_to_memory_ratio'])}), bad refs {len(persona['relations']['bad_references'])}",
            f"- retrieval shadow: ratio {_fmt(persona['retrieval_shadow']['token_ratio'])},"
            f" cold reactivated {persona['retrieval_shadow']['cold_reactivated']}",
            f"- impressions: canonical {persona['impressions']['canonical']}, "
            f"aliases {persona['impressions']['aliases']} "
            f"(learned {persona['impressions']['aliases_learned']}), "
            f"soft {persona['impressions']['soft_entities']}",
            f"- ideas: {ideas['count']} ({_fmt(ideas['by_motif'])}) · converted {ideas['converted']}"
            f" · refined {ideas.get('refined', 0)} · weeks with an idea "
            f"{len(ideas.get('weeks_with_idea') or [])}"
            f" · skipped {ideas.get('skipped', 0)} {_fmt(ideas.get('skipped_by_reason'))}"
            f" · rejected-by-motif {_fmt(ideas.get('rejected_by_motif'))}"
            f" · rejected {ideas['rejected']} · goal-linked {_fmt(ideas['goal_link_rate'])}"
            f" · potential {_fmt(ideas['potential_range'])}",
            f"- idea clusters (same insight, reworded): {len(ideas['insight_clusters'])}",
        ]
        for cluster in ideas["insight_clusters"][:3]:
            lines.append(
                f"  - {len(cluster['ideas'])} ideas {cluster['weeks']} → "
                f"{len(cluster['converted_methods'])} methods"
            )
        lines += [
            f"- methods: {methods['methods']} {_fmt(methods['by_status'])} · sources {_fmt(methods['by_source_type'])}"
            f" · incomplete structure {len(methods.get('structure_incomplete') or [])}"
            f" · complete structure {_fmt(methods['complete_structure_share'])}",
            f"- method structure by source: {_fmt({k: v['avg_steps'] for k, v in methods['structure_by_source'].items()})} (avg steps)",
            f"- selection: {selection['events']} events, top1 share {_fmt(selection['top1_share'])},"
            f" distinct {selection['distinct_methods_selected']}, never selected {len(selection['methods_never_selected'])},"
            f" explored share {_fmt(selection['explored_share'])}, explored-repeat {_fmt(selection['explored_repeat_share'])}",
            f"- outcomes: {outcomes['events']} events, reward {_fmt(outcomes['reward_min'])}..{_fmt(outcomes['reward_max'])}"
            f" (mean {_fmt(outcomes['reward_mean'])}, positive share {_fmt(outcomes['positive_share'])})",
            f"- skills without any method: {_fmt(methods['skills_without_method'])}",
            f"- weekly-start vitality: median {_fmt((persona.get('vitality') or {}).get('median'))} "
            f"(<20 {_fmt((persona.get('vitality') or {}).get('share_below_20'))}, "
            f"=100 {_fmt((persona.get('vitality') or {}).get('share_at_100'))})",
        ]
        hints = persona.get("method_hints") or {}
        offers_summary = hints.get("offers") or {}
        adoptions_summary = hints.get("adoptions") or {}
        practice_summary = hints.get("practice") or {}
        plan_summary = hints.get("used_in_plan") or {}
        if offers_summary.get("events"):
            lines += [
                f"- method hints: {offers_summary['events']} offers "
                f"({offers_summary['distinct_methods']} methods, {offers_summary['weeks']} weeks, "
                f"{_fmt(offers_summary['block_chars_mean'])} chars/offer, max "
                f"{offers_summary['block_chars_max']}) {_fmt(offers_summary['roles'])}",
                f"- hint adoptions: {adoptions_summary['events']} "
                f"({adoptions_summary['distinct_methods']} methods, rate "
                f"{_fmt(adoptions_summary['adoption_rate'])}, matched by "
                f"{_fmt(adoptions_summary['matched_by'])})",
                f"- hint practice: {practice_summary['applied']} applied -> "
                f"{practice_summary['real_outcomes']} real outcomes "
                f"(mean reward {_fmt(practice_summary['reward_mean'])}, real share "
                f"{_fmt(practice_summary['real_share_of_adoptions'])}, counterfactual "
                f"{practice_summary['shadow_outcomes']})",
                f"- hint evidence integrity: free reinforcement "
                f"{len(practice_summary['free_reinforcement'] or [])}, updates without practice "
                f"{len(practice_summary['real_updates_without_practice'] or [])}, double counted "
                f"{len(practice_summary['double_counted_activities'] or [])}, unoffered "
                f"{len(adoptions_summary['unoffered'] or [])}",
                f"- used-in-plan (KI-10): {plan_summary['events']} events, "
                f"{plan_summary['distinct_memories']} memories / {plan_summary['distinct_methods']} methods; "
                f"adoptions missing it {len(plan_summary['adoptions_missing_used_in_plan'] or [])}",
            ]
        choices = persona.get("method_choices") or {}
        declines = choices.get("declines") or {}
        if declines.get("events"):
            bandit = choices.get("bandit") or {}
            situation = choices.get("situation") or {}
            lines += [
                f"- menu answer (phase 5): {declines['events']} declines "
                f"({_fmt(declines['reasons'])}); weeks with no adoption "
                f"{declines['weeks_with_no_adoption']}/{declines['weeks_total']}",
                f"- offer situation: {situation.get('distinct_context_keys')} context keys "
                f"{_fmt(situation.get('context_keys'))}, tags {_fmt(situation.get('tag_counts'))}, "
                f"offers carrying a goal {situation.get('offers_with_goal')}",
                f"- bandit: enabled {bandit.get('enabled')}, explore offers "
                f"{bandit.get('explore_offers')} (adopted {bandit.get('explore_adoptions')}), "
                f"longest run of the same exploit method {bandit.get('longest_same_exploit_method')}w, "
                f"offers with a non-zero ignored streak {bandit.get('offers_with_ignored_streak')}",
            ]
        baseline = choices.get("reward_baseline_shadow") or {}
        if baseline.get("available"):
            lines += [
                f"- KI-8 shadow baseline: {baseline['outcomes']} outcomes; live positive share "
                f"{_fmt((baseline['live'] or {}).get('positive_share'))} -> centred "
                f"{_fmt((baseline['shadow'] or {}).get('positive_share'))}, "
                f"negative {_fmt((baseline['shadow'] or {}).get('negative'))} "
                f"(<= -0.1: {_fmt((baseline['shadow'] or {}).get('below_minus_0_1'))}), "
                f"range {_fmt((baseline['shadow'] or {}).get('min'))}..{_fmt((baseline['shadow'] or {}).get('max'))}",
                f"- KI-8 projection: status transitions {_fmt(baseline['projected_status_transitions'])}, "
                f"deprecated reachable live {baseline['deprecated_reachable']} "
                f"(shadow {baseline.get('shadow_deprecated_reachable')}), "
                f"live replay exact {_fmt((baseline['live_replay_matches_recorded'] or {}).get('exact'))}, "
                f"ranking Spearman {_fmt(baseline.get('ranking_spearman'))}",
            ]
        lines.append("")

    fragmentation = report.get("skill_fragmentation") or {}
    if fragmentation:
        lines += [
            "## Skill-name families (phase-5 follow-up)",
            "",
            f"- alias file: {fragmentation.get('alias_file')} "
            f"({fragmentation.get('families_defined')} families defined, "
            f"{fragmentation.get('skills_observed')} skills observed)",
            f"- fragmented families: {fragmentation.get('fragmented_family_count')} "
            f"{_fmt(fragmentation.get('fragmented_families'))}",
            f"- skills no family claims: {len(fragmentation.get('unmapped_skills') or [])} "
            f"{_fmt(fragmentation.get('unmapped_skills'))}",
            "",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Run directory name under data/ or full path")
    parser.add_argument("--output", help="Optional JSON report path")
    parser.add_argument("--markdown", help="Optional markdown summary path")
    args = parser.parse_args()

    run_dir = Path(args.data_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / "data" / run_dir
    if not run_dir.exists():
        raise SystemExit(f"run not found: {run_dir}")

    report = audit_run(run_dir)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        if not output.is_absolute():
            output = ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        print(output)
    else:
        print(text)
    if args.markdown:
        markdown_path = Path(args.markdown)
        if not markdown_path.is_absolute():
            markdown_path = ROOT / markdown_path
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_markdown(report), encoding="utf-8")
        print(markdown_path)


if __name__ == "__main__":
    main()
