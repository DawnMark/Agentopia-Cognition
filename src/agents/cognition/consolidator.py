"""Weekly memory consolidation (design doc 6 / 9, design draft §4 and §6.3).

Per character per week:

1. collect the week's evidence from the ledger (activity records + the week's
   diary entry), numbered so the model can cite indices;
2. one LLM call turns it into structured memories;
3. validate every memory against the ledger (sources must exist, entities must
   be part of the world's vocabulary, skills must be skills the character has);
4. fold it into the existing memories: same semantic key -> reinforced,
   near-duplicate -> merged, opposite outcome on the same anchor -> both kept
   and marked contradicted;
5. rebuild the relation graph over all memories (rules only, capped);
6. (at SETTLE) recompute strength/tier for every memory.

Everything here is inert unless the shadow switch is on, and nothing here
touches prompts, plans, state or skill numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from src.agents.cognition import memory_views
from src.agents.cognition.lessons import (
    LessonConfig,
    LessonDiff,
    collect_lesson_diff,
    derive_polarity,
    derive_topics,
    lesson_disambiguator,
    lesson_skill_ids,
    provenance,
    resolve_entities,
)
from src.agents.cognition.impressions import (
    SOFT_ENTITY_CONFIDENCE_FACTOR,
    ensure_impressions,
    normalize_name,
)
from src.agents.cognition.memory_models import (
    MEMORY_TOPICS,
    MemoryItem,
    jaccard,
    memory_id_for,
    semantic_key,
    shared_anchor,
    token_set,
    validate_memory,
)
from src.agents.cognition.memory_store import MemoryEventStore, RelationEventStore
from src.agents.cognition.memory_strength import (
    DEFAULT_HALF_LIFE_WEEKS,
    is_protected,
    settle_strengths,
    tier_for,
)
from src.agents.cognition.prompts import (
    build_memory_extraction_prompt,
    format_memory_evidence,
    parse_memory_extraction,
    split_diary,
)
from src.agents.cognition.relation_graph import (
    DEFAULT_MAX_RELATIONS_PER_WEEK,
    build_relations,
)
from src.utils import get_logger

MEMORY_LOGGER = get_logger("cognition", quiet=True)

NEAR_DUPLICATE_MERGE_THRESHOLD = 0.6
DEFAULT_MAX_NEW_PER_WEEK = 5
DEFAULT_MAX_EXTRACT_CALLS_PER_WEEK = 1


@dataclass
class MemoryConfig:
    """`world.cognition` memory settings, with inert defaults."""

    enabled: bool = False
    max_new_per_week: int = DEFAULT_MAX_NEW_PER_WEEK
    max_relations_per_week: int = DEFAULT_MAX_RELATIONS_PER_WEEK
    relation_llm_judge: bool = False  # phase 2 is rule-only (design draft Q2)
    half_life_weeks: int = DEFAULT_HALF_LIFE_WEEKS
    max_extract_calls_per_week: int = DEFAULT_MAX_EXTRACT_CALLS_PER_WEEK

    @staticmethod
    def from_world_config(world_cfg: Dict[str, Any]) -> "MemoryConfig":
        section = (world_cfg or {}).get("cognition") or {}

        def _int(key: str, default: int) -> int:
            try:
                return int(section.get(key, default))
            except (TypeError, ValueError):
                return default

        return MemoryConfig(
            enabled=bool(section.get("memory_shadow", False)),
            max_new_per_week=max(0, _int("memory_max_new_per_week", DEFAULT_MAX_NEW_PER_WEEK)),
            max_relations_per_week=max(
                0, _int("memory_relation_max_per_week", DEFAULT_MAX_RELATIONS_PER_WEEK)
            ),
            relation_llm_judge=bool(section.get("memory_relation_llm_judge", False)),
            half_life_weeks=max(1, _int("memory_half_life_weeks", DEFAULT_HALF_LIFE_WEEKS)),
            max_extract_calls_per_week=max(
                0, _int("memory_max_extract_calls_per_week", DEFAULT_MAX_EXTRACT_CALLS_PER_WEEK)
            ),
        )


class Consolidator:
    """Weekly consolidation for one character."""

    def __init__(
        self,
        *,
        dm,
        clock,
        agent_name: str,
        model: str,
        config: Optional[MemoryConfig] = None,
        language: str = "en",
        world_cognition: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.dm = dm
        self.clock = clock
        self.agent_name = agent_name
        self.model = model
        self.config = config or MemoryConfig()
        self.language = language
        self.store = MemoryEventStore(dm)
        self.relations = RelationEventStore(dm)
        self._memories: Optional[List[MemoryItem]] = None
        self._calls: Dict[str, int] = {}
        self._impressions = None
        self.lesson_config = LessonConfig.from_world_config(
            {"cognition": dict(world_cognition or {})}
        )
        self._last_lesson_diff: Optional[LessonDiff] = None

    # -- gating ------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self.config.enabled

    # -- world knowledge used for validation -------------------------------
    def known_skills(self) -> Set[str]:
        try:
            state = self.dm.read_state(exclude_cur_t=False)
        except (IndexError, FileNotFoundError):
            return set()
        return {str(k) for k in (state.get("skills") or {}).keys()}

    def impressions(self):
        """The persona's alias book (seeded on first use, KI-2)."""
        if self._impressions is None:
            _keeper, view = ensure_impressions(self.dm, self.clock, week=self._week_key())
            self._impressions = view
        return self._impressions

    def known_entities(self) -> Set[str]:
        """Personas, places, possessions, known contacts and learned aliases.

        Mirrors the design rule "people must exist in the world", but the world
        is now read through the persona's impressions (`impressions.json`), so a
        Chinese place name or a short form of a name resolves to its canonical
        entity instead of being dropped (KI-2).
        """
        known: Set[str] = set()
        persona_root = Path("data") / self.dm.world / "persona"
        try:
            for path in persona_root.iterdir():
                if path.is_dir():
                    known.add(path.name)
        except OSError:
            pass
        try:
            for conv in self.dm.contact.glob("*.jsonl"):
                if conv.stem and conv.stem != "sig":
                    known.add(conv.stem)
        except OSError:
            pass
        try:
            state = self.dm.read_state(exclude_cur_t=False)
            for item in (state.get("assets") or {}).get("possessions") or []:
                name = item.get("name") if isinstance(item, dict) else item
                if name:
                    known.add(str(name))
        except (IndexError, FileNotFoundError, AttributeError):
            pass
        try:
            store = getattr(self.dm, "location_store", None)
            if store is not None:
                for key, location in list(getattr(store, "public", {}).items()) + list(
                    getattr(store, "private", {}).items()
                ):
                    known.add(str(key))
                    display = getattr(location, "display_name", None) or (
                        location.get("display_name") if isinstance(location, dict) else None
                    )
                    if display:
                        known.add(str(display))
        except Exception:
            pass
        return known

    # -- memories ----------------------------------------------------------
    def memories(self) -> List[MemoryItem]:
        if self._memories is None:
            self._memories = memory_views.load_memories(self.dm)
        return self._memories

    def _invalidate(self) -> None:
        self._memories = None
        self.store.invalidate()
        self.relations.invalidate()

    # -- the character's own lessons (phase 4.5) ---------------------------
    def ingest_lessons(self) -> Dict[str, int]:
        """Turn the lesson lines in the character's own notes into memories.

        Deterministic and LLM-free: the line the character wrote *is* the memory
        content, its evidence is the scratchpad snapshot's ledger id, and its
        confidence stays below an observation's (it is a belief, design doc
        invariant 3). Runs before the weekly extraction so the extractor sees
        the same week's evidence and cannot silently duplicate them — and
        independently of the extraction budget, because it costs no call.
        """
        result = {
            "lessons_created": 0,
            "lessons_carried": 0,
            "lessons_rewritten": 0,
            "lessons_dropped": 0,
            "lessons_rejected": 0,
        }
        if not self.enabled or not self.lesson_config.enabled:
            return result

        diff = collect_lesson_diff(self.dm, self.lesson_config)
        self._last_lesson_diff = diff
        result["lessons_carried"] = len(diff.carried)
        result["lessons_rewritten"] = len(diff.rewritten)
        result["lessons_dropped"] = len(diff.dropped)
        if not diff.new:
            return result

        existing_ids = {memory.memory_id for memory in self.memories()}
        known_skills = self.known_skills()
        known_entities = self.known_entities() | set(self.impressions().known())
        week = self._week_key()
        for line in diff.new[: self.lesson_config.max_new_per_week]:
            if not line.ledger_event_id:
                # A lesson without a ledger identity cannot be cited, and an
                # uncitable memory would break invariant 2.
                result["lessons_rejected"] += 1
                continue
            topics = derive_topics(line.text)
            polarity = derive_polarity(line.text)
            entities = resolve_entities(line.text, known_entities=known_entities)
            key = semantic_key(kind="lesson", topics=topics, entities=entities)
            memory_id = memory_id_for(
                persona=self.agent_name,
                kind="lesson",
                key=key,
                disambiguator=lesson_disambiguator(
                    polarity=polarity, content=line.text
                ),
            )
            if memory_id in existing_ids:
                # Same words already ingested (a re-run or a re-added line).
                continue
            try:
                item = MemoryItem(
                    memory_id=memory_id,
                    kind="lesson",
                    content=line.text,
                    persona=self.agent_name,
                    semantic_key=key,
                    entities=entities,
                    topics=topics,
                    skill_ids=lesson_skill_ids(line.text, known_skills),
                    source_event_ids=[line.ledger_event_id],
                    confidence=self.lesson_config.confidence,
                    salience=self.lesson_config.salience,
                    outcome_polarity=polarity,
                    created_at=str(self.clock.get_time()),
                    origin="scratchpad_lesson",
                )
            except ValueError:
                result["lessons_rejected"] += 1
                continue
            item.protected = is_protected(
                kind=item.kind, emotion=item.emotion, status=item.status, is_long_term_goal=False
            )
            payload = self._created_payload(item)
            payload["provenance"] = provenance(line.source_file, line.ledger_event_id)
            self.store.created(memory_id=memory_id, week=week, payload=payload)
            existing_ids.add(memory_id)
            result["lessons_created"] += 1

        self._invalidate()
        if any(result.values()):
            MEMORY_LOGGER.info(
                f"[{self.agent_name}] lesson bridge {week}: {result} "
                f"(snapshot {diff.snapshot_time})"
            )
        return result

    # -- weekly consolidation ---------------------------------------------
    def consolidate_week(self) -> Dict[str, int]:
        """Run one week of consolidation. Returns counts for logging/tests."""
        result = {
            "created": 0,
            "reinforced": 0,
            "merged": 0,
            "contradicted": 0,
            "rejected": 0,
            "relations": 0,
            "aliases_learned": 0,
            "soft_entities": 0,
        }
        if not self.enabled:
            return result
        if not self._budget_available():
            MEMORY_LOGGER.info(
                f"[{self.agent_name}] memory extraction budget exhausted for {self._week_key()}"
            )
            return result

        evidence = self.week_evidence()
        if not evidence:
            return result

        self._consume_budget()
        extracted = self._call_extraction(evidence)
        if not extracted:
            return result

        existing = self.memories()
        by_key = {m.semantic_key: m for m in existing if m.semantic_key}
        id_to_event = {item["event_id"]: item for item in evidence}
        known_skills = self.known_skills()
        known_entities = self.known_entities()
        known_event_ids = set(id_to_event)

        keeper, impressions = ensure_impressions(
            self.dm, self.clock, week=self._week_key()
        )
        self._impressions = impressions
        known_entities |= impressions.known()

        for raw in extracted[: self.config.max_new_per_week]:
            item = self._build_item(raw, evidence=evidence, known_skills=known_skills)
            if item is None:
                result["rejected"] += 1
                continue
            item = self._canonicalize_entities(
                item, keeper=keeper, impressions=impressions, result=result
            )
            outcome = validate_memory(
                item,
                known_event_ids=known_event_ids,
                known_skills=known_skills,
                known_entities=known_entities,
            )
            if not outcome.ok or outcome.item is None:
                result["rejected"] += 1
                MEMORY_LOGGER.info(
                    f"[{self.agent_name}] memory rejected ({outcome.reason}): "
                    f"{item.content[:60]}"
                )
                continue
            item = outcome.item
            self._store_memory(item, existing=existing, by_key=by_key, result=result)

        self._invalidate()
        result["relations"] = self._build_relations()

        if any(result[k] for k in ("created", "reinforced", "merged", "contradicted")):
            MEMORY_LOGGER.info(
                f"[{self.agent_name}] memory consolidation {self._week_key()}: {result}"
            )
        return result

    def _store_memory(
        self,
        item: MemoryItem,
        *,
        existing: List[MemoryItem],
        by_key: Dict[str, MemoryItem],
        result: Dict[str, int],
    ) -> None:
        week = self._week_key()

        # 1) opposite outcome on a shared anchor -> keep both and flag it.
        # This runs *before* the same-key check: two memories with the same
        # (kind, topics, entities) but opposite outcomes are a contradiction,
        # not a reinforcement (design doc 8.1: contradictions are high-value
        # signals and must never be silently merged away).
        conflict = self._find_conflict(item, existing)
        if conflict is not None:
            # Two contradictory claims about one anchor are two memories: give
            # this one a deterministic id that cannot collide with the original
            # (the semantic key names the anchor, not the claim).
            import hashlib

            item.memory_id = memory_id_for(
                persona=self.agent_name,
                kind=item.kind,
                key=item.semantic_key,
                disambiguator=(
                    f"{item.outcome_polarity}:"
                    f"{hashlib.sha1(item.content.encode('utf-8')).hexdigest()[:6]}"
                ),
            )
            self.store.created(
                memory_id=item.memory_id,
                week=week,
                payload=self._created_payload(item),
            )
            self.store.contradicted(
                memory_id=item.memory_id,
                week=week,
                conflicts_with=conflict.memory_id,
                reason=(
                    f"same anchor, opposite outcome: {item.outcome_polarity} vs "
                    f"{conflict.outcome_polarity}"
                ),
            )
            self.store.contradicted(
                memory_id=conflict.memory_id,
                week=week,
                conflicts_with=item.memory_id,
                reason=(
                    f"same anchor, opposite outcome: {conflict.outcome_polarity} vs "
                    f"{item.outcome_polarity}"
                ),
            )
            result["created"] += 1
            result["contradicted"] += 1
            existing.append(item)
            by_key[item.semantic_key] = item
            return

        # 2) exact semantic key -> reinforce the memory we already have
        same_key = by_key.get(item.semantic_key)
        if same_key is not None:
            self.store.reinforced(
                memory_id=same_key.memory_id,
                week=week,
                source_event_ids=item.source_event_ids,
                payload={"confidence": max(same_key.confidence, item.confidence)},
            )
            result["reinforced"] += 1
            return

        # 3) near-duplicate wording of the same kind -> merge, keeping the older id
        for candidate in existing:
            if candidate.kind != item.kind or candidate.semantic_key == item.semantic_key:
                continue
            if jaccard(token_set(candidate.content), token_set(item.content)) >= NEAR_DUPLICATE_MERGE_THRESHOLD:
                self.store.merged(
                    memory_id=candidate.memory_id,
                    week=week,
                    merged_from=[item.memory_id],
                    payload={
                        "source_event_ids": item.source_event_ids,
                        "content": item.content,
                    },
                )
                result["merged"] += 1
                return

        # 4) genuinely new memory
        self.store.created(
            memory_id=item.memory_id, week=week, payload=self._created_payload(item)
        )
        result["created"] += 1
        existing.append(item)
        by_key[item.semantic_key] = item

    def _created_payload(self, item: MemoryItem) -> Dict[str, Any]:
        week = self._week_key()
        payload = item.to_dict()
        payload.pop("memory_id", None)
        payload.update(
            {
                "week": week,
                "created_at": str(self.clock.get_time()),
                "strength": round(item.salience, 4),
                "tier": tier_for(item.salience),
            }
        )
        return payload

    @staticmethod
    def _find_conflict(item: MemoryItem, existing: Sequence[MemoryItem]) -> Optional[MemoryItem]:
        if item.outcome_polarity not in ("positive", "negative"):
            return None
        opposite = "negative" if item.outcome_polarity == "positive" else "positive"
        for candidate in existing:
            if candidate.outcome_polarity != opposite:
                continue
            # KI-4: the anchor must be concrete. Sharing only a coarse topic
            # used to make one negative memory contradict every positive memory
            # in the same topic.
            if shared_anchor(item, candidate):
                return candidate
        return None

    def _canonicalize_entities(
        self,
        item: MemoryItem,
        *,
        keeper,
        impressions,
        result: Dict[str, int],
    ) -> MemoryItem:
        """Resolve extracted names against the persona's impressions (KI-2).

        Canonical names stay in `entities` and drive the relation graph; names
        that could not be resolved are *kept* as `soft_entities` with a small
        confidence discount instead of being dropped, and are recorded so a
        later alias can promote them. The semantic key/id are recomputed because
        the identity of a memory includes its entities.
        """
        raw_names = list(item.entities)
        if not raw_names:
            return item
        canonical, unresolved = impressions.canonicalize(raw_names)
        learned = 0
        for name in raw_names:
            found, how = impressions.resolve(name)
            if found and normalize_name(name) != normalize_name(found):
                if keeper.learn_alias(
                    alias=name,
                    canonical=found,
                    week=self._week_key(),
                    source=f"memory_extraction:{how}",
                ):
                    learned += 1
        item.entities = canonical
        if unresolved:
            item.soft_entities = unresolved
            item.confidence = round(
                float(item.confidence) * SOFT_ENTITY_CONFIDENCE_FACTOR, 4
            )
            keeper.record_soft(names=unresolved, week=self._week_key())
        if learned:
            self._impressions = keeper.impressions()
            impressions = self._impressions
            result["aliases_learned"] = result.get("aliases_learned", 0) + learned
        if canonical or unresolved:
            item.semantic_key = semantic_key(
                kind=item.kind, topics=item.topics, entities=item.entities
            )
            item.memory_id = memory_id_for(
                persona=self.agent_name, kind=item.kind, key=item.semantic_key
            )
        return item

    def _build_item(
        self,
        raw: Dict[str, Any],
        *,
        evidence: List[Dict[str, Any]],
        known_skills: Set[str],
    ) -> Optional[MemoryItem]:
        sources = [
            evidence[idx]["event_id"]
            for idx in raw.get("sources") or []
            if isinstance(idx, int) and 0 <= idx < len(evidence)
        ]
        topics = [t for t in (raw.get("topics") or []) if t in MEMORY_TOPICS]
        entities = list(raw.get("entities") or [])
        kind = str(raw.get("kind") or "episodic")
        if kind not in ("episodic", "semantic", "relationship", "goal", "belief", "lesson"):
            kind = "episodic"
        key = semantic_key(kind=kind, topics=topics, entities=entities)
        memory_id = memory_id_for(persona=self.agent_name, kind=kind, key=key)
        try:
            item = MemoryItem(
                memory_id=memory_id,
                kind=kind,
                content=str(raw.get("content") or ""),
                persona=self.agent_name,
                semantic_key=key,
                entities=entities,
                topics=topics,
                goal_ids=[str(raw.get("goal"))] if raw.get("goal") else [],
                skill_ids=[s for s in (raw.get("skills") or []) if s in known_skills],
                source_event_ids=sources,
                confidence=raw.get("confidence", 0.5),
                salience=raw.get("salience", 0.5),
                outcome_polarity=str(raw.get("outcome_polarity") or "neutral"),
                resources=list(raw.get("resources") or []),
                obstacles=list(raw.get("obstacles") or []),
                emotion=raw.get("emotion", 0.0),
                created_at=str(self.clock.get_time()),
            )
        except ValueError:
            return None
        item.protected = is_protected(
            kind=item.kind,
            emotion=item.emotion,
            status=item.status,
            is_long_term_goal=item.kind == "goal",
        )
        return item

    def _build_relations(self) -> int:
        memories = self.memories()
        if len(memories) < 2:
            return 0
        from src.agents.cognition.methodology_policy import deterministic_rng

        relations = build_relations(
            memories,
            week=self._week_key(),
            rng=deterministic_rng(self.agent_name, self._week_key(), "memory-relations"),
            max_relations=self.config.max_relations_per_week,
        )
        written = 0
        for relation in relations:
            _, created = self.relations.asserted(relation)
            if created:
                written += 1
        return written

    # -- weekly settlement (SETTLE stage) -----------------------------------
    def settle_strengths(self) -> int:
        """Recompute strength/tier and write `MEMORY_STRENGTH_UPDATED` events."""
        if not self.enabled:
            return 0
        view = memory_views.build_memory_views(self.dm)["memories"]
        memories = list(view["memories"].values())
        if not memories:
            return 0

        positive_outcomes: Dict[str, int] = {}
        contradictions: Dict[str, int] = {}
        used_in_plan: Dict[str, int] = {}
        for memory_id, memory in view["memories"].items():
            if memory.get("outcome_polarity") == "positive":
                positive_outcomes[memory_id] = 1
            contradictions[memory_id] = len(memory.get("conflicts_with") or [])
            # KI-10: now that phase 4 lets an adopted method cite its memories,
            # real use feeds the strength formula instead of staying at zero.
            used_in_plan[memory_id] = int(memory.get("used_in_plan_count") or 0)

        updates = settle_strengths(
            memories,
            current_time=str(self.clock.get_time()),
            positive_outcome_counts=positive_outcomes,
            contradiction_counts=contradictions,
            used_in_plan_counts=used_in_plan,
            half_life_weeks=self.config.half_life_weeks,
        )
        week = self._week_key()
        written = 0
        for update in updates:
            _, created = self.store.strength_updated(
                memory_id=str(update["memory_id"]),
                week=week,
                strength=float(update["strength"]),
                tier=str(update["tier"]),
                factors=dict(update["factors"]),
            )
            if created:
                written += 1
        self._invalidate()
        return written

    # -- evidence ----------------------------------------------------------
    def week_evidence(self) -> List[Dict[str, Any]]:
        """Numbered, citable evidence for this week.

        Two kinds of item, kept apart on purpose:

        * **objective** — the activity record (what was attempted) and the
          environment model's outcome/deltas;
        * **subjective** — the character's own reflection, per activity and at
          the end of the week (the "教训" it wrote for itself).

        The subjective items are what the lesson system is made of. They are
        rendered as their own evidence entries so a memory can cite "what I
        concluded" separately from "what happened", and so the extractor can
        only use them for `belief` / `lesson` memories (design doc invariants
        2-4: a reflection is never objective evidence and never moves a value).
        """
        records: List[Dict[str, Any]] = []
        week = self._week_key()
        activity_path = self.dm.root / "activity.jsonl"
        if activity_path.exists():
            import json

            for line in activity_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not str(record.get("time") or "").startswith(week):
                    continue
                records.append(record)
                reflection = str(record.get("reflection") or "").strip()
                if reflection:
                    records.append(
                        {
                            "ledger_event_id": record.get("ledger_event_id"),
                            "time": record.get("time"),
                            "type": "reflection",
                            "activity_type": record.get("type"),
                            "content": reflection,
                            "outcome": {"outcome": ""},
                        }
                    )

        diary_path = self.dm.weekly_diary
        if diary_path.exists():
            import json

            for line in diary_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not str(record.get("time") or "").startswith(week):
                    continue
                # The diary is "Summary: … \n\n Reflection: …". They are split
                # because the reflection is the part the character writes *for
                # itself* (lessons, resolutions), and it used to be cut off
                # entirely by the old 400-char clip.
                summary, reflection = split_diary(str(record.get("content") or ""))
                for item_type, text in (
                    ("weekly_review_summary", summary),
                    ("weekly_review_reflection", reflection),
                ):
                    if not text:
                        continue
                    records.append(
                        {
                            "ledger_event_id": record.get("ledger_event_id"),
                            "time": record.get("time"),
                            "type": item_type,
                            "content": text,
                            "outcome": {"outcome": ""},
                        }
                    )
                break
        return format_memory_evidence(records)

    # -- LLM ---------------------------------------------------------------
    def _call_extraction(self, evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        from src.utils import get_response_with_retry

        prompt = build_memory_extraction_prompt(
            char=self.agent_name,
            evidence_items=evidence,
            topic_list=list(MEMORY_TOPICS),
            language=self.language,
        )

        def _validator(response: str, **kwargs):
            return parse_memory_extraction(response)

        try:
            messages = [{"role": "system", "content": prompt}]
            data = get_response_with_retry(
                post_processing_funcs=[_validator],
                model=self.model,
                messages=messages,
                max_retry=2,
            )
            from src.agents.cognition.usage import record_llm_usage

            record_llm_usage(
                self.dm,
                "memory_extraction",
                messages=messages,
                response=data,
                model=self.model,
            )
        except Exception as e:  # pragma: no cover - shadow never breaks a run
            MEMORY_LOGGER.warning(f"[{self.agent_name}] memory extraction failed: {e!r}")
            return []
        if not isinstance(data, dict):
            return []
        memories = data.get("memories")
        return memories if isinstance(memories, list) else []

    # -- budget / week key --------------------------------------------------
    def _week_key(self) -> str:
        t = self.clock.get_time()
        return f"Y{t.year}-W{t.week:02d}"

    def _budget_available(self) -> bool:
        return self._calls.get(self._week_key(), 0) < self.config.max_extract_calls_per_week

    def _consume_budget(self) -> None:
        key = self._week_key()
        self._calls[key] = self._calls.get(key, 0) + 1
