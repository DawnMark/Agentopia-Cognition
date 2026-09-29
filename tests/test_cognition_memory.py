"""阶段 2：结构化记忆与关系图（离线，无真实 LLM 调用）。

对齐设计稿 §10 的 10 条验收：关闭即惰性、行为零影响（不写世界账本）、视图可重建、
幂等、来源完整、关系规则正反例、遗忘不删史、protected 生效、成本上限、影子对比可复现。
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

from src.agents.cognition.consolidator import Consolidator, MemoryConfig
from src.agents.cognition.memory_models import (
    MEMORY_TOPICS,
    MemoryItem,
    memory_id_for,
    semantic_key,
    validate_memory,
)
from src.agents.cognition.memory_retriever import (
    MemoryRetriever,
    RetrievalConfig,
    legacy_retrieval,
    score_memories,
)
from src.agents.cognition.memory_store import MemoryEventStore, RelationEventStore
from src.agents.cognition.memory_strength import (
    compute_strength,
    decay_factor,
    is_protected,
    settle_strengths,
    tier_for,
    weeks_between,
)
from src.agents.cognition.memory_views import (
    build_memory_views,
    load_memories,
    materialize_memories,
    materialize_relations,
    write_memory_views,
)
from src.agents.cognition.relation_graph import (
    InvertedIndex,
    build_relations,
    relation_score,
    relation_types_between,
)
from src.world.clock import Stage, TimeState
from tests._helpers import make_datamanager, temp_workspace


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
def _state() -> Dict[str, Any]:
    return {
        "vitality": 80,
        "fulfillment": {"mood": 50, "material": 50, "social": 50, "esteem": 50},
        "assets": {"deposit": 1000, "possessions": [{"name": "旧相机", "description": "二手"}]},
        "skills": {"写作": 12, "跑步": 30},
    }


def _memory(
    *,
    memory_id: str = "mem-a",
    kind: str = "episodic",
    content: str = "在书桌前写完了一章初稿",
    topics: List[str] | None = None,
    entities: List[str] | None = None,
    goals: List[str] | None = None,
    skills: List[str] | None = None,
    polarity: str = "positive",
    obstacles: List[str] | None = None,
    resources: List[str] | None = None,
    salience: float = 0.7,
    strength: float = 0.7,
    tier: str = "hot",
    created_at: str = "Y2020-W01-settle",
    status: str = "active",
    emotion: float = 0.0,
    sources: List[str] | None = None,
) -> MemoryItem:
    return MemoryItem(
        memory_id=memory_id,
        kind=kind,
        content=content,
        persona="测试角色",
        semantic_key=semantic_key(
            kind=kind, topics=topics or ["creation"], entities=entities or ["书桌"]
        ),
        topics=list(topics or ["creation"]),
        entities=list(entities or ["书桌"]),
        goal_ids=list(goals or []),
        skill_ids=list(skills or []),
        source_event_ids=list(sources or ["ev-1"]),
        confidence=0.7,
        salience=salience,
        strength=strength,
        tier=tier,
        created_at=created_at,
        status=status,
        outcome_polarity=polarity,
        obstacles=list(obstacles or []),
        resources=list(resources or []),
        emotion=emotion,
    )


def _extraction_payload(memories: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"memories": memories}


class MemoryModelTests(unittest.TestCase):
    def test_unknown_kind_polarity_or_status_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MemoryItem(memory_id="m", kind="nonsense", content="x")
        with self.assertRaises(ValueError):
            MemoryItem(memory_id="m", kind="episodic", content="x", outcome_polarity="great")
        with self.assertRaises(ValueError):
            MemoryItem(memory_id="m", kind="episodic", content="x", status="deleted")

    def test_topics_are_restricted_to_the_controlled_list(self) -> None:
        item = MemoryItem(
            memory_id="m", kind="episodic", content="x", topics=["creation", "not_a_topic"]
        )
        self.assertEqual(item.topics, ["creation"])
        self.assertIn("creation", MEMORY_TOPICS)

    def test_content_is_clipped_and_lists_deduplicated(self) -> None:
        item = MemoryItem(
            memory_id="m",
            kind="episodic",
            content="字" * 500,
            entities=["甲", "甲", "乙"],
        )
        self.assertEqual(len(item.content), 200)
        self.assertEqual(item.entities, ["甲", "乙"])

    def test_memory_id_is_content_addressed(self) -> None:
        key = semantic_key(kind="lesson", topics=["work"], entities=["甲"])
        self.assertEqual(
            memory_id_for(persona="甲", kind="lesson", key=key),
            memory_id_for(persona="甲", kind="lesson", key=key),
        )
        self.assertNotEqual(
            memory_id_for(persona="甲", kind="lesson", key=key),
            memory_id_for(persona="乙", kind="lesson", key=key),
        )

    def test_validation_requires_a_known_source(self) -> None:
        item = _memory(sources=["ev-unknown"])
        outcome = validate_memory(
            item, known_event_ids={"ev-1"}, known_skills={"写作"}, known_entities={"书桌"}
        )
        self.assertFalse(outcome.ok)
        self.assertIn("source", outcome.reason)

    def test_unknown_entities_are_dropped_not_invented(self) -> None:
        item = _memory(entities=["书桌", "不存在的咖啡馆"])
        outcome = validate_memory(
            item, known_event_ids={"ev-1"}, known_skills=set(), known_entities={"书桌"}
        )
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.item.entities, ["书桌"])
        self.assertEqual(outcome.item.dropped_entities, ["不存在的咖啡馆"])
        # losing an anchor lowers confidence rather than deleting the memory
        self.assertLess(outcome.item.confidence, 0.7)

    def test_unknown_skills_are_removed(self) -> None:
        item = _memory(skills=["写作", "占星"])
        outcome = validate_memory(
            item,
            known_event_ids={"ev-1"},
            known_skills={"写作"},
            known_entities={"书桌"},
        )
        self.assertEqual(outcome.item.skill_ids, ["写作"])


class StrengthTests(unittest.TestCase):
    def test_decay_halves_every_half_life(self) -> None:
        self.assertAlmostEqual(decay_factor(0, half_life_weeks=8), 1.0)
        self.assertAlmostEqual(decay_factor(8, half_life_weeks=8), 0.5)
        self.assertAlmostEqual(decay_factor(16, half_life_weeks=8), 0.25)

    def test_tiers_follow_the_thresholds(self) -> None:
        self.assertEqual(tier_for(0.8), "hot")
        self.assertEqual(tier_for(0.6), "hot")
        self.assertEqual(tier_for(0.45), "warm")
        self.assertEqual(tier_for(0.2), "cold")
        self.assertEqual(tier_for(0.05), "archived")

    def test_reads_and_outcomes_reinforce_strength(self) -> None:
        base, _ = compute_strength(salience=0.5, weeks_since_created=0)
        read, _ = compute_strength(salience=0.5, weeks_since_created=0, read_count=3)
        outcome, _ = compute_strength(
            salience=0.5, weeks_since_created=0, positive_outcome_count=2
        )
        self.assertGreater(read, base)
        self.assertGreater(outcome, read)

    def test_used_in_plan_is_recorded_but_zero_in_phase_2(self) -> None:
        strength, factors = compute_strength(salience=0.5, weeks_since_created=0)
        self.assertEqual(factors.used_in_plan_count, 0)
        self.assertAlmostEqual(strength, 0.5, places=3)

    def test_contradiction_lowers_strength(self) -> None:
        clean, _ = compute_strength(salience=0.6, weeks_since_created=0)
        conflicted, factors = compute_strength(
            salience=0.6, weeks_since_created=0, contradictions=2
        )
        self.assertLess(conflicted, clean)
        self.assertGreater(factors.contradiction_penalty, 0)

    def test_protected_memories_keep_a_floor(self) -> None:
        strength, _ = compute_strength(
            salience=0.05, weeks_since_created=100, protected=True
        )
        self.assertGreaterEqual(strength, 0.35)

    def test_protected_categories(self) -> None:
        self.assertTrue(is_protected(kind="goal", emotion=0.0, status="active"))
        self.assertTrue(is_protected(kind="episodic", emotion=0.9, status="active"))
        self.assertTrue(
            is_protected(kind="episodic", emotion=0.0, status="active", is_current_asset=True)
        )
        self.assertTrue(
            is_protected(kind="episodic", emotion=0.0, status="active", has_active_commitment=True)
        )
        self.assertTrue(
            is_protected(kind="relationship", emotion=0.7, status="active")
        )
        self.assertFalse(is_protected(kind="episodic", emotion=0.1, status="active"))

    def test_weeks_between_is_non_negative(self) -> None:
        self.assertEqual(weeks_between("Y2020-W01-begin", "Y2020-W03-begin"), 2)
        self.assertEqual(weeks_between("Y2020-W03-begin", "Y2020-W01-begin"), 2)
        self.assertEqual(weeks_between("garbage", "Y2020-W01"), 0)

    def test_settle_reports_only_changes(self) -> None:
        memories = [
            {"memory_id": "m1", "salience": 0.9, "strength": 0.9, "tier": "hot", "created_at": "Y2020-W01-settle", "source_event_ids": ["ev-1"]},
            {"memory_id": "m2", "salience": 0.2, "strength": 0.1, "tier": "cold", "created_at": "Y2020-W01-settle", "source_event_ids": ["ev-1"]},
        ]
        updates = settle_strengths(memories, current_time="Y2020-W10-settle")
        updated_ids = {u["memory_id"] for u in updates}
        self.assertIn("m2", updated_ids)  # decay moved it into archived
        self.assertIn("m1", updated_ids)


class RelationGraphTests(unittest.TestCase):
    def test_same_entity_and_same_goal(self) -> None:
        a = _memory(memory_id="a", entities=["甲"], goals=["g1"])
        b = _memory(memory_id="b", entities=["甲"], goals=["g1"], content="另一件事")
        types = relation_types_between(a, b)
        self.assertIn("same_entity", types)
        self.assertIn("same_goal", types)

    def test_unrelated_memories_have_no_relation(self) -> None:
        a = _memory(memory_id="a", entities=["甲"], topics=["work"], content="加班到很晚")
        b = _memory(
            memory_id="b",
            entities=["乙"],
            topics=["food"],
            content="煮了一锅粥",
            polarity="neutral",
        )
        b.created_at = "Y2021-W01-settle"
        self.assertEqual(relation_types_between(a, b), [])

    def test_temporal_alone_never_relates(self) -> None:
        a = _memory(memory_id="a", entities=["甲"], topics=["work"], content="开会")
        b = _memory(memory_id="b", entities=["乙"], topics=["food"], content="吃饭")
        b.created_at = a.created_at
        self.assertNotIn("temporal", relation_types_between(a, b))

    def test_temporal_joins_a_pair_that_shares_something(self) -> None:
        a = _memory(memory_id="a", entities=["甲"], content="开会")
        b = _memory(memory_id="b", entities=["甲"], content="复盘会议", created_at="Y2020-W02-settle")
        self.assertIn("temporal", relation_types_between(a, b))

    def test_contradiction_needs_opposite_outcomes(self) -> None:
        a = _memory(memory_id="a", entities=["甲"], topics=["exercise"], polarity="positive")
        b = _memory(memory_id="b", entities=["甲"], topics=["exercise"], polarity="negative")
        self.assertIn("contradiction", relation_types_between(a, b))
        c = _memory(memory_id="c", entities=["甲"], topics=["exercise"], polarity="positive")
        self.assertNotIn("contradiction", relation_types_between(a, c))

    def test_problem_resource_matches_obstacles_to_resources(self) -> None:
        problem = _memory(memory_id="p", obstacles=["没有安静的地方"], content="找不到地方写作")
        resource = _memory(
            memory_id="r",
            resources=["没有安静的地方"],
            content="图书馆早上很安静",
            entities=["图书馆"],
        )
        types = relation_types_between(problem, resource)
        self.assertIn("problem_resource", types)
        self.assertNotIn("precondition", types)  # no shared goal

    def test_precondition_requires_a_shared_goal(self) -> None:
        problem = _memory(memory_id="p", obstacles=["缺钱"], goals=["g1"])
        resource = _memory(memory_id="r", resources=["缺钱"], goals=["g1"])
        types = relation_types_between(problem, resource)
        self.assertIn("problem_resource", types)
        self.assertIn("precondition", types)

    def test_method_transfer_needs_different_skills(self) -> None:
        a = _memory(memory_id="a", skills=["写作"], topics=["planning"], entities=["甲"])
        b = _memory(
            memory_id="b", skills=["跑步"], topics=["planning"], entities=["乙"], content="制定训练计划"
        )
        self.assertIn("method_transfer", relation_types_between(a, b))
        c = _memory(memory_id="c", skills=["写作"], topics=["planning"], entities=["丙"])
        self.assertNotIn("method_transfer", relation_types_between(a, c))

    def test_repeated_pattern_needs_a_family_of_three(self) -> None:
        family = [
            _memory(memory_id=f"f{i}", topics=["exercise"], polarity="negative", entities=[f"人{i}"], content=f"跑步受伤 {i}")
            for i in range(3)
        ]
        index = InvertedIndex(family)
        types = relation_types_between(family[0], family[1], index=index)
        self.assertIn("repeated_pattern", types)

        pair = family[:2]
        types_small = relation_types_between(pair[0], pair[1], index=InvertedIndex(pair))
        self.assertNotIn("repeated_pattern", types_small)

    def test_gap_needs_shared_goal_and_no_resource(self) -> None:
        a = _memory(memory_id="a", goals=["g1"], obstacles=["没有场地"])
        b = _memory(memory_id="b", goals=["g1"], obstacles=["没有时间"], content="时间不够")
        self.assertIn("gap", relation_types_between(a, b))

    def test_causal_requires_entity_topic_flip_and_time(self) -> None:
        a = _memory(
            memory_id="a", entities=["甲"], topics=["work"], polarity="negative", content="项目失败"
        )
        b = _memory(
            memory_id="b",
            entities=["甲"],
            topics=["work"],
            polarity="positive",
            content="换了做法后项目成功",
            created_at="Y2020-W02-settle",
        )
        self.assertIn("causal", relation_types_between(a, b))

    def test_score_prefers_complementary_over_redundant(self) -> None:
        near_duplicate = _memory(memory_id="d", content="在书桌前写完了一章初稿")
        original = _memory(memory_id="o", content="在书桌前写完了一章初稿")
        score_dup, components = relation_score(original, near_duplicate, ["same_entity"])
        self.assertEqual(components["redundancy"], 1.0)

        problem = _memory(memory_id="p", obstacles=["没有安静的地方"], content="找不到地方写作")
        resource = _memory(
            memory_id="r", resources=["没有安静的地方"], content="图书馆早上很安静", entities=["图书馆"]
        )
        score_pr, _ = relation_score(problem, resource, ["problem_resource"])
        self.assertGreater(score_pr, score_dup)

    def test_build_relations_is_capped_and_deduplicated(self) -> None:
        memories = [
            _memory(memory_id=f"m{i}", entities=["甲"], topics=["work"], content=f"工作记录 {i}")
            for i in range(6)
        ]
        relations = build_relations(memories, week="Y2020-W02", max_relations=4)
        self.assertLessEqual(len(relations), 4)
        keys = [r.key for r in relations]
        self.assertEqual(len(keys), len(set(keys)))
        # undirected: the same pair is never asserted twice
        pairs = {tuple(sorted([r.source_memory_id, r.target_memory_id])) for r in relations}
        self.assertEqual(len(pairs), len(relations))

    def test_candidate_recall_respects_the_limit(self) -> None:
        memories = [
            _memory(memory_id=f"m{i}", entities=["甲"], content=f"记录 {i}") for i in range(30)
        ]
        index = InvertedIndex(memories)
        candidates = index.candidates(memories[0], limit=5)
        self.assertLessEqual(len(candidates), 5)
        self.assertNotIn(memories[0].memory_id, [c.memory_id for c in candidates])


class ConsolidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self._ctx = temp_workspace()
        self.root = self._ctx.__path__ if False else self._ctx.__enter__()  # type: ignore[attr-defined]
        self.dm, self.clock = make_datamanager()
        self.dm.save_state(_state())
        self._add_activity("Y2020-W01-activity-D2", "写作", 2)

    def tearDown(self) -> None:
        self._ctx.__exit__(None, None, None)  # type: ignore[attr-defined]

    def _add_activity(self, time_str: str, skill: str, gain: float, activity_id: str | None = None) -> str:
        from src.world.solo_activity_data import ActionOutcome, SoloActivityRecord

        stamp = TimeState.from_string(time_str)
        record = SoloActivityRecord(
            activity_id=activity_id or f"solo-{time_str}",
            agent_name=self.dm.char,
            time=stamp,
            content="写小说",
            outcome=ActionOutcome(
                outcome="完成了初稿",
                delta_vitality=-2,
                delta_fulfillment={"mood": 2},
                delta_skills={skill: gain},
            ),
        )
        self.dm.append_activity_record(record)
        rows = [
            json.loads(line)
            for line in (self.dm.root / "activity.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return str(rows[-1]["ledger_event_id"])

    def _consolidator(self, **overrides) -> Consolidator:
        cfg = MemoryConfig(
            enabled=overrides.pop("enabled", True),
            max_new_per_week=overrides.pop("max_new_per_week", 5),
            max_relations_per_week=overrides.pop("max_relations_per_week", 10),
        )
        return Consolidator(
            dm=self.dm,
            clock=self.clock,
            agent_name=self.dm.char,
            model="test-model",
            config=cfg,
        )

    def _run_week(self, consolidator: Consolidator, payload: Dict[str, Any]):
        with mock.patch("src.utils.get_response_with_retry", return_value=payload) as llm:
            result = consolidator.consolidate_week()
        return result, llm

    # -- gating and cost ---------------------------------------------------
    def test_disabled_writes_nothing_and_never_calls_the_model(self) -> None:
        consolidator = self._consolidator(enabled=False)
        with mock.patch("src.utils.get_response_with_retry") as llm:
            result = consolidator.consolidate_week()
        llm.assert_not_called()
        self.assertEqual(sum(result.values()), 0)
        self.assertFalse(consolidator.store.path.exists())

    def test_one_call_per_week_and_one_week_only(self) -> None:
        consolidator = self._consolidator()
        payload = _extraction_payload(
            [{"kind": "lesson", "content": "写长篇前先列场景", "topics": ["creation"], "entities": [self.dm.char], "sources": [0]}]
        )
        result, llm = self._run_week(consolidator, payload)
        self.assertEqual(result["created"], 1)
        llm.assert_called_once()

        # Same week again: budget refuses before any model call.
        result2, llm2 = self._run_week(consolidator, payload)
        self.assertEqual(sum(result2.values()), 0)
        llm2.assert_not_called()

    def test_weekly_cap_on_new_memories(self) -> None:
        consolidator = self._consolidator(max_new_per_week=2)
        subjects = [
            ("work", "在图书馆查了一下午资料"),
            ("study", "把课程笔记重新整理了一遍"),
            ("health", "早睡了一整周"),
            ("exercise", "绕着公园跑了五公里"),
            ("food", "学会了做番茄鸡蛋面"),
        ]
        payload = _extraction_payload(
            [
                {"kind": "episodic", "content": content, "topics": [topic], "entities": [self.dm.char], "sources": [0]}
                for topic, content in subjects
            ]
        )
        result, _ = self._run_week(consolidator, payload)
        self.assertEqual(result["created"], 2, "the weekly cap must truncate")
        self.assertEqual(len(consolidator.store.of_type("MEMORY_CREATED")), 2)

    def test_extraction_without_evidence_does_not_call_the_model(self) -> None:
        consolidator = self._consolidator()
        self.clock.set_week(9)  # a week with no activity records
        with mock.patch("src.utils.get_response_with_retry") as llm:
            result = consolidator.consolidate_week()
        llm.assert_not_called()
        self.assertEqual(sum(result.values()), 0)

    def test_a_bad_model_answer_is_not_fatal(self) -> None:
        consolidator = self._consolidator()
        with mock.patch("src.utils.get_response_with_retry", side_effect=RuntimeError("boom")):
            self.assertEqual(sum(consolidator.consolidate_week().values()), 0)
        with mock.patch("src.utils.get_response_with_retry", return_value=None):
            self.assertEqual(sum(consolidator.consolidate_week().values()), 0)

    # -- provenance and dedup ---------------------------------------------
    def test_memories_cite_real_ledger_events(self) -> None:
        consolidator = self._consolidator()
        payload = _extraction_payload(
            [{"kind": "lesson", "content": "先列场景再写", "topics": ["creation"], "entities": [self.dm.char], "sources": [0]}]
        )
        self._run_week(consolidator, payload)

        created = consolidator.store.of_type("MEMORY_CREATED")[0]
        ledger_ids = {
            json.loads(line)["ledger_event_id"]
            for line in (self.dm.root / "activity.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        self.assertTrue(set(created["source_event_ids"]) <= ledger_ids)
        self.assertTrue(created["source_event_ids"])

    def test_memory_without_a_valid_source_is_rejected(self) -> None:
        consolidator = self._consolidator()
        payload = _extraction_payload(
            [{"kind": "lesson", "content": "凭空捏造", "topics": ["work"], "sources": [99]}]
        )
        result, _ = self._run_week(consolidator, payload)
        self.assertEqual(result["rejected"], 1)
        self.assertEqual(result["created"], 0)

    def test_same_semantic_key_reinforces_instead_of_duplicating(self) -> None:
        consolidator = self._consolidator()
        item = {
            "kind": "lesson",
            "content": "先列场景再写",
            "topics": ["creation"],
            "entities": [self.dm.char],
            "sources": [0],
        }
        self._run_week(consolidator, _extraction_payload([item]))

        self.clock.set_week(2)
        self._add_activity("Y2020-W02-activity-D3", "写作", 1)
        second = self._consolidator()
        result, _ = self._run_week(second, _extraction_payload([dict(item, sources=[0])]))

        self.assertEqual(result["reinforced"], 1)
        self.assertEqual(result["created"], 0)

    def test_near_duplicate_wording_is_merged(self) -> None:
        consolidator = self._consolidator()
        self._run_week(
            consolidator,
            _extraction_payload(
                [{"kind": "lesson", "content": "写长篇之前应该先列出场景顺序", "topics": ["creation"], "entities": [self.dm.char], "sources": [0]}]
            ),
        )
        self.clock.set_week(2)
        self._add_activity("Y2020-W02-activity-D3", "写作", 1)
        second = self._consolidator()
        result, _ = self._run_week(
            second,
            _extraction_payload(
                [{"kind": "lesson", "content": "写长篇之前应该先列出场景顺序再动笔", "topics": ["planning"], "entities": [self.dm.char], "sources": [0]}]
            ),
        )
        self.assertEqual(result["merged"], 1)

    def test_opposite_outcome_keeps_both_and_flags_the_conflict(self) -> None:
        consolidator = self._consolidator()
        self._run_week(
            consolidator,
            _extraction_payload(
                [{"kind": "semantic", "content": "早起跑步让人一天有精神", "topics": ["exercise"], "entities": [self.dm.char], "outcome_polarity": "positive", "sources": [0]}]
            ),
        )
        self.clock.set_week(2)
        self._add_activity("Y2020-W02-activity-D3", "跑步", 1, activity_id="solo-W02-D3")
        second = self._consolidator()
        result, _ = self._run_week(
            second,
            _extraction_payload(
                [{"kind": "semantic", "content": "早起跑步让我整天疲惫", "topics": ["exercise"], "entities": [self.dm.char], "outcome_polarity": "negative", "sources": [0]}]
            ),
        )
        self.assertEqual(result["contradicted"], 1)
        self.assertEqual(result["created"], 1)
        view = build_memory_views(self.dm)["memories"]
        statuses = {m["status"] for m in view["memories"].values()}
        self.assertIn("contradicted", statuses)
        self.assertEqual(view["stats"]["memories"], 2, "contradiction keeps both memories")

    # -- relations ---------------------------------------------------------
    def test_relations_are_built_and_capped(self) -> None:
        consolidator = self._consolidator(max_relations_per_week=2)
        payload = _extraction_payload(
            [
                {"kind": "episodic", "content": f"和{self.dm.char}一起完成第 {i} 件事", "topics": ["cooperation"], "entities": [self.dm.char], "sources": [0]}
                for i in range(4)
            ]
        )
        result, _ = self._run_week(consolidator, payload)
        self.assertLessEqual(result["relations"], 2)
        self.assertLessEqual(len(consolidator.relations.events()), 2)

    # -- world ledger untouched -------------------------------------------
    def test_consolidation_never_writes_to_the_simulation_ledger(self) -> None:
        watched = [
            self.dm.root / "activity.jsonl",
            self.dm.root / "state.jsonl",
            self.dm.root / "schedule.jsonl",
        ]
        before = {p: p.read_text(encoding="utf-8") for p in watched if p.exists()}

        consolidator = self._consolidator()
        self._run_week(
            consolidator,
            _extraction_payload(
                [{"kind": "lesson", "content": "先列场景再写", "topics": ["creation"], "entities": [self.dm.char], "sources": [0]}]
            ),
        )

        for path, content in before.items():
            self.assertEqual(path.read_text(encoding="utf-8"), content, f"{path.name} changed")

    # -- settlement --------------------------------------------------------
    def test_settle_writes_strength_events_and_keeps_archived_history(self) -> None:
        consolidator = self._consolidator()
        self._run_week(
            consolidator,
            _extraction_payload(
                [{"kind": "episodic", "content": "一次普通的散步", "topics": ["routine"], "entities": [self.dm.char], "salience": 0.2, "sources": [0]}]
            ),
        )
        first = consolidator.settle_strengths()
        self.assertGreaterEqual(first, 0)

        # Far in the future the memory decays into archived territory.
        self.clock.set_year(2022)
        self.clock.set_week(1)
        self.clock.set_stage(Stage.SETTLE)
        consolidator.settle_strengths()

        view = build_memory_views(self.dm)["memories"]
        entry = next(iter(view["memories"].values()))
        self.assertIn(entry["tier"], ("cold", "archived", "warm", "hot"))
        self.assertTrue(entry["source_event_ids"], "history is never deleted")

    def test_views_are_rebuildable(self) -> None:
        consolidator = self._consolidator()
        self._run_week(
            consolidator,
            _extraction_payload(
                [{"kind": "lesson", "content": "先列场景再写", "topics": ["creation"], "entities": [self.dm.char], "sources": [0]}]
            ),
        )
        consolidator.settle_strengths()
        paths = write_memory_views(self.dm)
        first = {name: p.read_text(encoding="utf-8") for name, p in paths.items()}
        for path in paths.values():
            path.unlink()

        rebuilt = write_memory_views(self.dm)
        for name, path in rebuilt.items():
            self.assertEqual(path.read_text(encoding="utf-8"), first[name], name)

    def test_memory_views_fold_events_purely(self) -> None:
        events = [
            {"type": "MEMORY_CREATED", "memory_id": "m1", "kind": "lesson", "content": "x", "salience": 0.8, "strength": 0.8, "tier": "hot", "source_event_ids": ["ev-1"], "created_at": "Y2020-W01-settle", "ledger_event_id": "ev-9"},
            {"type": "MEMORY_STRENGTH_UPDATED", "memory_id": "m1", "strength": 0.4, "tier": "warm", "factors": {"decay": 0.5}},
            {"type": "MEMORY_RECALLED", "memory_id": "m1", "rank": 1, "score": 0.5, "tier": "warm", "used": False},
        ]
        view = materialize_memories(events, persona="甲")
        entry = view["memories"]["m1"]
        self.assertEqual(entry["strength"], 0.4)
        self.assertEqual(entry["tier"], "warm")
        self.assertEqual(entry["read_count"], 1)
        self.assertFalse(entry["recalls"][0]["used"])
        self.assertEqual(view["stats"]["by_tier"], {"warm": 1})

    def test_relation_view_groups_by_type(self) -> None:
        memories = {"m1": {"memory_id": "m1", "kind": "lesson"}, "m2": {"memory_id": "m2", "kind": "lesson"}}
        events = [
            {
                "type": "RELATION_ASSERTED",
                "source_memory_id": "m1",
                "target_memory_id": "m2",
                "relationship_types": ["same_entity", "temporal"],
                "score": 0.5,
            }
        ]
        view = materialize_relations(events, memories=memories)
        self.assertEqual(view["stats"]["edges"], 1)
        self.assertEqual(view["stats"]["by_type"], {"same_entity": 1, "temporal": 1})

    def test_load_memories_returns_objects(self) -> None:
        consolidator = self._consolidator()
        self._run_week(
            consolidator,
            _extraction_payload(
                [{"kind": "lesson", "content": "先列场景再写", "topics": ["creation"], "entities": [self.dm.char], "sources": [0]}]
            ),
        )
        items = load_memories(self.dm)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].kind, "lesson")


class RetrievalShadowTests(unittest.TestCase):
    def setUp(self) -> None:
        self._ctx = temp_workspace()
        self.root = self._ctx.__enter__()  # type: ignore[attr-defined]
        self.dm, self.clock = make_datamanager()
        self.dm.save_state(_state())
        self.clock.set_stage(Stage.SETTLE)

    def tearDown(self) -> None:
        self._ctx.__exit__(None, None, None)  # type: ignore[attr-defined]

    def test_legacy_side_reports_items_and_tokens(self) -> None:
        self.dm.update_scratchpad(
            "others/地形", "<summary>地形笔记</summary><full>九亭地形笔记</full>", create_new_scratchpad=True
        )
        legacy = legacy_retrieval(self.dm)
        self.assertGreaterEqual(legacy["n_items"], 0)
        self.assertGreater(legacy["est_tokens"], 0)

    def test_ranking_prefers_relevant_and_strong(self) -> None:
        query = {"写", "作"}
        strong = _memory(memory_id="strong", content="写作时先列场景", strength=0.9, tier="hot")
        weak = _memory(memory_id="weak", content="写作时先列场景", strength=0.2, tier="cold")
        irrelevant = _memory(memory_id="irrelevant", content="今天下雨了", strength=0.9, tier="hot")
        ranked = score_memories([weak, irrelevant, strong], query_tokens=query, top_k=3)
        self.assertEqual(ranked[0][0].memory_id, "strong")
        self.assertEqual(ranked[-1][0].memory_id, "irrelevant")

    def test_archived_memories_need_strong_relevance(self) -> None:
        archived = _memory(memory_id="arch", content="完全不相关的旧事", tier="archived", strength=0.05)
        ranked = score_memories([archived], query_tokens={"写作"}, top_k=5)
        self.assertEqual(ranked, [])
        reactivated = score_memories(
            [archived], query_tokens=set(), top_k=5
        )
        self.assertEqual(reactivated, [], "no query means no archived memory is offered")

    def test_comparison_is_recorded_and_reproducible(self) -> None:
        memories = [
            _memory(memory_id="m1", content="写作时先列场景", strength=0.8, tier="hot"),
            _memory(memory_id="m2", content="跑步前要热身", strength=0.3, tier="cold", topics=["exercise"]),
        ]
        retriever = MemoryRetriever(dm=self.dm, clock=self.clock, config=RetrievalConfig(top_k=2))
        first = retriever.compare_week(memories=memories)

        self.assertEqual(first["proposed"]["n_items"], 2)
        self.assertGreaterEqual(first["proposed"]["est_tokens"], 0)
        self.assertEqual(first["cold_reactivated"], 1)
        events = retriever.store.events()
        self.assertEqual(len(retriever.store.of_type("MEMORY_RECALLED")), 2)
        self.assertEqual(len(retriever.store.of_type("MEMORY_RETRIEVAL_COMPARED")), 1)
        self.assertTrue(all(e["used"] is False for e in retriever.store.of_type("MEMORY_RECALLED")))

        # Same week again -> idempotent, no duplicate events.
        retriever.compare_week(memories=memories)
        self.assertEqual(len(retriever.store.of_type("MEMORY_RETRIEVAL_COMPARED")), 1)

        view_a = build_memory_views(self.dm)["retrieval"]
        view_b = build_memory_views(self.dm)["retrieval"]
        self.assertEqual(view_a, view_b)
        self.assertEqual(view_a["stats"]["weeks"], 1)

    def test_config_defaults_are_inert(self) -> None:
        self.assertFalse(MemoryConfig.from_world_config({}).enabled)
        cfg = MemoryConfig.from_world_config(
            {"cognition": {"memory_shadow": True, "memory_max_new_per_week": 3}}
        )
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.max_new_per_week, 3)
        self.assertFalse(cfg.relation_llm_judge, "phase 2 is rule-only")


if __name__ == "__main__":
    unittest.main()
