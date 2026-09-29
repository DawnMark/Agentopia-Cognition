"""离线回归：Idea 产出率（KI-15，2026-09-26）。

背景：`data/shanghai_apartment_09261721`（阶段 4 对照运行）里三个角色整年只产出
1 / 5 / 10 条 idea，且集中在头几周。定位到两个结构性原因：

1. `_resource_meets_obstacle()` 只返回**精确相等**的障碍/资源字符串，而 KI-3 之后
   `problem_resource` 关系是按分级亲和度判定的 —— 于是大量候选带着空的
   obstacles/resources 进入打分，complementarity 恒为 0、IdeaPotential 恒为 0，
   必然 `below_min_potential`，却仍占掉每周 4 个措辞槽位之一。
2. 每周预算按 REVIEW 时刻的 vitality 计算（"vitality < 20 → 0"），而本世界的
   vitality 从 70 单调流失到 0、没有任何恢复机制 —— 一年里后半段直接不再产出。

这里钉住修复后的口径：候选载荷、候选排序与每 motif 配额、预算曲线、以及
"本周为什么没有产出"的可审计事件。
"""

from __future__ import annotations

import json
import unittest
from typing import Any, Dict, List, Optional
from unittest import mock

from src.agents.cognition.idea_engine import (
    MAX_CANDIDATES_PER_MOTIF,
    IdeaConfig,
    _candidate_prior,
    _complementarity,
    _resource_meets_obstacle,
    find_motif_candidates,
    weekly_idea_budget,
)
from src.agents.cognition.idea_pipeline import IdeaEngine
from src.agents.cognition.memory_models import MemoryItem, semantic_key
from src.world.clock import Stage
from tests._helpers import make_datamanager, temp_workspace


def _memory(
    *,
    memory_id: str,
    content: str = "内容",
    topics: Optional[List[str]] = None,
    entities: Optional[List[str]] = None,
    goals: Optional[List[str]] = None,
    polarity: str = "positive",
    obstacles: Optional[List[str]] = None,
    resources: Optional[List[str]] = None,
    skills: Optional[List[str]] = None,
) -> MemoryItem:
    return MemoryItem(
        memory_id=memory_id,
        kind="episodic",
        content=content,
        persona="测试角色",
        semantic_key=semantic_key(
            kind="episodic", topics=topics or ["work"], entities=entities or []
        ),
        topics=list(topics or ["work"]),
        entities=list(entities or []),
        goal_ids=list(goals or []),
        skill_ids=list(skills or []),
        source_event_ids=["ev-1"],
        confidence=0.7,
        outcome_polarity=polarity,
        obstacles=list(obstacles or []),
        resources=list(resources or []),
    )


def _edge(source: str, target: str, types: List[str], score: float = 0.3) -> Dict[str, Any]:
    return {"source": source, "target": target, "types": types, "score": score, "shared_focus": "work"}


class CandidatePayloadTests(unittest.TestCase):
    """KI-15 原因 1：候选必须携带真实的障碍/资源文本，否则永远得 0 分。"""

    def test_weak_match_carries_the_real_texts(self) -> None:
        need = _memory(memory_id="n", obstacles=["房间太暗"], topics=["housing"])
        supply = _memory(memory_id="s", resources=["台灯"], topics=["housing"])
        obstacles, resources = _resource_meets_obstacle(need, supply)
        self.assertEqual(obstacles, ["房间太暗"])
        self.assertEqual(resources, ["台灯"])

    def test_containment_match_is_directional(self) -> None:
        need = _memory(memory_id="n", obstacles=["承重墙打孔"], topics=["housing"])
        supply = _memory(memory_id="s", resources=["冲击钻和膨胀螺丝"], topics=["housing"])
        obstacles, resources = _resource_meets_obstacle(need, supply)
        self.assertEqual(obstacles, ["承重墙打孔"])
        self.assertEqual(resources, ["冲击钻和膨胀螺丝"])
        # ... and the reversed pair picks the other direction.
        reversed_pair = _resource_meets_obstacle(
            _memory(memory_id="n2", obstacles=["没有时间"], topics=["work"]),
            _memory(memory_id="s2", resources=["没有时间"], topics=["work"]),
        )
        self.assertEqual(reversed_pair[0], ["没有时间"])

    def test_unrelated_pair_has_no_payload(self) -> None:
        self.assertEqual(
            _resource_meets_obstacle(
                _memory(memory_id="a", topics=["food"]),
                _memory(memory_id="b", topics=["work"]),
            ),
            ([], []),
        )

    def test_candidate_gets_non_zero_complementarity(self) -> None:
        memories = [
            _memory(memory_id="n", obstacles=["房间太暗"], topics=["housing"], goals=["把书房用起来"]),
            _memory(memory_id="s", resources=["台灯"], topics=["housing"], goals=["把书房用起来"]),
        ]
        candidates = find_motif_candidates(
            memories,
            [_edge("n", "s", ["problem_resource", "temporal"])],
            traits={"creativity": 70, "curiosity": 70},
        )
        self.assertTrue(candidates)
        top = candidates[0]
        self.assertEqual(top.motif, "goal_obstacle_resource")
        self.assertTrue(top.obstacles and top.resources)
        index = {m.memory_id: m for m in memories}
        self.assertGreater(_complementarity(top, index), 0.0)

    def test_candidate_without_both_sides_is_never_emitted(self) -> None:
        memories = [
            _memory(memory_id="a", obstacles=[], topics=["housing"]),
            _memory(memory_id="b", resources=[], topics=["housing"]),
        ]
        candidates = find_motif_candidates(
            memories, [_edge("a", "b", ["problem_resource"])]
        )
        self.assertEqual([c for c in candidates if c.motif == "goal_obstacle_resource"], [])


class CandidateRankingTests(unittest.TestCase):
    """KI-15 原因 1 的后果：高边分但注定被拒的候选不得占满措辞槽位。"""

    def _memories(self) -> List[MemoryItem]:
        items = [
            _memory(memory_id="empty-need", obstacles=["没有安静的地方"], topics=["creation"]),
            _memory(memory_id="empty-supply", resources=["咖啡馆"], topics=["creation"]),
            _memory(memory_id="hub", content="晨跑后复盘很顺", polarity="positive", topics=["exercise", "routine"]),
            _memory(memory_id="p2", content="第二周晨跑后又复盘了一遍", polarity="positive", topics=["exercise", "routine"]),
            _memory(memory_id="p3", content="第三周晨跑复盘成了习惯", polarity="positive", topics=["exercise", "routine"]),
        ]
        return items

    def test_a_diverse_motif_still_gets_a_slot(self) -> None:
        memories = self._memories()
        index = {m.memory_id: m for m in memories}
        # Many high-scoring problem_resource edges compete with one repeated_pattern.
        edges = [
            _edge("empty-need", "empty-supply", ["problem_resource"], score=0.9),
            _edge("hub", "p2", ["problem_resource"], score=0.85),
            _edge("hub", "p3", ["problem_resource"], score=0.8),
            _edge("p2", "p3", ["repeated_pattern"], score=0.2),
        ]
        candidates = find_motif_candidates(
            memories, edges, traits={"creativity": 70, "curiosity": 70}, max_candidates=4
        )
        motifs = [c.motif for c in candidates]
        self.assertIn("repeated_pattern", motifs)

    def test_per_motif_share_is_capped(self) -> None:
        """Two loud motifs must not crowd out each other (or a third one)."""
        memories: List[MemoryItem] = []
        edges: List[Dict[str, Any]] = []
        for i in range(4):
            memories.append(_memory(memory_id=f"n{i}", obstacles=[f"障碍{i}"], topics=["work"]))
            memories.append(_memory(memory_id=f"s{i}", resources=[f"资源{i}"], topics=["work"]))
            edges.append(_edge(f"n{i}", f"s{i}", ["problem_resource"], score=0.9 - i * 0.01))
        for i in range(3):
            memories.append(
                _memory(
                    memory_id=f"r{i}",
                    content=f"第 {i} 次晨跑后复盘",
                    topics=["exercise", "routine"],
                )
            )
        edges.append(_edge("r0", "r1", ["repeated_pattern"], score=0.3))
        edges.append(_edge("r1", "r2", ["repeated_pattern"], score=0.3))
        edges.append(_edge("r0", "r2", ["repeated_pattern"], score=0.3))
        candidates = find_motif_candidates(
            memories, edges, traits={"creativity": 70, "curiosity": 70}, max_candidates=4
        )
        from collections import Counter

        counts = Counter(c.motif for c in candidates)
        for motif, count in counts.items():
            self.assertLessEqual(count, MAX_CANDIDATES_PER_MOTIF, motif)
        self.assertLessEqual(len(candidates), 4)
        self.assertIn("repeated_pattern", counts)

    def test_prior_prefers_a_scorable_candidate(self) -> None:
        memories = self._memories()
        index = {m.memory_id: m for m in memories}
        candidates = find_motif_candidates(
            memories,
            [
                _edge("empty-need", "empty-supply", ["problem_resource"], score=0.9),
                _edge("p2", "p3", ["repeated_pattern"], score=0.2),
            ],
            traits={"creativity": 70, "curiosity": 70},
        )
        priors = {c.motif: _candidate_prior(c, index, {"creativity": 70, "curiosity": 70}) for c in candidates}
        self.assertGreater(priors["repeated_pattern"], 0.0)


class BudgetCurveTests(unittest.TestCase):
    """KI-15 原因 2：预算不得因为"世界里的 vitality 一定会流干"而变成 0。"""

    def test_drained_character_still_gets_one_idea(self) -> None:
        config = IdeaConfig(enabled=True, max_ideas_per_week=2)
        for vitality in (0, 1, 19, 39):
            self.assertEqual(weekly_idea_budget(config, vitality=vitality), 1, vitality)

    def test_full_budget_when_rested(self) -> None:
        config = IdeaConfig(enabled=True, max_ideas_per_week=2)
        self.assertEqual(weekly_idea_budget(config, vitality=70), 2)

    def test_explicit_floor_restores_the_hard_stop(self) -> None:
        config = IdeaConfig(enabled=True, max_ideas_per_week=2, min_vitality=20)
        self.assertEqual(weekly_idea_budget(config, vitality=19), 0)
        self.assertEqual(weekly_idea_budget(config, vitality=20), 1)

    def test_config_reads_the_floor_from_world_config(self) -> None:
        cfg = IdeaConfig.from_world_config(
            {"cognition": {"idea_engine": True, "idea_min_vitality": 25}}
        )
        self.assertEqual(cfg.min_vitality, 25.0)


class SkipObservabilityTests(unittest.TestCase):
    """没有产出时必须留下可审计的事件，而不是只写一行日志。"""

    def setUp(self) -> None:
        self._tmp = temp_workspace()
        self._ctx = self._tmp.__enter__()
        self.dm, self.clock = make_datamanager(stage=Stage.REVIEW)

    def tearDown(self) -> None:
        self._tmp.__exit__(None, None, None)

    def _engine(self, **kwargs) -> IdeaEngine:
        return IdeaEngine(
            dm=self.dm,
            clock=self.clock,
            agent_name=self.dm.char,
            model="test-model",
            traits={"creativity": 70, "curiosity": 70, "intelligence": 60, "confidence": 70},
            config=IdeaConfig(enabled=True, **kwargs),
        )

    def test_no_memories_is_recorded(self) -> None:
        engine = self._engine()
        engine.weekly(vitality=80)
        skipped = engine.store.of_type("IDEA_SKIPPED")
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["reason"], "not_enough_memories")

    def test_no_candidates_is_recorded(self) -> None:
        from src.agents.cognition.memory_store import MemoryEventStore

        store = MemoryEventStore(self.dm)
        for i in range(2):
            store.created(
                memory_id=f"mem-{i}",
                week="Y2020-W01",
                payload={
                    "kind": "episodic",
                    "content": f"第 {i} 条",
                    "persona": self.dm.char,
                    "topics": ["work"],
                    "entities": [],
                    "source_event_ids": ["ev-1"],
                    "created_at": "Y2020-W01",
                },
            )
        engine = self._engine()
        with mock.patch("src.utils.get_response_with_retry") as llm:
            counts = engine.weekly(vitality=80)
        llm.assert_not_called()
        self.assertEqual(counts["candidates"], 0)
        skipped = engine.store.of_type("IDEA_SKIPPED")
        self.assertEqual([s["reason"] for s in skipped], ["no_candidates"])

    def test_phrasing_failure_is_recorded(self) -> None:
        memories = [
            _memory(memory_id="n", obstacles=["房间太暗"], topics=["housing"], goals=["用书房"]),
            _memory(memory_id="s", resources=["台灯"], topics=["housing"], goals=["用书房"]),
        ]
        from src.agents.cognition.memory_store import MemoryEventStore

        store = MemoryEventStore(self.dm)
        for memory in memories:
            store.created(
                memory_id=memory.memory_id,
                week="Y2020-W01",
                payload={
                    "kind": memory.kind,
                    "content": memory.content,
                    "persona": self.dm.char,
                    "topics": memory.topics,
                    "entities": [],
                    "obstacles": memory.obstacles,
                    "resources": memory.resources,
                    "goal_ids": memory.goal_ids,
                    "source_event_ids": ["ev-1"],
                    "created_at": "Y2020-W01",
                },
            )
        from src.agents.cognition.memory_store import RelationEventStore
        from src.agents.cognition.memory_models import MemoryRelation

        RelationEventStore(self.dm).asserted(
            MemoryRelation(
                source_memory_id="n",
                target_memory_id="s",
                relationship_types=["problem_resource"],
                score=0.4,
                week="Y2020-W01",
            )
        )
        engine = self._engine()
        with mock.patch("src.utils.get_response_with_retry", side_effect=RuntimeError("boom")):
            engine.weekly(vitality=80)
        skipped = engine.store.of_type("IDEA_SKIPPED")
        self.assertEqual([s["reason"] for s in skipped], ["phrasing_failed"])

    def test_rejection_carries_diagnostics(self) -> None:
        """被拒的候选要能回答"为什么被拒"，否则 below_min_potential 是死胡同。"""
        memories = [
            _memory(memory_id="n", obstacles=["房间太暗"], topics=["housing"], goals=["用书房"]),
            _memory(memory_id="s", resources=["台灯"], topics=["housing"], goals=["用书房"]),
        ]
        from src.agents.cognition.memory_store import MemoryEventStore, RelationEventStore
        from src.agents.cognition.memory_models import MemoryRelation

        store = MemoryEventStore(self.dm)
        for memory in memories:
            store.created(
                memory_id=memory.memory_id,
                week="Y2020-W01",
                payload={
                    "kind": memory.kind,
                    "content": memory.content,
                    "persona": self.dm.char,
                    "topics": memory.topics,
                    "entities": [],
                    "obstacles": memory.obstacles,
                    "resources": memory.resources,
                    "goal_ids": memory.goal_ids,
                    "source_event_ids": ["ev-1"],
                    "created_at": "Y2020-W01",
                },
            )
        RelationEventStore(self.dm).asserted(
            MemoryRelation(
                source_memory_id="n",
                target_memory_id="s",
                relationship_types=["problem_resource"],
                score=0.4,
                week="Y2020-W01",
            )
        )
        engine = self._engine()
        engine.config.min_potential = 0.99  # force a below-threshold rejection
        payload = {
            "ideas": [
                {
                    "candidate_index": 0,
                    "idea_type": "hypothesis",
                    "content": "把台灯搬到书桌上试试",
                    "test_plan": "本周把台灯搬过去，记录是否更容易写作",
                    "requires": {"skills": [], "entities": [], "money": 0},
                    "confidence": 0.3,
                }
            ]
        }
        with mock.patch("src.utils.get_response_with_retry", return_value=payload):
            counts = engine.weekly(vitality=80)
        self.assertEqual(counts["rejected"], 1)
        rejected = engine.store.of_type("IDEA_REJECTED")[0]
        self.assertIn("below_min_potential", rejected["reasons"])
        self.assertIn("potential", rejected)
        self.assertIn("score_components", rejected)
        self.assertIn("complementarity", rejected["score_components"])


if __name__ == "__main__":
    unittest.main()
