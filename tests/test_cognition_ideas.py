"""阶段 3：Idea Engine（离线，无真实 LLM 调用）。

对齐设计文档 §8.6–8.9 与阶段 3 交付：
- 六种 relation motif 的规则识别；
- 五项检查（grounding / novelty / feasibility / testability / safety）作为准入门槛；
- IdeaPotential 乘积评分 + 命名惩罚项；
- 每角色每周 0–2 个 Idea，人格只影响候选范围与筛选；
- Idea 可产生 **candidate** methodology（value 0），绝不直接改变能力；
- 关闭开关即惰性、事件可重建视图。
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

from src.agents.cognition import materializer
from src.agents.cognition.event_store import CapabilityEventStore
from src.agents.cognition.idea_engine import (
    IdeaConfig,
    IdeaWorldState,
    build_idea,
    evaluate_candidate,
    find_motif_candidates,
    weekly_idea_budget,
    world_state_from,
)
from src.agents.cognition.idea_models import (
    IDEA_MOTIFS,
    Idea,
    IdeaCandidate,
    IdeaPotential,
    idea_id_for,
    method_id_for_idea,
)
from src.agents.cognition.idea_pipeline import IdeaEngine
from src.agents.cognition.idea_store import (
    IdeaEventStore,
    build_idea_view,
    load_ideas,
    materialize_ideas,
    write_idea_view,
)
from src.agents.cognition.memory_models import MemoryItem, semantic_key
from src.world.clock import Stage
from tests._helpers import make_datamanager, temp_workspace


def _memory(
    *,
    memory_id: str,
    content: str,
    topics: List[str] | None = None,
    entities: List[str] | None = None,
    goals: List[str] | None = None,
    skills: List[str] | None = None,
    polarity: str = "positive",
    obstacles: List[str] | None = None,
    resources: List[str] | None = None,
    tier: str = "hot",
    confidence: float = 0.7,
    sources: List[str] | None = None,
) -> MemoryItem:
    return MemoryItem(
        memory_id=memory_id,
        kind="episodic",
        content=content,
        persona="测试角色",
        semantic_key=semantic_key(
            kind="episodic", topics=topics or ["work"], entities=entities or ["甲"]
        ),
        topics=list(topics or ["work"]),
        entities=list(entities or ["甲"]),
        goal_ids=list(goals or []),
        skill_ids=list(skills or []),
        source_event_ids=list(sources) if sources is not None else ["ev-1"],
        confidence=confidence,
        salience=0.6,
        strength=0.6,
        tier=tier,
        outcome_polarity=polarity,
        obstacles=list(obstacles or []),
        resources=list(resources or []),
    )


def _edge(source: str, target: str, types: List[str], score: float = 0.5) -> Dict[str, Any]:
    return {
        "source": source,
        "target": target,
        "types": types,
        "score": score,
        "shared_focus": "fitness",
    }


class IdeaModelTests(unittest.TestCase):
    def test_vocabulary_is_closed(self) -> None:
        with self.assertRaises(ValueError):
            Idea(idea_id="i", content="x", idea_type="certainty")
        with self.assertRaises(ValueError):
            Idea(idea_id="i", content="x", motif="not_a_motif")
        self.assertIn("contradiction", IDEA_MOTIFS)

    def test_candidate_confidence_is_capped(self) -> None:
        idea = Idea(idea_id="i", content="x", confidence=0.95)
        self.assertLessEqual(idea.confidence, 0.5, "a hypothesis may not sound certain")

    def test_idea_id_is_content_addressed(self) -> None:
        a = idea_id_for(persona="甲", motif="contradiction", source_memory_ids=["m1", "m2"])
        b = idea_id_for(persona="甲", motif="contradiction", source_memory_ids=["m2", "m1"])
        c = idea_id_for(persona="甲", motif="contradiction", source_memory_ids=["m1", "m3"])
        self.assertEqual(a, b, "order must not matter")
        self.assertNotEqual(a, c)

    def test_method_id_for_idea_is_stable(self) -> None:
        self.assertEqual(
            method_id_for_idea("idea-1", "写作"), method_id_for_idea("idea-1", "写作")
        )
        self.assertNotEqual(
            method_id_for_idea("idea-1", "写作"), method_id_for_idea("idea-2", "写作")
        )

    def test_potential_is_a_product_with_penalties(self) -> None:
        p = IdeaPotential(
            groundedness=1.0,
            complementarity=1.0,
            novelty=1.0,
            actionability=1.0,
            goal_relevance=1.0,
            persona_fit=1.0,
            evidence_confidence=1.0,
        )
        self.assertAlmostEqual(p.potential, 1.0)
        p.penalty_factor = 0.5
        p.penalties = ["duplicates_existing_plan"]
        self.assertAlmostEqual(p.potential, 0.5)


class MotifDetectionTests(unittest.TestCase):
    def test_problem_resource_motif(self) -> None:
        problem = _memory(
            memory_id="p", content="找不到安静地方写作", obstacles=["没有安静的地方"]
        )
        resource = _memory(
            memory_id="r",
            content="图书馆早上很安静",
            resources=["没有安静的地方"],
            entities=["图书馆"],
        )
        candidates = find_motif_candidates(
            [problem, resource], [_edge("p", "r", ["problem_resource"], 0.6)]
        )
        self.assertEqual([c.motif for c in candidates], ["goal_obstacle_resource"])
        self.assertEqual(candidates[0].memory_ids, ["p", "r"])
        self.assertEqual(candidates[0].obstacles, ["没有安静的地方"])

    def test_contradiction_motif(self) -> None:
        a = _memory(memory_id="a", content="早起跑步让我清爽", polarity="positive")
        b = _memory(memory_id="b", content="早起跑步让我疲惫", polarity="negative")
        candidates = find_motif_candidates([a, b], [_edge("a", "b", ["contradiction"], 0.45)])
        self.assertEqual([c.motif for c in candidates], ["contradiction"])

    def test_gap_motif(self) -> None:
        a = _memory(memory_id="a", content="想练耐力但没场地", goals=["提升耐力"], obstacles=["没场地"])
        b = _memory(memory_id="b", content="想练耐力但没时间", goals=["提升耐力"], obstacles=["没时间"])
        candidates = find_motif_candidates([a, b], [_edge("a", "b", ["gap"], 0.4)])
        self.assertEqual([c.motif for c in candidates], ["causal_gap"])

    def test_repeated_pattern_motif_collects_the_family(self) -> None:
        family = [
            _memory(memory_id=f"f{i}", content=f"夜跑后失眠 {i}", polarity="negative", entities=[f"人{i}"])
            for i in range(3)
        ]
        candidates = find_motif_candidates(
            family, [_edge("f0", "f1", ["repeated_pattern"], 0.5)]
        )
        self.assertEqual([c.motif for c in candidates], ["repeated_pattern"])
        self.assertGreaterEqual(len(candidates[0].memory_ids), 3)

    def test_unused_resource_motif(self) -> None:
        resource = _memory(
            memory_id="r",
            content="家里有一台旧跑步机",
            resources=["旧跑步机"],
            goals=["提升耐力"],
        )
        goal = _memory(
            memory_id="g",
            content="耐力一直没有进步",
            goals=["提升耐力"],
            obstacles=["没时间去健身房"],
            polarity="negative",
        )
        # No relation edge between them at all: the motif is found by rule.
        candidates = find_motif_candidates([resource, goal], [])
        self.assertEqual([c.motif for c in candidates], ["unused_resource"])
        self.assertEqual(candidates[0].resources, ["旧跑步机"])

    def test_cross_domain_analogy_needs_creativity(self) -> None:
        a = _memory(memory_id="a", content="写作先列提纲", skills=["写作"], topics=["planning"])
        b = _memory(
            memory_id="b", content="训练先列计划", skills=["跑步"], topics=["planning"], entities=["乙"]
        )
        edge = [_edge("a", "b", ["method_transfer"], 0.3)]
        creative = find_motif_candidates([a, b], edge, traits={"creativity": 90})
        cautious = find_motif_candidates([a, b], edge, traits={"creativity": 5})
        self.assertIn("cross_domain_analogy", [c.motif for c in creative])
        self.assertNotIn("cross_domain_analogy", [c.motif for c in cautious])

    def test_cold_memories_need_curiosity(self) -> None:
        a = _memory(memory_id="a", content="冬天跑步伤膝盖", tier="cold")
        b = _memory(memory_id="b", content="夏天跑步很舒服", tier="cold")
        edge = [_edge("a", "b", ["contradiction"], 0.4)]
        self.assertEqual(find_motif_candidates([a, b], edge, traits={"curiosity": 10}), [])
        self.assertTrue(find_motif_candidates([a, b], edge, traits={"curiosity": 80}))

    def test_candidates_are_deduplicated_and_capped(self) -> None:
        memories = [
            _memory(memory_id=f"m{i}", content=f"记录 {i}", entities=["甲"]) for i in range(6)
        ]
        edges = [_edge("m0", f"m{i}", ["contradiction"], 0.5) for i in range(1, 6)]
        candidates = find_motif_candidates(memories, edges, max_candidates=3)
        self.assertLessEqual(len(candidates), 3)
        keys = [(c.motif, tuple(sorted(c.memory_ids))) for c in candidates]
        self.assertEqual(len(keys), len(set(keys)))


class FiveChecksTests(unittest.TestCase):
    def setUp(self) -> None:
        self.memories = [
            _memory(
                memory_id="p",
                content="找不到安静地方写作",
                obstacles=["没有安静的地方"],
                goals=["完成长篇"],
                entities=["甲"],
            ),
            _memory(
                memory_id="r",
                content="图书馆早上很安静",
                resources=["没有安静的地方"],
                goals=["完成长篇"],
                entities=["图书馆"],
            ),
        ]
        self.candidate = IdeaCandidate(
            motif="goal_obstacle_resource",
            memory_ids=["p", "r"],
            relationship_types=["problem_resource", "same_goal"],
            goal="完成长篇",
            obstacles=["没有安静的地方"],
            resources=["没有安静的地方"],
            score=0.6,
        )
        self.world = IdeaWorldState(
            skills={"写作"},
            entities={"甲", "图书馆"},
            locations={"图书馆"},
            deposit=1000.0,
            vitality=80.0,
        )

    def _evaluate(self, **overrides):
        kwargs = dict(
            content="先口述初稿再整理成文，可能降低启动阻力",
            test_plan="本周挑一天口述一章，观察是否更快进入状态",
            requires={"skills": [], "entities": [], "money": 0},
            memories=self.memories,
            existing_ideas=[],
            existing_methods=[],
            world_state=self.world,
            traits={"creativity": 60, "intelligence": 60},
        )
        kwargs.update(overrides)
        return evaluate_candidate(self.candidate, **kwargs)

    def test_a_good_candidate_passes(self) -> None:
        decision = self._evaluate()
        self.assertTrue(decision.accepted, decision.reasons)
        self.assertGreater(decision.potential.potential, 0)
        self.assertEqual(decision.potential.groundedness, 0.5)

    def test_check_1_grounding_requires_two_sourced_memories(self) -> None:
        thin = [
            _memory(memory_id="p", content="x", obstacles=["没有安静的地方"], sources=[]),
            _memory(memory_id="r", content="y", resources=["没有安静的地方"], sources=["ev-2"]),
        ]
        decision = self._evaluate(memories=thin)
        self.assertFalse(decision.accepted)
        self.assertIn("not_grounded", decision.reasons)

    def test_check_2_novelty_rejects_a_rewrite(self) -> None:
        existing = [
            Idea(
                idea_id="old",
                content="先口述初稿再整理成文，可能降低启动阻力",
                idea_type="hypothesis",
            )
        ]
        decision = self._evaluate(existing_ideas=existing)
        self.assertFalse(decision.accepted)
        self.assertIn("duplicates_existing_idea_or_method", decision.reasons)

    def test_check_2_novelty_also_compares_against_methods(self) -> None:
        class _Method:
            title = "先口述初稿再整理成文"
            description = "可能降低启动阻力"

        decision = self._evaluate(existing_methods=[_Method()])
        self.assertFalse(decision.accepted)
        self.assertIn("duplicates_existing_idea_or_method", decision.reasons)

    def test_check_3_feasibility_rejects_missing_entities(self) -> None:
        decision = self._evaluate(
            requires={"skills": [], "entities": ["不存在的健身房"], "money": 0}
        )
        self.assertFalse(decision.accepted)
        self.assertIn("depends_on_missing_entity", decision.reasons)
        self.assertLess(decision.feasibility, 0.5)

    def test_check_3_rejects_missing_skills_and_money(self) -> None:
        self.assertFalse(
            self._evaluate(requires={"skills": ["潜水"], "entities": [], "money": 0}).accepted
        )
        money = self._evaluate(requires={"skills": [], "entities": [], "money": 999999})
        self.assertFalse(money.accepted)
        self.assertIn("conflicts_with_world_state", money.reasons)

    def test_check_4_testability(self) -> None:
        decision = self._evaluate(test_plan="")
        self.assertFalse(decision.accepted)
        self.assertIn("not_testable", decision.reasons)

    def test_check_5_low_confidence_sources_only_penalise(self) -> None:
        weak = [
            _memory(memory_id="p", content="x", obstacles=["没有安静的地方"], confidence=0.2),
            _memory(memory_id="r", content="y", resources=["没有安静的地方"], confidence=0.2),
        ]
        decision = self._evaluate(memories=weak)
        self.assertIn("sources_low_confidence", decision.reasons)
        self.assertLess(decision.potential.penalty_factor, 1.0)

    def test_asserting_a_fact_is_penalised(self) -> None:
        decision = self._evaluate(
            requires={"skills": [], "entities": [], "money": 0, "asserts_fact": True}
        )
        self.assertIn("conflicts_with_world_state", decision.reasons)

    def test_intelligence_shapes_the_threshold_only(self) -> None:
        low = self._evaluate(traits={"intelligence": 5})
        high = self._evaluate(traits={"intelligence": 95})
        self.assertGreater(low.potential.potential, 0)
        self.assertEqual(low.potential.potential, high.potential.potential)


class BudgetTests(unittest.TestCase):
    def test_vitality_limits_the_budget(self) -> None:
        """Low vitality shrinks the budget; it no longer switches the engine off.

        Changed 2026-09-26 (KI-15): in this world vitality only drains (70 → 0
        over a year, no rest mechanic), so the old `vitality < 20 -> 0` rule
        silently stopped idea generation for the second half of every run.
        A drained character now thinks once a week; a hard stop is opt-in via
        `idea_min_vitality`.
        """
        config = IdeaConfig(enabled=True, max_ideas_per_week=2)
        self.assertEqual(weekly_idea_budget(config, vitality=90), 2)
        self.assertEqual(weekly_idea_budget(config, vitality=30), 1)
        self.assertEqual(weekly_idea_budget(config, vitality=10), 1)
        self.assertEqual(weekly_idea_budget(config, vitality=0), 1)

    def test_opt_in_vitality_floor_can_still_stop_the_engine(self) -> None:
        config = IdeaConfig(enabled=True, max_ideas_per_week=2, min_vitality=20)
        self.assertEqual(weekly_idea_budget(config, vitality=10), 0)
        self.assertEqual(weekly_idea_budget(config, vitality=25), 1)

    def test_low_confidence_limits_the_budget(self) -> None:
        config = IdeaConfig(enabled=True, max_ideas_per_week=2)
        self.assertEqual(weekly_idea_budget(config, vitality=90, traits={"confidence": 10}), 1)

    def test_config_defaults_are_inert(self) -> None:
        self.assertFalse(IdeaConfig.from_world_config({}).enabled)
        cfg = IdeaConfig.from_world_config({"cognition": {"idea_engine": True, "idea_max_per_week": 1}})
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.max_ideas_per_week, 1)
        self.assertEqual(
            IdeaConfig.from_world_config(
                {"cognition": {"idea_engine": True, "idea_min_potential": "x"}}
            ).min_potential,
            0.05,
        )


class IdeaStoreTests(unittest.TestCase):
    def test_materialize_folds_events(self) -> None:
        events = [
            {
                "type": "IDEA_CREATED",
                "idea_id": "idea-1",
                "content": "先口述再写",
                "idea_type": "hypothesis",
                "motif": "contradiction",
                "status": "candidate",
                "source_memory_ids": ["m1", "m2"],
                "potential": 0.2,
                "week": "Y2020-W01",
                "ledger_event_id": "ev-1",
            },
            {"type": "IDEA_SCORED", "idea_id": "idea-1", "potential": 0.2},
            {"type": "IDEA_CONVERTED", "idea_id": "idea-1", "candidate_method_id": "method-idea-x"},
            {"type": "IDEA_REJECTED", "idea_id": "idea-2", "reasons": ["not_testable"]},
        ]
        view = materialize_ideas(events, persona="甲")
        self.assertEqual(view["stats"]["ideas"], 1)
        self.assertEqual(view["stats"]["rejected"], 1)
        self.assertEqual(view["stats"]["by_motif"], {"contradiction": 1})
        self.assertEqual(view["ideas"]["idea-1"]["converted_method_id"], "method-idea-x")

    def test_conversion_is_visible_from_the_capability_stream(self) -> None:
        events = [
            {
                "type": "IDEA_CREATED",
                "idea_id": "idea-1",
                "content": "x",
                "idea_type": "hypothesis",
                "status": "candidate",
                "source_memory_ids": ["m1", "m2"],
            }
        ]
        capability = [
            {
                "type": "METHOD_PROPOSED",
                "method_id": "method-idea-1",
                "source_type": "idea_conversion",
                "source_idea_id": "idea-1",
            }
        ]
        view = materialize_ideas(events, capability, persona="甲")
        self.assertEqual(view["ideas"]["idea-1"]["converted_method_id"], "method-idea-1")
        self.assertEqual(view["stats"]["converted_from_capability"], ["idea-1"])


class IdeaPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self._ctx = temp_workspace()
        self.root = self._ctx.__enter__()  # type: ignore[attr-defined]
        self.dm, self.clock = make_datamanager()
        self.dm.save_state(
            {
                "vitality": 80,
                "fulfillment": {"mood": 50, "material": 50, "social": 50, "esteem": 50},
                "assets": {"deposit": 1000, "possessions": [{"name": "旧跑步机"}]},
                "skills": {"写作": 12, "跑步": 30},
            }
        )
        self.clock.set_stage(Stage.REVIEW)

    def tearDown(self) -> None:
        self._ctx.__exit__(None, None, None)  # type: ignore[attr-defined]

    def _seed_memories(self) -> None:
        from src.agents.cognition.memory_store import MemoryEventStore

        store = MemoryEventStore(self.dm)
        for memory in (
            _memory(
                memory_id="mem-p",
                content="写作时找不到安静的地方，一直拖延",
                obstacles=["没有安静的地方"],
                goals=["完成长篇"],
                topics=["creation"],
                sources=["ev-1"],
            ),
            _memory(
                memory_id="mem-r",
                content="图书馆早上很安静，效率很高",
                resources=["没有安静的地方"],
                goals=["完成长篇"],
                topics=["creation"],
                entities=["图书馆"],
                sources=["ev-2"],
            ),
        ):
            payload = memory.to_dict()
            memory_id = payload.pop("memory_id")
            store.created(memory_id=memory_id, week="Y2020-W01", payload=payload)
        from src.agents.cognition.relation_graph import build_relations

        relations = build_relations(
            [
                _memory(
                    memory_id="mem-p",
                    content="写作时找不到安静的地方，一直拖延",
                    obstacles=["没有安静的地方"],
                    goals=["完成长篇"],
                    topics=["creation"],
                    sources=["ev-1"],
                ),
                _memory(
                    memory_id="mem-r",
                    content="图书馆早上很安静，效率很高",
                    resources=["没有安静的地方"],
                    goals=["完成长篇"],
                    topics=["creation"],
                    entities=["图书馆"],
                    sources=["ev-2"],
                ),
            ],
            week="Y2020-W01",
        )
        from src.agents.cognition.memory_store import RelationEventStore

        rel_store = RelationEventStore(self.dm)
        for relation in relations:
            rel_store.asserted(relation)

    def _engine(self, *, enabled: bool = True, max_ideas: int = 2) -> IdeaEngine:
        return IdeaEngine(
            dm=self.dm,
            clock=self.clock,
            agent_name=self.dm.char,
            model="test-model",
            traits={"creativity": 70, "curiosity": 70, "intelligence": 60, "confidence": 70},
            config=IdeaConfig(enabled=enabled, max_ideas_per_week=max_ideas),
        )

    def _llm_payload(self, content: str = "先口述初稿再整理成文，可能降低启动阻力") -> Dict[str, Any]:
        return {
            "ideas": [
                {
                    "candidate_index": 0,
                    "idea_type": "hypothesis",
                    "content": content,
                    "test_plan": "本周挑一天口述一章，观察是否更容易开始",
                    "skills": ["写作"],
                    "requires": {"skills": [], "entities": [], "money": 0},
                    "confidence": 0.4,
                }
            ]
        }

    # -- gating ------------------------------------------------------------
    def test_disabled_engine_writes_nothing(self) -> None:
        self._seed_memories()
        engine = self._engine(enabled=False)
        with mock.patch("src.utils.get_response_with_retry") as llm:
            counts = engine.weekly(vitality=80)
        llm.assert_not_called()
        self.assertEqual(sum(counts.values()), 0)
        self.assertFalse(engine.store.path.exists())

    def test_no_memories_means_no_call(self) -> None:
        engine = self._engine()
        with mock.patch("src.utils.get_response_with_retry") as llm:
            counts = engine.weekly(vitality=80)
        llm.assert_not_called()
        self.assertEqual(counts["candidates"], 0)

    def test_low_vitality_still_generates_one_idea(self) -> None:
        """KI-15: a tired character still gets one idea, and it is recorded."""
        self._seed_memories()
        engine = self._engine()
        with mock.patch(
            "src.utils.get_response_with_retry", return_value=self._llm_payload()
        ):
            counts = engine.weekly(vitality=10)
        self.assertEqual(counts["stored"], 1)
        self.assertEqual(engine.store.of_type("IDEA_SKIPPED"), [])

    def test_opt_in_vitality_floor_skips_generation_and_records_it(self) -> None:
        self._seed_memories()
        engine = self._engine()
        engine.config.min_vitality = 20
        with mock.patch("src.utils.get_response_with_retry") as llm:
            counts = engine.weekly(vitality=10)
        llm.assert_not_called()
        self.assertEqual(counts["stored"], 0)
        skipped = engine.store.of_type("IDEA_SKIPPED")
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["reason"], "no_budget")

    # -- generation --------------------------------------------------------
    def test_a_grounded_idea_is_stored_and_scored(self) -> None:
        self._seed_memories()
        engine = self._engine()
        with mock.patch("src.utils.get_response_with_retry", return_value=self._llm_payload()):
            counts = engine.weekly(vitality=80)

        self.assertEqual(counts["stored"], 1)
        created = engine.store.of_type("IDEA_CREATED")[0]
        self.assertEqual(created["status"], "candidate")
        self.assertLessEqual(created["confidence"], 0.5)
        self.assertEqual(len(created["source_memory_ids"]), 2)
        self.assertTrue(created["test_plan"])
        self.assertGreater(created["potential"], 0)
        self.assertEqual(len(engine.store.of_type("IDEA_SCORED")), 1)

    def test_weekly_call_budget(self) -> None:
        """One phrasing call per week, plus at most one methodisation call (KI-6)."""
        self._seed_memories()
        engine = self._engine()
        payload = self._llm_payload()
        with mock.patch("src.utils.get_response_with_retry", return_value=payload) as llm:
            engine.weekly(vitality=80)
            after_first = llm.call_count
            engine.weekly(vitality=80)  # same week: both budgets refuse
            after_second = llm.call_count
        self.assertEqual(after_first, 2)  # phrasing + methodisation
        self.assertEqual(after_second, after_first)

    def test_weekly_cap_is_enforced(self) -> None:
        self._seed_memories()
        engine = self._engine(max_ideas=1)
        payload = {
            "ideas": [
                {
                    "candidate_index": 0,
                    "idea_type": "hypothesis",
                    "content": f"尝试方案 {i}",
                    "test_plan": f"本周试第 {i} 种做法",
                    "requires": {"skills": [], "entities": [], "money": 0},
                    "confidence": 0.3,
                }
                for i in range(3)
            ]
        }
        with mock.patch("src.utils.get_response_with_retry", return_value=payload):
            counts = engine.weekly(vitality=80)
        self.assertEqual(counts["stored"], 1)

    def test_ungrounded_idea_is_rejected_and_recorded(self) -> None:
        self._seed_memories()
        engine = self._engine()
        payload = self._llm_payload()
        payload["ideas"][0]["requires"] = {
            "skills": [],
            "entities": ["不存在的地方"],
            "money": 0,
        }
        with mock.patch("src.utils.get_response_with_retry", return_value=payload):
            counts = engine.weekly(vitality=80)

        self.assertEqual(counts["stored"], 0)
        self.assertEqual(counts["rejected"], 1)
        rejected = engine.store.of_type("IDEA_REJECTED")[0]
        self.assertIn("depends_on_missing_entity", rejected["reasons"])

    def test_a_bad_model_answer_is_not_fatal(self) -> None:
        self._seed_memories()
        engine = self._engine()
        with mock.patch("src.utils.get_response_with_retry", side_effect=RuntimeError("boom")):
            self.assertEqual(engine.weekly(vitality=80)["stored"], 0)
        with mock.patch("src.utils.get_response_with_retry", return_value=None):
            self.assertEqual(engine.weekly(vitality=80)["stored"], 0)

    def test_same_evidence_yields_the_same_idea_id(self) -> None:
        """Re-deriving an idea from the same memories is idempotent."""
        self._seed_memories()
        engine = self._engine()
        payload = self._llm_payload()
        with mock.patch("src.utils.get_response_with_retry", return_value=payload):
            engine.weekly(vitality=80)
        first = engine.store.of_type("IDEA_CREATED")[0]

        self.clock.set_week(2)
        engine2 = self._engine()
        with mock.patch("src.utils.get_response_with_retry", return_value=payload):
            counts = engine2.weekly(vitality=80)

        created = engine2.store.of_type("IDEA_CREATED")
        self.assertEqual(len(created), 1, "the same idea must not be created twice")
        self.assertEqual(counts["stored"], 0)

    # -- conversion --------------------------------------------------------
    def test_idea_becomes_a_candidate_methodology_with_zero_value(self) -> None:
        self._seed_memories()
        engine = self._engine()
        with mock.patch("src.utils.get_response_with_retry", return_value=self._llm_payload()):
            counts = engine.weekly(vitality=80)

        self.assertEqual(counts["converted"], 1)
        methods = CapabilityEventStore(self.dm).of_type("METHOD_PROPOSED")
        self.assertEqual(len(methods), 1)
        method = methods[0]
        self.assertEqual(method["source_type"], "idea_conversion")
        self.assertEqual(method["status"], "proposed")
        self.assertEqual(method["global_value"], 0.0)
        self.assertEqual(method["confidence"], 0.0)
        self.assertTrue(method["source_idea_id"])
        self.assertEqual(len(engine.store.of_type("IDEA_CONVERTED")), 1)

        # The capability view shows a proposal, never evidence.
        view = materializer.build_capability_view(self.dm)
        entry = next(iter(view["methodologies"].values()))
        self.assertEqual(entry["status"], "proposed")
        self.assertEqual(entry["practice_count"], 0)

    def test_conversion_is_idempotent(self) -> None:
        self._seed_memories()
        engine = self._engine()
        payload = self._llm_payload()
        with mock.patch("src.utils.get_response_with_retry", return_value=payload):
            engine.weekly(vitality=80)
        self.clock.set_week(3)
        engine2 = self._engine()
        with mock.patch("src.utils.get_response_with_retry", return_value=payload):
            engine2.weekly(vitality=80)
        self.assertEqual(len(CapabilityEventStore(self.dm).of_type("METHOD_PROPOSED")), 1)

    # -- views and ledger --------------------------------------------------
    def test_view_is_rebuildable(self) -> None:
        self._seed_memories()
        engine = self._engine()
        with mock.patch("src.utils.get_response_with_retry", return_value=self._llm_payload()):
            engine.weekly(vitality=80)

        path = write_idea_view(self.dm)
        first = path.read_text(encoding="utf-8")
        path.unlink()
        rebuilt = write_idea_view(self.dm)
        self.assertEqual(rebuilt.read_text(encoding="utf-8"), first)
        self.assertEqual(build_idea_view(self.dm)["stats"]["ideas"], 1)
        self.assertEqual(len(load_ideas(self.dm)), 1)

    def test_idea_generation_never_writes_to_the_simulation_ledger(self) -> None:
        self._seed_memories()
        watched = [
            self.dm.root / "activity.jsonl",
            self.dm.root / "state.jsonl",
            self.dm.root / "schedule.jsonl",
        ]
        before = {p: p.read_text(encoding="utf-8") for p in watched if p.exists()}

        engine = self._engine()
        with mock.patch("src.utils.get_response_with_retry", return_value=self._llm_payload()):
            engine.weekly(vitality=80)

        for path, content in before.items():
            self.assertEqual(path.read_text(encoding="utf-8"), content, f"{path.name} changed")

    def test_skill_numbers_are_untouched(self) -> None:
        self._seed_memories()
        before = self.dm.read_state(exclude_cur_t=False)["skills"]
        engine = self._engine()
        with mock.patch("src.utils.get_response_with_retry", return_value=self._llm_payload()):
            engine.weekly(vitality=80)
        self.assertEqual(self.dm.read_state(exclude_cur_t=False)["skills"], before)

    def test_world_state_reads_the_live_world(self) -> None:
        world = world_state_from(self.dm)
        self.assertIn("写作", world.skills)
        self.assertIn("旧跑步机", world.entities)
        self.assertAlmostEqual(world.deposit, 1000.0)
        self.assertEqual(world.missing_entities(["旧跑步机"]), [])
        self.assertEqual(world.missing_entities(["不存在"]), ["不存在"])


if __name__ == "__main__":
    unittest.main()
