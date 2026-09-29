"""阶段 4：被动方法提示与采用追踪（离线，无真实 LLM 调用）。

阶段 4 是第一个改变角色**看到什么**的阶段，因此边界必须可测：
- 提示只是菜单：渲染文本明确写"可选、可忽略"，不写任何强制措辞；
- **被提供不算强化**（不变量 #6）：`METHOD_HINTED` 不动任何数值；
- 采用是角色自己的声明：只有它写了 `<method>…</method>` 才记录；
- **真实采用与反事实证据可分离**：被真实采用占用的活动，影子（阶段 1）不再对同一活动
  做反事实归因；两类证据都带 `attribution` / `evidence_kind`；
- 采用会把方法背后的记忆标为"用于计划"（KI-10），从而让 strength 公式里那一项真正生效。
"""

from __future__ import annotations

import json
import unittest
from typing import Any, Dict, List
from unittest import mock

from src.agents.cognition import materializer
from src.agents.cognition.event_store import CapabilityEventStore
from src.agents.cognition.idea_models import method_id_for_idea
from src.agents.cognition.memory_store import MemoryEventStore
from src.agents.cognition.memory_strength import compute_strength, settle_strengths
from src.agents.cognition.memory_views import build_memory_views
from src.agents.cognition.method_hints import (
    HintConfig,
    MethodHintProvider,
    normalize_title,
)
from src.agents.cognition.models import Methodology
from src.world.clock import Stage
from tests._helpers import make_datamanager, temp_workspace


def _state() -> Dict[str, Any]:
    return {
        "vitality": 80,
        "fulfillment": {"mood": 50, "material": 50, "social": 50, "esteem": 50},
        "assets": {"deposit": 1000, "possessions": []},
        "skills": {"写作": 12, "跑步": 30},
    }


def _method(
    *,
    method_id: str = "method-a",
    title: str = "先定冲突再写场景",
    skill_id: str = "写作",
    status: str = "tested",
    value: float = 0.4,
    confidence: float = 0.5,
    practice: int = 3,
    success: int = 2,
    steps: List[str] | None = None,
    contexts: List[str] | None = None,
    source_memories: List[str] | None = None,
    idea_id: str = "",
) -> Methodology:
    return Methodology(
        method_id=method_id,
        skill_id=skill_id,
        title=title,
        description="复杂写作任务中先确定目标与冲突",
        status=status,
        global_value=value,
        confidence=confidence,
        practice_count=practice,
        success_count=success,
        steps=list(steps) if steps is not None else ["明确目标", "列出冲突", "排场景顺序"],
        checks=["每个场景是否推动冲突"],
        failure_modes=["规划过度导致迟迟不开始"],
        applicable_contexts=list(contexts or ["long_form", "complex_structure"]),
        source_memory_ids=list(source_memories or []),
        source_idea_id=idea_id or None,
    )


class HintProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._ctx = temp_workspace()
        self.root = self._ctx.__enter__()  # type: ignore[attr-defined]
        self.dm, self.clock = make_datamanager()
        self.dm.save_state(_state())
        self.clock.set_stage(Stage.PLAN)

    def tearDown(self) -> None:
        self._ctx.__exit__(None, None, None)  # type: ignore[attr-defined]

    def _seed(self, *methods: Methodology, incomplete: bool = False) -> None:
        """Seed the capability stream the way the real writers do.

        `METHOD_PROPOSED` carries the structure; the authoritative numbers come
        from `METHOD_VALUE_UPDATED` (the view deliberately ignores value/status
        on a proposal, because a proposal is not evidence).
        """
        store = CapabilityEventStore(self.dm)
        for method in methods:
            payload = method.to_dict()
            method_id = payload.pop("method_id")
            if incomplete:
                payload["structure_incomplete"] = True
                payload["structure_missing"] = ["checks"]
            store.append(
                "METHOD_PROPOSED",
                method_id=method_id,
                payload=payload,
                idempotency_key=f"METHOD_PROPOSED:{method_id}:-:seed",
            )
            if method.practice_count or method.global_value or method.status != "proposed":
                store.append(
                    "METHOD_VALUE_UPDATED",
                    method_id=method_id,
                    payload={
                        "global_value": method.global_value,
                        "confidence": method.confidence,
                        "practice_count": method.practice_count,
                        "success_count": method.success_count,
                        "status": method.status,
                        "context_values": dict(method.context_values),
                    },
                    idempotency_key=f"METHOD_VALUE_UPDATED:{method_id}:-:seed",
                )

    def _provider(self, **overrides) -> MethodHintProvider:
        config = HintConfig(
            enabled=overrides.pop("enabled", True),
            top_k=overrides.pop("top_k", 3),
            max_chars=overrides.pop("max_chars", 1400),
            exploration_slots=overrides.pop("exploration_slots", 1),
        )
        return MethodHintProvider(
            dm=self.dm,
            clock=self.clock,
            agent_name=self.dm.char,
            traits={"creativity": 60, "curiosity": 60},
            config=config,
            language="zh",
        )

    # -- selection ---------------------------------------------------------
    def test_own_untried_idea_gets_the_exploration_slot(self) -> None:
        """记忆 → idea → 方法 → 行动：自己想到的办法必须先有机会被自己用上。

        Measured on run 09261617: only 1-2 of 26 offers were idea-derived, so the
        loop this phase exists for barely fired. The exploration slot now prefers
        the character's own, never-offered ideas.
        """
        practice = [_method(method_id=f"m-practice-{i}", value=0.6, status="validated") for i in range(3)]
        own = _method(
            method_id="method-idea-own",
            title="练一休一写进周计划",
            status="proposed",
            value=0.0,
            confidence=0.0,
            practice=0,
            success=0,
            source_memories=["mem-1", "mem-2"],
            idea_id="idea-1",
        )
        own.source_type = "idea_conversion"
        self._seed(*practice, own)
        provider = self._provider(exploration_slots=1)
        offers = provider.select_offers()
        explored = [o for o in offers if o.role == "explore"]
        self.assertEqual(len(explored), 1)
        self.assertEqual(explored[0].method_id, "method-idea-own")
        self.assertEqual(explored[0].origin, "own_idea")
        self.assertEqual(explored[0].source_memory_ids, ["mem-1", "mem-2"])

    def test_own_idea_is_prioritised_only_until_it_has_been_offered(self) -> None:
        own = _method(
            method_id="method-idea-own",
            title="练一休一写进周计划",
            status="proposed",
            value=0.0,
            confidence=0.0,
            practice=0,
            success=0,
        )
        own.source_type = "idea_conversion"
        self._seed(own)
        provider = self._provider()
        first = provider.prepare_week()  # writes METHOD_HINTED
        self.assertIn("你自己最近想到的做法", first)
        self.assertIn("method-idea-own", provider._offered_method_ids())
        # A second week may still offer it, but no longer *because* it is new.
        provider2 = self._provider()
        provider2.select_offers()
        self.assertIn("method-idea-own", provider2._offered_method_ids())

    def test_menu_marks_the_origin_only_for_own_ideas(self) -> None:
        practice = _method(method_id="m-practice", value=0.5)
        own = _method(method_id="method-idea-x", title="自己的假设", status="proposed", value=0.0, practice=0, success=0)
        own.source_type = "idea_conversion"
        self._seed(practice, own)
        provider = self._provider(top_k=3, exploration_slots=1)
        block = provider.prepare_week()
        self.assertIn("自己的假设（你自己最近想到的做法）", block)
        self.assertNotIn("先定冲突再写场景（你自己最近想到的做法）", block)

    def test_no_methods_means_no_hint_block(self) -> None:
        provider = self._provider()
        self.assertEqual(provider.select_offers(), [])
        self.assertEqual(provider.prepare_week(), "")

    def test_disabled_provider_is_inert(self) -> None:
        self._seed(_method())
        store = CapabilityEventStore(self.dm)
        before = len(store.events())

        provider = self._provider(enabled=False)
        provider.prepare_week()
        self.assertIsNone(provider.record_adoption("<method>先定冲突再写场景</method>"))
        self.assertFalse(provider.observe_activity({"activity_id": "a1"}))

        self.assertEqual(len(store.events()), before, "a disabled provider writes nothing")

    def test_top_k_is_respected_and_best_method_leads(self) -> None:
        self._seed(
            _method(method_id="m-best", title="最好", value=0.9, confidence=0.9, practice=10, success=9),
            _method(method_id="m-two", title="第二", value=0.5),
            _method(method_id="m-three", title="第三", value=0.3),
            _method(method_id="m-four", title="第四", value=0.1, status="validated"),
        )
        provider = self._provider(top_k=2, exploration_slots=0)
        offers = provider.select_offers()
        self.assertEqual(len(offers), 2)
        self.assertEqual(offers[0].method_id, "m-best")
        self.assertEqual(offers[0].role, "exploit")

    def test_incomplete_or_unusable_methods_are_not_offered(self) -> None:
        self._seed(_method(method_id="m-incomplete", title="残缺"), incomplete=True)
        provider = self._provider()
        self.assertEqual(provider.select_offers(), [])

        self.dm.root.joinpath("cognition", "capability_events.jsonl").unlink()
        self._seed(_method(method_id="m-nosteps", title="没步骤", steps=[]))
        provider2 = self._provider()
        self.assertEqual(provider2.select_offers(), [])

        self.dm.root.joinpath("cognition", "capability_events.jsonl").unlink()
        self._seed(_method(method_id="m-archived", title="已归档", status="archived"))
        provider3 = self._provider()
        self.assertEqual(provider3.select_offers(), [])

    def test_exploration_offers_an_unproven_method(self) -> None:
        self._seed(
            _method(method_id="m-proven", title="已验证", status="validated", value=0.9, practice=10, success=9),
            _method(method_id="m-fresh", title="新方法", status="proposed", value=0.0, confidence=0.0, practice=0, success=0),
        )
        provider = self._provider(top_k=2, exploration_slots=1)
        offers = provider.select_offers()
        roles = {o.method_id: o.role for o in offers}
        self.assertEqual(roles.get("m-proven"), "exploit")
        self.assertEqual(roles.get("m-fresh"), "explore")

    def test_selection_is_deterministic(self) -> None:
        self._seed(
            _method(method_id="m1", title="一", status="proposed", value=0.1),
            _method(method_id="m2", title="二", status="proposed", value=0.2),
            _method(method_id="m3", title="三", status="proposed", value=0.3),
        )
        first = [o.method_id for o in self._provider().select_offers()]
        second = [o.method_id for o in self._provider().select_offers()]
        self.assertEqual(first, second)

    # -- rendering ---------------------------------------------------------
    def test_render_is_a_menu_not_an_order(self) -> None:
        from src.agents.cognition.method_hints import HintOffer

        method = _method(method_id="m1", title="先定冲突再写场景")
        offer = HintOffer(
            method_id=method.method_id,
            title=method.title,
            skill_id=method.skill_id,
            status=method.status,
            role="exploit",
            score=0.5,
            steps=list(method.steps),
            checks=list(method.checks),
            failure_modes=list(method.failure_modes),
            applicable_contexts=list(method.applicable_contexts),
        )
        block = self._provider().render([offer])

        self.assertIn("先定冲突再写场景", block)
        self.assertIn("明确目标", block)
        self.assertIn("自检", block)
        self.assertIn("常见失败", block)
        self.assertIn("用不用完全由你决定", block)
        self.assertIn("<method>", block, "the adoption format is shown to the character")
        # It must read as a menu: no obligation language.
        for forbidden in ("你必须", "你应该采用", "按要求使用", "务必"):
            self.assertNotIn(forbidden, block)

    def test_render_respects_the_character_budget(self) -> None:
        from src.agents.cognition.method_hints import HintOffer

        offer = HintOffer(
            method_id="m1",
            title="很长的标题" * 20,
            skill_id="写作",
            status="tested",
            role="exploit",
            score=0.5,
            steps=["步骤" * 40],
        )
        provider = self._provider(max_chars=300)
        block = provider.render([offer])
        self.assertLessEqual(len(block), 320)
        self.assertIn("…", block)

    def test_english_world_gets_the_english_block(self) -> None:
        from src.agents.cognition.method_hints import HintOffer

        provider = MethodHintProvider(
            dm=self.dm,
            clock=self.clock,
            agent_name=self.dm.char,
            config=HintConfig(enabled=True),
            language="en",
        )
        block = provider.render(
            [HintOffer(method_id="m", title="Outline first", skill_id="writing", status="tested", role="exploit", score=0.4)]
        )
        self.assertIn("optional, not an order", block)

    # -- offers are recorded, and do not reinforce -------------------------
    def test_offering_records_events_without_touching_value(self) -> None:
        self._seed(_method(method_id="m1", title="方法一", value=0.4))
        provider = self._provider()
        block = provider.prepare_week()

        self.assertTrue(block)
        events = provider.capability.events()
        hinted = [e for e in events if e["type"] == "METHOD_HINTED"]
        self.assertEqual(len(hinted), 1)
        self.assertEqual(hinted[0]["week"], "Y2020-W01")
        self.assertIn("block_chars", hinted[0])
        # Invariant #6: being offered moves nothing.
        self.assertEqual(
            [
                e
                for e in events
                if e["type"] == "METHOD_VALUE_UPDATED" and e.get("week") == "Y2020-W01"
            ],
            [],
        )
        view = materializer.build_capability_view(self.dm)
        self.assertEqual(next(iter(view["methodologies"].values()))["global_value"], 0.4)

    def test_offers_are_idempotent_within_a_week(self) -> None:
        self._seed(_method(method_id="m1", title="方法一"))
        provider = self._provider()
        provider.prepare_week()
        provider.prepare_week()
        self.assertEqual(len(provider.capability.of_type("METHOD_HINTED")), 1)

    # -- adoption ----------------------------------------------------------
    def test_no_declaration_means_no_adoption(self) -> None:
        self._seed(_method(method_id="m1", title="方法一"))
        provider = self._provider()
        provider.prepare_week()
        self.assertIsNone(provider.record_adoption("这周就按老样子过，没打算试新方法。"))
        self.assertEqual(provider.capability.of_type("METHOD_SELECTED"), [])

    def test_declared_title_is_recorded_as_a_real_selection(self) -> None:
        self._seed(_method(method_id="m1", title="先定冲突再写场景", source_memories=["mem-1", "mem-2"], idea_id="idea-1"))
        provider = self._provider()
        provider.prepare_week()

        adoption = provider.record_adoption("计划：写第三章。<method>先定冲突再写场景</method>")

        self.assertIsNotNone(adoption)
        selected = provider.capability.of_type("METHOD_SELECTED")[0]
        self.assertFalse(selected["shadow"], "a real adoption is not a shadow selection")
        self.assertEqual(selected["source"], "hint")
        self.assertTrue(selected["adoption"])
        self.assertEqual(selected["matched_by"], "title")

    def test_declared_id_is_recorded(self) -> None:
        self._seed(_method(method_id="method-idea-abc", title="标题"))
        provider = self._provider()
        provider.prepare_week()
        adoption = provider.record_adoption("<method>method-idea-abc</method>")
        self.assertIsNotNone(adoption)
        self.assertEqual(adoption.matched_by, "id")

    def test_an_unoffered_method_cannot_be_adopted(self) -> None:
        self._seed(_method(method_id="m1", title="方法一"))
        provider = self._provider()
        provider.prepare_week()
        self.assertIsNone(provider.record_adoption("<method>我自己想出来的方法</method>"))
        self.assertEqual(provider.capability.of_type("METHOD_SELECTED"), [])

    def test_adoption_marks_the_source_memories_as_used(self) -> None:
        self._seed(
            _method(method_id="m1", title="先定冲突", source_memories=["mem-1", "mem-2"], idea_id="idea-1")
        )
        provider = self._provider()
        provider.prepare_week()
        provider.record_adoption("<method>先定冲突</method>")

        used = provider.memories.of_type("MEMORY_USED_IN_PLAN")
        self.assertEqual({e["memory_id"] for e in used}, {"mem-1", "mem-2"})
        self.assertEqual(used[0]["method_id"], "m1")

        # Idempotent: adopting the same method again does not double count.
        provider.record_adoption("<method>先定冲突</method>")
        self.assertEqual(len(provider.memories.of_type("MEMORY_USED_IN_PLAN")), 2)

    def test_normalize_title_is_punctuation_insensitive(self) -> None:
        self.assertEqual(normalize_title("先定冲突，再写场景。"), normalize_title("先定冲突 再写场景"))

    # -- practice attribution ---------------------------------------------
    def _record(self, *, activity_id: str = "solo-W01-D2", gains=None, vitality: int = -2):
        return {
            "type": "solo",
            "activity_id": activity_id,
            "time": "Y2020-W01-activity-D2",
            "content": "写小说",
            "outcome": {
                "outcome": "写完一章",
                "delta_vitality": vitality,
                "delta_fulfillment": {"mood": 2},
                "delta_skills": dict(gains if gains is not None else {"写作": 3}),
                "delta_money": 0,
            },
        }

    def _adopt(self, provider: MethodHintProvider, title: str = "先定冲突") -> None:
        provider.prepare_week()
        provider.record_adoption(f"<method>{title}</method>")

    def test_an_activity_without_adoption_is_left_to_the_shadow(self) -> None:
        self._seed(_method(method_id="m1", title="先定冲突"))
        provider = self._provider()
        provider.prepare_week()  # offers, but the character said nothing
        self.assertFalse(provider.observe_activity(self._record()))

    def test_a_relevant_activity_is_claimed_and_attributed_for_real(self) -> None:
        self._seed(_method(method_id="m1", title="先定冲突", skill_id="写作"))
        provider = self._provider()
        self._adopt(provider)

        claimed = provider.observe_activity(self._record(gains={"写作": 3}))

        self.assertTrue(claimed)
        applied = provider.capability.of_type("METHOD_APPLIED")[0]
        self.assertEqual(applied["activity_id"], "solo-W01-D2")
        self.assertEqual(applied["source"], "hint")
        observed = provider.capability.of_type("METHOD_OUTCOME_OBSERVED")[0]
        self.assertEqual(observed["attribution"], "real_adoption")
        self.assertGreater(observed["reward"], 0)
        updated = [
            e
            for e in provider.capability.of_type("METHOD_VALUE_UPDATED")
            if e.get("evidence_kind") == "real_adoption"
        ][0]
        self.assertEqual(updated["evidence_kind"], "real_adoption")
        self.assertEqual(updated["practice_count"], 4)
        self.assertGreater(updated["global_value"], 0.4)

    def test_only_one_activity_per_adoption_is_claimed(self) -> None:
        self._seed(_method(method_id="m1", title="先定冲突", skill_id="写作"))
        provider = self._provider()
        self._adopt(provider)
        self.assertTrue(provider.observe_activity(self._record(activity_id="a1")))
        self.assertFalse(
            provider.observe_activity(self._record(activity_id="a2")),
            "one practice per adoption: the second activity goes back to the shadow",
        )
        self.assertEqual(len(provider.capability.of_type("METHOD_APPLIED")), 1)

    def test_an_irrelevant_activity_is_not_claimed(self) -> None:
        self._seed(_method(method_id="m1", title="先定冲突", skill_id="写作"))
        provider = self._provider()
        provider.prepare_week()
        provider.record_adoption("<method>先定冲突</method>")
        relevant_free = self._record(gains={"跑步": 2}, activity_id="run-1")
        self.assertFalse(provider.observe_activity(relevant_free))

    def test_claimed_activity_is_reported_to_the_dispatcher(self) -> None:
        """The dispatcher must not let the shadow touch a claimed activity."""
        from src.agents.role_agent import RoleAgent

        agent = RoleAgent.__new__(RoleAgent)
        agent.logger = mock.Mock()
        agent.hint_config = HintConfig(enabled=True)
        agent.method_hints = mock.Mock()
        agent.method_hints.observe_activity.return_value = True
        agent.shadow = mock.Mock()

        agent._observe_activity_record({"activity_id": "a1"})

        agent.method_hints.observe_activity.assert_called_once()
        agent.shadow.observe_activity.assert_not_called()

        agent.method_hints.observe_activity.return_value = False
        agent._observe_activity_record({"activity_id": "a2"})
        agent.shadow.observe_activity.assert_called_once()

    # -- KI-10: used in plan now feeds strength ---------------------------
    def test_used_in_plan_strength_factor_is_no_longer_pinned_to_zero(self) -> None:
        base, factors = compute_strength(salience=0.6, weeks_since_created=0)
        self.assertEqual(factors.used_in_plan_count, 0)

        used, factors_used = compute_strength(
            salience=0.6, weeks_since_created=0, used_in_plan_count=2
        )
        self.assertGreater(used, base)
        self.assertEqual(factors_used.used_in_plan_count, 2)

    def test_settlement_reads_used_in_plan_from_the_view(self) -> None:
        # A real memory, then an adopted method that cites it.
        self.dm.append_ledger_record(
            self.dm.root / "cognition" / "memory_events.jsonl",
            {
                "type": "MEMORY_CREATED",
                "memory_id": "mem-1",
                "kind": "lesson",
                "content": "先列场景再写",
                "topics": ["creation"],
                "salience": 0.6,
                "strength": 0.6,
                "tier": "hot",
                "source_event_ids": ["ev-1"],
                "created_at": "Y2020-W01-settle",
                "week": "Y2020-W01",
            },
            idempotency_key="MEMORY_CREATED:mem-1:Y2020-W01",
        )
        self._seed(_method(method_id="m1", title="先定冲突", source_memories=["mem-1"]))
        provider = self._provider()
        provider.prepare_week()
        provider.record_adoption("<method>先定冲突</method>")

        view = build_memory_views(self.dm)["memories"]
        self.assertEqual(view["memories"]["mem-1"]["used_in_plan_count"], 1)

        updates = settle_strengths(
            list(view["memories"].values()),
            current_time="Y2020-W01-settle",
            used_in_plan_counts={"mem-1": 1},
        )
        self.assertTrue(updates, "using a memory in an adopted method must move its strength")
        self.assertEqual(updates[0]["factors"]["used_in_plan_count"], 1)
        self.assertGreater(updates[0]["strength"], 0.6)

    # -- no side effects ---------------------------------------------------
    def test_hints_never_write_to_the_simulation_ledger(self) -> None:
        self._seed(_method(method_id="m1", title="先定冲突", skill_id="写作"))
        watched = [
            self.dm.root / "activity.jsonl",
            self.dm.root / "state.jsonl",
            self.dm.root / "schedule.jsonl",
        ]
        before = {p: p.read_text(encoding="utf-8") for p in watched if p.exists()}
        skills_before = self.dm.read_state(exclude_cur_t=False)["skills"]

        provider = self._provider()
        provider.prepare_week()
        provider.record_adoption("<method>先定冲突</method>")
        provider.observe_activity(self._record())

        for path, content in before.items():
            self.assertEqual(path.read_text(encoding="utf-8"), content, f"{path.name} changed")
        self.assertEqual(self.dm.read_state(exclude_cur_t=False)["skills"], skills_before)

    def test_config_defaults_are_inert(self) -> None:
        self.assertFalse(HintConfig.from_world_config({}).enabled)
        cfg = HintConfig.from_world_config(
            {"cognition": {"method_hints": True, "method_hints_top_k": 2, "method_hints_max_chars": "x"}}
        )
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.top_k, 2)
        self.assertEqual(cfg.max_chars, 1400)


if __name__ == "__main__":
    unittest.main()
