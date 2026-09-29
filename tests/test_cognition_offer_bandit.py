"""阶段 5：方法菜单作为 contextual bandit，加上 KI-8 基线影子仪表（离线）。

阶段 5 改的是"给角色看什么"，所以边界仍然是可测的：

- **菜单还是菜单**：bandit 只决定展示哪些方法、按什么顺序，从不决定角色做什么；
  采用依旧是角色自己写下的 `<method>…</method>`（不变量 #11）。
- **拒绝也不动数值**：`METHOD_DECLINED` 只是提供者自己那一半反馈，价值仍然只随真实练习变化。
- **关闭即旧行为**：`method_hints_bandit=False` 时选法与阶段 4 逐字一致（不变量 #10）。
- **基线只观测不生效**：`shadow_reward` 跟着每条真实练习写进事件，但 value 仍走旧口径（KI-8 §6.1）。
"""

from __future__ import annotations

import json
import unittest
from typing import Any, Dict, List

from src.agents.cognition import materializer
from src.agents.cognition.event_store import CapabilityEventStore
from src.agents.cognition.method_hints import HintConfig, MethodHintProvider
from src.agents.cognition.models import ContextSignature, Methodology
from src.agents.cognition.offer_bandit import (
    MethodStat,
    consecutive_weeks_used,
    context_estimate,
    exploration_coefficient,
    ignored_streak,
    is_unproven,
    method_stats,
    rank_offers,
    score_offer,
    tags_from_records,
    week_signature,
)
from src.agents.cognition.reward_model import (
    BaselineTracker,
    baseline_tracker_from_events,
    shadow_baseline_for,
    signals_from_activity_record,
)
from src.world.clock import Stage
from tests._helpers import make_datamanager, temp_workspace


def _state(vitality: int = 80, deposit: int = 1000) -> Dict[str, Any]:
    return {
        "vitality": vitality,
        "fulfillment": {"mood": 50, "material": 50, "social": 50, "esteem": 50},
        "assets": {"deposit": deposit, "possessions": []},
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
    contexts: Dict[str, Dict[str, float]] | None = None,
    source_motif: str = "",
    source_type: str = "practice_reflection",
) -> Methodology:
    method = Methodology(
        method_id=method_id,
        skill_id=skill_id,
        title=title,
        description="复杂写作任务中先确定目标与冲突",
        status=status,
        source_type=source_type,
        global_value=value,
        confidence=confidence,
        practice_count=practice,
        success_count=success,
        steps=["明确目标", "列出冲突", "排场景顺序"],
        checks=["每个场景是否推动冲突"],
        failure_modes=["规划过度导致迟迟不开始"],
        applicable_contexts=["long_form", "complex_structure"],
        context_values=dict(contexts or {}),
    )
    method.source_motif = source_motif or None
    return method


def _record(
    *,
    activity_id: str = "act-1",
    activity_type: str = "solo",
    skill: str = "写作",
    gain: float = 2.0,
    turns: int = 3,
) -> Dict[str, Any]:
    return {
        "activity_id": activity_id,
        "type": activity_type,
        "turns": turns,
        "content": "写一段初稿",
        "outcome": {
            "delta_skills": {skill: gain},
            "delta_vitality": -2,
            "delta_money": 0,
            "delta_fulfillment": {},
        },
    }


# ---------------------------------------------------------------------------
# The bandit itself
# ---------------------------------------------------------------------------


class ContextEstimateTests(unittest.TestCase):
    def test_thin_context_evidence_is_shrunk_towards_the_pooled_value(self) -> None:
        method = _method(value=0.2, contexts={"solo+learning": {"value": 0.9, "count": 1}})
        signature = ContextSignature(skill_ids=["写作"], context_tags=["solo", "learning"])
        estimate = context_estimate(method, signature)
        self.assertIsNotNone(estimate)
        # weight 1 vs shrinkage 3 -> a quarter of the way from 0.2 to 0.9
        self.assertGreater(estimate.value, 0.2)
        self.assertLess(estimate.value, 0.9)
        self.assertEqual(estimate.samples, 1)

    def test_more_evidence_moves_the_estimate_further(self) -> None:
        signature = ContextSignature(context_tags=["solo", "learning"])
        thin = context_estimate(
            _method(value=0.2, contexts={"solo+learning": {"value": 0.9, "count": 1}}),
            signature,
        )
        thick = context_estimate(
            _method(value=0.2, contexts={"solo+learning": {"value": 0.9, "count": 9}}),
            signature,
        )
        self.assertGreater(thick.value, thin.value)

    def test_unrelated_contexts_are_ignored(self) -> None:
        method = _method(value=0.3, contexts={"work+time_pressure": {"value": -0.8, "count": 5}})
        signature = ContextSignature(context_tags=["solo", "learning"])
        self.assertIsNone(context_estimate(method, signature))

    def test_partial_overlap_counts_less_than_an_exact_match(self) -> None:
        signature = ContextSignature(context_tags=["solo", "learning"])
        partial = context_estimate(
            _method(value=0.0, contexts={"solo": {"value": 1.0, "count": 4}}), signature
        )
        exact = context_estimate(
            _method(value=0.0, contexts={"solo+learning": {"value": 1.0, "count": 4}}),
            signature,
        )
        self.assertGreater(exact.weight, partial.weight)
        self.assertTrue(exact.exact)
        self.assertFalse(partial.exact)


class ExplorationTests(unittest.TestCase):
    def test_exploration_follows_creativity_and_curiosity(self) -> None:
        low = exploration_coefficient({"creativity": 10, "curiosity": 20})
        mid = exploration_coefficient({"creativity": 50, "curiosity": 50})
        high = exploration_coefficient({"creativity": 95, "curiosity": 90})
        self.assertLess(low, mid)
        self.assertLess(mid, high)

    def test_exploration_stays_bounded(self) -> None:
        self.assertLessEqual(exploration_coefficient({"creativity": 100, "curiosity": 100}), 0.5)
        self.assertGreater(exploration_coefficient({"creativity": 0, "curiosity": 0}), 0.0)

    def test_missing_traits_fall_back_to_the_base_coefficient(self) -> None:
        self.assertAlmostEqual(exploration_coefficient({}), 0.22, places=4)

    def test_ucb_bonus_decays_as_a_method_keeps_being_offered(self) -> None:
        method = _method(status="proposed", value=0.0, confidence=0.0, practice=0, success=0)
        signature = ContextSignature(context_tags=["solo"])
        first = score_offer(method, signature, stat=MethodStat("m", offers=0))
        later = score_offer(method, signature, stat=MethodStat("m", offers=6))
        self.assertGreater(first.parts["explore"], later.parts["explore"])


class RankingTests(unittest.TestCase):
    def test_offered_and_ignored_pushes_a_method_down_the_menu(self) -> None:
        signature = ContextSignature(context_tags=["solo", "learning"])
        fresh = _method(method_id="m-fresh", value=0.4)
        stale = _method(method_id="m-stale", value=0.4)
        ignored = MethodStat("m-stale", offers=5, declines=5, ignored_streak=5)
        ranked = rank_offers(
            [stale, fresh], signature, stats={"m-stale": ignored}, exploration_c=0.0
        )
        self.assertEqual(ranked[0].method.method_id, "m-fresh")
        self.assertLess(ranked[1].parts["decline"], 0.0)
        self.assertLess(ranked[1].parts["stale"], 0.0)

    def test_a_situation_value_can_outrank_a_higher_pooled_value(self) -> None:
        signature = ContextSignature(context_tags=["solo", "time_pressure"])
        generalist = _method(method_id="m-general", value=0.5)
        specialist = _method(
            method_id="m-special",
            value=0.1,
            contexts={"solo+time_pressure": {"value": 0.95, "count": 8}},
        )
        ranked = rank_offers([generalist, specialist], signature, exploration_c=0.0)
        self.assertEqual(ranked[0].method.method_id, "m-special")

    def test_ranking_is_deterministic(self) -> None:
        signature = ContextSignature(context_tags=["solo"])
        methods = [_method(method_id=f"m-{i}", value=0.3) for i in range(6)]
        stats = {"m-2": MethodStat("m-2", offers=2, declines=1, ignored_streak=2)}
        first = [s.method.method_id for s in rank_offers(methods, signature, stats=stats)]
        second = [s.method.method_id for s in rank_offers(methods, signature, stats=stats)]
        self.assertEqual(first, second)

    def test_own_lesson_keeps_its_advantage(self) -> None:
        signature = ContextSignature(context_tags=["solo"])
        lesson = _method(
            method_id="m-lesson",
            status="proposed",
            value=0.0,
            confidence=0.0,
            practice=0,
            success=0,
            source_motif="lesson_application",
        )
        routine = _method(method_id="m-routine", value=0.25)
        ranked = rank_offers([routine, lesson], signature, exploration_c=0.0)
        self.assertEqual(ranked[0].method.method_id, "m-lesson")

    def test_contraindicated_method_loses_to_a_clean_one(self) -> None:
        signature = ContextSignature(context_tags=["solo", "time_pressure"])
        risky = _method(method_id="m-risky", value=0.5)
        risky.contraindications = ["time_pressure"]
        clean = _method(method_id="m-clean", value=0.4)
        ranked = rank_offers([risky, clean], signature, exploration_c=0.0)
        self.assertEqual(ranked[0].method.method_id, "m-clean")

    def test_explore_role_is_reserved_for_unproven_methods(self) -> None:
        signature = ContextSignature(context_tags=["solo"])
        # Two candidates hold their places on merit; the third only reaches the
        # menu because of the exploration bonus, which is what `explore` means.
        kept = _method(method_id="m-kept", value=0.05, confidence=0.05, status="tested")
        crowded_out = _method(
            method_id="m-p2",
            status="tested",
            value=0.25,
            confidence=0.0,
            practice=0,
            success=0,
        )
        rescued = _method(
            method_id="m-u", status="proposed", value=0.0, confidence=0.0, practice=0, success=0
        )
        stats = {"m-p2": MethodStat("m-p2", offers=3, declines=0, ignored_streak=3)}
        ranked = rank_offers(
            [kept, crowded_out, rescued],
            signature,
            stats=stats,
            top_k=2,
            exploration_c=0.6,
        )
        roles = {s.method.method_id: s.role for s in ranked}
        self.assertEqual(sorted(roles), ["m-kept", "m-u"])
        self.assertEqual(roles["m-kept"], "exploit")
        self.assertEqual(roles["m-u"], "explore")
        self.assertTrue(is_unproven(rescued))

    def test_unproven_alone_does_not_earn_the_explore_label(self) -> None:
        """`explore` 的意思是"靠探索奖励进菜单"，不是"还没练过"。

        年轻的运行里几乎所有方法都是 `proposed`；只按"没练过"打标签，
        运行 09271745 会把 45 次提供里的 21 次记成探索，读数字就失去意义。
        """
        signature = ContextSignature(context_tags=["solo"])
        strong = [
            _method(
                method_id=f"m-s{i}",
                status="proposed",
                value=0.6 - i * 0.1,
                confidence=0.0,
                practice=0,
                success=0,
            )
            for i in range(2)
        ]
        rescued = _method(
            method_id="m-r",
            status="proposed",
            value=0.0,
            confidence=0.0,
            practice=0,
            success=0,
        )
        ranked = rank_offers([*strong, rescued], signature, top_k=2)
        labels = {s.method.method_id: s.role for s in ranked}
        self.assertEqual(len(ranked), 2)
        # Both slots are held on merit; nothing needed the exploration bonus.
        self.assertEqual(sorted(labels), ["m-s0", "m-s1"])
        self.assertEqual(set(labels.values()), {"exploit", "filler"})

    def test_an_unproven_method_inside_the_menu_on_merit_is_not_labelled_explore(self) -> None:
        signature = ContextSignature(context_tags=["solo"])
        proven = _method(method_id="m-p", value=0.5, status="validated")
        untried = _method(
            method_id="m-u", status="proposed", value=0.45, confidence=0.0, practice=0, success=0
        )
        ranked = rank_offers([proven, untried], signature, top_k=2)
        labels = {s.method.method_id: s.role for s in ranked}
        self.assertEqual(labels["m-p"], "exploit")
        self.assertEqual(labels["m-u"], "filler")


class MethodStatsTests(unittest.TestCase):
    def test_stats_read_offers_adoptions_declines_and_the_streak(self) -> None:
        events = [
            {"type": "METHOD_HINTED", "method_id": "m-1", "week": "Y2020-W01"},
            {"type": "METHOD_SELECTED", "method_id": "m-1", "week": "Y2020-W01", "adoption": True},
            {"type": "METHOD_OUTCOME_OBSERVED", "method_id": "m-1", "week": "Y2020-W02",
             "attribution": "real_adoption"},
            {"type": "METHOD_HINTED", "method_id": "m-1", "week": "Y2020-W03"},
            {"type": "METHOD_DECLINED", "method_id": "m-1", "week": "Y2020-W03"},
            {"type": "METHOD_HINTED", "method_id": "m-1", "week": "Y2020-W04"},
            {"type": "METHOD_DECLINED", "method_id": "m-1", "week": "Y2020-W04"},
            {"type": "METHOD_HINTED", "method_id": "m-2", "week": "Y2020-W02"},
            {"type": "METHOD_DECLINED", "method_id": "m-2", "week": "Y2020-W02"},
        ]
        stats = method_stats(events)
        self.assertEqual(stats["m-1"].offers, 3)
        self.assertEqual(stats["m-1"].adoptions, 1)
        self.assertEqual(stats["m-1"].declines, 2)
        self.assertEqual(stats["m-1"].practices, 1)
        # offered in W03 and W04 after the W01 adoption
        self.assertEqual(ignored_streak(stats["m-1"]), 2)
        # never adopted: every offer counts
        self.assertEqual(ignored_streak(stats["m-2"]), 1)

    def test_a_counterfactual_outcome_is_not_a_practice(self) -> None:
        stats = method_stats(
            [
                {"type": "METHOD_OUTCOME_OBSERVED", "method_id": "m-1",
                 "attribution": "shadow_counterfactual"},
            ]
        )
        self.assertEqual(stats["m-1"].practices, 0)

    def test_a_method_used_week_after_week_stops_leading_the_menu(self) -> None:
        """阶段 5 欠账（探索槽 0 采用）：连续用同一个方法时，价值不动，首格让出来。

        实测运行 09271745：三个角色都在被提供的五周里采用了**同一个**方法，
        每练一次 value/confidence 就涨一次，于是它的 exploit 分数逐周升高
        （0.31 → 0.84 → 1.24 → 1.29），菜单再也不轮换；而探索槽提供的东西
        因为从没被采用过，也永远拿不到证据。
        """
        signature = ContextSignature(context_tags=["solo"])
        veteran = _method(method_id="m-veteran", value=0.55, confidence=0.6, practice=5, success=4)
        rival = _method(method_id="m-rival", value=0.45, confidence=0.5, practice=4, success=3)
        stats = {
            "m-veteran": MethodStat(
                "m-veteran",
                offers=5,
                adoptions=5,
                practices=5,
                adopted_weeks=["Y2020-W02", "Y2020-W03", "Y2020-W04", "Y2020-W05", "Y2020-W06"],
                last_adopted_week="Y2020-W06",
                ignored_streak=0,
            )
        }
        first_week = rank_offers([veteran, rival], signature, exploration_c=0.0)
        self.assertEqual(first_week[0].method.method_id, "m-veteran")

        ranked = rank_offers([veteran, rival], signature, stats=stats, exploration_c=0.0)
        self.assertEqual(ranked[0].method.method_id, "m-rival")
        self.assertLess(ranked[1].parts["repeat"], 0.0)
        # 只有"提供什么"受影响：方法的价值、置信度、练习次数都没被碰过
        self.assertEqual(veteran.global_value, 0.55)
        self.assertEqual(veteran.confidence, 0.6)

    def test_the_first_week_of_use_is_not_penalised(self) -> None:
        signature = ContextSignature(context_tags=["solo"])
        method = _method(method_id="m-new", value=0.5)
        stat = MethodStat("m-new", offers=1, adoptions=1, adopted_weeks=["Y2020-W02"])
        scored = score_offer(method, signature, stat=stat, exploration_c=0.0)
        self.assertNotIn("repeat", scored.parts)

    def test_the_usage_streak_survives_a_week_gap(self) -> None:
        events = [
            {"type": "METHOD_SELECTED", "method_id": "m", "week": week, "adoption": True}
            for week in ("Y2020-W02", "Y2020-W03", "Y2020-W06")
        ]
        stat = method_stats(events)["m"]
        self.assertEqual(consecutive_weeks_used(stat), 1)

    def test_the_usage_streak_crosses_the_year_boundary(self) -> None:
        events = [
            {"type": "METHOD_SELECTED", "method_id": "m", "week": week, "adoption": True}
            for week in ("Y2020-W09", "Y2020-W10", "Y2021-W01")
        ]
        stat = method_stats(events)["m"]
        self.assertEqual(consecutive_weeks_used(stat), 3)


class WeekSignatureTests(unittest.TestCase):
    def test_signature_uses_only_the_controlled_vocabulary(self) -> None:
        signature = week_signature(
            skills=["写作"],
            state=_state(vitality=10, deposit=50),
            goal_memories=[{"content": "把小说写完"}],
            recent_records=[_record()],
        )
        from src.agents.cognition.models import CONTEXT_TAGS

        self.assertTrue(set(signature.context_tags) <= set(CONTEXT_TAGS))
        self.assertIn("time_pressure", signature.context_tags)
        self.assertIn("low_money", signature.context_tags)
        self.assertIn("solo", signature.context_tags)
        self.assertEqual(signature.goal, "把小说写完")

    def test_a_rested_rich_week_looks_different(self) -> None:
        calm = week_signature(skills=["写作"], state=_state(vitality=85, deposit=5000))
        tight = week_signature(skills=["写作"], state=_state(vitality=15, deposit=10))
        self.assertIn("sufficient_time", calm.context_tags)
        self.assertNotEqual(calm.context_key(), tight.context_key())

    def test_tags_from_records_follow_last_weeks_shape(self) -> None:
        records = [
            {"activity_id": "a", "type": "solo", "outcome": {}},
            {"activity_id": "b", "type": "solo", "outcome": {}},
            {"activity_id": "c", "type": "joint", "outcome": {}},
        ]
        tags = tags_from_records(records)
        self.assertEqual(tags[0], "solo")
        self.assertIn("joint", tags)

    def test_no_records_means_no_guessed_tags(self) -> None:
        self.assertEqual(tags_from_records([]), [])


# ---------------------------------------------------------------------------
# The menu, as the character experiences it
# ---------------------------------------------------------------------------


class OfferProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._ctx = temp_workspace()
        self.root = self._ctx.__enter__()  # type: ignore[attr-defined]
        self.dm, self.clock = make_datamanager()
        self.dm.save_state(_state())
        self.clock.set_stage(Stage.PLAN)

    def tearDown(self) -> None:
        self._ctx.__exit__(None, None, None)  # type: ignore[attr-defined]

    def _seed(self, *methods: Methodology) -> None:
        store = CapabilityEventStore(self.dm)
        for method in methods:
            payload = method.to_dict()
            method_id = payload.pop("method_id")
            store.append(
                "METHOD_PROPOSED",
                method_id=method_id,
                payload=payload,
                idempotency_key=f"METHOD_PROPOSED:{method_id}:-:seed",
            )
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
            bandit=overrides.pop("bandit", True),
            record_declines=overrides.pop("record_declines", True),
        )
        return MethodHintProvider(
            dm=self.dm,
            clock=self.clock,
            agent_name=self.dm.char,
            traits={"creativity": 60, "curiosity": 60},
            config=config,
            language="zh",
        )

    def _events(self, *types: str) -> List[Dict[str, Any]]:
        return [
            e
            for e in CapabilityEventStore(self.dm).events()
            if not types or e.get("type") in types
        ]

    # -- observation: the half phase 4 could not see -----------------------
    def test_ignoring_the_whole_menu_is_recorded_as_a_decline(self) -> None:
        self._seed(*[_method(method_id=f"m-{i}", value=0.4) for i in range(3)])
        provider = self._provider()
        provider.prepare_week()
        self.assertIsNone(provider.record_adoption("这周我打算专心写初稿，没有别的安排。"))
        declines = self._events("METHOD_DECLINED")
        self.assertEqual(len(declines), 3)
        self.assertTrue(all(e["reason"] == "no_declaration" for e in declines))

    def test_taking_one_method_up_declines_the_others(self) -> None:
        self._seed(*[_method(method_id=f"m-{i}", title=f"方法{i}", value=0.4) for i in range(3)])
        provider = self._provider()
        offered = provider.select_offers()
        provider.prepare_week()
        chosen = offered[0]
        adoption = provider.record_adoption(f"我打算用 <method>{chosen.title}</method> 试试。")
        self.assertIsNotNone(adoption)
        declines = self._events("METHOD_DECLINED")
        self.assertEqual(len(declines), 2)
        self.assertTrue(all(e["reason"] == "chose_other" for e in declines))
        self.assertNotIn(chosen.method_id, {e["method_id"] for e in declines})

    def test_an_empty_plan_is_declined_for_a_different_reason(self) -> None:
        self._seed(_method(method_id="m-0", value=0.4))
        provider = self._provider()
        provider.prepare_week()
        provider.record_adoption("")
        declines = self._events("METHOD_DECLINED")
        self.assertEqual([e["reason"] for e in declines], ["no_plan"])

    def test_declines_are_idempotent_within_a_week(self) -> None:
        self._seed(_method(method_id="m-0", value=0.4))
        provider = self._provider()
        provider.prepare_week()
        provider.record_adoption("没有采用任何方法。")
        provider.record_adoption("没有采用任何方法。")
        self.assertEqual(len(self._events("METHOD_DECLINED")), 1)

    def test_a_decline_moves_no_value(self) -> None:
        """不变量 #11：被提供、被忽略都不改变方法价值。"""
        self._seed(_method(method_id="m-0", value=0.4, practice=3, success=2))
        before = len(self._events("METHOD_VALUE_UPDATED"))
        provider = self._provider()
        provider.prepare_week()
        provider.record_adoption("这周不用任何方法。")
        self.assertEqual(len(self._events("METHOD_VALUE_UPDATED")), before)
        view = materializer.build_capability_view(self.dm)
        self.assertEqual(view["methodologies"]["m-0"]["global_value"], 0.4)
        self.assertEqual(view["methodologies"]["m-0"]["practice_count"], 3)
        self.assertEqual(view["methodologies"]["m-0"]["declines"], 1)

    def test_a_disabled_provider_records_nothing(self) -> None:
        self._seed(_method(method_id="m-0", value=0.4))
        provider = self._provider(enabled=False)
        self.assertEqual(provider.prepare_week(), "")
        self.assertIsNone(provider.record_adoption("随便写点什么"))
        self.assertEqual(self._events("METHOD_DECLINED"), [])

    def test_the_offer_carries_the_situation_it_was_made_in(self) -> None:
        self._seed(_method(method_id="m-0", value=0.4))
        provider = self._provider()
        provider.prepare_week()
        offer = self._events("METHOD_HINTED")[0]
        self.assertIn("context_key", offer)
        self.assertTrue(offer["context_tags"])
        self.assertIn("score_parts", offer)
        self.assertTrue(offer["bandit"])

    def test_the_situation_follows_the_character_state(self) -> None:
        self._seed(_method(method_id="m-0", value=0.4))
        rested = self._provider()._week_signature().context_key()

        self.dm.save_state(_state(vitality=12, deposit=20))
        broke = self._provider()._week_signature().context_key()
        self.assertNotEqual(rested, broke)
        self.assertIn("time_pressure", broke)
        self.assertIn("low_money", broke)

    def test_the_menu_says_when_a_method_has_been_used_for_weeks(self) -> None:
        """角色看不到自己的菜单历史，所以这件事要说给它听——但仍然是菜单。"""
        method = _method(method_id="m-veteran", title="双光情境对照检查法", value=0.55)
        self._seed(method)
        store = CapabilityEventStore(self.dm)
        for index, week in enumerate(("Y2020-W02", "Y2020-W03", "Y2020-W04"), start=2):
            store.append(
                "METHOD_HINTED", method_id="m-veteran",
                payload={"week": week, "role": "exploit"},
                idempotency_key=f"METHOD_HINTED:m-veteran:{week}",
            )
            store.append(
                "METHOD_SELECTED", method_id="m-veteran",
                payload={"week": week, "adoption": True, "source": "hint"},
                idempotency_key=f"METHOD_SELECTED:m-veteran:{week}:hint",
            )
        provider = self._provider()
        block = provider.prepare_week()
        self.assertIn("你已经连续", block)
        self.assertIn("可选，不是命令", block)

    # -- the switch keeps the old behaviour --------------------------------
    def test_bandit_off_reproduces_the_phase_four_menu(self) -> None:
        """不变量 #10：关闭新机制必须回到旧行为（同输入同菜单）。"""
        practice = [_method(method_id=f"m-p{i}", value=0.6) for i in range(3)]
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
        self._seed(*practice, own)

        provider = self._provider(bandit=False)
        offers = provider.select_offers()
        self.assertEqual([o.role for o in offers][0], "exploit")
        self.assertIn("explore", [o.role for o in offers])
        self.assertEqual(
            [o.method_id for o in offers if o.role == "explore"], ["method-idea-own"]
        )
        # The phase-4 path does not fill in the bandit bookkeeping.
        self.assertEqual(offers[0].parts, {})

    def test_bandit_on_still_offers_the_characters_own_untried_idea(self) -> None:
        practice = [_method(method_id=f"m-p{i}", value=0.6) for i in range(3)]
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
        self._seed(*practice, own)
        provider = self._provider(bandit=True)
        offers = provider.select_offers()
        self.assertIn("method-idea-own", [o.method_id for o in offers])
        self.assertEqual(
            [o.role for o in offers if o.method_id == "method-idea-own"], ["explore"]
        )

    def test_a_method_the_character_keeps_ignoring_leaves_the_menu(self) -> None:
        """阶段 4 复核 §3 #3：exploit 槽不该被同一个方法连占四周。"""
        self._seed(
            _method(method_id="m-stale", value=0.55),
            _method(method_id="m-fresh", value=0.5),
        )
        store = CapabilityEventStore(self.dm)
        for week in ("Y2020-W01", "Y2020-W02", "Y2020-W03"):
            store.append(
                "METHOD_HINTED",
                method_id="m-stale",
                payload={"week": week, "role": "exploit"},
                idempotency_key=f"METHOD_HINTED:m-stale:{week}",
            )
            store.append(
                "METHOD_DECLINED",
                method_id="m-stale",
                payload={"week": week, "reason": "no_declaration"},
                idempotency_key=f"METHOD_DECLINED:m-stale:{week}",
            )
        provider = self._provider()
        offers = provider.select_offers()
        self.assertEqual(offers[0].method_id, "m-fresh")
        stale = [o for o in offers if o.method_id == "m-stale"]
        self.assertTrue(stale and stale[0].ignored_streak == 3)


# ---------------------------------------------------------------------------
# KI-8 shadow baseline: observed, never applied
# ---------------------------------------------------------------------------


class ShadowBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self._ctx = temp_workspace()
        self.root = self._ctx.__enter__()  # type: ignore[attr-defined]
        self.dm, self.clock = make_datamanager()
        self.dm.save_state(_state())
        self.clock.set_stage(Stage.PLAN)

    def tearDown(self) -> None:
        self._ctx.__exit__(None, None, None)  # type: ignore[attr-defined]

    def test_a_tracker_is_rebuilt_from_the_event_stream(self) -> None:
        events = [
            {"type": "METHOD_OUTCOME_OBSERVED", "context_key": "solo",
             "components": {"skill_gain": 0.2}}
            for _ in range(5)
        ] + [
            {"type": "METHOD_OUTCOME_OBSERVED", "context_key": "work",
             "components": {"skill_gain": 0.0}},
        ]
        tracker = baseline_tracker_from_events(events)
        self.assertEqual(tracker.samples("solo"), 5)
        # this character usually gets 0.2 here, so the baseline sits below the
        # neutral prior and a 0.2 outcome reads as "as expected"
        self.assertLess(tracker.baseline_for("solo"), 0.5)
        self.assertGreater(tracker.baseline_for("solo"), 0.2)

    def test_the_fallback_cascade_reaches_the_characters_own_history(self) -> None:
        """上下文键很细时，基线必须退到"同类活动/本人全部"，而不是停在中性先验。

        Measured in the first phase-5 run: offer-time keys are predicted
        situations, practice-time keys are observed ones, so a six-week run
        fills almost no exact bucket. With only an exact→"default" fallback (and
        nothing writing "default") every outcome was scored against the neutral
        prior 0.5, which reports a character-wide shift instead of "better or
        worse than usual".
        """
        tracker = BaselineTracker()
        for index in range(6):
            tracker.observe(f"solo+tag{index}", 0.2, activity_type="solo")
        # no exact bucket, but six outcomes of the same activity type
        self.assertEqual(tracker.samples("solo+never_seen", activity_type="solo"), 6)
        self.assertLess(tracker.baseline_for("solo+never_seen", activity_type="solo"), 0.4)
        # the exact bucket wins as soon as it has enough evidence of its own
        for _ in range(3):
            tracker.observe("solo+time_pressure", 0.9, activity_type="solo")
        self.assertEqual(tracker.samples("solo+time_pressure", activity_type="solo"), 3)
        self.assertGreater(
            tracker.baseline_for("solo+time_pressure", activity_type="solo"), 0.5
        )

    def test_a_character_with_no_history_is_scored_against_the_neutral_prior(self) -> None:
        tracker = BaselineTracker()
        self.assertEqual(tracker.baseline_for("solo+learning", activity_type="solo"), 0.5)
        self.assertEqual(tracker.samples("solo+learning", activity_type="solo"), 0)

    def test_the_shadow_reward_goes_negative_below_the_baseline(self) -> None:
        tracker = BaselineTracker()
        for _ in range(6):
            tracker.observe("solo", 0.8)  # this character usually does well here
        weak = signals_from_activity_record(_record(gain=0.0, turns=6))
        shadow = shadow_baseline_for(
            weak, skill_id="写作", context_key="solo", tracker=tracker
        )
        self.assertLess(shadow.quality_delta, 0.0)
        self.assertLess(shadow.reward, 0.0)

    def test_a_good_outcome_above_the_baseline_stays_positive(self) -> None:
        tracker = BaselineTracker()
        for _ in range(6):
            tracker.observe("solo", 0.05)
        strong = signals_from_activity_record(_record(gain=3.0, turns=3))
        shadow = shadow_baseline_for(
            strong, skill_id="写作", context_key="solo", tracker=tracker
        )
        self.assertGreater(shadow.quality_delta, 0.0)
        self.assertGreater(shadow.reward, 0.0)

    def test_practice_events_carry_the_shadow_score_but_keep_the_live_value(self) -> None:
        store = CapabilityEventStore(self.dm)
        method = _method(method_id="m-0", value=0.0, confidence=0.0, practice=0, success=0)
        payload = method.to_dict()
        method_id = payload.pop("method_id")
        store.append(
            "METHOD_PROPOSED", method_id=method_id, payload=payload,
            idempotency_key="METHOD_PROPOSED:m-0:-:seed",
        )
        provider = MethodHintProvider(
            dm=self.dm,
            clock=self.clock,
            agent_name=self.dm.char,
            traits={},
            config=HintConfig(enabled=True, top_k=1, bandit=True),
            language="zh",
        )
        provider.prepare_week()
        offer = provider.select_offers()[0]
        provider.record_adoption(f"我试试 <method>{offer.title}</method>。")
        self.assertTrue(provider.observe_activity(_record()))

        outcomes = [e for e in store.events() if e.get("type") == "METHOD_OUTCOME_OBSERVED"]
        self.assertEqual(len(outcomes), 1)
        outcome = outcomes[0]
        self.assertIn("shadow_reward", outcome)
        self.assertIn("shadow_baseline", outcome)
        self.assertFalse(outcome["components"]["used_baseline"])

        updates = [e for e in store.events() if e.get("type") == "METHOD_VALUE_UPDATED"]
        self.assertEqual(len(updates), 1)
        # The value moved on the live reward, not the shadow one.
        self.assertAlmostEqual(updates[0]["reward"], outcome["reward"], places=6)
        self.assertNotAlmostEqual(updates[0]["reward"], outcome["shadow_reward"], places=6)


class MaterializerTests(unittest.TestCase):
    def test_declines_are_folded_into_the_view(self) -> None:
        view = materializer.materialize(
            [
                {"type": "METHOD_PROPOSED", "method_id": "m-1", "title": "方法",
                 "ledger_event_id": "ev-1"},
                {"type": "METHOD_DECLINED", "method_id": "m-1", "week": "Y2020-W02",
                 "reason": "no_declaration", "role": "exploit", "ledger_event_id": "ev-2"},
            ],
            skills={"写作": 10},
        )
        entry = view["methodologies"]["m-1"]
        self.assertEqual(entry["declines"], 1)
        self.assertEqual(entry["last_declined"]["reason"], "no_declaration")
        # A decline is not evidence: nothing about the method's numbers moved.
        self.assertEqual(entry["global_value"], 0.0)
        self.assertEqual(entry["practice_count"], 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
