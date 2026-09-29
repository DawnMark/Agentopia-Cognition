"""阶段 5：目标完成度（goal_progress）与生命周期接线的离线测试。

这一组锁定四件事：

- **受控字段**：世界模型只能给 0–1 的一个数，越界会被夹紧，给不出数就丢弃，
  而不是让整次活动评估失败；
- **默认惰性**：`world.cognition.goal_progress` 关闭时，活动评估 prompt 与
  阶段 -1~4 逐字一致（不变量 #10），reward 的 goal 项恒为 0；
- **相对基线**：开启后 goal_progress 也走"比自己常规更好还是更差"的中心化口径；
- **结算接线**：方法生命周期在周结算里归档失败且长期未用的方法，视图保留记录。
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, Dict, List

from src.agents.cognition import materializer
from src.agents.cognition.event_store import CapabilityEventStore
from src.agents.cognition.method_hints import HintConfig
from src.agents.cognition.method_lifecycle import LifecycleConfig, MethodLifecycle
from src.agents.cognition.models import ContextSignature, Methodology
from src.agents.cognition.reward_model import (
    BaselineTracker,
    alternative_caliber,
    live_reward_for,
    weights_for_caliber,
    OutcomeSignals,
    RewardWeights,
    compute_reward,
    goal_baseline_tracker_from_events,
    shadow_baseline_for,
    signals_from_activity_record,
)
from src.world.clock import Stage
from tests._helpers import make_datamanager, temp_workspace


def _record(**overrides: Any) -> Dict[str, Any]:
    outcome = {
        "delta_skills": {"写作": 2.0},
        "delta_vitality": -2,
        "delta_money": 0,
        "delta_fulfillment": {"mood": 2},
    }
    outcome.update(overrides.pop("outcome", {}))
    row = {
        "activity_id": "act-1",
        "type": "solo",
        "turns": 3,
        "content": "写一段初稿",
        "outcome": outcome,
    }
    row.update(overrides)
    return row


class GodFieldTests(unittest.TestCase):
    """世界模型侧：字段是受控的，且默认不出现。"""

    def test_a_reported_value_is_bounded(self) -> None:
        from src.world.god import _coerce_goal_progress

        data = {"goal_progress": 1.7}
        _coerce_goal_progress(data)
        self.assertEqual(data["goal_progress"], 1.0)
        data = {"goal_progress": -3}
        _coerce_goal_progress(data)
        self.assertEqual(data["goal_progress"], 0.0)

    def test_an_unusable_value_is_dropped_not_fatal(self) -> None:
        from src.world.god import _coerce_goal_progress

        data = {"goal_progress": "quite well"}
        _coerce_goal_progress(data)
        self.assertNotIn("goal_progress", data)

    def test_an_absent_value_stays_absent(self) -> None:
        from src.world.god import _coerce_goal_progress, _goal_progress_of

        data: Dict[str, Any] = {}
        _coerce_goal_progress(data)
        self.assertEqual(data, {})
        self.assertIsNone(_goal_progress_of({}))

    def test_the_prompt_asks_for_it_only_when_switched_on(self) -> None:
        """不变量 #10：关闭时必须与阶段 -1~4 的 prompt 逐字一致。"""
        from src.agents import prompts
        from src.config import get_config

        config = get_config()
        cognition = config["world"].setdefault("cognition", {})
        original = cognition.get("goal_progress", False)
        try:
            cognition["goal_progress"] = False
            off = prompts.build_god_eval_solo_activity_prompt()
            self.assertNotIn("Goal Progress", off)
            self.assertNotIn("__GOAL_PROGRESS_RULES__", off)

            cognition["goal_progress"] = True
            on = prompts.build_god_eval_solo_activity_prompt()
            self.assertIn("Goal Progress", on)
            # Turning it on only *adds* the block; nothing else moves.
            self.assertNotIn("__GOAL_PROGRESS_RULES__", on)
            self.assertIn("delta_skills", on)
        finally:
            cognition["goal_progress"] = original

    def test_the_joint_and_public_prompts_carry_the_same_rules(self) -> None:
        from src.agents import prompts
        from src.config import get_config

        config = get_config()
        cognition = config["world"].setdefault("cognition", {})
        original = cognition.get("goal_progress", False)
        try:
            cognition["goal_progress"] = True
            for builder in (
                prompts.build_god_eval_joint_activity_prompt,
                prompts.build_god_eval_public_activity_prompt,
            ):
                self.assertIn("Goal Progress", builder())
        finally:
            cognition["goal_progress"] = original


class GoalProgressRewardTests(unittest.TestCase):
    def _signals(self, goal: float | None) -> OutcomeSignals:
        row = _record()
        if goal is None:
            row["outcome"].pop("goal_progress", None)
        else:
            row["outcome"]["goal_progress"] = goal
        return signals_from_activity_record(row)

    def test_an_absent_report_contributes_nothing(self) -> None:
        weights = RewardWeights(use_baseline=True, goal_progress_weight=0.30)
        components = compute_reward(
            self._signals(None), skill_id="写作", weights=weights, baseline=0.2
        )
        self.assertIsNone(components.goal_progress)
        self.assertEqual(components.goal_progress_delta, 0.0)
        # identical to the same score without a goal term at all
        plain = compute_reward(
            self._signals(None),
            skill_id="写作",
            weights=RewardWeights(use_baseline=True),
            baseline=0.2,
        )
        self.assertAlmostEqual(components.total, plain.total, places=6)

    def test_goal_progress_is_scored_against_the_characters_own_usual(self) -> None:
        weights = RewardWeights(use_baseline=True, goal_progress_weight=0.30)
        tracker = BaselineTracker()
        for _ in range(6):
            tracker.observe("solo", 0.2, activity_type="solo")
        good = compute_reward(
            self._signals(0.9),
            skill_id="写作",
            weights=weights,
            baseline=0.2,
            goal_baseline=tracker.baseline_for("solo", activity_type="solo"),
        )
        poor = compute_reward(
            self._signals(0.0),
            skill_id="写作",
            weights=weights,
            baseline=0.2,
            goal_baseline=tracker.baseline_for("solo", activity_type="solo"),
        )
        self.assertGreater(good.goal_progress_delta, 0.0)
        self.assertLess(poor.goal_progress_delta, 0.0)
        self.assertGreater(good.total, poor.total)

    def test_the_default_weight_keeps_the_term_inert(self) -> None:
        components = compute_reward(
            self._signals(1.0),
            skill_id="写作",
            weights=RewardWeights(use_baseline=True),
            baseline=0.2,
        )
        plain = compute_reward(
            self._signals(0.0),
            skill_id="写作",
            weights=RewardWeights(use_baseline=True),
            baseline=0.2,
        )
        self.assertAlmostEqual(components.total, plain.total, places=6)

    def test_the_goal_baseline_is_rebuilt_from_its_own_series(self) -> None:
        events = [
            {"type": "METHOD_OUTCOME_OBSERVED", "context_key": "solo",
             "activity_type": "solo", "goal_progress": 0.8},
            {"type": "METHOD_OUTCOME_OBSERVED", "context_key": "solo",
             "activity_type": "solo", "goal_progress": 0.9},
            {"type": "METHOD_OUTCOME_OBSERVED", "context_key": "solo",
             "activity_type": "solo", "goal_progress": 1.0},
        ]
        tracker = goal_baseline_tracker_from_events(events)
        self.assertEqual(tracker.samples("solo", activity_type="solo"), 3)
        self.assertGreater(tracker.baseline_for("solo", activity_type="solo"), 0.5)

    def test_an_outcome_without_the_field_does_not_enter_the_goal_series(self) -> None:
        tracker = goal_baseline_tracker_from_events(
            [{"type": "METHOD_OUTCOME_OBSERVED", "context_key": "solo"}]
        )
        self.assertEqual(tracker.samples("solo"), 0)

    def test_the_shadow_payload_carries_the_goal_numbers(self) -> None:
        shadow = shadow_baseline_for(
            self._signals(0.75),
            skill_id="写作",
            context_key="solo",
            weights=RewardWeights(use_baseline=True, goal_progress_weight=0.30),
        )
        payload = shadow.to_payload()
        self.assertEqual(payload["goal_progress"], 0.75)
        self.assertIn("goal_baseline", payload)


class LifecycleWiringTests(unittest.TestCase):
    def setUp(self) -> None:
        self._ctx = temp_workspace()
        self.root = self._ctx.__enter__()  # type: ignore[attr-defined]
        self.dm, self.clock = make_datamanager()
        self.clock.set_stage(Stage.SETTLE)
        self.clock.set_year(2020)
        self.clock.set_week(9)

    def tearDown(self) -> None:
        self._ctx.__exit__(None, None, None)  # type: ignore[attr-defined]

    def _seed_failing(self) -> None:
        store = CapabilityEventStore(self.dm)
        method = Methodology(
            method_id="m-fail",
            skill_id="写作",
            title="硬撑着写下去",
            status="deprecated",
            global_value=-0.5,
            practice_count=4,
            success_count=0,
            last_used="Y2020-W01-activity-D2",
            steps=["写"],
        )
        payload = method.to_dict()
        payload.pop("method_id")
        store.append("METHOD_PROPOSED", method_id="m-fail", payload=payload,
                     idempotency_key="METHOD_PROPOSED:m-fail:-:seed")
        store.append(
            "METHOD_VALUE_UPDATED",
            method_id="m-fail",
            payload={
                "global_value": -0.5, "confidence": 0.5, "practice_count": 4,
                "success_count": 0, "status": "deprecated", "context_values": {},
                "last_used": "Y2020-W01-activity-D2",
            },
            idempotency_key="METHOD_VALUE_UPDATED:m-fail:-:seed",
        )

    def test_the_settlement_archives_and_writes_the_view(self) -> None:
        self._seed_failing()
        lifecycle = MethodLifecycle(
            dm=self.dm, clock=self.clock, config=LifecycleConfig(enabled=True)
        )
        self.assertEqual(lifecycle.archive_failing_methods(), 1)
        view = materializer.build_capability_view(self.dm)
        entry = view["methodologies"]["m-fail"]
        self.assertEqual(entry["status"], "archived")
        # 归档只是降低可访问性：方法本身与它的来源事件都还在
        self.assertEqual(entry["global_value"], -0.5)
        self.assertTrue(entry["source_event_ids"])

    def test_hint_config_reads_the_phase_five_switches(self) -> None:
        config = HintConfig.from_world_config(
            {
                "cognition": {
                    "method_hints": True,
                    "method_hints_bandit": True,
                    "method_lifecycle": True,
                    "method_specialized_margin": 0.3,
                    "reward_caliber": "signed",
                    "goal_progress_weight": 0.3,
                }
            }
        )
        self.assertTrue(config.bandit)
        self.assertEqual(config.reward_caliber, "signed")
        self.assertAlmostEqual(config.specialized_margin, 0.3, places=6)
        self.assertAlmostEqual(config.goal_progress_weight, 0.3, places=6)

    def test_the_run_default_caliber_is_the_signed_one(self) -> None:
        """用户决定（KI-19，2026-09-27）：默认走有符号口径；缺省配置也给 signed。"""
        from src.agents.cognition.method_hints import DEFAULT_REWARD_CALIBER

        config = HintConfig.from_world_config({"cognition": {"method_hints": True}})
        self.assertEqual(config.reward_caliber, DEFAULT_REWARD_CALIBER)
        self.assertEqual(config.reward_caliber, "signed")
        # 直接构造的配置保持惰性（阶段 1~3 口径），跑批才用 `from_world_config`
        self.assertEqual(HintConfig(enabled=True).reward_caliber, "off")

    def test_lifecycle_config_takes_the_week_length_from_the_world(self) -> None:
        config = LifecycleConfig.from_world_config(
            {"time": {"n_week": 10}, "cognition": {"method_lifecycle": True}}
        )
        self.assertTrue(config.enabled)
        self.assertEqual(config.weeks_per_year, 10)


class RewardCaliberTests(unittest.TestCase):
    """KI-19 决策落地：跑批用哪个口径，以及另一个口径同时留档。"""

    def test_the_three_calibers_map_to_the_expected_switches(self) -> None:
        off = weights_for_caliber("off")
        self.assertFalse(off.use_baseline)
        self.assertFalse(off.centred_performance)

        quality = weights_for_caliber("quality")
        self.assertTrue(quality.use_baseline)
        self.assertFalse(quality.centred_performance)

        signed = weights_for_caliber("signed")
        self.assertTrue(signed.use_baseline)
        self.assertTrue(signed.centred_performance)

    def test_an_unknown_caliber_falls_back_to_the_old_behaviour(self) -> None:
        weights = weights_for_caliber("nonsense")
        self.assertFalse(weights.use_baseline)
        self.assertFalse(weights.centred_performance)

    def test_the_goal_weight_is_carried_into_the_caliber(self) -> None:
        self.assertAlmostEqual(
            weights_for_caliber("signed", goal_progress_weight=0.3).goal_progress_weight,
            0.3,
            places=6,
        )

    def test_each_caliber_is_compared_against_the_other_one(self) -> None:
        self.assertEqual(alternative_caliber("signed"), "quality")
        self.assertEqual(alternative_caliber("quality"), "signed")
        # 阶段 1~3 的绝对口径下，值得对照的是要迁过去的那个
        self.assertEqual(alternative_caliber("off"), "signed")

    def test_the_live_score_uses_the_characters_own_baseline(self) -> None:
        """中心化口径下 live 必须走基线，否则会拿中性先验当"常规"。"""
        tracker = BaselineTracker()
        for _ in range(6):
            tracker.observe("solo", 0.05, activity_type="solo")
        signals = OutcomeSignals(
            activity_id="a", activity_type="solo", skill_gains={"写作": 1.5}, turns=3
        )
        with_baseline = live_reward_for(
            signals,
            skill_id="写作",
            context_key="solo",
            tracker=tracker,
            weights=weights_for_caliber("signed"),
        )
        without = live_reward_for(
            signals,
            skill_id="写作",
            context_key="solo",
            weights=weights_for_caliber("signed"),
        )
        # 同样的表现：跟"本人常规"（很低）比是超常，跟中性先验 0.5 比是不足
        self.assertGreater(with_baseline.total, without.total)
        self.assertGreater(with_baseline.total, 0.0)
        self.assertLess(without.total, 0.0)

    def test_an_absolute_caliber_ignores_the_baseline(self) -> None:
        tracker = BaselineTracker()
        for _ in range(6):
            tracker.observe("solo", 0.0, activity_type="solo")
        signals = OutcomeSignals(
            activity_id="a", activity_type="solo", skill_gains={"写作": 3.0}, turns=3
        )
        scored = live_reward_for(
            signals,
            skill_id="写作",
            context_key="solo",
            tracker=tracker,
            weights=weights_for_caliber("off"),
        )
        self.assertFalse(scored.used_baseline)
        self.assertGreater(scored.total, 0.0)


class ActivityTurnsTests(unittest.TestCase):
    """KI-18：活动记录要带上会话长度，时间成本分量才有数据。"""

    def test_a_joint_activity_counts_the_shared_dialog(self) -> None:
        from src.world.activity import activity_turns

        self.assertEqual(activity_turns(object(), dialog_lines=9), 9)

    def test_a_solo_activity_counts_the_characters_own_rounds(self) -> None:
        """`activity_context` 在记录写出前就被释放了，所以轮数由角色自己存下来。"""
        from src.world.activity import activity_turns

        self.assertEqual(activity_turns(SimpleNamespace(last_activity_turns=3)), 3)

    def test_a_missing_count_is_zero_not_an_error(self) -> None:
        from src.world.activity import activity_turns

        self.assertEqual(activity_turns(SimpleNamespace()), 0)
        self.assertEqual(activity_turns(SimpleNamespace(last_activity_turns=None)), 0)

    def test_the_round_counter_reads_the_activity_transcript(self) -> None:
        from src.agents.role_agent import activity_rounds

        self.assertEqual(
            activity_rounds(
                [
                    {"role": "user", "content": "prompt"},
                    {"role": "assistant", "content": "action"},
                    {"role": "user", "content": "tool result"},
                    {"role": "assistant", "content": "final"},
                ]
            ),
            2,
        )
        self.assertEqual(activity_rounds(None), 0)

    def test_the_agent_stashes_the_count_before_releasing_the_context(self) -> None:
        import io as _io
        import pathlib as _pathlib

        source = _io.open(
            _pathlib.Path(__file__).resolve().parents[1]
            / "src"
            / "agents"
            / "role_agent.py",
            encoding="utf-8",
        ).read()
        stashed = source.index("self.last_activity_turns = activity_rounds(")
        cleared = source.index("self.activity_context = None")
        self.assertLess(stashed, cleared, "轮数必须在 activity_context 被清空前存下来")

    def test_the_records_carry_the_field_round_trip(self) -> None:
        from src.world.joint_activity_data import JointActivityRecord
        from src.world.public_activity_data import PublicActivityRecord
        from src.world.solo_activity_data import SoloActivityRecord

        self.assertEqual(SoloActivityRecord.__dataclass_fields__["turns"].default, 0)
        self.assertEqual(JointActivityRecord.__dataclass_fields__["turns"].default, 0)
        self.assertEqual(PublicActivityRecord.__dataclass_fields__["turns"].default, 0)

    def test_a_recorded_turn_count_reaches_the_reward_signals(self) -> None:
        row = _record()
        row["turns"] = 7
        signals = signals_from_activity_record(row)
        self.assertEqual(signals.turns, 7)
        components = compute_reward(signals, skill_id="写作")
        # 12 轮算满：7 轮 = 0.0583 而不是"未知时长"占位值 0.25
        self.assertAlmostEqual(components.time_cost, 7 / 12.0, places=4)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
