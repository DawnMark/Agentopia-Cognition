"""离线回归：体力恢复机制（KI-16，2026-09-26）。

背景：vitality 从 70 起只减不增（没有任何恢复机制），于是每次运行到第 7~10 周，
全体角色的体力都会停在 0，并连带把依赖它的机制一起关掉（Idea 周预算、活动评估、
计划 prompt）。这里钉住新增的恢复规则：

- **每天固定恢复** `world.vitality_recovery.daily_recovery`（默认 1.5，2 年运行定稿，见设计正文 0.15.8）；
- 恢复按 `年-周-日` 幂等：恢复/重放同一天不会重复加；
- 活动自身的 `delta_vitality` 逻辑不变：大行为照旧消耗；
- **活动 delta 原样生效**：god 发多少就是多少，正负都不缩放（休息的收益由 god 自己给）；
- 上下限仍然是 [0, 100]。
"""

from __future__ import annotations

import unittest
from typing import Any, Dict

from src.agents.data_manager import DAILY_VITALITY_RECOVERY, DataManager
from src.world.solo_activity_data import ActionOutcome
from tests._helpers import make_datamanager, temp_workspace


def _state(vitality: int) -> Dict[str, Any]:
    return {
        "vitality": vitality,
        "fulfillment": {"mood": 50, "material": 50, "social": 50, "esteem": 50},
        "assets": {"deposit": 1000, "possessions": []},
        "skills": {"跑步": 10},
    }


def _outcome(delta_vitality: int) -> ActionOutcome:
    return ActionOutcome(
        outcome="测试活动的客观结果",
        delta_vitality=delta_vitality,
        delta_fulfillment={},
        delta_skills={},
        delta_money=0,
    )


class DailyRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._ctx = temp_workspace()
        self._ctx.__enter__()  # type: ignore[attr-defined]
        self.dm, self.clock = make_datamanager()

    def tearDown(self) -> None:
        self._ctx.__exit__(None, None, None)  # type: ignore[attr-defined]

    def _vitality(self) -> float:
        # Not int(): the daily recovery is a fraction (1.5 since the 2-year run).
        return float(self.dm.read_state(exclude_cur_t=False)["vitality"])

    def test_defaults_come_from_the_world_config(self) -> None:
        config = DataManager.vitality_recovery_config()
        self.assertTrue(config["enabled"])
        self.assertEqual(config["daily_recovery"], DAILY_VITALITY_RECOVERY)
        self.assertNotIn(
            "rest_bonus_multiplier",
            config,
            "the world layer must not scale god's vitality deltas",
        )

    def test_daily_recovery_adds_the_fixed_amount(self) -> None:
        self.dm.save_state(_state(40))
        state = self.dm.apply_daily_vitality_recovery(idempotency_key="d1")
        # 恢复量可以是小数（2 年运行把它定在 1.5），所以这里不做 int() 截断
        expected = 40 + DAILY_VITALITY_RECOVERY
        self.assertEqual(self._vitality(), expected)
        self.assertEqual(state["vitality"], expected)

    def test_recovery_is_idempotent_per_day(self) -> None:
        self.dm.save_state(_state(40))
        self.dm.apply_daily_vitality_recovery(idempotency_key="Y2020-W01-D1")
        again = self.dm.apply_daily_vitality_recovery(idempotency_key="Y2020-W01-D1")
        self.assertIsNone(again, "a re-entered day must not recover twice")
        self.assertEqual(self._vitality(), 40 + DAILY_VITALITY_RECOVERY)

    def test_recovery_caps_at_one_hundred(self) -> None:
        self.dm.save_state(_state(99))
        state = self.dm.apply_daily_vitality_recovery(idempotency_key="d1")
        self.assertEqual(state["vitality"], 100)

    def test_recovery_can_be_disabled(self) -> None:
        """`enabled: false` leaves vitality alone (and the config is restored).

        The saved/restored value must be the *descriptor* from the class dict:
        assigning the unwrapped function back would silently turn the
        staticmethod into an instance method for the rest of the test run.
        """
        original = DataManager.__dict__["vitality_recovery_config"]
        try:
            DataManager.vitality_recovery_config = staticmethod(  # type: ignore[assignment]
                lambda: {"enabled": False, "daily_recovery": 3}
            )
            self.dm.save_state(_state(40))
            self.assertIsNone(self.dm.apply_daily_vitality_recovery(idempotency_key="d1"))
            self.assertEqual(self._vitality(), 40)
        finally:
            DataManager.vitality_recovery_config = original  # type: ignore[assignment]

    def test_explicit_amount_overrides_the_config(self) -> None:
        self.dm.save_state(_state(40))
        state = self.dm.apply_daily_vitality_recovery(amount=7, idempotency_key="d1")
        self.assertEqual(state["vitality"], 47)


class VerbatimDeltaTests(unittest.TestCase):
    """活动 delta 严格按 god 发放生效：负的照扣，正的照加，一律不缩放。"""

    def setUp(self) -> None:
        self._ctx = temp_workspace()
        self._ctx.__enter__()  # type: ignore[attr-defined]
        self.dm, self.clock = make_datamanager()
        self.dm.save_state(_state(50))

    def tearDown(self) -> None:
        self._ctx.__exit__(None, None, None)  # type: ignore[attr-defined]

    def _vitality(self) -> float:
        # Not int(): the daily recovery is a fraction (1.5 since the 2-year run).
        return float(self.dm.read_state(exclude_cur_t=False)["vitality"])

    def test_negative_delta_is_applied_verbatim(self) -> None:
        self.dm.apply_activity_outcome(_outcome(-4), idempotency_key="a1")
        self.assertEqual(self._vitality(), 46)

    def test_positive_delta_is_applied_verbatim(self) -> None:
        self.dm.apply_activity_outcome(_outcome(4), idempotency_key="a2")
        self.assertEqual(self._vitality(), 54, "no bonus: god's number is the number")

    def test_zero_delta_changes_nothing(self) -> None:
        self.dm.apply_activity_outcome(_outcome(0), idempotency_key="a3")
        self.assertEqual(self._vitality(), 50)

    def test_delta_cannot_exceed_one_hundred(self) -> None:
        self.dm.save_state(_state(99))
        self.dm.apply_activity_outcome(_outcome(5), idempotency_key="a4")
        self.assertEqual(self._vitality(), 100)

    def test_light_day_recovers_and_heavy_day_still_costs(self) -> None:
        """校准依据：日常恢复 +1.5/天，因此"轻的一天"净增、"重的一天"净减。

        实测活动消耗是 6.3–8.0/周（约 1.3–1.6/天），所以 +1.5/天让普通一周大致打平、
        高强度的一周仍然为负 —— "大行为消耗较多"由 god 的数值直接体现。
        这个值由 2 年运行定稿：+2/天（+10/周）会让体力在约 10 周后稳定贴在 100，
        恢复不再是背景而是天花板；+1/天（+5/周）则净减 3.5/周、20 周掉到 0。
        """
        self.dm.apply_activity_outcome(_outcome(-1), idempotency_key="a5")
        self.dm.apply_daily_vitality_recovery(idempotency_key="Y2020-W01-D1")
        self.assertEqual(self._vitality(), 50.5, "a light day should net positive")

        self.dm.apply_activity_outcome(_outcome(-4), idempotency_key="a6")
        self.dm.apply_daily_vitality_recovery(idempotency_key="Y2020-W01-D2")
        self.assertEqual(self._vitality(), 48, "a heavy day should still end lower")


class WorldHookTests(unittest.TestCase):
    """世界必须每天真的调用恢复（而不是只在 DataManager 里可用）。"""

    def test_day_loop_calls_the_recovery(self) -> None:
        import inspect

        from src.world.world import World

        source = inspect.getsource(World)
        self.assertIn("_apply_daily_vitality_recovery", source)
        # ... and the call site is inside the per-day loop, before activities.
        day_loop = source.split('for day in range(1, self.config["time"]["n_day"] + 1):')[1]
        head = day_loop[:1200]
        self.assertIn("_apply_daily_vitality_recovery", head)
        self.assertLess(
            head.index("_apply_daily_vitality_recovery"),
            head.index("_build_today_activities_all_types"),
        )

    def test_recovery_key_is_unique_per_day(self) -> None:
        import inspect

        from src.world.world import World

        source = inspect.getsource(World._apply_daily_vitality_recovery)
        self.assertIn("daily-vitality:", source)
        self.assertIn("time_state.day", source)


if __name__ == "__main__":
    unittest.main()
