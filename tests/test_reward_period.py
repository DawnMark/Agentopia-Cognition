"""⑥ reward 周期：配置、提示词与实现必须同一个口径（离线，无 LLM 调用）。

背景：配置与提示词告诉角色"每 `period_weeks` 周结算一次奖励"
（prompts.py 会把这个数字插进角色提示词），但 `_calculate_rewards` 实际只在
年末调用一次；而主观奖励的窗口又直接取 `period_weeks`——`n_week=10`、
`period_weeks=5` 时，W01–W05 的满足度**永远不会**进入奖励统计。

本仓库的口径（用户已确认）：周期 = 一个模拟年（`period_weeks == n_week`），
`resolve_period_weeks` 把这条约束变成显式校验，主观窗口随周期覆盖整年。
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.config import get_config
from src.world.clock import Stage
from src.world.reward import resolve_period_weeks
from tests._helpers import make_datamanager, persona_root, temp_workspace


def _state_row(week: int, mood: int) -> str:
    state = {
        "vitality": 80,
        "fulfillment": {"mood": mood, "material": 50, "social": 50, "esteem": 50},
        "assets": {"deposit": 1000, "possessions": []},
        "skills": {},
    }
    return json.dumps(
        {"time": f"Y2020-W{week:02d}-begin", "content": state}, ensure_ascii=False
    )


class RewardPeriodTests(unittest.TestCase):
    def setUp(self) -> None:
        self._ctx = temp_workspace()
        self.root = self._ctx.__enter__()
        self.dm, self.clock = make_datamanager()

    def tearDown(self) -> None:
        self._ctx.__exit__(None, None, None)

    # -- config guard ------------------------------------------------------
    def test_shipped_config_has_period_equal_to_the_year(self) -> None:
        cfg = get_config()["world"]
        self.assertEqual(
            resolve_period_weeks(cfg),
            cfg["time"]["n_week"],
            "config.json must not advertise a settlement period the code does not implement",
        )

    def test_mismatched_period_is_aligned_to_the_year(self) -> None:
        """An old run config (period_weeks=5) must not shorten the window."""
        cfg = {"time": {"n_week": 10}, "reward": {"period_weeks": 5}}
        self.assertEqual(resolve_period_weeks(cfg), 10)
        # Aligned in place: the same dict feeds the agent-facing prompt.
        self.assertEqual(cfg["reward"]["period_weeks"], 10)

    # -- subjective window -------------------------------------------------
    def test_subjective_window_covers_the_whole_period(self) -> None:
        state_path = persona_root() / "state.jsonl"
        state_path.write_text(
            "\n".join(_state_row(w, 40 + w) for w in range(1, 11)) + "\n",
            encoding="utf-8",
        )
        self.clock.set_week(10)
        self.clock.set_stage(Stage.SETTLE)

        period = resolve_period_weeks(get_config()["world"])
        # N week-long intervals need N+1 boundary snapshots, so the window
        # spans the whole period: W01-begin through the settlement point.
        history = self.dm.get_fulfillment_history(n_weeks=period + 1)

        self.assertEqual(len(history), 11)
        self.assertEqual(history[0]["time"], "Y2020-W01-begin")
        self.assertEqual(history[-1]["fulfillment"]["mood"], 50)

        # The old window (period_weeks=5 while settling once a year) started
        # mid-year, so W01–W06 could never influence the reward.
        old_window = self.dm.get_fulfillment_history(n_weeks=5)
        self.assertEqual(old_window[0]["time"], "Y2020-W07-begin")
        self.assertLess(len(old_window), len(history))


if __name__ == "__main__":
    unittest.main()
