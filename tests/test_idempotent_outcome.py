"""阶段 0：幂等键——同一个实践事件不得重复生效（离线，无 LLM 调用）。

设计文档不变量 #8「同一个实践事件不得重复强化」。当前实现的风险点是活动结算
（`apply_activity_outcome`）：resume 后重入某个阶段、或上层重试，都可能把同一
个活动的 delta 再加一次。

修复后：调用方传入 `activity-outcome:<activity_id>:<agent>` 作为幂等键，
DataManager 以 `state.jsonl`（效果账本）为准判断是否已生效；重复调用只告警、
不写状态、不叠加数值。
"""

from __future__ import annotations

import json
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from tests._helpers import make_datamanager, temp_workspace


@dataclass
class _Outcome:
    delta_vitality: int = -5
    delta_fulfillment: Dict[str, int] = field(default_factory=lambda: {"mood": 3})
    delta_skills: Dict[str, float] = field(default_factory=lambda: {"写作": 1})
    delta_money: int = 10
    gain_items: List[str] = field(default_factory=list)


def _state_rows(dm) -> list:
    path = dm.root / "state.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


class IdempotentOutcomeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._ctx = temp_workspace()
        self.root = self._ctx.__enter__()
        self.dm, self.clock = make_datamanager()
        self.dm.save_state(
            {
                "vitality": 80,
                "fulfillment": {"mood": 50, "material": 50, "social": 50, "esteem": 50},
                "assets": {"deposit": 1000, "possessions": []},
                "skills": {"写作": 10},
            }
        )

    def tearDown(self) -> None:
        self._ctx.__exit__(None, None, None)

    def _apply(self, key=None, **kwargs):
        return self.dm.apply_activity_outcome(
            _Outcome(**kwargs), idempotency_key=key
        )

    # -- the guarantee -----------------------------------------------------
    def test_second_application_of_the_same_key_is_a_no_op(self) -> None:
        key = "activity-outcome:joint-Y2020-W01-D2-甲-爬山:甲"
        first = self._apply(key)
        self.assertEqual(first["vitality"], 75)
        self.assertEqual(first["fulfillment"]["mood"], 53)

        second = self._apply(key)

        self.assertEqual(second["vitality"], 75, "vitality must not be applied twice")
        self.assertEqual(second["fulfillment"]["mood"], 53)
        self.assertEqual(second["skills"]["写作"], 11)
        self.assertEqual(second["assets"]["deposit"], 1010)

    def test_duplicate_does_not_append_a_state_record(self) -> None:
        key = "activity-outcome:a1:甲"
        self._apply(key)
        rows_after_first = len(_state_rows(self.dm))
        self._apply(key)
        self.assertEqual(len(_state_rows(self.dm)), rows_after_first)

    def test_state_record_carries_the_key(self) -> None:
        key = "activity-outcome:a1:甲"
        self._apply(key)
        rows = _state_rows(self.dm)
        self.assertEqual(rows[-1]["idempotency_key"], key)

    def test_different_keys_both_apply(self) -> None:
        self._apply("activity-outcome:a1:甲")
        state = self._apply("activity-outcome:a2:甲")
        self.assertEqual(state["vitality"], 70)
        self.assertEqual(state["fulfillment"]["mood"], 56)

    def test_without_a_key_behaviour_is_unchanged(self) -> None:
        self._apply(None)
        state = self._apply(None)
        self.assertEqual(state["vitality"], 70, "legacy callers still accumulate")

    # -- durability across a resume ----------------------------------------
    def test_keys_are_reloaded_from_the_ledger_after_a_restart(self) -> None:
        key = "activity-outcome:a1:甲"
        self._apply(key)

        # A resumed run builds a fresh DataManager over the same files.
        resumed, _ = make_datamanager()
        state = resumed.apply_activity_outcome(_Outcome(), idempotency_key=key)

        self.assertEqual(state["vitality"], 75)
        self.assertEqual(len(_state_rows(resumed)), 2, "no new state record")

    def test_key_recorded_by_a_plain_save_state_is_honoured(self) -> None:
        """Any record carrying the key counts, not just outcome applications."""
        self.dm.save_state(
            {
                "vitality": 60,
                "fulfillment": {"mood": 40, "material": 40, "social": 40, "esteem": 40},
                "assets": {"deposit": 500, "possessions": []},
                "skills": {},
            },
            idempotency_key="activity-outcome:a9:甲",
        )
        state = self._apply("activity-outcome:a9:甲")
        self.assertEqual(state["vitality"], 60)


if __name__ == "__main__":
    unittest.main()
