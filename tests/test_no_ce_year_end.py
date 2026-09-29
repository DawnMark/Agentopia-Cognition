"""⑤ `--no-context-engineering` 对照组必须能跑完整年（离线，无 LLM 调用）。

背景：`no_context_engineering` 关闭了 REVIEW 阶段的周记写入
（role_agent.review 只在未开启时 append_weekly_summary），而年末的
`update_yearly_profile` 用 `assert summaries` 要求周记必须存在——
于是对照组跑到第一年年末就必然崩溃，方案不变量"新系统必须可以关闭并保留
旧行为作为对照组"无法成立。

修复后：没有周记时显式降级——不调用 LLM、原样沿用上一年的 profile，
并把原因写进 verify 日志。
"""

from __future__ import annotations

import copy
import unittest
from unittest import mock

import src.world.god as god


def _profile() -> dict:
    return {
        "personality_traits": {"qualitative": "内向", "quantitative": {"openness": 50}},
        "talents": {"qualitative": "普通", "quantitative": {"logic": 50}},
        "position": {"current": "学生"},
    }


class _StubDM:
    def __init__(self, profile: dict, summaries: list) -> None:
        self._profile = profile
        self._summaries = summaries

    def read_profile(self) -> dict:
        return copy.deepcopy(self._profile)

    def read_weekly_summaries(self, n_weeks: int) -> list:
        return copy.deepcopy(self._summaries)


class _StubAgent:
    def __init__(self, profile: dict, summaries: list) -> None:
        self.name = "测试角色"
        self.dm = _StubDM(profile, summaries)


class NoContextEngineeringYearEndTests(unittest.TestCase):
    def setUp(self) -> None:
        # The god module can write generation files; keep that out of the repo.
        from tests._helpers import temp_workspace

        self._ctx = temp_workspace()
        self._ctx.__enter__()

    def tearDown(self) -> None:
        self._ctx.__exit__(None, None, None)

    def test_missing_summaries_carry_the_profile_over_without_an_llm_call(self) -> None:
        profile = _profile()
        agent = _StubAgent(profile, summaries=[])
        calls: list = []

        def _unexpected_llm_call(*args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError("no weekly summaries -> no LLM call expected")

        with mock.patch.object(god, "get_response_with_retry", _unexpected_llm_call):
            new_profile = god.update_yearly_profile(agent, 2020, 2021)

        self.assertEqual(calls, [])
        self.assertEqual(new_profile, profile)
        # The caller writes this straight to profile/year=<next>.json, so it has
        # to stay a complete profile rather than a partial diff.
        self.assertIn("personality_traits", new_profile)
        self.assertIn("talents", new_profile)

    def test_missing_summaries_does_not_raise(self) -> None:
        """The old assert killed the run at the first year end."""
        agent = _StubAgent(_profile(), summaries=[])
        try:
            god.update_yearly_profile(agent, 2020, 2021)
        except AssertionError as e:  # pragma: no cover - regression guard
            self.fail(f"year-end profile update raised: {e}")

    def test_summaries_still_drive_an_llm_update(self) -> None:
        profile = _profile()
        summaries = [
            {"time": "Y2020-W01-review", "content": "第一周"},
            {"time": "Y2020-W02-review", "content": "第二周"},
        ]
        agent = _StubAgent(profile, summaries=summaries)

        updated = copy.deepcopy(profile)
        updated["talents"]["quantitative"]["logic"] = 53
        captured: dict = {}

        def _fake_llm(*args, **kwargs):
            captured["messages"] = kwargs.get("messages")
            return copy.deepcopy(updated)

        with mock.patch.object(god, "get_response_with_retry", _fake_llm):
            new_profile = god.update_yearly_profile(agent, 2020, 2021)

        self.assertEqual(new_profile["talents"]["quantitative"]["logic"], 53)
        # Both summaries must reach the prompt.
        prompt_text = captured["messages"][0]["content"]
        self.assertIn("第一周", prompt_text)
        self.assertIn("第二周", prompt_text)


if __name__ == "__main__":
    unittest.main()
