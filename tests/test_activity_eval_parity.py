"""⑦ 三类活动的 God 评估输入一致性（离线，无 LLM 调用）。

背景：`evaluate_solo_activity` 把 `character_prompt()`（完整人格 + talents +
当前状态）交给 God 模型打分，而 `evaluate_joint_activity` /
`evaluate_public_activity` 用的是 `get_profile_for_activity_eval()`
（裁剪外观 + 简述 + 定性人格 + 技能）——**没有 talents**。
同一个角色在单人活动和集体活动中被放在不同的信息基础上评价。

已确认的修复口径（用户决策 C）：只给 Joint/Public 补 talents，Solo 保持全量输入；
Public 与 Solo/Joint 的"状态应用 vs 反思"时序差异本次记录不改。
"""

from __future__ import annotations

import unittest
from typing import Dict, List
from unittest import mock

import src.world.god as god
from src.agents.data_manager import DataManager
from tests._helpers import make_datamanager, temp_workspace

YEAR = 2020


class _StubAgent:
    """Just enough agent for the evaluators: a name and a DataManager."""

    def __init__(self, name: str, dm: DataManager) -> None:
        self.name = name
        self.dm = dm


def _profile() -> Dict:
    return {
        "appearance_and_impression": "清瘦，戴眼镜",
        "brief_introduction": "在读大学生",
        "details": "住校，周末回家。",
        "personality_traits": {
            "qualitative": "内向，慢热",
            "quantitative": {"openness": 50},
        },
        "talents": {
            "qualitative": "逻辑与记忆突出，肢体协调一般",
            "quantitative": {"logic": 72, "memory": 60, "coordination": 35},
        },
        "position": {
            "current": "学生",
            "role": "student",
            "organization": "某大学",
            "weekly_income": 300,
        },
        "core_motivation": "把书写完",
        "conflicts": "时间不够",
        "values": "诚实",
        "preferences": "安静的图书馆",
        "init_skills": {"写作": 10},
    }


def _state() -> Dict:
    return {
        "vitality": 80,
        "fulfillment": {"mood": 50, "material": 50, "social": 50, "esteem": 50},
        "assets": {"deposit": 1000, "possessions": []},
        "skills": {"写作": 12},
    }


class ActivityEvalParityTests(unittest.TestCase):
    def setUp(self) -> None:
        self._ctx = temp_workspace()
        self.root = self._ctx.__enter__()
        self.dm, self.clock = make_datamanager()
        self.dm.write_profile(_profile(), year=YEAR)
        self.dm.save_state(_state())
        self.agent = _StubAgent(self.dm.char, self.dm)

    def tearDown(self) -> None:
        self._ctx.__exit__(None, None, None)

    # -- the shared profile block ------------------------------------------
    def test_eval_profile_includes_talents(self) -> None:
        text = self.dm.get_profile_for_activity_eval()
        self.assertIn("逻辑与记忆突出", text)  # qualitative
        self.assertIn("logic: 72", text)  # quantitative
        self.assertIn("memory: 60", text)
        self.assertIn("写作: 12", text)  # skills still present

    def test_eval_profile_survives_a_profile_without_talents(self) -> None:
        profile = _profile()
        profile.pop("talents")
        self.dm.write_profile(profile, year=YEAR)
        text = self.dm.get_profile_for_activity_eval()
        self.assertIn("Talents (innate, 0-100): None", text)

    # -- both collective evaluators use that block -------------------------
    def test_joint_and_public_use_the_shared_eval_profile(self) -> None:
        calls: List[str] = []
        original = DataManager.get_profile_for_activity_eval

        def _recorder(self):  # noqa: ANN001 - patched method
            calls.append(self.char)
            return original(self)

        joint_payload = {
            self.agent.name: {
                "delta_vitality": -1,
                "delta_fulfillment": {"mood": 2, "social": 3, "esteem": 1},
                "delta_skills": {"写作": 1},
            }
        }
        public_payload = {
            self.agent.name: {
                "delta_vitality": 1,
                "delta_fulfillment": {"mood": 1, "social": 1, "esteem": 0},
                "delta_skills": {},
            }
        }

        with mock.patch.object(
            DataManager, "get_profile_for_activity_eval", _recorder
        ):
            with mock.patch.object(
                god, "get_response_with_retry", lambda *a, **k: joint_payload
            ):
                god.evaluate_joint_activity(
                    agents=[self.agent],
                    activity_background="一起爬山",
                    dialog_history="甲：走吗",
                )
            with mock.patch.object(
                god, "get_response_with_retry", lambda *a, **k: public_payload
            ):
                god.evaluate_public_activity(
                    agents=[self.agent],
                    activity_name="社区活动",
                    event_description="周末市集",
                    participation_outputs={self.agent.name: "帮忙摆摊"},
                )

        self.assertEqual(calls, [self.dm.char, self.dm.char])

    # -- Solo stays on the full persona prompt (deliberate) ----------------
    def test_solo_still_uses_the_full_persona_prompt(self) -> None:
        calls: List[str] = []
        flags: List[bool] = []

        def _recorder(self, **kwargs):  # noqa: ANN001 - patched method
            calls.append(self.char)
            # 阶段 6：Solo 也必须要求能力块，否则三类活动又用了不同的评估基础
            flags.append(bool(kwargs.get("include_capability")))
            return "FULL PERSONA PROMPT"

        solo_payload = {
            "is_consumption_event": False,
            "outcome": "读完一章书",
            "delta_vitality": -1,
            "delta_fulfillment": {"mood": 2},
            "delta_skills": {"写作": 1},
            "delta_money": 0,
            "gain_items": [],
        }

        with mock.patch.object(DataManager, "character_prompt", _recorder):
            with mock.patch.object(
                god, "get_response_with_retry", lambda *a, **k: solo_payload
            ):
                outcome_text, is_consumption, deltas = god.evaluate_solo_activity(
                    self.agent, "读书"
                )

        self.assertEqual(calls, [self.dm.char])
        self.assertEqual(flags, [True])
        self.assertFalse(is_consumption)
        self.assertEqual(outcome_text, "读完一章书")
        self.assertEqual(deltas["delta_skills"], {"写作": 1})

    def test_the_capability_flag_is_off_for_every_other_prompt(self) -> None:
        """只有活动评估要能力块：计划/反思/联系 prompt 保持逐字节不变。"""
        self.assertEqual(self.dm.read_skills_prompt(), self.dm.read_skills_prompt(
            include_capability=False
        ))
        self.assertNotIn("Practised capability", self.dm.read_skills_prompt())
        self.assertNotIn(
            "Practised capability", self.dm.character_prompt()
        )  # 默认不带
        # 开关本身是关闭的，所以即使显式要求也不会有内容
        self.assertNotIn(
            "Practised capability",
            self.dm.character_prompt(include_capability=True),
        )


if __name__ == "__main__":
    unittest.main()
