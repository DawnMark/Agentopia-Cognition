"""阶段 6 第二步：按已练习能力给技能增益封顶（`capability_gain.py`）。

用户决策（2026-09-27）：反事实探针显示 God 对能力**数值**不敏感（14/100 与 55/100
没有区别），所以"能力已经很高就别再涨"这条设计意图不能只写在 prompt 里，必须由代码执行。
这里钉住规则本身（阈值、只封顶不抬高、负值不动）、默认关闭、事件可查，
以及**三类活动走同一条规则**（不变量 #7）。
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.agents.cognition.capability_gain import (
    CAPABILITY_GAIN_CAPPED,
    DEFAULT_CAP_HIGH,
    DEFAULT_CAP_HIGH_GAIN,
    DEFAULT_CAP_MID,
    DEFAULT_CAP_MID_GAIN,
    cap_agent_deltas,
    cap_deltas,
    cap_enabled,
    cap_rules,
    rule_for,
)
from tests._helpers import temp_workspace

NL = chr(10)


def _config(**cognition) -> dict:
    section = {
        "capability_gain_cap": False,
        "capability_cap_mid": DEFAULT_CAP_MID,
        "capability_cap_mid_gain": DEFAULT_CAP_MID_GAIN,
        "capability_cap_high": DEFAULT_CAP_HIGH,
        "capability_cap_high_gain": DEFAULT_CAP_HIGH_GAIN,
    }
    section.update(cognition)
    return {"world": {"cognition": section}}


def _activity(week: int, gains: dict) -> dict:
    return {
        "time": f"Y2020-W{int(week):02d}-activity-D1",
        "type": "solo",
        "outcome": {"delta_skills": gains},
    }


class RuleTests(unittest.TestCase):
    def test_the_rule_picks_the_first_threshold_it_meets(self) -> None:
        rules = cap_rules(_config())
        self.assertEqual(rule_for(0.05, rules), None)
        self.assertEqual(rule_for(DEFAULT_CAP_MID, rules).name, "mid")
        self.assertEqual(
            rule_for(DEFAULT_CAP_HIGH - 0.01, rules).name, "mid"
        )
        self.assertEqual(rule_for(DEFAULT_CAP_HIGH, rules).name, "high")
        self.assertEqual(rule_for(0.95, rules).name, "high")

    def test_the_thresholds_are_configurable(self) -> None:
        rules = cap_rules(
            _config(capability_cap_mid=0.2, capability_cap_high=0.5,
                    capability_cap_mid_gain=2.0, capability_cap_high_gain=1.0)
        )
        self.assertEqual(rule_for(0.3, rules).max_gain, 2.0)
        self.assertEqual(rule_for(0.6, rules).max_gain, 1.0)

    def test_gains_are_only_capped_never_raised(self) -> None:
        rules = cap_rules(_config())
        capped, notes = cap_deltas({"写作": 1.0, "急救": 0.0}, {"写作": 0.9, "急救": 0.9}, rules)
        self.assertEqual(capped, {"写作": 0.0, "急救": 0.0})
        self.assertEqual([note.skill for note in notes], ["写作"])

    def test_a_negative_delta_is_never_touched(self) -> None:
        """God 给了负分就照记：封顶只限制"还能涨多少"，不改判断的方向。"""
        rules = cap_rules(_config())
        capped, notes = cap_deltas({"写作": -2.0}, {"写作": 0.95}, rules)
        self.assertEqual(capped, {"写作": -2.0})
        self.assertEqual(notes, [])

    def test_an_unknown_skill_is_treated_as_unpractised(self) -> None:
        rules = cap_rules(_config())
        capped, notes = cap_deltas({"新技能": 3.0}, {}, rules)
        self.assertEqual(capped, {"新技能": 3.0})
        self.assertEqual(notes, [])

    def test_the_note_records_before_and_after(self) -> None:
        rules = cap_rules(_config())
        _, notes = cap_deltas({"写作": 3.0}, {"写作": DEFAULT_CAP_HIGH + 0.05}, rules)
        note = notes[0].to_dict()
        self.assertEqual(note["delta_before"], 3.0)
        self.assertEqual(note["delta_after"], DEFAULT_CAP_HIGH_GAIN)
        self.assertEqual(note["rule"], "high")
        self.assertAlmostEqual(note["capability"], DEFAULT_CAP_HIGH + 0.05)


class WiringTests(unittest.TestCase):
    """开关、投影与事件：关闭时不做任何事，打开时留下可查的痕迹。"""

    def setUp(self) -> None:
        from tests._helpers import make_datamanager

        self._ctx = temp_workspace()
        self.root = self._ctx.__enter__()  # type: ignore[attr-defined]
        # 真 DataManager：事件写入要走账本身份（append_ledger_record），
        # 用替身对象就测不到"封顶留下了可查的痕迹"这件事。
        self.dm, _clock = make_datamanager()
        self.persona = Path(self.dm.root)
        # 一个练得很熟的技能：12 次练习 → proficiency 0.545，没有方法 → 能力就是 0.545
        self._dump(
            self.persona / "activity.jsonl",
            [_activity(i, {"写作": 2}) for i in range(1, 13)],
        )
        self.agent = SimpleNamespace(name="测试角色", dm=self.dm)

    def tearDown(self) -> None:
        self._ctx.__exit__(None, None, None)  # type: ignore[attr-defined]

    def _dump(self, path: Path, rows: list) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + NL for row in rows),
            encoding="utf-8",
        )

    def _events(self) -> list:
        path = self.persona / "cognition" / "capability_events.jsonl"
        if not path.exists():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_the_switch_is_off_by_default(self) -> None:
        self.assertFalse(cap_enabled({"world": {}}))
        self.assertFalse(cap_enabled({"world": {"cognition": {}}}))
        self.assertFalse(cap_enabled({}))

    def test_switched_off_nothing_happens_not_even_a_projection_rebuild(self) -> None:
        deltas = cap_agent_deltas(self.agent, {"写作": 3.0}, config=_config())
        self.assertEqual(deltas, {"写作": 3.0})
        self.assertEqual(self._events(), [])  # 连事件都不写

    def test_switched_on_a_well_practised_skill_is_capped(self) -> None:
        # 12 次练习 → proficiency 0.545（≥ 下限阈值 0.45）→ high 档：该技能不再涨
        capped = cap_agent_deltas(
            self.agent, {"写作": 3.0}, activity_id="solo-1", config=_config(capability_gain_cap=True)
        )
        self.assertEqual(capped, {"写作": DEFAULT_CAP_HIGH_GAIN})
        events = self._events()
        self.assertEqual([event["type"] for event in events], [CAPABILITY_GAIN_CAPPED])
        note = events[0]
        self.assertEqual(note["skill"], "写作")
        self.assertEqual(note["activity_id"], "solo-1")
        self.assertEqual(note["rule"], "high")
        self.assertEqual(note["delta_after"], DEFAULT_CAP_HIGH_GAIN)

    def test_a_moderately_practised_skill_is_capped_to_one(self) -> None:
        # 4 次练习 → proficiency 0.286（≥ mid 阈值 0.25，< high）→ 一周最多 1 点
        self._dump(
            self.persona / "activity.jsonl",
            [_activity(i, {"写作": 2}) for i in range(1, 5)],
        )
        capped = cap_agent_deltas(
            self.agent, {"写作": 3.0}, activity_id="solo-mid", config=_config(capability_gain_cap=True)
        )
        self.assertEqual(capped, {"写作": DEFAULT_CAP_MID_GAIN})
        self.assertEqual(self._events()[0]["rule"], "mid")

    def test_a_very_well_practised_skill_stops_growing(self) -> None:
        # 30 次练习 → proficiency 0.75 → high 档：该技能这一周不再涨
        self._dump(
            self.persona / "activity.jsonl",
            [_activity(i, {"写作": 2}) for i in range(1, 31)],
        )
        capped = cap_agent_deltas(
            self.agent, {"写作": 3.0}, activity_id="solo-2", config=_config(capability_gain_cap=True)
        )
        self.assertEqual(capped, {"写作": DEFAULT_CAP_HIGH_GAIN})
        self.assertEqual(self._events()[0]["rule"], "high")

    def test_caps_in_different_weeks_are_separate_events(self) -> None:
        """幂等键必须带时间：早先的键是 `<技能>:<activity_id 或 '-'>:<档>`，
        而 Joint/Public 根本拿不到 activity_id、Solo 也只传了 None，于是同一技能同一档
        的第二次封顶起全部被当成重复行丢掉——2 年运行实际封顶 41 / 11 / 12 次，
        只记下 1 / 4 / 1 条（KI-23）。
        """
        for week in (1, 2, 3):
            self.dm.clock.set_week(week)
            cap_agent_deltas(
                self.agent, {"写作": 3.0}, config=_config(capability_gain_cap=True)
            )
        events = self._events()
        self.assertEqual(len(events), 3)
        self.assertEqual(len({event["idempotency_key"] for event in events}), 3)

    def test_the_same_cap_at_the_same_moment_is_written_once(self) -> None:
        """同一时刻、同一技能的同一档只写一条：重放/重试不该产生重复事件。"""
        for _ in range(3):
            cap_agent_deltas(
                self.agent, {"写作": 3.0}, config=_config(capability_gain_cap=True)
            )
        self.assertEqual(len(self._events()), 1)

    def test_a_never_practised_skill_keeps_its_gain(self) -> None:
        capped = cap_agent_deltas(
            self.agent, {"摄影": 3.0}, config=_config(capability_gain_cap=True)
        )
        self.assertEqual(capped, {"摄影": 3.0})
        self.assertEqual(self._events(), [])


class FoldedNameTests(unittest.TestCase):
    """能力族里的折叠名也必须被封顶。

    投影是按能力族归并的，而活动记录写的是原始技能名：`客户沟通` 折进 `人际沟通`。
    第一版按原始名直接查投影 → 查不到 → 永远绕开封顶（实测 2 年运行里 3 条越界记录
    全部是这类名字）。
    """

    def setUp(self) -> None:
        import json as _json

        from tests._helpers import make_datamanager

        self._ctx = temp_workspace()
        self.root = self._ctx.__enter__()  # type: ignore[attr-defined]
        self.dm, _clock = make_datamanager()
        self.persona = Path(self.dm.root)
        world_dir = Path("data") / "regression_world"
        world_dir.mkdir(parents=True, exist_ok=True)
        (world_dir / "skill_aliases.json").write_text(
            _json.dumps(
                {"version": 1, "skills": [{"canonical": "人际沟通", "aliases": ["客户沟通"]}]},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        # 12 次练习，全部记在**折叠名**上 → 折进 `人际沟通` 后 proficiency 0.545 → high 档
        self._dump(
            self.persona / "activity.jsonl",
            [_activity(i, {"客户沟通": 2}) for i in range(1, 13)],
        )
        self.agent = SimpleNamespace(name=self.dm.char, dm=self.dm)

    def tearDown(self) -> None:
        self._ctx.__exit__(None, None, None)  # type: ignore[attr-defined]

    def _dump(self, path: Path, rows: list) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + NL for row in rows),
            encoding="utf-8",
        )

    def test_a_folded_skill_name_is_still_capped(self) -> None:
        from src.agents.cognition.capability_gain import capabilities_for

        caps = capabilities_for(self.dm)
        self.assertGreater(caps.get("客户沟通", 0.0), 0.0)
        self.assertGreater(caps.get("人际沟通", 0.0), 0.0)
        capped = cap_agent_deltas(
            self.agent, {"客户沟通": 3.0}, config=_config(capability_gain_cap=True)
        )
        self.assertEqual(capped, {"客户沟通": DEFAULT_CAP_HIGH_GAIN})


class EndToEndCappingTests(unittest.TestCase):
    """三类活动的评估函数：God 给 3 分、代码封到 1 分（开关打开时）。"""

    def setUp(self) -> None:
        from tests._helpers import make_datamanager

        self._ctx = temp_workspace()
        self.root = self._ctx.__enter__()  # type: ignore[attr-defined]
        self.dm, _clock = make_datamanager()
        self.persona = Path(self.dm.root)
        self._dump(
            self.persona / "activity.jsonl",
            [_activity(i, {"写作": 2}) for i in range(1, 13)],  # 12 次 → prof 0.545 → high 档
        )
        self.profile = {
            "appearance_and_impression": "清瘦",
            "brief_introduction": "在读大学生",
            "details": "住校。",
            "personality_traits": {"qualitative": "内向", "quantitative": {}},
            "talents": {"qualitative": "记忆突出", "quantitative": {"memory": 60}},
            "position": {
                "current": "学生", "role": "student",
                "organization": "某大学", "weekly_income": 300,
            },
            "core_motivation": "把书写完",
            "conflicts": "时间不够",
            "values": "诚实",
            "preferences": "图书馆",
            "init_skills": {"写作": 10},
        }
        self.dm.write_profile(self.profile, year=2020)
        self.dm.save_state(
            {
                "vitality": 80,
                "fulfillment": {"mood": 50, "material": 50, "social": 50, "esteem": 50},
                "assets": {"deposit": 1000, "possessions": []},
                "skills": {"写作": 120},
            }
        )
        self.agent = SimpleNamespace(name=self.dm.char, dm=self.dm)

    def tearDown(self) -> None:
        self._ctx.__exit__(None, None, None)  # type: ignore[attr-defined]

    def _dump(self, path: Path, rows: list) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + NL for row in rows),
            encoding="utf-8",
        )

    def _with_config(self, **cognition):
        from src.config import get_config

        config = get_config()
        original = dict(config["world"].get("cognition") or {})
        config["world"].setdefault("cognition", {}).update(cognition)
        self.addCleanup(lambda: (config["world"]["cognition"].clear(),
                                 config["world"]["cognition"].update(original)))

    def test_solo_gains_are_capped(self) -> None:
        from unittest import mock

        import src.world.god as god

        self._with_config(capability_gain_cap=True, capability_input=True)
        payload = {
            "is_consumption_event": False,
            "outcome": "写完一章",
            "delta_vitality": -1,
            "delta_fulfillment": {"mood": 2},
            "delta_skills": {"写作": 3},
            "delta_money": 0,
            "gain_items": [],
        }
        with mock.patch.object(god, "get_response_with_retry", lambda *a, **k: payload):
            _text, _consumption, deltas = god.evaluate_solo_activity(self.agent, "写作")
        self.assertEqual(deltas["delta_skills"], {"写作": DEFAULT_CAP_HIGH_GAIN})

    def test_joint_and_public_gains_are_capped_by_the_same_rule(self) -> None:
        from unittest import mock

        import src.world.god as god

        self._with_config(capability_gain_cap=True, capability_input=True)
        joint_payload = {
            self.agent.name: {
                "delta_vitality": -1,
                "delta_fulfillment": {"mood": 2, "social": 3, "esteem": 1},
                "delta_skills": {"写作": 3},
            }
        }
        public_payload = {
            self.agent.name: {
                "delta_vitality": 1,
                "delta_fulfillment": {"mood": 1, "social": 1, "esteem": 0},
                "delta_skills": {"写作": 3},
            }
        }
        with mock.patch.object(god, "get_response_with_retry", lambda *a, **k: joint_payload):
            joint = god.evaluate_joint_activity(
                agents=[self.agent], activity_background="一起写东西",
                dialog_history="甲：写吧",
            )
        with mock.patch.object(god, "get_response_with_retry", lambda *a, **k: public_payload):
            public = god.evaluate_public_activity(
                agents=[self.agent], activity_name="写作工作坊",
                event_description="一起写", participation_outputs={self.agent.name: "写"},
            )
        self.assertEqual(
            joint[self.agent.name].delta_skills, {"写作": DEFAULT_CAP_HIGH_GAIN}
        )
        self.assertEqual(
            public[self.agent.name].delta_skills, {"写作": DEFAULT_CAP_HIGH_GAIN}
        )

    def test_with_the_switch_off_the_god_number_is_untouched(self) -> None:
        from unittest import mock

        import src.world.god as god

        self._with_config(capability_gain_cap=False)
        payload = {
            "is_consumption_event": False,
            "outcome": "写完一章",
            "delta_vitality": -1,
            "delta_fulfillment": {"mood": 2},
            "delta_skills": {"写作": 3},
            "delta_money": 0,
            "gain_items": [],
        }
        with mock.patch.object(god, "get_response_with_retry", lambda *a, **k: payload):
            _text, _consumption, deltas = god.evaluate_solo_activity(self.agent, "写作")
        self.assertEqual(deltas["delta_skills"], {"写作": 3})


class ParityTests(unittest.TestCase):
    """三类活动必须走同一条规则（不变量 #7）。"""

    def test_solo_joint_and_public_all_call_the_cap(self) -> None:
        import inspect

        import src.world.god as god

        for func in (
            god.evaluate_solo_activity,
            god.evaluate_joint_activity,
            god.evaluate_public_activity,
        ):
            source = inspect.getsource(func)
            self.assertIn(
                "_cap_skill_gains",
                source,
                f"{func.__name__} 没有按能力封顶增益",
            )

    def test_the_cap_helper_returns_the_input_when_there_is_nothing_to_cap(self) -> None:
        import src.world.god as god

        agent = SimpleNamespace(name="甲", dm=SimpleNamespace())
        self.assertEqual(god._cap_skill_gains(agent, {}), {})


if __name__ == "__main__":
    unittest.main()
