"""阶段 0：绕过 DataManager 的账本写入点也必须带事件身份（离线，无 LLM 调用）。

`DataManager._append_jsonl` 是代理侧的统一入口，但世界级/工具级还有几处自己
开文件的写入点：公共事件、职位申请日志、reward 视图、god 生成数据。它们此前
没有身份字段，于是重放对比工具对这些流"看不见"（显示 0 events），
验收会变成盲测。这里逐个确认它们现在都会盖章。
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.world.clock import Stage
from src.world.events import SCHEMA_VERSION, event_stream_id, stamped_record
from src.world.scheduling import PublicEvent
from tests._helpers import make_datamanager, temp_workspace


def _rows(path: Path) -> list:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class StampedRecordTests(unittest.TestCase):
    def test_stamped_record_is_a_copy_with_identity(self) -> None:
        original = {"time": "Y2020-W01-begin", "value": 1}
        stamped = stamped_record(original, path="data/run_a/public_events.jsonl")
        self.assertNotIn("ledger_event_id", original, "must not mutate the caller's dict")
        self.assertTrue(stamped["ledger_event_id"].startswith("ev-"))
        self.assertEqual(stamped["schema_version"], SCHEMA_VERSION)
        self.assertEqual(stamped["value"], 1)

    def test_same_event_in_two_run_dirs_gets_the_same_id(self) -> None:
        a = stamped_record({"time": "t", "value": 1}, path="data/run_a/x.jsonl")
        b = stamped_record({"time": "t", "value": 1}, path="data/run_b/x.jsonl")
        self.assertEqual(a["ledger_event_id"], b["ledger_event_id"])
        self.assertEqual(event_stream_id("data/run_a/x.jsonl"), "x.jsonl")


class StandaloneWriterTests(unittest.TestCase):
    def setUp(self) -> None:
        self._ctx = temp_workspace()
        self.root = self._ctx.__enter__()

    def tearDown(self) -> None:
        # init_god_module sets module globals: leaving them set makes later
        # tests write god generations into the real working directory.
        import src.world.god as god

        god._god_clock = None
        god._god_data_dir = None
        self._ctx.__exit__(None, None, None)

    def test_public_events_are_stamped(self) -> None:
        from src.world.world import append_public_events

        path = Path("data/regression_world/public_events.jsonl")
        events = [
            PublicEvent(
                event_id="public-2020-W01-1",
                event_name="社区市集",
                start_year=2020,
                start_week=1,
                start_day=3,
                repeat_weeks=1,
                description="周末市集",
                eligible_participants="all",
            )
        ]
        append_public_events(path, events, time_str="Y2020-W01-before_contact")

        row = _rows(path)[0]
        self.assertTrue(row["ledger_event_id"].startswith("ev-"))
        self.assertEqual(row["schema_version"], SCHEMA_VERSION)
        # The domain identifier keeps its own meaning and value.
        self.assertEqual(row["event_id"], "public-2020-W01-1")

    def test_position_application_log_is_stamped(self) -> None:
        from src.world.position_application import _append_position_application_log

        _append_position_application_log(
            world_name="regression_world",
            time_str="Y2020-W01-begin",
            round_num=1,
            agent_name="甲",
            wishes=["A", "B"],
            result="accepted",
            position_name="店员",
        )
        path = Path("data/regression_world/position_application_log.jsonl")
        row = _rows(path)[0]
        self.assertTrue(row["ledger_event_id"].startswith("ev-"))
        self.assertEqual(row["result"], "accepted")

    def test_reward_views_are_stamped(self) -> None:
        from src.world.reward import SocialRanking, _save_reward_jsonl

        ranking = SocialRanking(
            agent_name="甲",
            time="Y2020-W02-settle",
            affection_scores={"乙": 70},
            respect_scores={"乙": 60},
        )
        _save_reward_jsonl([ranking], "regression_world", "rankings", 2020, 2)
        path = Path("data/regression_world/reward/rankings/year=2020/week=2.jsonl")
        row = _rows(path)[0]
        self.assertTrue(row["ledger_event_id"].startswith("ev-"))
        self.assertEqual(row["agent_name"], "甲")

    def test_god_generations_are_stamped(self) -> None:
        import src.world.god as god
        from src.world.clock import Clock

        clock = Clock(start_year=2020, start_week=1)
        god.init_god_module(clock=clock, data_dir="regression_world")
        god.save_generation(
            feature="regression_feature",
            inputs=[{"role": "system", "content": "prompt"}],
            outputs=[{"role": "assistant", "content": "answer"}],
        )
        path = Path("data/regression_world/god/regression_feature/year=2020/week=1.jsonl")
        row = _rows(path)[0]
        self.assertTrue(row["ledger_event_id"].startswith("ev-"))
        self.assertEqual(row["schema_version"], SCHEMA_VERSION)


class SoloActivityIdTests(unittest.TestCase):
    """阶段 0「增加 activity ID」：solo 记录此前没有 activity_id。"""

    def setUp(self) -> None:
        self._ctx = temp_workspace()
        self.root = self._ctx.__enter__()
        self.dm, _ = make_datamanager()

    def tearDown(self) -> None:
        self._ctx.__exit__(None, None, None)

    def test_roundtrip_keeps_the_activity_id(self) -> None:
        from src.world.clock import TimeState
        from src.world.solo_activity_data import ActionOutcome, SoloActivityRecord

        record = SoloActivityRecord(
            activity_id="solo-Y2020-W01-activity-D2-甲",
            agent_name="甲",
            time=TimeState(2020, 1, Stage.ACTIVITY, 2, 0),
            content="读书",
            outcome=ActionOutcome(outcome="读完一章", delta_vitality=1, delta_fulfillment={}, delta_skills={}),
        )
        restored = SoloActivityRecord.from_dict(record.to_dict())
        self.assertEqual(restored.activity_id, "solo-Y2020-W01-activity-D2-甲")
        self.assertEqual(record.to_dict()["activity_id"], "solo-Y2020-W01-activity-D2-甲")

    def test_ledger_record_carries_the_activity_id(self) -> None:
        from src.world.clock import TimeState
        from src.world.solo_activity_data import ActionOutcome, SoloActivityRecord

        self.dm.append_activity_record(
            SoloActivityRecord(
                activity_id="solo-Y2020-W01-activity-D2-甲",
                agent_name=self.dm.char,
                time=TimeState(2020, 1, Stage.ACTIVITY, 2, 0),
                content="读书",
                outcome=ActionOutcome(
                    outcome="读完一章",
                    delta_vitality=1,
                    delta_fulfillment={},
                    delta_skills={},
                ),
            )
        )
        row = _rows(self.dm.root / "activity.jsonl")[0]
        self.assertEqual(row["activity_id"], "solo-Y2020-W01-activity-D2-甲")
        self.assertTrue(row["ledger_event_id"].startswith("ev-"))

    def test_legacy_solo_records_without_the_field_still_load(self) -> None:
        from src.world.solo_activity_data import SoloActivityRecord

        legacy = {
            "type": "solo",
            "agent_name": "甲",
            "time": "Y2020-W01-activity-D2",
            "content": "读书",
            "reflection": "",
            "outcome": None,
            "consumption_options_offered": [],
            "consumption_purchased": None,
            "purchase_response": "",
        }
        self.assertEqual(SoloActivityRecord.from_dict(legacy).activity_id, "")


if __name__ == "__main__":
    unittest.main()
