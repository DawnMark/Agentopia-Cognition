"""阶段 0：事件身份（ledger_event_id / schema_version / 幂等键）必须稳定且与顺序无关。

验收口径（设计文档阶段 0）：**中断恢复、重复重放、并行与串行执行产生一致的事件集合**。
前提是同一个逻辑事件在任何运行、任何线程调度下都得到同一个 `ledger_event_id`：
- 由 `(stream, time, payload)` 内容寻址，不含偏移量、行号、运行目录名；
- 标注（rejected / reject_reason / record_id）不改变身份，否则"打标"会变成"新事件"；
- 旧数据没有该字段，读取方必须容忍。
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.world.events import (
    SCHEMA_VERSION,
    attach_event_identity,
    event_stream_id,
    make_event_id,
)
from tests._helpers import make_datamanager, temp_workspace


def _records(path: Path) -> list:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class EventIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self._ctx = temp_workspace()
        self.root = self._ctx.__enter__()

    def tearDown(self) -> None:
        self._ctx.__exit__(None, None, None)

    # -- pure helpers ------------------------------------------------------
    def test_stream_id_drops_the_run_directory(self) -> None:
        self.assertEqual(
            event_stream_id("data/shanghai_apartment_09252339/persona/甲/state.jsonl"),
            "persona/甲/state.jsonl",
        )
        self.assertEqual(
            event_stream_id("data/run/public_events.jsonl"), "public_events.jsonl"
        )
        # Paths that are not inside a run directory keep their own identity.
        self.assertEqual(event_stream_id("logs/error.log"), "logs/error.log")

    def test_id_ignores_key_order_and_annotations(self) -> None:
        stream = "persona/甲/state.jsonl"
        time_str = "Y2020-W01-begin"
        base = make_event_id(
            stream=stream, time_str=time_str, payload={"time": time_str, "content": {"v": 1}}
        )
        reordered = make_event_id(
            stream=stream,
            time_str=time_str,
            payload={"content": {"v": 1}, "time": time_str},
        )
        annotated = make_event_id(
            stream=stream,
            time_str=time_str,
            payload={
                "time": time_str,
                "content": {"v": 1},
                "record_id": "Y2020-W01-begin#7",
                "schema_version": SCHEMA_VERSION,
                "idempotency_key": "k",
                "rejected": True,
                "reject_reason": "because",
            },
        )
        self.assertEqual(base, reordered)
        self.assertEqual(base, annotated)

    def test_id_changes_with_content_stream_or_time(self) -> None:
        base = make_event_id(
            stream="persona/甲/state.jsonl",
            time_str="Y2020-W01-begin",
            payload={"time": "Y2020-W01-begin", "content": {"v": 1}},
        )
        variants = [
            make_event_id(
                stream="persona/甲/state.jsonl",
                time_str="Y2020-W01-begin",
                payload={"time": "Y2020-W01-begin", "content": {"v": 2}},
            ),
            make_event_id(
                stream="persona/乙/state.jsonl",
                time_str="Y2020-W01-begin",
                payload={"time": "Y2020-W01-begin", "content": {"v": 1}},
            ),
            make_event_id(
                stream="persona/甲/state.jsonl",
                time_str="Y2020-W02-begin",
                payload={"time": "Y2020-W02-begin", "content": {"v": 1}},
            ),
        ]
        self.assertEqual(len({base, *variants}), 4)

    def test_attach_keeps_an_existing_id(self) -> None:
        record = {"time": "Y2020-W01-begin", "ledger_event_id": "ev-existing"}
        attach_event_identity(record, stream="persona/甲/state.jsonl")
        self.assertEqual(record["ledger_event_id"], "ev-existing")
        self.assertEqual(record["schema_version"], SCHEMA_VERSION)

    # -- through the ledger ------------------------------------------------
    def test_append_stamps_identity(self) -> None:
        dm, _ = make_datamanager()
        state_path = dm.root / "state.jsonl"
        dm.save_state({"vitality": 80})

        rows = _records(state_path)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["ledger_event_id"].startswith("ev-"))
        self.assertEqual(rows[0]["schema_version"], SCHEMA_VERSION)

    def test_identical_events_get_identical_ids(self) -> None:
        dm, _ = make_datamanager()
        state_path = dm.root / "state.jsonl"
        dm.save_state({"vitality": 80})
        dm.save_state({"vitality": 80})  # same time, same payload

        rows = _records(state_path)
        self.assertEqual(rows[0]["ledger_event_id"], rows[1]["ledger_event_id"])

    def test_different_events_get_different_ids(self) -> None:
        dm, clock = make_datamanager()
        state_path = dm.root / "state.jsonl"
        dm.save_state({"vitality": 80})
        clock.set_week(2)
        dm.save_state({"vitality": 70})

        rows = _records(state_path)
        self.assertNotEqual(rows[0]["ledger_event_id"], rows[1]["ledger_event_id"])

    def test_idempotency_key_is_recorded(self) -> None:
        dm, _ = make_datamanager()
        path = dm.root / "effects.jsonl"
        dm._append_jsonl(path, {"effect": "x"}, idempotency_key="activity-outcome:a1:甲")
        rows = _records(path)
        self.assertEqual(rows[0]["idempotency_key"], "activity-outcome:a1:甲")
        # The key is metadata: adding it must not change the event identity.
        without = attach_event_identity(
            {"time": rows[0]["time"], "effect": "x"},
            stream=event_stream_id(path),
        )
        self.assertEqual(rows[0]["ledger_event_id"], without["ledger_event_id"])

    def test_ids_are_run_independent(self) -> None:
        """A replay in a fresh run directory must produce the same ids."""
        first, _ = make_datamanager(world="world_a")
        first.save_state({"vitality": 80})
        first_path = first.root / "state.jsonl"

        second, _ = make_datamanager(world="world_b")
        second.save_state({"vitality": 80})
        second_path = second.root / "state.jsonl"

        self.assertNotEqual(first_path, second_path)
        self.assertEqual(
            _records(first_path)[0]["ledger_event_id"], _records(second_path)[0]["ledger_event_id"]
        )

    def test_rejection_annotation_preserves_identity(self) -> None:
        dm, _ = make_datamanager()
        record_id = dm.save_generation(
            inputs=[{"role": "user", "content": "hi"}],
            outputs=[{"role": "assistant", "content": "yo"}],
        )
        week_file = dm.generation / "year=2020" / "week=1.jsonl"
        before = _records(week_file)[0]["ledger_event_id"]

        dm.mark_generation_rejected("bad action", record_id)

        after = _records(week_file)[0]
        self.assertTrue(after["rejected"])
        self.assertEqual(after["ledger_event_id"], before)

    def test_legacy_records_without_ids_are_still_readable(self) -> None:
        dm, _ = make_datamanager()
        state_path = dm.root / "state.jsonl"
        legacy = {
            "time": "Y2020-W01-begin",
            "content": {
                "vitality": 80,
                "fulfillment": {"mood": 50, "material": 50, "social": 50, "esteem": 50},
                "assets": {"deposit": 100, "possessions": []},
                "skills": {},
            },
        }
        state_path.write_text(json.dumps(legacy, ensure_ascii=False) + "\n", encoding="utf-8")

        self.assertEqual(
            dm.read_state(at_t="Y2020-W01-begin", exclude_cur_t=False)["vitality"], 80
        )
        rows = _records(state_path)
        self.assertNotIn("ledger_event_id", rows[0])  # untouched on read


if __name__ == "__main__":
    unittest.main()
