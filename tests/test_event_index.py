"""阶段 0：事件索引视图与重放对比（离线，无 LLM 调用）。

`views/event_index.json` 是从账本**派生**的视图：可删除、可随时重建，不写账本。
对比按 `ledger_event_id` 进行，因此与物理行序、线程调度、运行目录名无关——这正是
"并行 vs 串行""原始 vs 重放""中断恢复 vs 未中断"需要的判据。
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.world.views import (
    build_event_index,
    compare_event_index,
    summarize_index,
    write_event_index,
)
from tests._helpers import temp_workspace


def _write_ledger(run_dir: Path, stream: str, rows: list) -> None:
    path = run_dir / stream
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )


def _row(time: str, event_id: str, **extra) -> dict:
    return {"time": time, "ledger_event_id": event_id, **extra}


class EventIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self._ctx = temp_workspace()
        self.root = self._ctx.__enter__()
        self.run_a = Path("data/run_a")
        self.run_b = Path("data/run_b")
        for run in (self.run_a, self.run_b):
            _write_ledger(
                run,
                "persona/甲/state.jsonl",
                [
                    _row("Y2020-W01-begin", "ev-1"),
                    _row("Y2020-W02-begin", "ev-2"),
                ],
            )
            _write_ledger(
                run,
                "persona/甲/contact/乙.jsonl",
                [_row("Y2020-W01-contact-S1", "ev-3", **{"from": "甲", "content": "hi"})],
            )

    def tearDown(self) -> None:
        self._ctx.__exit__(None, None, None)

    # -- view building -----------------------------------------------------
    def test_index_reports_streams_and_counts(self) -> None:
        index = build_event_index(self.run_a)
        self.assertEqual(index["records"], 3)
        self.assertEqual(index["legacy_records"], 0)
        self.assertEqual(
            sorted(index["streams"]), ["persona/甲/contact/乙.jsonl", "persona/甲/state.jsonl"]
        )
        self.assertEqual(index["streams"]["persona/甲/state.jsonl"]["event_ids"], ["ev-1", "ev-2"])
        self.assertEqual(
            index["streams"]["persona/甲/state.jsonl"]["first_time"], "Y2020-W01-begin"
        )
        self.assertEqual(
            index["streams"]["persona/甲/state.jsonl"]["last_time"], "Y2020-W02-begin"
        )

    def test_index_is_run_directory_independent(self) -> None:
        self.assertTrue(compare_event_index(build_event_index(self.run_a), build_event_index(self.run_b))["identical"])

    def test_legacy_records_are_counted_not_invented(self) -> None:
        _write_ledger(
            self.run_a,
            "persona/甲/weekly_diary.jsonl",
            [{"time": "Y2020-W01-review", "content": "old, no id"}],
        )
        index = build_event_index(self.run_a)
        self.assertEqual(index["legacy_records"], 1)
        self.assertEqual(index["streams"]["persona/甲/weekly_diary.jsonl"]["event_ids"], [])

    def test_duplicate_events_are_reported(self) -> None:
        _write_ledger(
            self.run_a,
            "persona/甲/dup.jsonl",
            [_row("Y2020-W01-begin", "ev-x"), _row("Y2020-W01-begin", "ev-x")],
        )
        index = build_event_index(self.run_a)
        self.assertEqual(index["streams"]["persona/甲/dup.jsonl"]["duplicate_events"], 1)
        self.assertEqual(index["streams"]["persona/甲/dup.jsonl"]["event_ids"], ["ev-x"])

    def test_views_directory_is_not_part_of_the_ledger(self) -> None:
        write_event_index(self.run_a)
        index = build_event_index(self.run_a)
        self.assertNotIn("views/event_index.json", index["streams"])
        self.assertEqual(index["records"], 3)

    # -- comparison --------------------------------------------------------
    def test_missing_event_is_detected(self) -> None:
        _write_ledger(
            self.run_b,
            "persona/甲/state.jsonl",
            [_row("Y2020-W01-begin", "ev-1")],
        )
        report = compare_event_index(build_event_index(self.run_a), build_event_index(self.run_b))
        self.assertFalse(report["identical"])
        self.assertEqual(
            report["differences"]["persona/甲/state.jsonl"]["missing_in_b"], ["ev-2"]
        )

    def test_extra_event_is_detected(self) -> None:
        _write_ledger(
            self.run_b,
            "persona/甲/state.jsonl",
            [
                _row("Y2020-W01-begin", "ev-1"),
                _row("Y2020-W02-begin", "ev-2"),
                _row("Y2020-W03-begin", "ev-9"),
            ],
        )
        report = compare_event_index(build_event_index(self.run_a), build_event_index(self.run_b))
        self.assertFalse(report["identical"])
        self.assertEqual(
            report["differences"]["persona/甲/state.jsonl"]["extra_in_b"], ["ev-9"]
        )

    def test_stream_only_in_one_run_is_detected(self) -> None:
        _write_ledger(self.run_b, "persona/乙/state.jsonl", [_row("Y2020-W01-begin", "ev-z")])
        report = compare_event_index(build_event_index(self.run_a), build_event_index(self.run_b))
        self.assertFalse(report["identical"])
        self.assertIn("persona/乙/state.jsonl", report["streams_only_in_b"])

    def test_line_order_does_not_matter(self) -> None:
        _write_ledger(
            self.run_b,
            "persona/甲/state.jsonl",
            [
                _row("Y2020-W02-begin", "ev-2"),
                _row("Y2020-W01-begin", "ev-1"),
            ],
        )
        report = compare_event_index(build_event_index(self.run_a), build_event_index(self.run_b))
        self.assertTrue(report["identical"], "order must not affect the comparison")

    # -- rebuildability ----------------------------------------------------
    def test_view_can_be_deleted_and_rebuilt_identically(self) -> None:
        out = write_event_index(self.run_a)
        first = out.read_text(encoding="utf-8")
        out.unlink()

        out2 = write_event_index(self.run_a)
        self.assertEqual(out2.read_text(encoding="utf-8"), first)

    def test_summary_rows_are_sorted(self) -> None:
        rows = summarize_index(build_event_index(self.run_a))
        self.assertEqual([r[0] for r in rows], sorted(r[0] for r in rows))
        self.assertEqual(dict((s, n) for s, n, _ in rows)["persona/甲/state.jsonl"], 2)


if __name__ == "__main__":
    unittest.main()
