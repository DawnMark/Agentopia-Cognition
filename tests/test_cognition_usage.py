"""认知层的 token 账（`capability` 之外的另一本账：新功能多花了多少 token）。

角色与上帝模型的调用写在 `generation/` / `god/` 日志里并带 token 数，而认知层自己的
抽取调用（方法抽取、记忆合并、Idea 生成、Idea→方法转换）**不写任何生成日志**——
"这一系列新功能比原版多花多少 token"这个问题没有这本账就答不了。
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.agents.cognition.usage import USAGE_FILENAME, estimate_tokens, record_llm_usage
from tests._helpers import make_datamanager, temp_workspace


class UsageRecordTests(unittest.TestCase):
    def setUp(self) -> None:
        self._ctx = temp_workspace()
        self.root = self._ctx.__enter__()  # type: ignore[attr-defined]
        self.dm, _clock = make_datamanager()

    def tearDown(self) -> None:
        self._ctx.__exit__(None, None, None)  # type: ignore[attr-defined]

    def _rows(self) -> list:
        path = Path(self.dm.root) / "cognition" / USAGE_FILENAME
        if not path.exists():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_a_call_is_recorded_with_its_feature_and_token_counts(self) -> None:
        record = record_llm_usage(
            self.dm,
            "memory_extraction",
            messages=[{"role": "system", "content": "把这几件事整理成记忆。" * 10}],
            response={"memories": [{"content": "独自参加活动能学到东西。"}]},
            model="test-model",
        )
        assert record is not None
        self.assertEqual(record["feature"], "memory_extraction")
        self.assertEqual(record["calls"], 1)
        self.assertGreater(record["input_tokens"], 0)
        self.assertGreater(record["output_tokens"], 0)
        rows = self._rows()
        self.assertEqual(len(rows), 1)
        # 走的是账本身份：审计的 ledger_integrity 会检查 persona/ 下的每一行
        self.assertTrue(rows[0].get("ledger_event_id"))
        self.assertEqual(rows[0]["model"], "test-model")

    def test_each_call_is_its_own_row(self) -> None:
        for index in range(3):
            self.dm.clock.set_week(index + 1)
            record_llm_usage(
                self.dm,
                "method_extraction",
                messages=[{"role": "system", "content": f"第 {index} 周"}],
                response="{}",
            )
        rows = self._rows()
        self.assertEqual(len(rows), 3)
        self.assertEqual({row["feature"] for row in rows}, {"method_extraction"})

    def test_a_broken_dm_never_breaks_the_extraction(self) -> None:
        class _Broken:
            root = Path("does/not/exist")

            def append_ledger_record(self, *args, **kwargs):  # noqa: ANN002, ANN003
                raise OSError("disk on fire")

        self.assertIsNone(
            record_llm_usage(
                _Broken(), "idea_phrasing", messages="x", response="y"
            )
        )

    def test_the_token_estimate_matches_the_generation_log_metric(self) -> None:
        """与 `DataManager.save_generation` 用同一个函数，两组数字可以直接相加。"""
        from src.utils import num_tokens_from_string

        text = "角色的计划：本周先把书稿写完，再去跑步。"
        self.assertEqual(estimate_tokens(text), num_tokens_from_string(text))
        self.assertEqual(
            estimate_tokens([{"role": "system", "content": text}]),
            num_tokens_from_string(text),
        )


class InstrumentationTests(unittest.TestCase):
    """四个抽取调用点都必须记账——少一个，"新功能多花了多少"就是错的。"""

    def test_every_cognition_llm_call_site_records_usage(self) -> None:
        import inspect

        from src.agents.cognition import consolidator, idea_pipeline, shadow

        for obj, name in (
            (shadow, "方法抽取"),
            (consolidator, "记忆合并"),
            (idea_pipeline, "Idea 生成 / 方法化"),
        ):
            source = inspect.getsource(obj)
            self.assertIn("record_llm_usage", source, f"{name} 没有记账")
            # 每一个 `get_response_with_retry(` 调用点后面都该跟着一次记账
            self.assertEqual(
                source.count("get_response_with_retry("),
                source.count("record_llm_usage("),
                f"{name}: 调用点数与记账数不一致",
            )


if __name__ == "__main__":
    unittest.main()
