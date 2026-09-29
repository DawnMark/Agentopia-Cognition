"""KI-1 回归：最终回答必须是 assistant 消息、周记不得被污染（离线，无 LLM 调用）。

背景见 `docs/known-issues.md`：生成循环可能以"工具结果"收尾，`review()` 又把
`outputs[-1]` 当作最终回答，解析器在找不到 `Summary:` 时整段返回，于是
`<think_brief>` 摘要头（或工具结果本身）被写进 weekly_diary，并流入后续 prompt
与阶段 1/2 的抽取证据。

修复分三层：
- A：只接受"最后一条 assistant 且有内容、无工具调用"的消息；否则补一次不带工具的
  收尾调用；空响应不再进缓存、也不再把空缓存当命中；
- B：解析器剥离 `<think_brief>` 并要求 `Summary:`；缺失时标记该 generation 为
  rejected 且不写周记；
- C：`review()` 关闭 compact 摘要，从源头去掉 `<think_brief>` 前缀。
"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

from src.agents.role_agent import RoleAgent
from src.utils import _ERROR_RESPONSE
from tests._helpers import temp_workspace


class _StubLogger:
    def __init__(self) -> None:
        self.warnings: List[str] = []
        self.infos: List[str] = []

    def warning(self, message: str, *args: Any) -> None:
        self.warnings.append(str(message))

    def info(self, message: str, *args: Any) -> None:
        self.infos.append(str(message))

    def debug(self, message: str, *args: Any) -> None:
        pass


class _StubDM:
    """Records what review() would persist, without touching the file system."""

    def __init__(self) -> None:
        self.summaries: List[str] = []
        self.rejected: List[tuple] = []

    def append_weekly_summary(self, content: str) -> None:
        self.summaries.append(content)

    def mark_generation_rejected(self, reason: str, record_id: Any = None) -> None:
        self.rejected.append((reason, record_id))


def _agent(*, no_ce: bool = False) -> RoleAgent:
    agent = RoleAgent.__new__(RoleAgent)  # no __init__: no config, no LLM
    agent.dm = _StubDM()
    agent.logger = _StubLogger()
    agent.no_context_engineering = no_ce
    agent.last_main_generation_id = "Y2020-W02-review#36"
    return agent


class FinalAnswerSelectionTests(unittest.TestCase):
    def test_a_tool_result_is_never_the_final_answer(self) -> None:
        outputs = [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "content": "SUCCESS: Content of characters/甲.txt: ..."},
        ]
        self.assertIsNone(RoleAgent._usable_final_answer(outputs))

    def test_an_assistant_answer_is_accepted(self) -> None:
        message = {"role": "assistant", "content": "Summary: 这一周……"}
        self.assertEqual(RoleAgent._usable_final_answer([message]), message)

    def test_tool_calls_and_empty_content_are_rejected(self) -> None:
        self.assertIsNone(
            RoleAgent._usable_final_answer(
                [{"role": "assistant", "content": "partial", "tool_calls": [{"id": "1"}]}]
            )
        )
        self.assertIsNone(
            RoleAgent._usable_final_answer([{"role": "assistant", "content": "   "}])
        )
        self.assertIsNone(RoleAgent._usable_final_answer([]))
        self.assertIsNone(RoleAgent._usable_final_answer([{"role": "system", "content": "x"}]))

    def test_a_mid_turn_assistant_message_is_not_reused(self) -> None:
        """The last message decides: an earlier draft is not the answer."""
        outputs = [
            {"role": "assistant", "content": "Summary: 早期草稿"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "content": "ok"},
        ]
        self.assertIsNone(RoleAgent._usable_final_answer(outputs))


class FinalizeCallTests(unittest.TestCase):
    def test_finalize_call_runs_without_tools_and_appends_the_answer(self) -> None:
        agent = _agent()
        outputs = [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "content": "SUCCESS: scratchpad"},
        ]
        answer = [{"role": "assistant", "content": "Summary: 最终答案"}]
        with mock.patch(
            "src.agents.role_agent.generate_with_fc", return_value=answer
        ) as llm:
            message = agent._finalize_without_tools(
                model="test-model", inputs=[{"role": "system", "content": "p"}],
                outputs=outputs, cache_file=".cache.pkl",
            )

        self.assertIsNotNone(message)
        self.assertEqual(message["content"], "Summary: 最终答案")
        self.assertEqual(outputs[-1], answer[0], "the finalize answer is recorded")
        kwargs = llm.call_args.kwargs
        self.assertEqual(kwargs["functions"], [], "no tools on the finalize call")
        self.assertEqual(kwargs["tool_choice"], "none")

    def test_finalize_failure_is_not_fatal(self) -> None:
        agent = _agent()
        with mock.patch(
            "src.agents.role_agent.generate_with_fc", side_effect=RuntimeError("boom")
        ):
            self.assertIsNone(
                agent._finalize_without_tools(
                    model="m", inputs=[], outputs=[], cache_file=".cache.pkl"
                )
            )

    def test_finalize_returning_an_empty_answer_is_not_usable(self) -> None:
        agent = _agent()
        with mock.patch("src.agents.role_agent.generate_with_fc", return_value=[]):
            self.assertIsNone(
                agent._finalize_without_tools(
                    model="m", inputs=[], outputs=[], cache_file=".cache.pkl"
                )
            )


class ReviewParsingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = _agent()

    def test_a_compact_header_is_stripped(self) -> None:
        text = (
            "<think_brief> (A summary of my thinking and function calling process)\n\n"
            "Background and motivations: ...\n</think_brief>\n\n"
            "Summary:\n这周完成了初稿。\n\nReflection:\n下次先列场景。"
        )
        parsed = self.agent._parse_review_response(text)
        self.assertTrue(parsed.startswith("Summary:"))
        self.assertNotIn("<think_brief>", parsed)

    def test_a_truncated_compact_header_is_stripped(self) -> None:
        text = "<think_brief> (A summary ...\nBackground and motivations: 被截断"
        self.assertEqual(self.agent._parse_review_response(text), "")

    def test_a_tool_result_yields_nothing(self) -> None:
        text = "SUCCESS: Content of characters/上官霄月.txt:\n关系是同事"
        self.assertEqual(self.agent._parse_review_response(text), "")

    def test_a_normal_review_is_kept_verbatim(self) -> None:
        text = "Thinking: ...\n\nSummary:\n写完了。\n\nReflection:\n保持节奏。"
        parsed = self.agent._parse_review_response(text)
        self.assertEqual(parsed, "Summary:\n写完了。\n\nReflection:\n保持节奏。")

    def test_empty_input_is_empty(self) -> None:
        self.assertEqual(self.agent._parse_review_response(""), "")
        self.assertEqual(self.agent._parse_review_response(_ERROR_RESPONSE), "")


class SaveReviewTests(unittest.TestCase):
    def test_a_good_review_is_stored(self) -> None:
        agent = _agent()
        agent._save_review("Summary:\n这周过得不错。")
        self.assertEqual(agent.dm.summaries, ["Summary:\n这周过得不错。"])
        self.assertEqual(agent.dm.rejected, [])

    def test_a_missing_summary_is_rejected_and_not_stored(self) -> None:
        agent = _agent()
        agent._save_review("")
        self.assertEqual(agent.dm.summaries, [])
        self.assertEqual(len(agent.dm.rejected), 1)
        reason, record_id = agent.dm.rejected[0]
        self.assertIn("Summary:", reason)
        self.assertEqual(record_id, "Y2020-W02-review#36")

    def test_the_control_group_still_records_rejections(self) -> None:
        agent = _agent(no_ce=True)
        agent._save_review("Summary:\n正常内容")
        self.assertEqual(agent.dm.summaries, [], "no-ce writes no diary")
        self.assertEqual(agent.dm.rejected, [])
        agent._save_review("")
        self.assertEqual(len(agent.dm.rejected), 1, "quality marking is independent")

    def test_review_disables_compact_reasoning(self) -> None:
        """C: the review must not carry a thinking-brief header at all."""
        import inspect

        source = inspect.getsource(RoleAgent.review)
        self.assertIn("keep_compact_reasoning=False", source)


class CacheHygieneTests(unittest.TestCase):
    """A: empty responses must not be stored, nor served as hits."""

    def setUp(self) -> None:
        self._ctx = temp_workspace()
        self.root = self._ctx.__enter__()  # type: ignore[attr-defined]

    def tearDown(self) -> None:
        self._ctx.__exit__(None, None, None)  # type: ignore[attr-defined]

    def _cached_fn(self, result: Any):
        from src import utils as utils_mod

        calls: List[int] = []

        @utils_mod.cached
        def probe(**kwargs):
            calls.append(1)
            return result

        return probe, calls

    def test_an_empty_result_is_not_cached(self) -> None:
        probe, calls = self._cached_fn([])
        probe(cache_file=str(Path("llm_cache/.probe.pkl")), marker="ki1")
        probe(cache_file=str(Path("llm_cache/.probe.pkl")), marker="ki1")
        self.assertEqual(len(calls), 2, "an empty answer must be asked again")

    def test_an_error_result_is_not_cached(self) -> None:
        probe, calls = self._cached_fn(_ERROR_RESPONSE)
        probe(cache_file=str(Path("llm_cache/.probe2.pkl")), marker="ki1")
        probe(cache_file=str(Path("llm_cache/.probe2.pkl")), marker="ki1")
        self.assertEqual(len(calls), 2)

    def test_a_real_answer_is_cached(self) -> None:
        probe, calls = self._cached_fn([{"role": "assistant", "content": "hello"}])
        first = probe(cache_file=str(Path("llm_cache/.probe3.pkl")), marker="ki1")
        second = probe(cache_file=str(Path("llm_cache/.probe3.pkl")), marker="ki1")
        self.assertEqual(first, second)
        self.assertEqual(len(calls), 1, "a non-empty answer is served from cache")


class GenerationLoopTests(unittest.TestCase):
    """End-to-end check of the loop's final-answer handling (mocked LLM)."""

    def setUp(self) -> None:
        self._ctx = temp_workspace()
        self.root = self._ctx.__enter__()  # type: ignore[attr-defined]

    def tearDown(self) -> None:
        self._ctx.__exit__(None, None, None)  # type: ignore[attr-defined]

    def _stub_agent(self) -> RoleAgent:
        agent = RoleAgent.__new__(RoleAgent)
        agent.name = "测试角色"
        agent.model = "test-model"
        agent.logger = _StubLogger()
        agent.config = {"response_validation": {"enabled": False}}
        agent.no_context_engineering = False
        agent.no_history = False
        agent.last_main_generation_id = None
        agent._opened_scratchpads = set()
        agent.dm = _StubDM()
        agent.dm.saved = []  # type: ignore[attr-defined]

        def save_generation(inputs, outputs, filename=None):
            agent.dm.saved.append(outputs)  # type: ignore[attr-defined]
            return "rec-1"

        agent.dm.save_generation = save_generation  # type: ignore[assignment]
        return agent

    def test_a_tool_result_is_replaced_by_a_real_answer(self) -> None:
        """The KI-1 shape: tool call in the last round, then an empty response."""
        agent = self._stub_agent()
        tool_call_round = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "1",
                        "type": "function",
                        "function": {"name": "not_allowed_tool", "arguments": "{}"},
                    }
                ],
            }
        ]
        empty_round: List[Dict[str, Any]] = []
        finalize_round = [{"role": "assistant", "content": "Summary: 真正的周记"}]

        with mock.patch(
            "src.agents.role_agent.generate_with_fc",
            side_effect=[tool_call_round, empty_round, finalize_round],
        ):
            outputs = agent._generate_with_functions(
                [{"role": "system", "content": "prompt"}],
                keep_compact_reasoning=False,
                save_to_week_response=False,
            )

        self.assertEqual(outputs[-1]["content"], "Summary: 真正的周记")
        # The saved generation record contains the answer that was used.
        self.assertEqual(agent.dm.saved[-1][-1]["content"], "Summary: 真正的周记")

    def test_a_normal_final_answer_is_used_without_an_extra_call(self) -> None:
        agent = self._stub_agent()
        with mock.patch(
            "src.agents.role_agent.generate_with_fc",
            return_value=[{"role": "assistant", "content": "Summary: 正常周记"}],
        ) as llm:
            outputs = agent._generate_with_functions(
                [{"role": "system", "content": "prompt"}],
                keep_compact_reasoning=False,
                save_to_week_response=False,
            )
        self.assertEqual(outputs[-1]["content"], "Summary: 正常周记")
        self.assertEqual(llm.call_count, 1, "no finalize call when the answer is fine")

    def test_no_usable_answer_falls_back_to_the_error_response(self) -> None:
        agent = self._stub_agent()
        with mock.patch(
            "src.agents.role_agent.generate_with_fc",
            return_value=[{"role": "tool", "content": "SUCCESS: scratchpad only"}],
        ):
            outputs = agent._generate_with_functions(
                [{"role": "system", "content": "prompt"}],
                keep_compact_reasoning=False,
                save_to_week_response=False,
            )
        self.assertEqual(outputs[-1]["content"], _ERROR_RESPONSE)



if __name__ == "__main__":
    unittest.main()
