"""② generation reject 标记必须落在真正出错的那条记录上（离线，无 LLM 调用）。

背景：一次响应会往当周 generation jsonl 追两条记录——主记录，以及紧随其后的
compact reasoning 摘要（`keep_compact_reasoning` 默认开启，contact/plan/review
都没有关闭）。动作报错时旧代码改写文件**最后一行**，于是被标记成
`rejected=true` 的是摘要记录，真正产生错误动作的记录被保留下来，
`scripts/build_rft_data.py` 又恰好按 `rejected` 过滤训练样本。
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from tests._helpers import make_datamanager, temp_workspace


def _week_file(year: int = 2020, week: int = 1) -> Path:
    return Path("data/regression_world/persona/测试角色/generation") / (
        f"year={year}/week={week}.jsonl"
    )


def _read_records(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class GenerationRejectTests(unittest.TestCase):
    def setUp(self) -> None:
        self._ctx = temp_workspace()
        self.root = self._ctx.__enter__()
        self.dm, self.clock = make_datamanager()

    def tearDown(self) -> None:
        self._ctx.__exit__(None, None, None)

    def _save(self, tag: str) -> str:
        return self.dm.save_generation(
            inputs=[{"role": "user", "content": f"input {tag}"}],
            outputs=[{"role": "assistant", "content": f"output {tag}"}],
        )

    # -- ids ---------------------------------------------------------------
    def test_save_generation_returns_and_embeds_record_id(self) -> None:
        record_id = self._save("main")
        self.assertTrue(record_id)

        records = _read_records(_week_file())
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["record_id"], record_id)
        self.assertNotIn("rejected", records[0])

    def test_ids_do_not_collide_after_resume(self) -> None:
        first = self._save("a")
        second = self._save("b")

        # A resumed run builds a fresh DataManager over the existing file.
        resumed, _ = make_datamanager()
        third = resumed.save_generation(
            inputs=[{"role": "user", "content": "input c"}],
            outputs=[{"role": "assistant", "content": "output c"}],
        )

        ids = {first, second, third}
        self.assertEqual(len(ids), 3)
        self.assertEqual(
            sorted(r["record_id"] for r in _read_records(_week_file())),
            sorted(ids),
        )

    # -- targeting ---------------------------------------------------------
    def test_reject_targets_the_action_record_not_the_compact_record(self) -> None:
        main_id = self._save("main")
        self._save("compact reasoning summary")

        self.dm.mark_generation_rejected("unparsable role_action block", main_id)

        records = _read_records(_week_file())
        by_tag = {r["outputs"][0]["content"]: r for r in records}

        main = by_tag["output main"]
        compact = by_tag["output compact reasoning summary"]
        self.assertTrue(main.get("rejected"))
        self.assertEqual(main["reject_reason"], "unparsable role_action block")
        self.assertNotIn("rejected", compact)

    def test_reject_can_target_a_non_last_record(self) -> None:
        first_id = self._save("first")
        self._save("second")
        self._save("third")

        self.dm.mark_generation_rejected("bad first", first_id)

        records = _read_records(_week_file())
        self.assertEqual(len(records), 3)
        flags = [bool(r.get("rejected")) for r in records]
        self.assertEqual(flags, [True, False, False])
        # ids and payloads of the untouched records survive the rewrite
        self.assertEqual(records[1]["outputs"][0]["content"], "output second")
        self.assertEqual(records[2]["outputs"][0]["content"], "output third")

    def test_reject_is_idempotent_per_record(self) -> None:
        target = self._save("target")
        self.dm.mark_generation_rejected("first reason", target)
        self.dm.mark_generation_rejected("second reason", target)
        records = _read_records(_week_file())
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["reject_reason"], "second reason")

    def test_unknown_or_missing_id_marks_nothing(self) -> None:
        self._save("only")

        self.dm.mark_generation_rejected("unknown id", "Y2020-W01-begin#999")
        self.dm.mark_generation_rejected("no id at all", None)

        records = _read_records(_week_file())
        self.assertEqual(len(records), 1)
        self.assertNotIn("rejected", records[0])

    def test_reject_without_a_generation_file_is_a_noop(self) -> None:
        self.dm.mark_generation_rejected("nothing saved yet", "Y2020-W01-begin#1")
        self.assertFalse(_week_file().exists())

    def test_reject_ignores_a_corrupt_line(self) -> None:
        target = self._save("good")
        path = _week_file()
        with path.open("a", encoding="utf-8") as f:
            f.write("{not json}\n")

        self.dm.mark_generation_rejected("still works", target)
        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.assertEqual(len(lines), 2)
        self.assertTrue(json.loads(lines[0]).get("rejected"))
        self.assertEqual(lines[1], "{not json}")

    # -- wiring ------------------------------------------------------------
    def test_agent_rejects_the_generation_it_acted_on(self) -> None:
        """RoleAgent must pass its own record id, not rely on file order."""
        from src.agents.role_agent import RoleAgent

        agent = RoleAgent.__new__(RoleAgent)  # no __init__: no LLM, no profile
        agent.dm = self.dm
        agent.last_main_generation_id = self._save("acted on")
        self._save("compact summary")

        agent.mark_last_generation_rejected("verification failed")

        records = _read_records(_week_file())
        by_content = {r["outputs"][0]["content"]: r for r in records}
        self.assertTrue(by_content["output acted on"].get("rejected"))
        self.assertNotIn("rejected", by_content["output compact summary"])


if __name__ == "__main__":
    unittest.main()
