"""③ CONTACT 读写竞态与四文件非事务（离线，无 LLM 调用）。

背景：CONTACT 阶段全体 agent 在同一个进程里并发跑 `contact()`，
`read_message()` 会去读**别人正在写**的 `contact/*.jsonl` 与 `sig.jsonl`，
而 `_read_jsonl` 当时完全不加锁——读到写了一半的行会直接抛
`ValueError: failed to parse line` 并终止整轮模拟。
`send_message` 又要顺序写 4 个文件（双方会话 + 双方 sig），没有事务：
崩溃或并发交错会漏 peer、丢消息。

修复后的语义：
- 读侧与写侧共用同一把以文件为键的锁（读共享 / 写独占）；
- 双方会话日志是事实源，先写；`sig.jsonl` 是派生缓存，后写，可重建；
- 因此四文件不再需要"要么全成、要么全不成"，读者以会话日志为准。
"""

from __future__ import annotations

import json
import threading
import unittest
from pathlib import Path

from src.world.clock import Stage
from tests._helpers import make_datamanager, temp_workspace

PEER = "邻居角色"
WORLD = "regression_world"


def _contact_dir(char: str) -> Path:
    return Path("data") / WORLD / "persona" / char / "contact"


class ContactRaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._ctx = temp_workspace()
        self.root = self._ctx.__enter__()
        # read_message is only meaningful during CONTACT; slot 2 so that the
        # "previous slot" exists inside the same week.
        self.dm, self.clock = make_datamanager(stage=Stage.CONTACT)
        self.clock.set_slot(2)
        # One world clock, shared by every persona (as World does it).
        self.peer_dm, _ = make_datamanager(PEER, clock=self.clock)

    def tearDown(self) -> None:
        self._ctx.__exit__(None, None, None)

    # -- basic behaviour ---------------------------------------------------
    def test_send_message_writes_both_sides_and_signals(self) -> None:
        self.assertTrue(self.dm.send_message(PEER, "早"))
        # A message becomes visible to the reader once the clock moves on
        # (reads exclude the current instant by design).
        self.clock.set_slot(3)
        my_conv = _contact_dir(self.dm.char) / f"{PEER}.jsonl"
        peer_conv = _contact_dir(PEER) / f"{self.dm.char}.jsonl"
        self.assertTrue(my_conv.exists() and peer_conv.exists())
        self.assertTrue((_contact_dir(PEER) / "sig.jsonl").exists())
        self.assertTrue((_contact_dir(self.dm.char) / "sig.jsonl").exists())

        read = self.dm.read_message()
        self.assertIn("早", read)

    def test_read_message_without_any_contact_returns_placeholder(self) -> None:
        self.assertEqual(self.dm.read_message(), self.dm.NO_CONTACT_MSG)

    # -- crash consistency -------------------------------------------------
    def test_missing_signal_row_still_reveals_the_peer(self) -> None:
        """A crash between the conversation write and the signal write."""
        self.dm.send_message(PEER, "第一句")
        # Simulate the interrupted second half: drop the peer's signal rows.
        sig = _contact_dir(self.dm.char) / "sig.jsonl"
        sig.write_text("", encoding="utf-8")

        self.clock.set_slot(3)
        read = self.dm.read_message()
        self.assertIn(PEER, read)
        self.assertIn("第一句", read)

    def test_foreign_or_malformed_signal_rows_are_ignored(self) -> None:
        """Once this aborted the run through `assert to == self.char`."""
        self.dm.send_message(PEER, "正常消息")
        self.clock.set_slot(3)
        sig = _contact_dir(self.dm.char) / "sig.jsonl"
        with sig.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"time": str(self.clock.get_time()), "from": "路人甲", "to": "路人乙"},
                    ensure_ascii=False,
                )
                + "\n"
            )

        read = self.dm.read_message()
        self.assertIn(PEER, read)
        self.assertNotIn("路人甲", read)

    def test_rebuild_restores_signals_from_conversation_logs(self) -> None:
        from src.world.cleanup import _rebuild_one_sig

        self.dm.send_message(PEER, "一")
        self.peer_dm.send_message(self.dm.char, "二")

        sig = _contact_dir(self.dm.char) / "sig.jsonl"
        sig.unlink()

        rows = _rebuild_one_sig(_contact_dir(self.dm.char), self.dm.char)
        self.assertGreaterEqual(rows, 2)
        rebuilt = [
            json.loads(l) for l in sig.read_text(encoding="utf-8").splitlines() if l.strip()
        ]
        pairs = {(r["from"], r["to"]) for r in rebuilt}
        self.assertIn((self.dm.char, PEER), pairs)
        self.assertIn((PEER, self.dm.char), pairs)

        # The rebuilt file must still accept appends (_append_jsonl rejects an
        # append older than the last row).
        self.assertTrue(self.dm.send_message(PEER, "三"))

    # -- concurrency -------------------------------------------------------
    def test_concurrent_send_and_read_never_tears_a_line(self) -> None:
        """Writers append large records while readers read the same files."""
        big = "长" * 4000  # larger than the text buffer, to force real flushes
        errors: list = []
        stop = threading.Event()
        rounds = 40

        def writer(char_dm, target: str, tag: str) -> None:
            try:
                for i in range(rounds):
                    char_dm.send_message(target, f"{tag}-{i}-{big}")
            except Exception as e:  # pragma: no cover - failure path
                errors.append(e)
            finally:
                stop.set()

        def reader(char_dm) -> None:
            try:
                while not stop.is_set():
                    char_dm.read_message()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(self.dm, PEER, "A")),
            threading.Thread(target=writer, args=(self.peer_dm, self.dm.char, "B")),
            threading.Thread(target=reader, args=(self.dm,)),
            threading.Thread(target=reader, args=(self.peer_dm,)),
        ]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=60)
            self.assertFalse(th.is_alive(), "thread did not finish")

        self.assertEqual(errors, [], f"concurrent contact raised: {errors!r}")

        # Both sides must hold every message exactly once.
        for char, other in ((self.dm.char, PEER), (PEER, self.dm.char)):
            conv = _contact_dir(char) / f"{other}.jsonl"
            rows = [
                json.loads(l)
                for l in conv.read_text(encoding="utf-8").splitlines()
                if l.strip()
            ]
            self.assertEqual(len(rows), rounds * 2, f"{char} lost or duplicated rows")

    def test_concurrent_appends_are_all_present(self) -> None:
        """Two writers, same file, no lost records."""
        errors: list = []

        def writer(tag: str) -> None:
            try:
                for i in range(60):
                    self.dm.send_message(PEER, f"{tag}-{i}")
            except Exception as e:  # pragma: no cover - failure path
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,)) for t in ("A", "B")]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=60)

        self.assertEqual(errors, [])
        conv = _contact_dir(self.dm.char) / f"{PEER}.jsonl"
        rows = [
            json.loads(l) for l in conv.read_text(encoding="utf-8").splitlines() if l.strip()
        ]
        self.assertEqual(len(rows), 120)
        # The per-sender seq counter makes the order recoverable even though the
        # physical order depends on thread scheduling; allocation must stay
        # unique under concurrent writers.
        seqs = [r["seq"] for r in rows if r["from"] == self.dm.char]
        self.assertEqual(sorted(seqs), list(range(1, 121)))


if __name__ == "__main__":
    unittest.main()
