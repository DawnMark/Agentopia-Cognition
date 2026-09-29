"""① scratchpad 名称解析：路径穿越必须被拒绝（离线，无 LLM 调用）。

背景：`read_scratchpad` / `update_scratchpad` 的 `s_name` 直接来自模型工具调用。
修复前只检查 `characters/` / `others/` 前缀，随后直接拼接路径，因此
`others/../../../../config` 可以写出 scratchpad 根目录之外，而
`characters/../../contact/<peer>` 可以读到其他角色的私聊记录。
"""

from __future__ import annotations

import unittest
from pathlib import Path

from tests._helpers import make_datamanager, persona_root, temp_workspace, TEST_CHAR, TEST_WORLD

# Names that must never resolve to a path.
TRAVERSAL_NAMES = [
    "others/../../../etc/passwd",
    "others/../../../../../../config.json",
    "others/../../contact/别人",
    "characters/../../contact/别人",
    "others/../../../../../../logs/error",
    "others/..",
    "others/.",
    "others/a/b",
    "others/./a",
    "others/../a",
    "others/.hidden",
    "others/CON",
    "others/nul.txt",
    "others/com1",
    "others/a:b",
    "others/a\\b",
    "others/名字.",
    "others/\x00x",
    "others/" + "x" * 200,
    "characters/",
    "others/",
    "",
    "   ",
    "general/../others/x",
    "working_memory/../general",
    "/others/x",
    "C:/others/x",
    "etc/passwd",
]


class ScratchpadPathTests(unittest.TestCase):
    def setUp(self) -> None:
        self._ctx = temp_workspace()
        self.root = self._ctx.__enter__()
        self.dm, self.clock = make_datamanager()

    def tearDown(self) -> None:
        self._ctx.__exit__(None, None, None)

    # -- rejection ---------------------------------------------------------
    def test_traversal_names_are_rejected_on_read(self) -> None:
        for name in TRAVERSAL_NAMES:
            with self.subTest(name=name):
                out = self.dm.read_scratchpad(name)
                self.assertTrue(out.startswith("ERROR:"), out)

    def test_traversal_names_are_rejected_on_create(self) -> None:
        for name in TRAVERSAL_NAMES:
            with self.subTest(name=name):
                out = self.dm.update_scratchpad(
                    name, "pwned", create_new_scratchpad=True
                )
                self.assertTrue(out.startswith("ERROR:"), out)

    def test_traversal_names_are_rejected_on_update(self) -> None:
        for name in TRAVERSAL_NAMES:
            with self.subTest(name=name):
                out = self.dm.update_scratchpad(name, "pwned")
                self.assertTrue(out.startswith("ERROR:"), out)

    def test_nothing_is_written_outside_the_scratchpad_root(self) -> None:
        # A file outside the scratchpad tree that the old code could overwrite.
        stray = Path("data") / TEST_WORLD / "secret.jsonl"
        stray.parent.mkdir(parents=True, exist_ok=True)
        stray.write_text('{"time": "Y2020-W01-begin", "content": "original"}\n', encoding="utf-8")

        def jsonl_files() -> set:
            return {p.resolve() for p in Path(".").rglob("*.jsonl")}

        before = jsonl_files()
        for name in TRAVERSAL_NAMES:
            self.dm.update_scratchpad(name, "pwned", create_new_scratchpad=True)
            self.dm.update_scratchpad(name, "pwned")

        # The stray file is untouched, and no new file appeared outside the
        # scratchpad root.
        self.assertNotIn("pwned", stray.read_text(encoding="utf-8"))

        scratch = (persona_root() / "memory" / "scratchpad").resolve()
        outside = [
            p
            for p in jsonl_files() - before
            if p != scratch and scratch not in p.parents
        ]
        self.assertEqual(outside, [], f"written outside scratchpad root: {outside}")

    # -- legal usage keeps working ----------------------------------------
    def test_legal_names_resolve(self) -> None:
        legal = [
            "general",
            "general.txt",
            "general.jsonl",
            "working_memory",
            "working_memory.txt",
            "others/九亭地形",
            "others/九亭地形.jsonl",
            "others/九亭地形.txt",
            "characters/吉日和",
            "characters/吉日和.jsonl",
        ]
        scratch = persona_root() / "memory" / "scratchpad"
        for name in legal:
            with self.subTest(name=name):
                path, canonical, err = self.dm._resolve_scratchpad_name(name)
                self.assertEqual(err, "", f"{name} -> {err}")
                self.assertIsNotNone(path)
                self.assertTrue(str(path.resolve()).startswith(str(scratch.resolve())))
                self.assertFalse(canonical.endswith(".jsonl"))

    def test_create_read_update_roundtrip(self) -> None:
        out = self.dm.update_scratchpad(
            "others/九亭地形", "<summary>地形笔记</summary><full>九亭地形笔记正文</full>",
            create_new_scratchpad=True,
        )
        self.assertTrue(out.startswith("SUCCESS"), out)
        created = persona_root() / "memory" / "scratchpad" / "others" / "九亭地形.jsonl"
        self.assertTrue(created.exists())

        # Reading requires the pad to have been created before "now".
        self.clock.set_week(2)
        read = self.dm.read_scratchpad("others/九亭地形")
        self.assertIn("九亭地形笔记正文", read)

        updated = self.dm.update_scratchpad("others/九亭地形.txt", "第二版正文")
        self.assertTrue(updated.startswith("SUCCESS"), updated)
        # Append-only reads are exclusive of the current instant, so advance
        # the clock before reading the version just written.
        self.clock.set_week(3)
        self.assertIn("第二版正文", self.dm.read_scratchpad("others/九亭地形"))

    def test_character_pad_creation_still_requires_permission(self) -> None:
        denied = self.dm.update_scratchpad(
            "characters/吉日和", "印象", create_new_scratchpad=True
        )
        self.assertTrue(denied.startswith("ERROR"), denied)
        self.assertIn("not allowed", denied)

        allowed = self.dm.update_scratchpad(
            "characters/吉日和",
            "印象",
            create_new_scratchpad=True,
            allow_characters_create=True,
        )
        self.assertTrue(allowed.startswith("SUCCESS"), allowed)

    def test_dotted_leaf_keeps_its_name(self) -> None:
        """`others/x.v2` used to collapse to `x.jsonl` via with_suffix()."""
        out = self.dm.update_scratchpad(
            "others/跑单与装备.v2", "正文", create_new_scratchpad=True
        )
        self.assertTrue(out.startswith("SUCCESS"), out)
        others = persona_root() / "memory" / "scratchpad" / "others"
        self.assertTrue((others / "跑单与装备.v2.jsonl").exists())
        self.assertFalse((others / "跑单与装备.jsonl").exists())

    def test_pad_id_stays_relative(self) -> None:
        self.dm.update_scratchpad("others/维修", "正文", create_new_scratchpad=True)
        pad = persona_root() / "memory" / "scratchpad" / "others" / "维修.jsonl"
        self.assertEqual(self.dm._pad_id_of(pad), "others/维修")

    def test_surrounding_whitespace_is_normalised(self) -> None:
        path_a, name_a, err_a = self.dm._resolve_scratchpad_name("  others/维修  ")
        path_b, _, err_b = self.dm._resolve_scratchpad_name("others/维修")
        self.assertEqual((err_a, err_b), ("", ""))
        self.assertEqual(path_a, path_b)
        self.assertEqual(name_a, "others/维修")


if __name__ == "__main__":
    unittest.main()
