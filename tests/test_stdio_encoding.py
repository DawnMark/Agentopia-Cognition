"""控制台编码必须容错：模型文本不能中断整轮模拟（离线，无 LLM 调用）。

背景：`utils.py` 会把模型生成的内容（cache-miss key、prompt 片段）用 `print`
写到 stdout。Windows 控制台/重定向默认使用本地代码页（中文 Windows 为 GBK），
只要出现一个 GBK 之外的字符（例如模型回复里的 ✓/✗），就会抛
`UnicodeEncodeError`；异常发生在 worker 线程里，会把整轮模拟带走。
真跑 B（`--no-ce`）首次运行就是这样崩的。
"""

from __future__ import annotations

import io
import sys
import unittest

from src.utils import configure_stdio_encoding

# A character that is not representable in GBK.
NON_GBK = "\u2717"  # ✗


def _gbk_stream() -> io.TextIOWrapper:
    return io.TextIOWrapper(io.BytesIO(), encoding="gbk", errors="strict")


class StdioEncodingTests(unittest.TestCase):
    def test_gbk_stream_raises_before_the_fix(self) -> None:
        """Documents the failure mode the helper removes."""
        stream = _gbk_stream()
        with self.assertRaises(UnicodeEncodeError):
            print(NON_GBK, file=stream)

    def test_configure_stdio_encoding_survives_non_gbk_text(self) -> None:
        stream = _gbk_stream()
        configure_stdio_encoding(streams=[stream])
        print(NON_GBK, file=stream)  # must not raise
        stream.flush()
        self.assertEqual(stream.encoding.lower().replace("-", ""), "utf8")

    def test_configure_stdio_encoding_accepts_default_streams(self) -> None:
        """Must not raise for the real stdout/stderr (or for detached ones)."""
        configure_stdio_encoding()

    def test_streams_without_reconfigure_are_skipped(self) -> None:
        class _NoReconfigure:
            def write(self, _data):  # pragma: no cover - not used
                return len(_data)

        configure_stdio_encoding(streams=[_NoReconfigure(), None])

    def test_run_world_entry_point_configures_stdio(self) -> None:
        """The entry point must call it before anything else can print."""
        import inspect

        import scripts.run_world as run_world

        source = inspect.getsource(run_world.main)
        self.assertIn("configure_stdio_encoding()", source.split("parser =")[0])


if __name__ == "__main__":
    unittest.main()
