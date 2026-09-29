"""`scripts/capability_input_probe.py` 的离线回归（不调用 LLM）。

反事实探针的全部意义在于"三档之间只差数字"，所以这里钉住三件事：
挑 prompt 的过滤条件、注入落点（必须在 `### Skills` 段之后、`## Current State` 之前，
与运行里的真实拼法一致），以及**高能力档只改数字、不改技能列表**。
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from capability_input_probe import (  # noqa: E402
    SKILLS_MARKER,
    high_variant,
    inject,
    parse_gains,
    pick_prompts,
    skills_in_block,
)

REAL_BLOCK = (
    "- Practised capability (derived from this character's own activity ledger, 0-100. ...):\n"
    "  - 厨艺: practised 4x, effective capability 29/100 (proficiency 29/100, methods x1.00)\n"
    "  - 长跑: practised 35x, effective capability 14/100 (proficiency 78/100, methods x0.19)\n"
    "  - Use this when deciding how much a character can still gain: ..."
)


def _prompt() -> str:
    return (
        "You are 甲, a 26-year-old woman.\n"
        "- Skills:\n"
        "### Skills\n"
        "    - 厨艺: 105\n"
        "\n"
        "## Current State\n"
        "\n"
        "### Vitality\n"
        " - Vitality: 71/100\n"
    )


class InjectionTests(unittest.TestCase):
    def test_the_block_lands_inside_the_skills_section(self) -> None:
        text = inject(_prompt(), REAL_BLOCK)
        self.assertIn(SKILLS_MARKER, text)
        self.assertLess(text.index("厨艺: 105"), text.index("Practised capability"))
        self.assertLess(text.index("Practised capability"), text.index("## Current State"))
        # 块之外的文字一个字符都不动（`\n` 是块自带的分隔，去掉块时一并去掉）
        self.assertEqual(text.replace(REAL_BLOCK + "\n", ""), _prompt())

    def test_an_empty_block_leaves_the_prompt_alone(self) -> None:
        self.assertEqual(inject(_prompt(), ""), _prompt())

    def test_a_prompt_without_the_skills_section_is_left_alone(self) -> None:
        text = "no skills here\n## Current State\n"
        self.assertEqual(inject(text, REAL_BLOCK), text)


class HighVariantTests(unittest.TestCase):
    def test_only_the_numbers_change(self) -> None:
        high = high_variant(REAL_BLOCK, points=55, practices=60)
        self.assertEqual(skills_in_block(high), skills_in_block(REAL_BLOCK))
        self.assertEqual(
            high.split("\n")[0], REAL_BLOCK.split("\n")[0]  # 抬头一字不动
        )
        self.assertEqual(high.count("practised 60x"), 2)
        self.assertEqual(high.count("effective capability 55/100"), 2)
        self.assertIn("(proficiency 70/100, methods x1.00)", high)
        # 真实块里那条 0.19 的折扣在高能力档里不该留下
        self.assertNotIn("x0.19", high)

    def test_the_skill_list_is_read_in_block_order(self) -> None:
        self.assertEqual(skills_in_block(REAL_BLOCK), ["厨艺", "长跑"])


class PromptPickingTests(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory(prefix="agentopia-probe-")
        self.run = Path(self._tmp.name) / "synthetic_world_00000000"
        self._write(
            "Y2021-W01",
            "You are 甲, a 26-year-old woman.\n### Skills\n  - 厨艺: 10\n\n## Current State\n",
            '{"delta_skills": {"厨艺": 2}}',
        )
        self._write(
            "Y2021-W02",
            "You are 甲, a 26-year-old woman.\n### Skills\n  - 厨艺: 20\n\n## Current State\n",
            '{"delta_skills": {}}',  # 没有增益 → 不该被挑中
        )
        self._write(
            "Y2021-W03",
            "You are 乙, a 30-year-old man.\n### Skills\n  - 长跑: 10\n\n## Current State\n",
            '{"delta_skills": {"长跑": 1}}',  # 别的角色 → 不该被挑中
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, time: str, prompt: str, response: str) -> None:
        path = self.run / "god" / "solo_activity" / "year=2021" / "week=1.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "time": time,
            "inputs": [{"role": "system", "content": prompt}],
            "outputs": [{"role": "assistant", "content": response}],
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def test_only_this_personas_prompts_with_gains_are_picked(self) -> None:
        picked = pick_prompts(self.run, "甲", 5)
        self.assertEqual([row["time"] for row in picked], ["Y2021-W01"])

    def test_the_latest_prompts_come_first(self) -> None:
        picked = pick_prompts(self.run, "甲", 5)
        self.assertEqual(picked[0]["time"], "Y2021-W01")  # 只有一条合格


class ParsingTests(unittest.TestCase):
    def test_deltas_are_read_out_of_the_json_answer(self) -> None:
        self.assertEqual(
            parse_gains('{"outcome": "ok", "delta_skills": {"厨艺": 2, "长跑": 0}}'),
            {"厨艺": 2.0, "长跑": 0.0},
        )

    def test_a_non_json_answer_yields_nothing_rather_than_raising(self) -> None:
        self.assertEqual(parse_gains("not json at all"), {})
        self.assertEqual(parse_gains(None), {})


if __name__ == "__main__":
    unittest.main()
