"""`scripts/compare_capability_input.py` 的离线回归（不调用任何 LLM）。

这个脚本是阶段 6"能力进评估输入"那一步的取证工具：它读两次运行的 `activity.jsonl`，
比较技能增益的分布。读数一旦算错，结论就会反着写（"模型没反应" / "模型少发了"），
所以这里用合成运行目录把三个读数钉住：分布、零增益、按前半程练习量分层。
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from compare_capability_input import compare, profile_run  # noqa: E402


def _write_run(root: Path, name: str, persona: str, activities: list[dict]) -> Path:
    run = root / name
    persona_dir = run / "persona" / persona
    persona_dir.mkdir(parents=True, exist_ok=True)
    text = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in activities
    )
    (persona_dir / "activity.jsonl").write_text(text, encoding="utf-8")
    return run


def _activity(week: int, gains: dict) -> dict:
    return {
        "time": f"Y2020-W{week:02d}-activity-D1",
        "type": "solo",
        "outcome": {"delta_skills": gains},
    }


class ComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory(prefix="agentopia-ab-")
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_the_grant_distribution_is_compared(self) -> None:
        control = _write_run(
            self.root,
            "control",
            "甲",
            [_activity(1, {"写作": 3}), _activity(2, {"写作": 3})],
        )
        treatment = _write_run(
            self.root,
            "treatment",
            "甲",
            [_activity(1, {"写作": 1}), _activity(2, {"写作": 1})],
        )
        result = compare(profile_run(control), profile_run(treatment))
        row = result["personas"][0]
        self.assertEqual(row["grant_mean"], -2.0)
        self.assertEqual(row["grant_sum"], -4.0)
        self.assertEqual(row["grants_per_activity"], [3.0, 1.0])

    def test_a_run_that_stops_granting_is_visible(self) -> None:
        """最典型的失败形态：prompt 被读成"什么都别给"，增益整体消失。"""
        control = _write_run(
            self.root, "control", "甲", [_activity(1, {"写作": 2})]
        )
        treatment = _write_run(
            self.root,
            "treatment",
            "甲",
            [
                _activity(1, {"写作": 0}),
                {"time": "Y2020-W02-activity-D1", "type": "solo", "outcome": {}},
            ],
        )
        profile = profile_run(treatment)["personas"]["甲"]
        self.assertEqual(profile["zero_share"], 1.0)
        self.assertEqual(profile["empty_share"], 0.5)
        result = compare(profile_run(control), profile_run(treatment))
        self.assertEqual(result["personas"][0]["zero_share"], 1.0)

    def test_grants_are_stratified_by_first_half_practice(self) -> None:
        """分层是唯一能区分"读懂了块"和"只是少发了几个点"的读数。

        分层规则：按周序号取中位数分前后半程，前半程练过 **5 次以上** 的技能归
        `practised_5_plus`。所以这里前半程（W01–W05）让写作练满 5 次，后半程再看它。
        """
        run = _write_run(
            self.root,
            "run",
            "甲",
            [
                # 前半程 W01-W05：写作每周 3 分 → 5 次练习
                _activity(1, {"写作": 3, "急救": 2}),
                _activity(2, {"写作": 3}),
                _activity(3, {"写作": 3}),
                _activity(4, {"写作": 3}),
                _activity(5, {"写作": 3}),
                # 后半程 W06-W10：写作只给 1 分；新技能（前半程没练过）给 3 分
                _activity(6, {"写作": 1, "新技能": 3}),
                _activity(7, {"写作": 1}),
                _activity(8, {"写作": 1}),
                _activity(9, {"写作": 1}),
                _activity(10, {"写作": 1}),
            ],
        )
        profile = profile_run(run)["personas"]["甲"]
        self.assertEqual(profile["midpoint_week"], 20206)
        strata = profile["second_half_by_first_half_practice"]
        self.assertEqual(strata["practised_5_plus"]["mean"], 1.0)
        self.assertEqual(strata["practised_0"]["mean"], 3.0)
        self.assertEqual(strata["practised_1_4"]["samples"], 0)


if __name__ == "__main__":
    unittest.main()
