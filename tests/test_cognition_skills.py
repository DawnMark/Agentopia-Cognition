"""阶段 5 欠账：技能名"能力族"旁路映射（不改数据，只做测量与报告）。

设计正文担心的是"未知 skill 名称可以直接创建，容易产生同义词碎片和数值通胀"。
实测（两次验收运行）把这件事分成了两半：

- **角色内部几乎没有碎片**：每个方法的 `skill_id` 都是该角色已有的技能（35 个方法里只有 1 个
  `unmapped`），抽取 prompt 里"用已有技能名"这条被遵守了；
- **角色之间碎片是真的**，而且来自环境模型：同一个能力在不同角色那里叫 `长跑` /
  `长跑耐力` / `长跑与体能`，`厨艺` / `烹饪`，`电动车维修` / `电动车保养`。

那些名字写在 `state.jsonl` 里，职位、工资、奖励计算都依赖它们，改名是阶段 6 的迁移，
所以这一版只做**旁路映射**：归族用于测量与报告，绝不改写技能、事件或方法。
"""

from __future__ import annotations

import io
import json
import unittest
from pathlib import Path

from src.agents.cognition.skills import (
    SkillFamilies,
    normalize_skill,
    similar_names,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
# `skill_aliases.json` is a world-side file (`data/<world>/skill_aliases.json`). The world
# it was authored against is not part of this repository, so the tests read the shipped
# copy from `tests/fixtures/` instead. Same file, same schema, same expectations.
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "skill_aliases.json"


class SkillFamilyTests(unittest.TestCase):
    def _families(self) -> SkillFamilies:
        return SkillFamilies.from_world_dir(FIXTURE.parent)

    def test_the_world_map_groups_the_observed_synonyms(self) -> None:
        families = self._families()
        self.assertTrue(families.members)
        self.assertEqual(families.canonical("长跑耐力"), "长跑")
        self.assertEqual(families.canonical("长跑与体能"), "长跑")
        self.assertEqual(families.canonical("烹饪"), "厨艺")
        self.assertEqual(families.canonical("客户沟通"), "人际沟通")

    def test_an_unknown_name_is_its_own_family(self) -> None:
        families = self._families()
        self.assertEqual(families.canonical("观星"), "观星")
        self.assertEqual(families.canonical(""), "")

    def test_only_real_fragmentation_is_reported(self) -> None:
        families = self._families()
        groups = families.fragments(
            ["长跑", "长跑耐力", "长跑与体能", "厨艺", "烹饪", "观星", "外语"]
        )
        self.assertEqual(sorted(groups), sorted(["长跑", "厨艺"]))
        self.assertEqual(sorted(groups["长跑"]), sorted(["长跑", "长跑与体能", "长跑耐力"]))
        # 只有一个写法的能力族不是碎片，不该出现在报告里
        self.assertNotIn("观星", groups)

    def test_normalization_ignores_punctuation_and_case(self) -> None:
        families = self._families()
        self.assertEqual(families.canonical(" 长跑 "), "长跑")
        self.assertEqual(normalize_skill("长 跑"), "长跑")
        self.assertEqual(normalize_skill("Running"), "running")

    def test_an_absent_map_is_not_an_error(self) -> None:
        families = SkillFamilies.from_world_dir(REPO_ROOT / "data" / "no_such_world")
        self.assertEqual(families.members, {})
        self.assertEqual(families.canonical("长跑"), "长跑")
        self.assertEqual(families.fragments(["长跑", "长跑耐力"]), {})

    def test_a_malformed_map_falls_back_to_no_mapping(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "skill_aliases.json"
            path.write_text("{not json", encoding="utf-8")
            families = SkillFamilies.from_world_dir(Path(tmp))
            self.assertEqual(families.members, {})

    def test_suggestions_point_at_the_closest_names(self) -> None:
        candidates = ["观星", "天象观测", "外语", "太极拳"]
        self.assertEqual(similar_names("观星术", candidates), ["观星"])
        self.assertEqual(similar_names("太长", candidates), [])
        self.assertEqual(similar_names("太极", ["太极", "太极拳"]), ["太极", "太极拳"])

    def test_the_shipped_map_is_loadable_and_documented(self) -> None:
        payload = json.loads(io.open(FIXTURE, encoding="utf-8").read())
        self.assertEqual(payload["version"], 1)
        self.assertIn("旁路映射", payload["comment"])
        for entry in payload["skills"]:
            self.assertTrue(entry["canonical"])
            self.assertIsInstance(entry["aliases"], list)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
