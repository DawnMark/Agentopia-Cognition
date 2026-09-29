"""阶段 6 影子投影：proficiency 来自练习、effective capability 来自方法论。

用户决策（2026-09-27）：

- 形态：先做影子对照投影，行为不变、默认关闭（world.cognition.proficiency_projection）；
- 尺度锚点：按**练习次数**定义（同一技能有效练习 10 次 → proficiency = 0.5），
  而不是按 God 给的点数——点数取决于环境模型一次给多少分，换世界/换模型就要重标；
- 有效练习：所有正 skill delta 都算，不分档；"带真实方法证据的次数"作为独立一列保留；
- 合成：proficiency x 方法论因子的加权几何平均。

这一组测试钉住口径、饱和、折扣语义，以及"能从账本重建"。
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.agents.cognition.proficiency import (
    METHOD_EXPONENTS,
    DISCOUNT_FLOOR,
    PROFICIENCY_HALF_PRACTICES,
    MethodFactors,
    build_projection,
    capability_points,
    effective_capability,
    method_factors_by_skill,
    practice_by_skill,
    proficiency_from_practices,
    write_projection,
)
from src.agents.cognition.skills import SkillFamilies
from tests._helpers import make_datamanager, temp_workspace

NL = chr(10)


def _activity(week: int, gains: dict) -> dict:
    return {
        "time": f"Y2020-W{int(week):02d}-activity-D1",
        "type": "solo",
        "outcome": {"delta_skills": gains},
    }


def _dump(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + NL for row in rows),
        encoding="utf-8",
    )


class ProficiencyCurveTests(unittest.TestCase):
    """尺度锚点是练习次数（用户决策 2026-09-27），不是 God 给的点数。"""

    def test_practice_count_maps_to_a_saturating_proficiency(self) -> None:
        self.assertEqual(proficiency_from_practices(0), 0.0)
        self.assertAlmostEqual(
            proficiency_from_practices(PROFICIENCY_HALF_PRACTICES), 0.5, places=4
        )
        self.assertLess(proficiency_from_practices(10_000), 1.0)
        self.assertGreater(proficiency_from_practices(10_000), 0.99)

    def test_more_practice_always_ranks_higher(self) -> None:
        self.assertLess(proficiency_from_practices(5), proficiency_from_practices(50))

    def test_negative_counts_do_not_inflate(self) -> None:
        self.assertEqual(proficiency_from_practices(-100), 0.0)

    def test_the_scale_ignores_how_many_points_god_granted(self) -> None:
        light = practice_by_skill([_activity(1, {"写作": 1})])["写作"]
        heavy = practice_by_skill([_activity(1, {"写作": 3})])["写作"]
        self.assertEqual(light.practices, heavy.practices)
        self.assertEqual(light.proficiency, heavy.proficiency)
        self.assertLess(light.units, heavy.units)


class PracticeEvidenceTests(unittest.TestCase):
    def test_only_positive_grants_count_as_practice(self) -> None:
        records = [
            _activity(1, {"写作": 2}),
            _activity(2, {"写作": 0}),
            _activity(3, {"写作": -1}),
            _activity(4, {"写作": 3, "跑步": 1}),
        ]
        practice = practice_by_skill(records)
        self.assertEqual(practice["写作"].practices, 2)
        self.assertAlmostEqual(practice["写作"].units, 5.0)
        self.assertEqual(practice["写作"].first_week, "Y2020-W01")
        self.assertEqual(practice["写作"].last_week, "Y2020-W04")
        self.assertEqual(practice["跑步"].practices, 1)

    def test_skill_names_are_canonicalised_when_families_are_given(self) -> None:
        families = SkillFamilies()
        families.family_of["长跑耐力"] = "长跑"
        families.family_of["长跑"] = "长跑"
        practice = practice_by_skill(
            [_activity(1, {"长跑耐力": 2}), _activity(2, {"长跑": 3})], families=families
        )
        self.assertEqual(list(practice), ["长跑"])
        self.assertEqual(practice["长跑"].practices, 2)

    def test_the_unmapped_sentinel_is_not_a_skill(self) -> None:
        practice = practice_by_skill([_activity(1, {"unmapped": 3, "写作": 1})])
        self.assertNotIn("unmapped", practice)
        self.assertIn("写作", practice)
        factors = method_factors_by_skill(
            [
                {"type": "METHOD_PROPOSED", "method_id": "m9", "skill_id": "unmapped"},
                {"type": "METHOD_PROPOSED", "method_id": "m8", "skill_id": "写作"},
            ]
        )
        self.assertNotIn("unmapped", factors)
        self.assertIn("写作", factors)


class MethodFactorTests(unittest.TestCase):
    def _events(self) -> list:
        return [
            {"type": "METHOD_PROPOSED", "method_id": "m1", "skill_id": "写作"},
            {"type": "METHOD_PROPOSED", "method_id": "m2", "skill_id": "写作"},
            {"type": "METHOD_OUTCOME_OBSERVED", "method_id": "m1",
             "attribution": "real_adoption", "reward": 0.4},
            {"type": "METHOD_OUTCOME_OBSERVED", "method_id": "m1",
             "attribution": "real_adoption", "reward": -0.2},
            {"type": "METHOD_VALUE_UPDATED", "method_id": "m1",
             "evidence_kind": "real_adoption", "status": "validated"},
            {"type": "METHOD_OUTCOME_OBSERVED", "method_id": "m2",
             "attribution": "shadow_counterfactual", "reward": 0.9},
        ]

    def test_only_real_adoptions_count(self) -> None:
        factors = method_factors_by_skill(self._events())["写作"]
        self.assertEqual(factors.methods, 2)
        self.assertEqual(factors.practised_methods, 1)
        self.assertEqual(factors.total_practices, 2)
        self.assertEqual(factors.successful_practices, 1)
        self.assertEqual(factors.reliable_methods, 1)

    def test_the_three_factors_have_the_documented_meanings(self) -> None:
        factors = method_factors_by_skill(self._events())["写作"]
        self.assertAlmostEqual(factors.coverage, 0.5, places=4)
        self.assertAlmostEqual(factors.selection_accuracy, 0.5, places=4)
        self.assertAlmostEqual(factors.verified_reliability, 1.0, places=4)

    def test_a_skill_without_methods_is_not_penalised(self) -> None:
        empty = MethodFactors(skill_id="观星")
        self.assertEqual(empty.coverage, 1.0)
        self.assertEqual(empty.selection_accuracy, 1.0)
        self.assertEqual(empty.verified_reliability, 1.0)


class CapabilityCompositionTests(unittest.TestCase):
    def test_perfect_methodology_leaves_proficiency_intact(self) -> None:
        self.assertAlmostEqual(
            effective_capability(0.5, MethodFactors(skill_id="x")), 0.5, places=4
        )
        self.assertAlmostEqual(
            effective_capability(0.8, MethodFactors(skill_id="x")), 0.8, places=4
        )

    def test_poor_methodology_discounts_rather_than_collapses(self) -> None:
        """折扣下限 0.5（用户决策 2026-09-27）：方法论再差也只打到半折。

        改这条口径的直接原因：真实账本上"练 35 次、熟练度 78"的技能被三因子打到
        14/100，反而低于一个"只练 4 次、没有任何方法"的技能（29/100）——
        一个"练得多不如不练"的排序无法解释，也不能拿去影响行为。
        """
        half = MethodFactors(
            skill_id="写作", methods=4, practised_methods=2,
            total_practices=4, successful_practices=2, reliable_methods=1,
        )
        capability = effective_capability(0.8, half)
        product = 0.8 * half.coverage * half.selection_accuracy * half.verified_reliability
        self.assertGreater(capability, product)
        self.assertLess(capability, 0.8)
        # 该技能的因子是 0.5/0.5/0.5，加权几何平均仍是 0.5，正好压在下限上
        self.assertAlmostEqual(capability, 0.8 * DISCOUNT_FLOOR, places=3)

    def test_the_discount_can_never_push_capability_below_half_of_proficiency(self) -> None:
        """最极端的方法论（三因子全部触底）也只打到五折。"""
        worst = MethodFactors(
            skill_id="写作", methods=10, practised_methods=0,
            total_practices=5, successful_practices=0, reliable_methods=0,
        )
        self.assertAlmostEqual(
            effective_capability(0.6, worst), 0.6 * DISCOUNT_FLOOR, places=3
        )

    def test_no_practice_means_no_capability(self) -> None:
        self.assertEqual(effective_capability(0.0, MethodFactors(skill_id="写作")), 0.0)

    def test_the_exponents_form_a_mean_over_the_method_factors(self) -> None:
        self.assertAlmostEqual(sum(METHOD_EXPONENTS.values()), 1.0, places=6)

    def test_capability_never_exceeds_proficiency(self) -> None:
        for proficiency, factors in (
            (0.2, MethodFactors(skill_id="x")),
            (0.9, MethodFactors(skill_id="x", methods=1, practised_methods=1,
                                total_practices=3, successful_practices=3,
                                reliable_methods=1)),
        ):
            self.assertLessEqual(
                effective_capability(proficiency, factors), proficiency + 1e-6
            )


class ProjectionRebuildTests(unittest.TestCase):
    def setUp(self) -> None:
        self._ctx = temp_workspace()
        self.root = self._ctx.__enter__()  # type: ignore[attr-defined]
        self.persona = Path("data") / "regression_world" / "persona" / "测试角色"
        self.persona.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self._ctx.__exit__(None, None, None)  # type: ignore[attr-defined]

    def _dm(self):
        return SimpleNamespace(root=self.persona, char="测试角色")

    def _seed_basic(self) -> None:
        _dump(
            self.persona / "activity.jsonl",
            [
                _activity(1, {"写作": 2}),
                _activity(2, {"写作": 3}),
                {"time": "Y2020-W02-review", "type": "solo", "outcome": {}},
            ],
        )
        _dump(
            self.persona / "cognition" / "capability_events.jsonl",
            [
                {"time": "Y2020-W02-review", "type": "METHOD_PROPOSED",
                 "method_id": "m1", "skill_id": "写作", "ledger_event_id": "ev-1"},
                {"time": "Y2020-W03-activity-D1", "type": "METHOD_OUTCOME_OBSERVED",
                 "method_id": "m1", "attribution": "real_adoption", "reward": 0.3,
                 "ledger_event_id": "ev-2"},
            ],
        )

    def test_the_projection_is_rebuilt_from_the_two_ledgers(self) -> None:
        self._seed_basic()
        projection = build_projection(self._dm())
        entry = projection["skills"]["写作"]
        self.assertEqual(entry["practices"], 2)
        self.assertAlmostEqual(entry["practice_units"], 5.0)
        self.assertEqual(entry["factors"]["methods"], 1)
        self.assertEqual(entry["factors"]["practised_methods"], 1)
        self.assertGreater(entry["effective_capability"], 0.0)

    def test_an_empty_ledger_yields_an_empty_projection_not_an_error(self) -> None:
        projection = build_projection(self._dm())
        self.assertEqual(projection["skills"], {})
        self.assertIsNone(projection["stats"]["capability_mean"])

    def test_the_written_view_is_reproducible(self) -> None:
        self._seed_basic()
        dm = self._dm()
        first = write_projection(dm)
        write_projection(dm)
        self.assertEqual(
            (self.persona / "cognition" / "views" / "proficiency.json").read_text(
                encoding="utf-8"
            ),
            first.read_text(encoding="utf-8"),
        )

    def test_a_heavily_practised_skill_with_failing_methods_is_only_discounted(self) -> None:
        """折扣下限改写后的命题：方法一直失败会被打折，但**不会**把练习量抹掉。

        用户决策 2026-09-27 之前，这条测试断言的是设计正文的原命题——"练了 5 次但
        方法全失败"的能力低于"只练 2 次、没有方法"。加了 0.5 的折扣下限之后，
        两者不再可比（前者 0.19、后者 0.17），这正是那次决策的目的：
        **练得多的人不该因为"方法没走到 validated"而被只练过两次的人超过**。
        现在钉住的是新口径：失败的方法确实扣分（低于同练习量的满分情形），
        但扣到底也只有五折。
        """
        rows = [_activity(i, {"写作": 2}) for i in range(1, 6)]
        rows += [_activity(6, {"急救": 2}), _activity(7, {"急救": 2})]
        _dump(self.persona / "activity.jsonl", rows)

        events = [
            {"time": "Y2020-W02-review", "type": "METHOD_PROPOSED",
             "method_id": "m1", "skill_id": "写作", "ledger_event_id": "ev-1"},
        ]
        events += [
            {"time": f"Y2020-W0{i}-activity-D1", "type": "METHOD_OUTCOME_OBSERVED",
             "method_id": "m1", "attribution": "real_adoption", "reward": -0.4,
             "ledger_event_id": f"ev-o{i}"}
            for i in range(3, 6)
        ]
        _dump(self.persona / "cognition" / "capability_events.jsonl", events)

        projection = build_projection(self._dm())
        writing = projection["skills"]["写作"]
        aid = projection["skills"]["急救"]
        self.assertGreater(writing["practices"], aid["practices"])
        self.assertGreater(writing["proficiency"], aid["proficiency"])
        self.assertEqual(writing["factors"]["selection_accuracy"], 0.0)
        # 打折：低于"同样熟练度但方法全成功"的情形
        self.assertLess(writing["effective_capability"], writing["proficiency"])
        # 但不会被打到底：至少还有熟练度的一半（这里的因子积已经低于下限，
        # 所以正好落在下限上——0.3333 × 0.5 = 0.1666，与"只练 2 次、没有方法"的
        # 0.1667 打平，而不是被反超）
        self.assertAlmostEqual(
            writing["effective_capability"], writing["proficiency"] * DISCOUNT_FLOOR,
            places=3,
        )
        # 而"没有方法"的技能不打折，它照旧等于熟练度
        self.assertAlmostEqual(aid["effective_capability"], aid["proficiency"], places=4)


class CapabilityPointsTests(unittest.TestCase):
    """对外量纲 0-100（用户决策 2026-09-27）：内部 0-1，阈值刻度不变。"""

    def test_the_conversion_is_a_plain_percentage(self) -> None:
        self.assertEqual(capability_points(0.0), 0)
        self.assertEqual(capability_points(0.5), 50)
        self.assertEqual(capability_points(0.4321), 43)
        self.assertEqual(capability_points(1.0), 100)

    def test_values_outside_the_range_are_clamped(self) -> None:
        self.assertEqual(capability_points(-0.4), 0)
        self.assertEqual(capability_points(1.7), 100)

    def test_the_projection_carries_both_scales(self) -> None:
        """0-1 用于计算与审计，0-100 用于 God prompt 里已有的"点数"阈值。

        必须在临时工作目录里跑：`_dump` 写的是相对路径，直接在仓库根目录跑会往
        `data/` 里落一个真实的运行目录（有过一次）。
        """
        with temp_workspace() as root:
            persona = Path("data") / "regression_world" / "persona" / "测试角色"
            dm = SimpleNamespace(root=persona, char="测试角色")
            half = int(PROFICIENCY_HALF_PRACTICES)
            _dump(
                persona / "activity.jsonl",
                [_activity(i, {"写作": 2}) for i in range(1, half + 1)],
            )
            projection = build_projection(dm)
            entry = projection["skills"]["写作"]
            self.assertEqual(entry["practices"], half)
            self.assertAlmostEqual(entry["proficiency"], 0.5, places=4)
            self.assertEqual(entry["proficiency_points"], 50)
            self.assertEqual(
                entry["effective_capability_points"],
                capability_points(entry["effective_capability"]),
            )
            del root


class FamilyFoldingTests(unittest.TestCase):
    """归并只作用于投影统计，`state.skills` 的键一个都不动（用户决策）。"""

    def setUp(self) -> None:
        self._ctx = temp_workspace()
        self.root = self._ctx.__enter__()  # type: ignore[attr-defined]
        self.persona = Path("data") / "regression_world" / "persona" / "测试角色"
        self.world_dir = Path("data") / "regression_world"
        self.world_dir.mkdir(parents=True, exist_ok=True)
        (self.world_dir / "skill_aliases.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "skills": [
                        {"canonical": "长跑", "aliases": ["长跑耐力", "跑步"]},
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self._ctx.__exit__(None, None, None)  # type: ignore[attr-defined]

    def _dm(self):
        return SimpleNamespace(root=self.persona, char="测试角色", world="regression_world")

    def _seed(self) -> None:
        _dump(
            self.persona / "activity.jsonl",
            [
                _activity(1, {"长跑": 2}),
                _activity(2, {"长跑耐力": 2}),
                _activity(3, {"跑步": 2}),
                _activity(4, {"写作": 2}),
            ],
        )

    def test_three_spellings_become_one_family(self) -> None:
        self._seed()
        projection = build_projection(self._dm())
        self.assertIn("长跑", projection["skills"])
        self.assertNotIn("长跑耐力", projection["skills"])
        entry = projection["skills"]["长跑"]
        self.assertEqual(entry["practices"], 3)  # 三次练习合成一族
        self.assertGreater(entry["proficiency"], 0.0)

    def test_the_projection_reports_what_was_folded(self) -> None:
        self._seed()
        stats = build_projection(self._dm())["stats"]
        self.assertEqual(stats["raw_names_observed"], 4)  # 三个写法 + 写作
        self.assertEqual(stats["practised_skills"], 2)  # 折叠成两族
        self.assertEqual(stats["merged_family_count"], 1)
        self.assertEqual(
            {k: sorted(v) for k, v in stats["merged_families"].items()},
            {"长跑": sorted(["长跑", "长跑耐力", "跑步"])},
        )

    def test_folding_changes_the_ranking_not_just_the_labels(self) -> None:
        """归并的意义在这里：不归并时三族各自都很低，归并后它们合成一个高分族。"""
        self._seed()
        projection = build_projection(self._dm())
        folded = projection["skills"]["长跑"]["proficiency"]
        unfolded = proficiency_from_practices(1)  # 每个写法只练了一次
        self.assertGreater(folded, unfolded)

    def test_an_absent_alias_file_means_no_folding(self) -> None:
        (self.world_dir / "skill_aliases.json").unlink()
        self._seed()
        projection = build_projection(self._dm())
        self.assertEqual(
            sorted(projection["skills"]), sorted(["长跑", "长跑耐力", "跑步", "写作"])
        )
        self.assertEqual(projection["stats"]["merged_family_count"], 0)


def _profile() -> dict:
    """最小可用 profile：`get_profile_for_activity_eval` 需要的字段。"""
    return {
        "appearance_and_impression": "清瘦，戴眼镜",
        "brief_introduction": "在读大学生",
        "personality_traits": {"qualitative": "内向，慢热", "quantitative": {}},
        "talents": {"qualitative": "记忆突出", "quantitative": {"memory": 60}},
        "position": {"current": "学生"},
        "init_skills": {"写作": 10},
    }


class CapabilityInputTests(unittest.TestCase):
    """只读接口：上帝评估看得到派生能力，但默认关闭、且不写任何东西。"""

    def setUp(self) -> None:
        self._ctx = temp_workspace()
        self.root = self._ctx.__enter__()  # type: ignore[attr-defined]
        self.dm, self.clock = make_datamanager()
        self.dm.write_profile(_profile(), year=2020)
        self.dm.save_state(
            {
                "vitality": 70,
                "fulfillment": {},
                "skills": {"写作": 120, "急救": 40},
                "assets": {"deposit": 100, "possessions": []},
            }
        )
        persona = Path("data") / "regression_world" / "persona" / self.dm.char
        _dump(
            persona / "activity.jsonl",
            [_activity(1, {"写作": 2}), _activity(2, {"写作": 3})],
        )

    def tearDown(self) -> None:
        self._ctx.__exit__(None, None, None)  # type: ignore[attr-defined]

    def test_the_block_is_off_by_default(self) -> None:
        from src.config import get_config

        config = get_config()
        original = dict(config["world"].get("cognition") or {})
        try:
            config["world"].setdefault("cognition", {})["capability_input"] = False
            block = self.dm.capability_input_block({"写作": 120})
            self.assertEqual(block, "")
            text = self.dm.get_profile_for_activity_eval()
            self.assertNotIn("Practised capability", text)
        finally:
            config["world"]["cognition"].clear()
            config["world"]["cognition"].update(original)

    def test_the_block_appears_when_switched_on_and_writes_nothing(self) -> None:
        from src.config import get_config

        config = get_config()
        original = dict(config["world"].get("cognition") or {})
        try:
            config["world"].setdefault("cognition", {})["capability_input"] = True
            block = self.dm.capability_input_block({"写作": 120})
            self.assertIn("Practised capability", block)
            self.assertIn("写作", block)
            self.assertIn("practised 2x", block)
            text = self.dm.get_profile_for_activity_eval()
            self.assertIn("Practised capability", text)
            # 只读：技能数值没有被碰过
            state = self.dm.read_state(exclude_cur_t=False)
            self.assertEqual(state["skills"]["写作"], 120)
        finally:
            config["world"]["cognition"].clear()
            config["world"]["cognition"].update(original)

    def test_the_block_reports_the_points_scale_and_the_method_discount(self) -> None:
        """0-100 是对外量纲（用户决策）：God prompt 里的阈值本来就是点数刻度。"""
        from src.config import get_config

        config = get_config()
        original = dict(config["world"].get("cognition") or {})
        try:
            config["world"].setdefault("cognition", {})["capability_input"] = True
            block = self.dm.capability_input_block({"写作": 120})
            self.assertIn("/100", block)
            self.assertIn("proficiency", block)
            self.assertIn("methods x", block)
            # 没有方法的技能折扣恒为 1.00（"没有方法"不等于"方法失败"）
            self.assertIn("methods x1.00", block)
            # 迁移语义必须在块里说明白：能力高则增益小，没练过的可以自由增长
            self.assertIn("should be small", block)
            self.assertIn("not been practised at all", block)
        finally:
            config["world"]["cognition"].clear()
            config["world"]["cognition"].update(original)

    def test_at_most_eight_skills_are_shown_and_the_rest_are_counted(self) -> None:
        """prompt 预算：只列能力最高的 8 个，其余用一行说明，避免悄悄截断。"""
        from src.config import get_config

        persona = Path("data") / "regression_world" / "persona" / self.dm.char
        rows = []
        for index in range(11):
            rows.append(_activity(index + 1, {f"技能{index:02d}": 2}))
            rows.append(_activity(index + 12, {f"技能{index:02d}": 2}))
        _dump(persona / "activity.jsonl", rows)

        config = get_config()
        original = dict(config["world"].get("cognition") or {})
        try:
            config["world"].setdefault("cognition", {})["capability_input"] = True
            skills = {f"技能{index:02d}": 100 for index in range(11)}
            block = self.dm.capability_input_block(skills)
            self.assertEqual(block.count("practised 2x"), 8)
            self.assertIn("3 more practised skills omitted", block)
        finally:
            config["world"]["cognition"].clear()
            config["world"]["cognition"].update(original)

    def test_a_failure_is_reported_once_rather_than_swallowed(self) -> None:
        """只读块失败不能安静——"开关开着但什么都不显示"是最坏的结果。"""
        import contextlib
        import io
        from unittest import mock

        import src.agents.cognition.proficiency as prof
        import src.agents.data_manager as dm_mod
        from src.config import get_config

        config = get_config()
        original = dict(config["world"].get("cognition") or {})
        warned_before = dm_mod._CAPABILITY_INPUT_WARNED
        try:
            config["world"].setdefault("cognition", {})["capability_input"] = True
            dm_mod._CAPABILITY_INPUT_WARNED = False
            stderr = io.StringIO()
            with mock.patch.object(prof, "build_projection", side_effect=RuntimeError("boom")):
                with contextlib.redirect_stderr(stderr):
                    self.assertEqual(self.dm.capability_input_block({"写作": 120}), "")
                    self.assertEqual(self.dm.capability_input_block({"写作": 120}), "")
            self.assertIn("capability_input block unavailable", stderr.getvalue())
            self.assertEqual(stderr.getvalue().count("unavailable"), 1)  # 只喊一次
        finally:
            dm_mod._CAPABILITY_INPUT_WARNED = warned_before
            config["world"]["cognition"].clear()
            config["world"]["cognition"].update(original)

    def test_a_skill_that_was_never_practised_is_not_listed(self) -> None:
        from src.config import get_config

        config = get_config()
        original = dict(config["world"].get("cognition") or {})
        try:
            config["world"].setdefault("cognition", {})["capability_input"] = True
            block = self.dm.capability_input_block({"急救": 40})
            self.assertEqual(block, "")
        finally:
            config["world"]["cognition"].clear()
            config["world"]["cognition"].update(original)

if __name__ == "__main__":  # pragma: no cover
    unittest.main()


