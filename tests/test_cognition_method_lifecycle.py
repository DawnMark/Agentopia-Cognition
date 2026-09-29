"""阶段 5：方法生命周期（specialized / archived / 版本演化）的离线测试。

`models.METHOD_STATUSES` 里 `specialized` 与 `archived` 从阶段 1 就声明了，但一直没有
任何写入者，所以"长期低 value 方法怎么办"（设计正文开放问题 9）和"同义方法重复率"
都无法回答。这一组测试锁定三条规则：

- **specialized 是相对自己**：某一情境明显高于该方法自己的整体 value 才算，且要有样本下限；
- **归档是降低可访问性，不是删除**：只归档"持续失败且长期没人再用"的方法，视图里保留；
- **版本演化**：后一周换了个说法重新得出的方法，是同一个方法的新版本，不是新方法。
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, List

from src.agents.cognition.method_lifecycle import (
    LifecycleConfig,
    MethodLifecycle,
    archive_reason,
    plan_refinement,
    refinement_target,
    specialization_of,
    title_similarity,
    week_number,
    weeks_between,
)
from src.agents.cognition.models import Methodology
from src.world.clock import Stage
from tests._helpers import make_datamanager, temp_workspace


def _method(
    *,
    method_id: str = "method-a",
    title: str = "先定冲突再写场景",
    status: str = "tested",
    value: float = 0.1,
    practice: int = 3,
    success: int = 1,
    version: int = 1,
    last_used: str = "",
    contexts: Dict[str, Dict[str, float]] | None = None,
) -> Methodology:
    return Methodology(
        method_id=method_id,
        skill_id="写作",
        title=title,
        status=status,
        global_value=value,
        confidence=0.5,
        practice_count=practice,
        success_count=success,
        version=version,
        last_used=last_used,
        steps=["明确目标", "列出冲突"],
        context_values=dict(contexts or {}),
    )


class WeekArithmeticTests(unittest.TestCase):
    def test_week_numbers_are_ordered(self) -> None:
        self.assertLess(week_number("Y2020-W09"), week_number("Y2020-W10"))
        self.assertLess(week_number("Y2020-W10"), week_number("Y2021-W01"))
        self.assertIsNone(week_number("nonsense"))

    def test_weeks_between_handles_the_year_boundary(self) -> None:
        self.assertEqual(weeks_between("Y2020-W08", "Y2020-W10"), 2)
        self.assertEqual(weeks_between("Y2020-W10", "Y2021-W01"), 1)
        self.assertIsNone(weeks_between("", "Y2020-W10"))


class SpecializationTests(unittest.TestCase):
    def test_a_clear_context_advantage_specialises_the_method(self) -> None:
        method = _method(
            value=0.1,
            contexts={"solo+time_pressure": {"value": 0.8, "count": 4}},
        )
        found = specialization_of(method)
        self.assertIsNotNone(found)
        self.assertEqual(found.context_key, "solo+time_pressure")
        self.assertAlmostEqual(found.margin, 0.7, places=4)

    def test_thin_evidence_does_not_specialise(self) -> None:
        method = _method(value=0.1, contexts={"solo": {"value": 0.9, "count": 1}})
        self.assertIsNone(specialization_of(method))

    def test_a_small_margin_is_not_specialisation(self) -> None:
        method = _method(value=0.4, contexts={"solo": {"value": 0.5, "count": 6}})
        self.assertIsNone(specialization_of(method))

    def test_a_method_that_is_good_everywhere_is_not_specialised(self) -> None:
        method = _method(
            value=0.7,
            contexts={
                "solo": {"value": 0.75, "count": 5},
                "work": {"value": 0.72, "count": 5},
            },
        )
        self.assertIsNone(specialization_of(method))

    def test_the_strongest_qualifying_context_wins(self) -> None:
        method = _method(
            value=0.0,
            contexts={
                "solo": {"value": 0.4, "count": 4},
                "work": {"value": 0.9, "count": 4},
            },
        )
        self.assertEqual(specialization_of(method).context_key, "work")


class ArchivingTests(unittest.TestCase):
    def test_a_failing_method_left_unused_is_archived(self) -> None:
        method = _method(status="deprecated", value=-0.4, practice=4, last_used="Y2020-W01-activity-D2")
        reason = archive_reason(method, current_week="Y2020-W08", after_weeks=6)
        self.assertIsNotNone(reason)
        self.assertIn("unused", reason)

    def test_a_failing_method_that_was_just_used_stays(self) -> None:
        method = _method(status="deprecated", value=-0.4, practice=4, last_used="Y2020-W07-activity-D1")
        self.assertIsNone(archive_reason(method, current_week="Y2020-W08", after_weeks=6))

    def test_a_new_method_is_never_archived(self) -> None:
        """没有实践证据的方法还没轮到它，不能因为"没人用"就被清掉。"""
        method = _method(status="proposed", value=0.0, practice=0, last_used="")
        self.assertIsNone(archive_reason(method, current_week="Y2030-W01", after_weeks=1))

    def test_a_healthy_method_is_never_archived(self) -> None:
        method = _method(status="validated", value=0.6, practice=8, last_used="Y2020-W01-activity-D1")
        self.assertIsNone(archive_reason(method, current_week="Y2020-W09", after_weeks=6))

    def test_an_already_archived_method_is_left_alone(self) -> None:
        method = _method(status="archived", value=-0.5, practice=5, last_used="Y2020-W01-activity-D1")
        self.assertIsNone(archive_reason(method, current_week="Y2020-W09"))

    def test_the_same_numbers_that_deprecate_also_archive(self) -> None:
        """状态可能是旧事件写下的旧值；归档判定要能独立重算。"""
        method = _method(status="tested", value=-0.4, practice=4, last_used="Y2020-W01-activity-D1")
        self.assertIsNotNone(archive_reason(method, current_week="Y2020-W09", after_weeks=6))


class VersionEvolutionTests(unittest.TestCase):
    def test_similar_titles_are_the_same_method(self) -> None:
        self.assertGreater(
            title_similarity("先定冲突再写场景", "先定冲突，再排场景顺序"),
            0.5,
        )
        self.assertLess(title_similarity("先定冲突再写场景", "每天晨跑五公里"), 0.3)

    def test_a_reworded_method_becomes_a_new_version(self) -> None:
        existing = [_method(method_id="method-x", title="先定冲突再写场景", version=2)]
        plan = plan_refinement({"title": "先定冲突再排场景"}, existing)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.method_id, "method-x")
        self.assertEqual(plan.version, 3)
        self.assertEqual(plan.parent_method_id, "method-x")

    def test_an_unrelated_method_is_not_a_refinement(self) -> None:
        existing = [_method(method_id="method-x", title="先定冲突再写场景")]
        self.assertIsNone(plan_refinement({"title": "每天晨跑五公里"}, existing))

    def test_the_closest_existing_method_wins(self) -> None:
        existing = [
            _method(method_id="method-a", title="先定冲突再写场景"),
            _method(method_id="method-b", title="先定冲突再排场景顺序"),
        ]
        target = refinement_target("先定冲突再排场景顺序", existing)
        self.assertEqual(target[0].method_id, "method-b")

    def test_an_empty_candidate_is_ignored(self) -> None:
        self.assertIsNone(plan_refinement({"title": ""}, [_method()]))


class LifecycleWriterTests(unittest.TestCase):
    def setUp(self) -> None:
        self._ctx = temp_workspace()
        self.root = self._ctx.__enter__()  # type: ignore[attr-defined]
        self.dm, self.clock = make_datamanager()
        self.clock.set_stage(Stage.SETTLE)
        self.clock.set_year(2020)
        self.clock.set_week(8)

    def tearDown(self) -> None:
        self._ctx.__exit__(None, None, None)  # type: ignore[attr-defined]

    def _seed(self, method: Methodology) -> None:
        from src.agents.cognition.event_store import CapabilityEventStore

        store = CapabilityEventStore(self.dm)
        payload = method.to_dict()
        method_id = payload.pop("method_id")
        store.append(
            "METHOD_PROPOSED", method_id=method_id, payload=payload,
            idempotency_key=f"METHOD_PROPOSED:{method_id}:-:seed",
        )
        store.append(
            "METHOD_VALUE_UPDATED",
            method_id=method_id,
            payload={
                "global_value": method.global_value,
                "confidence": method.confidence,
                "practice_count": method.practice_count,
                "success_count": method.success_count,
                "status": method.status,
                "context_values": dict(method.context_values),
                "last_used": method.last_used,
            },
            idempotency_key=f"METHOD_VALUE_UPDATED:{method_id}:-:seed",
        )

    def _lifecycle(self, enabled: bool = True) -> MethodLifecycle:
        return MethodLifecycle(
            dm=self.dm, clock=self.clock, config=LifecycleConfig(enabled=enabled)
        )

    def test_the_writer_archives_a_failing_unused_method(self) -> None:
        self._seed(
            _method(
                method_id="m-fail",
                status="deprecated",
                value=-0.5,
                practice=4,
                last_used="Y2020-W01-activity-D2",
            )
        )
        self.assertEqual(self._lifecycle().archive_failing_methods(), 1)
        from src.agents.cognition import materializer

        view = materializer.build_capability_view(self.dm)
        self.assertEqual(view["methodologies"]["m-fail"]["status"], "archived")

    def test_archiving_is_idempotent(self) -> None:
        self._seed(
            _method(
                method_id="m-fail",
                status="deprecated",
                value=-0.5,
                practice=4,
                last_used="Y2020-W01-activity-D2",
            )
        )
        lifecycle = self._lifecycle()
        self.assertEqual(lifecycle.archive_failing_methods(), 1)
        self.assertEqual(lifecycle.archive_failing_methods(), 0)

    def test_a_disabled_writer_does_nothing(self) -> None:
        self._seed(
            _method(
                method_id="m-fail",
                status="deprecated",
                value=-0.5,
                practice=4,
                last_used="Y2020-W01-activity-D2",
            )
        )
        self.assertEqual(self._lifecycle(enabled=False).archive_failing_methods(), 0)

    def test_archived_methods_leave_the_menu(self) -> None:
        from src.agents.cognition.method_hints import OFFERABLE_STATUSES

        self.assertNotIn("archived", OFFERABLE_STATUSES)
        self.assertNotIn("deprecated", OFFERABLE_STATUSES)

    def test_a_refinement_is_recorded_as_a_new_version(self) -> None:
        self._seed(_method(method_id="m-x", title="先定冲突再写场景", version=1))
        lifecycle = self._lifecycle()
        plan = lifecycle.record_refinement({"title": "先定冲突再排场景", "steps": ["新的一步"]})
        self.assertIsNotNone(plan)
        self.assertEqual(plan.version, 2)
        from src.agents.cognition import materializer

        view = materializer.build_capability_view(self.dm)
        entry = view["methodologies"]["m-x"]
        self.assertEqual(entry["version"], 2)
        self.assertEqual(entry["parent_method_id"], "m-x")
        self.assertEqual(entry["steps"], ["新的一步"])
        # A refinement is a re-wording of the same method, never a value change.
        self.assertEqual(entry["global_value"], 0.1)
        self.assertEqual(entry["practice_count"], 3)

    def test_a_refinement_without_a_target_is_not_written(self) -> None:
        self._seed(_method(method_id="m-x", title="先定冲突再写场景"))
        self.assertIsNone(self._lifecycle().record_refinement({"title": "每天晨跑五公里"}))


class SpecializationReachabilityTests(unittest.TestCase):
    """`specialized` 必须真的可达：它不该要求先 `validated`。

    实测 2 年运行 09272220：`specialized` 一次都没出现，因为实现多加了"先 overall
    validated"这个前提；而设计正文 §3.3 只要求"某一情境的 value 高于全局 ≥0.2"。
    这条前提本身也是矛盾的——"只在一个情境好用"的方法，整体上恰恰不该是 validated。
    """

    def test_a_specialised_method_does_not_have_to_be_validated_overall(self) -> None:
        from src.agents.cognition.reward_model import update_value

        update = update_value(
            value=0.05,
            practice_count=1,
            success_count=1,
            reward=0.2,
            current_status="tested",
            context_values={"learning+solo": {"value": 0.5, "count": 3}},
        )
        self.assertEqual(update.status, "specialized")
        self.assertIsNotNone(update.specialized_context)
        self.assertEqual(update.specialized_context["context_key"], "learning+solo")

    def test_a_method_that_is_good_everywhere_stays_validated(self) -> None:
        from src.agents.cognition.reward_model import update_value

        update = update_value(
            value=0.6,
            practice_count=3,
            success_count=3,
            reward=0.6,
            current_status="validated",
            context_values={
                "learning+solo": {"value": 0.65, "count": 4},
                "work+time_pressure": {"value": 0.62, "count": 4},
            },
        )
        self.assertEqual(update.status, "validated")
        self.assertIsNone(update.specialized_context)

    def test_a_deprecated_method_is_never_re_specialised(self) -> None:
        from src.agents.cognition.reward_model import update_value

        update = update_value(
            value=-0.4,
            practice_count=4,
            success_count=0,
            reward=-0.4,
            current_status="deprecated",
            context_values={"learning+solo": {"value": 0.9, "count": 5}},
        )
        self.assertEqual(update.status, "deprecated")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
