"""阶段 1：Methodology Shadow Mode（离线，无真实 LLM 调用）。

阶段 1 的硬约束（设计文档）：
- 抽取候选方法（周级、有预算）；
- 记录方法选择与实践证据，计算 value/confidence；
- **不影响角色行为**、**不改变现有 skill 数值**；
- 关掉开关即完全惰性（不产生事件、不调用模型）。
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

from src.agents.cognition.event_store import CapabilityEventStore
from src.agents.cognition.materializer import (
    build_capability_view,
    load_methodologies,
    materialize,
    write_capability_view,
)
from src.agents.cognition.methodology_policy import (
    context_signature_from_activity,
    deterministic_rng,
    exploration_rate_for,
    select_methods,
)
from src.agents.cognition.models import (
    Methodology,
    ContextSignature,
    method_id_for,
)
from src.agents.cognition.reward_model import (
    compute_reward,
    confidence_for,
    learning_rate,
    signals_from_activity_record,
    status_for,
    update_context_value,
    update_value,
)
from src.agents.cognition.shadow import MethodologyShadow, ShadowConfig
from src.world.clock import Stage
from tests._helpers import make_datamanager, temp_workspace


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
def _state(skills: Dict[str, float] | None = None) -> Dict[str, Any]:
    return {
        "vitality": 80,
        "fulfillment": {"mood": 50, "material": 50, "social": 50, "esteem": 50},
        "assets": {"deposit": 1000, "possessions": []},
        "skills": dict(skills or {"写作": 12, "跑步": 30}),
    }


def _activity_record(
    *,
    activity_id: str = "solo-Y2020-W02-activity-D2-甲",
    gains: Dict[str, float] | None = None,
    vitality: int = -2,
    money: int = 0,
    turns: int = 0,
    rejections: int = 0,
    content: str = "写小说",
) -> Dict[str, Any]:
    return {
        "type": "solo",
        "activity_id": activity_id,
        "time": "Y2020-W02-activity-D2",
        "content": content,
        "reflection": "感觉还行",
        "outcome": {
            "outcome": "写出了初稿",
            "delta_vitality": vitality,
            "delta_fulfillment": {"mood": 2},
            "delta_skills": dict(gains or {"写作": 2}),
            "delta_money": money,
        },
        "turns": turns,
        "verification_rejections": rejections,
    }


def _method(
    *,
    method_id: str = "method-写作-abc",
    skill_id: str = "写作",
    title: str = "先定冲突再写场景",
    status: str = "tested",
    value: float = 0.4,
    confidence: float = 0.5,
    practice: int = 4,
    success: int = 3,
    contexts: List[str] | None = None,
    contraindications: List[str] | None = None,
) -> Methodology:
    return Methodology(
        method_id=method_id,
        skill_id=skill_id,
        title=title,
        status=status,
        global_value=value,
        confidence=confidence,
        practice_count=practice,
        success_count=success,
        applicable_contexts=list(contexts or ["long_form"]),
        contraindications=list(contraindications or []),
    )


# --------------------------------------------------------------------------
class ModelContractTests(unittest.TestCase):
    def test_proposal_is_not_evidence(self) -> None:
        m = Methodology(method_id="m1", skill_id="写作", title="t")
        self.assertEqual(m.status, "proposed")
        self.assertEqual(m.global_value, 0.0)
        self.assertEqual(m.confidence, 0.0)
        self.assertFalse(m.is_evidence_backed)

    def test_unknown_status_or_source_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Methodology(method_id="m1", skill_id="写作", status="made_up")
        with self.assertRaises(ValueError):
            Methodology(method_id="m1", skill_id="写作", source_type="vibes")

    def test_roundtrip(self) -> None:
        m = _method()
        self.assertEqual(Methodology.from_dict(m.to_dict()).to_dict(), m.to_dict())

    def test_method_id_is_content_addressed_and_versioned(self) -> None:
        a = method_id_for("写作", "先定冲突")
        b = method_id_for("写作", "先定冲突")
        c = method_id_for("写作", "先写结尾")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertTrue(a.startswith("method-写作-"))
        self.assertTrue(method_id_for("写作", "先定冲突", version=2).endswith(".v2"))

    def test_context_signature_uses_the_controlled_vocabulary(self) -> None:
        sig = ContextSignature(
            skill_ids=["写作"],
            context_tags=["long_form", "not_a_real_tag", "long_form", "solo"],
        )
        self.assertEqual(sig.context_tags, ["long_form", "solo"])
        self.assertEqual(sig.context_key(), "long_form+solo")
        self.assertEqual(ContextSignature().context_key(), "default")


class RewardModelTests(unittest.TestCase):
    def test_signals_come_from_the_record_only(self) -> None:
        signals = signals_from_activity_record(_activity_record())
        self.assertEqual(signals.skill_gains, {"写作": 2})
        self.assertEqual(signals.delta_vitality, -2)
        # The reflection is not part of the objective signals.
        self.assertFalse(hasattr(signals, "reflection"))

    def test_reward_is_bounded_and_positive_for_a_clean_gain(self) -> None:
        signals = signals_from_activity_record(_activity_record(gains={"写作": 6}))
        reward = compute_reward(signals, skill_id="写作")
        self.assertGreater(reward.total, 0)
        self.assertLessEqual(reward.total, 1.0)
        self.assertEqual(reward.skill_gain, 1.0)
        self.assertEqual(reward.quality, 1.0)

    def test_costs_reduce_the_reward(self) -> None:
        cheap = compute_reward(
            signals_from_activity_record(_activity_record(gains={"写作": 3}, vitality=0)),
            skill_id="写作",
        )
        costly = compute_reward(
            signals_from_activity_record(
                _activity_record(gains={"写作": 3}, vitality=-5, money=-200, turns=12)
            ),
            skill_id="写作",
        )
        self.assertLess(costly.total, cheap.total)

    def test_rejections_and_social_damage_lower_quality(self) -> None:
        signals = signals_from_activity_record(_activity_record(rejections=3))
        reward = compute_reward(signals, skill_id="写作")
        self.assertEqual(reward.quality, 0.0)

    def test_spillover_counts_as_transferability(self) -> None:
        signals = signals_from_activity_record(
            _activity_record(gains={"写作": 2, "跑步": 2})
        )
        reward = compute_reward(signals, skill_id="写作")
        self.assertGreater(reward.transferability, 0)

    def test_learning_rate_shrinks_with_evidence(self) -> None:
        self.assertGreater(learning_rate(0), learning_rate(4))
        self.assertGreater(learning_rate(4), learning_rate(20))

    def test_value_and_confidence_are_separate(self) -> None:
        update = update_value(
            value=0.0, practice_count=0, success_count=0, reward=0.8
        )
        self.assertGreater(update.value, 0.3)  # one success moves value a lot
        self.assertLess(update.confidence, 0.25)  # but confidence stays low
        self.assertEqual(update.status, "tested")

    def test_value_converges_towards_the_reward(self) -> None:
        value, practice, success = 0.0, 0, 0
        for _ in range(6):
            u = update_value(
                value=value, practice_count=practice, success_count=success, reward=0.6
            )
            value, practice, success = u.value, u.practice_count, u.success_count
        self.assertAlmostEqual(value, 0.6, delta=0.12)
        self.assertEqual(practice, 6)
        self.assertEqual(success, 6)
        self.assertEqual(u.status, "validated")

    def test_repeated_failure_deprecates_a_method(self) -> None:
        self.assertEqual(
            status_for(practice_count=4, success_count=0, value=-0.4, current_status="tested"),
            "deprecated",
        )

    def test_context_values_are_tracked_separately(self) -> None:
        ctx = update_context_value({}, context_key="long_form", reward=0.5)
        ctx = update_context_value(ctx, context_key="time_pressure", reward=-0.5)
        self.assertEqual(ctx["long_form"]["count"], 1)
        self.assertGreater(ctx["long_form"]["value"], ctx["time_pressure"]["value"])

    def test_confidence_grows_with_practice(self) -> None:
        self.assertLess(confidence_for(1), confidence_for(8))


class PolicyTests(unittest.TestCase):
    def test_context_signature_is_built_from_objective_fields(self) -> None:
        sig = context_signature_from_activity(_activity_record(), skill_id="写作")
        self.assertEqual(sig.skill_ids, ["写作"])
        self.assertIn("learning", sig.context_tags)
        self.assertIn("solo", sig.context_tags)

    def test_selection_prefers_the_better_method(self) -> None:
        good = _method(method_id="m-good", value=0.8, confidence=0.9, practice=10, success=9)
        bad = _method(method_id="m-bad", title="另一个", value=-0.2, confidence=0.2)
        selection = select_methods(
            [bad, good],
            ContextSignature(context_tags=["long_form"]),
            exploration_rate=0.0,
        )
        self.assertEqual(selection.primary.method.method_id, "m-good")
        self.assertEqual(selection.context_key, "long_form")

    def test_supporting_method_comes_from_another_skill(self) -> None:
        a = _method(method_id="m-a", skill_id="写作", value=0.5)
        b = _method(method_id="m-b-same", skill_id="写作", title="同技能", value=0.45)
        c = _method(method_id="m-c-other", skill_id="跑步", title="跨技能", value=0.4)
        selection = select_methods(
            [a, b, c], ContextSignature(context_tags=["long_form"]), exploration_rate=0.0
        )
        self.assertEqual(selection.primary.method.method_id, "m-a")
        self.assertEqual(selection.supporting.method.method_id, "m-c-other")

    def test_contraindicated_method_is_avoided(self) -> None:
        risky = _method(
            method_id="m-risky",
            value=0.9,
            contraindications=["time_pressure"],
        )
        safe = _method(method_id="m-safe", title="稳妥", value=0.2)
        selection = select_methods(
            [risky, safe],
            ContextSignature(context_tags=["time_pressure"]),
            exploration_rate=0.0,
        )
        self.assertEqual(selection.primary.method.method_id, "m-safe")
        self.assertTrue(
            next(s for s in selection.candidates if s.method.method_id == "m-risky").contraindicated
        )

    def test_exploration_can_pick_an_unproven_method(self) -> None:
        proven = _method(method_id="m-proven", value=0.9, status="validated", practice=10, success=9)
        fresh = _method(
            method_id="m-fresh", title="新方法", value=0.0, status="proposed", practice=0, success=0, confidence=0.0
        )
        selection = select_methods(
            [proven, fresh],
            ContextSignature(context_tags=["long_form"]),
            exploration_rate=1.0,
            rng=deterministic_rng("test"),
        )
        self.assertTrue(selection.explored)
        self.assertEqual(selection.primary.method.method_id, "m-fresh")

    def test_selection_is_deterministic(self) -> None:
        methods = [_method(method_id=f"m{i}", title=f"t{i}", value=0.1 * i) for i in range(5)]
        sig = ContextSignature(context_tags=["long_form", "solo"])
        first = select_methods(methods, sig, exploration_rate=0.3, rng=deterministic_rng("x", 1))
        second = select_methods(methods, sig, exploration_rate=0.3, rng=deterministic_rng("x", 1))
        self.assertEqual(
            first.primary.method.method_id, second.primary.method.method_id
        )

    def test_exploration_rate_follows_personality(self) -> None:
        low = exploration_rate_for(creativity=10, curiosity=10)
        high = exploration_rate_for(creativity=95, curiosity=90)
        self.assertLess(low, high)
        self.assertGreaterEqual(low, 0.05)
        self.assertLessEqual(high, 0.5)

    def test_empty_candidate_set_selects_nothing(self) -> None:
        self.assertTrue(select_methods([], ContextSignature()).empty)


class MaterializerTests(unittest.TestCase):
    def test_fold_builds_methods_skills_and_stats(self) -> None:
        events = [
            {
                "type": "METHOD_PROPOSED",
                "method_id": "m1",
                "skill_id": "写作",
                "title": "先定冲突",
                "source_type": "practice_reflection",
                "status": "proposed",
                "ledger_event_id": "ev-1",
                "time": "Y2020-W01-review",
            },
            {
                "type": "METHOD_SELECTED",
                "method_id": "m1",
                "activity_id": "a1",
                "context_key": "long_form",
                "role": "primary",
                "time": "Y2020-W02-activity-D2",
            },
            {
                "type": "METHOD_VALUE_UPDATED",
                "method_id": "m1",
                "global_value": 0.4,
                "confidence": 0.2,
                "practice_count": 1,
                "success_count": 1,
                "status": "tested",
                "context_values": {"long_form": {"value": 0.4, "count": 1}},
                "time": "Y2020-W02-activity-D2",
            },
        ]
        view = materialize(events, skills={"写作": 12}, persona="甲")
        method = view["methodologies"]["m1"]
        self.assertEqual(method["title"], "先定冲突")
        self.assertEqual(method["global_value"], 0.4)
        self.assertEqual(method["selections"], 1)
        self.assertEqual(view["skills"]["写作"]["methodology_ids"], ["m1"])
        self.assertEqual(view["stats"]["by_status"], {"tested": 1})
        self.assertEqual(method["proposed_week"], "Y2020-W01")
        self.assertEqual(view["stats"]["methods"], 1)

    def test_unmapped_skill_is_flagged_not_invented(self) -> None:
        events = [
            {
                "type": "METHOD_PROPOSED",
                "method_id": "m2",
                "skill_id": "占星",
                "skill_mapped": False,
                "title": "看星象",
                "time": "Y2020-W01-review",
            }
        ]
        view = materialize(events, skills={"写作": 12}, persona="甲")
        self.assertIn("m2", view["stats"]["unmapped_skill_methods"])
        self.assertNotIn("占星", view["skills"])

    def test_load_methodologies_round_trips_into_objects(self) -> None:
        events = [
            {
                "type": "METHOD_PROPOSED",
                "method_id": "m1",
                "skill_id": "写作",
                "title": "t",
                "time": "Y2020-W01-review",
            }
        ]
        view = materialize(events, skills={"写作": 1})
        methods = load_methodologies(view)
        self.assertEqual([m.method_id for m in methods], ["m1"])


class ShadowModeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._ctx = temp_workspace()
        self.root = self._ctx.__enter__()
        self.dm, self.clock = make_datamanager()
        self.dm.save_state(_state())
        self.clock.set_stage(Stage.ACTIVITY)
        self.clock.set_day(2)

    def tearDown(self) -> None:
        self._ctx.__exit__(None, None, None)

    def _shadow(self, *, enabled: bool = True, model: str = "test-model") -> MethodologyShadow:
        return MethodologyShadow(
            dm=self.dm,
            clock=self.clock,
            agent_name=self.dm.char,
            model=model,
            traits={"creativity": 60, "curiosity": 60},
            config=ShadowConfig(enabled=enabled, max_extract_calls_per_week=1),
        )

    def _next_week(self) -> None:
        """Move the clock one week on: proposals become evidence-eligible.

        Methods proposed in week N may only collect evidence from week N+1
        onwards (reflection cannot reinforce itself).
        """
        self.clock.set_week(2)
        self.clock.set_stage(Stage.ACTIVITY)
        self.clock.set_day(2)

    def _propose(self, shadow: MethodologyShadow, **overrides) -> str:
        skill_id = overrides.pop("skill_id", "写作")
        title = overrides.pop("title", "先定冲突再写场景")
        method_id = method_id_for(skill_id, title)
        shadow.store.append(
            "METHOD_PROPOSED",
            method_id=method_id,
            payload={
                "skill_id": skill_id,
                "title": title,
                "source_type": "practice_reflection",
                "status": "proposed",
                "week": "Y2020-W01",
                **overrides,
            },
        )
        shadow._invalidate()
        return method_id

    # -- gating ------------------------------------------------------------
    def test_disabled_shadow_writes_nothing(self) -> None:
        shadow = self._shadow(enabled=False)
        self.assertIsNone(shadow.observe_activity(_activity_record()))
        self.assertEqual(shadow.after_review(), 0)
        self.assertFalse(shadow.store.path.exists())

    def test_disabled_shadow_never_calls_the_model(self) -> None:
        shadow = self._shadow(enabled=False)
        with mock.patch("src.utils.get_response_with_retry") as llm:
            shadow.after_review()
        llm.assert_not_called()

    def test_no_candidates_means_no_selection(self) -> None:
        shadow = self._shadow()
        self.assertIsNone(shadow.observe_activity(_activity_record()))
        self.assertFalse(shadow.store.path.exists())

    # -- selection + evidence ---------------------------------------------
    def test_activity_selection_records_shadow_attribution_and_value(self) -> None:
        shadow = self._shadow()
        method_id = self._propose(shadow)
        self._next_week()

        record = shadow.observe_activity(_activity_record(gains={"写作": 4}))

        self.assertIsNotNone(record)
        events = shadow.store.events()
        types = [e["type"] for e in events]
        self.assertIn("METHOD_SELECTED", types)
        self.assertIn("METHOD_OUTCOME_OBSERVED", types)
        self.assertIn("METHOD_VALUE_UPDATED", types)

        selected = next(e for e in events if e["type"] == "METHOD_SELECTED")
        self.assertTrue(selected["shadow"])
        self.assertEqual(selected["method_id"], method_id)
        self.assertEqual(selected["role"], "primary")

        observed = next(e for e in events if e["type"] == "METHOD_OUTCOME_OBSERVED")
        self.assertEqual(observed["attribution"], "shadow_counterfactual")
        self.assertGreater(observed["reward"], 0)

        update = next(e for e in events if e["type"] == "METHOD_VALUE_UPDATED")
        self.assertEqual(update["practice_count"], 1)
        self.assertEqual(update["status"], "tested")
        self.assertGreater(update["global_value"], 0.0)
        self.assertEqual(update["evidence_kind"], "shadow_counterfactual")

    def test_value_never_moves_for_a_proposal_in_the_same_week(self) -> None:
        """A method proposed at REVIEW cannot be reinforced by that same week."""
        shadow = self._shadow()
        self._propose(shadow, week="Y2020-W01")  # the clock stands in W01
        shadow.observe_activity(_activity_record())
        updates = [e for e in shadow.store.events() if e["type"] == "METHOD_VALUE_UPDATED"]
        self.assertEqual(updates, [])
        self.assertEqual(shadow.selectable_methods(), [])

    def test_a_proposal_from_an_earlier_week_is_evidence_eligible(self) -> None:
        shadow = self._shadow()
        self._propose(shadow, week="Y2020-W01")
        self.clock.set_week(2)
        self.clock.set_stage(Stage.ACTIVITY)
        self.clock.set_day(2)

        record = shadow.observe_activity(_activity_record(activity_id="solo-W02-D2"))

        self.assertIsNotNone(record)
        self.assertEqual(len(shadow.selectable_methods()), 1)

    def test_repeated_observation_of_the_same_activity_is_idempotent(self) -> None:
        shadow = self._shadow()
        self._propose(shadow)
        self._next_week()
        record = _activity_record()

        first = shadow.observe_activity(record)
        self.assertIsNotNone(first)
        before = len(shadow.store.events())

        second = shadow.observe_activity(record)

        self.assertIsNotNone(second)
        self.assertTrue(second.get("duplicate"))
        self.assertEqual(len(shadow.store.events()), before)

    def test_skill_numbers_are_never_touched(self) -> None:
        shadow = self._shadow()
        self._propose(shadow)
        self._next_week()
        before = self.dm.read_state(exclude_cur_t=False)["skills"]
        shadow.observe_activity(_activity_record(gains={"写作": 4}))
        after = self.dm.read_state(exclude_cur_t=False)["skills"]
        self.assertEqual(before, after)

    # -- weekly extraction -------------------------------------------------
    def test_extraction_records_proposed_methods(self) -> None:
        shadow = self._shadow()
        payload = {
            "methods": [
                {
                    "skill_id": "写作",
                    "skill_mapped": True,
                    "title": "先写结尾再回填",
                    "description": "适用于结构复杂的长文",
                    "applicable_contexts": ["long_form"],
                    "contraindications": ["time_pressure"],
                    "steps": ["写结尾", "回填中段"],
                    "checks": ["每段都有落点"],
                    "failure_modes": ["结尾与主题脱节"],
                }
            ]
        }
        with mock.patch("src.utils.get_response_with_retry", return_value=payload) as llm:
            written = shadow.after_review(week_records=[_activity_record()])

        self.assertEqual(written, 1)
        llm.assert_called_once()
        events = shadow.store.events()
        self.assertEqual([e["type"] for e in events], ["METHOD_PROPOSED"])
        self.assertEqual(events[0]["status"], "proposed")
        self.assertEqual(events[0]["global_value"], 0.0)
        self.assertEqual(events[0]["confidence"], 0.0)

    def test_extraction_is_idempotent_and_skips_known_methods(self) -> None:
        shadow = self._shadow()
        self._propose(shadow, title="先定冲突再写场景")
        payload = {
            "methods": [
                {"skill_id": "写作", "title": "先定冲突再写场景"},
                {"skill_id": "写作", "title": "全新方法"},
            ]
        }
        with mock.patch("src.utils.get_response_with_retry", return_value=payload):
            written = shadow.after_review(week_records=[_activity_record()])
        self.assertEqual(written, 1, "the known method must not be duplicated")

        # A second call in the same week is refused by the budget, not by dedup.
        with mock.patch("src.utils.get_response_with_retry", return_value=payload) as llm:
            self.assertEqual(shadow.after_review(week_records=[_activity_record()]), 0)
        llm.assert_not_called()

    def test_extraction_survives_a_bad_model_answer(self) -> None:
        shadow = self._shadow()
        with mock.patch("src.utils.get_response_with_retry", side_effect=RuntimeError("boom")):
            self.assertEqual(shadow.after_review(week_records=[_activity_record()]), 0)
        with mock.patch("src.utils.get_response_with_retry", return_value=None):
            self.assertEqual(shadow.after_review(week_records=[_activity_record()]), 0)

    def test_extraction_skips_a_week_without_evidence(self) -> None:
        shadow = self._shadow()
        with mock.patch("src.utils.get_response_with_retry") as llm:
            self.assertEqual(shadow.after_review(week_records=[]), 0)
        llm.assert_not_called()

    def test_budget_can_be_zero(self) -> None:
        shadow = MethodologyShadow(
            dm=self.dm,
            clock=self.clock,
            agent_name=self.dm.char,
            model="test-model",
            config=ShadowConfig(enabled=True, max_extract_calls_per_week=0),
        )
        with mock.patch("src.utils.get_response_with_retry") as llm:
            self.assertEqual(shadow.after_review(week_records=[_activity_record()]), 0)
        llm.assert_not_called()

    # -- views -------------------------------------------------------------
    def test_view_is_rebuilt_from_events(self) -> None:
        shadow = self._shadow()
        self._propose(shadow)
        self._next_week()
        shadow.observe_activity(_activity_record(gains={"写作": 4}))

        path = write_capability_view(self.dm)
        first = path.read_text(encoding="utf-8")
        path.unlink()

        rebuilt = write_capability_view(self.dm)
        self.assertEqual(rebuilt.read_text(encoding="utf-8"), first)

        view = build_capability_view(self.dm)
        self.assertEqual(view["persona"], self.dm.char)
        self.assertEqual(view["stats"]["methods"], 1)
        self.assertEqual(view["skills"]["写作"]["methodology_ids"], list(view["methodologies"]))
        self.assertGreater(view["event_count"], 0)

    def test_config_defaults_to_disabled(self) -> None:
        self.assertFalse(ShadowConfig.from_world_config({}).enabled)
        self.assertFalse(
            ShadowConfig.from_world_config({"cognition": {}}).enabled
        )
        cfg = ShadowConfig.from_world_config(
            {"cognition": {"methodology_shadow": True, "max_extract_calls_per_week": 2}}
        )
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.max_extract_calls_per_week, 2)
        # A malformed cap degrades to the default instead of raising.
        self.assertEqual(
            ShadowConfig.from_world_config(
                {"cognition": {"methodology_shadow": True, "max_extract_calls_per_week": "x"}}
            ).max_extract_calls_per_week,
            1,
        )


class WiringTests(unittest.TestCase):
    """阶段 1 与既有链路的接线：观察者钩子与"忽略 cognition 流"的对比。"""

    def setUp(self) -> None:
        self._ctx = temp_workspace()
        self.root = self._ctx.__enter__()
        self.dm, self.clock = make_datamanager()
        self.dm.save_state(_state())

    def tearDown(self) -> None:
        self._ctx.__exit__(None, None, None)

    def _solo_record(self, activity_id: str = "solo-W01-D1"):
        from src.world.clock import TimeState
        from src.world.solo_activity_data import ActionOutcome, SoloActivityRecord

        return SoloActivityRecord(
            activity_id=activity_id,
            agent_name=self.dm.char,
            time=TimeState(2020, 1, Stage.ACTIVITY, 1, 0),
            content="读书",
            outcome=ActionOutcome(
                outcome="读完一章",
                delta_vitality=1,
                delta_fulfillment={},
                delta_skills={"写作": 1},
            ),
        )

    def test_activity_records_notify_the_observer(self) -> None:
        seen: List[Dict[str, Any]] = []
        self.dm.activity_record_observer = seen.append

        self.dm.append_activity_record(self._solo_record())

        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]["activity_id"], "solo-W01-D1")
        self.assertEqual(seen[0]["outcome"]["delta_skills"], {"写作": 1})

    def test_a_failing_observer_never_breaks_the_ledger(self) -> None:
        def boom(_record):
            raise RuntimeError("observer exploded")

        self.dm.activity_record_observer = boom
        self.dm.append_activity_record(self._solo_record())

        rows = [
            json.loads(l)
            for l in (self.dm.root / "activity.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["ledger_event_id"].startswith("ev-"))

    def test_shadow_streams_can_be_ignored_when_comparing_runs(self) -> None:
        from src.world.views import build_event_index, compare_event_index, filter_streams

        run_a = Path("data/run_a")
        run_b = Path("data/run_b")
        for run in (run_a, run_b):
            path = run / "persona/甲/state.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"time": "Y2020-W01-begin", "ledger_event_id": "ev-1"}) + chr(10),
                encoding="utf-8",
            )
        # run_b additionally carries a cognition stream.
        extra = run_b / "persona/甲/cognition/capability_events.jsonl"
        extra.parent.mkdir(parents=True, exist_ok=True)
        extra.write_text(
            json.dumps({"time": "Y2020-W01-review", "ledger_event_id": "ev-2"}) + chr(10),
            encoding="utf-8",
        )

        raw = compare_event_index(build_event_index(run_a), build_event_index(run_b))
        self.assertFalse(raw["identical"], "the raw comparison must see the extra stream")

        filtered = compare_event_index(
            filter_streams(build_event_index(run_a), ["persona/*/cognition/*"]),
            filter_streams(build_event_index(run_b), ["persona/*/cognition/*"]),
        )
        self.assertTrue(filtered["identical"], "the ledger itself is unchanged")


if __name__ == "__main__":
    unittest.main()
