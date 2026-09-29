"""验收工具自身的回归测试（离线，无 LLM 调用）。

`scripts/audit_cognition_run.py` 是阶段 1-3 的验收取证工具，它必须和真实视图
schema 对齐，否则会给出错误的结论。本轮（1 年真跑验收）就发现过一次这样的
事故：`memory_graph.json` 的边用 `source` / `target` / `types`，旧审计脚本按
`source_memory_id` / `relation_type` 读取，于是把所有边都判成
`bad_references=unknown`、`by_type` 全空。

这里用一份合成运行目录把这些口径钉住：关系视图字段、未运行角色的识别、
实体丢弃率、同一洞见的重述检测、idea 转换方法的结构对比、账本重复行。
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_cognition_run import audit_run  # noqa: E402


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def _memory(memory_id: str, content: str, *, entities=None, dropped=None) -> dict:
    return {
        "memory_id": memory_id,
        "persona": "甲",
        "kind": "lesson",
        "content": content,
        "entities": entities or [],
        "dropped_entities": dropped or [],
        "source_event_ids": ["ev-1"],
        "status": "active",
        "tier": "hot",
        "protected": False,
        "recall_count": 1,
        "created_at": "Y2020-W01-review",
    }


def _build_run(root: Path) -> Path:
    run = root / "synthetic_world_00000000"
    _write_json(
        run / "config.json",
        {
            "world": {
                "name": "synthetic_world",
                "language": "zh",
                "time": {"n_year": 1, "n_week": 2},
                "cognition": {"methodology_shadow": True, "memory_shadow": True},
            },
            "role_model": "stub",
            "god_model": "stub",
            "fallback_model": "stub",
        },
    )

    persona = run / "persona" / "甲"
    (persona / "profile").mkdir(parents=True, exist_ok=True)
    (run / "persona" / "乙" / "profile").mkdir(parents=True, exist_ok=True)

    _write_jsonl(
        persona / "activity.jsonl",
        [
            {
                "time": "Y2020-W01-activity-D1",
                "type": "solo",
                "activity_id": "solo-Y2020-W01-activity-D1-甲",
                "ledger_event_id": "ev-a1",
                "schema_version": 1,
            },
            {
                "time": "Y2020-W02-activity-D1",
                "type": "public",
                "activity_id": "public-Y2020-W02-activity-D1-晨跑团",
                "ledger_event_id": "ev-a2",
                "schema_version": 1,
            },
        ],
    )
    _write_jsonl(
        persona / "memory" / "weekly_diary.jsonl",
        [
            {"time": "Y2020-W01", "content": "Summary: 第一周"},
            {"time": "Y2020-W02", "content": "Summary: 第二周"},
        ],
    )
    _write_jsonl(
        persona / "generation" / "year=2020" / "week=1.jsonl",
        [
            {
                "record_id": "Y2020-W01-review#1",
                "time": "Y2020-W01-review",
                "outputs": [{"role": "assistant", "content": "Summary: ok"}],
                "rejected": False,
            }
        ],
    )

    capability_events = [
        {
            "time": "Y2020-W01-review",
            "type": "METHOD_PROPOSED",
            "idempotency_key": "METHOD_PROPOSED:method-practice-1:-:Y2020-W01",
            "ledger_event_id": "ev-c1",
            "schema_version": 1,
            "method_id": "method-practice-1",
            "skill_id": "厨艺",
            "title": "收工顺路备菜做饭法",
            "description": "顺路买菜、快速成菜。",
            "source_type": "practice_reflection",
            "steps": ["先想路线", "只买耐放的", "一锅出"],
            "checks": ["当天吃上热饭"],
            "failure_modes": ["太累点外卖"],
            "applicable_contexts": ["routine", "low_money"],
        },
        {
            "time": "Y2020-W02-review",
            "type": "METHOD_PROPOSED",
            "idempotency_key": "METHOD_PROPOSED:method-idea-1:-:idea-conversion",
            "ledger_event_id": "ev-c2",
            "schema_version": 1,
            "method_id": "method-idea-1",
            "skill_id": "人际沟通",
            "title": "既然上周上课能学到东西又能带回菜，可以再报一次课并主动向同组学员开口请教做法",
            "description": "整句假设被当成方法名。",
            "source_type": "idea_conversion",
            "source_idea_id": "idea-contradictio-1",
            "steps": ["本周再报一次课并主动开口。"],
            "checks": [],
            "failure_modes": [],
            "applicable_contexts": [],
            "global_value": 0.0,
            "confidence": 0.0,
        },
        {
            "time": "Y2020-W02-activity-D1",
            "type": "METHOD_SELECTED",
            "idempotency_key": "METHOD_SELECTED:method-idea-1:solo-Y2020-W02-activity-D1-甲:",
            "ledger_event_id": "ev-c3",
            "schema_version": 1,
            "method_id": "method-idea-1",
            "activity_id": "solo-Y2020-W02-activity-D1-甲",
            "role": "primary",
            "context_key": "social",
            "explored": True,
            "score": 0.0,
        },
        {
            "time": "Y2020-W02-activity-D1",
            "type": "METHOD_OUTCOME_OBSERVED",
            "idempotency_key": "METHOD_OUTCOME_OBSERVED:method-idea-1:solo-Y2020-W02-activity-D1-甲:Y2020-W02",
            "ledger_event_id": "ev-c4",
            "schema_version": 1,
            "method_id": "method-idea-1",
            "activity_id": "solo-Y2020-W02-activity-D1-甲",
            "context_key": "social",
            "reward": 0.2,
        },
        {
            "time": "Y2020-W02-activity-D1",
            "type": "METHOD_VALUE_UPDATED",
            "idempotency_key": "METHOD_VALUE_UPDATED:method-idea-1:solo-Y2020-W02-activity-D1-甲:Y2020-W02",
            "ledger_event_id": "ev-c5",
            "schema_version": 1,
            "method_id": "method-idea-1",
            "activity_id": "solo-Y2020-W02-activity-D1-甲",
            "global_value": 0.1,
            "confidence": 0.2,
            "practice_count": 1,
            "success_count": 1,
        },
    ]
    _write_jsonl(persona / "cognition" / "capability_events.jsonl", capability_events)

    memory_events = [
        {
            "time": "Y2020-W01-review",
            "type": "MEMORY_CREATED",
            "idempotency_key": "MEMORY_CREATED:mem-1:Y2020-W01",
            "ledger_event_id": "ev-m1",
            "schema_version": 1,
            "memory_id": "mem-1",
            "week": "Y2020-W01",
            "kind": "lesson",
            "content": "独自参加活动能学到东西，但认识人得主动开口。",
            "entities": ["吉日和"],
            "dropped_entities": ["九亭公园", "菜市场"],
            "source_event_ids": ["ev-a1"],
            "confidence": 0.6,
        },
        {
            "time": "Y2020-W02-review",
            "type": "MEMORY_CREATED",
            "idempotency_key": "MEMORY_CREATED:mem-2:Y2020-W02",
            "ledger_event_id": "ev-m2",
            "schema_version": 1,
            "memory_id": "mem-2",
            "week": "Y2020-W02",
            "kind": "episodic",
            "content": "棋摊上先看棋再递烟，跟大爷们聊上了。",
            "entities": [],
            "dropped_entities": ["苏州河", "棋摊"],
            "source_event_ids": ["ev-a2"],
            "confidence": 0.6,
        },
    ]
    _write_jsonl(persona / "cognition" / "memory_events.jsonl", memory_events)
    _write_jsonl(
        persona / "cognition" / "memory_relation_events.jsonl",
        [
            {
                "time": "Y2020-W02-review",
                "type": "RELATION_ASSERTED",
                "idempotency_key": "RELATION_ASSERTED:mem-1:mem-2:contradiction,temporal",
                "ledger_event_id": "ev-r1",
                "schema_version": 1,
                "source_memory_id": "mem-1",
                "target_memory_id": "mem-2",
                "relationship_types": ["temporal", "contradiction"],
                "score": 0.31,
                "week": "Y2020-W02",
            }
        ],
    )

    idea_events = [
        {
            "time": "Y2020-W01-review",
            "type": "IDEA_CREATED",
            "idempotency_key": "IDEA_CREATED:idea-contradictio-1:Y2020-W01",
            "ledger_event_id": "ev-i1",
            "schema_version": 1,
            "idea_id": "idea-contradictio-1",
            "content": "在体验课上用带回的菜当由头，主动跟同课的人聊做法。",
            "motif": "contradiction",
            "source_memory_ids": ["mem-1", "mem-2"],
            "test_plan": "本周再上一次课，主动跟一个人说话并记录回应。",
            "potential": 0.2,
        },
        {
            "time": "Y2020-W02-review",
            "type": "IDEA_CREATED",
            "idempotency_key": "IDEA_CREATED:idea-contradictio-2:Y2020-W02",
            "ledger_event_id": "ev-i2",
            "schema_version": 1,
            "idea_id": "idea-contradictio-2",
            # Same insight re-proposed a week later with the same plan: exactly
            # what `idea_engine._semantic_duplicate` (KI-5) must fold.
            "content": "在体验课上用带回的菜当由头，主动跟同课的人聊做法，顺便认识人。",
            "motif": "contradiction",
            "source_memory_ids": ["mem-1", "mem-2"],
            "test_plan": "本周再上一次课，主动跟一个人说话并记录回应。",
            "potential": 0.19,
        },
    ]
    _write_jsonl(persona / "cognition" / "idea_events.jsonl", idea_events)

    views = persona / "cognition" / "views"
    _write_json(
        views / "memories.json",
        {
            "persona": "甲",
            "memories": {
                "mem-1": _memory("mem-1", "独自参加活动能学到东西", entities=["吉日和"], dropped=["九亭公园"]),
                "mem-2": _memory("mem-2", "棋摊上先看棋再递烟", dropped=["苏州河"]),
            },
            "stats": {},
        },
    )
    _write_json(
        views / "memory_graph.json",
        {
            "nodes": {"mem-1": {}, "mem-2": {}},
            "edges": [
                {
                    "source": "mem-1",
                    "target": "mem-2",
                    "types": ["temporal", "contradiction"],
                    "score": 0.31,
                }
            ],
            "stats": {},
        },
    )
    _write_json(
        views / "ideas.json",
        {
            "persona": "甲",
            "ideas": {
                "idea-contradictio-1": {
                    "idea_id": "idea-contradictio-1",
                    "content": "在体验课上用带回的菜当由头，主动跟同课的人聊做法。",
                    "motif": "contradiction",
                    "status": "candidate",
                    "confidence": 0.5,
                    "potential": 0.2,
                    "created_at": "Y2020-W01-review",
                    "test_plan": "本周再上一次课，主动跟一个人说话并记录回应。",
                    "source_memory_ids": ["mem-1", "mem-2"],
                    "related_goal_ids": ["建立联系"],
                    "converted_method_id": "method-idea-1",
                },
                "idea-contradictio-2": {
                    "idea_id": "idea-contradictio-2",
                    "content": "在体验课上用带回的菜当由头，主动跟同课的人聊做法，顺便认识人。",
                    "motif": "contradiction",
                    "status": "candidate",
                    "confidence": 0.5,
                    "potential": 0.19,
                    "created_at": "Y2020-W02-review",
                    "test_plan": "本周再上一次课，主动跟一个人说话并记录回应。",
                    "source_memory_ids": ["mem-1", "mem-2"],
                    "related_goal_ids": [],
                    "converted_method_id": None,
                },
            },
            "rejected": [],
            "stats": {},
        },
    )
    _write_json(
        views / "retrieval_shadow.json",
        {
            "stats": {"weeks": 1},
            "weeks": [
                {
                    "week": "Y2020-W01",
                    "legacy": {"n_items": 9, "est_tokens": 1000},
                    "proposed": {"n_items": 2, "est_tokens": 200},
                    "overlap": {"shared": 0},
                    "cold_reactivated": 0,
                }
            ],
        },
    )
    return run


class AuditRunTests(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory(prefix="agentopia-audit-")
        self.run_dir = _build_run(Path(self._tmp.name))
        self.report = audit_run(self.run_dir)

    def _persona(self, name: str = "甲") -> dict:
        for persona in self.report["personas"]:
            if persona["persona"] == name:
                return persona
        raise AssertionError(f"persona {name} missing from the report")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_relation_view_schema_is_read_correctly(self) -> None:
        """Regression: edges use source/target/types (not *_memory_id/relation_type)."""
        persona = self._persona()
        self.assertEqual(persona["relations"]["bad_references"], [])
        self.assertEqual(persona["relations"]["by_type"], {"contradiction": 1, "temporal": 1})

    def test_persona_without_activity_is_reported_as_not_simulated(self) -> None:
        self.assertEqual(self.report["personas_simulated"], 1)
        self.assertEqual(self.report["personas_not_simulated"], ["乙"])

    def test_entity_drop_rate_and_flag(self) -> None:
        persona = self._persona()
        self.assertEqual(persona["memory"]["accepted_entities"], 1)
        self.assertEqual(persona["memory"]["dropped_entities"], 4)
        self.assertAlmostEqual(persona["memory"]["entity_drop_rate"], 0.8, places=4)
        self.assertTrue(any("entities were dropped" in f for f in self.report["red_flags"]))

    def test_reworded_ideas_are_clustered_as_one_insight(self) -> None:
        """KI-5: a re-proposal the engine should have folded shows up as a cluster."""
        persona = self._persona()
        clusters = persona["ideas"]["insight_clusters"]
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]["ideas"], ["idea-contradictio-1", "idea-contradictio-2"])

    def test_shared_evidence_alone_is_not_a_duplicate(self) -> None:
        """Two different hypotheses built on the same memories are not one insight."""
        persona = self._persona()
        self.assertEqual(
            [c["ideas"] for c in persona["ideas"]["shared_evidence_clusters"]],
            [["idea-contradictio-1", "idea-contradictio-2"]],
        )
        # ... and the engine-aligned cluster rule must not fire for a pair whose
        # wording and plan genuinely differ.
        from src.agents.cognition.idea_engine import _semantic_duplicate
        from src.agents.cognition.idea_models import Idea

        existing = Idea(
            idea_id="idea-old",
            content="在家常菜课上主动跟同组的人搭话，可能更容易认识人",
            test_plan="本周上一次课，主动跟一个人说话并记录回应",
            motif="contradiction",
            source_memory_ids=["mem-1", "mem-2"],
        )
        self.assertIsNone(
            _semantic_duplicate(
                candidate=type("C", (), {"memory_ids": ["mem-1", "mem-2", "mem-3"]})(),
                content="把练一休一写进训练周计划，观察恢复是否变好",
                test_plan="本周隔天训练，长距离跑后只做拉伸，记录每天体感",
                existing_ideas=[existing],
                existing_methods=[],
            )
        )

    def test_idea_conversion_methods_are_flagged_as_structurally_poorer(self) -> None:
        persona = self._persona()
        summary = persona["methodologies"]["structure_by_source"]
        self.assertEqual(summary["idea_conversion"]["avg_steps"], 1.0)
        self.assertEqual(summary["idea_conversion"]["avg_checks"], 0.0)
        self.assertEqual(summary["practice_reflection"]["avg_steps"], 3.0)
        self.assertTrue(
            any("structurally poorer" in f for f in self.report["red_flags"])
        )

    def test_all_positive_outcomes_are_flagged(self) -> None:
        self.assertTrue(
            any("every observed outcome reward is positive" in f for f in self.report["red_flags"])
        )

    def test_ledger_duplicate_rows_are_detected(self) -> None:
        # Duplicate the same record: content-addressed identity means one id twice.
        path = self.run_dir / "persona" / "甲" / "activity.jsonl"
        row = path.read_text(encoding="utf-8").splitlines()[0]
        path.write_text(
            path.read_text(encoding="utf-8") + row + "\n", encoding="utf-8"
        )
        report = audit_run(self.run_dir)
        self.assertTrue(report["ledger"]["duplicate_ids"])
        self.assertLess(report["ledger"]["unique_ids"], report["ledger"]["records"])
        self.assertIn("ev-a1", report["ledger"]["duplicate_ids"])

    def test_markdown_renders(self) -> None:
        from audit_cognition_run import render_markdown

        text = render_markdown(self.report)
        self.assertIn("# Cognition run audit", text)
        self.assertIn("## Red flags", text)


def _enable_hints_with_events(run: Path, *, leak: bool = False) -> None:
    """Turn the synthetic run into a phase-4 run with one full hint cycle.

    The appended events are exactly what `method_hints.py` writes: two offers,
    one adoption, one claimed practice with real attribution, and the
    `MEMORY_USED_IN_PLAN` row the adoption triggers (KI-10).
    """
    persona = run / "persona" / "甲"
    caps = persona / "cognition" / "capabilities.json"
    del caps  # the view is rebuilt by the audit tool, nothing to keep in sync

    activity_id = "solo-Y2020-W02-activity-D1-甲"
    counterfactual_activity = "solo-Y2020-W01-activity-D1-甲"
    cap_path = persona / "cognition" / "capability_events.jsonl"
    rows = [
        json.loads(line) for line in cap_path.read_text(encoding="utf-8").splitlines() if line
    ]
    # The adopted method carries source memories, so the KI-10 link is checkable.
    # The pre-existing outcome evidence becomes counterfactual evidence about a
    # *different* activity: a claimed practice is never observed by the shadow.
    for row in rows:
        if row.get("method_id") == "method-idea-1":
            row["source_memory_ids"] = ["mem-1", "mem-2"]
        if row.get("type") == "METHOD_OUTCOME_OBSERVED":
            row["attribution"] = "shadow_counterfactual"
            row["activity_id"] = counterfactual_activity
        if row.get("type") == "METHOD_VALUE_UPDATED":
            row["evidence_kind"] = "shadow_counterfactual"
            row["activity_id"] = counterfactual_activity

    def _row(kind: str, suffix: str, **fields) -> dict:
        return {
            "time": "Y2020-W02",
            "type": kind,
            "idempotency_key": suffix,
            "ledger_event_id": f"ev-hint-{suffix}",
            "schema_version": 1,
            **fields,
        }

    rows += [
        _row(
            "METHOD_HINTED",
            "offered-1",
            method_id="method-idea-1",
            week="Y2020-W02",
            role="exploit",
            score=0.0,
            block_chars=600,
        ),
        _row(
            "METHOD_HINTED",
            "offered-2",
            method_id="method-practice-1",
            week="Y2020-W02",
            role="explore",
            score=0.0,
            block_chars=600,
        ),
        _row(
            "METHOD_SELECTED",
            "adopted-1",
            method_id="method-idea-1",
            activity_id=activity_id,
            week="Y2020-W02",
            role="primary",
            source="hint",
            adoption=True,
            matched_by="title",
            context_key="social",
            shadow=False,
        ),
        _row(
            "METHOD_APPLIED",
            "applied-1",
            method_id="method-idea-1",
            activity_id=activity_id,
            week="Y2020-W02",
            source="hint",
            binding="skill_or_context_match",
        ),
        _row(
            "METHOD_OUTCOME_OBSERVED",
            "real-1",
            method_id="method-idea-1",
            activity_id=activity_id,
            week="Y2020-W02",
            reward=0.3,
            attribution="real_adoption",
        ),
        _row(
            "METHOD_VALUE_UPDATED",
            "real-1",
            method_id="method-idea-1",
            activity_id=activity_id,
            week="Y2020-W02",
            global_value=0.1,
            practice_count=1,
            success_count=1,
            evidence_kind="real_adoption",
        ),
    ]
    if leak:
        # A method that moved without any practice behind it, plus the same
        # activity feeding the counterfactual estimate as well.
        rows.append(
            _row(
                "METHOD_VALUE_UPDATED",
                "leak-1",
                method_id="method-practice-1",
                activity_id="",
                week="Y2020-W03",
                global_value=0.2,
                practice_count=1,
                evidence_kind="real_adoption",
            )
        )
        rows.append(
            _row(
                "METHOD_OUTCOME_OBSERVED",
                "leak-2",
                method_id="method-idea-1",
                activity_id=activity_id,
                week="Y2020-W02",
                reward=0.1,
                attribution="shadow_counterfactual",
            )
        )
    _write_jsonl(cap_path, rows)

    mem_path = persona / "cognition" / "memory_events.jsonl"
    mem_rows = [
        json.loads(line) for line in mem_path.read_text(encoding="utf-8").splitlines() if line
    ]
    if not leak:
        mem_rows.append(
            {
                "time": "Y2020-W02",
                "type": "MEMORY_USED_IN_PLAN",
                "idempotency_key": "MEMORY_USED_IN_PLAN:mem-1:method-idea-1:Y2020-W02",
                "ledger_event_id": "ev-hint-used-1",
                "schema_version": 1,
                "memory_id": "mem-1",
                "method_id": "method-idea-1",
                "week": "Y2020-W02",
            }
        )
    _write_jsonl(mem_path, mem_rows)

    config_path = run / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["world"]["cognition"].update(
        {
            "method_hints": True,
            "method_hints_top_k": 3,
            "method_hints_max_chars": 1400,
            "method_hints_exploration_slots": 1,
        }
    )
    _write_json(config_path, config)


class Phase4AuditTests(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory(prefix="agentopia-audit-p4-")
        self.run_dir = _build_run(Path(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _audit(self, *, leak: bool = False) -> tuple[dict, dict]:
        _enable_hints_with_events(self.run_dir, leak=leak)
        report = audit_run(self.run_dir)
        persona = next(p for p in report["personas"] if p["persona"] == "甲")
        return report, persona

    def test_offer_adopt_practise_chain_is_measured(self) -> None:
        report, persona = self._audit()
        hints = persona["method_hints"]
        self.assertEqual(hints["offers"]["events"], 2)
        self.assertEqual(hints["offers"]["weeks"], 1)
        self.assertEqual(hints["offers"]["block_chars_max"], 600)
        self.assertEqual(hints["offers"]["roles"], {"exploit": 1, "explore": 1})
        self.assertEqual(hints["adoptions"]["events"], 1)
        self.assertEqual(hints["adoptions"]["adoption_rate"], 0.5)
        self.assertEqual(hints["adoptions"]["matched_by"], {"title": 1})
        self.assertEqual(hints["adoptions"]["unoffered"], [])
        self.assertEqual(hints["practice"]["applied"], 1)
        self.assertEqual(hints["practice"]["real_outcomes"], 1)
        self.assertEqual(hints["practice"]["real_share_of_adoptions"], 1.0)
        self.assertEqual(hints["practice"]["shadow_outcomes"], 1)
        self.assertEqual(hints["practice"]["free_reinforcement"], [])
        self.assertEqual(hints["practice"]["double_counted_activities"], [])
        self.assertEqual(hints["used_in_plan"]["events"], 1)
        self.assertEqual(hints["used_in_plan"]["adoptions_missing_used_in_plan"], [])

    def test_phase4_gates_pass_on_a_clean_chain(self) -> None:
        report, _ = self._audit()
        gates = report["acceptance_gates"]["gates"]
        for name in (
            "hint_offers_recorded",
            "hint_adoption_rate",
            "hint_adoption_practice_rate",
            "hint_evidence_leaks",
            "hint_block_within_budget",
        ):
            self.assertIn(name, gates)
            self.assertTrue(gates[name]["pass"], f"{name} should pass")
        self.assertNotIn("hint_evidence_leaks", report["acceptance_gates"]["failed"])

    def test_free_reinforcement_and_double_counting_are_caught(self) -> None:
        report, persona = self._audit(leak=True)
        hints = persona["method_hints"]
        self.assertEqual(hints["practice"]["free_reinforcement"], [("method-practice-1", "Y2020-W03")])
        self.assertEqual(hints["practice"]["real_updates_without_practice"][0]["method_id"], "method-practice-1")
        self.assertEqual(
            hints["practice"]["double_counted_activities"],
            ["solo-Y2020-W02-activity-D1-甲"],
        )
        gates = report["acceptance_gates"]["gates"]
        self.assertFalse(gates["hint_evidence_leaks"]["pass"])
        self.assertTrue(
            any("invariant #6" in flag or "counted twice" in flag for flag in report["red_flags"]),
            report["red_flags"],
        )

    def test_missing_used_in_plan_is_a_leak_for_adopted_methods(self) -> None:
        # The adoption happened, the method carries memories, but nothing marked
        # them as used -> KI-10 did not actually close.
        _enable_hints_with_events(self.run_dir)
        mem_path = self.run_dir / "persona" / "甲" / "cognition" / "memory_events.jsonl"
        rows = [
            json.loads(line)
            for line in mem_path.read_text(encoding="utf-8").splitlines()
            if line and "MEMORY_USED_IN_PLAN" not in line
        ]
        _write_jsonl(mem_path, rows)
        report = audit_run(self.run_dir)
        persona = next(p for p in report["personas"] if p["persona"] == "甲")
        self.assertEqual(
            persona["method_hints"]["used_in_plan"]["adoptions_missing_used_in_plan"],
            ["method-idea-1"],
        )
        self.assertFalse(report["acceptance_gates"]["gates"]["hint_evidence_leaks"]["pass"])

    def test_hint_gates_are_absent_when_hints_are_off(self) -> None:
        report = audit_run(self.run_dir)
        gates = report["acceptance_gates"]["gates"]
        self.assertFalse([name for name in gates if name.startswith("hint_")])

    def test_markdown_reports_the_hint_readings(self) -> None:
        from audit_cognition_run import render_markdown

        report, _ = self._audit()
        text = render_markdown(report)
        self.assertIn("method hints:", text)
        self.assertIn("hint adoptions:", text)
        self.assertIn("used-in-plan (KI-10):", text)


def _enable_phase_five(run: Path) -> None:
    """Add the phase-5 events: declines, a situation, shadow scores, lifecycle."""
    persona = run / "persona" / "甲"
    cap_path = persona / "cognition" / "capability_events.jsonl"
    rows = [
        json.loads(line) for line in cap_path.read_text(encoding="utf-8").splitlines() if line
    ]

    def _row(kind: str, suffix: str, **fields) -> dict:
        return {
            "time": "Y2020-W03",
            "type": kind,
            "idempotency_key": suffix,
            "ledger_event_id": f"ev-p5-{suffix}",
            "schema_version": 1,
            **fields,
        }

    for row in rows:
        if row.get("type") == "METHOD_HINTED":
            row.update(
                {
                    "context_key": "learning+solo",
                    "context_tags": ["solo", "learning"],
                    "goal": "把小说写完",
                    "ignored_streak": 0,
                    "bandit": True,
                    "score_parts": {"value": 0.2, "explore": 0.1},
                }
            )
        if row.get("type") == "METHOD_OUTCOME_OBSERVED":
            row.setdefault("activity_type", "solo")
            row.update(
                {
                    "shadow_reward": -0.18,
                    "shadow_baseline": 0.22,
                    "shadow_baseline_samples": 4,
                    "shadow_quality_delta": -0.5,
                    "components": {"skill_gain": 0.0, "transferability": 0.0},
                    # The audit recomputes the centred score from these, so the
                    # fixture has to carry the baseline-independent terms.
                    "shadow_components": {
                        "skill_gain": 0.0,
                        "quality": 0.28,
                        "transferability": 0.0,
                        "time_cost": 0.25,
                        "money_cost": 0.0,
                        "vitality_cost": 0.4,
                        "social_cost": 0.0,
                    },
                }
            )

    rows += [
        _row(
            "METHOD_DECLINED",
            "declined-1",
            method_id="method-practice-1",
            week="Y2020-W02",
            reason="chose_other",
            role="explore",
            context_key="learning+solo",
        ),
        _row(
            "METHOD_ARCHIVED",
            "archived-1",
            method_id="method-practice-1",
            week="Y2020-W03",
            reason="failing_and_unused_for_7_weeks",
            global_value=-0.4,
            practice_count=4,
        ),
        _row(
            "METHOD_REFINED",
            "refined-1",
            method_id="method-idea-1",
            week="Y2020-W03",
            version=2,
            parent_method_id="method-idea-1",
            title="换了个说法的同一个方法",
        ),
    ]
    _write_jsonl(cap_path, rows)

    config_path = run / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["world"]["cognition"].update(
        {
            "method_hints": True,
            "method_hints_bandit": True,
            "method_lifecycle": True,
        }
    )
    _write_json(config_path, config)


class Phase5AuditTests(unittest.TestCase):
    """阶段 5 读数：拒绝的另一半、情境签名、KI-8 基线影子、生命周期。"""

    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory(prefix="agentopia-audit-p5-")
        self.run_dir = _build_run(Path(self._tmp.name))
        _enable_hints_with_events(self.run_dir)
        _enable_phase_five(self.run_dir)
        self.report = audit_run(self.run_dir)
        self.persona = next(p for p in self.report["personas"] if p["persona"] == "甲")
        self.choices = self.persona["method_choices"]

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_declines_are_reported_with_their_reason_and_role(self) -> None:
        declines = self.choices["declines"]
        self.assertEqual(declines["events"], 1)
        self.assertEqual(declines["reasons"], {"chose_other": 1})
        self.assertEqual(declines["by_role"], {"explore": 1})

    def test_the_offer_situation_is_reported(self) -> None:
        situation = self.choices["situation"]
        self.assertEqual(situation["distinct_context_keys"], 1)
        self.assertEqual(situation["context_keys"], {"learning+solo": 2})
        self.assertEqual(situation["offers_with_goal"], 2)
        self.assertEqual(situation["tag_counts"], {"solo": 2, "learning": 2})

    def test_the_bandit_readings_are_reported(self) -> None:
        bandit = self.choices["bandit"]
        self.assertTrue(bandit["enabled"])
        self.assertEqual(bandit["explore_offers"], 1)
        self.assertEqual(bandit["weeks_with_a_new_lead"], 1)

    def test_the_shadow_baseline_distribution_is_read_off_the_run(self) -> None:
        baseline = self.choices["reward_baseline_shadow"]
        self.assertTrue(baseline["available"])
        # both recorded outcomes carry the centred score, and both are negative
        self.assertEqual(baseline["shadow"]["negative"], 2)
        self.assertEqual(baseline["shadow"]["positive_share"], 0.0)
        self.assertIn("projected_status_transitions", baseline)
        self.assertIn("ranking_spearman", baseline)

    def test_the_lifecycle_readings_are_reported(self) -> None:
        lifecycle = self.choices["lifecycle"]
        self.assertEqual(lifecycle["archived"]["events"], 1)
        self.assertEqual(lifecycle["refined"]["events"], 1)
        self.assertEqual(lifecycle["versioned_methods"], {"method-idea-1": 2})
        self.assertEqual(lifecycle["by_status"].get("archived"), 1)

    def test_the_phase_five_gates_are_present(self) -> None:
        gates = self.report["acceptance_gates"]["gates"]
        for name in (
            "declines_recorded",
            "reward_baseline_shadow_available",
            "reward_baseline_shadow_signs",
            "method_lifecycle_reachable",
        ):
            self.assertIn(name, gates)

    def test_the_decline_gate_is_absent_when_hints_are_off(self) -> None:
        run = _build_run(Path(self._tmp.name) / "second")
        report = audit_run(run)
        self.assertNotIn("declines_recorded", report["acceptance_gates"]["gates"])

    def test_markdown_reports_the_phase_five_readings(self) -> None:
        from audit_cognition_run import render_markdown

        text = render_markdown(self.report)
        self.assertIn("menu answer (phase 5):", text)
        self.assertIn("KI-8 shadow baseline:", text)
        self.assertIn("KI-8 projection:", text)


class AdoptionSlotTests(unittest.TestCase):
    """`adoptions_by_slot` 必须把采用对齐到**当周提供**的槽位。

    旧实现去读 `METHOD_SELECTED` 的 `role` 字段——采用事件里根本没有槽位
    （那里写的是 `role="primary"`），于是 `explore_adoptions` 恒为 0，两轮验收
    与一次选法改动都建立在这个错误读数上。运行 09272220 的真实读数是
    exploit 21/57、explore 12/43、filler 21/69。
    """

    def test_the_slot_comes_from_the_offer_not_from_the_adoption(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="agentopia-audit-slot-") as tmp:
            run = _build_run(Path(tmp))
            _enable_hints_with_events(run)
            cap = run / "persona" / "甲" / "cognition" / "capability_events.jsonl"
            rows = [
                json.loads(line)
                for line in cap.read_text(encoding="utf-8").splitlines()
                if line
            ]
            for row in rows:
                if (
                    row.get("type") == "METHOD_HINTED"
                    and row.get("method_id") == "method-idea-1"
                ):
                    row["role"] = "explore"
            _write_jsonl(cap, rows)

            report = audit_run(run)
            entry = next(p for p in report["personas"] if p["persona"] == "甲")
            bandit = (entry.get("method_choices") or {}).get("bandit") or {}
            self.assertEqual(bandit["explore_adoptions"], 1)
            self.assertEqual(bandit["adoptions_by_slot"]["explore"]["adoptions"], 1)
            # both offered methods now sit in the explore slot; the point is that
            # the *adoption* is attributed to the slot it was offered in
            self.assertEqual(bandit["adoptions_by_slot"]["explore"]["offers"], 2)
            self.assertEqual(bandit["adoptions_by_slot"]["explore"]["rate"], 0.5)



class Phase6AuditTests(unittest.TestCase):
    """阶段 6 读数：能力投影必须真的按世界的能力族归并。

    这里钉的是一个**已经发生过的错误读数**：审计用一个只有 `root` / `char` 的
    替身对象去建投影，`SkillFamilies.from_data_manager` 拿不到 `world` 就静默退回
    "没有家族"，于是 `folded` 与 `unfolded` 两列恒等——看起来像"归并没有作用"，
    实际是审计根本没归并。合成运行目录里带上能力族文件，两列就必须分开。
    """

    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory(prefix="agentopia-audit-p6-")
        self.run_dir = _build_run(Path(self._tmp.name))
        persona = self.run_dir / "persona" / "甲"
        _write_jsonl(
            persona / "activity.jsonl",
            [
                _gain("Y2020-W01-activity-D1", {"长跑": 2}),
                _gain("Y2020-W02-activity-D1", {"长跑耐力": 2}),
                _gain("Y2020-W03-activity-D1", {"跑步": 2}),
                _gain("Y2020-W04-activity-D1", {"写作": 2}),
            ],
        )
        _write_json(
            self.run_dir / "skill_aliases.json",
            {
                "version": 1,
                "skills": [{"canonical": "长跑", "aliases": ["长跑耐力", "跑步"]}],
            },
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _projection(self) -> dict:
        report = audit_run(self.run_dir)
        entry = next(p for p in report["personas"] if p["persona"] == "甲")
        return entry["proficiency_projection"]

    def test_the_projection_really_folds_when_the_run_has_alias_families(self) -> None:
        folding = self._projection()["family_folding"]
        self.assertTrue(folding["available"])
        self.assertEqual(folding["unfolded_families"], 4)
        self.assertEqual(folding["folded_families"], 2)
        self.assertEqual(folding["merged_family_count"], 1)
        self.assertEqual(folding["names_folded_away"], 2)

    def test_the_two_columns_agree_when_there_is_nothing_to_fold(self) -> None:
        (self.run_dir / "skill_aliases.json").unlink()
        folding = self._projection()["family_folding"]
        self.assertEqual(folding["folded_families"], folding["unfolded_families"])
        self.assertEqual(folding["merged_family_count"], 0)

    def test_both_scales_are_reported_for_every_ranked_skill(self) -> None:
        top = self._projection()["top_by_capability"]
        self.assertTrue(top)
        for entry in top:
            self.assertIsNotNone(entry["proficiency_points"])
            self.assertIsNotNone(entry["effective_capability_points"])

    def test_the_unmapped_sentinel_is_not_counted_as_a_skill(self) -> None:
        cap = self.run_dir / "persona" / "甲" / "cognition" / "capability_events.jsonl"
        rows = [
            json.loads(line) for line in cap.read_text(encoding="utf-8").splitlines() if line
        ]
        rows.append(
            {
                "time": "Y2020-W03-review",
                "type": "METHOD_PROPOSED",
                "idempotency_key": "METHOD_PROPOSED:method-sentinel:-:Y2020-W03",
                "ledger_event_id": "ev-sentinel",
                "schema_version": 1,
                "method_id": "method-sentinel",
                "skill_id": "unmapped",
                "title": "没有技能可挂的方法",
            }
        )
        _write_jsonl(cap, rows)
        report = audit_run(self.run_dir)
        fragmentation = report["skill_fragmentation"]
        self.assertEqual(fragmentation["unmapped_sentinel_methods"], 1)
        self.assertNotIn("unmapped", fragmentation["unmapped_skills"])
        entry = next(p for p in report["personas"] if p["persona"] == "甲")
        self.assertNotIn("unmapped", entry["proficiency_projection"]["stats"].get(
            "merged_families", {}
        ))


def _gain(time: str, gains: dict) -> dict:
    """一条带 God 发放的技能增益的活动记录（投影只认这个）。"""
    return {
        "time": time,
        "type": "solo",
        "outcome": {"delta_skills": gains},
        "ledger_event_id": f"ev-{time}",
        "schema_version": 1,
    }


if __name__ == "__main__":
    unittest.main()
