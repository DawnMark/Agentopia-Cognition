"""阶段 4.5：教训桥——角色自己写下的教训进入认知层（离线，无真实 LLM 调用）。

这一层要证明四件事：

1. **读得到**：周记的 Reflection 与活动反思真的进了记忆抽取的证据（旧代码把它们裁掉了：
   周记先截 400 字、再截 160 字，而反思从 330~390 字才开始，等于 0% 可见）；
2. **存得住**：角色笔记里的【教训】行按"快照差分 + 改写去重 + 消失追踪"变成 `belief`/`lesson`
   记忆，内容逐字保留、来源可追到快照的账本身份、置信度低于客观观察；
3. **变成假设**：教训能生成 `lesson_application` 的 Idea 候选，且**不挤掉**原有的随机组合 motif；
4. **优先实践**：由教训长出来的方法在菜单里排在最前，并被标注"你自己在笔记里总结的教训"。
"""

from __future__ import annotations

import json
import unittest
from typing import Any, Dict, List
from unittest import mock

from src.agents.cognition.consolidator import Consolidator, MemoryConfig
from src.agents.cognition.idea_engine import find_motif_candidates
from src.agents.cognition.idea_models import IDEA_MOTIFS, Idea, IdeaCandidate
from src.agents.cognition.lessons import (
    LessonConfig,
    bigram_jaccard,
    is_same_lesson,
    classify_lessons,
    collect_lesson_diff,
    derive_polarity,
    derive_topics,
    extract_lesson_lines,
    lesson_disambiguator,
    read_scratchpad_snapshots,
)
from src.agents.cognition.memory_models import MemoryItem, memory_id_for, semantic_key
from src.agents.cognition.memory_views import build_memory_views
from src.agents.cognition.method_hints import HintConfig, MethodHintProvider, origin_of
from src.agents.cognition.models import Methodology
from src.agents.cognition.prompts import format_memory_evidence, split_diary
from src.world.clock import Stage
from tests._helpers import make_datamanager, temp_workspace


def _state() -> Dict[str, Any]:
    return {
        "vitality": 80,
        "fulfillment": {"mood": 50, "material": 50, "social": 50, "esteem": 50},
        "assets": {"deposit": 1000, "possessions": []},
        "skills": {"写作": 12, "跑步": 30, "情报搜集与反侦察": 20},
    }


def _notes(*lessons: str, header: str = "【教训与注意】") -> str:
    body = "\n".join(f"- {line}" for line in lessons)
    return f"【长期目标】\n- 写一本小说\n\n{header}\n{body}\n\n【生活标准】moderate"


class LessonParsingTests(unittest.TestCase):
    def test_lines_under_a_lesson_header_are_extracted(self) -> None:
        text = _notes("谈不到就当天排备选市内活", "出门必锁门")
        self.assertEqual(
            extract_lesson_lines(text),
            ["谈不到就当天排备选市内活", "出门必锁门"],
        )

    def test_other_sections_are_ignored(self) -> None:
        text = "【长期目标】\n- 写一本小说\n\n【本周计划】\n- 去图书馆\n"
        self.assertEqual(extract_lesson_lines(text), [])

    def test_header_variants_and_bullet_styles(self) -> None:
        for header in ("【教训/提醒】", "【反思与教训】", "**教训**", "## Lessons"):
            text = f"{header}\n1. 先定锚点\n• 别贪多\n"
            self.assertEqual(
                extract_lesson_lines(text),
                ["先定锚点", "别贪多"],
                msg=header,
            )

    def test_reworded_lessons_are_recognised(self) -> None:
        # Measured on the acceptance run: the character rewords its lessons
        # every week; this pair scores 0.38 Jaccard / 0.57 containment.
        a = "谈不到城际活就当天排备选市内活"
        b = "谈不到城际的活，就当天接市内兜底活"
        self.assertGreaterEqual(bigram_jaccard(a, b), 0.35)
        self.assertTrue(is_same_lesson(a, b))

    def test_unrelated_lessons_are_not_similar(self) -> None:
        self.assertFalse(is_same_lesson("出门必锁门", "谈不到就接市内活"))
        self.assertFalse(is_same_lesson("新的一条：白天留给接活", "旧的一条"))
        self.assertFalse(is_same_lesson("白天留给接活和练手", "娱乐排晚间"))


class LessonDiffTests(unittest.TestCase):
    def test_new_rewritten_carried_and_dropped(self) -> None:
        previous = ["出门必锁门", "谈不到城际活就当天排备选市内活", "旧的一条"]
        current = [
            "出门必锁门",
            "谈不到城际的活，就当天接市内兜底活",
            "新的一条：白天留给接活",
        ]
        diff = classify_lessons(current, previous)
        self.assertEqual([line.text for line in diff.new], ["新的一条：白天留给接活"])
        self.assertEqual(len(diff.rewritten), 1)
        self.assertEqual([line.text for line in diff.carried], ["出门必锁门"])
        self.assertEqual(diff.dropped, ["旧的一条"])

    def test_snapshots_are_read_from_the_scratchpad_streams(self) -> None:
        with temp_workspace() as root:
            dm, _clock = make_datamanager()
            path = dm.root / "memory" / "scratchpad" / "general.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            rows = [
                {
                    "time": "Y2020-W01-review",
                    "ledger_event_id": "ev-1",
                    "schema_version": 1,
                    "content": _notes("出门必锁门"),
                },
                {
                    "time": "Y2020-W02-review",
                    "ledger_event_id": "ev-2",
                    "schema_version": 1,
                    "content": _notes("出门必锁门", "白天留给接活"),
                },
            ]
            path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
                encoding="utf-8",
            )
            snapshots = read_scratchpad_snapshots(dm)
            self.assertEqual([s[1] for s in snapshots], ["Y2020-W01-review", "Y2020-W02-review"])
            diff = collect_lesson_diff(dm, LessonConfig(enabled=True))
            self.assertEqual([line.text for line in diff.new], ["白天留给接活"])
            self.assertEqual(diff.snapshot_event_id, "ev-2")
            self.assertEqual(diff.snapshot_time, "Y2020-W02-review")


class LessonIngestTests(unittest.TestCase):
    def setUp(self) -> None:
        self._ctx = temp_workspace()
        self.root = self._ctx.__enter__()  # type: ignore[attr-defined]
        self.dm, self.clock = make_datamanager()
        self.dm.save_state(_state())
        self.clock.set_stage(Stage.REVIEW)

    def tearDown(self) -> None:
        self._ctx.__exit__(None, None, None)  # type: ignore[attr-defined]

    def _write_notes(self, *snapshots: tuple[str, str, List[str]]) -> None:
        path = self.dm.root / "memory" / "scratchpad" / "general.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "time": time_str,
                "ledger_event_id": event_id,
                "schema_version": 1,
                "content": _notes(*lessons),
            }
            for time_str, event_id, lessons in snapshots
        ]
        path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8"
        )

    def _consolidator(self, **overrides) -> Consolidator:
        cfg = MemoryConfig(enabled=overrides.pop("enabled", True))
        return Consolidator(
            dm=self.dm,
            clock=self.clock,
            agent_name=self.dm.char,
            model="regression-model",
            config=cfg,
            language="zh",
            world_cognition={
                "lesson_ingest": overrides.pop("lesson_ingest", True),
                "lesson_max_new_per_week": overrides.pop("max_new", 6),
            },
        )

    def _lessons(self) -> List[Dict[str, Any]]:
        view = build_memory_views(self.dm)["memories"]
        return [
            m
            for m in view["memories"].values()
            if str(m.get("origin")) == "scratchpad_lesson"
        ]

    def test_disabled_ingestion_writes_nothing(self) -> None:
        self._write_notes(("Y2020-W01-review", "ev-1", ["谈不到就接市内活"]))
        cons = self._consolidator(lesson_ingest=False)
        counts = cons.ingest_lessons()
        self.assertEqual(counts["lessons_created"], 0)
        self.assertEqual(self._lessons(), [])

    def test_new_lesson_becomes_a_citable_memory(self) -> None:
        self._write_notes(
            ("Y2020-W01-review", "ev-1", ["出门必锁门"]),
            ("Y2020-W02-review", "ev-2", ["出门必锁门", "谈不到城际活就当天排备选市内活"]),
        )
        counts = self._consolidator().ingest_lessons()
        self.assertEqual(counts["lessons_created"], 1)
        lessons = self._lessons()
        self.assertEqual(len(lessons), 1)
        lesson = lessons[0]
        # The character's own words, verbatim, traceable to the snapshot.
        self.assertEqual(lesson["content"], "谈不到城际活就当天排备选市内活")
        self.assertEqual(lesson["kind"], "lesson")
        self.assertEqual(lesson["source_event_ids"], ["ev-2"])
        self.assertEqual(lesson["origin"], "scratchpad_lesson")
        self.assertEqual(lesson["provenance"], "scratchpad:general.jsonl#ev-2")
        # A belief, not an observation: above the idea engine's floor (0.4) but
        # below a measured outcome.
        self.assertGreaterEqual(lesson["confidence"], 0.4)
        self.assertLess(lesson["confidence"], 0.8)
        # A procedural lesson stays neutral: only "don't do X" wording is
        # negative, so lessons never manufacture contradictions by accident.
        self.assertEqual(lesson["outcome_polarity"], "neutral")
        self.assertIn("planning", lesson["topics"])

    def test_ingestion_is_idempotent(self) -> None:
        self._write_notes(
            ("Y2020-W01-review", "ev-1", ["出门必锁门"]),
            ("Y2020-W02-review", "ev-2", ["出门必锁门", "白天留给接活"]),
        )
        cons = self._consolidator()
        first = cons.ingest_lessons()
        second = cons.ingest_lessons()
        self.assertEqual(first["lessons_created"], 1)
        self.assertEqual(second["lessons_created"], 0)
        self.assertEqual(len(self._lessons()), 1)

    def test_reworded_lesson_is_not_stored_twice(self) -> None:
        self._write_notes(
            ("Y2020-W01-review", "ev-1", ["谈不到城际活就当天排备选市内活"]),
            ("Y2020-W02-review", "ev-2", ["谈不到城际的活，就当天接市内兜底活"]),
        )
        counts = self._consolidator().ingest_lessons()
        self.assertEqual(counts["lessons_created"], 0)
        self.assertEqual(counts["lessons_rewritten"], 1)
        self.assertEqual(self._lessons(), [])

    def test_two_lessons_with_one_anchor_stay_two_memories(self) -> None:
        self._write_notes(
            ("Y2020-W01-review", "ev-1", ["出门必锁门"]),
            (
                "Y2020-W02-review",
                "ev-2",
                ["出门必锁门", "白天留给接活和练手", "娱乐排晚间"],
            ),
        )
        self.assertEqual(self._consolidator().ingest_lessons()["lessons_created"], 2)
        contents = sorted(lesson["content"] for lesson in self._lessons())
        self.assertEqual(sorted(["白天留给接活和练手", "娱乐排晚间"]), contents)

    def test_a_lesson_without_a_ledger_id_is_rejected(self) -> None:
        self._write_notes(
            ("Y2020-W01-review", "", ["出门必锁门"]),
            ("Y2020-W02-review", "", ["出门必锁门", "白天留给接活"]),
        )
        counts = self._consolidator().ingest_lessons()
        self.assertEqual(counts["lessons_created"], 0)
        self.assertEqual(counts["lessons_rejected"], 1)

    def test_lessons_are_tagged_deterministically(self) -> None:
        self.assertEqual(derive_topics("接活前先绕外围走一圈"), ["work"])
        self.assertIn("exercise", derive_topics("白天留给接活和练手"))
        self.assertEqual(derive_polarity("别再贪多"), "negative")
        self.assertEqual(derive_polarity("白天留给接活"), "neutral")
        self.assertNotEqual(
            lesson_disambiguator(polarity="neutral", content="a"),
            lesson_disambiguator(polarity="neutral", content="b"),
        )


class EvidencePipelineTests(unittest.TestCase):
    """The bug that made this whole phase necessary: reflections were cut off."""

    def setUp(self) -> None:
        self._ctx = temp_workspace()
        self.root = self._ctx.__enter__()  # type: ignore[attr-defined]
        self.dm, self.clock = make_datamanager()
        self.dm.save_state(_state())
        self.clock.set_stage(Stage.REVIEW)

    def tearDown(self) -> None:
        self._ctx.__exit__(None, None, None)  # type: ignore[attr-defined]

    def _consolidator(self) -> Consolidator:
        return Consolidator(
            dm=self.dm,
            clock=self.clock,
            agent_name=self.dm.char,
            model="regression-model",
            config=MemoryConfig(enabled=True),
            language="zh",
        )

    def _add_activity(self, time_str: str, reflection: str) -> str:
        from src.world.solo_activity_data import ActionOutcome, SoloActivityRecord
        from src.world.clock import TimeState

        record = SoloActivityRecord(
            activity_id=f"solo-{time_str}",
            agent_name=self.dm.char,
            time=TimeState.from_string(time_str),
            content="去图书馆查资料",
            outcome=ActionOutcome(outcome="查到两条线索", delta_vitality=-1, delta_skills={"写作": 1}),
        )
        record.reflection = reflection
        self.dm.append_activity_record(record)
        rows = [
            json.loads(line)
            for line in (self.dm.root / "activity.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return str(rows[-1]["ledger_event_id"])

    def _write_diary(self, summary: str, reflection: str) -> str:
        path = self.dm.memory_root / "weekly_diary.jsonl" if hasattr(self.dm, "memory_root") else None
        if path is None:  # pragma: no cover - DataManager exposes weekly_diary
            path = self.dm.weekly_diary
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "time": "Y2020-W01-review",
            "ledger_event_id": "ev-diary",
            "schema_version": 1,
            "content": f"Summary: {summary}\n\nReflection: {reflection}",
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return "ev-diary"

    def test_diary_reflection_reaches_the_extraction_prompt(self) -> None:
        summary = "这周把停太久的接活线接上了。" * 12  # long enough to have hidden it before
        reflection = "最该记的教训：谈不到城际活，就当天接市内兜底活。"
        self._write_diary(summary, reflection)
        evidence = self._consolidator().week_evidence()
        types = [item["text"] for item in evidence]
        self.assertTrue(
            any("weekly_review_reflection" in text for text in types),
            types,
        )
        self.assertTrue(
            any("谈不到城际活" in text for text in types),
            "the reflection text itself must be visible",
        )
        reflection_item = next(t for t in types if "weekly_review_reflection" in t)
        self.assertIn("当天接市内兜底活", reflection_item)

    def test_activity_reflection_is_its_own_citable_item(self) -> None:
        event_id = self._add_activity(
            "Y2020-W01-activity-D2", "这活干得干净：外围走一圈、不掏本子。"
        )
        evidence = self._consolidator().week_evidence()
        objective = [i for i in evidence if "did:" in i["text"]]
        reflective = [i for i in evidence if "reflection:" in i["text"]]
        self.assertEqual(len(objective), 1)
        self.assertEqual(len(reflective), 1)
        self.assertEqual(reflective[0]["event_id"], event_id)
        self.assertIn("外围走一圈", reflective[0]["text"])
        self.assertIn("reflection(solo)", reflective[0]["text"])

    def test_reflections_are_labelled_and_bounded(self) -> None:
        self._add_activity("Y2020-W01-activity-D2", "短反思" * 200)
        evidence = self._consolidator().week_evidence()
        item = next(i for i in evidence if "reflection:" in i["text"])
        self.assertLessEqual(len(item["text"]), 420 + 60)
        self.assertLessEqual(sum(len(i["text"]) for i in evidence), 4400)

    def test_split_diary_handles_both_halves(self) -> None:
        summary, reflection = split_diary("Summary: 甲\n\nReflection: 乙")
        self.assertEqual(summary, "甲")
        self.assertEqual(reflection, "乙")
        self.assertEqual(split_diary("没有标记的周记"), ("没有标记的周记", ""))


class LessonIdeaTests(unittest.TestCase):
    def _memory(self, **overrides) -> MemoryItem:
        payload: Dict[str, Any] = {
            "memory_id": "mem-x",
            "kind": "episodic",
            "content": "去图书馆查了资料",
            "persona": "甲",
            "source_event_ids": ["ev-x"],
            "confidence": 0.6,
            "salience": 0.6,
            "strength": 0.6,
            "topics": ["work"],
            "entities": ["图书馆"],
        }
        payload.update(overrides)
        return MemoryItem(**payload)

    def test_lesson_pairs_with_a_goal_into_a_candidate(self) -> None:
        lesson = self._memory(
            memory_id="mem-lesson-1",
            kind="lesson",
            content="谈不到城际活就当天排备选市内活",
            origin="scratchpad_lesson",
            topics=["work"],
            entities=["事务所"],
            obstacles=["城际线铺不动"],
        )
        goal = self._memory(
            memory_id="mem-goal-1",
            kind="goal",
            content="把接活的渠道重新盘活",
            topics=["work"],
            entities=["事务所"],
        )
        candidates = find_motif_candidates([lesson, goal], [], traits={"creativity": 50})
        motifs = [c.motif for c in candidates]
        self.assertIn("lesson_application", motifs)
        candidate = next(c for c in candidates if c.motif == "lesson_application")
        self.assertEqual(sorted(candidate.memory_ids), ["mem-goal-1", "mem-lesson-1"])
        self.assertIn("谈不到城际活", candidate.hint)
        self.assertEqual(candidate.goal, "把接活的渠道重新盘活")

    def test_lessons_do_not_replace_the_random_combination_motifs(self) -> None:
        lesson = self._memory(
            memory_id="mem-lesson-1",
            kind="lesson",
            content="出门必锁门",
            origin="scratchpad_lesson",
        )
        a = self._memory(memory_id="mem-a", topics=["friendship"])
        b = self._memory(memory_id="mem-b", topics=["friendship"])
        edges = [
            {
                "source": "mem-a",
                "target": "mem-b",
                "types": ["contradiction"],
                "score": 0.8,
                "shared_focus": "朋友",
            }
        ]
        motifs = {c.motif for c in find_motif_candidates([lesson, a, b], edges)}
        self.assertIn("contradiction", motifs)

    def test_lesson_motif_is_part_of_the_closed_vocabulary(self) -> None:
        self.assertIn("lesson_application", IDEA_MOTIFS)
        idea = Idea(
            idea_id="idea-1",
            motif="lesson_application",
            content="把这条教训变成这周的一件事",
            persona="甲",
        )
        self.assertEqual(idea.motif, "lesson_application")

    def test_a_lesson_keeps_one_phrasing_slot(self) -> None:
        """Pattern motifs carry more sources and would always outrank a lesson."""
        lesson = self._memory(
            memory_id="mem-lesson-1",
            kind="lesson",
            content="谈不到城际活就当天排备选市内活",
            origin="scratchpad_lesson",
            topics=["work"],
        )
        goal = self._memory(memory_id="mem-goal-1", kind="goal", content="把接活渠道盘活", topics=["work"])
        pattern = [
            self._memory(memory_id=f"mem-p{i}", topics=["friendship"]) for i in range(4)
        ]
        edges = [
            {
                "source": "mem-p0",
                "target": "mem-p1",
                "types": ["repeated_pattern"],
                "score": 0.9,
                "shared_focus": "朋友",
            },
            {
                "source": "mem-p2",
                "target": "mem-p3",
                "types": ["repeated_pattern"],
                "score": 0.9,
                "shared_focus": "朋友",
            },
            {
                "source": "mem-p0",
                "target": "mem-p2",
                "types": ["contradiction"],
                "score": 0.9,
                "shared_focus": "朋友",
            },
        ]
        candidates = find_motif_candidates(
            [lesson, goal, *pattern], edges, traits={"creativity": 60, "curiosity": 60}
        )
        motifs = [c.motif for c in candidates]
        self.assertIn("lesson_application", motifs)
        # The random-combination motifs keep their share of the slots.
        self.assertGreaterEqual(motifs.count("repeated_pattern") + motifs.count("contradiction"), 2)

    def test_lesson_candidates_outrank_on_their_own(self) -> None:
        """The prior nudge is what lets a two-source lesson compete at all."""
        lesson = self._memory(
            memory_id="mem-lesson-1",
            kind="lesson",
            content="谈不到城际活就当天排备选市内活",
            origin="scratchpad_lesson",
            topics=["work"],
        )
        goal = self._memory(memory_id="mem-goal-1", kind="goal", content="把接活渠道盘活", topics=["work"])
        candidate = next(
            c
            for c in find_motif_candidates([lesson, goal], [], traits={"creativity": 60})
            if c.motif == "lesson_application"
        )
        from src.agents.cognition.idea_engine import _by_id, _candidate_prior

        index = _by_id([lesson, goal])
        boosted = _candidate_prior(candidate, index, {"creativity": 60})
        candidate.motif = "unused_resource"
        plain = _candidate_prior(candidate, index, {"creativity": 60})
        self.assertGreater(boosted, plain * 1.2)

    def test_a_lesson_without_a_partner_makes_no_candidate(self) -> None:
        lesson = self._memory(
            memory_id="mem-lesson-1",
            kind="lesson",
            content="出门必锁门",
            origin="scratchpad_lesson",
            topics=[],
            entities=[],
        )
        self.assertEqual(
            [c.motif for c in find_motif_candidates([lesson], [])],
            [],
        )


class LessonPracticePriorityTests(unittest.TestCase):
    def setUp(self) -> None:
        self._ctx = temp_workspace()
        self.root = self._ctx.__enter__()  # type: ignore[attr-defined]
        self.dm, self.clock = make_datamanager()
        self.dm.save_state(_state())
        self.clock.set_stage(Stage.PLAN)

    def tearDown(self) -> None:
        self._ctx.__exit__(None, None, None)  # type: ignore[attr-defined]

    def _seed(self, *methods: Methodology) -> None:
        from src.agents.cognition.event_store import CapabilityEventStore

        store = CapabilityEventStore(self.dm)
        for method in methods:
            payload = method.to_dict()
            method_id = payload.pop("method_id")
            store.append(
                "METHOD_PROPOSED",
                method_id=method_id,
                payload=payload,
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
                    "context_values": {},
                },
                idempotency_key=f"METHOD_VALUE_UPDATED:{method_id}:-:seed",
            )

    def _method(self, **overrides) -> Methodology:
        payload: Dict[str, Any] = {
            "method_id": "method-a",
            "skill_id": "写作",
            "title": "先定冲突再写场景",
            "description": "复杂写作任务中先确定目标与冲突",
            "status": "proposed",
            "global_value": 0.6,
            "confidence": 0.6,
            "practice_count": 4,
            "steps": ["明确目标", "列出冲突"],
            "checks": ["场景推动冲突"],
            "failure_modes": ["迟迟不开始"],
            "applicable_contexts": ["long_form"],
        }
        payload.update(overrides)
        return Methodology(**payload)

    def _provider(self) -> MethodHintProvider:
        return MethodHintProvider(
            dm=self.dm,
            clock=self.clock,
            agent_name=self.dm.char,
            traits={"creativity": 60, "curiosity": 60},
            config=HintConfig(enabled=True, top_k=3, max_chars=1400, exploration_slots=1),
            language="zh",
        )

    def test_origin_of_reads_the_source_motif(self) -> None:
        lesson_method = self._method(source_motif="lesson_application", source_type="idea_conversion")
        idea_method = self._method(source_motif="contradiction", source_type="idea_conversion")
        routine = self._method(source_motif=None, source_type="practice_reflection")
        self.assertEqual(origin_of(lesson_method), "own_lesson")
        self.assertEqual(origin_of(idea_method), "own_idea")
        self.assertEqual(origin_of(routine), "practice")

    def test_an_untried_lesson_method_leads_the_menu(self) -> None:
        self._seed(
            self._method(
                method_id="method-strong",
                title="先定冲突再写场景",
                global_value=0.9,
                confidence=0.9,
                practice_count=8,
                status="validated",
            ),
            self._method(
                method_id="method-lesson",
                title="谈不到就当天接市内活",
                skill_id="写作",
                global_value=0.0,
                confidence=0.0,
                practice_count=0,
                status="proposed",
                source_type="idea_conversion",
                source_motif="lesson_application",
            ),
        )
        provider = self._provider()
        offers = provider.select_offers()
        self.assertEqual(offers[0].method_id, "method-lesson")
        self.assertEqual(offers[0].origin, "own_lesson")
        block = provider.render(offers)
        self.assertIn("你自己在笔记里总结的教训", block)
        for forbidden in ("你必须", "你应该采用", "务必"):
            self.assertNotIn(forbidden, block)

    def test_a_practised_lesson_method_still_outranks_plain_methods(self) -> None:
        self._seed(
            self._method(
                method_id="method-lesson",
                title="谈不到就当天接市内活",
                global_value=0.1,
                confidence=0.2,
                practice_count=2,
                status="tested",
                source_type="idea_conversion",
                source_motif="lesson_application",
            ),
            self._method(
                method_id="method-strong",
                title="先定冲突再写场景",
                global_value=0.5,
                confidence=0.6,
                practice_count=3,
                status="tested",
            ),
        )
        provider = self._provider()
        offers = provider.select_offers()
        self.assertEqual(offers[0].method_id, "method-lesson")
        self.assertEqual(offers[0].origin, "own_lesson")

    def test_hinted_events_record_the_origin(self) -> None:
        self._seed(
            self._method(
                method_id="method-lesson",
                title="谈不到就当天接市内活",
                source_type="idea_conversion",
                source_motif="lesson_application",
            )
        )
        provider = self._provider()
        provider.prepare_week()
        events = [e for e in provider.capability.events() if e["type"] == "METHOD_HINTED"]
        self.assertEqual(events[0].get("origin"), "own_lesson")


if __name__ == "__main__":
    unittest.main()
