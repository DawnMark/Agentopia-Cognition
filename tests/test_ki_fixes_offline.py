"""离线回归：KI-2 ~ KI-12 的修复口径（不调用真实 LLM）。

覆盖本轮按用户批注实施的改动：

- KI-2  印象文档（别称词典）+ 实体归一 + soft_entities
- KI-3  关系规则：家族键、obstacle↔resource 匹配、gap
- KI-4  矛盾必须共享锚点
- KI-5  Idea 语义去重（同一洞见不再重生）+ IDEA_REFINED
- KI-6  idea→method "方法化"与结构不完整判定
- KI-7  可行性使用印象词典（软实体降级而不是硬拒）
- KI-9  探索真的在可探索集合里抽样
- KI-12 反思解析 fallback 取最后一段散文
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest import mock

from src.agents.cognition.idea_engine import (
    IdeaCandidate,
    IdeaWorldState,
    _semantic_duplicate,
    evaluate_candidate,
)
from src.agents.cognition.idea_models import Idea
from src.agents.cognition.idea_pipeline import _structure_gaps
from src.agents.cognition.impressions import (
    Impressions,
    build_seed,
    ensure_impressions,
    name_alias_candidates,
    normalize_name,
)
from src.agents.cognition.memory_models import (
    MemoryItem,
    _affinity,
    semantic_key,
    shared_anchor,
)
from src.agents.cognition.methodology_policy import (
    context_signature_from_activity,
    deterministic_rng,
    select_methods,
)
from src.agents.cognition.models import Methodology
from src.agents.cognition.prompts import parse_methodization_response
from src.agents.cognition.relation_graph import (
    REPEATED_PATTERN_MIN_FAMILY,
    InvertedIndex,
    _problem_resource_overlap,
    relation_types_between,
)
from src.agents.role_agent import _last_prose_block
from tests._helpers import make_datamanager, temp_workspace


def _memory(
    *,
    memory_id: str,
    content: str = "内容",
    topics: Optional[List[str]] = None,
    entities: Optional[List[str]] = None,
    goals: Optional[List[str]] = None,
    polarity: str = "positive",
    obstacles: Optional[List[str]] = None,
    resources: Optional[List[str]] = None,
) -> MemoryItem:
    return MemoryItem(
        memory_id=memory_id,
        kind="episodic",
        content=content,
        persona="测试角色",
        semantic_key=semantic_key(
            kind="episodic", topics=topics or ["work"], entities=entities or []
        ),
        topics=list(topics or ["work"]),
        entities=list(entities or []),
        goal_ids=list(goals or []),
        source_event_ids=["ev-1"],
        outcome_polarity=polarity,
        obstacles=list(obstacles or []),
        resources=list(resources or []),
    )


# ---------------------------------------------------------------------------
# KI-2 · impressions
# ---------------------------------------------------------------------------


class ImpressionsTests(unittest.TestCase):
    def test_short_name_candidates(self) -> None:
        self.assertIn("日和", name_alias_candidates("吉日和"))
        self.assertIn("霄月", name_alias_candidates("上官霄月"))
        self.assertIn("亦岚屋", name_alias_candidates("萧亦岚"))
        self.assertEqual(normalize_name(" 日 和！"), "日和")

    def test_seed_resolves_aliases_and_keeps_soft_entities(self) -> None:
        with temp_workspace() as tmp:
            world = tmp / "data" / "regression_world"
            for name in ("测试角色", "吉日和", "萧亦岚"):
                (world / "persona" / name).mkdir(parents=True, exist_ok=True)
            (world / "entity_aliases.json").write_text(
                json.dumps(
                    {
                        "entities": [
                            {
                                "canonical": "Vegetable_Market",
                                "kind": "place",
                                "aliases": ["菜市场", "菜场"],
                            },
                            {
                                "canonical": "吉日和",
                                "kind": "person",
                                "aliases": ["日和"],
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            dm, clock = make_datamanager()
            canonical, aliases = build_seed(dm, persona="测试角色")
            self.assertIn("吉日和", canonical["people"])
            self.assertIn("Vegetable_Market", canonical["places"])
            self.assertEqual(aliases[normalize_name("菜市场")], "Vegetable_Market")
            self.assertEqual(aliases[normalize_name("日和")], "吉日和")

            keeper, view = ensure_impressions(dm, clock, week="Y2020-W01")
            self.assertTrue(view.seeded)
            self.assertTrue(keeper.view_path.exists())

            resolved, unresolved = view.canonicalize(["菜场", "日和", "中介张哥"])
            self.assertEqual(sorted(resolved), ["Vegetable_Market", "吉日和"])
            self.assertEqual(unresolved, ["中介张哥"])

            # Containment still resolves a longer spoken form.
            self.assertEqual(view.resolve("九亭菜市场")[0], "Vegetable_Market")
            # Soft entities are remembered, not dropped.
            keeper.record_soft(names=["中介张哥"], week="Y2020-W01")
            keeper.invalidate()
            self.assertTrue(keeper.impressions().mentions("中介张哥"))

    def test_view_is_rebuildable_from_events(self) -> None:
        with temp_workspace():
            dm, clock = make_datamanager()
            keeper, _view = ensure_impressions(dm, clock, week="Y2020-W01")
            keeper.learn_alias(alias="大高个", canonical="吉日和", week="Y2020-W02")
            rebuilt = Impressions.from_events(keeper.events(), persona=dm.char)
            self.assertEqual(rebuilt.aliases.get(normalize_name("大高个")), "吉日和")


# ---------------------------------------------------------------------------
# KI-3 / KI-4 · relation rules
# ---------------------------------------------------------------------------


class RelationRuleTests(unittest.TestCase):
    def test_repeated_pattern_needs_a_shared_topic_family(self) -> None:
        memories = [
            _memory(memory_id=f"m{i}", content=f"跑单第{i}周", topics=["work", "routine"])
            for i in range(REPEATED_PATTERN_MIN_FAMILY)
        ]
        index = InvertedIndex(memories)
        types = relation_types_between(memories[0], memories[1], index=index)
        self.assertIn("repeated_pattern", types)
        # Only one member: no family, no repeated pattern.
        single = InvertedIndex(memories[:1] + [_memory(memory_id="x", topics=["work", "food"])])
        self.assertNotIn(
            "repeated_pattern",
            relation_types_between(single.memories[0], single.memories[1], index=single),
        )

    def test_problem_resource_matches_beyond_equality(self) -> None:
        need = _memory(
            memory_id="n",
            content="墙上打孔装窗帘杆很费劲",
            topics=["housing"],
            goals=["把窗帘杆装上"],
            obstacles=["承重墙打孔"],
        )
        supply = _memory(
            memory_id="s",
            content="家里有冲击钻和膨胀螺丝",
            topics=["housing"],
            goals=["把窗帘杆装上"],
            resources=["冲击钻和膨胀螺丝"],
        )
        self.assertGreater(_problem_resource_overlap(need, supply), 0)
        types = relation_types_between(need, supply)
        self.assertIn("problem_resource", types)
        self.assertIn("precondition", relation_types_between(need, supply, index=InvertedIndex([need, supply])))

        # Same topic, no lexical overlap at all → weak match, still reachable.
        vague_need = _memory(memory_id="n2", topics=["housing"], obstacles=["房间太暗"])
        vague_supply = _memory(memory_id="s2", topics=["housing"], resources=["台灯"])
        weak = _problem_resource_overlap(vague_need, vague_supply)
        self.assertGreater(weak, 0)
        self.assertLess(weak, 0.5)

        unrelated = _memory(memory_id="n3", topics=["food"], obstacles=["房间太暗"])
        self.assertEqual(_problem_resource_overlap(unrelated, vague_supply), 0.0)

    def test_gap_fires_when_neither_side_answers_the_other(self) -> None:
        a = _memory(
            memory_id="a",
            content="想开单但不敢开口",
            topics=["work"],
            goals=["开单"],
            obstacles=["不敢跟店长开口"],
        )
        b = _memory(
            memory_id="b",
            content="试课客户名单还没拿到",
            topics=["work"],
            goals=["开单"],
            obstacles=["没有客户名单"],
        )
        self.assertIn("gap", relation_types_between(a, b))
        # A resource that answers the obstacle turns it into problem_resource.
        c = _memory(
            memory_id="c",
            content="店长其实愿意给名单",
            topics=["work"],
            goals=["开单"],
            obstacles=["不敢跟店长开口"],
            resources=["不敢跟店长开口"],
        )
        self.assertIn("problem_resource", relation_types_between(a, c))

    def test_contradiction_requires_an_anchor(self) -> None:
        positive = _memory(
            memory_id="p",
            content="学会了土豆丝要泡水",
            topics=["learning"],
            polarity="positive",
        )
        negative = _memory(
            memory_id="n",
            content="独自参加活动没跟人搭上话",
            topics=["learning"],
            polarity="negative",
        )
        # KI-4: sharing only a coarse topic is not an anchor any more.
        self.assertEqual(shared_anchor(positive, negative), [])
        self.assertNotIn("contradiction", relation_types_between(positive, negative))

        anchored = _memory(
            memory_id="n2",
            content="和日和在棋摊没聊起来",
            topics=["learning"],
            entities=["吉日和"],
            polarity="negative",
        )
        same_person = _memory(
            memory_id="p2",
            content="和日和一起做饭聊得很好",
            topics=["learning"],
            entities=["吉日和"],
            polarity="positive",
        )
        self.assertIn(
            "contradiction", relation_types_between(anchored, same_person)
        )


# ---------------------------------------------------------------------------
# KI-5 · semantic idea de-duplication
# ---------------------------------------------------------------------------


class IdeaDedupTests(unittest.TestCase):
    def _candidate(self) -> IdeaCandidate:
        return IdeaCandidate(
            motif="contradiction",
            memory_ids=["hub", "p2"],
            relationship_types=["contradiction"],
            goal="建立联系",
            shared_focus="friendship",
        )

    def _existing(self, *, sources: List[str], plan: str) -> Idea:
        return Idea(
            idea_id="idea-old",
            content="在家常菜课上主动跟同组的人搭话，可能更容易认识人",
            test_plan=plan,
            motif="contradiction",
            source_memory_ids=sources,
            status="candidate",
        )

    def test_same_insight_with_a_new_partner_is_a_refinement(self) -> None:
        duplicate_of = _semantic_duplicate(
            candidate=self._candidate(),
            content="在家常菜课上主动跟同组学员搭话，可能更容易认识人",
            test_plan="本周上一次课，主动跟一个人说话并记录回应",
            existing_ideas=[
                self._existing(
                    sources=["hub", "p1"],
                    plan="本周上一次课，主动跟一个人说话并记录回应次数",
                )
            ],
            existing_methods=[],
        )
        self.assertEqual(duplicate_of, "idea-old")

    def test_a_genuinely_new_plan_is_not_a_duplicate(self) -> None:
        duplicate_of = _semantic_duplicate(
            candidate=self._candidate(),
            content="把练一休一写进训练周计划，观察恢复是否变好",
            test_plan="本周隔天训练，长距离跑后只做拉伸，记录每天体感",
            existing_ideas=[
                self._existing(
                    sources=["hub", "p1"],
                    plan="本周上一次课，主动跟一个人说话并记录回应次数",
                )
            ],
            existing_methods=[],
        )
        self.assertIsNone(duplicate_of)

    def test_evaluate_candidate_reports_the_duplicate(self) -> None:
        hub = _memory(memory_id="hub", content="独自参加活动没聊起来", polarity="negative")
        partner = _memory(memory_id="p2", content="课上主动开口聊上了")
        decision = evaluate_candidate(
            self._candidate(),
            content="在家常菜课上主动跟同组的人搭话，可能更容易认识人",
            test_plan="本周上一次课，主动跟一个人说话并记录回应",
            requires={"skills": [], "entities": [], "money": 0},
            memories=[hub, partner],
            existing_ideas=[
                self._existing(
                    sources=["hub", "p1"],
                    plan="本周上一次课，主动跟一个人说话并记录回应次数",
                )
            ],
            existing_methods=[],
            world_state=IdeaWorldState(skills={"厨艺"}, entities={"吉日和"}),
        )
        self.assertIn("duplicates_existing_idea_semantics", decision.reasons)
        self.assertEqual(decision.duplicate_of, "idea-old")
        self.assertFalse(decision.accepted)


# ---------------------------------------------------------------------------
# KI-6 · methodisation
# ---------------------------------------------------------------------------


class MethodisationTests(unittest.TestCase):
    def test_structure_gaps(self) -> None:
        self.assertEqual(
            _structure_gaps(
                {
                    "title": "备菜法",
                    "steps": ["先想路线", "只买耐放的"],
                    "checks": ["当天吃上热饭"],
                    "failure_modes": ["太累点外卖"],
                    "applicable_contexts": ["routine"],
                }
            ),
            [],
        )
        gaps = _structure_gaps({"title": "备菜法", "steps": ["只有一步"]})
        self.assertEqual(
            sorted(gaps), ["applicable_contexts", "checks", "failure_modes", "steps"]
        )

    def test_parser_filters_context_tags(self) -> None:
        parsed = parse_methodization_response(
            json.dumps(
                {
                    "title": "备菜法",
                    "description": "顺路买菜",
                    "steps": ["a", "b"],
                    "checks": ["c"],
                    "failure_modes": ["f"],
                    "applicable_contexts": ["routine", "not_a_tag"],
                    "contraindications": ["time_pressure", "bogus"],
                },
                ensure_ascii=False,
            )
        )
        self.assertEqual(parsed["applicable_contexts"], ["routine"])
        self.assertEqual(parsed["contraindications"], ["time_pressure"])


# ---------------------------------------------------------------------------
# KI-9 · exploration sampling
# ---------------------------------------------------------------------------


def _methodology(method_id: str, *, status: str = "proposed", value: float = 0.0) -> Methodology:
    return Methodology(
        method_id=method_id,
        skill_id="写作",
        title=method_id,
        description="",
        status=status,
        global_value=value,
        confidence=0.0,
    )


class ExplorationTests(unittest.TestCase):
    def test_exploration_samples_the_whole_explorable_set(self) -> None:
        methods = [_methodology(f"m{i}") for i in range(4)]
        signature = context_signature_from_activity(
            {"type": "solo", "content": "写一章", "outcome": {}}, skill_id="写作"
        )
        picked = set()
        for seed in range(40):
            selection = select_methods(
                methods,
                signature,
                exploration_rate=1.0,
                rng=deterministic_rng("test", seed),
            )
            if selection.explored and selection.primary is not None:
                picked.add(selection.primary.method.method_id)
        self.assertGreater(len(picked), 1, "exploration must not always pick the same method")

    def test_same_seed_replays_exactly(self) -> None:
        methods = [_methodology(f"m{i}") for i in range(4)]
        signature = context_signature_from_activity(
            {"type": "solo", "content": "写一章", "outcome": {}}, skill_id="写作"
        )
        first = select_methods(
            methods, signature, exploration_rate=1.0, rng=deterministic_rng("test", 7)
        )
        second = select_methods(
            methods, signature, exploration_rate=1.0, rng=deterministic_rng("test", 7)
        )
        self.assertEqual(
            first.primary.method.method_id, second.primary.method.method_id
        )

    def test_no_exploration_keeps_the_best_method(self) -> None:
        methods = [
            _methodology("proposed-low", value=0.0),
            _methodology("validated-high", status="validated", value=0.9),
        ]
        signature = context_signature_from_activity(
            {"type": "solo", "content": "写一章", "outcome": {}}, skill_id="写作"
        )
        selection = select_methods(
            methods, signature, exploration_rate=0.0, rng=deterministic_rng("x")
        )
        self.assertFalse(selection.explored)
        self.assertEqual(selection.primary.method.method_id, "validated-high")


# ---------------------------------------------------------------------------
# KI-7 · soft entities in the feasibility check
# ---------------------------------------------------------------------------


class SoftEntityFeasibilityTests(unittest.TestCase):
    def test_alias_and_observed_names_do_not_hard_fail(self) -> None:
        view = Impressions(persona="甲")
        view.aliases[normalize_name("日和")] = "吉日和"
        view.soft_entities["中介张哥"] = 1
        state = IdeaWorldState(
            skills={"厨艺"}, entities={"吉日和"}, impressions=view, observed={"中介张哥"}
        )
        self.assertEqual(state.missing_entities(["日和"]), [])
        self.assertEqual(state.resolve_entity("日和")[0], "吉日和")
        self.assertEqual(state.soft_requirements(["中介张哥"]), ["中介张哥"])
        self.assertEqual(state.missing_entities(["不存在的人"]), ["不存在的人"])

    def test_soft_requirement_only_dents_feasibility(self) -> None:
        candidate = IdeaCandidate(
            motif="unused_resource",
            memory_ids=["m1", "m2"],
            relationship_types=["same_goal"],
        )
        decision = evaluate_candidate(
            candidate,
            content="让中介张哥帮忙问一句房东能不能装空调",
            test_plan="本周联系中介张哥问一次，记录回复",
            requires={"skills": [], "entities": ["中介张哥"], "money": 0},
            memories=[_memory(memory_id="m1"), _memory(memory_id="m2")],
            existing_ideas=[],
            existing_methods=[],
            world_state=IdeaWorldState(
                skills=set(), entities=set(), observed={"中介张哥"}
            ),
        )
        self.assertNotIn("depends_on_missing_entity", decision.reasons)
        self.assertAlmostEqual(decision.feasibility, 0.9, places=4)


# ---------------------------------------------------------------------------
# KI-12 · reflection fallback
# ---------------------------------------------------------------------------


class ReflectionFallbackTests(unittest.TestCase):
    def test_last_prose_block_ignores_markers_and_stage_directions(self) -> None:
        response = "（旁白。）\n\n课散了，腿有点沉，但脑子是清的。\n\nReflection: 这一课值了。"
        self.assertEqual(_last_prose_block(response), "这一课值了。")

    def test_marker_only_response_returns_the_body(self) -> None:
        response = "Summary of the Activity:\n分房定死。\n\nReflection:\n饭桌上把话说开了。"
        self.assertEqual(_last_prose_block(response), "饭桌上把话说开了。")

    def test_empty_response_is_empty(self) -> None:
        self.assertEqual(_last_prose_block(""), "")


if __name__ == "__main__":
    unittest.main()


class FinalizeWithoutToolsShapeTests(unittest.TestCase):
    """KI-25：`generate_with_fc` 失败时返回**字符串**，`list.extend` 把它拆成字符。

    3 年验收运行跑到第 29 周崩在 `save_generation` 的 `m.get("content")` 上——
    因为 `outputs` 里混进了单字符元素。这里钉住规范化函数本身，
    它不依赖 RoleAgent 实例（staticmethod）。
    """

    def _normalise(self, output):
        from src.agents.role_agent import RoleAgent

        return RoleAgent._normalise_generation_output(output)

    def test_the_error_sentinel_is_dropped_not_split_into_characters(self) -> None:
        from src.utils import _ERROR_RESPONSE

        self.assertEqual(self._normalise(_ERROR_RESPONSE), [])
        # 这就是事故发生时的形态：每个字符各占一项
        self.assertEqual(len(list(_ERROR_RESPONSE)), len("NO_RESPONSE"))

    def test_a_normal_message_list_passes_through(self) -> None:
        messages = [{"role": "assistant", "content": "答案"}]
        self.assertEqual(self._normalise(messages), messages)

    def test_a_bare_string_becomes_one_assistant_message(self) -> None:
        self.assertEqual(
            self._normalise("就写这一句"),
            [{"role": "assistant", "content": "就写这一句"}],
        )

    def test_a_bare_dict_becomes_a_single_item_list(self) -> None:
        message = {"role": "assistant", "content": "答案"}
        self.assertEqual(self._normalise(message), [message])

    def test_an_empty_answer_yields_nothing(self) -> None:
        self.assertEqual(self._normalise(""), [])
        self.assertEqual(self._normalise(None), [])
        self.assertEqual(self._normalise([]), [])

    def test_save_generation_tolerates_a_stray_string(self) -> None:
        """第二道防线：即使有字符串混进 outputs，也不再崩掉整轮运行。"""
        from tests._helpers import make_datamanager, temp_workspace

        import json

        with temp_workspace():
            dm, _clock = make_datamanager()
            dm.save_generation(
                [{"role": "system", "content": "提示"}],
                [{"role": "assistant", "content": "答案"}, "NO_RESPONSE"],
            )
            written = sorted(Path(dm.generation).glob("year=*/week=*.jsonl"))[-1]
            record = json.loads(written.read_text(encoding="utf-8").splitlines()[-1])
            self.assertGreater(record["output_tokens"], 0)
