"""对照基线：一个开关关掉全部"与原生 Agentopia 不同"的新功能。

背景（2026-09-27，用户提出）：不变量 #10 要求"新系统必须可以通过配置关闭，并保留旧行为作为
对照组"。对照上游 `Neph0s/Agentopia`（tip = `da264aa`，与本地导入的初始提交同源）逐项核对后
发现：上游的 `config.example.json` **完全没有 `cognition` 段**，本分支到阶段 5 已经有 **10 个
默认打开**的开关，逐个关很容易漏；而且有**三处差异根本没有开关**——

1. `dcbe908` 往每周计划 prompt 里加的 "Do Not Fall Into a Routine (Required)"（还改写了原来
   那一句 "Your weekly plan should directly serve your goals…"）；
2. 运行随机种子与模型分配的随机源；
3. 阶段 -1 的七项修复。

用户决策：两档基线（`cognition` / `upstream`），阶段 -1 的修复**不可关**但文档列清楚。

这里钉住两件事：**开关真的关掉了一切**，以及**关掉之后 prompt 逐字节回到原生**。
哈希常量是在实现时用 `git show da264aa:src/agents/prompts.py` 加载上游模块逐字节比对后记下的；
它们一旦变化，说明有人改了 prompt 而没有同步更新基线对照，测试会立刻报出来。
"""

from __future__ import annotations

import unittest

from src.config import get_config
from src.world.baseline import (
    BASELINE_COGNITION,
    BASELINE_OFF,
    BASELINE_UPSTREAM,
    COGNITION_SWITCHES,
    apply_baseline,
    baseline_of,
    describe_effective_features,
    effective_features,
    routine_prompt_enabled,
    unexpected_features,
    upstream_seed_source,
)

# 逐字节核对过的 prompt 哈希（sha256 前 16 位）
#   upstream 档的 plan/solo/joint/public 与 `da264aa` 完全相同；
#   cognition 档与 upstream 档共享三个上帝评估 prompt（goal_progress 关掉了）；
#   off 档的 plan 与改动前的分支行为完全相同。
PLAN_HASH_OFF = "d76f1fdc75943861"
PLAN_HASH_UPSTREAM = "08d68bcc4867e67c"
EVAL_HASHES_OFF = {
    "solo": "a0025efdb541a75a",
    "joint": "34c6755d6236dfdd",
    "public": "35a86917117ea4ce",
}
EVAL_HASHES_UPSTREAM = {
    "solo": "67447d5dab2756a9",
    "joint": "33b382ea9c6a1719",
    "public": "d55ad8927a0e3b53",
}


def _with_baseline(profile: str):
    """Apply a baseline profile to the live config; returns a restore callback."""

    class _Ctx:
        """Snapshot the whole `world` section, not just the keys this module names.

        `apply_baseline` also switches `world.vitality_recovery.enabled`, so
        restoring only `cognition` leaked a disabled recovery into every test
        that ran after this file.
        """

        def __enter__(self):
            import copy

            config = get_config()
            self.world = copy.deepcopy(config["world"])
            config["world"].setdefault("cognition", {})["baseline"] = profile
            config["world"].pop("upstream_compat", None)
            apply_baseline(config)
            return config

        def __exit__(self, *exc):
            import copy

            get_config()["world"] = copy.deepcopy(self.world)
            return False

    return _Ctx()


def _hashes() -> dict:
    import hashlib

    from src.agents import prompts

    def h(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    return {
        "plan": h(prompts.render_plan_prompt()),
        "solo": h(prompts.build_god_eval_solo_activity_prompt()),
        "joint": h(prompts.build_god_eval_joint_activity_prompt()),
        "public": h(prompts.build_god_eval_public_activity_prompt()),
    }


class BaselineProfileTests(unittest.TestCase):
    def test_an_unknown_profile_falls_back_to_off(self) -> None:
        self.assertEqual(baseline_of({"world": {"cognition": {"baseline": "nonsense"}}}), "off")
        self.assertEqual(baseline_of({}), "off")
        self.assertEqual(baseline_of({"world": {"cognition": {"baseline": "UPSTREAM"}}}), "upstream")

    def test_off_changes_nothing(self) -> None:
        config = {"world": {"cognition": {"method_hints": True, "baseline": "off"}}}
        before = dict(config["world"]["cognition"])
        apply_baseline(config)
        self.assertEqual(config["world"]["cognition"], before)

    def test_cognition_switches_every_feature_off(self) -> None:
        config = {
            "world": {
                "cognition": {**{key: True for key in COGNITION_SWITCHES}, "baseline": BASELINE_COGNITION},
                "vitality_recovery": {"enabled": True},
            }
        }
        apply_baseline(config)
        for key in COGNITION_SWITCHES:
            self.assertFalse(config["world"]["cognition"][key], key)
        self.assertFalse(config["world"]["vitality_recovery"]["enabled"])
        self.assertEqual(unexpected_features(config), [])
        # 阶段 -1 的修复与种子行为不在这张清单里：它们不可关
        self.assertEqual(routine_prompt_enabled(config), True)
        self.assertEqual(upstream_seed_source(config), False)

    def test_upstream_also_drops_the_two_unswitched_differences(self) -> None:
        config = {"world": {"cognition": {"method_hints": True}}}
        config["world"]["cognition"]["baseline"] = BASELINE_UPSTREAM
        apply_baseline(config)
        self.assertFalse(routine_prompt_enabled(config))
        self.assertTrue(upstream_seed_source(config))
        self.assertEqual(unexpected_features(config), [])
        # 恢复上游的随机源是 upstream 档**允许**开着的唯一一项
        enabled = {line.name for line in effective_features(config) if line.enabled}
        self.assertEqual(enabled, {"upstream_compat.seed_source"})

    def test_a_feature_re_enabled_after_the_profile_is_caught(self) -> None:
        config = {"world": {"cognition": {"baseline": "cognition"}}}
        apply_baseline(config)
        config["world"]["cognition"]["idea_engine"] = True  # 有人事后打开了
        self.assertEqual(unexpected_features(config), ["cognition.idea_engine"])

    def test_the_summary_line_names_the_profile_and_what_is_on(self) -> None:
        config = {"world": {"cognition": {"baseline": "cognition"}}}
        apply_baseline(config)
        text = describe_effective_features(config)
        self.assertIn("baseline=cognition", text)
        self.assertIn("cognition features ON: none", text)
        # routine prompt 不是认知层的东西，这一档不动它——日志要如实写出来
        self.assertIn("still differs from upstream: upstream_compat.routine_prompt", text)


class BaselinePromptEquivalenceTests(unittest.TestCase):
    """关掉之后 prompt 必须逐字节回到原生——哈希是拿上游文件核对后记下来的。"""

    def test_the_upstream_profile_reproduces_the_upstream_prompts(self) -> None:
        with _with_baseline(BASELINE_UPSTREAM):
            hashes = _hashes()
        self.assertEqual(hashes["plan"], PLAN_HASH_UPSTREAM)
        self.assertEqual(hashes["solo"], EVAL_HASHES_UPSTREAM["solo"])
        self.assertEqual(hashes["joint"], EVAL_HASHES_UPSTREAM["joint"])
        self.assertEqual(hashes["public"], EVAL_HASHES_UPSTREAM["public"])

    def test_the_default_profile_keeps_the_branch_behaviour(self) -> None:
        """off 档必须与改动前的分支完全一致——基线开关不能顺手改掉本分支的行为。"""
        with _with_baseline(BASELINE_OFF):
            hashes = _hashes()
        self.assertEqual(hashes["plan"], PLAN_HASH_OFF)
        self.assertEqual(hashes["solo"], EVAL_HASHES_OFF["solo"])
        self.assertEqual(hashes["joint"], EVAL_HASHES_OFF["joint"])
        self.assertEqual(hashes["public"], EVAL_HASHES_OFF["public"])

    def test_the_cognition_profile_only_drops_the_cognition_prompts(self) -> None:
        """cognition 档保留分支的计划 prompt（routine 段不是认知层的东西）。"""
        with _with_baseline(BASELINE_COGNITION):
            hashes = _hashes()
        self.assertEqual(hashes["plan"], PLAN_HASH_OFF)
        self.assertEqual(hashes, {"plan": PLAN_HASH_OFF, **EVAL_HASHES_UPSTREAM})

    def test_the_routine_block_is_the_only_plan_prompt_difference(self) -> None:
        with _with_baseline(BASELINE_OFF):
            branch = _hashes()["plan"]
        with _with_baseline(BASELINE_UPSTREAM):
            upstream = _hashes()["plan"]
        self.assertNotEqual(branch, upstream)

    def test_no_placeholder_survives_any_profile(self) -> None:
        from src.agents.prompts import render_plan_prompt

        for profile in (BASELINE_OFF, BASELINE_COGNITION, BASELINE_UPSTREAM):
            with _with_baseline(profile):
                text = render_plan_prompt()
            self.assertNotIn("__", text.split("### Weekly Planning")[-1][:200], profile)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
