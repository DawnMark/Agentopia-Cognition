"""对照基线：一个开关，让"与原生 Agentopia 不同的东西"全部不生效。

设计正文不变量 #10 要求"新系统必须可以通过配置关闭，并保留旧行为作为对照组"。
逐项核对（2026-09-27，对照上游 `Neph0s/Agentopia` 的 `da264aa`）之后发现三件事：

1. 上游的 `config.example.json` **完全没有 `cognition` 段**，所以认知层整层都是新增；
2. 到这一轮为止已经有 **10 个默认打开**的开关，逐个关容易漏掉一个，而那一个就足以让
   对照失去意义；
3. 有三处差异**根本没有开关**：`dcbe908` 往每周计划 prompt 里加的
   "Do Not Fall Into a Routine (Required)"、运行随机种子的来源改动、
   以及阶段 -1 的七项修复。

这个模块给出用户决策（2026-09-27）后的两档基线：

    baseline = "off"       正常运行（默认）
    baseline = "cognition" 关掉全部认知层功能；保留阶段 -1 的修复与种子改进。
                           用来回答"认知层给模拟带来了什么变化"。
    baseline = "upstream"  再关掉 routine prompt 与种子改动，最大程度贴近原生。
                           用来回答"这个分支整体给模拟带来了什么变化"。

**阶段 -1 的修复不可关**（用户决策）：它们是修 bug，做成开关等于提供一条"故意复现已知
缺陷"的代码路径。它们与上游的差异在 `docs/baseline-comparison.md` 里逐条列出，
对照时要连同这一点一起解释。

用法：

    from src.world.baseline import apply_baseline, describe_effective_features

    config = apply_baseline(config)          # 按 config["world"]["cognition"]["baseline"] 归一
    print(describe_effective_features(config))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

BASELINE_OFF = "off"
BASELINE_COGNITION = "cognition"
BASELINE_UPSTREAM = "upstream"
BASELINE_PROFILES = (BASELINE_OFF, BASELINE_COGNITION, BASELINE_UPSTREAM)

# 认知层里每一个"会改变行为或写事件"的开关。默认值全部按 **运行时** 的配置来，
# 不按代码里的 dataclass 默认值来——用户看到的应该是"这次运行实际生效了什么"。
COGNITION_SWITCHES: Tuple[str, ...] = (
    "methodology_shadow",
    "memory_shadow",
    "idea_engine",
    "lesson_ingest",
    "method_hints",
    "method_hints_bandit",
    "method_lifecycle",
    "goal_progress",
    "proficiency_projection",
    "capability_input",
    "capability_gain_cap",
)

# 不在 `world.cognition` 段下、但同样属于"新增且会改变行为"的开关。
WORLD_SWITCHES: Tuple[Tuple[str, str], ...] = (
    ("vitality_recovery", "enabled"),
)


@dataclass(frozen=True)
class FeatureLine:
    """一行"这次运行实际生效了什么"。"""

    name: str
    enabled: bool
    source: str  # 配置里的路径，便于定位

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "enabled": self.enabled, "path": self.source}


def baseline_of(config: Dict[str, Any]) -> str:
    """This run's baseline profile, normalised (unknown values fall back to off)."""
    section = ((config.get("world") or {}).get("cognition")) or {}
    value = str(section.get("baseline", BASELINE_OFF) or BASELINE_OFF).strip().lower()
    return value if value in BASELINE_PROFILES else BASELINE_OFF


def apply_baseline(config: Dict[str, Any]) -> Dict[str, Any]:
    """Fold the baseline profile into the config, in place, and return it.

    Called from `run_world.py` *and* from `World.__init__`, so a caller that
    builds a `World` directly cannot accidentally run a "baseline" comparison
    with the new features still switched on.
    """
    profile = baseline_of(config)
    if profile == BASELINE_OFF:
        return config

    world = config.setdefault("world", {})
    cognition = world.setdefault("cognition", {})

    if profile in (BASELINE_COGNITION, BASELINE_UPSTREAM):
        for key in COGNITION_SWITCHES:
            cognition[key] = False
        # `method_hints_bandit` implies hints; both are already False above.
        for section_name, key in WORLD_SWITCHES:
            section = world.setdefault(section_name, {})
            section[key] = False

    if profile == BASELINE_UPSTREAM:
        # The two differences that never had a switch of their own.
        world.setdefault("upstream_compat", {})["routine_prompt"] = False
        world.setdefault("upstream_compat", {})["seed_source"] = True

    cognition["baseline"] = profile
    return config


def routine_prompt_enabled(config: Dict[str, Any]) -> bool:
    """Whether the "Do Not Fall Into a Routine" plan block is added.

    It came in with `dcbe908`, before the cognition layer, and had no switch;
    `baseline = "upstream"` is that switch. Default is on (the branch's own
    behaviour), so only an explicit `upstream_compat.routine_prompt = false`
    turns it off.
    """
    section = ((config.get("world") or {}).get("upstream_compat")) or {}
    return bool(section.get("routine_prompt", True))


def upstream_seed_source(config: Dict[str, Any]) -> bool:
    """Whether to reproduce upstream's model-assignment RNG source.

    Upstream seeded the assignment with the *run directory name*, which makes a
    replay in a fresh directory produce a different assignment. The branch seeds
    it with `world name + run seed` instead. `baseline = "upstream"` restores the
    original source, so a comparison run reproduces upstream's assignment.
    """
    section = ((config.get("world") or {}).get("upstream_compat")) or {}
    return bool(section.get("seed_source", False))


def effective_features(config: Dict[str, Any]) -> List[FeatureLine]:
    """Every new feature and whether this run actually has it on."""
    world = config.get("world") or {}
    cognition = world.get("cognition") or {}
    lines: List[FeatureLine] = []
    for key in COGNITION_SWITCHES:
        name = f"cognition.{key}"
        lines.append(
            FeatureLine(
                name=name,
                enabled=bool(cognition.get(key, False)),
                source=f"world.cognition.{key}",
            )
        )
    for section_name, key in WORLD_SWITCHES:
        section = world.get(section_name) or {}
        lines.append(
            FeatureLine(
                name=f"{section_name}.{key}",
                enabled=bool(section.get(key, True)),
                source=f"world.{section_name}.{key}",
            )
        )
    lines.append(
        FeatureLine(
            name="upstream_compat.routine_prompt",
            enabled=routine_prompt_enabled(config),
            source="world.upstream_compat.routine_prompt",
        )
    )
    lines.append(
        FeatureLine(
            name="upstream_compat.seed_source",
            enabled=upstream_seed_source(config),
            source="world.upstream_compat.seed_source",
        )
    )
    return lines


def describe_effective_features(config: Dict[str, Any]) -> str:
    """One line per feature for the run log — a comparison run documents itself."""
    profile = baseline_of(config)
    lines = effective_features(config)
    cognition_on = [
        line.name for line in lines if line.enabled and line.name.startswith("cognition.")
    ]
    # 与上游仍有差异、但这一档不动的东西（写在日志里，免得对照时误判）。
    # `upstream_compat.seed_source` 开着是"恢复上游做法"，不算仍存差异。
    residual = [
        line.name
        for line in lines
        if line.enabled
        and not line.name.startswith("cognition.")
        and line.name != "upstream_compat.seed_source"
    ]
    parts = [f"baseline={profile}"]
    if profile == BASELINE_UPSTREAM:
        parts.append("upstream-compat: routine prompt off, upstream seed source restored")
    if profile == BASELINE_COGNITION:
        parts.append("phase -1 fixes and seed behaviour still active (not switchable)")
    parts.append(
        "cognition features ON: " + (", ".join(cognition_on) if cognition_on else "none")
    )
    if residual:
        parts.append("still differs from upstream: " + ", ".join(residual))
    return " | ".join(parts)


def unexpected_features(config: Dict[str, Any]) -> List[str]:
    """Features still on that the baseline profile says should be off.

    Empty for every well-formed config; non-empty means something re-enabled a
    switch after `apply_baseline` ran, which would silently invalidate a
    comparison run.
    """
    profile = baseline_of(config)
    if profile == BASELINE_OFF:
        return []
    lines = effective_features(config)
    # 每一档允许保持开启的项：cognition 档不动本分支的计划 prompt（routine 段不是
    # 认知层的东西，它由 upstream 档负责关）；upstream 档则允许"恢复上游随机源"开着。
    allowed = (
        {"upstream_compat.seed_source"}
        if profile == BASELINE_UPSTREAM
        else {"upstream_compat.routine_prompt"}
    )
    return [line.name for line in lines if line.enabled and line.name not in allowed]
