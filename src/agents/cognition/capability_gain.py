"""阶段 6 第二步：把"能力已经很高就别再涨"从 prompt 里的请求变成代码里的规则。

## 为什么要有这个文件

`capability_input` 把派生能力放进了 God 的评估输入，第一版只在 prompt 里写了一句
"能力已经很高时增益应当很小"。反事实探针（2026-09-27，96 次调用，真实 2 年 prompt）
测出来的结果是：

| 档位 | 注入的数字 | 增益均值 | 中位数 |
|---|---|---|---|
| off | 不注入 | 1.625 | 2.0 |
| real | 真实值（最高 29/100） | 1.333 | 1.0 |
| high | 55/100 | 1.375 | 1.0 |
| extreme | 95/100 | 1.167 | 1.0 |

也就是说：**块的存在会把典型增益压一档，但数字本身几乎不进口判断**（14 与 55 没有区别）。
这不是"模型不听话"，而是"让语言模型按一个它看不到量纲、也无法验证的数值做分段函数"
本身就不是可靠的做法。用户因此决定（2026-09-27）：**规则落到代码**——
在 God 给出增益之后、写进活动记录之前，按角色**当前**的有效能力给 `delta_skills` 封顶。

## 规则

按能力从高到低匹配第一条满足的规则（阈值与上限都可配）：

    capability >= high  → 该技能增益最多 high_gain（默认 0）
    capability >= mid   → 该技能增益最多 mid_gain（默认 1）
    否则                → 不动

只**封顶**、从不抬高，也永远不动负值：God 的判断仍然是判断，代码只负责"练到一定程度之后
再练也涨不了多少"这一条设计意图。每一次封顶都写一条 `CAPABILITY_GAIN_CAPPED` 事件，
所以"这条规则到底有没有生效、在哪些技能上生效"是可查的，而不是靠推断。

## 关闭时的行为

`world.cognition.capability_gain_cap = false`（默认）：函数直接返回原值，
不重建投影、不写事件，运行的 prompt 与账本与本轮之前逐字节相同。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.agents.data_manager import DataManager

CAPABILITY_GAIN_CAPPED = "CAPABILITY_GAIN_CAPPED"

# 阈值是照着**真实账本回放**定的，不是拍的：把 2 年运行的账本按周回放，看每种阈值组合
# 会触发多少次、压掉多少点（`docs/phase6-acceptance.md` §6 有完整表格）。
#
#   阈值组合        上官霄月        吉日和        萧亦岚
#   0.40 / 0.70     0 次 / 0 点     0 / 0         0 / 0        ← 一次都不触发
#   0.30 / 0.50     13 次 / 14%     5 / 6%        4 / 3%
#   0.25 / 0.45     14 次 / 15%     9 / 11%       12 / 8%     ← 选定
#   0.20 / 0.35     18 次 / 28%     15 / 16%      15 / 11%
#
# 第一版默认 0.40 / 0.70 是错的：新口径下 2 年里最高的有效能力只有 39/100，
# 那条规则永远不会触发——"声明了却一次都不写入"与没实现是一回事（KI-15 同类）。
DEFAULT_CAP_MID = 0.25
DEFAULT_CAP_MID_GAIN = 1.0
DEFAULT_CAP_HIGH = 0.45
DEFAULT_CAP_HIGH_GAIN = 0.0


@dataclass(frozen=True)
class CapRule:
    """一条封顶规则：能力达到 `threshold` 时，增益最多 `max_gain`。"""

    threshold: float
    max_gain: float
    name: str

    def to_dict(self) -> Dict[str, Any]:
        return {"rule": self.name, "threshold": self.threshold, "max_gain": self.max_gain}


@dataclass(frozen=True)
class CapNote:
    """一次实际发生的封顶（写进事件，供审计读）。"""

    skill: str
    before: float
    after: float
    capability: float
    rule: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill": self.skill,
            "delta_before": self.before,
            "delta_after": self.after,
            "capability": round(self.capability, 4),
            "rule": self.rule,
        }


def cap_rules(config: Dict[str, Any]) -> Tuple[CapRule, ...]:
    """从配置读出规则，按阈值从高到低排序（第一条命中的生效）。"""
    cognition = ((config.get("world") or {}).get("cognition")) or {}
    mid = float(cognition.get("capability_cap_mid", DEFAULT_CAP_MID))
    mid_gain = float(cognition.get("capability_cap_mid_gain", DEFAULT_CAP_MID_GAIN))
    high = float(cognition.get("capability_cap_high", DEFAULT_CAP_HIGH))
    high_gain = float(cognition.get("capability_cap_high_gain", DEFAULT_CAP_HIGH_GAIN))
    rules = (
        CapRule(threshold=high, max_gain=high_gain, name="high"),
        CapRule(threshold=mid, max_gain=mid_gain, name="mid"),
    )
    return tuple(sorted(rules, key=lambda rule: -rule.threshold))


def cap_enabled(config: Dict[str, Any]) -> bool:
    cognition = ((config.get("world") or {}).get("cognition")) or {}
    return bool(cognition.get("capability_gain_cap", False))


def rule_for(capability: float, rules: Tuple[CapRule, ...]) -> Optional[CapRule]:
    for rule in rules:
        if capability >= rule.threshold:
            return rule
    return None


def cap_deltas(
    deltas: Dict[str, Any],
    capabilities: Dict[str, float],
    rules: Tuple[CapRule, ...],
) -> Tuple[Dict[str, float], List[CapNote]]:
    """给一组 `delta_skills` 封顶；返回 (新的增益表, 实际发生的封顶)。"""
    capped: Dict[str, float] = {}
    notes: List[CapNote] = []
    for skill, value in (deltas or {}).items():
        try:
            delta = float(value)
        except (TypeError, ValueError):
            continue
        capability = float(capabilities.get(str(skill), 0.0))
        rule = rule_for(capability, rules)
        if rule is not None and delta > rule.max_gain:
            notes.append(
                CapNote(
                    skill=str(skill),
                    before=delta,
                    after=rule.max_gain,
                    capability=capability,
                    rule=rule.name,
                )
            )
            delta = rule.max_gain
        capped[str(skill)] = delta
    return capped, notes


def capabilities_for(dm: "DataManager") -> Dict[str, float]:
    """角色当前每个技能的有效能力，直接从账本重建（阶段 6 的投影）。

    投影是按**能力族**归并的（`skill_aliases.json`，见 `proficiency` 的说明），而活动
    记录里写的是**原始技能名**——`客户沟通` 会折进 `人际沟通`，直接按原始名查投影是查
    不到的。这里把每个族展开回账本里出现过的写法，并额外用族映射兜一次底，
    否则这类技能会**永远绕开封顶**（实测：2 年运行里 3 条越界记录全部是折叠名）。
    """
    from src.agents.cognition.proficiency import build_projection
    from src.agents.cognition.skills import SkillFamilies

    projection = build_projection(dm)
    skills = projection.get("skills") or {}
    caps = {
        skill: float(entry.get("effective_capability") or 0.0)
        for skill, entry in skills.items()
    }
    for skill, entry in skills.items():
        capability = float(entry.get("effective_capability") or 0.0)
        for raw in entry.get("raw_names") or []:
            caps.setdefault(str(raw), capability)
    families = SkillFamilies.from_data_manager(dm)
    return _ExpandedCapabilities(caps, families)


class _ExpandedCapabilities(dict):
    """按原始技能名查能力；查不到时再用能力族映射查一次。

    对 `cap_deltas` 来说它就是一个普通 dict，多出来的只是"折叠名也能查到"。
    """

    def __init__(self, caps: Dict[str, float], families: Any) -> None:
        super().__init__(caps)
        self._families = families

    def get(self, key, default=None):  # type: ignore[override]
        if key in self:
            return dict.get(self, key)
        canonical = self._families.canonical(str(key)) if self._families else str(key)
        if canonical in self:
            return dict.get(self, canonical)
        return default


def _timestamp(dm: Any) -> str:
    """事件用的时间戳：事件的 `time` 由账本写入时统一补，键里带上它才唯一。"""
    clock = getattr(dm, "clock", None)
    try:
        return str(clock.get_time()) if clock is not None else ""
    except Exception:  # pragma: no cover - defensive
        return ""


def cap_agent_deltas(
    agent: Any,
    deltas: Dict[str, Any],
    *,
    activity_id: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """按角色当前能力给这次活动的技能增益封顶，并把封顶写进能力事件流。

    只在 `world.cognition.capability_gain_cap` 打开时做任何事；关闭时原样返回
    （连投影都不重建），这样默认运行的账本与 prompt 完全不受影响。
    """
    if config is None:
        from src.config import get_config

        config = get_config()
    if not cap_enabled(config):
        return {str(k): float(v) for k, v in (deltas or {}).items()}

    dm = getattr(agent, "dm", None)
    if dm is None:  # pragma: no cover - defensive
        return {str(k): float(v) for k, v in (deltas or {}).items()}

    capped, notes = cap_deltas(deltas or {}, capabilities_for(dm), cap_rules(config))
    if notes:
        from src.agents.cognition.event_store import CapabilityEventStore

        store = CapabilityEventStore(dm)
        stamp = _timestamp(dm)
        for note in notes:
            # 幂等键必须**唯一到"这一次封顶"**：早先的键是
            # `CAPABILITY_GAIN_CAPPED:<技能>:<activity_id 或 '-'>:<档>`，而 Joint / Public
            # 路径根本拿不到 activity_id（Solo 也只传了 None），于是同一技能同一档的
            # 第二次封顶开始全部被当成重复行丢掉——2 年运行实际封顶 41 / 11 / 12 次，
            # 只记下 1 / 4 / 1 条。键里加时间戳与前后值，既唯一又可读。
            store.append(
                CAPABILITY_GAIN_CAPPED,
                activity_id=activity_id,
                payload=note.to_dict(),
                idempotency_key=(
                    f"{CAPABILITY_GAIN_CAPPED}:{note.skill}:{stamp}:"
                    f"{note.rule}:{note.before}->{note.after}"
                ),
            )
    return capped
