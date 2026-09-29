"""阶段 6 的第一步：把"能力"从累加计数改成可重建的投影（影子模式）。

现状是 `state.skills[skill] = max(0, skills[skill] + God 给的 delta)`——只增不减、无上限、
不可从账本重建，而且"练过"与"用对方法练过"混在同一个数里。设计正文阶段 6 要的是：

    proficiency        来自有效练习
    method value       来自情境化实践结果
    effective capability = proficiency + methodology

这一版**只做影子投影**（用户决策 2026-09-27）：并行算出"由练习派生的 proficiency"与
"由方法派生的有效能力"，写进独立视图，行为完全不变、默认关闭，等读数站得住再谈迁移。

三个口径（都来自用户决策）：

1. **有效练习 = 所有正 skill delta**，不分档。练习量是环境模型客观发放的 delta 之和，
   不是角色自述、也不是反思（不变量 #2.3）。分档（"带方法证据的才算"）在本轮里
   作为**独立的一列**保留下来供以后使用，但不参与 proficiency。
2. **proficiency 是饱和的**：`units / (units + HALF)`，而不是无上限累加——设计正文说的是
   "随有效练习**缓慢**增长"，而当前实现会让任何勤快的角色无限通胀。
3. **effective capability 用加权几何平均**：设计正文写的是连乘，但四个都小于 1 的因子相乘
   会让能力整体塌向 0、失去区分度；加权几何平均保留"任一因子为 0 则能力为 0"的语义，
   又不会因为四个 0.8 相乘变成 0.41。

投影里的每个数都能从两本账重算：`activity.jsonl`（God 发放的 delta）与
`capability_events.jsonl`（方法、练习、结果）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

from src.agents.cognition.skills import SkillFamilies

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.agents.data_manager import DataManager

PROFICIENCY_VIEW_FILENAME = "proficiency.json"

# 方法抽取在没有技能可用时会写这个哨兵值（`method_id_for(skill_id or "unmapped", …)`）。
# 它不是技能：让它进投影会凭空多出一个"技能"，而且单位数与因子都恒为 0。
# 这类方法由审计的 `unmapped_skill_methods` 单独统计。
UNMAPPED_SKILL = "unmapped"

# 饱和曲线的半程点：**同一技能有效练习 10 次**时 proficiency = 0.5（用户决策 2026-09-27）。
#
# 为什么按次数而不是按点数：点数（God 给的 delta 之和）取决于环境模型一星期给多少分，
# 换世界、换模型、换活动配置都要重标；次数与运行长度无关，标定一次长期稳定。
#
# 为什么是 10：这个世界一年 = 10 周，所以 10 次 = **每周练一次、坚持整年**，是一个
# 生活模拟里"确实在练"的门槛。第一版按 25 次标定是错的——那是 2 年半的量，实测 2 年
# 运行里练得最多的技能也才 35 次、中位技能 3 次，于是 10 周运行的能力值只有个位数，
# 机制在任何实际长度的运行里都咬合不上（见 docs/phase6-acceptance.md §4）。
# 按 10 次标定：2 年里的高频技能（35 次）→ 0.78，中位技能（3 次）→ 0.23，
# 10 周运行里的领头技能（约 8 次）→ 0.44。
PROFICIENCY_HALF_PRACTICES = 10.0

# 因子下限：方法类因子为 0 会让几何平均直接塌成 0（"没有任何方法"不等于"用错了方法"），
# 所以对已确认存在方法、但一次都没成功的技能给一个小下限而不是零。
FACTOR_FLOOR = 0.05

# effective capability = proficiency × 方法论因子的加权几何平均。
#
# 设计正文写的是四个因子连乘（proficiency × 覆盖度 × 准确率 × 可靠性）。纯乘积在我方数据上
# 会塌：四个 0.8 相乘得 0.41，四个 0.6 得 0.13，能力值挤在低位、失去区分度。但把 proficiency
# 也当成四个等权因子之一（指数和为 1 的加权几何平均）会往另一个方向错：proficiency 的偏离被
# 指数压扁，0.5 的熟练度配上完美方法会算成 0.76——**比练习证据支持的还高**。
#
# 所以口径是：**proficiency 是基数，方法论是折扣**。
#   capability = proficiency × coverage^(1/3) × accuracy^(1/3) × reliability^(1/3)
# 全部方法因子为 1（没有方法，或方法都好好的）时 capability 就等于 proficiency；
# 任一因子为 0 时按 FACTOR_FLOOR 打折而不是归零（"还没成功过"不该等于"没有能力"）。
# 折扣下限（用户决策 2026-09-27）：方法侧再差也只打到 5 折。
#
# 起因是实测：`情报搜集与反侦察` 练了 35 次、熟练度 78/100，三因子是
# 0.33 / 0.38 / 0.00（覆盖率低是因为抽取引擎两年给它提了 27 个方法，而它只试了 9 个；
# 验证率为 0 是因为整轮只有 6 个方法到过 validated）——乘出来 0.19，有效能力 14/100，
# 反而低于一个只练了 4 次、**根本没有任何方法**的技能（那个三因子恒为 1，拿 29/100）。
# 一个"练得多不如不练"的排序无法解释，也无法拿去影响行为。
#
# 所以口径改成：**方法论只能打折，不能抹掉练习**——折扣取
# `max(DISCOUNT_FLOOR, ∏ 因子^w)`。取 `max` 而不是线性混合，是因为下限就该是下限：
# 低于 0.5 的方法论一律等于 0.5（会形成一小段平台，与 `FACTOR_FLOOR` 是同一个约定——
# "差到一定程度就不再区分"),而 0.5 以上的部分仍然保留全部区分度。
DISCOUNT_FLOOR = 0.5

METHOD_EXPONENTS = {
    "coverage": 1.0 / 3.0,
    "selection_accuracy": 1.0 / 3.0,
    "verified_reliability": 1.0 / 3.0,
}


@dataclass
class SkillPractice:
    """一个技能上的练习证据，全部来自账本。"""

    skill_id: str
    practices: int = 0
    units: float = 0.0
    first_week: str = ""
    last_week: str = ""
    # 独立保留、不参与 proficiency：其中还带真实方法证据的练习次数（用户决策：
    # "有效练习"不分档，但这一列对以后的迁移有用）。
    method_backed_practices: int = 0
    # 折进这个能力族的原始技能名（用户决策 2026-09-27：家族归并只作用于投影统计，
    # 不改 `state.skills`）。`raw_names` 只有一个名字时，这一族没有被归并过。
    raw_names: List[str] = field(default_factory=list)

    @property
    def proficiency(self) -> float:
        return proficiency_from_practices(self.practices)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "practices": int(self.practices),
            "practice_units": round(self.units, 2),  # 记录用：不再驱动 proficiency
            "method_backed_practices": int(self.method_backed_practices),
            "first_week": self.first_week,
            "last_week": self.last_week,
            "proficiency": self.proficiency,
            "proficiency_points": capability_points(self.proficiency),
            "raw_names": list(self.raw_names),
        }


@dataclass
class MethodFactors:
    """一个技能上的方法论因子（设计正文阶段 6 的后三项）。"""

    skill_id: str
    methods: int = 0
    practised_methods: int = 0
    successful_practices: int = 0
    total_practices: int = 0
    reliable_methods: int = 0

    @property
    def coverage(self) -> float:
        """练过的方法占该技能方法总数的比例；没有方法时视为 1（无事可覆盖）。"""
        if self.methods <= 0:
            return 1.0
        return self.practised_methods / self.methods

    @property
    def selection_accuracy(self) -> float:
        """练习里"达到或超过本人常规"的比例；没有练习时视为 1（还没得选错）。"""
        if self.total_practices <= 0:
            return 1.0
        return self.successful_practices / self.total_practices

    @property
    def verified_reliability(self) -> float:
        """练过的方法里走到 `validated` / `specialized` 的比例。"""
        if self.practised_methods <= 0:
            return 1.0
        return self.reliable_methods / self.practised_methods

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "methods": self.methods,
            "practised_methods": self.practised_methods,
            "reliable_methods": self.reliable_methods,
            "total_practices": self.total_practices,
            "successful_practices": self.successful_practices,
            "coverage": round(self.coverage, 4),
            "selection_accuracy": round(self.selection_accuracy, 4),
            "verified_reliability": round(self.verified_reliability, 4),
        }


def proficiency_from_practices(practices: float) -> float:
    """同一技能的**有效练习次数** → proficiency ∈ [0, 1)。

    次数是"练过多少次"（God 每次给了正 delta 记一次），与它给多少分无关——
    见 `PROFICIENCY_HALF_PRACTICES` 的说明。
    """
    value = max(0.0, float(practices))
    return round(value / (value + PROFICIENCY_HALF_PRACTICES), 4)


def capability_points(value: float) -> int:
    """0-1 的能力值 → 0-100 的对外量纲（用户决策 2026-09-27）。

    内部一律用 0-1（有明确的概率式含义），但对外的消费点（职位要求的
    `min_skills`、God prompt 里"比现有最高再高 50 点"这类阈值）用的是点数刻度。
    迁移时把 0-1 直接丢进点数阈值会把所有门槛变成 1 分，所以对外统一乘 100：
    这样"50 点"这句话的语义不变，只是它现在衡量的是练习深度而不是累加量。
    """
    return int(round(max(0.0, min(1.0, float(value))) * 100))


def effective_capability(
    proficiency: float,
    factors: MethodFactors,
    *,
    exponents: Optional[Dict[str, float]] = None,
) -> float:
    """`proficiency × ∏ method_factor^w`（见 `METHOD_EXPONENTS` 的说明）。

    没有方法（或方法全部成功）时方法论因子都是 1，capability 就等于 proficiency；
    方法论差则打折。proficiency 为 0 —— 没有练习 —— 时 capability 为 0。
    """
    import math

    base = max(0.0, float(proficiency))
    if base <= 0:
        return 0.0

    methodology = methodology_discount(factors, exponents=exponents)
    return round(min(1.0, base * max(DISCOUNT_FLOOR, methodology)), 4)


def methodology_discount(
    factors: Any,
    *,
    exponents: Optional[Dict[str, float]] = None,
) -> float:
    """方法论因子 → [FACTOR_FLOOR, 1] 之间的折扣（不含 `DISCOUNT_FLOOR` 下限）。

    `factors` 可以是 `MethodFactors`，也可以是任何带 `coverage` /
    `selection_accuracy` / `verified_reliability` 三个属性的对象——显示层
    （God prompt 里的 `methods x0.59`）与计算层必须走同一条式子，各写一份迟早会分叉。
    """
    import math

    w = exponents or METHOD_EXPONENTS
    parts = {
        "coverage": max(FACTOR_FLOOR, float(getattr(factors, "coverage", 1.0))),
        "selection_accuracy": max(
            FACTOR_FLOOR, float(getattr(factors, "selection_accuracy", 1.0))
        ),
        "verified_reliability": max(
            FACTOR_FLOOR, float(getattr(factors, "verified_reliability", 1.0))
        ),
    }
    return math.exp(sum(w[key] * math.log(parts[key]) for key in parts))


# --- 从账本重建 ---------------------------------------------------------------


def _week_of(time_str: Any) -> str:
    parts = str(time_str or "").split("-")
    if len(parts) >= 2 and parts[0].startswith("Y") and parts[1].startswith("W"):
        return f"{parts[0]}-{parts[1]}"
    return ""


def practice_by_skill(
    activity_records: Iterable[Dict[str, Any]],
    *,
    families: Optional[SkillFamilies] = None,
) -> Dict[str, SkillPractice]:
    """每个技能上的练习量，全部来自 God 在活动结算时发放的正 delta。

    技能名按 `skill_aliases.json` 折进能力族（用户决策 2026-09-27：归并只作用于
    投影统计，`state.skills` 的键不动）。折进哪几个原始名字记在 `raw_names` 里，
    这样"归并到底改变了多少"在投影里就能直接读出来，不需要另一套统计。
    """
    out: Dict[str, SkillPractice] = {}
    for record in activity_records:
        outcome = record.get("outcome") or {}
        gains = outcome.get("delta_skills") or {}
        if not isinstance(gains, dict):
            continue
        week = _week_of(record.get("time"))
        for name, delta in gains.items():
            try:
                value = float(delta)
            except (TypeError, ValueError):
                continue
            if value <= 0:
                continue  # 只算正 delta（用户决策：有效练习 = God 给的正增益）
            raw = str(name).strip()
            skill = families.canonical(str(name)) if families else raw
            if not skill or skill == UNMAPPED_SKILL:
                continue
            entry = out.setdefault(skill, SkillPractice(skill_id=skill))
            entry.practices += 1
            entry.units += value
            if raw and raw not in entry.raw_names:
                entry.raw_names.append(raw)
            if week:
                entry.first_week = entry.first_week or week
                entry.last_week = week
    for entry in out.values():
        entry.raw_names.sort()
    return out


def method_factors_by_skill(
    capability_events: Iterable[Dict[str, Any]],
    *,
    families: Optional[SkillFamilies] = None,
) -> Dict[str, MethodFactors]:
    """每个技能上的方法覆盖、选择准确率与已验证可靠性。

    只认**真实采用**的练习证据：`METHOD_OUTCOME_OBSERVED(attribution="real_adoption")`。
    反事实证据（影子对"本来会选哪个"的估计）不进这里——它不是角色的行为。
    """
    events = list(capability_events)
    out: Dict[str, MethodFactors] = {}

    def factors_for(skill: str) -> MethodFactors:
        return out.setdefault(skill, MethodFactors(skill_id=skill))

    method_skill: Dict[str, str] = {}
    for event in events:
        if str(event.get("type") or "") != "METHOD_PROPOSED":
            continue
        method_id = str(event.get("method_id") or "")
        skill = str(event.get("skill_id") or "").strip()
        if not method_id or not skill or skill == UNMAPPED_SKILL:
            continue
        canonical = families.canonical(skill) if families else skill
        method_skill[method_id] = canonical
        factors_for(canonical).methods += 1

    practised: set = set()
    reliable: set = set()
    for event in events:
        event_type = str(event.get("type") or "")
        if event_type == "METHOD_OUTCOME_OBSERVED":
            if str(event.get("attribution") or "") != "real_adoption":
                continue
            method_id = str(event.get("method_id") or "")
            skill = method_skill.get(method_id)
            if skill is None:
                continue
            factors = factors_for(skill)
            factors.total_practices += 1
            if float(event.get("reward") or 0.0) >= 0.0:
                factors.successful_practices += 1
            practised.add(method_id)
        elif event_type == "METHOD_VALUE_UPDATED":
            if str(event.get("evidence_kind") or "") != "real_adoption":
                continue
            if str(event.get("status") or "") in ("validated", "specialized"):
                reliable.add(str(event.get("method_id") or ""))

    for method_id in sorted(practised):
        skill = method_skill.get(method_id)
        if skill is not None:
            factors_for(skill).practised_methods += 1
    for method_id in sorted(reliable):
        skill = method_skill.get(method_id)
        if skill is not None and method_id in practised:
            factors_for(skill).reliable_methods += 1

    return out


def build_projection(
    dm: "DataManager",
    *,
    families: Optional[SkillFamilies] = None,
) -> Dict[str, Any]:
    """把两本账折成一份能力投影（可直接写入 `views/proficiency.json`）。"""
    if families is None:
        families = SkillFamilies.from_data_manager(dm)

    activity_records: List[Dict[str, Any]] = []
    path = Path(dm.root) / "activity.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                activity_records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    events: List[Dict[str, Any]] = []
    from src.agents.cognition.event_store import CapabilityEventStore

    try:
        events = CapabilityEventStore(dm).events()
    except (OSError, ValueError):
        events = []

    practice = practice_by_skill(activity_records, families=families)
    factors = method_factors_by_skill(events, families=families)

    # 方法证据回填到练习上（只作记录，不参与 proficiency）
    method_skills: Dict[str, set] = {}
    for event in events:
        if str(event.get("type") or "") != "METHOD_OUTCOME_OBSERVED":
            continue
        if str(event.get("attribution") or "") != "real_adoption":
            continue
        method_id = str(event.get("method_id") or "")
        skill = next(
            (
                str(e.get("skill_id") or "")
                for e in events
                if str(e.get("type")) == "METHOD_PROPOSED"
                and str(e.get("method_id")) == method_id
            ),
            "",
        )
        if skill:
            canonical = families.canonical(skill)
            method_skills.setdefault(canonical, set()).add(method_id)
    for skill, ids in method_skills.items():
        if skill in practice:
            practice[skill].method_backed_practices = len(ids)

    skills: Dict[str, Any] = {}
    for skill in sorted(set(practice) | set(factors)):
        entry = practice.get(skill) or SkillPractice(skill_id=skill)
        factor = factors.get(skill) or MethodFactors(skill_id=skill)
        capability = effective_capability(entry.proficiency, factor)
        skills[skill] = {
            **entry.to_dict(),
            "factors": factor.to_dict(),
            "effective_capability": capability,
            "effective_capability_points": capability_points(capability),
        }

    # 归并读数（用户决策 2026-09-27）：先看清"归并前后差多少"，再谈要不要
    # 真去改 `state.skills` 的键。
    raw_names = sorted({name for entry in practice.values() for name in entry.raw_names})
    merged_families = {
        skill: entry.raw_names for skill, entry in practice.items() if len(entry.raw_names) > 1
    }
    return {
        "persona": getattr(dm, "char", ""),
        "activities": len(activity_records),
        "events": len(events),
        "skills": skills,
        "stats": {
            "skills": len(skills),
            "practised_skills": sum(1 for s in skills.values() if s["practices"] > 0),
            "skills_with_methods": sum(1 for s in skills.values() if s["factors"]["methods"] > 0),
            "capability_mean": (
                round(
                    sum(s["effective_capability"] for s in skills.values()) / len(skills), 4
                )
                if skills
                else None
            ),
            # 归并前后：`raw_names_observed` 是账本里出现过的原始技能名个数，
            # `practised_skills` 是折叠之后的族数，`merged_families` 是真正合并过
            # 多个名字的族。
            "raw_names_observed": len(raw_names),
            "merged_family_count": len(merged_families),
            "merged_families": merged_families,
        },
    }


def write_projection(dm: "DataManager", projection: Optional[Dict[str, Any]] = None) -> Path:
    """写入 `persona/<name>/cognition/views/proficiency.json`（可删可重建）。"""
    if projection is None:
        projection = build_projection(dm)
    target = Path(dm.root) / "cognition" / "views" / PROFICIENCY_VIEW_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(projection, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target
