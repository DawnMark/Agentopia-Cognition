#!/usr/bin/env python
"""对照两次运行的**技能增益分布**：能力块进评估输入之后，God 还发多少分。

存在的理由（阶段 6，用户决策 2026-09-27）：`capability_input` 是"能力进入行为"的
第一步，而它进入的是**判定增益的输入**。这一步不能只看两边跑没跑起来，要看下面三个
问题各自有没有答案：

1. 增益有没有变小？——按技能聚合 `delta_skills`，比较两次运行的均值/中位数/总量；
2. 有没有"直接不发了"？——零增益与"整条活动没有 delta_skills"的比例；一条指令被
   过度解读成"什么都别给"是这类 prompt 改动最典型的失败形态；
3. 变化是不是发生在**该变的地方**？——按"该技能在运行前半程的练习量"分层，看高练习
   量的技能增益是否比对照组下降更多。这是唯一能区分"模型读懂了块"和"模型只是少发了
   几个点"的读数。

两次运行的账本不可能逐事件相同（采样是随机的），所以这里**只做分布比较**，不比事件集：
事件集对比是 `scripts/compare_runs.py` 的事，它需要的是同一份 prompt 与同一份随机源。

    .venv/Scripts/python.exe scripts/compare_capability_input.py \
        --control shanghai_apartment_09280504 --treatment shanghai_apartment_09280506
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def resolve_run(name: str) -> Path:
    path = Path(name)
    if not path.is_absolute():
        path = ROOT / "data" / name
    return path


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _gains(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """每条活动的技能增益，附上周次（用于分层）。"""
    out: List[Dict[str, Any]] = []
    for row in rows:
        outcome = row.get("outcome") or {}
        gains = outcome.get("delta_skills") or {}
        if not isinstance(gains, dict):
            gains = {}
        entry: Dict[str, Any] = {}
        for name, value in gains.items():
            try:
                entry[str(name)] = float(value)
            except (TypeError, ValueError):
                continue
        out.append({"time": str(row.get("time") or ""), "gains": entry})
    return out


def _week_index(time_str: str) -> Optional[int]:
    """`Y2020-W07-activity-D1` → 全局周序号（年*10+周），只用于分层。"""
    parts = time_str.split("-")
    if len(parts) < 2 or not parts[0].startswith("Y") or not parts[1].startswith("W"):
        return None
    try:
        year = int(parts[0][1:])
        week = int(parts[1][1:])
    except ValueError:
        return None
    return year * 10 + week


def _summary(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"samples": 0}
    return {
        "samples": len(values),
        "mean": round(statistics.fmean(values), 4),
        "median": round(statistics.median(values), 4),
        "max": round(max(values), 4),
        "sum": round(sum(values), 4),
    }


def profile_run(run_dir: Path) -> Dict[str, Any]:
    """一个运行里每个角色的技能增益分布。"""
    personas: Dict[str, Any] = {}
    persona_root = run_dir / "persona"
    if not persona_root.exists():
        return {"run": run_dir.name, "personas": personas}

    for path in sorted(persona_root.iterdir(), key=lambda p: p.name):
        if not path.is_dir():
            continue
        activities = _gains(read_jsonl(path / "activity.jsonl"))
        if not activities:
            continue
        per_skill: Dict[str, List[float]] = {}
        grants: List[float] = []
        activities_with_gains = 0
        for activity in activities:
            if activity["gains"]:
                activities_with_gains += 1
            for skill, value in activity["gains"].items():
                per_skill.setdefault(skill, []).append(value)
                grants.append(value)

        # 前半程练习量 → 该技能在后半程的增益（唯一能区分"读懂块"与"少发点"的读数）
        weeks = sorted(
            {w for w in (_week_index(a["time"]) for a in activities) if w is not None}
        )
        midpoint = weeks[len(weeks) // 2] if weeks else None
        first_half: Dict[str, int] = {}
        second_half: Dict[str, List[float]] = {}
        for activity in activities:
            week = _week_index(activity["time"])
            if week is None or midpoint is None:
                continue
            target = first_half if week < midpoint else second_half
            for skill, value in activity["gains"].items():
                if target is first_half:
                    first_half[skill] = first_half.get(skill, 0) + (1 if value > 0 else 0)
                else:
                    target.setdefault(skill, []).append(value)

        # 按前半程练习量分层：0 次（没练过）、1-4 次、5 次以上
        strata: Dict[str, List[float]] = {"practised_0": [], "practised_1_4": [], "practised_5_plus": []}
        for skill, values in second_half.items():
            count = first_half.get(skill, 0)
            key = "practised_0" if count == 0 else ("practised_1_4" if count < 5 else "practised_5_plus")
            strata[key].extend(values)

        personas[path.name] = {
            "activities": len(activities),
            "activities_with_gains": activities_with_gains,
            "empty_share": round(1 - activities_with_gains / len(activities), 4),
            "skills_practised": len(per_skill),
            "grants": _summary(grants),
            "zero_grants": sum(1 for v in grants if v == 0.0),
            "zero_share": round(
                sum(1 for v in grants if v == 0.0) / len(grants), 4
            ) if grants else None,
            "per_skill": {
                skill: _summary(values)
                for skill, values in sorted(per_skill.items())
            },
            "second_half_by_first_half_practice": {
                key: _summary(values) for key, values in strata.items()
            },
            "midpoint_week": midpoint,
        }
    return {"run": run_dir.name, "personas": personas}


def _delta(control: Optional[float], treatment: Optional[float]) -> Optional[float]:
    if control is None or treatment is None:
        return None
    return round(treatment - control, 4)


def compare(control: Dict[str, Any], treatment: Dict[str, Any]) -> Dict[str, Any]:
    rows = []
    for name, c in sorted(control["personas"].items()):
        t = treatment["personas"].get(name)
        if not t:
            rows.append({"persona": name, "missing_in_treatment": True})
            continue
        rows.append(
            {
                "persona": name,
                "activities": [c["activities"], t["activities"]],
                "grants_per_activity": [
                    round(c["grants"]["sum"] / c["activities"], 4),
                    round(t["grants"]["sum"] / t["activities"], 4),
                ],
                "grant_mean": _delta(c["grants"].get("mean"), t["grants"].get("mean")),
                "grant_median": _delta(c["grants"].get("median"), t["grants"].get("median")),
                "grant_sum": _delta(c["grants"].get("sum"), t["grants"].get("sum")),
                "empty_share": _delta(c.get("empty_share"), t.get("empty_share")),
                "zero_share": _delta(c.get("zero_share"), t.get("zero_share")),
                "skills_practised": [c["skills_practised"], t["skills_practised"]],
                "strata_mean": {
                    key: [
                        c["second_half_by_first_half_practice"].get(key, {}).get("mean"),
                        t["second_half_by_first_half_practice"].get(key, {}).get("mean"),
                    ]
                    for key in ("practised_0", "practised_1_4", "practised_5_plus")
                },
            }
        )
    return {"personas": rows}


def render_markdown(result: Dict[str, Any], control_name: str, treatment_name: str) -> str:
    lines = [
        f"# capability_input A/B：{control_name} → {treatment_name}",
        "",
        "只比较技能增益的分布；两次运行的账本不逐事件相同（采样随机），事件集对比见 compare_runs.py。",
        "",
    ]
    for row in result["personas"]:
        if row.get("missing_in_treatment"):
            lines.append(f"- {row['persona']}：对照组有、实验组没有")
            continue
        lines.append(f"## {row['persona']}")
        lines.append("")
        lines.append("| 读数 | 对照组 | 实验组 | 差 |")
        lines.append("|---|---|---|---|")
        lines.append(
            f"| 活动数 | {row['activities'][0]} | {row['activities'][1]} | - |"
        )
        lines.append(
            f"| 每条活动总增益 | {row['grants_per_activity'][0]} | {row['grants_per_activity'][1]} | {_delta(row['grants_per_activity'][0], row['grants_per_activity'][1])} |"
        )
        lines.append(f"| 单次增益均值 | - | - | {row['grant_mean']} |")
        lines.append(f"| 单次增益中位数 | - | - | {row['grant_median']} |")
        lines.append(f"| 零增益占比 | - | - | {row['zero_share']} |")
        lines.append(f"| 无增益活动占比 | - | - | {row['empty_share']} |")
        lines.append("")
        lines.append("| 前半程练习量分层（后半程增益均值） | 对照组 | 实验组 |")
        lines.append("|---|---|---|")
        for key, pair in row["strata_mean"].items():
            lines.append(f"| {key} | {pair[0]} | {pair[1]} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True)
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--markdown", default=None)
    args = parser.parse_args()

    control_dir, treatment_dir = resolve_run(args.control), resolve_run(args.treatment)
    for path in (control_dir, treatment_dir):
        if not path.exists():
            raise SystemExit(f"run not found: {path}")

    control, treatment = profile_run(control_dir), profile_run(treatment_dir)
    result = {
        "control": control,
        "treatment": treatment,
        "comparison": compare(control, treatment),
    }
    if args.output:
        Path(args.output).write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(Path(args.output).resolve())
    if args.markdown:
        Path(args.markdown).write_text(
            render_markdown(result["comparison"], control_dir.name, treatment_dir.name),
            encoding="utf-8",
        )
        print(Path(args.markdown).resolve())
    if not args.output and not args.markdown:
        print(json.dumps(result["comparison"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
