#!/usr/bin/env python
"""反事实探针：把"高能力"注入**真实的** God 评估 prompt，看增益会不会随之下降。

存在的理由（阶段 6，用户决策 2026-09-27）：`capability_input` 的 10 周 A/B 读数是"没有
可测差别"，但期末最高有效能力只有 11/100——块里"能力已经很高就别再涨"的触发条件整轮
都没成立，所以那次 A/B 说明不了指令的效力。要分开验证两件事：

1. **指令有没有效力**（本脚本）：同一条真实 prompt、只改能力块里的数字，看 `delta_skills`；
2. **数值能不能达到"高"**（离线算）：PROFICIENCY_HALF_PRACTICES 与三因子折扣决定上限。

设计上刻意做成**配对**：每条 prompt 在三档下各跑 N 次，只有块文本不同，其余逐字节相同。

    档位            注入的块
    off             不注入（就是当年真实发给 God 的那一条）
    real            用**当前代码**从该角色 2 年账本重建的真实块
    high            同样的措辞、合成的"高能力"数字

调用是花钱的，所以默认很小：3 条 prompt × 3 档 × 5 次 = 45 次。

    .venv/Scripts/python.exe scripts/capability_input_probe.py \
        --run shanghai_apartment_09272220 --persona 上官霄月
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 能力块在 prompt 里的落点：`### Skills` 段落之后、`## Current State` 之前，
# 与 `DataManager.read_skills_prompt` 的实际拼法一致。
INSERT_BEFORE = "\n\n## Current State"
SKILLS_MARKER = "### Skills"

HIGH_BLOCK_TEMPLATE = (
    "- Practised capability (derived from this character's own activity ledger, 0-100. "
    "It counts how often the character actually practised each skill and how well its own "
    "methods worked out, so unlike the skill numbers above it does not include the persona's "
    "starting endowment):\n"
    "{rows}"
    "  - Use this when deciding how much a character can still gain: a high effective "
    "capability means the skill is already well practised and further gains should be small, "
    "while a skill that is not listed has not been practised at all and can still gain freely."
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def resolve_run(name: str) -> Path:
    path = Path(name)
    if not path.is_absolute():
        path = ROOT / "data" / name
    return path


def real_block(run_dir: Path, persona: str, skills: Dict[str, Any]) -> str:
    """用**当前代码**从该角色的账本重建真实能力块（与运行里那条同源）。"""
    from src.agents.data_manager import DataManager
    from src.config import get_config
    from src.world.clock import Clock

    cognition = get_config()["world"].setdefault("cognition", {})
    cognition["capability_input"] = True
    dm = DataManager(
        char=persona,
        world=run_dir.name,
        clock=Clock(start_year=2020, start_week=1),
        model="probe",
    )
    return dm.capability_input_block(skills)


ROW_PATTERN = re.compile(
    r"practised (\d+)x, effective capability (\d+)/100 "
    r"\(proficiency (\d+)/100, methods x([0-9.]+)\)"
)
SKILL_PATTERN = re.compile(r"^  - ([^:]+): practised", re.M)


def high_variant(block: str, *, points: int = 55, practices: int = 60) -> str:
    """把真实块里的**数字**换掉，技能列表、顺序、措辞一字不动。

    只改数字是刻意的：这样三档之间的差别就是"God 看到的数值"，而不是"prompt 长什么样"。
    """
    return ROW_PATTERN.sub(
        lambda m: (
            f"practised {practices}x, effective capability {points}/100 "
            f"(proficiency {min(95, points + 15)}/100, methods x1.00)"
        ),
        block,
    )


def skills_in_block(block: str) -> List[str]:
    """块里列出的技能（按块的顺序）——测量这几项就够，不必看全部增益。"""
    return [m.group(1).strip() for m in SKILL_PATTERN.finditer(block)]


def inject(prompt: str, block: str) -> str:
    """把块插到 `### Skills` 段落后面；没有这个段落就原样返回。"""
    if not block or SKILLS_MARKER not in prompt:
        return prompt
    if INSERT_BEFORE not in prompt:
        return prompt
    head, tail = prompt.split(INSERT_BEFORE, 1)
    return f"{head}\n{block}{INSERT_BEFORE}{tail}"


def _skills_from_response(content: str) -> List[str]:
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        return []
    gains = payload.get("delta_skills") or {}
    return [str(k) for k in gains] if isinstance(gains, dict) else []


def pick_prompts(run_dir: Path, persona: str, limit: int) -> List[Dict[str, Any]]:
    """挑出"后半程、有技能增益"的真实 solo 评估 prompt（状态最成熟的那几条）。"""
    rows: List[Dict[str, Any]] = []
    for path in sorted(run_dir.glob("god/solo_activity/year=*/week=*.jsonl")):
        for row in read_jsonl(path):
            inputs = row.get("inputs") or []
            if not inputs or inputs[0].get("role") != "system":
                continue
            prompt = str(inputs[0].get("content") or "")
            if "You are " + persona not in prompt or SKILLS_MARKER not in prompt:
                continue
            outputs = row.get("outputs") or []
            content = str(outputs[0].get("content") or "") if outputs else ""
            if not _skills_from_response(content):
                continue
            rows.append({"time": row.get("time"), "prompt": prompt, "recorded": content})
    rows.sort(key=lambda r: str(r["time"]), reverse=True)
    return rows[:limit]


def parse_gains(content: Any) -> Dict[str, float]:
    if not isinstance(content, str):
        return {}
    match = re.search(r"\{.*\}", content, re.S)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
    except ValueError:
        return {}
    gains = payload.get("delta_skills") or {}
    out: Dict[str, float] = {}
    if isinstance(gains, dict):
        for key, value in gains.items():
            try:
                out[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
    return out


def summarise(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"samples": 0}
    return {
        "samples": len(values),
        "mean": round(statistics.fmean(values), 4),
        "median": round(statistics.median(values), 4),
        "max": round(max(values), 4),
        "zero_share": round(sum(1 for v in values if v == 0.0) / len(values), 4),
    }


def run_probe(args: argparse.Namespace) -> Dict[str, Any]:
    from src.config import get_config, load_config
    from src.utils import _get_response

    load_config("config.json")
    run_dir = resolve_run(args.run)
    if not run_dir.exists():
        raise SystemExit(f"run not found: {run_dir}")

    # 角色当前的技能表（用来重建真实块）
    state_rows = read_jsonl(run_dir / "persona" / args.persona / "state.jsonl")
    # 状态行把状态放在 `content` 里（账本身份字段在外层）。
    last = (state_rows[-1].get("content") if state_rows else {}) or {}
    skills = last.get("skills") or {}
    block_real = real_block(run_dir, args.persona, skills)
    focus = skills_in_block(block_real)

    # 挑 prompt：优先挑"练到的那项技能就在块里"的——否则注入的数字对这条活动没有意义
    # （实测第一条被挑中的是射箭，而射箭正好落在块的"还有 3 个没列出来"里）。
    candidates = pick_prompts(run_dir, args.persona, args.prompts * 6)
    prompts = [
        item
        for item in candidates
        if set(parse_gains(item["recorded"])) & set(focus)
    ][: args.prompts]
    if not prompts:
        prompts = candidates[: args.prompts]
    if not prompts:
        raise SystemExit(f"no solo-activity prompt found for {args.persona} in {run_dir}")
    block_high = high_variant(block_real)
    block_extreme = high_variant(block_real, points=95, practices=200)

    results: List[Dict[str, Any]] = []
    for item in prompts:
        recorded = parse_gains(item["recorded"])
        # 这条活动练到、且**块里列着**的技能：只有这些技能的增益才可能被块影响。
        changed = [s for s in recorded if s in focus] or list(recorded)
        record: Dict[str, Any] = {
            "time": item["time"],
            "recorded": recorded,
            "measured_skills": changed,
            "arms": {},
        }
        for arm, block in (
            ("off", ""),
            ("real", block_real),
            ("high", block_high),
            ("extreme", block_extreme),
        ):
            prompt = inject(item["prompt"], block) if block else item["prompt"]
            gains: List[Dict[str, float]] = []
            for _ in range(args.repeats):
                # `get_response_with_retry` 自己管 `force_regenerate`，这里通过
                # `max_retry=0` 之外没有别的钩子——所以直接调 `_get_response`，
                # 每次都强制重算，保证 N 次是不同的采样而不是同一个缓存命中。
                response = _get_response(
                    model=get_config()["god_model"],
                    messages=[{"role": "system", "content": prompt}],
                    cache_file=args.cache_file,
                    force_regenerate=True,
                )
                gains.append(parse_gains(response))  # `_get_response` 返回文本
            totals = [sum(g.values()) for g in gains]
            # 只测"这条活动真正练到的技能"——块里的数字只对这个技能有语义。
            focus_values = [g.get(skill, 0.0) for g in gains for skill in changed]
            record["arms"][arm] = {
                "total": summarise([float(t) for t in totals]),
                "focus_skills": focus,
                "focus": summarise([float(v) for v in focus_values]),
                "raw": gains,
            }
            print(
                f"[probe] {item['time']} arm={arm} "
                f"total_mean={record['arms'][arm]['total'].get('mean')} "
                f"focus_mean={record['arms'][arm]['focus'].get('mean')}"
            )
        results.append(record)

    payload: Dict[str, Any] = {
        "run": run_dir.name,
        "persona": args.persona,
        "model": get_config()["god_model"],
        "repeats": args.repeats,
        "focus_skills": focus,
        "real_block": block_real,
        "high_block": block_high,
        "extreme_block": block_extreme,
        "prompts": results,
    }
    payload["summary"] = summarise_arms(payload)
    return payload


ARMS = ("off", "real", "high", "extreme")


def summarise_arms(result: Dict[str, Any]) -> Dict[str, Any]:
    """按档汇总全部样本，并给出与 `off` 的差与 Welch t。

    样本是 `{1, 2}` 这种小整数、每档只有几十条，所以除了均值还要给 t：
    **均值差 0.2 在这点样本量下什么都不是**，不把 t 写出来就会被读成"有效果"。
    """
    import math

    per_arm: Dict[str, List[float]] = {arm: [] for arm in ARMS}
    paired: Dict[str, List[float]] = {arm: [] for arm in ARMS if arm != "off"}
    for item in result["prompts"]:
        measured = item.get("measured_skills") or []
        for arm in ARMS:
            samples = item["arms"][arm]["raw"]
            values = [
                float(g.get(skill, 0.0)) for g in samples for skill in measured
            ]
            per_arm[arm].extend(values)
            if arm != "off":
                off_values = [
                    float(g.get(skill, 0.0))
                    for g in item["arms"]["off"]["raw"]
                    for skill in measured
                ]
                paired[arm].append(
                    round(statistics.fmean(values) - statistics.fmean(off_values), 4)
                )

    def welch(a: List[float], b: List[float]) -> Optional[float]:
        if len(a) < 2 or len(b) < 2:
            return None
        sa, sb = statistics.variance(a), statistics.variance(b)
        denom = math.sqrt(sa / len(a) + sb / len(b))
        if denom == 0:
            return None
        return round((statistics.fmean(a) - statistics.fmean(b)) / denom, 3)

    off = per_arm["off"]
    return {
        "arms": {
            arm: {**summarise(values), "t_vs_off": welch(values, off)}
            for arm, values in per_arm.items()
        },
        "paired_prompt_means": paired,
    }


def render_markdown(result: Dict[str, Any]) -> str:
    lines = [
        f"# capability_input 反事实探针：{result['run']} / {result['persona']}",
        "",
        f"- 模型：{result['model']}；每档重复：{result['repeats']} 次",
        f"- 块里列出的技能：{'、'.join(result['focus_skills'])}",
        "- 三档只差**数字**：off 不注入、real 用当前代码重建的真实值、"
        "high 把数字换成 55/100、extreme 换成 95/100（技能列表与措辞完全一致）",
        "",
        "| prompt | 档位 | 总增益均值 | 总增益中位数 | 关注技能增益均值 | 零增益占比 |",
        "|---|---|---|---|---|---|",
    ]
    for item in result["prompts"]:
        for arm in ("off", "real", "high", "extreme"):
            entry = item["arms"][arm]
            lines.append(
                f"| {item['time']} | {arm} | {entry['total'].get('mean')} | "
                f"{entry['total'].get('median')} | {entry['focus'].get('mean')} | "
                f"{entry['total'].get('zero_share')} |"
            )
    summary = result.get("summary") or summarise_arms(result)
    lines += [
        "",
        "## 汇总（所有 prompt 的全部样本）",
        "",
        "| 档位 | 样本 | 均值 | 中位数 | 零增益占比 | 与 off 的 Welch t |",
        "|---|---|---|---|---|---|",
    ]
    for arm in ARMS:
        entry = summary["arms"][arm]
        lines.append(
            f"| {arm} | {entry.get('samples')} | {entry.get('mean')} | "
            f"{entry.get('median')} | {entry.get('zero_share')} | {entry.get('t_vs_off')} |"
        )
    lines += [
        "",
        "每条 prompt 上（该档均值 − off 均值）："
        + "；".join(
            f"{arm}: {values}"
            for arm, values in summary["paired_prompt_means"].items()
        ),
        "",
        "## 注入的真实块",
        "",
        "```text",
        result["real_block"] or "(空：账本里没有练过的技能)",
        "```",
        "",
        "## 注入的高能力块（high：55/100）",
        "",
        "```text",
        result["high_block"],
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="shanghai_apartment_09272220")
    parser.add_argument("--persona", default="上官霄月")
    parser.add_argument("--prompts", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cache-file", default="llm_cache/probe_capability_input.pkl")
    parser.add_argument("--output", default=None)
    parser.add_argument("--markdown", default=None)
    parser.add_argument(
        "--from-json",
        default=None,
        help="不再调用 LLM：读回已有的探针结果，重算汇总并重新渲染",
    )
    args = parser.parse_args()

    if args.from_json:
        result = json.loads(Path(args.from_json).read_text(encoding="utf-8"))
        result["summary"] = summarise_arms(result)
    else:
        result = run_probe(args)
    if args.output:
        Path(args.output).write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(Path(args.output).resolve())
    if args.markdown:
        Path(args.markdown).write_text(render_markdown(result), encoding="utf-8")
        print(Path(args.markdown).resolve())
    if not args.output and not args.markdown:
        print(json.dumps(result, ensure_ascii=False, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
