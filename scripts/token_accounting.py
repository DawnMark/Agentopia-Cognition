#!/usr/bin/env python
"""Token 账：这次开发带来的新功能，比原版 Agentopia 多花了多少 token、花在哪里。

## 口径（先说清楚，否则数字会被误读）

- token 数不是 API 返回的 `usage`，而是用 `num_tokens_from_string`（cl100k_base）
  对 prompt / completion **文本**估出来的——与运行目录里 `generation/`、`god/` 日志
  用的是同一个函数，所以可以直接相加，但它衡量的是"文本长度"，不是账单；
- 三本账：
  1. `persona/*/generation/` —— 角色自己的每一次生成（计划、联系、活动、反思、结算）；
  2. `god/*` —— 上帝模型的每一次评估（活动、公开事件、职位、人设更新）；
  3. `persona/*/cognition/token_usage.jsonl` —— **认知层自己的抽取调用**
     （方法抽取、记忆合并、Idea 生成、Idea→方法转换）。第三本账是这一轮才补上的：
     这些调用此前不写任何生成日志，于是"新功能多花了多少"根本无法回答。
- 角色模型与上帝模型分开统计（它们是不同的模型与不同的价钱）。

    .venv/Scripts/python.exe scripts/token_accounting.py --run shanghai_apartment_09281239
    .venv/Scripts/python.exe scripts/token_accounting.py --compare \
        --branch shanghai_apartment_09281239 --upstream shanghai_apartment_09281336
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 角色侧生成记录的 `record_id` 形态：Y2020-W01-<stage>#n
_RECORD_ID = re.compile(r"Y\d+-W\d+-([a-zA-Z_]+)")


def resolve_run(name: str) -> Path:
    path = Path(name)
    return path if path.is_absolute() else ROOT / "data" / name


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


def _add(
    table: Dict[str, Dict[str, Any]],
    key: str,
    *,
    side: str,
    calls: int,
    input_tokens: int,
    output_tokens: int,
) -> None:
    entry = table.setdefault(
        key, {"side": side, "calls": 0, "input_tokens": 0, "output_tokens": 0}
    )
    entry["calls"] += calls
    entry["input_tokens"] += input_tokens
    entry["output_tokens"] += output_tokens


def profile_run(run_dir: Path) -> Dict[str, Any]:
    """一个运行的三本 token 账。"""
    table: Dict[str, Dict[str, Any]] = {}
    weeks: Dict[str, set] = defaultdict(set)

    # 1) 角色侧
    for path in run_dir.glob("persona/*/generation/year=*/week=*.jsonl"):
        for row in read_jsonl(path):
            match = _RECORD_ID.match(str(row.get("record_id") or ""))
            stage = match.group(1) if match else "other"
            _add(
                table,
                f"role.{stage}",
                side="role",
                calls=1,
                input_tokens=int(row.get("input_tokens") or 0),
                output_tokens=int(row.get("output_tokens") or 0),
            )
            week = str(row.get("time") or "")[:11]
            if week:
                weeks[f"role.{stage}"].add(week)

    # 2) 上帝侧
    for directory in sorted(run_dir.glob("god/*")):
        if not directory.is_dir():
            continue
        for path in directory.glob("year=*/week=*.jsonl"):
            for row in read_jsonl(path):
                _add(
                    table,
                    f"god.{directory.name}",
                    side="god",
                    calls=1,
                    input_tokens=int(row.get("input_tokens") or 0),
                    output_tokens=int(row.get("output_tokens") or 0),
                )

    # 3) 认知层抽取（新功能）
    cognition_rows = [
        row
        for path in run_dir.glob("persona/*/cognition/token_usage.jsonl")
        for row in read_jsonl(path)
    ]
    for row in cognition_rows:
        _add(
            table,
            f"cognition.{row.get('feature') or 'unknown'}",
            side="cognition",
            calls=int(row.get("calls") or 1),
            input_tokens=int(row.get("input_tokens") or 0),
            output_tokens=int(row.get("output_tokens") or 0),
        )

    for entry in table.values():
        entry["input_per_call"] = (
            round(entry["input_tokens"] / entry["calls"], 1) if entry["calls"] else None
        )

    # 按周汇总（有周次的才计），用于"每周多花多少"
    weeks_count = len(
        {
            str(row.get("time") or "")[:11]
            for path in run_dir.glob("persona/*/generation/year=*/week=*.jsonl")
            for row in read_jsonl(path)
        }
    )
    totals = {
        "calls": sum(e["calls"] for e in table.values()),
        "input_tokens": sum(e["input_tokens"] for e in table.values()),
        "output_tokens": sum(e["output_tokens"] for e in table.values()),
    }
    totals["total_tokens"] = totals["input_tokens"] + totals["output_tokens"]
    by_side: Dict[str, Dict[str, int]] = {}
    for entry in table.values():
        side = by_side.setdefault(
            entry["side"], {"calls": 0, "input_tokens": 0, "output_tokens": 0}
        )
        side["calls"] += entry["calls"]
        side["input_tokens"] += entry["input_tokens"]
        side["output_tokens"] += entry["output_tokens"]
    return {
        "run": run_dir.name,
        "weeks_observed": weeks_count,
        "by_feature": {key: table[key] for key in sorted(table)},
        "by_side": by_side,
        "totals": totals,
        "cognition_log_present": bool(cognition_rows),
    }


def compare(branch: Dict[str, Any], upstream: Dict[str, Any]) -> Dict[str, Any]:
    rows = []
    keys = sorted(set(branch["by_feature"]) | set(upstream["by_feature"]))
    for key in keys:
        b = branch["by_feature"].get(key)
        u = upstream["by_feature"].get(key)
        rows.append(
            {
                "feature": key,
                "branch_calls": (b or {}).get("calls", 0),
                "upstream_calls": (u or {}).get("calls", 0),
                "branch_input": (b or {}).get("input_tokens", 0),
                "upstream_input": (u or {}).get("input_tokens", 0),
                "delta_input": (b or {}).get("input_tokens", 0)
                - (u or {}).get("input_tokens", 0),
                "branch_output": (b or {}).get("output_tokens", 0),
                "upstream_output": (u or {}).get("output_tokens", 0),
            }
        )
    b_tot, u_tot = branch["totals"], upstream["totals"]
    return {
        "rows": rows,
        "totals": {
            "branch": {**b_tot, "weeks": branch["weeks_observed"]},
            "upstream": {**u_tot, "weeks": upstream["weeks_observed"]},
            "delta_input": b_tot["input_tokens"] - u_tot["input_tokens"],
            "delta_output": b_tot["output_tokens"] - u_tot["output_tokens"],
            "delta_calls": b_tot["calls"] - u_tot["calls"],
            "input_ratio": (
                round(b_tot["input_tokens"] / u_tot["input_tokens"], 4)
                if u_tot["input_tokens"]
                else None
            ),
        },
    }


def render_markdown(result: Dict[str, Any], *, branch: str, upstream: Optional[str]) -> str:
    lines: List[str] = []
    if upstream:
        lines += [
            f"# Token 账：{branch}（本分支） vs {upstream}（原版基线）",
            "",
            "| 特征 | 本分支调用 | 原版调用 | 本分支输入 | 原版输入 | 输入差 |",
            "|---|---|---|---|---|---|",
        ]
        for row in result["rows"]:
            lines.append(
                f"| {row['feature']} | {row['branch_calls']} | {row['upstream_calls']} | "
                f"{row['branch_input']:,} | {row['upstream_input']:,} | "
                f"{row['delta_input']:+,} |"
            )
        t = result["totals"]
        lines += [
            "",
            "| 合计 | 调用 | 输入 token | 输出 token |",
            "|---|---|---|---|",
            f"| 本分支 | {t['branch']['calls']:,} | {t['branch']['input_tokens']:,} | "
            f"{t['branch']['output_tokens']:,} |",
            f"| 原版基线 | {t['upstream']['calls']:,} | {t['upstream']['input_tokens']:,} | "
            f"{t['upstream']['output_tokens']:,} |",
            f"| 差 | {t['delta_calls']:+,} | {t['delta_input']:+,} | {t['delta_output']:+,} |",
            "",
            f"- 输入 token 比：**{t['input_ratio']}**（本分支 / 原版）",
            f"- 两边观察到的周数：{t['branch']['weeks']} / {t['upstream']['weeks']}",
            "",
        ]
    else:
        lines += [
            f"# Token 账：{result['run']}",
            "",
            f"- 观察周数：{result['weeks_observed']}",
            f"- 认知层抽取账本存在：{result['cognition_log_present']}",
            "",
            "| 特征 | 侧 | 调用 | 输入 | 输出 | 每次输入 |",
            "|---|---|---|---|---|---|",
        ]
        for key, entry in result["by_feature"].items():
            lines.append(
                f"| {key} | {entry['side']} | {entry['calls']:,} | "
                f"{entry['input_tokens']:,} | {entry['output_tokens']:,} | "
                f"{entry['input_per_call']:,} |"
            )
        totals = result["totals"]
        lines += [
            "",
            f"合计：{totals['calls']:,} 次调用、输入 {totals['input_tokens']:,}、"
            f"输出 {totals['output_tokens']:,} tokens。",
            "",
        ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=None, help="单个运行目录（或 data/ 下的名字）")
    parser.add_argument("--compare", action="store_true", help="对比两个运行")
    parser.add_argument("--branch", default=None)
    parser.add_argument("--upstream", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--markdown", default=None)
    args = parser.parse_args()

    if args.compare:
        if not args.branch or not args.upstream:
            raise SystemExit("--compare 需要 --branch 与 --upstream")
        branch, upstream = resolve_run(args.branch), resolve_run(args.upstream)
        for path in (branch, upstream):
            if not path.exists():
                raise SystemExit(f"run not found: {path}")
        result: Dict[str, Any] = {
            "branch": profile_run(branch),
            "upstream": profile_run(upstream),
        }
        result["comparison"] = compare(result["branch"], result["upstream"])
        markdown = render_markdown(
            result["comparison"], branch=branch.name, upstream=upstream.name
        )
    else:
        if not args.run:
            raise SystemExit("需要 --run 或 --compare")
        run_dir = resolve_run(args.run)
        if not run_dir.exists():
            raise SystemExit(f"run not found: {run_dir}")
        result = profile_run(run_dir)
        markdown = render_markdown(result, branch=run_dir.name, upstream=None)

    if args.output:
        Path(args.output).write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(Path(args.output).resolve())
    if args.markdown:
        Path(args.markdown).write_text(markdown, encoding="utf-8")
        print(Path(args.markdown).resolve())
    if not args.output and not args.markdown:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
