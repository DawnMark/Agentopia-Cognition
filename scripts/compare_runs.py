#!/usr/bin/env python
"""对照两次运行：功能清单、配置差异、账本事件集合。

存在的理由：设计正文不变量 #10 要求"新系统必须可以通过配置关闭，并保留旧行为作为
对照组"，而对照组只有在**能说清楚两次运行差在哪**的时候才有意义。这个脚本不调用任何
LLM，只读两次运行已经落盘的东西：

    .venv/Scripts/python.exe scripts/compare_runs.py --a <runA> --b <runB>

输出三段：

1. **生效的功能清单**：两次运行各自开了哪些新功能（从运行目录里保存的 config.json 读，
   所以它记录的是当时实际生效的东西，不是今天的默认值）；
2. **配置差异**：两次运行之间所有不一致的配置项——这同时也是"prompt 差在哪"的摘要，
   因为 prompt 完全由配置与人设派生；
3. **账本对比**：每条流的记录数、以及按 `ledger_event_id` 比对的事件集合
   （只在一边出现的 id 会列出来）。阶段 -1 之后每条账本记录都带稳定身份，
   所以"两次运行是不是同一批事件"是可以精确回答的。

`--ignore-stream` 可以排除派生视图类文件（例如 `persona/*/cognition/*`），
用法与 `scripts/verify_replay.py` 一致。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DERIVED_VIEW_SUFFIXES = ("/cognition/views/",)

# 对照运行时最该先看的几项：它们直接决定 prompt 与角色行为。
PROMPT_RELEVANT_KEYS = (
    "world.name",
    "world.language",
    "world.cognition.baseline",
    "world.upstream_compat.routine_prompt",
    "world.upstream_compat.seed_source",
    "world.vitality_recovery.enabled",
    "world.vitality_recovery.daily_recovery",
    "role_model",
    "god_model",
    "temperature",
)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def flatten(value: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            out.update(flatten(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        out[prefix] = json.dumps(value, ensure_ascii=False)
    else:
        out[prefix] = value
    return out


def stream_files(run_dir: Path, ignore: Iterable[str]) -> List[Path]:
    out: List[Path] = []
    for path in sorted(run_dir.rglob("*.jsonl")):
        rel = path.relative_to(run_dir).as_posix()
        if any(rel.startswith(pattern.rstrip("*")) or pattern in rel for pattern in ignore):
            continue
        if any(suffix in rel for suffix in DERIVED_VIEW_SUFFIXES):
            continue
        out.append(path)
    return out


def event_ids(path: Path) -> Tuple[int, Set[str], Counter]:
    """Rows, their `ledger_event_id`s, and a type histogram (when present)."""
    rows = 0
    ids: Set[str] = set()
    types: Counter = Counter()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return 0, ids, types
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows += 1
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            event_id = record.get("ledger_event_id") or record.get("event_id")
            if event_id:
                ids.add(str(event_id))
            if record.get("type"):
                types[str(record["type"])] += 1
    return rows, ids, types


def effective_features(config: Dict[str, Any]) -> List[str]:
    from src.world.baseline import effective_features as lines

    return [line.name for line in lines(config) if line.enabled]


def compare(run_a: Path, run_b: Path, ignore: List[str]) -> Dict[str, Any]:
    cfg_a = read_json(run_a / "config.json", {}) or {}
    cfg_b = read_json(run_b / "config.json", {}) or {}

    flat_a, flat_b = flatten(cfg_a), flatten(cfg_b)
    keys = sorted(set(flat_a) | set(flat_b))
    config_diff = {
        key: {"a": flat_a.get(key, "<absent>"), "b": flat_b.get(key, "<absent>")}
        for key in keys
        if flat_a.get(key, "<absent>") != flat_b.get(key, "<absent>")
    }

    files_a = {p.relative_to(run_a).as_posix(): p for p in stream_files(run_a, ignore)}
    files_b = {p.relative_to(run_b).as_posix(): p for p in stream_files(run_b, ignore)}
    streams: Dict[str, Any] = {}
    for rel in sorted(set(files_a) | set(files_b)):
        rows_a, ids_a, types_a = (
            event_ids(files_a[rel]) if rel in files_a else (0, set(), Counter())
        )
        rows_b, ids_b, types_b = (
            event_ids(files_b[rel]) if rel in files_b else (0, set(), Counter())
        )
        streams[rel] = {
            "rows": {"a": rows_a, "b": rows_b},
            "events": {"a": len(ids_a), "b": len(ids_b)},
            "only_in_a": sorted(ids_a - ids_b),
            "only_in_b": sorted(ids_b - ids_a),
            "shared": len(ids_a & ids_b),
            "types_a": dict(sorted(types_a.items())),
            "types_b": dict(sorted(types_b.items())),
        }

    return {
        "a": {"run": run_a.name, "features_on": effective_features(cfg_a)},
        "b": {"run": run_b.name, "features_on": effective_features(cfg_b)},
        "config_diff": config_diff,
        "streams": streams,
    }


def render(report: Dict[str, Any]) -> str:
    a, b = report["a"], report["b"]
    lines = [
        f"# Run comparison · {a['run']}  vs  {b['run']}",
        "",
        f"- A features on: {', '.join(a['features_on']) or 'none'}",
        f"- B features on: {', '.join(b['features_on']) or 'none'}",
        "",
        f"## Config differences ({len(report['config_diff'])})",
        "",
        "| key | A | B |",
        "|---|---|---|",
    ]
    for key, values in sorted(report["config_diff"].items()):
        lines.append(f"| `{key}` | {values['a']} | {values['b']} |")

    identical, differing = [], []
    for rel, info in report["streams"].items():
        (identical if not info["only_in_a"] and not info["only_in_b"] else differing).append(
            (rel, info)
        )

    lines += [
        "",
        f"## Ledger streams",
        "",
        f"- identical event sets: {len(identical)} streams",
        f"- differing: {len(differing)} streams",
        "",
    ]
    if differing:
        lines += [
            "| stream | rows A/B | events A/B | only A | only B | shared |",
            "|---|---|---|---|---|---|",
        ]
        for rel, info in differing:
            lines.append(
                f"| `{rel}` | {info['rows']['a']}/{info['rows']['b']} | "
                f"{info['events']['a']}/{info['events']['b']} | "
                f"{len(info['only_in_a'])} | {len(info['only_in_b'])} | {info['shared']} |"
            )
        lines.append("")
        for rel, info in differing[:5]:
            if info["only_in_a"][:3] or info["only_in_b"][:3]:
                lines.append(f"- `{rel}`: A-only {info['only_in_a'][:3]}, B-only {info['only_in_b'][:3]}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True, help="Run directory name under data/ or a path")
    parser.add_argument("--b", required=True)
    parser.add_argument(
        "--ignore-stream",
        action="append",
        default=[],
        help="Substring of a stream path to exclude (repeatable)",
    )
    parser.add_argument("--output", help="Optional JSON report path")
    parser.add_argument("--markdown", help="Optional markdown summary path")
    args = parser.parse_args()

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else ROOT / "data" / path

    run_a, run_b = resolve(args.a), resolve(args.b)
    for run in (run_a, run_b):
        if not run.exists():
            raise SystemExit(f"run not found: {run}")

    report = compare(run_a, run_b, list(args.ignore_stream))
    text = render(report)
    print(text)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {target}")
    if args.markdown:
        target = Path(args.markdown)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        print(f"wrote {target}")


if __name__ == "__main__":
    main()
