#!/usr/bin/env python3
"""Read one run and tell the lesson-bridge story in the character's own words.

    python scripts/lesson_bridge_story.py --data-dir shanghai_apartment_09262206

Prints, per persona: the lessons it wrote, the ideas those lessons produced, the
methods those ideas became, what the menu offered, what the character said in its
plan, what it actually did, and what came back from the world. Everything is read
from the run directory — no LLM, no recomputation of business rules.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]


sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def ideas_of(persona_dir: Path) -> Dict[str, Dict[str, Any]]:
    path = persona_dir / "cognition" / "views" / "ideas.json"
    if not path.exists():
        return {}
    return (json.loads(path.read_text(encoding="utf-8")).get("ideas") or {})


def methods_of(persona_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Methods rebuilt from the capability stream (the view file is written at
    settlement, so it may lag behind a finished-but-unsaved run)."""
    events = read_jsonl(persona_dir / "cognition" / "capability_events.jsonl")
    if not events:
        return {}
    path = persona_dir / "cognition" / "views" / "capabilities.json"
    if path.exists():
        view = json.loads(path.read_text(encoding="utf-8"))
        if (view.get("methodologies") or {}):
            return view["methodologies"]
    from src.agents.cognition.materializer import materialize

    return materialize(events, skills=None, persona=persona_dir.name).get("methodologies") or {}


def memories_of(persona_dir: Path) -> Dict[str, Dict[str, Any]]:
    path = persona_dir / "cognition" / "views" / "memories.json"
    if not path.exists():
        return {}
    return (json.loads(path.read_text(encoding="utf-8")).get("memories") or {})


def plan_text_around(persona_dir: Path, needle: str) -> List[str]:
    """The character's own sentences around a mention (plan / reflection)."""
    out: List[str] = []
    for path in sorted((persona_dir / "generation").glob("year=*/week=*.jsonl")):
        for row in read_jsonl(path):
            outputs = row.get("outputs") or []
            text = str(outputs[-1].get("content") or "") if outputs else ""
            if needle and needle in text:
                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                hit = next((i for i, ln in enumerate(lines) if needle in ln), None)
                if hit is not None:
                    out.append(
                        f"[{row.get('time')}] " + " / ".join(lines[max(0, hit - 1) : hit + 2])
                    )
    return out


def story_for(persona_dir: Path) -> List[str]:
    name = persona_dir.name
    caps = read_jsonl(persona_dir / "cognition" / "capability_events.jsonl")
    memories = memories_of(persona_dir)
    ideas = ideas_of(persona_dir)
    methods = methods_of(persona_dir)
    activities = {str(r.get("activity_id")): r for r in read_jsonl(persona_dir / "activity.jsonl")}

    lines: List[str] = [f"######## {name}"]

    lessons = [m for m in memories.values() if str(m.get("origin")) == "scratchpad_lesson"]
    lines.append(f"-- 角色自己写下的教训（入库 {len(lessons)} 条）")
    for m in lessons:
        lines.append(
            f"   [{m.get('created_at')}] {m.get('content')}   "
            f"(confidence {m.get('confidence')}, {m.get('provenance')})"
        )

    lesson_ideas = [i for i in ideas.values() if str(i.get("motif")) == "lesson_application"]
    lines.append(f"-- 这些教训生成的 Idea（{len(lesson_ideas)} 条）")
    for idea in lesson_ideas:
        lines.append(
            f"   [{idea.get('created_at')}] status={idea.get('status')} potential={idea.get('potential')}\n"
            f"       想试：{idea.get('content')}\n"
            f"       怎么试：{idea.get('test_plan')}\n"
            f"       来源记忆：{idea.get('source_memory_ids')} → 方法 {idea.get('converted_method_id')}"
        )

    lesson_methods = {
        mid: m for mid, m in methods.items() if str(m.get("source_motif")) == "lesson_application"
    }
    lines.append(f"-- 它们变成的方法（{len(lesson_methods)} 个）")
    for mid, m in lesson_methods.items():
        lines.append(
            f"   [{mid}] {m.get('title')} status={m.get('status')} "
            f"value={m.get('global_value')} practice={m.get('practice_count')} "
            f"success={m.get('success_count')}\n"
            f"       步骤：{' → '.join(m.get('steps') or [])}"
        )

    lines.append("-- 提示与采用")
    offers = [r for r in caps if r.get("type") == "METHOD_HINTED"]
    picks = [r for r in caps if r.get("type") == "METHOD_SELECTED" and r.get("source") == "hint"]
    applied = [r for r in caps if r.get("type") == "METHOD_APPLIED"]
    outcomes = [
        r for r in caps if r.get("type") == "METHOD_OUTCOME_OBSERVED" and r.get("attribution") == "real_adoption"
    ]
    lesson_ids = set(lesson_methods)
    lines.append(
        f"   本周提示总数 {len(offers)}，其中来自教训的方法 "
        f"{sum(1 for r in offers if r.get('origin') == 'own_lesson')} 次"
    )
    for r in offers:
        if r.get("origin") == "own_lesson":
            m = lesson_methods.get(str(r.get("method_id")), {})
            title = m.get("title") or str(r.get("title") or r.get("method_id"))
            lines.append(f"   [{r.get('week')}] 菜单提供：{title}（{r.get('role')}）")
    for r in picks:
        if str(r.get("method_id")) in lesson_ids:
            m = lesson_methods[str(r.get("method_id"))]
            lines.append(f"   [{r.get('week')}] 角色采用：{m.get('title')}（{r.get('matched_by')}）")
            for snippet in plan_text_around(persona_dir, str(m.get("title") or ""))[:2]:
                lines.append(f"        计划原文：{snippet[:220]}")
    for r in applied:
        if str(r.get("method_id")) in lesson_ids:
            activity = activities.get(str(r.get("activity_id")), {})
            lines.append(
                f"   [{r.get('week')}] 真的做了：{activity.get('type')} "
                f"{str(activity.get('content') or activity.get('reflection') or '')[:120]}"
            )
            lines.append(f"        世界反馈：{str((activity.get('outcome') or {}).get('outcome') or '')[:160]}")
    for r in outcomes:
        if str(r.get("method_id")) in lesson_ids:
            lines.append(f"   [{r.get('week')}] 结果：reward {r.get('reward')}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    run_dir = Path(args.data_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / "data" / run_dir
    lines: List[str] = [f"# lesson bridge story · {run_dir.name}"]
    for persona_dir in sorted((run_dir / "persona").iterdir()):
        if not (persona_dir / "profile").exists() or not (persona_dir / "activity.jsonl").exists():
            continue
        lines += story_for(persona_dir)
    text = "\n".join(lines) + "\n"
    if args.output:
        out = Path(args.output)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(out)
    else:
        print(text)


if __name__ == "__main__":
    main()
