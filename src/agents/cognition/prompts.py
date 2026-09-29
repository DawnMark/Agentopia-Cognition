"""Prompts for the cognition layer.

Kept out of the (already very large) `src/agents/prompts.py` so the phase 1
subsystem is reviewable on its own. The extraction prompt is the only LLM call
phase 1 adds, and it is deliberately narrow:

- it runs **once per character per simulated week**, after REVIEW;
- it may only name skills the character already has (design decision: methods
  hang off the existing `state.skills` vocabulary; anything else is flagged as
  unmapped rather than invented);
- it returns a bounded JSON list, so the post-processor can reject anything
  malformed and the caller can simply record nothing.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from src.agents.cognition.idea_models import CANDIDATE_CONFIDENCE_CEILING

# Hard limits so a chatty model cannot blow up the event stream.
MAX_METHODS_PER_WEEK = 3
MAX_LIST_ITEMS = 6
MAX_TEXT_LEN = 400

def language_name(code: str) -> str:
    """Human-readable language for the extraction prompts.

    The evidence is written in the world's language, and an English prompt
    otherwise produces English output for Chinese characters (observed in the
    phase 2 acceptance run), so the language is stated explicitly.
    """
    return "简体中文" if str(code).lower() in ("zh", "cn") else "English"


METHOD_EXTRACTION_PROMPT = """You are analysing one week of a person's life to extract **reusable methods** \
they could apply again.

A method is a *way of solving a class of problems*, not a diary entry:
- it must be repeatable (the person could do it again in a similar situation);
- it must name concrete steps, in order;
- it must say when it does **not** apply;
- it must be tied to one of the person's existing skills: {skill_list}
- write `title` and `description` in {language}, the language of the evidence

What counts as evidence this week (already recorded, objective):
{week_evidence}

Avoid these mistakes:
- Do not restate what happened ("I went to the gym"). Extract *how* it was done.
- Do not invent skills that are not in the list above.
- Do not propose more than {max_methods} methods; quality over quantity.
- Do not propose a method that is only a rephrasing of an existing one:
{existing_methods}

# Output

Return one JSON object and nothing else:

{{
  "methods": [
    {{
      "skill_id": "<one of the person's skills, exactly as written>",
      "title": "<short imperative title, e.g. 先定冲突再写场景>",
      "description": "<1-2 sentences: the situation it solves>",
      "applicable_contexts": ["<from: {allowed_tags}>"],
      "contraindications": ["<from the same list, when it backfires>"],
      "steps": ["<ordered, concrete>"],
      "checks": ["<how the person can tell it worked>"],
      "failure_modes": ["<how it typically goes wrong>"]
    }}
  ]
}}

If this week produced no genuinely reusable method, return {{"methods": []}}.
"""


def build_method_extraction_prompt(
    *,
    char: str,
    skill_list: List[str],
    week_evidence: str,
    existing_methods: List[str],
    allowed_tags: List[str],
    language: str = "en",
) -> str:
    """Render the extraction prompt for one character-week."""
    return METHOD_EXTRACTION_PROMPT.format(
        char=char,
        skill_list=", ".join(skill_list) if skill_list else "(none)",
        week_evidence=week_evidence.strip() or "(nothing recorded)",
        existing_methods=(
            "\n".join(f"  - {t}" for t in existing_methods[:10]) or "  (none yet)"
        ),
        max_methods=MAX_METHODS_PER_WEEK,
        allowed_tags=", ".join(allowed_tags),
        language=language_name(language),
    )


def _clean_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in out:
            out.append(text[:MAX_TEXT_LEN])
        if len(out) >= MAX_LIST_ITEMS:
            break
    return out


def parse_method_extraction(
    response: str, *, known_skills: Optional[List[str]] = None, **kwargs
) -> Optional[Dict[str, Any]]:
    """Post-processor for the extraction call: validated `{"methods": [...]}`.

    Returns None to trigger a retry; the caller treats an exhausted retry as
    "no methods this week", never as an error that could abort a run.
    """
    from src.utils import extract_json

    data = extract_json(response, **kwargs)
    if isinstance(data, list):
        data = {"methods": data}
    if not isinstance(data, dict):
        return None

    raw_methods = data.get("methods")
    if raw_methods is None:
        return None
    if not isinstance(raw_methods, list):
        return None

    known = {str(s).strip() for s in (known_skills or [])}
    cleaned: List[Dict[str, Any]] = []
    for item in raw_methods:
        if not isinstance(item, dict):
            continue
        skill_id = str(item.get("skill_id") or "").strip()
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        cleaned.append(
            {
                "skill_id": skill_id,
                "skill_mapped": (skill_id in known) if known else bool(skill_id),
                "title": title[:120],
                "description": str(item.get("description") or "").strip()[:MAX_TEXT_LEN],
                "applicable_contexts": _clean_list(item.get("applicable_contexts")),
                "contraindications": _clean_list(item.get("contraindications")),
                "steps": _clean_list(item.get("steps")),
                "checks": _clean_list(item.get("checks")),
                "failure_modes": _clean_list(item.get("failure_modes")),
            }
        )
        if len(cleaned) >= MAX_METHODS_PER_WEEK:
            break

    return {"methods": cleaned}


def format_week_evidence(records: List[Dict[str, Any]], *, max_records: int = 8) -> str:
    """Compact, objective rendering of one week's activity records.

    Only ledger facts are shown (type, what was done, the environment model's
    outcome, skill/vitality/money deltas). The character's reflection is
    deliberately excluded: reflection may inspire a hypothesis but must not be
    mistaken for evidence.
    """
    lines: List[str] = []
    for record in records[:max_records]:
        outcome = record.get("outcome") or {}
        if not isinstance(outcome, dict):
            outcome = {}
        deltas: List[str] = []
        if outcome.get("delta_skills"):
            deltas.append(
                "skills "
                + ", ".join(f"{k}{v:+}" for k, v in outcome["delta_skills"].items())
            )
        if outcome.get("delta_vitality"):
            deltas.append(f"vitality {int(outcome['delta_vitality']):+}")
        if outcome.get("delta_money"):
            deltas.append(f"money {int(outcome['delta_money']):+}")
        if record.get("verification_rejections"):
            deltas.append(f"rejections {int(record['verification_rejections'])}")

        lines.append(
            "- [{time}] {type}: {content}\n    outcome: {outcome}\n    deltas: {deltas}".format(
                time=record.get("time", "?"),
                type=record.get("type", "?"),
                content=str(record.get("content") or "").strip()[:200],
                outcome=str(outcome.get("outcome") or "").strip()[:200],
                deltas="; ".join(deltas) or "none",
            )
        )
    omitted = len(records) - len(lines)
    if omitted > 0:
        lines.append(f"({omitted} further activities omitted)")
    return "\n".join(lines)


def dumps_methods(methods: List[Dict[str, Any]]) -> str:
    """Small helper for logs/tests."""
    return json.dumps({"methods": methods}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Phase 2: weekly memory consolidation
# ---------------------------------------------------------------------------

MAX_MEMORIES_PER_WEEK = 5

MEMORY_EXTRACTION_PROMPT = """You are structuring one week of a person's life into **memories**.

A memory is one durable, specific thing this person now knows or remembers — not a
summary of the week. It must be usable months later without the original context.

Rules:
- Cite your evidence: each memory lists the indices of the evidence items it came
  from. Never write a memory that no evidence item supports.
- Items typed `reflection` / `weekly_review_reflection` are the person's **own
  words about their week**. They can support `belief` and `lesson` memories
  ("what I concluded"), and they can never support `semantic` or `episodic`
  facts on their own — pair them with an objective item when the memory claims
  something happened.
- Only mention people, places and things that appear in the evidence. Do not
  introduce new names.
- Write `content`, `obstacles` and `resources` in {language}, the language of
  the evidence. Do not translate them.
- `topics` must come from this list: {topic_list}
- `kind` is one of: episodic (a specific event), semantic (a general fact learned),
  relationship (about a person), goal (an intention), belief (what the person
  thinks is true), lesson (what to do differently next time).
- `outcome_polarity` is positive / negative / mixed / neutral for the *outcome*.
- `obstacles` are things that got in the way; `resources` are things that helped.
  Use short noun phrases, and reuse the same wording across weeks when it is the
  same obstacle or resource.
- At most {max_memories} memories. Skip anything trivial or already obvious.

# Evidence from this week

{evidence}

# Output

Return one JSON object and nothing else:

{{
  "memories": [
    {{
      "kind": "lesson",
      "content": "<one sentence, <= 200 characters>",
      "topics": ["<from the list>"],
      "entities": ["<person/place/thing from the evidence>"],
      "skills": ["<skill names used or improved, may be empty>"],
      "goal": "<the intention this relates to, may be empty>",
      "outcome_polarity": "positive|negative|mixed|neutral",
      "obstacles": ["<short phrase>"],
      "resources": ["<short phrase>"],
      "confidence": 0.0,
      "salience": 0.0,
      "emotion": 0.0,
      "sources": [0, 1]
    }}
  ]
}}

`confidence` is how sure you are the memory is accurate; `salience` is how much it
matters to this person; `emotion` is -1 (very bad) to 1 (very good).
If nothing this week is worth remembering, return {{"memories": []}}.
"""


def build_memory_extraction_prompt(
    *,
    char: str,
    evidence_items: List[Dict[str, Any]],
    topic_list: List[str],
    language: str = "en",
) -> str:
    """Render the weekly consolidation prompt.

    `evidence_items` are `{index, text}` entries; the model cites indices, which
    the consolidator maps back to ledger event ids (models are unreliable at
    copying long ids).
    """
    lines: List[str] = []
    for item in evidence_items:
        lines.append(f"[{item['index']}] {item['text']}")
    evidence_text = "\n\n".join(lines) if lines else "(nothing recorded)"
    return MEMORY_EXTRACTION_PROMPT.format(
        language=language_name(language),
        topic_list=", ".join(topic_list),
        max_memories=MAX_MEMORIES_PER_WEEK,
        evidence=evidence_text,
    )


def parse_memory_extraction(response: str, **kwargs) -> Optional[Dict[str, Any]]:
    """Validated `{"memories": [...]}` or None to trigger a retry."""
    from src.utils import extract_json

    data = extract_json(response, **kwargs)
    if isinstance(data, list):
        data = {"memories": data}
    if not isinstance(data, dict):
        return None
    raw = data.get("memories")
    if raw is None or not isinstance(raw, list):
        return None

    cleaned: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        sources: List[int] = []
        for value in item.get("sources") or []:
            try:
                idx = int(value)
            except (TypeError, ValueError):
                continue
            if idx >= 0 and idx not in sources:
                sources.append(idx)
        cleaned.append(
            {
                "kind": str(item.get("kind") or "episodic").strip(),
                "content": content[:200],
                "topics": _clean_list(item.get("topics")),
                "entities": _clean_list(item.get("entities")),
                "skills": _clean_list(item.get("skills")),
                "goal": str(item.get("goal") or "").strip(),
                "outcome_polarity": str(item.get("outcome_polarity") or "neutral").strip(),
                "obstacles": _clean_list(item.get("obstacles")),
                "resources": _clean_list(item.get("resources")),
                "confidence": item.get("confidence", 0.5),
                "salience": item.get("salience", 0.5),
                "emotion": item.get("emotion", 0.0),
                "sources": sources,
            }
        )
        if len(cleaned) >= MAX_MEMORIES_PER_WEEK:
            break
    return {"memories": cleaned}


# How much of one evidence item reaches the extraction prompt. Objective items
# are compressed hard; the character's own reflection gets more room because it
# is the raw material of beliefs and lessons (it used to be clipped to nothing:
# the diary was cut at 400 chars and then at 160, so the reflection half — which
# starts around char 330-390 — never reached the model at all).
EVIDENCE_CLIP_OBJECTIVE = 200
EVIDENCE_CLIP_REFLECTION = 420
# Total budget for one week's evidence block, so more items cannot silently
# inflate the prompt.
EVIDENCE_TOTAL_CHARS = 4200

_REFLECTION_TYPES = ("reflection", "weekly_review_reflection", "weekly_review")


def split_diary(content: str) -> tuple[str, str]:
    """Split a weekly diary entry into its Summary and Reflection halves."""
    text = str(content or "").strip()
    if not text:
        return "", ""
    marker = re.search(r"\bReflection\s*:", text)
    if not marker:
        return text, ""
    summary = text[: marker.start()].strip()
    reflection = text[marker.end() :].strip()
    summary = re.sub(r"^\s*Summary\s*:\s*", "", summary).strip()
    return summary, reflection


def format_memory_evidence(
    records: List[Dict[str, Any]],
    *,
    max_items: int = 14,
    total_chars: int = EVIDENCE_TOTAL_CHARS,
) -> List[Dict[str, Any]]:
    """Turn this week's ledger records into numbered evidence items.

    Only records that carry a ledger id can be cited, which keeps every memory
    traceable back to the append-only ledger. Objective items and the
    character's own reflections are rendered differently — `did / outcome /
    deltas` versus `reflection` — so the extractor can tell "what happened"
    from "what I concluded about it".
    """
    items: List[Dict[str, Any]] = []
    used = 0
    for record in records[:max_items]:
        event_id = str(record.get("ledger_event_id") or "")
        if not event_id:
            continue
        item_type = str(record.get("type") or "?")
        outcome = record.get("outcome") or {}
        if not isinstance(outcome, dict):
            outcome = {}
        content = str(record.get("content") or "").strip()
        if item_type in _REFLECTION_TYPES:
            clip = EVIDENCE_CLIP_REFLECTION
            label = (
                f"{item_type}({record['activity_type']})"
                if record.get("activity_type")
                else item_type
            )
            text = "{time} | {type} | reflection: {body}".format(
                time=record.get("time", "?"),
                type=label,
                body=content[:clip],
            )
        else:
            clip = EVIDENCE_CLIP_OBJECTIVE
            deltas: List[str] = []
            if outcome.get("delta_skills"):
                deltas.append(
                    "skills " + ", ".join(f"{k}{v:+}" for k, v in outcome["delta_skills"].items())
                )
            if outcome.get("delta_vitality"):
                deltas.append(f"vitality {int(outcome['delta_vitality']):+}")
            if outcome.get("delta_money"):
                deltas.append(f"money {int(outcome['delta_money']):+}")
            text = (
                "{time} | {type} | did: {content} | outcome: {outcome} | {deltas}".format(
                    time=record.get("time", "?"),
                    type=item_type,
                    content=content[:clip] or "(no description in the record)",
                    outcome=str(outcome.get("outcome") or "").strip()[:clip],
                    deltas="; ".join(deltas) or "no measured delta",
                )
            )
        if used + len(text) > total_chars and items:
            break
        used += len(text)
        items.append({"index": len(items), "text": text, "event_id": event_id})
    return items


# ---------------------------------------------------------------------------
# Phase 3: idea phrasing
# ---------------------------------------------------------------------------

IDEA_PROMPT = """You are helping one person turn a pattern in their own experience into a **testable idea**.

You are given {n_candidates} candidate motifs found in this person's memories. For each, produce one idea. An idea is a *hypothesis or an opportunity*, never a fact and never a decision.

Hard rules:
- The idea must be grounded in the given memories only. Do not introduce people,   places or things that are not in the evidence.
- Keep it small enough to try within a week.
- `test_plan` must describe one concrete action plus what the person would   observe to know whether it worked. If you cannot describe such a test, return   an empty test_plan for that motif (it will be rejected).
- `requires` lists only what the attempt needs: skills, entities (people,   places, things) and money. Leave a list empty when nothing is needed. The   system checks these against the world, so do not pad them.
- Write in {language}, the language of the evidence.
- `confidence` is your own confidence that this is worth trying (0-1). It will   be capped at {confidence_ceiling} because nothing has been tested yet.
- When `motif=lesson_application` the memory quoted in `hint` is a lesson the   person wrote for themselves in their own notes ("next time, do X"). Turn it   into one concrete action for this week and say what to watch for. Do not   restate the lesson; the idea is the *test* of it.

# What this person currently has

Skills: {skills}
Can reach: {entities}
Money: {money}

# Candidate motifs

{candidates}

# Output

Return one JSON object and nothing else:

{{
  "ideas": [
    {{
      "candidate_index": 0,
      "idea_type": "hypothesis|opportunity|method_candidate|goal_adjustment",
      "content": "<one sentence: what to try, and why it might work>",
      "test_plan": "<what to do this week and what to observe>",
      "skills": ["<used or needed skills>"],
      "requires": {{"skills": [], "entities": [], "money": 0}},
      "confidence": 0.0
    }}
  ]
}}
"""


METHODIZATION_PROMPT = """You are turning one person's **testable idea** into a usable **method** they could actually follow.

Idea (a hypothesis, not a fact):
  {idea_content}

How they planned to test it:
  {test_plan}

Motif it came from: {motif}
Shared focus: {shared_focus}
Goal: {goal}
Obstacles: {obstacles}
Resources: {resources}
Skill it belongs to: {skill}
Evidence memories:
{memories}

Write the method, not the hypothesis. Rules:
- `title`: a short method name (at most 12 Chinese characters / 8 words). Not a sentence, no "maybe", no "I will".
- `description`: one or two sentences saying what the method is and when it applies.
- `steps`: 3 to 6 concrete actions, in the order the person would do them.
- `checks`: 1 to 3 observable signs that tell the person whether it worked.
- `failure_modes`: 1 to 3 concrete ways this method fails or backfires.
- `applicable_contexts`: 1 to 4 tags from exactly this list: {context_tags}
- `contraindications`: 0 to 3 tags from the same list, situations where it should NOT be used.
- Everything must stay within the evidence above: do not invent people, places, tools or skills.
- Write in {language}.

Return JSON only:
{{"title": "...", "description": "...", "steps": ["..."], "checks": ["..."],
  "failure_modes": ["..."], "applicable_contexts": ["..."], "contraindications": ["..."]}}"""


def build_methodization_prompt(
    *,
    idea_content: str,
    test_plan: str,
    motif: str,
    skill: str,
    shared_focus: str = "",
    goal: str = "",
    obstacles: Optional[List[str]] = None,
    resources: Optional[List[str]] = None,
    memories: Optional[List[str]] = None,
    language: str = "en",
) -> str:
    """Render the idea → method ("methodisation") prompt (KI-6)."""
    from src.agents.cognition.models import CONTEXT_TAGS

    memory_lines = chr(10).join(f"  - {m}" for m in (memories or [])) or "  - (none)"
    return METHODIZATION_PROMPT.format(
        idea_content=str(idea_content or "").strip()[:400],
        test_plan=str(test_plan or "").strip()[:300] or "(none)",
        motif=motif or "unknown",
        shared_focus=shared_focus or "(none)",
        goal=goal or "(none)",
        obstacles=", ".join(obstacles or []) or "(none)",
        resources=", ".join(resources or []) or "(none)",
        skill=skill or "(none)",
        memories=memory_lines,
        context_tags=", ".join(CONTEXT_TAGS),
        language=language_name(language),
    )


def parse_methodization_response(response: str, **kwargs) -> Optional[Dict[str, Any]]:
    """Validated method fields; None triggers a retry.

    Returns whatever the model produced (possibly still incomplete) so the
    caller can decide, mark and log it — a structurally incomplete method is
    treated as a program error rather than silently dropped (KI-6).
    """
    from src.agents.cognition.models import CONTEXT_TAGS
    from src.utils import extract_json

    data = extract_json(response, **kwargs)
    if not isinstance(data, dict):
        return None
    if isinstance(data.get("method"), dict):
        data = data["method"]

    allowed = set(CONTEXT_TAGS)
    contexts = [t for t in _clean_list(data.get("applicable_contexts")) if t in allowed]
    contra = [t for t in _clean_list(data.get("contraindications")) if t in allowed]
    return {
        "title": str(data.get("title") or "").strip()[:80],
        "description": str(data.get("description") or "").strip()[:400],
        "steps": _clean_list(data.get("steps")),
        "checks": _clean_list(data.get("checks")),
        "failure_modes": _clean_list(data.get("failure_modes")),
        "applicable_contexts": contexts,
        "contraindications": contra,
    }


def build_idea_prompt(
    *,
    char: str,
    candidates: List[Dict[str, Any]],
    skills: List[str],
    entities: List[str],
    money: float,
    language: str = "en",
) -> str:
    """Render the idea-phrasing prompt for the week's motif candidates."""
    lines: List[str] = []
    for idx, candidate in enumerate(candidates):
        memory_lines = "\n".join(
            f"      - {m}" for m in (candidate.get("memory_contents") or [])
        ) or "      - (no excerpt available)"
        block = "\n".join(
            [
                f"[{idx}] motif={candidate.get('motif', '?')} "
                f"(relation score {candidate.get('score', 0)})",
                f"    hint: {candidate.get('hint', '')}",
                f"    shared focus: {candidate.get('shared_focus') or '(none)'}",
                f"    goal: {candidate.get('goal') or '(none)'}",
                f"    obstacles: {', '.join(candidate.get('obstacles') or []) or '(none)'}",
                f"    resources: {', '.join(candidate.get('resources') or []) or '(none)'}",
                "    memories:",
                memory_lines,
            ]
        )
        lines.append(block)
    candidates_block = "\n\n".join(lines) if lines else "(no candidates)"
    return IDEA_PROMPT.format(
        n_candidates=len(candidates),
        language=language_name(language),
        confidence_ceiling=CANDIDATE_CONFIDENCE_CEILING,
        skills=", ".join(skills) or "(none)",
        entities=", ".join(entities[:25]) or "(none)",
        money=round(float(money), 2),
        candidates=candidates_block,
    )


def parse_idea_response(response: str, **kwargs) -> Optional[Dict[str, Any]]:
    """Validated `{"ideas": [...]}`; None triggers a retry."""
    from src.utils import extract_json

    data = extract_json(response, **kwargs)
    if isinstance(data, list):
        data = {"ideas": data}
    if not isinstance(data, dict):
        return None
    raw = data.get("ideas")
    if raw is None or not isinstance(raw, list):
        return None

    cleaned: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        try:
            index = int(item.get("candidate_index", len(cleaned)))
        except (TypeError, ValueError):
            index = len(cleaned)
        requires = item.get("requires") or {}
        if not isinstance(requires, dict):
            requires = {}
        cleaned.append(
            {
                "candidate_index": index,
                "idea_type": str(item.get("idea_type") or "hypothesis").strip(),
                "content": content[:300],
                "test_plan": str(item.get("test_plan") or "").strip()[:300],
                "skills": _clean_list(item.get("skills")),
                "requires": {
                    "skills": _clean_list(requires.get("skills")),
                    "entities": _clean_list(requires.get("entities")),
                    "money": requires.get("money", 0),
                },
                "confidence": item.get("confidence", 0.3),
            }
        )
    return {"ideas": cleaned}
