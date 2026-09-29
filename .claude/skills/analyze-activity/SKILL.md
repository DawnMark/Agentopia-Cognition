---
name: analyze-activity
description: Analyze whether a role-play agent's activity output reads like a real human. Use when the user says "分析对话", "analyze activity", or "检查角色扮演".
allowed-tools: Bash(python:*), Read, Grep, Glob
---

# Analyze Activity Skill

## Purpose

Extract a role-play agent's utterances (output) from the activity phase in the system run logs, and analyze whether they read like things a real human would say or do.

## Core question: does this read like a real human?

Take the perspective of an ordinary person and judge whether each piece of output reads like something a real human would say or do in that situation.

**Ask yourself:**
- If I met this person in real life and heard them talk this way, would it feel normal?
- If a coworker or friend expressed themselves like this, would it feel strange?

**Common "not human-like" signs (examples only, not exhaustive):**
- Language too formal or literary: real people don't talk this way day to day
- Action descriptions too dramatic: real people don't move this way
- Unnatural emotional expression: too exaggerated or too suppressed
- Off conversational rhythm: too fast or too slow, transitions too abrupt
- Information density too high: cramming too much into one sentence
- Repetition or wordiness: saying the same thing over and over
- Inconsistent character: behavior or personality contradicts itself
- Hallucination: mentioning things that don't exist
- Format issues: not conforming to the output spec
- ...(anything that makes you think "a real person wouldn't do this")

## Notes

- Character dialogue uses format markers, for example:
    (action description) "line" (continued action)
    <private>inner monologue</private>
    These format markers are allowed; they are used only as tags.

## Steps

### Step 1: User provides run_id

Wait for the user to specify the run_id to analyze, in a format like `data/school_01082234`.

### Step 2: Read the Principles

Read `.claude/skills/analyze-activity/PRINCIPLES.md` to learn the existing judgment criteria.

### Step 3: Extract dialogue records

```bash
python scripts/extract_activity_dialogues.py <run_id>
```

This generates `logs/activity/<run_name>.log` (e.g., `logs/activity/school_01082234.log`).

### Step 4: Read the dialogue records

Use the Read tool to read the generated dialogue record file.

### Step 5: Analyze item by item

Analyze each piece of output:
1. Check it against the principles in PRINCIPLES.md
2. Start with an overall impression: does this read like a real human?
3. Pinpoint the specific problems: where does it fall short, and why?
4. Suggest a direction for improvement

### Step 6: Update the Claude Principles

If you find a new, representative problem pattern (not already in the existing principles), use the Edit tool to add it to the "Claude Principles" section of PRINCIPLES.md.

Format:
```
### C{N}. {principle description}

- Example: {original bad case}
- Revision: {revised version}
```

Notes:
- Only add representative, recurring problems
- Do not modify the "Human Annotated Principles" section
- If a principle is best illustrated with an example (e.g., "avoid piling on metaphorical imagery" -> "example: dandelion roots dig into the cracks"), give an example; if it's better explained in words, skip the example

### Step 7: Output the analysis report

```
## Activity Analysis Report

### Run ID: <run_id>

---

### Problem 1
**Location**: [turn X, character name] in Activity "activity name"
**Original**: "..."
**Problem**: [specifically describe what doesn't read like a real human]
**Why it's off**: [explanation]
**A real person might**: [give a more natural way to express it]

---

### Problem 2
...

---

## Summary

### Overall impression
[Does this batch of output read like a real human overall? What are the main problems?]

### Problem breakdown
[Count by problem type, e.g., "language too literary: X instances"]

### Prompt improvement suggestions
[Based on the problems found, suggest how to adjust the prompt]
```

## Guiding principles

1. **Real humans are the standard** - the question isn't "is it good?" but "is it human-like?"
2. **Consider the character's background** - a 17-year-old high schooler and a 30-year-old office worker talk differently
3. **Consider the situation** - chatting among friends differs from a formal setting
4. **Pay attention to detail** - a single word or action can reveal the "AI flavor"
5. **Trust your overall sense** - sometimes the whole thing just feels off, even when you can't name the exact spot
