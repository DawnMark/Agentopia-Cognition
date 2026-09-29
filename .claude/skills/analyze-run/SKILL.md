---
name: analyze-run
description: Deeply analyze a simulation run's results — agent experiences, inner journey, personality growth, etc. Use when the user says "分析run", "analyze run", or "分析这个运行".
allowed-tools: Read, Grep, Glob, Write, Bash(mkdir:*, ls:*)
---

# Analyze Run Skill

## Purpose

Perform a deep analysis of a complete simulation run, producing both system-level and agent-level analysis reports.

## Input

The user provides a `RUN_ID`, for example: `school_03020040`

## Output

All artifacts go under the `data/{RUN_ID}/run_analysis/` folder:

```
data/{RUN_ID}/run_analysis/
├── system_analysis.md       # System-level analysis
├── agent_A.md               # Detailed analysis of Agent A
├── agent_B.md               # Detailed analysis of Agent B
├── agent_C.md               # ...(one file per agent)
└── ...
```

---

## Execution Steps

### Step 1: Get the RUN_ID

Wait for the user to provide a RUN_ID. If the user does not specify one, use the following command to find the latest run:

```bash
ls -t data/ | grep "school_" | head -1
```

### Step 2: Verify the data directory exists

```bash
ls data/{RUN_ID}/persona/
```

Confirm that the persona directory exists and contains agent subdirectories.

### Step 3: Create the output directory

```bash
mkdir -p data/{RUN_ID}/run_analysis
```

### Step 4: Collect data

For each agent, read the following key data:

1. **Profile** (character background)
   - `data/{RUN_ID}/persona/{name}/profile/year=2020.json`

2. **Weekly Diary** (core analysis material)
   - `data/{RUN_ID}/persona/{name}/memory/weekly_diary.jsonl`

3. **General Scratchpad** (long-term plans)
   - `data/{RUN_ID}/persona/{name}/memory/scratchpad/general.jsonl`

4. **Character Notes** (impressions of others)
   - `data/{RUN_ID}/persona/{name}/memory/scratchpad/characters/*.jsonl`

5. **State History** (state changes)
   - `data/{RUN_ID}/persona/{name}/state.jsonl`

6. **Schedule** (activity scheduling)
   - `data/{RUN_ID}/persona/{name}/schedule.jsonl`

7. **Contact Records** (contact logs)
   - `data/{RUN_ID}/persona/{name}/contact/*.jsonl`

### Step 5: Analysis dimensions

#### 5.1 System-level analysis dimensions

1. **Run Overview**
   - Number of weeks run, number of agents, total number of activities
   - Key statistics

2. **Relationship Network Evolution**
   - Which relationships formed / deepened / drifted apart
   - Who are the social hubs
   - Interesting relationship dynamics

3. **Group Behavior Patterns**
   - Jointly attended activities
   - Small groups that formed
   - Conflicts and reconciliations

4. **Notable Highlights**
   - The most lifelike behaviors
   - The most dramatic turning points
   - The performances most true to character

5. **Overall Authenticity Assessment**
   - An authenticity score and brief note for each agent

#### 5.2 Agent-level analysis dimensions (about 1000 words per agent)

1. **Character Introduction**
   - Background introduction based on the profile
   - Core motivation and inner conflict

2. **Experience Recap**
   - Walk through the main activities and events week by week
   - Key interactions and decisions

3. **Inner Journey**
   - Emotional change curve
   - Key lines from inner monologues
   - Shifts in thinking and beliefs

4. **Personality Analysis**
   - Personality traits displayed
   - Which behaviors embody the character profile
   - Degree of alignment with the preset personality

5. **Growth and Change**
   - Improvements in abilities/skills
   - Changes in mindset/outlook
   - Development of relationships

6. **Interpreting Key Behaviors**
   - The 3-5 behaviors that best embody the personality
   - Why these behaviors are characteristic

7. **Authenticity Assessment**
   - Does this agent feel like a real person?
   - Which performances are the most lifelike?
   - Which performances feel slightly stereotyped?

---

## Output Format

### system_analysis.md template

```markdown
# {RUN_ID} System Analysis Report

## 1. Run Overview

- **Run Period**: Y{year}-W{start} ~ Y{year}-W{end}
- **Number of Agents**: {N}
- **Total Activities**: {M}

## 2. Agent Ensemble

| Agent | Personality Type | Core Conflict | Authenticity Score |
|-------|------------------|---------------|--------------------|
| Agent A | INTJ top student | academic pressure vs. desire for social connection | ⭐⭐⭐⭐⭐ |
| ... | ... | ... | ... |

## 3. Relationship Network Evolution

### 3.1 Relationship Change Overview
[Describe the formation, deepening, and drifting apart of major relationships]

### 3.2 Key Relationship Threads
[e.g., Agent A & Agent D study partners, Agent C & Agent D artistic exchange, Agent E & Agent F brotherhood]

## 4. Group Behavior Patterns

[Analyze joint activities, formation of small groups, etc.]

## 5. Notable Highlights

### 5.1 Most Lifelike Behaviors
[List 3-5 behaviors that feel the most human]

### 5.2 Interesting Turning Points
[List 2-3 unexpected or dramatic developments]

## 6. Overall Authenticity Assessment

[Overall assessment of whether this batch of agents behaves like real people]

## 7. Issues and Suggestions for Improvement

[If there are clearly unnatural behaviors, list suggestions for improvement]
```

### agent_{name}.md template

```markdown
# {name} - Character Analysis Report

## 1. Character Introduction

**Basic Information**
- Personality type: {MBTI}
- Core motivation: {motivation}
- Inner conflict: {conflict}

**Background Setting**
[Description based on the profile, about 100 words]

## 2. {N}-Week Experience Recap

### W1: [keyword]
[Main activities and events of the week, about 50-80 words]

### W2: [keyword]
...

## 3. Inner Journey

### 3.1 Emotional Change Curve

[Describe the change in emotional state, from low point to high point]

### 3.2 Key Inner Monologues

> "{key line excerpted from the weekly diary 1}"

> "{key line excerpted from the weekly diary 2}"

### 3.3 Shifts in Outlook

[Describe how their thinking changed, about 100 words]

## 4. Personality Analysis

### 4.1 Personality Traits Displayed

- **{Trait 1}**: [specific manifestation]
- **{Trait 2}**: [specific manifestation]
- ...

### 4.2 Alignment with the Character Profile

[Analyze how well the behavior aligns with the preset personality, about 80 words]

## 5. Growth and Change

### 5.1 Ability Improvements
[Changes in skills, knowledge, and abilities]

### 5.2 Mindset Changes
[Changes in mindset, attitude, and outlook]

### 5.3 Relationship Development
[Changes in relationships with others]

## 6. Interpreting Key Behaviors

### Behavior 1: {behavior description}
**Context**: [in what situation it occurred]
**Significance**: [why this behavior strongly embodies their personality]

### Behavior 2: ...
...

## 7. Authenticity Assessment

### 7.1 Overall Score: ⭐⭐⭐⭐⭐

### 7.2 Most Lifelike Performances
[List 2-3 behaviors that feel the most human]

### 7.3 Areas for Improvement
[If there are any unnatural performances, point them out]

---

**Analysis Summary**

[Summarize this agent's performance and growth in this run in a single sentence]
```

---

## Analysis Principles

1. **People-first** - Always ask yourself: would a real person do this? Think this way?

2. **Watch the change** - Don't just look at single-point states; focus on trends of change and trajectories of growth.

3. **Quote the source** - Use the agent's own words from the weekly diary as the basis for analysis.

4. **Objective evaluation** - Honestly point out both strengths and shortcomings; don't praise indiscriminately.

5. **Make it accessible** - Use plain language to explain complex psychological changes.

6. **Mind the details** - Small behaviors are often the most revealing of personality.

---

## Notes

- **Data first**: Base the analysis on actual data; do not make things up.
- **Cite sources**: When quoting weekly diary content, note the source (e.g., W3 weekly diary).
- **Length control**: System-level analysis about 2000-3000 words, agent-level analysis about 1000 words.
- **Readability**: Use clear structure, appropriate subheadings, and bullet points.
