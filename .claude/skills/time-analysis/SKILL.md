---
name: time-analysis
description: Measure wall-clock time consumed per week of a simulation run. Use when the user says "时间分析", "time analysis", "每周耗时", or "运行时间".
allowed-tools: Bash(python scripts/time_analysis.py *)
---

# Time Analysis Skill

## Purpose

Analyze the per-week wall-clock time of a simulation run, reporting each week's start/end time (GMT+8) and duration.

## Usage

Run the matching command based on the user's request:

### Analyze a specific run

```bash
python scripts/time_analysis.py <run_id>
```

### Analyze multiple runs (comparison)

```bash
python scripts/time_analysis.py <run_id1> <run_id2> <run_id3>
```

### Analyze the most recent N runs

```bash
python scripts/time_analysis.py --latest 3
```

### Analyze all runs

```bash
python scripts/time_analysis.py --all
```

### Print only, without saving a file

```bash
python scripts/time_analysis.py --no-save <run_id>
```

## Output

- **stdout**: a compact table per run (Week / Start / End / Duration).
- **File**: `data/{run_id}/time_analysis.md`, containing the full markdown report.

## Notes

- Times are converted from the server's PDT (UTC-7) to GMT+8.
- If early weeks like W1/W2 take very little time (<10min), it indicates a cache hit (resumed from a previous run).
- If the same week has multiple PLAN STAGE entries (resume), use the last one as the start time.
