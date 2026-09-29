---
name: run-metrics
description: Compute numerical metrics for a simulation run (as opposed to the qualitative analysis done by analyze-run). Use when the user says "run metrics", "计算指标", or "compute metrics".
---

# run-metrics

Compute numerical metrics for a simulation run (as opposed to the qualitative analysis done by analyze-run).

## Trigger

Use when the user asks to compute run metrics. See the trigger phrases in the frontmatter `description`.

## Workflow

### 1. Determine data_dir
- Provided by the user → use it directly.
- Not provided → run `ls -t data/ | head -5` to list the most recent directories and let the user choose.

### 2. Run compute_metrics
```bash
python scripts/compute_metrics.py --data-dir school_03121031
```

The world name is inferred from the run config; no extra argument is needed.

### 3. Produce a summary
Read `analysis/results/{data_dir}_metrics.json` and output:

- **Innate stats**: distribution of income/deposit/relationship (min, max, mean, median).
- **Per-year key metrics**: average deposit change per year, fulfillment trend, social eval trend.
- **Outliers**: agents whose metrics are extremely high or low.
- **Correlation insights**: relationships between innate conditions and growth metrics (e.g. income vs deposit_diff).

Format the output as concise tables/lists so the user can grasp the overall picture quickly.
