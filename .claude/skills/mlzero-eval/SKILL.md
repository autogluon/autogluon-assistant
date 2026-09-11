---
name: mlzero-eval
description: Use after a mlzero run to pick the best candidate node on the validation (select) set, then score that single winner on the untouched hold-out with a deterministic, task-appropriate metric.
---

# mlzero-eval

**Purpose.** Produce a TRUSTWORTHY generalization metric in two honest stages, deterministically and
without any LLM:
1. **Select** — among the candidate nodes (mlzero's chosen "best" is often wrong / leaky), choose the
   winner by its score on the `select` (validation) set.
2. **Test** — score that ONE winner on the `holdout` set exactly once → the reported true metric.
The hold-out is NEVER used for selection (that would bias the metric via selection-on-test).

**Inputs.** `<ws>/spec.json`, `<ws>/mlzero_result.json` (with `candidates`), `<ws>/data/{train,select,holdout}/`.

**Steps.**
1. Run the evaluator:
   ```
   python .claude/skills/mlzero-eval/evaluate.py --workspace <ws>
   ```
   For each candidate it reloads the node's saved model (or re-runs its code on the eval data),
   scores on `select`, picks the winner, then scores the winner on `holdout` once.
2. Read `<ws>/eval/holdout_metrics.json`: the TRUE metric, `winner_node_id`, `selected_by`, and the
   `selection_table` (per-candidate select scores). Note if the winner differs from mlzero's `best_run`.

**Outputs.** `<ws>/eval/holdout_metrics.json` (true score, metric, method [reload|rerun|failed],
selected_by, winner_node_id, selection_table, leakage_flag, sanity) and `<ws>/eval/predictions_holdout.csv`.

**Notes/guardrails.**
- The metric is TASK-APPROPRIATE, keyed by `spec.eval_metric` (accuracy/F1/AUC for classification;
  RMSE/MAE/MAPE for regression; MASE/etc. for forecasts; extend the registry for others). Never hardcode one.
- **Never rank candidates on the test set** — only on `select`. If there is no `select` set or only one
  candidate, the winner falls back to mlzero's `best_run` (recorded as `selected_by=mlzero_best_run`).
- **Sanity/leakage:** flag near-perfect scores, constant predictions, wrong row counts, target-in-features.
- `method="failed"` (exit 2) means the winner produced no predictions via reload or rerun — surface it,
  never treat as a pass.
