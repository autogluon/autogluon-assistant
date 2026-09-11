---
name: mlzero-solve
description: Use when the user asks to build, train, fit, predict, classify, or forecast an ML model/solution from a data folder or dataset (any modality — tabular, text, image, audio, document, time series) with a natural-language goal. The end-to-end orchestrator.
---

# mlzero-solve

**Purpose.** Turn a data folder + a natural-language goal into a verifiable ML solution:
profile the data, clarify the spec with the user, prepare a leakage-safe hold-out split,
run the `mlzero` CLI, evaluate on the untouched hold-out with a deterministic metric, and
write a clean report. This is the ONLY skill users trigger directly; it delegates each phase
to a sub-skill and owns the per-run workspace.

**Inputs.** A path to the user's data (folder or CSV) and their goal in natural language.

**Steps.**
1. **Create workspace.** `ws=runs/<slug-of-task>-<UTC-ts>/`. Make `ws/`, `ws/data/`,
   `ws/env/`, `ws/eval/`, `ws/solution/`. Use absolute paths everywhere.
2. **Profile** — run the `mlzero-profile` skill → `ws/profile.json` (task_type, modality,
   target_candidates, split_strategy_recommendation, per-column stats + leakage flags).
3. **Clarify with the user (required).** Present the inferred spec in plain language and ask
   ONLY genuinely ambiguous items, pre-filling defaults from the profile:
   - target column (skip if non-tabular / mlzero infers it),
   - evaluation metric (propose a task-appropriate default; do NOT hardcode),
   - split strategy (confirm the recommendation),
   - **forecast only:** horizon, and which covariates are known-in-advance.
   Do not ask questions the profile already answers unambiguously.
4. **Write `ws/spec.json`** from profile + user answers (general schema; time-series fields
   only when `task_type=forecast`). See `mlzero-spec` for the schema. Also make ONE **compute-budget**
   assessment here from the data size + available compute, and thread it through consistently: it sets
   the training intensity/preset (`mlzero-spec` → `description.md`) AND the search-iteration budget `N`
   (`mlzero-run`). Larger/slower data ⇒ lighter preset + fewer iterations; small/fast data ⇒ heavier
   preset + more iterations. (Never encode it as a hard training time limit.)
5. **Prepare** — run `mlzero-prepare` (split + drop unknown-future covariates) → `ws/data/*`.
6. **Spec** — run `mlzero-spec` → `ws/data/train/description.md`.
7. **Env** — run `mlzero-env` → `ws/env/env_path.txt`.
8. **Run** — run `mlzero-run` (invoke mlzero + normalize) → `ws/mlzero_run/`, `ws/mlzero_result.json`.
9. **Eval** — run `mlzero-eval` → `ws/eval/holdout_metrics.json`, `ws/eval/predictions_holdout.csv`.
10. **Report** — run `mlzero-report` → `ws/solution/{report.md,run_summary.json,reproduce.sh}`.

**Auto-retry (once).** After step 9, inspect `holdout_metrics.json`. If it carries a
`leakage_flag`, an error, or an implausible score (e.g. near-perfect / far better than the
mlzero-reported validation), **report the concern honestly** and retry mlzero EXACTLY ONCE:
append a sharper anti-leakage note to `description.md` (name the suspected leaking columns /
the hidden hold-out), re-run `mlzero-run` then `mlzero-eval`, then stop regardless of outcome.
Track a `retried` flag so you never loop.

**Outputs.** The full `ws/` tree; the deliverable is `ws/solution/report.md`.

**Notes/guardrails.**
- Never let mlzero see `ws/data/holdout/` — it only receives `ws/data/train/`.
- Keep the workspace as the single source of truth; pass `ws` to every sub-skill.
- Return the absolute path to `ws/solution/report.md` and a one-line honest summary of the
  true hold-out metric.
