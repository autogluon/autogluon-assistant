---
name: mlzero-prepare
description: Use after the spec is written to create a guaranteed leakage-safe 3-way split (train / select / hold-out) with a strategy chosen from the profile, and to drop covariates that would not be known at prediction time.
---

# mlzero-prepare

**Purpose.** Carve THREE disjoint sets so evaluation is honest: `train` (the only data mlzero sees),
`select` (a validation set the skill uses to CHOOSE among candidate solutions — never given to mlzero),
and `holdout` (the test set, used ONCE to report the chosen solution's true metric — never used for
selection). Also removes leakage-prone (unknown-future) covariates so all three are honest.

**Inputs.** `<ws>/profile.json`, `<ws>/spec.json`, and the source data.

**Steps.**
1. **Pick the strategy** from `spec.split.strategy` (defaulted from `profile.split_policy_recommendation`):
   - `random` — general default;
   - `stratified` — classification (preserve class balance; needs `--target`);
   - `group` — hold out whole entities so none straddle any split (needs `--id-cols`);
   - `temporal` / `last_h_per_entity` — panel time series (per entity: train < select < holdout);
   - `global_cutoff` — time series with ≈1 timestamp per entity.
2. **Run the split**:
   ```
   python .claude/skills/mlzero-prepare/split.py \
     --data <source.csv> --strategy <strategy> --target <y> --out-dir <ws>/data \
     [--holdout-frac 0.2] [--select-frac 0.15] [--id-cols a,b,c] \
     [--time-col t] [--horizon H] [--min-history 2H] [--seed 42] [--no-select]
   ```
   `--holdout-frac`/`--select-frac` apply to random/stratified/group; time/horizon flags to temporal
   (temporal reserves the last H timestamps for holdout and the H before that for select). Pass
   `--no-select` only if you will NOT re-rank candidates (then eval falls back to mlzero's best_run).
3. **Drop unknown-future covariates.** Per `spec.known_covariates`, remove any column NOT confirmed
   known-in-advance from all three splits, and record it.
4. Read `<ws>/data/split_manifest.json` and confirm its `assertions` all passed (holdout & select
   non-empty; no overlap; for temporal, train < select < holdout).

**Outputs.** `<ws>/data/train/<name>.csv`, `<ws>/data/select/<name>.csv`, `<ws>/data/holdout/<name>.csv`,
`<ws>/data/split_manifest.json`.

**Notes/guardrails.**
- Forecasting is ONE branch, not the default — do not force temporal flags on non-forecast tasks.
- The `select` set exists so node selection NEVER touches the test set (avoids selection-on-test bias).
- If an assertion fails, STOP and surface it; do not proceed to a run.
- For folder/multimodal data, split on the label/manifest CSV; if none exists, note the limitation and
  prefer a `random` file-level split.
