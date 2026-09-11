---
name: mlzero-profile
description: Use as the first step of an mlzero-solve run to inspect a dataset and infer its task shape — modality, task_type, target candidates, a recommended hold-out split strategy, and per-column leakage risks.
---

# mlzero-profile

**Purpose.** Produce a machine-readable profile of the input data so the orchestrator can
propose a spec and pick a split strategy. Task-agnostic: works for tabular, text, image,
audio, document, and time-series inputs.

**Inputs.** The user's data path (CSV or folder) from the workspace context.

**Steps.**
1. Run the helper:
   ```
   python .claude/skills/mlzero-profile/profile.py \
     --data <csv|folder> --out <ws>/profile.json [--sample-rows 200000]
   ```
2. Read `<ws>/profile.json` and sanity-check it before returning.

**Outputs.** `<ws>/profile.json`, containing at least:
- `problem_type` / `task_type` — `classification | regression | forecast | other`.
- `modality` — `tabular | text | image | audio | document | multimodal | mixed`.
- `target_candidates` — ranked likely target column(s); empty for non-tabular.
- `split_policy_recommendation` — `random | stratified | group | temporal |
  last_h_per_entity | global_cutoff`.
- per-column stats with `leakage_risk` flags (constant, id-like, same-timestamp covariate, etc.).
- **forecast only:** `time_column`, `entity_id_columns`, `freq`, `timestamps_per_entity`.

**Notes/guardrails.**
- For non-CSV / multimodal folder inputs, the profiler defers detail to mlzero and
  recommends a **random** hold-out; expect sparse column stats — that is fine.
- Recommendation heuristics (surfaced to the user in `mlzero-solve`'s clarify step):
  discrete/low-cardinality target ⇒ classification (stratified); continuous target ⇒
  regression (random); a time column with many timestamps per entity ⇒ forecast
  (`last_h_per_entity`), or `global_cutoff` when timestamps/entity ≈ 1; distinct entities that
  must not straddle the split ⇒ group.
- Read-only: this skill never mutates the source data.
