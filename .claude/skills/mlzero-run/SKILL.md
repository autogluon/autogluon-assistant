---
name: mlzero-run
description: Use to assemble and execute the mlzero CLI command against the prepared training data, then normalize mlzero's real output layout into a stable result schema.
---

# mlzero-run

**Purpose.** Invoke the `mlzero` CLI on the train-only data with the pinned env, then map its
actual output (which rarely matches the requested contract) into `mlzero_result.json`.

**Inputs.** `<ws>/data/train/` (incl. `description.md`), `<ws>/env/env_path.txt`, `<ws>/spec.json`,
and the max-iteration budget `N`.

**Steps.**
0. **Choose the iteration budget `N` from the size/compute assessment** (the same one that sets the
   training preset in `mlzero-spec`): use more iterations when each run is cheap (small/fast data) so
   the search explores more candidates, and fewer when each run is expensive (large data or a heavy
   preset) to bound wall-clock. Keep `N` and the preset consistent — never pair the heaviest preset
   with a large `N` on a large dataset. Use judgment on the actual scale rather than fixed numbers.
1. Read the env prefix from `<ws>/env/env_path.txt` and the first ~15 lines of
   `description.md` for the `-t` instruction.
1b. **Derive the fixed-scorer inputs from the `select` split** (so every node's reward is the SAME
   deterministic metric, not the LLM's stdout scrape):
   ```
   python .claude/skills/mlzero-prepare/make_eval_inputs.py \
     --select-csv <ws>/data/select/<name>.csv --target <target> --out-dir <ws>/eval
   ```
   → `<ws>/eval/validation_features.csv` (given to each node to predict on) and
   `<ws>/eval/validation_ground_truth.csv` (read only by the scorer). Skip this only when there is no
   `select` split (e.g. `--no-select`), in which case mlzero falls back to its LLM-scraped score.
2. Run mlzero (never point it at the hold-out), passing the fixed scorer so node rewards are comparable:
   ```
   mlzero -i <ws>/data/train -o <ws>/mlzero_run \
     --provider bedrock --conda-env <env_path> \
     -n <N> --continuous_improvement -t "<top lines of the spec>" \
     --eval-script .claude/skills/mlzero-eval/score_one.py \
     --eval-ground-truth <ws>/eval/validation_ground_truth.csv \
     --eval-validation-features <ws>/eval/validation_features.csv \
     --eval-metric <metric>
   ```
3. Normalize the result — follows the `best_run` symlink and maps the real layout
   (`node_*/output/results.csv`, `model_<ts>/` predictor dir, `states/`, etc.):
   ```
   python .claude/skills/mlzero-run/normalize_result.py \
     --mlzero-run <ws>/mlzero_run --out <ws>/mlzero_result.json
   ```
4. Confirm `<ws>/mlzero_result.json` lists `candidates` (successful nodes, each with its own
   code/model/tool) plus mlzero's own `mlzero_best_node`. `mlzero-eval` re-ranks the candidates on the
   validation (`select`) set — mlzero's `best_run` is kept only for reference, not trusted as final.

**Outputs.** `<ws>/mlzero_run/` (full mlzero tree, incl. `best_run` symlink) and
`<ws>/mlzero_result.json` (normalized: `candidates` list [node id, code, model dir, tool], `mlzero_best_node`,
and mlzero's `mlzero_internal_score` for reference only — it is LLM-scraped and cross-node-inconsistent).

**Notes/guardrails.**
- NEVER assume the coder honored the OUTPUT CONTRACT — always trust `normalize_result.py` over
  the requested `./model` / `./output` paths.
- `--continuous_improvement` keeps searching past the first success for a better solution;
  `--conda-env` makes mlzero *activate* the prepared env instead of building its own (new flag
  added by this initiative).
- Hard agents (coder, perception) run on Opus 4.8 per mlzero's default config; override models
  via `-c <config.yaml>` only if the user asks.
- If mlzero produces no successful node, record that in the result and let `mlzero-solve`
  decide on the single retry.
