---
name: mlzero-spec
description: Use after preparing the split to render the open-format description.md that mlzero reads — the task spec plus an output contract, determinism request, and a note that a hidden hold-out exists.
---

# mlzero-spec

**Purpose.** Write `data/train/description.md`, the single natural-language brief mlzero
consumes. It stays OPEN-FORMAT (no forced flags): it states the task, target, metric, and
split context, and requests an output contract — but downstream never assumes the contract
was honored (`mlzero-run` normalizes reality).

**Inputs.** `<ws>/spec.json` (and `<ws>/profile.json` for context).

**spec.json schema (general).** `task` (NL goal), `task_type`, `modality`, `target`,
`eval_metric`, `split` (`strategy` + `holdout_frac`/`id_cols`), `known_covariates`,
`output_contract`, `seed`. **Forecast only:** a `forecast` block with `horizon`,
`time_column`, `id_cols`, `freq`, `known_future_covariates`.

**Steps.**
1. Render `<ws>/data/train/description.md`. Put the most important facts FIRST: the task in
   plain language, task_type, target column (if any), and the chosen `eval_metric`.
2. State the **VALIDATION & EVALUATION METRIC explicitly** (this is what makes mlzero's internal
   node ranking trustworthy — otherwise it scrapes whatever number it finds and compares
   incomparable metrics across candidates). Write, verbatim, using `spec.eval_metric`:
   - "Use `<eval_metric>` as BOTH the internal validation metric AND the single reported score."
   - "Configure the predictor's `eval_metric`/scoring to `<eval_metric>` (do not substitute another
     metric such as R²/accuracy)."
   - "Compute `<eval_metric>` on an internal validation split and print exactly one line to stdout:
     `VALIDATION_SCORE: <number>` — the raw `<eval_metric>` value, and nothing else on that line."
   - "Also record it in `./output/metrics.json` as
     `{\"metric\": \"<eval_metric>\", \"value\": <number>, \"split\": \"internal_validation\"}`."
3. Include the **OUTPUT CONTRACT** verbatim (as a request):
   - save the trained predictor/model to `./model/`;
   - write predictions to `./output/predictions.csv` and scores to `./output/metrics.json`;
   - make the solution re-runnable on new data via the env var `MLZERO_DATA_PATH`;
   - for reproducibility, set library random seeds (e.g. numpy/random and the framework's own
     seed utility) to `spec.seed` — but do NOT pass a seed/`random_seed` keyword to the trainer's
     `fit()`/`train()` call unless that framework documents it (some, e.g. AutoGluon, reject it).
4. State that **a hidden hold-out set exists** and will score the solution, so the code must
   not leak or peek beyond `data/train`.
5. Add task-specific detail ONLY when relevant (e.g. forecast horizon/freq/known-future
   covariates; class list for classification). Omit time-series vocabulary for non-forecast tasks.
6. **Compute budget (size/compute-aware; NEVER a hard training `time_limit`).** State the training
   intensity to use, chosen to match the dataset size and available compute and to stay consistent
   with the search-iteration budget (`N`, decided in `mlzero-run` from the same assessment): heavier
   when data is small or compute is ample, lighter when data is large or compute is limited. For
   AutoGluon this maps to `presets` (lighter → heavier: medium_quality, good_quality, best_quality,
   extreme; the heaviest need a GPU plus foundation-model dependencies); other frameworks have
   analogous knobs. Use your judgment on the actual size/compute rather than fixed thresholds, phrase
   it as guidance (not a mandate), and never impose a training `time_limit` that could truncate a run.

**Outputs.** `<ws>/data/train/description.md` (keep < 16 KB; key facts first).

**Notes/guardrails.**
- The contract is a REQUEST, not a guarantee — never rely on the exact paths downstream;
  `mlzero-run/normalize_result.py` maps mlzero's real layout.
- Do not impose a training time limit or forbid libraries; keep it open-format.
- If `spec` lacks a metric, do not invent one silently — resolve it in `mlzero-solve`'s clarify.
