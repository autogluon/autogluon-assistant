---
name: mlzero-benchmark
description: Use to benchmark the MLZero approach across one or more tasks — freeze a leakage-safe hold-out per task, run each arm (trivial baseline, raw mlzero, the mlzero-agent orchestrator) SERIALLY, score every arm on the SAME frozen hold-out with the one shared scorer, and tabulate a leaderboard (mean-rank + improvement-over-trivial). Generalizes to any task; there is no fixed task schema or hardcoded harness.
---

# mlzero-benchmark

**Purpose.** Fair, apples-to-apples comparison of approaches on held-out data. Every arm is scored on
the **same frozen, leakage-safe hold-out** by the **same shared metric**
(`.claude/skills/mlzero-eval/score_one.py`). You (Claude Code) drive it — adapt the tasks, arms, and
metrics to the situation rather than following a rigid script.

**Inputs.** A list of tasks. Each task is just a data path + one-line goal; infer its shape with
`mlzero-profile` (target, task_type, metric, split strategy) or take them from the user. Arms to
compare (default: `trivial`, `mlzero_raw`, `mlzero_agent`).

**Steps.**
1. **Freeze the split once per task (fixed seed).** Run `mlzero-prepare` (`split.py`) →
   `train/`, `select/`, `holdout/`. Derive `holdout/features.csv` (drop the target) and
   `holdout/labels.csv` (target only — keep HIDDEN from every arm). Run
   `mlzero-prepare/make_eval_inputs.py` on `select` for the mlzero arms' per-node scorer inputs. The
   SAME frozen split is reused by every arm and repeat, so all are scored on identical rows.
2. **Run each (task, arm) SERIALLY** — never concurrently (parallel AutoGluon OOMs the host) and with a
   per-arm wall-clock bound. Each arm must emit predictions on `holdout/features.csv`, row-aligned:
   - **trivial** — `python .claude/skills/mlzero-benchmark/trivial_baseline.py --train <train.csv>
     --features <holdout/features.csv> --target <t> --task-type <tt> --out <arm>/predictions.csv`.
   - **mlzero_raw** — `mlzero -i <train_dir> -o <arm>/run --conda-env <env> -n <N> --continuous_improvement
     --eval-script .claude/skills/mlzero-eval/score_one.py --eval-ground-truth <select GT>
     --eval-validation-features <select feats> --eval-metric <m>`; then reload the winner and predict on
     `holdout/features.csv` (reuse `mlzero-eval/evaluate.py`'s reload/rerun logic) → `predictions.csv`.
   - **mlzero_agent** — `mlzero-agent -i <train_dir> -t "<goal>" -o <arm>/ws`; then predict its winning
     model on `holdout/features.csv` the same way → `predictions.csv`.
3. **Score every arm the same way.** `python .claude/skills/mlzero-eval/score_one.py
   --ground-truth <holdout/labels.csv> --predictions <arm>/predictions.csv --metric <m>` →
   `MLZERO_SCORE: <float>` (higher-is-better; the scorer negates lower-is-better metrics). Record it.
4. **Aggregate a leaderboard.** For each task rank arms by score (desc) and compute improvement over the
   `trivial` arm; report per-task scores plus overall **mean-rank** AND **mean-improvement-over-trivial**.
   Write `leaderboard.md` (+ a small JSON if useful).

**Outputs.** A `runs/<id>/` workspace: per-arm `predictions.csv` + `result` per (task, arm), and
`leaderboard.md`.

**Notes / guardrails.**
- Serialize heavy arms; bound each with a wall-clock; a host OOM (exit 137) must be caught, not fatal.
- Never let any arm see `holdout/labels.csv`. The frozen split + shared scorer are what make it fair.
- Flag SMALL datasets (e.g. the 1000-row FTG sample) as indicative, not production numbers.
- This is standalone — it does NOT use `maab/` (a separate, older harness).
