---
name: mlzero-report
description: Use as the final step to assemble the clean deliverable — report.md, run_summary.json, a reproduce.sh, and copies of the saved model and code — from the run workspace.
---

# mlzero-report

**Purpose.** Produce the human-facing, verifiable deliverable: an honest report of what was
built and how it scored on the hold-out, plus everything needed to reproduce it.

**Inputs.** The whole workspace — `<ws>/spec.json`, `<ws>/profile.json`,
`<ws>/mlzero_result.json`, `<ws>/eval/*`, `<ws>/data/split_manifest.json`, `<ws>/env/env_path.txt`.

**Steps.**
1. Run the helper:
   ```
   python .claude/skills/mlzero-report/make_report.py --workspace <ws>
   ```
   It writes the report, summary, and reproduce script, and copies the model + code into
   `<ws>/solution/`.
2. Read `<ws>/solution/report.md` and confirm it states the task, the split strategy, the TRUE
   hold-out metric (with metric name), and any leakage flags / retry that occurred.

**Outputs.** `<ws>/solution/report.md`, `<ws>/solution/run_summary.json`,
`<ws>/solution/reproduce.sh`, plus copied `<ws>/solution/{model/,train_code.py,evaluate.py}`.

**Notes/guardrails.**
- Verifiable = `reproduce.sh` + saved model + `report.md`/`run_summary.json`. Ensure
  `reproduce.sh` references the pinned env from `env_path.txt` and passes data via
  `MLZERO_DATA_PATH`.
- Report the TRUE hold-out metric prominently and honestly; if it diverges from mlzero's
  validation score or a leakage flag fired, say so plainly — do not oversell.
- Return the absolute path to `<ws>/solution/report.md`.
