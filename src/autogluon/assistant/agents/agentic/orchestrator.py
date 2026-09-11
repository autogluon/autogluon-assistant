#!/usr/bin/env python3
"""`mlzero-agent` — a headless, end-to-end Claude-Agent orchestrator (P3).

Launches an autonomous Claude Agent (via the Claude Agent SDK, on the ambient
CLAUDE_CODE_USE_BEDROCK — no API key) that runs the whole `mlzero-solve` flow: perceive the data,
prepare a leakage-safe split, write the spec, provision the env, invoke `mlzero` (with the fixed
per-node scorer), evaluate the winner on the untouched hold-out, and write a clean report. This is
exactly what a human running the mlzero-* skills does by hand — packaged so it can be benchmarked.

Usage:
    mlzero-agent -i <data_folder> -t "<natural-language goal>" [-o <output_dir>] [--model M] [--max-turns N]
"""

import argparse
import os
import sys
from pathlib import Path

# repo root = five parents up from .../src/autogluon/assistant/agents/agentic/orchestrator.py
REPO_ROOT = Path(__file__).resolve().parents[5]

SYSTEM_PROMPT = """You are the MLZero end-to-end orchestrator, running FULLY AUTONOMOUS and HEADLESS
(there is no human to answer questions — never pause to ask; make sensible defaults from the data and
proceed). Your job: turn a data folder + a one-line goal into a VERIFIABLE ML solution and a clean report,
by following the repo's `mlzero-solve` skill flow. The skills live under `.claude/skills/mlzero-*` (each has
a SKILL.md and helper .py scripts); read them as needed and run their scripts with Bash.

Flow (do them in order; use ABSOLUTE paths; put everything under the given workspace):
1. profile      — `.claude/skills/mlzero-profile/profile.py --data <input> --out <ws>/profile.json`
2. spec         — decide problem_type / target / metric / split from the profile (no human); ALSO pick ONE
                  size/compute budget that sets both the training preset and the mlzero iteration count N.
                  Write `<ws>/spec.json`.
3. prepare      — `.claude/skills/mlzero-prepare/split.py` → train/select/holdout under `<ws>/data/`; then
                  `.claude/skills/mlzero-prepare/make_eval_inputs.py --select-csv <ws>/data/select/<f>.csv
                  --target <target> --out-dir <ws>/eval` → validation_features.csv + validation_ground_truth.csv.
4. description  — write `<ws>/data/train/description.md` per `mlzero-spec` (open-format spec + output contract
                  + size-aware preset guidance; NO hard training time_limit).
5. env          — REUSE the already-pinned conda env you are running in (its prefix is the output of
                  `python -c "import sys;print(sys.prefix)"`, typically /opt/conda/envs/mlzero); write it to
                  `<ws>/env/env_path.txt`. Do NOT build a new env.
6. run          — invoke mlzero on the TRAIN data only, handing it the reused env AND the fixed scorer so every
                  node's reward is the same deterministic metric. mlzero can run for many MINUTES–HOURS, which
                  exceeds the Bash tool's timeout, so launch it in the BACKGROUND and poll:
                  `nohup mlzero -i <ws>/data/train -o <ws>/mlzero_run --provider bedrock --conda-env <env>
                   -n <N> --continuous_improvement -t "<spec top lines>"
                   --eval-script .claude/skills/mlzero-eval/score_one.py
                   --eval-ground-truth <ws>/eval/validation_ground_truth.csv
                   --eval-validation-features <ws>/eval/validation_features.csv --eval-metric <metric>
                   > <ws>/mlzero_cli.log 2>&1 &`
                  then poll in a loop: `sleep 240; tail -3 <ws>/mlzero_run/logs.txt` until it logs
                  "MCTS search completed" (or the process is gone). Keep N SMALL and the preset LIGHT for large
                  data so it finishes. Finally run `.claude/skills/mlzero-run/normalize_result.py` →
                  `<ws>/mlzero_result.json`.
7. eval         — `.claude/skills/mlzero-eval/evaluate.py --workspace <ws>` → pick the winner on `select`,
                  score it ONCE on `holdout` → `<ws>/eval/holdout_metrics.json`.
8. report       — `.claude/skills/mlzero-report/make_report.py --workspace <ws>` → `<ws>/solution/report.md`
                  (+ run_summary.json + reproduce.sh + copied model/code).

Rules: never let mlzero see `<ws>/data/holdout/`. Never fabricate results. If mlzero produces no successful
node, say so honestly in the report. When done, print the absolute path to `<ws>/solution/report.md` on the
LAST line, prefixed exactly with `REPORT: `.
"""


def _task_prompt(input_dir: str, goal: str, workspace: str) -> str:
    return (
        f"Input data folder: {input_dir}\n"
        f"Goal: {goal or '(infer a sensible predictive task from the data)'}\n"
        f"Workspace (create it; put all outputs here): {workspace}\n\n"
        "Run the full mlzero-solve flow now, autonomously, and finish by printing the REPORT: <path> line."
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", required=True, help="Path to the data folder")
    ap.add_argument("-t", "--goal", default="", help="One-line natural-language goal")
    ap.add_argument("-o", "--output", default=None, help="Workspace dir (default: runs/agent-<name>)")
    ap.add_argument("--model", default=None, help="Model id override (else the SDK/Bedrock default)")
    ap.add_argument("--max-turns", type=int, default=150)
    args = ap.parse_args()

    from .sdk_runtime import run_orchestrator_sync, sdk_available

    if not sdk_available():
        sys.exit("claude-agent-sdk is not installed; `pip install claude-agent-sdk` to use mlzero-agent.")

    input_dir = os.path.abspath(args.input)
    if not os.path.exists(input_dir):
        sys.exit(f"input not found: {input_dir}")
    workspace = os.path.abspath(args.output or os.path.join("runs", "agent-" + Path(input_dir).name))
    os.makedirs(workspace, exist_ok=True)

    def _log(msg):
        name = type(msg).__name__
        if name in ("AssistantMessage", "ResultMessage", "SystemMessage"):
            print(f"[{name}]", flush=True)

    final = run_orchestrator_sync(
        prompt=_task_prompt(input_dir, args.goal, workspace),
        cwd=str(REPO_ROOT),
        system_prompt=SYSTEM_PROMPT,
        model=args.model,
        max_turns=args.max_turns,
        on_message=_log,
    )
    print("\n===== agent final =====")
    print(final or "(no final message)")
    # Locate the report regardless of what the agent printed.
    report = os.path.join(workspace, "solution", "report.md")
    if os.path.isfile(report):
        print(f"\nREPORT: {report}")
    else:
        print(f"\n(no report.md found under {workspace}/solution)")


if __name__ == "__main__":
    main()
