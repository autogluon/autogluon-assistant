#!/usr/bin/env python3
"""`mlzero-agent` — a headless Claude-Agent orchestrator for the `mlzero` skill.

Launches an autonomous Claude Agent (via the Claude Agent SDK, on the ambient
CLAUDE_CODE_USE_BEDROCK — no API key) whose entire job is to invoke this repository's `mlzero`
project skill and follow it, from a data folder plus a one-line goal through to a report. The
skill is the workflow; this script only supplies the inputs, the working directory and the
constraints that come from running with no human present, then reports where the agent's report
landed. Packaged as a console script so the same flow a person drives by hand can be run headless
and benchmarked.

Usage:
    mlzero-agent -i <data_folder> -t "<natural-language goal>" [-o <output_dir>] [--model M] [--max-turns N]
"""

import argparse
import os
import sys
from pathlib import Path

# repo root = five parents up from .../src/autogluon/assistant/agents/agentic/orchestrator.py
REPO_ROOT = Path(__file__).resolve().parents[5]

SYSTEM_PROMPT = """You are running FULLY AUTONOMOUS and HEADLESS: there is no human to answer questions, so
never pause to ask one. Where a choice is genuinely open, decide it from what the data and this repository
actually show you, write down the decision and the reason you chose it, and proceed.

This repository ships a project skill named `mlzero`, whose directory is `.claude/skills/mlzero/` under your
working directory. That skill is the authority on how this work is done, and it is the workflow: invoke it
before you plan anything, read its entry document in full, and follow it — including the further material it
tells you to open, at the point where it tells you to open it. Do not substitute a procedure of your own for
it, and do not skip the parts that ask you to prove something because they look like overhead; those parts
are why the result is worth reporting at all. If the Skill tool is not available to you in this session, read
the skill from that directory and follow it exactly the same way.

Constraints this session adds on top of the skill:
- Put everything you produce under the workspace directory named in the task, and use absolute paths
  throughout.
- Work you launch may run far longer than a single tool call is permitted to last. Start anything
  long-running detached with its output redirected to a file under the workspace, then poll that file — and
  whether the process still exists — until it reports that it finished. Letting a tool call time out kills
  the work you were waiting on.
- Every claim you make has to be traceable to output that something actually printed. Keep what you verified
  that way separate from what you only requested or intended: a request nobody checked is not a finding. If
  the work cannot deliver what was asked for, say so plainly and say why — an honest negative result is a
  correct outcome, and a number you did not measure never is.

Finish by writing your report to a file under the workspace and printing that file's absolute path on the
LAST line, prefixed exactly with `REPORT: `. That line is how the caller of this session finds your work.
"""


def _task_prompt(input_dir: str, goal: str, workspace: str) -> str:
    return (
        f"Input data folder: {input_dir}\n"
        f"Goal: {goal or '(none given — derive the objective from what the data itself shows)'}\n"
        f"Workspace (create it; put all outputs here): {workspace}\n\n"
        "Invoke the `mlzero` skill now, carry it through end to end autonomously, and finish by printing "
        "the REPORT: <path> line."
    )


def _reported_report_path(final: str):
    """Return the report path the agent printed, but only if that path really is a file.

    The skill fixes no report filename, so there is nothing here to guess at: either the agent handed
    back a path that resolves, or this run produced no report and must not be described as if it had.
    """
    for line in reversed((final or "").splitlines()):
        stripped = line.strip()
        if stripped.startswith("REPORT:"):
            path = os.path.abspath(stripped[len("REPORT:") :].strip())
            if os.path.isfile(path):
                return path
    return None


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
    report = _reported_report_path(final)
    if report:
        print(f"\nREPORT: {report}")
    else:
        print(f"\n(no reported report file exists; inspect the workspace directly: {workspace})")


if __name__ == "__main__":
    main()
