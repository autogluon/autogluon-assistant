#!/usr/bin/env python3
"""Normalize mlzero's real output layout into a stable mlzero_result.json (DEFENSIVE).

mlzero does NOT reliably honor the requested output contract, and its layout varies. This
follows `best_run` (symlink) → mlzero's chosen node, AND enumerates the other successful nodes
as `candidates` so the skill can re-rank them on a *validation* set (never the test set). Each
node entry carries its own artifacts (code, predictor dir, inferred tool). Missing pieces become
null, not crashes.

Usage: python normalize_result.py --mlzero-run <run_dir> --out <mlzero_result.json> [--max-candidates 6]
"""

import argparse
import glob
import json
import os
import re


def _find_predictor_dir(root):
    # Return a real top-level predictor dir, or None. AutoGluon's internal DyStack sub-fit
    # artifacts (…/ds_sub_fit/sub_fit_ho/…) are NOT independently reloadable, so a node whose
    # only predictor is a sub-fit is treated as having no usable model (returns None) — it must
    # not become a scoreable candidate.
    for marker in ("predictor.pkl", "learner.pkl"):
        dirs = [os.path.dirname(h) for h in glob.glob(os.path.join(root, "**", marker), recursive=True)]
        external = [d for d in dirs if "ds_sub_fit" not in d and "sub_fit" not in d]
        if external:
            return sorted(external, key=len)[0]
    return None


def _infer_tool(states_dir, code_path):
    p = os.path.join(states_dir, "selected_tool.txt")
    if os.path.isfile(p):
        lines = open(p).read().strip().splitlines()
        if lines:
            return lines[0].split(",")[0].strip()
    if code_path and os.path.isfile(code_path):
        src = open(code_path, errors="ignore").read()
        for marker, tool in (
            ("TimeSeriesPredictor", "autogluon.timeseries"),
            ("TabularPredictor", "autogluon.tabular"),
            ("MultiModalPredictor", "autogluon.multimodal"),
        ):
            if marker in src:
                return tool
        if "sklearn" in src or "train_test_split" in src:
            return "machine learning"
    return "unknown"


def _node_entry(node_dir):
    """Build a candidate entry from a node dir; None if it has no runnable code."""
    if not node_dir or not os.path.isdir(node_dir):
        return None
    cand = os.path.join(node_dir, "generated_code.py")
    code = cand if os.path.isfile(cand) else next(iter(glob.glob(os.path.join(node_dir, "*.py"))), None)
    if not code:
        return None
    states = os.path.join(node_dir, "states")
    csvs = glob.glob(os.path.join(node_dir, "output", "**", "*.csv"), recursive=True)
    # model dir: prefer node-local (never run-root — that can belong to another node)
    model_dir = _find_predictor_dir(os.path.join(node_dir, "output")) or _find_predictor_dir(node_dir)
    decision = ""
    # The executer saves its verdict per-iteration as decision_<uuid>.txt (not literally
    # "decision.txt"), so read the most recent decision_*.txt.
    dfiles = sorted(glob.glob(os.path.join(states, "decision*.txt")), key=os.path.getmtime)
    if dfiles:
        decision = open(dfiles[-1]).read().strip()
    m = re.search(r"node_(\d+)", os.path.basename(node_dir))
    return {
        "node_id": int(m.group(1)) if m else None,
        "node_path": node_dir,
        "generated_code_path": code,
        "model_dir": model_dir,
        "tool_used": _infer_tool(states, code),
        "predictions_path": csvs[0] if csvs else None,
        "produced_output": bool(csvs or model_dir),
        "decision": decision,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlzero-run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-candidates", type=int, default=6)
    args = ap.parse_args()
    run = args.mlzero_run

    best = os.path.join(run, "best_run")
    best_node = os.path.realpath(best) if os.path.exists(best) else None
    best_entry = _node_entry(best_node)

    # enumerate all node_* dirs; a candidate = produced output OR SUCCESS decision
    entries = []
    for nd in sorted(glob.glob(os.path.join(run, "node_*")), key=lambda p: len(p)):
        e = _node_entry(os.path.realpath(nd))
        if e and (e["produced_output"] or e["decision"].upper() == "SUCCESS"):
            entries.append(e)
    # de-dup by node_path, put mlzero's best first, cap at K
    seen, candidates = set(), []
    for e in ([best_entry] if best_entry else []) + entries:
        if e and e["node_path"] not in seen:
            seen.add(e["node_path"])
            candidates.append(e)
    candidates = candidates[: args.max_candidates]

    # run-level logs (mlzero's own, untrustworthy score kept only for reference)
    score = None
    tp = os.path.join(run, "logs.txt")
    if os.path.isfile(tp):
        m = re.findall(r"Best validation score:\s*([-\d.eE]+)", open(tp, errors="ignore").read())
        if m:
            try:
                score = float(m[-1])
            except ValueError:
                pass
    tu = {}
    tup = os.path.join(run, "token_usage.json")
    if os.path.isfile(tup):
        try:
            tu = json.load(open(tup)).get("total", {})
        except Exception:  # noqa: BLE001
            tu = {}

    result = {
        "success": bool(best_entry and best_entry["generated_code_path"]),
        "mlzero_best_node": best_entry,  # mlzero's own pick (NOT trusted for final selection)
        "mlzero_internal_score": score,  # reference only; LLM-scraped, cross-node-inconsistent
        "candidates": candidates,  # to be re-ranked on the SELECT (validation) set
        "n_candidates": len(candidates),
        "token_usage": tu,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(
        json.dumps(
            {
                "success": result["success"],
                "n_candidates": result["n_candidates"],
                "mlzero_best_node_id": (best_entry or {}).get("node_id"),
                "candidate_node_ids": [c["node_id"] for c in candidates],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
