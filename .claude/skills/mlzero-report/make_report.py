#!/usr/bin/env python3
"""Assemble the verifiable deliverable for an mlzero-solve run.

Writes solution/report.md (honest, hold-out-first), run_summary.json (machine-readable), and
reproduce.sh (rebuild env → re-run winning code on train → re-eval on holdout → assert metric),
and copies the saved model + winning code + evaluate.py into solution/.

Usage: python make_report.py --workspace <ws>
"""
import argparse
import json
import os
import shutil


def _load(p, default=None):
    try:
        return json.load(open(p))
    except Exception:  # noqa: BLE001
        return default if default is not None else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    args = ap.parse_args()
    ws = os.path.abspath(args.workspace)
    sol = os.path.join(ws, "solution")
    os.makedirs(sol, exist_ok=True)

    spec = _load(os.path.join(ws, "spec.json"))
    result = _load(os.path.join(ws, "mlzero_result.json"))
    metrics = _load(os.path.join(ws, "eval", "holdout_metrics.json"))
    manifest = _load(os.path.join(ws, "data", "split_manifest.json"))
    env_path = ""
    ep = os.path.join(ws, "env", "env_path.txt")
    if os.path.isfile(ep):
        env_path = open(ep).read().strip()

    hv, hname, hmethod = metrics.get("value"), metrics.get("metric"), metrics.get("method")
    direction = metrics.get("direction", "")
    leak = metrics.get("leakage_flag")

    # Resolve the ACTUAL winning node: eval selects it on the validation (select) set, so the
    # winner lives in holdout_metrics.winner_node_id, not at the top level of mlzero_result.json.
    cands = result.get("candidates") or []
    wid = metrics.get("winner_node_id")
    winner = next((c for c in cands if str(c.get("node_id")) == str(wid)), None) or {}
    if not winner and isinstance(result.get("mlzero_best_node"), dict):
        winner = result["mlzero_best_node"]
    tool_used = winner.get("tool_used") or result.get("tool_used") or "?"
    best_node_id = winner.get("node_id", result.get("best_node_id", "?"))
    model_dir = winner.get("model_dir") or result.get("model_dir")
    code_path = winner.get("generated_code_path") or result.get("generated_code_path")
    internal = winner.get("internal_score") or result.get("mlzero_internal_score") or result.get("internal_score")
    token_usage = result.get("token_usage") or {}

    # copy artifacts (best-effort)
    if model_dir and os.path.isdir(model_dir):
        shutil.copytree(model_dir, os.path.join(sol, "model"), dirs_exist_ok=True)
    if code_path and os.path.isfile(code_path):
        shutil.copy2(code_path, os.path.join(sol, "train_code.py"))
    ev = os.path.join(os.path.dirname(__file__), "..", "mlzero-eval", "evaluate.py")
    if os.path.isfile(ev):
        shutil.copy2(ev, os.path.join(sol, "evaluate.py"))

    # report.md
    task_line = spec.get("task", spec.get("task_type", "(task)"))
    rep = f"""# MLZero solution report

## Task
{task_line}
- task_type: **{spec.get('task_type','?')}** · target: `{spec.get('target','?')}` · metric: **{hname or spec.get('eval_metric','?')}**

## Data & split
- strategy: `{(manifest.get('params') or {}).get('strategy', spec.get('split',{}).get('strategy','?'))}`
- train rows: {manifest.get('n_train_rows','?')} · hold-out rows: {manifest.get('n_holdout_rows','?')} · dropped entities: {manifest.get('n_dropped_entities',0)}
- split assertions: {', '.join(f"{a['name']}={a['passed']}" for a in manifest.get('assertions', []))}

## Approach (from mlzero)
- tool: `{tool_used}` · winning node: {best_node_id} (selected on validation set) · code: `{code_path or '?'}`

## Results
| metric | value | source |
|---|---|---|
| **{hname}** (TRUE, held-out; {direction}) | **{hv}** | skill `eval` ({hmethod}) |
| mlzero internal validation | {internal} | mlzero self-report |

{"> ⚠️ **Leakage / sanity flag raised** — the held-out result is suspect; see holdout_metrics.json." if leak else ""}
- sanity: {json.dumps(metrics.get('sanity', {}))}

## Cost
- tokens: {token_usage}
- selection: winner chosen on the validation (`select`) set among {len(cands)} candidate node(s); hold-out used once.

## Reproduce
```
bash solution/reproduce.sh
```
Rebuilds the pinned env, re-trains on `data/train/`, re-evaluates on the held-out set, and asserts
the same {hname}. Artifacts: `solution/model/`, `solution/train_code.py`, `solution/evaluate.py`.
"""
    open(os.path.join(sol, "report.md"), "w").write(rep)

    run_summary = {
        "task_type": spec.get("task_type"),
        "target": spec.get("target"),
        "tool": tool_used,
        "best_node": best_node_id,
        "internal_score": internal,
        "holdout_metric": {"name": hname, "value": hv, "direction": direction, "method": hmethod},
        "selected_by": metrics.get("selected_by"),
        "n_candidates": len(cands),
        "leakage_flag": leak,
        "split": manifest.get("params"),
        "token_usage": token_usage,
        "artifacts": {"report": "solution/report.md", "reproduce": "solution/reproduce.sh",
                      "predictions": "eval/predictions_holdout.csv", "model": "solution/model"},
    }
    json.dump(run_summary, open(os.path.join(sol, "run_summary.json"), "w"), indent=2, default=str)

    tol = "0.05"
    reproduce = f"""#!/bin/bash
set -euo pipefail
# Reproduce this mlzero solution end-to-end.
WS="{ws}"
ENV="{env_path or '<conda-env-prefix>'}"
# 1. (Re)build the pinned env if needed:
#    conda create -y -p "$ENV" python=3.11 && "$ENV/bin/pip" install "autogluon==1.6.1"
# 2. Re-train on the training split (seed fixed in the code / description):
MLZERO_DATA_PATH="$WS/data/train" "$ENV/bin/python" "$WS/solution/train_code.py"
# 3. Re-evaluate on the held-out set and assert the metric is within tolerance ({tol}):
"$ENV/bin/python" "$WS/solution/evaluate.py" --workspace "$WS"
python - "$WS" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]+"/eval/holdout_metrics.json"))
rec={hv}
assert m["value"] is not None and abs(float(m["value"])-float(rec))<= {tol}*max(1,abs(float(rec))), \
    f"metric drift: got {{m['value']}} vs recorded {{rec}}"
print("REPRODUCED: {hname} =", m["value"])
PY
"""
    rp = os.path.join(sol, "reproduce.sh")
    open(rp, "w").write(reproduce)
    os.chmod(rp, 0o755)
    print(f"report: {os.path.join(sol,'report.md')}\nsummary: {os.path.join(sol,'run_summary.json')}\nreproduce: {rp}")


if __name__ == "__main__":
    main()
