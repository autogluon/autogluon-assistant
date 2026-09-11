#!/usr/bin/env python3
"""Deterministic held-out evaluator for the mlzero skills suite (GENERAL, no LLM).

Two-stage, statistically honest:
  1. SELECT — among the candidate nodes, choose the winner by its score on the `select` (validation)
     set. The test set is NEVER used for selection.
  2. TEST   — score the single chosen winner on the `holdout` set ONCE → the reported true metric
     (an unbiased generalization estimate for the selected solution).

Predictions per node come "support both": reload the node's saved model, else re-run its code on the
eval data (via MLZERO_DATA_PATH). If neither works for the winner, the eval FAILS EXPLICITLY.

Usage: python evaluate.py --workspace <ws>
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd


# ----------------------------- general metric registry -----------------------------
def _rmse(y, p, **k):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def _mae(y, p, **k):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def _mape(y, p, **k):
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = y != 0
    return float(np.mean(np.abs((y[m] - p[m]) / y[m]))) if m.any() else float("nan")


def _mase(y, p, *, train=None, **k):
    y, p = np.asarray(y, float), np.asarray(p, float)
    d = np.mean(np.abs(np.diff(np.asarray(train, float)))) if train is not None and len(train) > 1 else np.nan
    return float(np.mean(np.abs(y - p)) / d) if d and d == d and d != 0 else float("nan")


def _accuracy(y, p, **k):
    from sklearn.metrics import accuracy_score

    return float(accuracy_score(y, p))


def _f1_macro(y, p, **k):
    from sklearn.metrics import f1_score

    return float(f1_score(y, p, average="macro"))


def _roc_auc(y, p, **k):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p))


SCORERS = {
    "rmse": (_rmse, "lower_is_better"),
    "root_mean_squared_error": (_rmse, "lower_is_better"),
    "mae": (_mae, "lower_is_better"),
    "mean_absolute_error": (_mae, "lower_is_better"),
    "mape": (_mape, "lower_is_better"),
    "mase": (_mase, "lower_is_better"),
    "accuracy": (_accuracy, "higher_is_better"),
    "f1": (_f1_macro, "higher_is_better"),
    "f1_macro": (_f1_macro, "higher_is_better"),
    "roc_auc": (_roc_auc, "higher_is_better"),
}


def score(metric, y_true, y_pred, train_target=None):
    fn, direction = SCORERS.get((metric or "rmse").lower(), SCORERS["rmse"])
    return fn(y_true, y_pred, train=train_target), direction


def _only_csv(d):
    fs = glob.glob(os.path.join(d, "*.csv"))
    return fs[0] if fs else None


def _predict(cand, spec, train_df, eval_df, eval_dir, ws, tag, allow_rerun=True):
    """Return predictions (Series aligned to eval_df rows) for one candidate node, or None.

    allow_rerun=False disables the (expensive) retrain-from-code fallback — used during the SELECT
    stage so one candidate whose saved model won't reload can't trigger a full retrain: it is simply
    skipped (returns None) and cannot win. The rerun fallback is reserved for the single chosen winner.
    """
    target = spec["target"]
    model_dir, tool = cand.get("model_dir"), (cand.get("tool_used") or "").lower()
    # ---- reload ----
    if model_dir and os.path.isdir(model_dir):
        try:
            if "tabular" in tool:
                from autogluon.tabular import TabularPredictor

                p = TabularPredictor.load(model_dir)
                return pd.Series(p.predict(eval_df.drop(columns=[target], errors="ignore"))).reset_index(drop=True)
            if "timeseries" in tool:
                from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

                p = TimeSeriesPredictor.load(model_dir)
                idc = (spec.get("entity_id_columns") or ["item_id"])[0]
                tcol = spec.get("time_column")
                hist = TimeSeriesDataFrame.from_data_frame(train_df, id_column=idc, timestamp_column=tcol)
                fc = p.predict(hist).reset_index()
                key = [idc, tcol]
                merged = eval_df.merge(fc.rename(columns={"mean": f"{target}_pred"}), on=key, how="left")
                if merged[f"{target}_pred"].notna().any():
                    return merged[f"{target}_pred"].reset_index(drop=True)
        except Exception as e:  # noqa: BLE001
            print(f"[reload:{tag}] {cand.get('node_id')} failed: {e}", file=sys.stderr)
    # ---- rerun ----
    code = cand.get("generated_code_path")
    if allow_rerun and code and os.path.isfile(code):
        out = os.path.join(ws, "eval", f"_rerun_{tag}_{cand.get('node_id')}.csv")
        env = dict(
            os.environ, MLZERO_DATA_PATH=eval_dir, MLZERO_OUTPUT_PATH=os.path.join(ws, "eval"), MLZERO_PRED_FILE=out
        )
        ep = os.path.join(ws, "env", "env_path.txt")
        py = os.path.join(open(ep).read().strip(), "bin", "python") if os.path.isfile(ep) else sys.executable
        alt = os.path.join(ws, "eval", "_rerun_predictions.csv")
        try:
            if os.path.isfile(alt):
                os.remove(alt)
            subprocess.run([py, code], env=env, timeout=3600, check=False)
            pf = out if os.path.isfile(out) else (alt if os.path.isfile(alt) else None)
            if pf:
                return pd.read_csv(pf).iloc[:, -1].reset_index(drop=True)
        except Exception as e:  # noqa: BLE001
            print(f"[rerun:{tag}] {cand.get('node_id')} failed: {e}", file=sys.stderr)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    args = ap.parse_args()
    ws = args.workspace
    spec = json.load(open(os.path.join(ws, "spec.json")))
    result = json.load(open(os.path.join(ws, "mlzero_result.json")))
    target, metric = spec["target"], spec.get("eval_metric", "rmse")
    _, direction = score(metric, [0, 1], [0, 1])  # direction only

    train_df = pd.read_csv(_only_csv(os.path.join(ws, "data", "train")))
    holdout_dir = os.path.join(ws, "data", "holdout")
    holdout_df = pd.read_csv(_only_csv(holdout_dir))
    select_dir = os.path.join(ws, "data", "select")
    select_csv = _only_csv(select_dir) if os.path.isdir(select_dir) else None
    select_df = pd.read_csv(select_csv) if select_csv else None

    candidates = result.get("candidates") or ([result["mlzero_best_node"]] if result.get("mlzero_best_node") else [])
    os.makedirs(os.path.join(ws, "eval"), exist_ok=True)

    # ---- stage 1: SELECT on the validation set (never the test set) ----
    selection_table, winner, selected_by = [], None, None
    if select_df is not None and len(candidates) > 1:
        best_val = None
        for c in candidates:
            # reload-only during selection: never retrain a candidate just to rank it
            preds = _predict(c, spec, train_df, select_df, select_dir, ws, "select", allow_rerun=False)
            row = {"node_id": c.get("node_id"), "tool": c.get("tool_used")}
            if preds is None:
                row["select_score"] = None
            else:
                n = min(len(select_df), len(preds))
                val, _ = score(
                    metric,
                    select_df[target].reset_index(drop=True).iloc[:n],
                    pd.Series(list(preds)[:n]),
                    train_target=train_df[target] if target in train_df else None,
                )
                row["select_score"] = val
                # An implausibly perfect validation score (zero error / ~perfect) signals leakage
                # or a degenerate artifact — record it but do NOT let it win selection.
                leaky = (val <= 0) if direction == "lower_is_better" else (val >= 0.9999)
                if leaky:
                    row["leaky"] = True
                else:
                    better = best_val is None or (val < best_val if direction == "lower_is_better" else val > best_val)
                    if better:
                        best_val, winner = val, c
            selection_table.append(row)
        selected_by = "validation(select)"
    if winner is None:  # no select set, single candidate, or all select-preds failed
        winner = result.get("mlzero_best_node") or (candidates[0] if candidates else None)
        selected_by = selected_by or "mlzero_best_run"

    # ---- stage 2: TEST the single winner on the hold-out ONCE ----
    if winner is None:
        out = {"metric": metric, "value": None, "method": "failed", "error": "no candidate nodes"}
        json.dump(out, open(os.path.join(ws, "eval", "holdout_metrics.json"), "w"), indent=2)
        print(json.dumps(out, indent=2))
        sys.exit(2)

    preds = _predict(winner, spec, train_df, holdout_df, holdout_dir, ws, "holdout")
    method = (
        "reload" if (winner.get("model_dir") and preds is not None) else ("rerun" if preds is not None else "failed")
    )
    if preds is None:
        out = {
            "metric": metric,
            "value": None,
            "method": "failed",
            "selected_by": selected_by,
            "winner_node_id": winner.get("node_id"),
            "error": "winner produced no predictions",
            "selection_table": selection_table,
        }
        json.dump(out, open(os.path.join(ws, "eval", "holdout_metrics.json"), "w"), indent=2, default=str)
        print(json.dumps(out, indent=2, default=str))
        sys.exit(2)

    y_true = holdout_df[target].reset_index(drop=True)
    n = min(len(y_true), len(preds))
    y_true, preds = y_true.iloc[:n], pd.Series(list(preds)[:n])
    value, _ = score(metric, y_true, preds, train_target=train_df[target] if target in train_df else None)
    out = {
        "metric": metric,
        "value": value,
        "direction": direction,
        "method": method,
        "selected_by": selected_by,
        "winner_node_id": winner.get("node_id"),
        "n_candidates": len(candidates),
        "selection_table": selection_table,
        "sanity": {
            "n_rows_expected": int(len(holdout_df)),
            "n_rows_got": int(n),
            "constant_pred": bool(pd.Series(preds).nunique() <= 1),
            "nan_count": int(pd.Series(preds).isna().sum()),
        },
        "leakage_flag": bool(value == 0 or (direction == "higher_is_better" and value >= 0.999)),
        "notes": ["Winner selected on the validation (select) set; hold-out used once for this metric."],
    }
    json.dump(out, open(os.path.join(ws, "eval", "holdout_metrics.json"), "w"), indent=2, default=str)
    pd.DataFrame({target: y_true, f"{target}_pred": preds}).to_csv(
        os.path.join(ws, "eval", "predictions_holdout.csv"), index=False
    )
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
