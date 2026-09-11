#!/usr/bin/env python3
"""Single-shot fixed scorer — the ONE metric-of-record used everywhere.

Compares a predictions CSV to a ground-truth CSV (row-aligned) under one metric and prints exactly
one line ``MLZERO_SCORE: <float>`` in the higher-is-better convention (lower-is-better metrics are
negated). Reuses evaluate.py's SCORERS registry so the per-node MCTS reward, the final hold-out
metric, and the benchmark leaderboard all use identical scoring.

Contract (matches NodeManager._run_external_scorer):
    python score_one.py --ground-truth GT.csv --predictions PRED.csv [--metric M] [--target COL] [--train-target T.csv]
    -> stdout "MLZERO_SCORE: <float>"; exit 0 on success, non-zero (no score line) on any problem
       (row mismatch / non-finite / bad input) so the caller treats the node as unvalidated.
"""

import argparse
import math
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import score  # noqa: E402  (SCORERS registry + metric direction)


def _col(df, name):
    """The named column if present, else the last column (predictions/GT fallback)."""
    if name and name in df.columns:
        return df[name]
    return df.iloc[:, -1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--metric", default="rmse")
    ap.add_argument("--target", default=None)
    ap.add_argument("--train-target", default=None, help="optional training-target CSV, for MASE's scale")
    a = ap.parse_args()

    try:
        gt = pd.read_csv(a.ground_truth)
        pr = pd.read_csv(a.predictions)
    except Exception as e:  # noqa: BLE001
        print(f"score_one: could not read inputs: {e}", file=sys.stderr)
        sys.exit(2)

    y = _col(gt, a.target).reset_index(drop=True)
    p = _col(pr, "prediction").reset_index(drop=True)
    if len(y) == 0 or len(y) != len(p):
        print(f"score_one: row mismatch gt={len(y)} pred={len(p)}", file=sys.stderr)
        sys.exit(2)

    train_target = None
    if a.train_target and os.path.isfile(a.train_target):
        train_target = pd.read_csv(a.train_target).iloc[:, -1]

    try:
        value, direction = score(a.metric, y, p, train_target=train_target)
    except Exception as e:  # noqa: BLE001
        print(f"score_one: metric '{a.metric}' failed: {e}", file=sys.stderr)
        sys.exit(2)

    if value is None or not math.isfinite(float(value)):
        print(f"score_one: non-finite score ({value})", file=sys.stderr)
        sys.exit(2)

    higher_is_better = float(value) if direction == "higher_is_better" else -float(value)
    print(f"MLZERO_SCORE: {higher_is_better}")


if __name__ == "__main__":
    main()
