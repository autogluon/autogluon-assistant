#!/usr/bin/env python3
"""Trivial baseline arm: predict the train target's mean (regression/forecast) or majority
class (classification) for every hold-out row. Deterministic; a floor every real arm must beat.

    python trivial_baseline.py --train train.csv --features holdout/features.csv --target COL \
        --task-type regression --out predictions.csv
"""

import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--task-type", default="regression")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tr = pd.read_csv(a.train)
    feats = pd.read_csv(a.features)
    y = tr[a.target]
    if a.task_type in ("classification", "binary", "multiclass"):
        pred = y.mode().iloc[0]
    else:  # regression / forecast
        pred = float(y.mean())
    pd.DataFrame({"prediction": [pred] * len(feats)}).to_csv(a.out, index=False)
    print(f"trivial_baseline: wrote {len(feats)} constant predictions ({pred}) to {a.out}")


if __name__ == "__main__":
    main()
