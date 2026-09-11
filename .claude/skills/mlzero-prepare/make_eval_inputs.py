#!/usr/bin/env python3
"""Derive the per-node scorer inputs from the frozen 'select' (validation) split.

mlzero gives each node the FEATURES file to predict on (target removed), and the fixed scorer reads
the GROUND-TRUTH file. Row order is preserved so predictions align to ground truth by position.

    python make_eval_inputs.py --select-csv <ws>/data/select/<name>.csv --target <col> --out-dir <ws>/eval
    -> <out-dir>/validation_features.csv   (select minus target; handed to mlzero via --eval-validation-features)
    -> <out-dir>/validation_ground_truth.csv (the target column; handed to mlzero via --eval-ground-truth)
"""

import argparse
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select-csv", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    df = pd.read_csv(a.select_csv)
    if a.target not in df.columns:
        raise SystemExit(f"target '{a.target}' not in {a.select_csv} columns {list(df.columns)}")
    os.makedirs(a.out_dir, exist_ok=True)
    feats = os.path.join(a.out_dir, "validation_features.csv")
    gt = os.path.join(a.out_dir, "validation_ground_truth.csv")
    df.drop(columns=[a.target]).to_csv(feats, index=False)
    df[[a.target]].to_csv(gt, index=False)
    print(f"wrote {feats} ({df.shape[0]} rows) and {gt}")


if __name__ == "__main__":
    main()
