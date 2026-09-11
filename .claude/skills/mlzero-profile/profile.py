#!/usr/bin/env python3
"""Deterministic data profiler for the mlzero skills suite.

Reads a CSV (samples if huge) and emits profile.json: per-column stats + inferred
time column / entity id columns / target candidates / freq / problem type, plus a
recommended split policy. Inference is heuristic — `solve`/clarify confirms it.

Usage:
    python profile.py --data <csv|folder> --out profile.json [--sample-rows 200000]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

DATEY = ("date", "time", "timestamp", "day", "dt", "period")


def _pick_csv(path):
    if os.path.isdir(path):
        csvs = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(".csv")]
        if not csvs:
            sys.exit(f"no CSV found in {path}")
        return max(csvs, key=os.path.getsize)
    return path


def _looks_datetime(name, s):
    """Try to parse a column as datetimes; return parsed series if >90% parse, else None."""
    if s.dtype.kind in "mM":
        return pd.to_datetime(s, errors="coerce")
    # Only attempt on strings, or numerics whose NAME looks temporal (avoid parsing zip codes as epochs)
    namey = any(k in name.lower() for k in DATEY)
    if s.dtype == object or namey:
        sample = s.dropna().astype(str).head(500)
        parsed = pd.to_datetime(sample, errors="coerce")
        if len(sample) and parsed.notna().mean() > 0.9:
            return pd.to_datetime(s, errors="coerce")
    return None


def _infer_freq(ts):
    d = pd.Series(sorted(pd.Series(ts).dropna().unique()))
    if len(d) < 3:
        return None
    deltas = d.diff().dropna()
    med = deltas.median()
    days = med / np.timedelta64(1, "D")
    for n, label in [(1, "D"), (7, "W"), (28, "M"), (30, "M"), (31, "M"), (365, "Y"), (366, "Y")]:
        if abs(days - n) <= max(1, 0.15 * n):
            return label
    if days < 1:
        return "H"
    return f"~{days:.0f}D"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample-rows", type=int, default=200000)
    args = ap.parse_args()

    path = _pick_csv(args.data)
    total = sum(1 for _ in open(path)) - 1
    if total > args.sample_rows:
        df = pd.read_csv(path, nrows=args.sample_rows)
        sampled = True
    else:
        df = pd.read_csv(path)
        sampled = False

    n = len(df)
    time_col, parsed_time = None, None
    columns = []
    for c in df.columns:
        s = df[c]
        pt = _looks_datetime(c, s)
        is_dt = pt is not None
        if is_dt and time_col is None:
            time_col, parsed_time = c, pt
        nun = int(s.nunique(dropna=True))
        columns.append(
            {
                "name": c,
                "dtype": str(s.dtype),
                "n_missing": int(s.isna().sum()),
                "n_unique": nun,
                "examples": [str(x) for x in s.dropna().unique()[:3]],
                "is_datetime": bool(is_dt),
                "is_id_candidate": bool((s.dtype == object) or (nun > 0.5 * n and s.dtype.kind in "iu")),
                "is_target_candidate": bool(s.dtype.kind == "f" or (s.dtype.kind in "iu" and nun > 10)),
                "leakage_risk": False,  # set below (numeric, non-id, non-time → same-timestamp covariate)
            }
        )

    # entity id columns = id candidates that are not the time column
    id_cols = [c["name"] for c in columns if c["is_id_candidate"] and c["name"] != time_col]
    # target candidates = numeric, not an id, not the time col
    target_cands = [
        c["name"] for c in columns if c["is_target_candidate"] and c["name"] not in id_cols and c["name"] != time_col
    ]
    # leakage-risk = numeric covariates (candidate targets we did NOT pick) measured same-timestamp
    for c in columns:
        if c["name"] != time_col and c["name"] not in id_cols and c["dtype"].startswith(("float", "int")):
            c["leakage_risk"] = True

    # panel structure
    ts_per_entity = None
    problem_type = "tabular"
    split_reco = "random"
    if time_col and id_cols:
        df["_t"] = parsed_time
        g = df.groupby(id_cols)["_t"].nunique()
        ts_per_entity = {"entities": int(len(g)), "min": int(g.min()), "median": int(g.median()), "max": int(g.max())}
        if g.median() >= 3:
            problem_type, split_reco = "timeseries-forecast (panel)", "last_h_per_entity"
        else:
            problem_type, split_reco = "cross-sectional / global-temporal", "global_cutoff"
    elif time_col:
        problem_type, split_reco = "temporal (single/few series)", "global_cutoff"

    profile = {
        "data_path": os.path.abspath(path),
        "n_rows": total,
        "n_cols": len(df.columns),
        "sampled_for_profiling": sampled,
        "columns": columns,
        "inferred": {
            "time_column": time_col,
            "entity_id_columns": id_cols,
            "target_candidates": target_cands,
            "freq": _infer_freq(parsed_time) if parsed_time is not None else None,
            "problem_type": problem_type,
            "split_policy_recommendation": split_reco,
            "timestamps_per_entity": ts_per_entity,
        },
        "notes": [
            "Inference is heuristic; confirm target/metric/split in clarify.",
            "leakage_risk=True flags same-timestamp numeric covariates (unknown at forecast time → drop unless known-in-advance).",
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(profile, f, indent=2, default=str)
    print(json.dumps(profile["inferred"], indent=2, default=str))


if __name__ == "__main__":
    main()
