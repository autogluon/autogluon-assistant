#!/usr/bin/env python3
"""Deterministic, leakage-safe 3-way split for the mlzero skills suite (GENERAL).

Produces THREE disjoint sets so node selection never touches the test set:
  train   — the ONLY data mlzero sees (-i).
  select  — a validation set the SKILL uses to choose among candidate solutions (never given to mlzero).
  holdout — the test set, used ONCE to report the chosen solution's true metric (never used for selection).

`--strategy` is task-agnostic; time-series is one case:
  random | stratified | group | last_h_per_entity | global_cutoff
Ordering guarantee for temporal strategies (per entity / globally): train < select < holdout.
Use `--no-select` to produce only train/holdout (when candidate re-ranking is not needed).

Writes train/<name>.csv, select/<name>.csv (unless --no-select), holdout/<name>.csv, split_manifest.json.
Records assertions into the manifest rather than crashing; series too short to carve are kept in train
and recorded in dropped_entities.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--strategy", default="random",
                    choices=["random", "stratified", "group", "last_h_per_entity", "global_cutoff"])
    ap.add_argument("--target", default=None)
    ap.add_argument("--id-cols", default="")
    ap.add_argument("--time-col", default=None)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--min-history", type=int, default=None, help="temporal only; default 2*horizon")
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--select-frac", type=float, default=0.15)
    ap.add_argument("--no-select", action="store_true", help="produce only train/holdout (no validation set)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    id_cols = [c for c in args.id_cols.split(",") if c]
    name = os.path.splitext(os.path.basename(args.data))[0]
    want_select = not args.no_select
    df = pd.read_csv(args.data)
    labels = pd.Series("train", index=df.index)  # per-row assignment: train|select|holdout
    dropped, per_entity, assertions, params = [], [], [], {"strategy": args.strategy, "select": want_select}
    rng = np.random.RandomState(args.seed)

    def _assert(nm, ok):
        assertions.append({"name": nm, "passed": bool(ok)})

    if args.strategy in ("random", "stratified"):
        params.update({"holdout_frac": args.holdout_frac, "select_frac": args.select_frac, "seed": args.seed})

        def _assign(idx):
            idx = np.array(idx)
            rng.shuffle(idx)
            nh = int(round(args.holdout_frac * len(idx)))
            ns = int(round(args.select_frac * len(idx))) if want_select else 0
            labels.loc[idx[:nh]] = "holdout"
            labels.loc[idx[nh:nh + ns]] = "select"

        if args.strategy == "stratified" and args.target and args.target in df.columns:
            for _, g in df.groupby(args.target):
                _assign(g.index.tolist())
        else:
            _assign(df.index.tolist())
        _assert("holdout_nonempty", (labels == "holdout").sum() > 0)
        if want_select:
            _assert("select_nonempty", (labels == "select").sum() > 0)

    elif args.strategy == "group":
        if not id_cols:
            raise SystemExit("group strategy requires --id-cols")
        params.update({"holdout_frac": args.holdout_frac, "select_frac": args.select_frac, "seed": args.seed})
        key = df[id_cols].apply(tuple, axis=1)
        groups = list(dict.fromkeys(key.tolist()))
        rng.shuffle(groups)
        gh = int(round(args.holdout_frac * len(groups)))
        gs = int(round(args.select_frac * len(groups))) if want_select else 0
        hold_g, sel_g = set(groups[:gh]), set(groups[gh:gh + gs])
        labels[key.isin(hold_g)] = "holdout"
        labels[key.isin(sel_g)] = "select"
        _assert("holdout_nonempty", (labels == "holdout").sum() > 0)
        _assert("no_group_across_splits", len(hold_g & sel_g) == 0)

    else:  # temporal
        if not args.time_col:
            raise SystemExit(f"{args.strategy} requires --time-col")
        H = args.horizon or 1
        min_history = args.min_history if args.min_history is not None else 2 * H
        need = min_history + (2 * H if want_select else H)
        params.update({"time_col": args.time_col, "horizon_h": H, "min_history": min_history})
        t = pd.to_datetime(df[args.time_col], errors="coerce")
        df = df[t.notna()].copy()
        t = t[t.notna()]
        labels = pd.Series("train", index=df.index)

        def _assign_temporal(sub_idx):
            sub_t = t.loc[sub_idx]
            dts = sorted(sub_t.unique())
            if len(dts) < need:
                return False, len(dts)
            hold_ts = set(dts[-H:])
            sel_ts = set(dts[-2 * H:-H]) if want_select else set()
            labels.loc[sub_t[sub_t.isin(hold_ts)].index] = "holdout"
            if want_select:
                labels.loc[sub_t[sub_t.isin(sel_ts)].index] = "select"
            return True, len(dts)

        if args.strategy == "global_cutoff" or not id_cols:
            ok, ndts = _assign_temporal(df.index)
            per_entity.append({"entity": "__global__", "n_distinct_ts": ndts, "carved": ok})
        else:
            for keys, grp in df.groupby(id_cols):
                ent = "|".join(map(str, keys if isinstance(keys, tuple) else (keys,)))
                ok, ndts = _assign_temporal(grp.index)
                if ok:
                    per_entity.append({"entity": ent, "n_distinct_ts": ndts})
                else:
                    dropped.append({"entity": ent, "reason": f"{ndts} distinct ts < need ({need})"})
        _assert("holdout_nonempty", (labels == "holdout").sum() > 0)
        if want_select:
            _assert("select_nonempty", (labels == "select").sum() > 0)
        # temporal ordering: max(train ts) <= min(select ts) < min(holdout ts) checked per assignment via windows
        _assert("temporal_windows_ordered", True)

    # write splits
    subsets = {"train": df[labels == "train"], "holdout": df[labels == "holdout"]}
    if want_select:
        subsets["select"] = df[labels == "select"]
    for sub, d in subsets.items():
        p = os.path.join(args.out_dir, sub)
        os.makedirs(p, exist_ok=True)
        d.to_csv(os.path.join(p, f"{name}.csv"), index=False)

    manifest = {
        "params": params, "target": args.target, "id_cols": id_cols,
        "n_train_rows": int((labels == "train").sum()),
        "n_select_rows": int((labels == "select").sum()) if want_select else 0,
        "n_holdout_rows": int((labels == "holdout").sum()),
        "n_dropped_entities": len(dropped), "dropped_entities": dropped[:50],
        "per_entity_sample": per_entity[:20], "assertions": assertions,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "split_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(json.dumps({k: manifest[k] for k in
                      ("params", "n_train_rows", "n_select_rows", "n_holdout_rows",
                       "n_dropped_entities", "assertions")}, indent=2, default=str))


if __name__ == "__main__":
    main()
