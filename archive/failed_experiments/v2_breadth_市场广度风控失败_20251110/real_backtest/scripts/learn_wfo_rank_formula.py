#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Learn a simple, data-driven WFO ranking formula that best correlates with realized performance.

Inputs:
    - WFO run dir: results/run_<ts>/ (expects all_combos.parquet)
    - Real backtest csv (single/reco frequency): results_combo_wfo/<ts>/top*_full.csv  [for targets: sharpe|annual]
    - All-frequency scan csv: results_combo_wfo/<ts>/all_freq_scan_*.csv           [for targets: robust_*|reco_*]
    - Robust ranking csv (optional): results_combo_wfo/<ts>/robust_ranking_*.csv    [preferred if available]

Outputs:
    - JSON with learned coefficients and CV metrics: results/run_<ts>/learned_wfo_rank_formula_<ts>.json
    - CSV with combos ranked by the learned score: results/run_<ts>/wfo_learned_ranking_<ts>.csv

Method:
    - Join WFO features with target labels by 'combo'
    - Feature set (if present): mean_oos_ic, stability_score, combo_size, best_rebalance_freq
            * Derived: is_freq_21, is_freq_in_band(6-13,21), dist_to_21
            * Interaction: mean_oos_ic*stability_score
    - Standardize features (z-score) and solve least squares for target y
    - Use 5-fold CV (random split) to report Spearman correlation between predicted score and target y

Targets supported via --target:
    - sharpe | annual                      -> from single-frequency RB csv (top*_full)
    - robust_score                         -> from robust_ranking csv, or computed from all_freq_scan with default weights
    - robust_composite                     -> same as robust_score but ignoring turnover term
    - reco_sharpe | reco_annual            -> realized Sharpe/annual at recommended frequency (from robust ranking or computed)

Usage examples:
    python learn_wfo_rank_formula.py --run_dir results/run_20251106_184657 \
        --rb_csv results_combo_wfo/20251106_184657/top100_backtest_by_ic_20251106_184657_full.csv --target sharpe

    python learn_wfo_rank_formula.py --run_dir results/run_20251106_184657 \
        --allfreq_csv results_combo_wfo/20251106_184657/all_freq_scan_20251106_184657.csv --target robust_score
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import os

PREF_FREQ_SET = set([6,7,8,9,10,11,12,13,21])

# Default robust weights (aligned with robust_rank_from_allfreq.py)
W_MEAN = float(os.environ.get('ROBUST_W_MEAN', 0.4))
W_P20  = float(os.environ.get('ROBUST_W_P20',  0.6))
W_STD  = float(os.environ.get('ROBUST_W_STD',  0.3))
W_DD   = float(os.environ.get('ROBUST_W_DD',   0.1))
W_TURN = float(os.environ.get('ROBUST_W_TURN', 0.0))
PENALTY_DD_REF = float(os.environ.get('ROBUST_DD_REF', 0.25))
GOOD_SHARPE_TH = float(os.environ.get('ROBUST_GOOD_SHARPE', 0.5))


def zscore(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if sd <= 1e-12:
        return np.zeros_like(x)
    return (x - mu) / sd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    # raw
    for col in ["mean_oos_ic", "stability_score", "combo_size", "best_rebalance_freq"]:
        if col in df.columns:
            out[col] = df[col]
    # derived
    if "best_rebalance_freq" in out.columns:
        fr = out["best_rebalance_freq"].astype(float).fillna(-1)
        out["is_freq_21"] = (fr == 21).astype(float)
        out["is_freq_pref"] = fr.isin(list(PREF_FREQ_SET)).astype(float)
        out["dist_to_21"] = np.abs(fr - 21)
    else:
        out["is_freq_21"] = 0.0
        out["is_freq_pref"] = 0.0
        out["dist_to_21"] = 0.0
    # interactions
    if set(["mean_oos_ic", "stability_score"]).issubset(out.columns):
        out["ic_x_stab"] = out["mean_oos_ic"] * out["stability_score"]
    else:
        out["ic_x_stab"] = 0.0
    # fillna
    out = out.fillna(0.0)
    return out


def _ensure_test_freq_col(df: pd.DataFrame) -> pd.DataFrame:
    if 'test_freq' in df.columns:
        return df
    if 'freq' in df.columns:
        return df.rename(columns={'freq': 'test_freq'})
    raise ValueError('Neither test_freq nor freq column found in all-frequency csv')


def _robust_aggregate_from_allfreq(df: pd.DataFrame,
                                   w_mean=W_MEAN, w_p20=W_P20, w_std=W_STD, w_dd=W_DD, w_turn=W_TURN,
                                   dd_ref=PENALTY_DD_REF) -> pd.DataFrame:
    """Compute robust aggregates and recommended_freq from an all-frequency scan DataFrame.
    Returns columns: [combo, robust_score, sharpe_mean, sharpe_std, sharpe_p20, sharpe_p50,
                      best_sharpe, best_freq_sharpe, sharpe_21, n_freq_ge_0_5,
                      annual_mean, annual_max, max_dd_mean, turnover_mean, recommended_freq]
    """
    required = ['combo', 'sharpe', 'annual_ret', 'max_dd', 'test_freq']
    for c in required:
        if c not in df.columns:
            raise ValueError(f'all-frequency csv missing column: {c}')
    if 'avg_turnover' not in df.columns:
        df = df.copy(); df['avg_turnover'] = 0.0

    rows = []
    for combo, g in df.groupby('combo'):
        g = g.copy()
        sharpe_mean = g['sharpe'].mean()
        sharpe_std  = g['sharpe'].std(ddof=0)
        sharpe_p20  = g['sharpe'].quantile(0.2)
        sharpe_p50  = g['sharpe'].quantile(0.5)
        ann_mean    = g['annual_ret'].mean()
        ann_max     = g['annual_ret'].max()
        dd_mean     = g['max_dd'].mean()
        turn_mean   = g['avg_turnover'].mean()

        idx_best = g['sharpe'].idxmax()
        best_sharpe = float(g.loc[idx_best, 'sharpe'])
        best_freq   = int(g.loc[idx_best, 'test_freq']) if pd.notna(g.loc[idx_best, 'test_freq']) else None

        n_good = int((g['sharpe'] >= GOOD_SHARPE_TH).sum())
        dd_pen = max(0.0, abs(dd_mean) - dd_ref)

        score = (
            w_mean * sharpe_mean +
            w_p20  * sharpe_p20  -
            w_std  * (sharpe_std if pd.notna(sharpe_std) else 0.0) -
            w_dd   * dd_pen -
            w_turn * turn_mean
        )

        # recommend freq: prefer 21 if not worse than median; else best in preferred set; else global best
        reco_freq = best_freq
        sharpe_21 = None
        if 21 in set(g['test_freq'].dropna().astype(int).tolist()):
            sharpe_21 = float(g.loc[g['test_freq']==21, 'sharpe'].iloc[0])
            if sharpe_21 >= sharpe_p50:
                reco_freq = 21
        if reco_freq != 21:
            gp = g[g['test_freq'].isin(PREF_FREQ_SET)]
            if len(gp) > 0:
                idxp = gp['sharpe'].idxmax()
                reco_freq = int(gp.loc[idxp, 'test_freq'])

        rows.append({
            'combo': combo,
            'robust_score': score,
            'sharpe_mean': sharpe_mean,
            'sharpe_std': sharpe_std,
            'sharpe_p20': sharpe_p20,
            'sharpe_p50': sharpe_p50,
            'best_sharpe': best_sharpe,
            'best_freq_sharpe': best_freq,
            'sharpe_21': sharpe_21,
            'n_freq_ge_0_5': n_good,
            'annual_mean': ann_mean,
            'annual_max': ann_max,
            'max_dd_mean': dd_mean,
            'turnover_mean': turn_mean,
            'recommended_freq': reco_freq,
        })
    return pd.DataFrame(rows)


def fit_least_squares(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Add bias term
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    coef, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return coef  # includes bias as last element


def predict_with_coef(X: np.ndarray, coef: np.ndarray) -> np.ndarray:
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    return Xb @ coef


def kfold_spearman(X: np.ndarray, y: np.ndarray, k=5, seed=42) -> dict:
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    cors = []
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        Xtr, ytr = X[train_idx], y[train_idx]
        Xte, yte = X[test_idx], y[test_idx]
        coef = fit_least_squares(Xtr, ytr)
        pred = predict_with_coef(Xte, coef)
        r, p = spearmanr(pred, yte)
        cors.append((float(r), float(p)))
    r_mean = float(np.nanmean([c[0] for c in cors]))
    p_mean = float(np.nanmean([c[1] for c in cors]))
    return {"k": k, "mean_spearman": r_mean, "mean_p": p_mean, "folds": cors}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Path to results/run_<ts> directory")
    ap.add_argument("--rb_csv", required=False, help="Path to real backtest csv (top*_full.csv) for single/reco targets")
    ap.add_argument("--allfreq_csv", required=False, help="Path to all-frequency scan csv for robust/reco targets")
    ap.add_argument("--robust_csv", required=False, help="Path to robust ranking csv (if present, preferred)")
    ap.add_argument("--target", default="sharpe", choices=["sharpe", "annual", "robust_score", "robust_composite", "reco_sharpe", "reco_annual"], help="Which label to fit")
    ap.add_argument("--robust_w_mean", type=float, default=W_MEAN)
    ap.add_argument("--robust_w_p20", type=float, default=W_P20)
    ap.add_argument("--robust_w_std", type=float, default=W_STD)
    ap.add_argument("--robust_w_dd", type=float, default=W_DD)
    ap.add_argument("--robust_w_turn", type=float, default=W_TURN)
    ap.add_argument("--robust_dd_ref", type=float, default=PENALTY_DD_REF)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    assert run_dir.exists(), f"run_dir not found: {run_dir}"

    # load wfo combos
    wfo_all = pd.read_parquet(run_dir / "all_combos.parquet")
    wfo_all = wfo_all.reset_index(drop=True)
    # build target labels by target type
    target = args.target

    # normalize common column names
    if "combo" not in wfo_all.columns:
        # try 'name'
        if "name" in wfo_all.columns:
            wfo_all = wfo_all.rename(columns={"name": "combo"})
        else:
            raise ValueError("all_combos.parquet missing 'combo' column")

    label_df: pd.DataFrame
    if target in ("sharpe", "annual"):
        if not args.rb_csv:
            raise ValueError("--rb_csv is required for target sharpe/annual")
        rb_csv = Path(args.rb_csv)
        assert rb_csv.exists(), f"rb_csv not found: {rb_csv}"
        rb = pd.read_csv(rb_csv)
        if "combo" not in rb.columns:
            raise ValueError("rb_csv missing 'combo'")
        if "sharpe" not in rb.columns or "annual_ret" not in rb.columns:
            raise ValueError("rb_csv missing 'sharpe'/'annual_ret'")

        def pick_row(g: pd.DataFrame) -> pd.Series:
            if {"wfo_freq", "test_freq"}.issubset(g.columns):
                gg = g[g["wfo_freq"] == g["test_freq"]]
                if len(gg) > 0:
                    g = gg
            return g.sort_values("sharpe", ascending=False).iloc[0]

        rb_combo = rb.groupby("combo", as_index=False).apply(pick_row).reset_index(drop=True)
        label_df = rb_combo[["combo", "sharpe", "annual_ret"]].copy()
    else:
        # robust/reco based targets
        if not args.allfreq_csv and not args.robust_csv:
            raise ValueError("--allfreq_csv or --robust_csv is required for robust/reco targets")
        robust_df = None
        if args.robust_csv:
            rpath = Path(args.robust_csv)
            assert rpath.exists(), f"robust_csv not found: {rpath}"
            robust_df = pd.read_csv(rpath)
        af_df = None
        if args.allfreq_csv:
            apath = Path(args.allfreq_csv)
            assert apath.exists(), f"allfreq_csv not found: {apath}"
            af_df = pd.read_csv(apath)
            af_df = _ensure_test_freq_col(af_df)

        if robust_df is None:
            if af_df is None:
                raise ValueError("Provide either --robust_csv or --allfreq_csv")
            robust_df = _robust_aggregate_from_allfreq(
                af_df,
                w_mean=args.robust_w_mean, w_p20=args.robust_w_p20,
                w_std=args.robust_w_std, w_dd=args.robust_w_dd,
                w_turn=args.robust_w_turn, dd_ref=args.robust_dd_ref,
            )

        label_df = robust_df[["combo", "robust_score", "recommended_freq", "sharpe_mean", "sharpe_p20", "sharpe_std"]].copy()
        if target in ("reco_sharpe", "reco_annual"):
            if af_df is None:
                raise ValueError("--allfreq_csv is required for reco_sharpe/reco_annual")
            tmp = robust_df.merge(af_df[["combo", "test_freq", "sharpe", "annual_ret"]], left_on=["combo", "recommended_freq"], right_on=["combo", "test_freq"], how="left")
            col = "sharpe" if target == "reco_sharpe" else "annual_ret"
            label_df[col] = tmp[col]

    # inner join by combo for features + label
    if target in ("sharpe", "annual"):
        m = wfo_all.merge(label_df, on="combo", how="inner")
        y = m["sharpe" if target == "sharpe" else "annual_ret"].values.astype(float)
    elif target == "robust_score":
        m = wfo_all.merge(label_df[["combo", "robust_score"]], on="combo", how="inner")
        y = m["robust_score"].values.astype(float)
    elif target == "robust_composite":
        if not {"sharpe_mean", "sharpe_p20", "sharpe_std"}.issubset(label_df.columns):
            raise ValueError("robust_composite requires sharpe_mean/p20/std from robust/allfreq")
        composite = (
            args.robust_w_mean * label_df["sharpe_mean"].values +
            args.robust_w_p20  * label_df["sharpe_p20"].values -
            args.robust_w_std  * label_df["sharpe_std"].fillna(0.0).values
        )
        lab2 = pd.DataFrame({"combo": label_df["combo"].values, "y": composite})
        m = wfo_all.merge(lab2, on="combo", how="inner")
        y = m["y"].values.astype(float)
    elif target in ("reco_sharpe", "reco_annual"):
        col = "sharpe" if target == "reco_sharpe" else "annual_ret"
        if col not in label_df.columns:
            raise ValueError(f"{target} requires {col} from allfreq+robust")
        m = wfo_all.merge(label_df[["combo", col]], on="combo", how="inner")
        y = m[col].values.astype(float)
    else:
        raise ValueError(f"Unknown target: {target}")

    if len(m) < 50:
        raise RuntimeError(f"Too few matches between WFO and labels: {len(m)}")

    # features
    feats = build_features(m)

    # standardize
    X = np.column_stack([zscore(feats[col].values.astype(float)) for col in feats.columns])
    feat_names = list(feats.columns)

    # CV
    cv = kfold_spearman(X, y, k=5, seed=42)

    # Fit on all
    coef = fit_least_squares(X, y)
    pred = predict_with_coef(X, coef)
    r_full, p_full = spearmanr(pred, y)

    # pack results
    ts = run_dir.name.replace("run_", "")
    out_json = run_dir / f"learned_wfo_rank_formula_{ts}.json"
    out_csv = run_dir / f"wfo_learned_ranking_{ts}.csv"

    coef_map = {name: float(c) for name, c in zip(feat_names + ["bias"], coef)}
    result = {
        "target": args.target,
        "features": feat_names,
        "coef": coef_map,
        "cv": cv,
        "spearman_full": float(r_full),
        "spearman_full_p": float(p_full),
        "n_samples": int(len(y)),
        "note": "X are z-scored; score = sum(coef_i * z(feature_i)) + bias",
    }

    # save json
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # produce ranking on full WFO set (including combos without realized label)
    feats_all = build_features(wfo_all)
    X_all = np.column_stack([zscore(feats_all[col].values.astype(float)) for col in feats_all.columns])
    score_all = predict_with_coef(X_all, coef)
    rank_df = wfo_all.copy()
    rank_df["learned_score"] = score_all
    rank_df = rank_df.sort_values("learned_score", ascending=False)
    rank_df.to_csv(out_csv, index=False)

    # also print concise summary
    print(f"Saved learned formula: {out_json}")
    print(f"Saved learned ranking: {out_csv}")
    print(f"Spearman(full) vs {args.target}: r={r_full:.3f}, p={p_full:.3g}; CV mean r={cv['mean_spearman']:.3f}")
    print("Top 10 feature coefficients (abs):")
    for name, w in sorted(coef_map.items(), key=lambda x: abs(x[1]), reverse=True)[:10]:
        print(f"  {name:>20s}: {w:+.4f}")


if __name__ == "__main__":
    main()
