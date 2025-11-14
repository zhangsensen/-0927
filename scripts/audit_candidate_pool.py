#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit the candidate pool (Top2000) and the selected set (Top500).
- Analyze factor overlap, frequency/industry distribution, duplicates/concentration
- Optionally evaluate realized performance using the full backtest CSV if available
- Emit a compact Markdown and JSON report under run_dir/selection/audit/

Usage:
  python scripts/audit_candidate_pool.py --run-dir results/run_YYYYMMDD_HHMMSS [--backtest-csv PATH]

Outputs:
  {run_dir}/selection/audit/
    - audit_report.md
    - audit_report.json
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    # Low-memory=False to avoid dtype guessing pitfalls on wide tables
    return pd.read_csv(path, low_memory=False)


def _ensure_outdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _detect_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def _parse_factor_list(val: Any) -> List[str]:
    if isinstance(val, list):
        return [str(x) for x in val]
    if pd.isna(val):
        return []
    s = str(val).strip()
    # Try JSON-like list
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            parsed = json.loads(s.replace("(", "[").replace(")", "]"))
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass
    # Fallback split
    if "|" in s:
        return [x.strip() for x in s.split("|") if x.strip()]
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    if s:
        return [s]
    return []


def _compute_hhi(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        return float("nan")
    w = w / w.sum()
    return float(np.sum(w ** 2))


def _sample_pairwise_jaccard(factors_series: pd.Series, samples: int = 50000, random_state: int = 42) -> Dict[str, float]:
    rng = np.random.default_rng(random_state)
    lists = [set(_parse_factor_list(x)) for x in factors_series.tolist()]
    n = len(lists)
    if n < 2:
        return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan")}
    m = min(samples, n * (n - 1) // 2)
    idx = rng.integers(0, n, size=(m, 2))
    # ensure different indices
    mask = idx[:, 0] != idx[:, 1]
    idx = idx[mask]
    j_scores = []
    for i, j in idx:
        a, b = lists[i], lists[j]
        if not a and not b:
            j_scores.append(1.0)
            continue
        inter = len(a & b)
        union = len(a | b)
        j = inter / union if union else 0.0
        j_scores.append(j)
    if not j_scores:
        return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan")}
    arr = np.asarray(j_scores)
    return {"mean": float(arr.mean()), "p50": float(np.median(arr)), "p90": float(np.quantile(arr, 0.9))}


def _realized_col(df: pd.DataFrame) -> Optional[str]:
    return _detect_col(df.columns.tolist(), [
        "realized_sharpe", "sharpe", "oos_sharpe", "backtest_sharpe",
        "realized_sharpe_ratio", "sr",
    ])


def _id_col(df: pd.DataFrame) -> Optional[str]:
    return _detect_col(df.columns.tolist(), ["combo_id", "id", "combo_key", "key"])


def _factor_col(df: pd.DataFrame) -> Optional[str]:
    return _detect_col(df.columns.tolist(), ["factors", "factor_list", "factor_names", "features"])


def _freq_col(df: pd.DataFrame) -> Optional[str]:
    return _detect_col(df.columns.tolist(), ["freq", "frequency", "bar", "tf"])


def _industry_col(df: pd.DataFrame) -> Optional[str]:
    return _detect_col(df.columns.tolist(), ["industry", "sector", "universe", "category"])


def _pred_col(df: pd.DataFrame) -> Optional[str]:
    return _detect_col(df.columns.tolist(), ["calibrated_sharpe_pred", "calibrated_sharpe_full", "predicted_sharpe", "score"])


def _stability_col(df: pd.DataFrame) -> Optional[str]:
    return _detect_col(df.columns.tolist(), ["stability_score", "stability", "robustness"])


def _dedup_key(df: pd.DataFrame) -> pd.Series:
    # Construct a dedup key from most informative columns available
    cols = []
    for getter in (_id_col, _factor_col, _freq_col, _industry_col):
        c = getter(df)
        if c:
            cols.append(c)
    if not cols:
        cols = df.columns.tolist()[:5]
    # Join row-wise with '|'
    return df[cols].astype(str).agg("|".join, axis=1).rename("dedup_key")


def _load_backtest_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p, low_memory=False)
    except Exception:
        return None


def audit(run_dir: Path, backtest_csv_override: Optional[str] = None) -> Tuple[Path, Dict[str, Any]]:
    sel_dir = run_dir / "selection"
    cand_path = sel_dir / "candidate_top2000.csv"
    top500_path = sel_dir / "selected_top500.csv"
    out_dir = sel_dir / "audit"
    _ensure_outdir(out_dir)

    summary_path = sel_dir / "selection_summary.json"
    summary = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            summary = {}

    df_cand = _read_csv(cand_path)
    df_500 = _read_csv(top500_path)

    # Basic schema
    id_c = _id_col(df_cand)
    pred_c = _pred_col(df_cand)
    stab_c = _stability_col(df_cand)
    fac_c = _factor_col(df_cand)
    frq_c = _freq_col(df_cand)
    ind_c = _industry_col(df_cand)

    report: Dict[str, Any] = {
        "paths": {"run_dir": str(run_dir), "candidate": str(cand_path), "top500": str(top500_path)},
        "row_counts": {"candidate": int(len(df_cand)), "top500": int(len(df_500))},
        "columns": {
            "id": id_c, "pred": pred_c, "stability": stab_c,
            "factors": fac_c, "frequency": frq_c, "industry": ind_c,
        },
    }

    # Duplicates
    dedup_key_cand = _dedup_key(df_cand)
    dup_counts = dedup_key_cand.value_counts()
    n_dups = int((dup_counts > 1).sum())
    report["duplicates"] = {
        "duplicate_keys": n_dups,
        "max_dup_count": int(dup_counts.max()) if not dup_counts.empty else 0,
        "top_examples": dup_counts.head(10).astype(int).to_dict(),
    }

    # Distributions
    dist: Dict[str, Any] = {}
    if frq_c:
        dist["frequency"] = df_cand[frq_c].value_counts().head(20).astype(int).to_dict()
    if ind_c:
        dist["industry"] = df_cand[ind_c].value_counts().head(20).astype(int).to_dict()
    report["distributions"] = dist

    # Factor coverage & overlap (approximate)
    if fac_c:
        # prevalence
        fac_lists = df_cand[fac_c].map(_parse_factor_list)
        all_factors = [f for sub in fac_lists for f in sub]
        fac_counts = pd.Series(all_factors).value_counts()
        report["factor_prevalence_top"] = fac_counts.head(30).astype(int).to_dict()
        # overlap sample
        j_stats = _sample_pairwise_jaccard(df_cand[fac_c])
        report["pairwise_factor_overlap_jaccard"] = j_stats
    else:
        report["factor_prevalence_top"] = {}
        report["pairwise_factor_overlap_jaccard"] = {"mean": None, "p50": None, "p90": None}

    # Concentration via HHI over Top500 equal-weight combos
    weights = np.ones(len(df_500), dtype=float)
    report["top500_hhi_combo_equal_weight"] = _compute_hhi(weights)

    # Realized metrics (if backtest csv available)
    backtest_csv = backtest_csv_override or summary.get("backtest_csv")
    df_bt = _load_backtest_csv(backtest_csv)
    realized_stats: Dict[str, Any] = {"available": False}
    if df_bt is not None:
        id_bt = _id_col(df_bt)
        rcol = _realized_col(df_bt)
        if id_bt and rcol:
            realized_stats["available"] = True
            realized_stats["backtest_csv"] = backtest_csv
            df_cand_k = df_cand.copy()
            if id_c is None:
                # try to align by index if no id
                df_cand_k = df_cand_k.reset_index().rename(columns={"index": "combo_id_fallback"})
                id_c_local = "combo_id_fallback"
            else:
                id_c_local = id_c
            merged = df_cand_k[[id_c_local] + ([pred_c] if pred_c else [])].merge(
                df_bt[[id_bt, rcol]], left_on=id_c_local, right_on=id_bt, how="inner"
            )
            # Rank-based metrics
            if pred_c in merged.columns:
                merged = merged.dropna(subset=[pred_c, rcol])
                merged = merged.sort_values(pred_c, ascending=False)
                realized = merged[rcol].to_numpy()
                predicted_rank = np.arange(1, len(merged)+1)
                # Spearman via rankcorr
                spearman = pd.Series(merged[pred_c]).rank(ascending=False).corr(pd.Series(merged[rcol]).rank())
                pearson = float(np.corrcoef(merged[pred_c].to_numpy(), realized)[0,1])
                realized_stats["pearson"] = float(pearson)
                realized_stats["spearman"] = float(spearman)
                # Precision@K
                def precision_at_k(k: int) -> float:
                    top_pred_ids = merged.head(k)[id_bt].tolist()
                    top_real_ids = merged.nlargest(k, rcol)[id_bt].tolist()
                    inter = len(set(top_pred_ids) & set(top_real_ids))
                    return inter / float(k)
                realized_stats["precision_at"] = {k: precision_at_k(k) for k in [50,100,200,500,1000,2000] if k <= len(merged)}
                # Summary stats on realized sharpe for selected_top500 if possible
                if id_c:
                    top500_ids = set(df_500[id_c].astype(str).tolist())
                    sel = merged[merged[id_bt].astype(str).isin(top500_ids)][rcol]
                    realized_stats["selected_top500_realized_sharpe_summary"] = {
                        "count": int(sel.shape[0]),
                        "mean": float(sel.mean()),
                        "median": float(sel.median()),
                        "p25": float(sel.quantile(0.25)),
                        "p75": float(sel.quantile(0.75)),
                    }
        else:
            realized_stats["note"] = "Backtest CSV missing id or realized sharpe column; skip realized metrics."
    report["realized_metrics"] = realized_stats

    # Persist outputs
    md_lines = []
    md_lines.append(f"# Candidate Pool Audit\n")
    md_lines.append(f"Run dir: `{run_dir}`\n")
    md_lines.append(f"- Candidate rows: {report['row_counts']['candidate']}\n")
    md_lines.append(f"- Top500 rows: {report['row_counts']['top500']}\n")
    md_lines.append(f"- Columns detected: {json.dumps(report['columns'], ensure_ascii=False)}\n")
    md_lines.append(f"\n## Duplicates\n")
    md_lines.append(f"- Duplicate keys: {report['duplicates']['duplicate_keys']} (max dup count: {report['duplicates']['max_dup_count']})\n")
    if report['duplicates']['top_examples']:
        md_lines.append("Top duplicate keys (examples):\n")
        for k, v in list(report['duplicates']['top_examples'].items())[:10]:
            md_lines.append(f"- `{k}`: {v}\n")
    md_lines.append("\n## Distributions\n")
    for name, dist in report["distributions"].items():
        md_lines.append(f"### {name}\n")
        for k, v in dist.items():
            md_lines.append(f"- {k}: {v}\n")
    md_lines.append("\n## Factor coverage & overlap\n")
    if report["factor_prevalence_top"]:
        md_lines.append("Top factors by prevalence:\n")
        for k, v in list(report["factor_prevalence_top"].items())[:30]:
            md_lines.append(f"- {k}: {v}\n")
        j = report["pairwise_factor_overlap_jaccard"]
        md_lines.append(f"\nApprox pairwise Jaccard overlap — mean: {j['mean']:.4f}, p50: {j['p50']:.4f}, p90: {j['p90']:.4f}\n")
    else:
        md_lines.append("Factor list column not found; skipped.\n")
    md_lines.append("\n## Concentration\n")
    md_lines.append(f"Top500 combo equal-weight HHI: {report['top500_hhi_combo_equal_weight']:.6f} (lower is better; ~0.002 is perfectly equal-weight)\n")

    md_lines.append("\n## Realized metrics (if available)\n")
    if report["realized_metrics"].get("available"):
        rm = report["realized_metrics"]
        md_lines.append(f"- Pearson: {rm.get('pearson'):.4f}\n")
        md_lines.append(f"- Spearman: {rm.get('spearman'):.4f}\n")
        if "precision_at" in rm:
            md_lines.append("- Precision@K:\n")
            for k, v in rm["precision_at"].items():
                md_lines.append(f"  - @{k}: {v:.3f}\n")
        if "selected_top500_realized_sharpe_summary" in rm:
            s = rm["selected_top500_realized_sharpe_summary"]
            md_lines.append("- Selected Top500 realized Sharpe summary:\n")
            md_lines.append(f"  - count: {s['count']} | mean: {s['mean']:.4f} | median: {s['median']:.4f} | p25: {s['p25']:.4f} | p75: {s['p75']:.4f}\n")
    else:
        md_lines.append("Backtest CSV not available or missing required columns.\n")

    (out_dir / "audit_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (out_dir / "audit_report.md").write_text("".join(md_lines))

    return out_dir, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Path to a results/run_*/ directory")
    ap.add_argument("--backtest-csv", default=None, help="Optional path to full backtest CSV; overrides selection_summary.json")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir}")
    out_dir, _ = audit(run_dir, args.backtest_csv)
    print(f"✅ Audit completed. Report at: {out_dir}")


if __name__ == "__main__":
    main()
