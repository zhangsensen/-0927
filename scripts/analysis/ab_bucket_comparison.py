#!/usr/bin/env python3
"""
A/B 对照: 跨桶约束 vs 无约束

读取全量 VEC + Rolling OOS + Holdout + Triple validation 结果,
按桶约束 post-filter 分成两组, 输出对照表.

用法:
  uv run python scripts/analysis/ab_bucket_comparison.py \
    --vec-dir results/vec_from_wfo_XXXXXXXX \
    --rolling-dir results/rolling_oos_consistency_XXXXXXXX \
    --holdout-dir results/holdout_validation_XXXXXXXX \
    --triple-dir results/final_triple_validation_XXXXXXXX
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from etf_strategy.core.factor_buckets import (
    FACTOR_BUCKETS,
    check_cross_bucket_constraint,
    get_bucket_coverage,
)


def parse_factor_combo(combo_str: str) -> list[str]:
    """Parse factor combo string like 'ADX_14D+OBV_SLOPE_10D+...' into list."""
    if isinstance(combo_str, str):
        return combo_str.split("+")
    return []


def tag_bucket_info(df: pd.DataFrame, combo_col: str = "combo") -> pd.DataFrame:
    """Add bucket constraint pass/fail and coverage info to DataFrame."""
    results = []
    for _, row in df.iterrows():
        factors = parse_factor_combo(row[combo_col])
        passes, reason = check_cross_bucket_constraint(
            factors, min_buckets=3, max_per_bucket=2
        )
        coverage = get_bucket_coverage(factors)
        results.append({
            "bucket_pass": passes,
            "n_buckets": len(coverage),
            "buckets": ",".join(sorted(coverage.keys())),
            "bucket_dist": "|".join(
                f"{k}={len(v)}" for k, v in sorted(coverage.items())
            ),
        })
    tag_df = pd.DataFrame(results, index=df.index)
    return pd.concat([df, tag_df], axis=1)


def compare_groups(df: pd.DataFrame, metric_cols: list[str], label: str):
    """Print A/B comparison table for given metrics."""
    group_a = df[~df["bucket_pass"]]  # fails constraint (unconstrained-only)
    group_b = df[df["bucket_pass"]]   # passes constraint

    print(f"\n{'='*80}")
    print(f"📊 {label}")
    print(f"{'='*80}")
    print(f"  A (unconstrained-only): {len(group_a)} combos")
    print(f"  B (cross-bucket pass):  {len(group_b)} combos")
    print(f"  A+B total:              {len(df)} combos")
    print()

    header = f"  {'Metric':<30} {'A(uncons-only)':>14} {'B(cross-bkt)':>14} {'Delta':>10} {'Winner':>8}"
    print(header)
    print(f"  {'-'*30} {'-'*14} {'-'*14} {'-'*10} {'-'*8}")

    for col in metric_cols:
        if col not in df.columns:
            continue
        a_vals = group_a[col].dropna()
        b_vals = group_b[col].dropna()
        if len(a_vals) == 0 or len(b_vals) == 0:
            continue

        a_med = a_vals.median()
        b_med = b_vals.median()
        delta = b_med - a_med
        # For MDD/drawdown columns, smaller (less negative) is better
        is_risk = any(kw in col.lower() for kw in ["mdd", "dd", "drawdown", "worst"])
        if is_risk:
            winner = "B" if b_med > a_med else "A" if a_med > b_med else "="
        else:
            winner = "B" if b_med > a_med else "A" if a_med > b_med else "="
        print(f"  {col:<30} {a_med:>14.4f} {b_med:>14.4f} {delta:>+10.4f} {winner:>8}")

    # Percentile analysis for key return metrics
    ret_cols = [c for c in metric_cols if "ret" in c.lower() or "return" in c.lower()]
    if ret_cols:
        print(f"\n  尾部分析 (首个收益指标: {ret_cols[0]}):")
        col = ret_cols[0]
        for pct_label, pct in [("10%分位", 0.10), ("25%分位", 0.25), ("75%分位", 0.75)]:
            a_p = group_a[col].dropna().quantile(pct)
            b_p = group_b[col].dropna().quantile(pct)
            print(f"    {pct_label}: A={a_p:+.4f}  B={b_p:+.4f}  delta={b_p-a_p:+.4f}")
        a_pos = (group_a[col].dropna() > 0).mean()
        b_pos = (group_b[col].dropna() > 0).mean()
        print(f"    正收益比例: A={a_pos:.1%}  B={b_pos:.1%}  delta={b_pos-a_pos:+.1%}")


def analyze_bucket_distribution(df: pd.DataFrame, label: str):
    """Analyze bucket coverage distribution."""
    print(f"\n{'='*80}")
    print(f"📊 桶覆盖分布 — {label}")
    print(f"{'='*80}")

    # Distribution of bucket count
    bucket_counts = df["n_buckets"].value_counts().sort_index()
    print(f"\n  桶数分布:")
    for n, count in bucket_counts.items():
        pct = count / len(df)
        print(f"    {n}桶: {count:>6} ({pct:.1%})")

    # Per-bucket selection frequency
    all_buckets = sorted(FACTOR_BUCKETS.keys())
    print(f"\n  各桶入选频率:")
    for bucket in all_buckets:
        in_bucket = df["buckets"].str.contains(bucket).sum()
        pct = in_bucket / len(df) if len(df) > 0 else 0
        print(f"    {bucket:<25} {in_bucket:>6} ({pct:.1%})")


def find_latest_dir(prefix: str) -> Path | None:
    """Find latest results directory matching prefix."""
    results_dir = ROOT / "results"
    candidates = sorted(results_dir.glob(f"{prefix}_*"), reverse=True)
    return candidates[0] if candidates else None


def main():
    parser = argparse.ArgumentParser(description="A/B bucket constraint comparison")
    parser.add_argument("--vec-dir", type=str, default=None)
    parser.add_argument("--rolling-dir", type=str, default=None)
    parser.add_argument("--holdout-dir", type=str, default=None)
    parser.add_argument("--triple-dir", type=str, default=None)
    args = parser.parse_args()

    print("=" * 80)
    print("🔬 A/B 对照: 跨桶约束 (min_buckets=3, max_per_bucket=2)")
    print("=" * 80)

    # Auto-detect latest directories if not specified
    vec_dir = Path(args.vec_dir) if args.vec_dir else find_latest_dir("vec_from_wfo")
    rolling_dir = Path(args.rolling_dir) if args.rolling_dir else find_latest_dir("rolling_oos_consistency")
    holdout_dir = Path(args.holdout_dir) if args.holdout_dir else find_latest_dir("holdout_validation")
    triple_dir = Path(args.triple_dir) if args.triple_dir else find_latest_dir("final_triple_validation")

    # ─── VEC Results ─────────────────────────────────────────────
    if vec_dir and vec_dir.exists():
        vec_file = vec_dir / "full_space_results.parquet"
        if vec_file.exists():
            print(f"\n📁 VEC: {vec_dir.name}")
            vec_df = pd.read_parquet(vec_file)
            vec_df = tag_bucket_info(vec_df)

            vec_metrics = [c for c in vec_df.columns if c not in [
                "combo", "bucket_pass", "n_buckets", "buckets", "bucket_dist",
                "factor_indices", "combo_idx",
            ]]
            compare_groups(vec_df, vec_metrics, "VEC 全量结果")
            analyze_bucket_distribution(vec_df, "VEC 全量")

            # Save tagged results
            out_path = ROOT / "results" / "ab_bucket_comparison"
            out_path.mkdir(parents=True, exist_ok=True)
            vec_df.to_parquet(out_path / "vec_tagged.parquet")
        else:
            print(f"  ⚠️ VEC results not found: {vec_file}")
    else:
        print(f"  ⚠️ VEC directory not found")

    # ─── Rolling OOS Results ─────────────────────────────────────
    if rolling_dir and rolling_dir.exists():
        rolling_file = rolling_dir / "rolling_oos_summary.parquet"
        if rolling_file.exists():
            print(f"\n📁 Rolling: {rolling_dir.name}")
            roll_df = pd.read_parquet(rolling_file)
            roll_df = tag_bucket_info(roll_df)

            roll_metrics = [c for c in roll_df.columns if c not in [
                "combo", "bucket_pass", "n_buckets", "buckets", "bucket_dist",
            ]]
            compare_groups(roll_df, roll_metrics, "Rolling OOS 结果")

    # ─── Holdout Results ─────────────────────────────────────────
    if holdout_dir and holdout_dir.exists():
        holdout_file = holdout_dir / "holdout_results.parquet"
        if holdout_file.exists():
            print(f"\n📁 Holdout: {holdout_dir.name}")
            ho_df = pd.read_parquet(holdout_file)
            ho_df = tag_bucket_info(ho_df)

            ho_metrics = [c for c in ho_df.columns if c not in [
                "combo", "bucket_pass", "n_buckets", "buckets", "bucket_dist",
            ]]
            compare_groups(ho_df, ho_metrics, "Holdout 结果")

    # ─── Triple Validation Candidates ────────────────────────────
    if triple_dir and triple_dir.exists():
        cand_file = triple_dir / "final_candidates.parquet"
        if cand_file.exists():
            print(f"\n📁 Triple: {triple_dir.name}")
            cand_df = pd.read_parquet(cand_file)
            cand_df = tag_bucket_info(cand_df)

            n_total = len(cand_df)
            n_pass = cand_df["bucket_pass"].sum()
            n_fail = n_total - n_pass
            print(f"\n  三重验证通过的 candidates: {n_total}")
            print(f"    跨桶通过: {n_pass} ({n_pass/n_total:.1%})")
            print(f"    跨桶不通过: {n_fail} ({n_fail/n_total:.1%})")

            if n_total > 0:
                cand_metrics = [c for c in cand_df.columns if c not in [
                    "combo", "bucket_pass", "n_buckets", "buckets", "bucket_dist",
                ]]
                if n_pass > 0 and n_fail > 0:
                    compare_groups(cand_df, cand_metrics, "Triple Validation Candidates")
                analyze_bucket_distribution(cand_df, "Triple Validation Candidates")

    # ─── Final Verdict ───────────────────────────────────────────
    print(f"\n{'='*80}")
    print("📋 A/B 判定 (需满足至少 2 条)")
    print("=" * 80)
    print("  1) Holdout median ≥ A + 1pp")
    print("  2) Holdout 10%分位 ≥ A + 1pp (尾部不退化)")
    print("  3) Candidates 数量 ≥ A")
    print("  4) Rolling OOS 通过率更高或最差季度更好")
    print("  5) 搜索空间减半 (工程效率)")
    print(f"\n  结果已保存至: results/ab_bucket_comparison/")


if __name__ == "__main__":
    main()
