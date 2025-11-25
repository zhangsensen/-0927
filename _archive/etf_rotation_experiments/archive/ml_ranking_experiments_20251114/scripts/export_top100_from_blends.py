#!/usr/bin/env python3
"""
从 ranking_blends 目录导出 Top100 组合列表到 run 目录：
  - top100_ic_combos.csv（baseline 排序）
  - top100_calibrated_combos.csv（calibrated 排序）
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Export Top100 combos CSVs from ranking_blends")
    ap.add_argument("--run-ts", type=str, required=True, help="Target run timestamp (e.g., 20251113_145102)")
    ap.add_argument("--topk", type=int, default=100, help="TopK to export; default 100")
    args = ap.parse_args()

    exp_root = Path(__file__).resolve().parent.parent
    run_dir = exp_root / "results" / f"run_{args.run_ts}"
    blend_dir = run_dir / "ranking_blends"
    base_path = blend_dir / "ranking_baseline.parquet"
    cal_path = blend_dir / "ranking_lightgbm.parquet"
    if not base_path.exists() or not cal_path.exists():
        raise FileNotFoundError(f"缺少 ranking_blends 文件: {base_path} / {cal_path}")

    base = pd.read_parquet(base_path)
    cal = pd.read_parquet(cal_path)

    base_top = base.head(args.topk)[[c for c in base.columns if c in ("combo", "rank_score", "mean_oos_ic")]].copy()
    if "rank_score" not in base_top.columns and "mean_oos_ic" in base_top.columns:
        base_top["rank_score"] = base_top["mean_oos_ic"].astype(float)
    base_top.to_csv(run_dir / f"top{args.topk}_ic_combos.csv", index=False)

    cal_top = cal.head(args.topk)[[c for c in cal.columns if c in ("combo", "rank_score", "calibrated_sharpe_pred")]].copy()
    if "rank_score" not in cal_top.columns and "calibrated_sharpe_pred" in cal_top.columns:
        cal_top["rank_score"] = cal_top["calibrated_sharpe_pred"].astype(float)
    cal_top.to_csv(run_dir / f"top{args.topk}_calibrated_combos.csv", index=False)

    # Also write canonical names expected by run_top100_comparison_backtest
    if args.topk == 100:
        base_top.to_csv(run_dir / "top100_ic_combos.csv", index=False)
        cal_top.to_csv(run_dir / "top100_calibrated_combos.csv", index=False)

    print(f"✅ 导出完成: {run_dir}/top{args.topk}_ic_combos.csv | top{args.topk}_calibrated_combos.csv")


if __name__ == "__main__":
    main()
