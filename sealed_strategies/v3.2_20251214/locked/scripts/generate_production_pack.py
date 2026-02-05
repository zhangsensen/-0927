#!/usr/bin/env python3
"""scripts/generate_production_pack.py

Generate Production Pack
------------------------
Merges leakage-controlled Final Triple Validation results (VEC + Rolling + Holdout)
with BT Audit results.

For v3.2 delivery, BT (event-driven) is treated as Ground Truth for returns.
We therefore prioritize BT-split returns (train/holdout) to avoid any ambiguity
caused by differing execution assumptions between VEC and BT.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse


def _percentile_rank(values: pd.Series) -> pd.Series:
    # Deterministic [0, 1] rank, higher is better.
    # If all values are identical, return 0.5.
    if values.nunique(dropna=True) <= 1:
        return pd.Series(0.5, index=values.index)
    return values.rank(pct=True, method="average")


def _require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def main():
    parser = argparse.ArgumentParser(description="Generate Production Pack")
    parser.add_argument(
        "--candidates", type=str, required=True, help="Path to final_candidates.parquet"
    )
    parser.add_argument(
        "--bt-results", type=str, required=True, help="Path to bt_results.parquet"
    )
    parser.add_argument(
        "--top-n", type=int, default=20, help="Number of top strategies to include"
    )
    args = parser.parse_args()

    path_candidates = Path(args.candidates)
    path_bt = Path(args.bt_results)

    print(f"Loading Candidates: {path_candidates}")
    candidates = pd.read_parquet(path_candidates)

    print(f"Loading BT Results: {path_bt}")
    bt_results = pd.read_parquet(path_bt)

    _require_columns(
        candidates,
        [
            "combo",
            "vec_return",
            "holdout_return",
            "roll_all_segment_positive_rate",
            "roll_all_segment_worst_return",
            "composite_score",
        ],
        name="candidates",
    )
    _require_columns(
        bt_results,
        [
            "combo",
            "bt_return",
            "bt_train_return",
            "bt_holdout_return",
            "bt_max_drawdown",
            "bt_calmar_ratio",
            "bt_win_rate",
            "bt_profit_factor",
            "bt_total_trades",
            "bt_margin_failures",
        ],
        name="bt_results",
    )

    print("Merging datasets...")
    merged = candidates.merge(bt_results, on="combo", how="inner")
    print(f"Merged: {len(merged)} strategies")

    # Safety: ensure all candidates are BT-audited with no margin failures
    merged["bt_pass"] = merged["bt_margin_failures"].fillna(0).astype(int).eq(0)

    # Alignment diagnostics (NOT used as a strict gate; BT is ground truth)
    merged["diff_train_return_vec_vs_bt"] = (
        merged["vec_return"] - merged["bt_train_return"]
    ).abs()
    merged["diff_holdout_return_vec_vs_bt"] = (
        merged["holdout_return"] - merged["bt_holdout_return"]
    ).abs()

    # Production score (BT-ground-truth, deterministic)
    # Priority: OOS (holdout) return, then full-period robustness (calmar, MDD), then train return.
    r_hold = _percentile_rank(merged["bt_holdout_return"])
    r_train = _percentile_rank(merged["bt_train_return"])
    r_calmar = _percentile_rank(merged["bt_calmar_ratio"])
    r_mdd = _percentile_rank(
        -merged["bt_max_drawdown"]
    )  # smaller drawdown => higher rank

    merged["prod_score_bt"] = (
        0.45 * r_hold + 0.20 * r_calmar + 0.20 * r_mdd + 0.15 * r_train
    )

    # Default sort: production score first, then composite score (tie-breaker)
    merged = merged.sort_values(
        ["prod_score_bt", "composite_score"], ascending=[False, False]
    )

    # Select Top N (BT-pass only)
    merged_bt_pass = merged[merged["bt_pass"]].copy()
    top_n = merged_bt_pass.head(args.top_n).copy()

    # Output Directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/production_pack_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save artifacts
    out_all = output_dir / "production_all_candidates.parquet"
    out_top = output_dir / "production_candidates.parquet"
    merged_bt_pass.to_parquet(out_all)
    top_n.to_parquet(out_top)
    print(f"Saved ALL candidates to: {out_all}")
    print(f"Saved Top {args.top_n} candidates to: {out_top}")

    # Generate Report
    report_path = output_dir / "PRODUCTION_REPORT.md"

    with open(report_path, "w") as f:
        f.write(f"# 🚀 Production Strategy Pack v3.2 (Top {args.top_n})\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(
            "**Source**: Leakage-Controlled Triple Validation (screening) + Backtrader Audit (ground truth)\n\n"
        )

        f.write("## 1. Executive Summary\n")
        f.write(
            "This pack is intended to be *audit-grade* (no ambiguous comparisons):\n"
        )
        f.write(
            "1.  **Screening (VEC + Rolling + Holdout)**: Candidate discovery + stability filtering (leakage-controlled).\n"
        )
        f.write(
            "2.  **Audit (BT Ground Truth)**: Final production metrics use BT-split returns (train/holdout) and BT risk stats.\n\n"
        )
        f.write(
            "**Key Principle**: When VEC and BT differ due to execution assumptions, BT wins.\n\n"
        )

        f.write("### Data Splits\n")
        f.write("- **Train**: 2020-01-01 → 2025-04-30 (from config)\n")
        f.write("- **Holdout**: 2025-05-01 → 2025-10-14\n\n")

        top_strat = top_n.iloc[0]
        f.write(f"### 🏆 Best Strategy: `{top_strat['combo']}`\n")
        f.write(f"- **Train Return (BT)**: {top_strat['bt_train_return']:.2%}\n")
        f.write(f"- **Holdout Return (BT)**: {top_strat['bt_holdout_return']:.2%}\n")
        f.write(f"- **Total Return (BT, full)**: {top_strat['bt_return']:.2%}\n")
        f.write(f"- **Max Drawdown (BT, full)**: {top_strat['bt_max_drawdown']:.2%}\n")
        f.write(f"- **Calmar Ratio (BT, full)**: {top_strat['bt_calmar_ratio']:.2f}\n")
        f.write(f"- **Win Rate (trade)**: {top_strat['bt_win_rate']:.2%}\n")
        f.write(f"- **Profit Factor**: {top_strat['bt_profit_factor']:.2f}\n")
        f.write(f"- **Total Trades**: {int(top_strat['bt_total_trades'])}\n\n")

        f.write("## 2. Top Strategies List\n")
        f.write("Sorted by `prod_score_bt` (BT-ground-truth, OOS-first).\n\n")

        cols_display = [
            "combo",
            "prod_score_bt",
            "bt_train_return",
            "bt_holdout_return",
            "bt_return",
            "bt_max_drawdown",
            "bt_calmar_ratio",
            "bt_win_rate",
            "bt_profit_factor",
            "bt_total_trades",
            "roll_all_segment_positive_rate",
            "roll_all_segment_worst_return",
        ]

        display_df = top_n[cols_display].copy()
        display_df.columns = [
            "Combo",
            "ProdScore",
            "BT Train Ret",
            "BT Holdout Ret",
            "BT Total Ret",
            "BT MDD",
            "BT Calmar",
            "BT WinRate",
            "ProfitFactor",
            "Trades",
            "Roll Win%",
            "Roll Worst",
        ]

        # Formatting
        display_df["ProdScore"] = display_df["ProdScore"].apply(lambda x: f"{x:.3f}")
        for c in [
            "BT Train Ret",
            "BT Holdout Ret",
            "BT Total Ret",
            "BT MDD",
            "BT WinRate",
            "Roll Worst",
        ]:
            display_df[c] = display_df[c].apply(lambda x: f"{x:.2%}")
        display_df["BT Calmar"] = display_df["BT Calmar"].apply(lambda x: f"{x:.2f}")
        display_df["ProfitFactor"] = display_df["ProfitFactor"].apply(
            lambda x: f"{x:.2f}"
        )
        display_df["Trades"] = display_df["Trades"].astype(int)
        display_df["Roll Win%"] = display_df["Roll Win%"].apply(lambda x: f"{x:.0%}")

        f.write(display_df.to_markdown(index=False))
        f.write("\n\n")

        f.write("## 3. Audit Notes (No-Ambiguity)\n")
        f.write(
            "- **Ground Truth**: All production returns shown in this report are from BT (event-driven).\n"
        )
        f.write(
            "- **VEC vs BT**: VEC is used for fast screening; it may diverge from BT due to execution modeling differences.\n"
        )
        f.write(
            "- **Leakage Control**: Rolling stability metrics are train-only (no holdout segment mixed).\n"
        )

        f.write("\n## 4. Risk Disclosures\n")
        f.write(
            "- **Holdout Length**: 2025-05 to 2025-10 is short; monitor live drawdowns and regime shifts.\n"
        )
        f.write(
            "- **QDII Exposure**: Many top strategies use QDII assets; US/HK regime shifts can impact OOS.\n"
        )
        f.write(
            "- **Execution**: BT assumes execution at bar prices; real slippage/taxes may reduce performance.\n"
        )

        f.write("\n## 5. Implementation Guide\n")
        f.write(
            "1.  **Select**: Choose 1-3 strategies (or a small basket) from the top list.\n"
        )
        f.write(
            "2.  **Monitor**: Track drawdown vs BT expectation and stop deploying if behavior breaks.\n"
        )
        f.write(
            "3.  **Rebalance**: Follow the fixed rule: FREQ=3 trading days, POS=2, no stop-loss.\n"
        )

    print(f"Report generated: {report_path}")


if __name__ == "__main__":
    main()
