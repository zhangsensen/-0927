import pandas as pd
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"


def main():
    # Find the latest BT results
    bt_dirs = sorted([d for d in RESULTS_DIR.glob("bt_backtest_full_*") if d.is_dir()])
    if not bt_dirs:
        print("No BT results found.")
        sys.exit(1)

    latest_bt_dir = bt_dirs[-1]
    bt_results_path = latest_bt_dir / "bt_results.csv"
    print(f"Loading BT results from {bt_results_path}")

    df = pd.read_csv(bt_results_path)

    # Calculate Score
    # Normalize metrics for scoring
    # We want high AnnRet, high Sharpe, low MaxDD

    # Handle potential NaNs or Infs
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["bt_annual_return", "bt_sharpe_ratio", "bt_max_drawdown"]
    )

    # Simple ranking score
    df["rank_ret"] = df["bt_annual_return"].rank(ascending=False)
    df["rank_sharpe"] = df["bt_sharpe_ratio"].rank(ascending=False)
    df["rank_dd"] = df["bt_max_drawdown"].rank(ascending=True)  # Lower DD is better

    # Weighted Rank Score (lower is better)
    # Weights: Ret 0.4, Sharpe 0.3, DD 0.3
    df["score_rank"] = (
        0.4 * df["rank_ret"] + 0.3 * df["rank_sharpe"] + 0.3 * df["rank_dd"]
    )

    # Sort by score
    df = df.sort_values("score_rank", ascending=True)

    # Assign Grades
    n = len(df)
    df["grade"] = "D"
    df.iloc[: int(n * 0.1), df.columns.get_loc("grade")] = "A"  # Top 10%
    df.iloc[int(n * 0.1) : int(n * 0.3), df.columns.get_loc("grade")] = "B"  # Next 20%
    df.iloc[int(n * 0.3) : int(n * 0.6), df.columns.get_loc("grade")] = "C"  # Next 30%

    # Save Grading
    output_path = RESULTS_DIR / "v3_top200_bt_grading_no_lookahead.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved grading to {output_path}")

    # Generate Markdown Summary
    summary_path = RESULTS_DIR / "v3_strategy_summary.md"
    with open(summary_path, "w") as f:
        f.write("# v3.1 Strategy Analysis Report (No Lookahead)\n\n")
        f.write(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Source**: {bt_results_path.name}\n\n")

        f.write("## 1. Grade Distribution\n")
        grade_counts = df["grade"].value_counts().sort_index()
        f.write(grade_counts.to_markdown())
        f.write("\n\n")

        f.write("## 2. Top 10 Strategies (Grade A)\n")
        cols = [
            "combo",
            "grade",
            "bt_annual_return",
            "bt_max_drawdown",
            "bt_sharpe_ratio",
            "bt_calmar_ratio",
        ]
        f.write(df.head(10)[cols].to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n")

        f.write("## 3. Factor Ecosystem\n")
        # Count factor usage in Top 50 (Grade A+B)
        top_df = df[df["grade"].isin(["A", "B"])]
        all_factors = []
        for combo in top_df["combo"]:
            factors = combo.split(" + ")
            all_factors.extend(factors)

        factor_counts = pd.Series(all_factors).value_counts()
        f.write("### Factor Frequency in Top Strategies (Grade A & B)\n")
        f.write(factor_counts.to_frame("count").to_markdown())
        f.write("\n\n")

        f.write("## 4. Best Strategy Details\n")
        best = df.iloc[0]
        f.write(f"**Combo**: `{best['combo']}`\n")
        f.write(f"- **Annual Return**: {best['bt_annual_return']:.2%}\n")
        f.write(f"- **Max Drawdown**: {best['bt_max_drawdown']:.2%}\n")
        f.write(f"- **Sharpe Ratio**: {best['bt_sharpe_ratio']:.4f}\n")
        f.write(f"- **Calmar Ratio**: {best['bt_calmar_ratio']:.4f}\n")

    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
