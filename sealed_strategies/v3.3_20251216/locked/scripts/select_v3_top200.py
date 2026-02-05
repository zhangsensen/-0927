import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent


def main():
    print("🔍 Selecting Top 200 V3 Candidates for Robustness Check...")

    # Load Results
    df = pd.read_csv(ROOT / "results/full_space_v3_results.csv")
    print(f"Total Strategies: {len(df)}")

    # 1. Hard Filters
    # 2022 Bear Market Survival (> 0%)
    # 2024 Volatility Survival (> 0%)
    # Max Drawdown Constraint (> -30%)
    # Annual Return (> 12%)

    mask = (
        (df["ret_2022"] > 0)
        & (df["ret_2024"] > 0)
        & (df["max_dd"] > -0.30)
        & (df["ann_return"] > 0.12)
    )

    df_filtered = df[mask].copy()
    print(f"After Filters: {len(df_filtered)}")

    # 2. Scoring
    # Normalize metrics to 0-1
    df_filtered["rank_sharpe"] = df_filtered["sharpe"].rank(pct=True)
    df_filtered["rank_ret"] = df_filtered["ann_return"].rank(pct=True)
    df_filtered["rank_dd"] = df_filtered["max_dd"].rank(
        pct=True
    )  # Higher is better (closer to 0)

    # Composite Score
    # Weights: Return 40%, Sharpe 30%, DD 30%
    df_filtered["score"] = (
        0.4 * df_filtered["rank_ret"]
        + 0.3 * df_filtered["rank_sharpe"]
        + 0.3 * df_filtered["rank_dd"]
    )

    # 3. Select Top 200
    top_200 = df_filtered.sort_values("score", ascending=False).head(200)

    # Save
    out_path = ROOT / "results/v3_top200_candidates.csv"
    top_200.to_csv(out_path, index=False)

    print(f"\n🏆 Top 200 Candidates saved to {out_path}")
    print(
        top_200[
            ["combo", "ann_return", "max_dd", "sharpe", "ret_2022", "ret_2024", "score"]
        ]
        .head(10)
        .to_markdown()
    )


if __name__ == "__main__":
    main()
