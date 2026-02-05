import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent


def main():
    print("🔍 Selecting Top 24 V3 Strategies...")

    # Load Results
    df = pd.read_csv(ROOT / "results/full_space_v3_results.csv")
    print(f"Total Strategies: {len(df)}")

    # 1. Robustness Filters
    # 2022 Bear Market Survival
    df_filtered = df[df["ret_2022"] > 0].copy()
    print(f"After 2022 Filter (>0%): {len(df_filtered)}")

    # 2024 Volatility Survival
    df_filtered = df_filtered[df_filtered["ret_2024"] > 0].copy()
    print(f"After 2024 Filter (>0%): {len(df_filtered)}")

    # Max Drawdown Constraint (e.g. < 30%)
    df_filtered = df_filtered[df_filtered["max_dd"] > -0.30].copy()
    print(f"After MaxDD Filter (<-30%): {len(df_filtered)}")

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

    # 3. Select Top 24
    top_24 = df_filtered.sort_values("score", ascending=False).head(24)

    # Save
    out_path = ROOT / "results/v3_top24_candidates.csv"
    top_24.to_csv(out_path, index=False)

    print(f"\n🏆 Top 24 Candidates saved to {out_path}")
    print(
        top_24[
            ["combo", "ann_return", "max_dd", "sharpe", "ret_2022", "ret_2024", "score"]
        ].to_markdown()
    )


if __name__ == "__main__":
    main()
