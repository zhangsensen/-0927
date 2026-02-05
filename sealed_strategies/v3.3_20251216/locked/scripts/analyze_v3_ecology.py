import pandas as pd
import numpy as np
from pathlib import Path
import re
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).parent.parent


def main():
    print("📊 Starting V3 Ecological Analysis...")

    # 1. Load Data
    print("Loading data...")
    df_v0 = pd.read_csv(ROOT / "results/ARCHIVE_unified_wfo_43etf_best/all_combos.csv")
    df_v3 = pd.read_csv(ROOT / "results/full_space_v3_results.csv")

    # Merge
    # V0 has 'combo', 'rank' (V0 Rank)
    # V3 has 'combo', metrics...
    df = pd.merge(df_v3, df_v0[["combo", "rank"]], on="combo", how="inner")
    df.rename(columns={"rank": "rank_v0"}, inplace=True)

    print(f"Merged Data Shape: {df.shape}")

    # 2. Calculate V3 Score and Rank
    # Normalize metrics
    df["rank_ret"] = df["ann_return"].rank(pct=True)
    df["rank_sharpe"] = df["sharpe"].rank(pct=True)
    df["rank_dd"] = df["max_dd"].rank(pct=True)  # Higher max_dd (closer to 0) is better

    # Composite Score (40% Ret, 30% Sharpe, 30% DD)
    df["score_v3"] = (
        0.4 * df["rank_ret"] + 0.3 * df["rank_sharpe"] + 0.3 * df["rank_dd"]
    )

    # V3 Rank
    df["rank_v3"] = df["score_v3"].rank(ascending=False)

    # Define Top 300
    top300 = df.sort_values("score_v3", ascending=False).head(300).copy()

    # 3. Factor Significance Analysis
    print("\n🔍 Analyzing Factor Significance...")

    def get_factors(combo_str):
        return [f.strip() for f in combo_str.split("+")]

    # Count in Top 300
    all_factors_top300 = []
    for c in top300["combo"]:
        all_factors_top300.extend(get_factors(c))

    freq_top300 = pd.Series(all_factors_top300).value_counts() / len(top300)

    # Count in Base (All)
    all_factors_base = []
    for c in df["combo"]:
        all_factors_base.extend(get_factors(c))

    freq_base = pd.Series(all_factors_base).value_counts() / len(df)

    # Lift Ratio
    factor_stats = pd.DataFrame({"freq_top300": freq_top300, "freq_base": freq_base})
    factor_stats["lift_ratio"] = factor_stats["freq_top300"] / factor_stats["freq_base"]
    factor_stats = factor_stats.sort_values("lift_ratio", ascending=False)

    # Save Factor Stats
    factor_stats.to_csv(ROOT / "results/factor_importance_v3.csv")
    print("Saved factor_importance_v3.csv")

    # 4. Window Bias Analysis
    print("\n🔍 Analyzing Lookback Window Bias...")

    def extract_window(factor_name):
        m = re.search(r"_(\d+)D?$", factor_name)
        if m:
            return int(m.group(1))
        return None

    window_counts = []
    for f in all_factors_top300:
        w = extract_window(f)
        if w:
            window_counts.append(w)

    win_series = pd.Series(window_counts).value_counts().sort_index()
    win_dist = win_series / win_series.sum()

    # 5. Rank Migration Analysis
    print("\n🔍 Analyzing Rank Migration...")

    # Categories
    # Robust Kings: V0 <= 300 & V3 <= 300
    robust_kings = df[(df["rank_v0"] <= 300) & (df["rank_v3"] <= 300)]

    # New Comers: V0 > 1000 & V3 <= 300
    new_comers = df[(df["rank_v0"] > 1000) & (df["rank_v3"] <= 300)]

    # Fallen Angels: V0 <= 300 & V3 > 1000
    fallen_angels = df[(df["rank_v0"] <= 300) & (df["rank_v3"] > 1000)]

    # Plot Scatter
    try:
        plt.figure(figsize=(10, 10))
        plt.scatter(
            df["rank_v0"], df["rank_v3"], alpha=0.1, s=1, c="gray", label="Others"
        )
        plt.scatter(
            robust_kings["rank_v0"],
            robust_kings["rank_v3"],
            c="green",
            s=10,
            label="Robust Kings",
        )
        plt.scatter(
            new_comers["rank_v0"],
            new_comers["rank_v3"],
            c="blue",
            s=10,
            label="New Comers",
        )
        plt.scatter(
            fallen_angels["rank_v0"],
            fallen_angels["rank_v3"],
            c="red",
            s=10,
            label="Fallen Angels",
        )
        plt.xlabel("Original Rank (V0 - IC Based)")
        plt.ylabel("New Rank (V3 - Vol Adaptive Score)")
        plt.title("Strategy Rank Migration: V0 vs V3")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        plt.savefig(ROOT / "results/rank_migration_scatter.png")
        print("Saved rank_migration_scatter.png")
    except Exception as e:
        print(f"Could not save plot: {e}")

    # 6. Performance Clustering
    print("\n🔍 Analyzing Performance Clusters...")

    # Define Quadrants
    # All Weather: 2022 > 0, 2024 > 20%
    q_all_weather = top300[(top300["ret_2022"] > 0) & (top300["ret_2024"] > 0.20)]

    # Crisis Alpha: 2022 > 20%, 2024 < 20%
    q_crisis_alpha = top300[(top300["ret_2022"] > 0.20) & (top300["ret_2024"] < 0.20)]

    # Bull Rider: 2022 < 0, 2024 > 20%
    q_bull_rider = top300[(top300["ret_2022"] < 0) & (top300["ret_2024"] > 0.20)]

    # Defensive: 2022 > 0, 2024 < 20% (Slow but steady)
    q_defensive = top300[
        (top300["ret_2022"] > 0)
        & (top300["ret_2024"] < 0.20)
        & (top300["ret_2022"] <= 0.20)
    ]

    # 7. Safety Margin
    top300_sharpe = top300["sharpe"].mean()
    base_sharpe = df["sharpe"].mean()

    # 8. Generate Report
    report = f"""# 🧬 V3 Strategy Ecology Report

## 1. Factor Significance (Top 300 vs Base)
| Factor | Lift Ratio | Freq Top300 | Freq Base |
| :--- | :--- | :--- | :--- |
"""
    for idx, row in factor_stats.head(10).iterrows():
        report += f"| {idx} | {row['lift_ratio']:.2f}x | {row['freq_top300']:.1%} | {row['freq_base']:.1%} |\n"

    report += "\n### Bottom 5 Factors (Losers)\n"
    report += "| Factor | Lift Ratio | Freq Top300 | Freq Base |\n| :--- | :--- | :--- | :--- |\n"
    for idx, row in factor_stats.tail(5).iterrows():
        report += f"| {idx} | {row['lift_ratio']:.2f}x | {row['freq_top300']:.1%} | {row['freq_base']:.1%} |\n"

    report += f"""
## 2. Lookback Window Bias
| Window | Share in Top 300 |
| :--- | :--- |
"""
    for w, val in win_dist.items():
        report += f"| {w}D | {val:.1%} |\n"

    report += f"""
## 3. Rank Migration (V0 -> V3)
- **Robust Kings** (Top 300 -> Top 300): {len(robust_kings)} strategies
- **New Comers** (Rank >1000 -> Top 300): {len(new_comers)} strategies
- **Fallen Angels** (Top 300 -> Rank >1000): {len(fallen_angels)} strategies

## 4. Performance Clusters (Top 300)
- **All Weather** (2022 > 0%, 2024 > 20%): {len(q_all_weather)} ({len(q_all_weather)/300:.1%})
- **Crisis Alpha** (2022 > 20%, 2024 < 20%): {len(q_crisis_alpha)} ({len(q_crisis_alpha)/300:.1%})
- **Bull Rider** (2022 < 0%, 2024 > 20%): {len(q_bull_rider)} ({len(q_bull_rider)/300:.1%})
- **Defensive** (2022 > 0%, 2024 < 20%): {len(q_defensive)} ({len(q_defensive)/300:.1%})

## 5. Safety Margin
- **Top 300 Avg Sharpe**: {top300_sharpe:.3f}
- **Population Avg Sharpe**: {base_sharpe:.3f}
- **Alpha Density**: {(top300_sharpe/base_sharpe - 1)*100:.1f}% improvement
"""

    with open(ROOT / "results/strategy_cluster_report.md", "w") as f:
        f.write(report)

    print("\n✅ Analysis Complete. Report saved to results/strategy_cluster_report.md")
    print(report)


if __name__ == "__main__":
    main()
