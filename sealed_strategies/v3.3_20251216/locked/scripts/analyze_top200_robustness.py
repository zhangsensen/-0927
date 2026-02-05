import pandas as pd
import numpy as np
from pathlib import Path
import re

ROOT = Path(__file__).parent.parent


def main():
    print("📊 Analyzing Top 200 Robustness...")

    df = pd.read_csv(ROOT / "results/v3_top200_bt_crosscheck.csv")

    # 1. Classification
    def classify(row):
        # Execution Robustness (The primary filter)
        # We ignore VEC-BT gap for grading, but we will report it.

        # Strong: Ratio >= 0.8 AND BT_B_Ret >= 10%
        if row["exec_return_ratio"] >= 0.8 and row["bt_b_ann_ret"] >= 0.10:
            return "A"
        # Good: Ratio >= 0.7 AND BT_B_Ret >= 8%
        elif row["exec_return_ratio"] >= 0.7 and row["bt_b_ann_ret"] >= 0.08:
            return "B"
        # Acceptable: Ratio >= 0.6 AND BT_B_Ret >= 5%
        elif row["exec_return_ratio"] >= 0.6 and row["bt_b_ann_ret"] >= 0.05:
            return "C"
        else:
            return "D"

    df["grade"] = df.apply(classify, axis=1)
    df["vec_optimism"] = df["vec_ann_ret"] - df["bt_a_ann_ret"]

    # Save classified
    df.to_csv(ROOT / "results/v3_top200_classified.csv", index=False)

    # 2. Statistics
    grade_counts = df["grade"].value_counts().sort_index()
    print("\nGrade Distribution:")
    print(grade_counts)

    # 3. Factor Analysis
    def get_factors(combo_str):
        return [f.strip() for f in combo_str.split("+")]

    factors_by_grade = {"A": [], "B": [], "C": [], "D": []}
    for _, row in df.iterrows():
        fs = get_factors(row["combo"])
        factors_by_grade[row["grade"]].extend(fs)

    # Calculate frequency per grade
    factor_freq = {}
    for g in ["A", "B", "C", "D"]:
        if len(df[df["grade"] == g]) > 0:
            s = pd.Series(factors_by_grade[g]).value_counts() / len(
                df[df["grade"] == g]
            )
            factor_freq[g] = s

    # Compare A vs D (Strong vs Weak)
    print("\nFactor Analysis (A vs D):")
    if "A" in factor_freq and "D" in factor_freq:
        common_factors = set(factor_freq["A"].index).union(set(factor_freq["D"].index))
        factor_comp = []
        for f in common_factors:
            freq_a = factor_freq["A"].get(f, 0)
            freq_d = factor_freq["D"].get(f, 0)
            factor_comp.append(
                {
                    "factor": f,
                    "freq_A": freq_a,
                    "freq_D": freq_d,
                    "ratio_A_D": freq_a / freq_d if freq_d > 0 else 999,
                }
            )

        comp_df = pd.DataFrame(factor_comp).sort_values("ratio_A_D", ascending=False)
        print(comp_df.head(10).to_markdown())

    # 4. Top Candidates
    # Sort by BT_B Sharpe within Grade A/B
    top_candidates = (
        df[df["grade"].isin(["A", "B"])]
        .sort_values(["grade", "bt_b_sharpe"], ascending=[True, False])
        .head(20)
    )

    print("\n🏆 Top Robust Candidates (Sorted by Grade then BT_B Sharpe):")
    cols = [
        "combo",
        "vec_ann_ret",
        "bt_a_ann_ret",
        "bt_b_ann_ret",
        "bt_b_max_dd",
        "bt_b_sharpe",
        "exec_return_ratio",
        "grade",
    ]
    print(top_candidates[cols].to_markdown())

    # 5. Generate Report Text
    report = f"""# 🛡️ Top 200 Execution Robustness Report

## 1. Overall Distribution
- **A (Strong)**: {grade_counts.get('A', 0)} - Robust execution & High Return (>10%).
- **B (Good)**: {grade_counts.get('B', 0)} - Decent execution & Return (>8%).
- **C (Acceptable)**: {grade_counts.get('C', 0)} - Low return or higher slippage.
- **D (Weak)**: {grade_counts.get('D', 0)} - Failed execution check.

## 2. Top Candidates
{top_candidates[cols].to_markdown(index=False)}

## 3. Factor Insights
"""
    if "A" in factor_freq and "D" in factor_freq:
        # High Ratio (A preferred)
        high_ratio = comp_df[comp_df["freq_A"] > 0.1].head(5)
        report += "### Robust Factors (Favored by A-Grade)\n"
        for _, row in high_ratio.iterrows():
            report += f"- **{row['factor']}**: {row['freq_A']:.1%} in A vs {row['freq_D']:.1%} in D (Ratio: {row['ratio_A_D']:.1f}x)\n"

        # Low Ratio (D preferred)
        low_ratio = comp_df[comp_df["freq_D"] > 0.1].tail(5)
        report += "\n### Fragile Factors (Favored by D-Grade)\n"
        for _, row in low_ratio.iterrows():
            ratio_val = 1 / row["ratio_A_D"] if row["ratio_A_D"] > 0 else 999
            report += f"- **{row['factor']}**: {row['freq_D']:.1%} in D vs {row['freq_A']:.1%} in A (Ratio: {ratio_val:.1f}x fragility)\n"

    with open(ROOT / "results/top200_robustness_report.md", "w") as f:
        f.write(report)

    print("\nReport saved to results/top200_robustness_report.md")


if __name__ == "__main__":
    main()
