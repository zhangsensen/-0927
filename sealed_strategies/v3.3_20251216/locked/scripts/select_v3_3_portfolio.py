"""Select a diverse portfolio of strategies for v3.3 release.

Selection Logic:
1. Pool: Gate ON survivors (31 strategies).
2. Ranking: Composite Score (descending).
3. Diversity: Jaccard Similarity of factors < 0.6 with any already selected strategy.
4. Target: 3-5 strategies.
"""

import pandas as pd
from pathlib import Path


def get_factors(combo_str):
    return set(f.strip() for f in combo_str.split("+"))


def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0


def main():
    p_cand = Path(
        "results/final_triple_validation_20251216_041418/final_candidates.parquet"
    )
    df = pd.read_parquet(p_cand)

    # Sort by composite score
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)

    selected_indices = []
    selected_factors = []

    print(f"Pool size: {len(df)}")
    print("-" * 60)

    # Greedy selection with diversity constraint
    for idx, row in df.iterrows():
        current_factors = get_factors(row["combo"])

        is_diverse = True
        for existing_factors in selected_factors:
            sim = jaccard_similarity(current_factors, existing_factors)
            if sim > 0.6:  # Threshold: max 60% overlap allowed
                is_diverse = False
                break

        if is_diverse:
            selected_indices.append(idx)
            selected_factors.append(current_factors)
            print(f"SELECTED #{len(selected_indices)}: {row['combo']}")
            print(
                f"  Score: {row['composite_score']:.4f}, Holdout Calmar: {row['holdout_calmar_ratio']:.4f}"
            )
            print(f"  Factors: {current_factors}")
            print("-" * 60)

        if len(selected_indices) >= 5:
            break

    # Create result dataframe
    portfolio = df.loc[selected_indices].copy()

    # Save to CSV for review
    out_path = Path("results/v3_3_portfolio_candidates.csv")
    portfolio.to_csv(out_path, index=False)
    print(f"\nSaved {len(portfolio)} strategies to {out_path}")

    # Print summary table for report
    cols = [
        "combo",
        "composite_score",
        "holdout_calmar_ratio",
        "vec_calmar_ratio",
        "holdout_return",
        "vec_max_drawdown",
    ]
    print("\nSelected Portfolio Summary:")
    print(portfolio[cols].to_markdown(index=False))


if __name__ == "__main__":
    main()
