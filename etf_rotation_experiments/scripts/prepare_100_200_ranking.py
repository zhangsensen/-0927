import pandas as pd
from pathlib import Path

base_path = Path("/home/sensen/dev/projects/-0927/etf_rotation_experiments/results/run_20251116_151853")
input_file = base_path / "ranking_ml_top2000.parquet"
output_file = base_path / "ranking_ml_100_200.parquet"

df = pd.read_parquet(input_file)

# Select ranks 100 to 200 (indices 100 to 199, since it's 0-indexed and sorted by rank usually)
# Let's verify sorting first.
if 'ltr_rank' in df.columns:
    df = df.sort_values('ltr_rank')
else:
    # Fallback if ltr_rank is not there, though it should be for ML ranking
    pass

subset = df.iloc[100:200].copy()
subset.to_parquet(output_file)

print(f"Saved {len(subset)} combos (Rank 101-200) to {output_file}")
print(subset[['combo', 'ltr_score']].head())
