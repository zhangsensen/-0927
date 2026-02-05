import pandas as pd
import sys

input_path = "results/vec_from_wfo_20251214_000418/full_space_results.csv"
output_path = "results/vec_from_wfo_20251214_000418/top2000_training_results.csv"

print(f"Reading {input_path}...")
df = pd.read_csv(input_path)
print(f"Total combos: {len(df)}")

# Sort by Calmar Ratio (or whatever metric is best)
df_sorted = df.sort_values("vec_calmar_ratio", ascending=False)

# Take top 2000
top_n = 2000
df_top = df_sorted.head(top_n)
print(f"Selected top {top_n} combos.")

df_top.to_csv(output_path, index=False)
print(f"Saved to {output_path}")
