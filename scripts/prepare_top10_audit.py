
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
csv_path = ROOT / 'results/vec_full_space_20251130_235418/full_space_results.csv'
output_path = ROOT / 'results/top10_audit.parquet'

print(f"Reading from: {csv_path}")
df = pd.read_csv(csv_path)

# Sort by Calmar
df_sorted = df.sort_values('vec_calmar_ratio', ascending=False).head(10)

print("Top 10 Strategies:")
print(df_sorted[['combo', 'vec_calmar_ratio']])

# Save to parquet
df_sorted.to_parquet(output_path)
print(f"Saved to: {output_path}")
