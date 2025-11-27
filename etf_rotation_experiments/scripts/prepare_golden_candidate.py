import pandas as pd
from pathlib import Path

base_path = Path("/home/sensen/dev/projects/-0927/etf_rotation_experiments/results/run_20251116_151853")
input_file = base_path / "ranking_ml_top500.parquet"
output_file = base_path / "golden_candidate.parquet"

df = pd.read_parquet(input_file)
target_combo = "ADX_14D + OBV_SLOPE_10D + PRICE_POSITION_20D + RSI_14 + VORTEX_14D"

subset = df[df['combo'] == target_combo].copy()

if len(subset) == 0:
    print(f"Error: Combo {target_combo} not found in {input_file}")
else:
    subset.to_parquet(output_file)
    print(f"Saved golden candidate to {output_file}")
    print(subset.iloc[0])
