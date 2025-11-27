import pandas as pd
from pathlib import Path

base_path = Path("/home/sensen/dev/projects/-0927/etf_rotation_experiments/results/run_20251116_151853")
input_file = base_path / "all_combos.parquet"

try:
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df)} strategies from {input_file.name}")
except Exception as e:
    print(f"Error: {e}")
