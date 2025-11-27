import pandas as pd
import sys

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

base_path = "/home/sensen/dev/projects/-0927/etf_rotation_experiments/results/run_20251116_151853"

try:
    df = pd.read_parquet(f"{base_path}/top100_by_ml.parquet")
    print(f"Loaded top100_by_ml.parquet with {len(df)} rows")
    print("Columns:", df.columns.tolist())
    print(df.head(3))
except Exception as e:
    print(f"Error loading top100: {e}")

try:
    df2 = pd.read_parquet(f"{base_path}/ranking_ml_top2000.parquet")
    print(f"\nLoaded ranking_ml_top2000.parquet with {len(df2)} rows")
    print("Columns:", df2.columns.tolist())
    print(df2.head(3))
except Exception as e:
    print(f"Error loading top2000: {e}")
