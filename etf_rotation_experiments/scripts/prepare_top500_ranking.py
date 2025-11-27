import pandas as pd
from pathlib import Path

base_path = Path("/home/sensen/dev/projects/-0927/etf_rotation_experiments/results/run_20251116_151853")
input_file = base_path / "ranking_ml_top2000.parquet"
output_file = base_path / "ranking_ml_top500.parquet"

try:
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df)} strategies from {input_file.name}")
    
    # Ensure sorted by ltr_rank if available, otherwise trust the order or sort by ltr_score
    if 'ltr_rank' in df.columns:
        df = df.sort_values('ltr_rank')
    elif 'ltr_score' in df.columns:
        df = df.sort_values('ltr_score', ascending=False)
        
    subset = df.head(500).copy()
    subset.to_parquet(output_file)
    
    print(f"Saved top {len(subset)} strategies to {output_file.name}")
    print("Top 3 combos:")
    print(subset[['combo', 'ltr_score']].head(3))
    print("Bottom 3 combos:")
    print(subset[['combo', 'ltr_score']].tail(3))

except Exception as e:
    print(f"Error: {e}")
