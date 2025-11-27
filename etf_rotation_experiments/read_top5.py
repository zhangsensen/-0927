import pandas as pd
import sys

path = "results/run_20251116_151853/ranking_ml_top2000.parquet"
try:
    df = pd.read_parquet(path)
    print("Top 5 Combos:")
    for i in range(5):
        print(f"Rank {i+1}: {df.iloc[i]['combo']}")
        # Check if ltr_score exists, otherwise print columns
        if 'ltr_score' in df.columns:
            print(f"       Score: {df.iloc[i]['ltr_score']:.4f}")
        else:
            print(f"       Columns: {df.columns}")
except Exception as e:
    print(e)
