import pandas as pd

combo = "MOM_20D + RSI_14 + VOL_RATIO_20D + VOL_RATIO_60D + VORTEX_14D"
df = pd.DataFrame([{'combo': combo, 'ltr_score': 0.0}]) # Score doesn't matter for backtest
df.to_parquet("/home/sensen/dev/projects/-0927/etf_rotation_experiments/results/run_20251116_151853/balanced_candidate.parquet")
print(f"Saved balanced candidate: {combo}")
