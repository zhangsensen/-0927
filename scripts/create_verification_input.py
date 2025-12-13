import pandas as pd
import os
from pathlib import Path

# Define the best combo
best_combo = "ADX_14D + MAX_DD_60D + PRICE_POSITION_120D + PRICE_POSITION_20D + SHARPE_RATIO_20D"

# Create a dataframe
df = pd.DataFrame([{
    "combo": best_combo,
    "combo_size": 5,
    "mean_oos_ic": 0.1495, # Dummy value from AGENTS.md
    "positive_rate": 0.6, # Dummy
    "stability_score": 1.0 # Dummy
}])

# Define output directory
output_dir = Path("results/run_verification")
output_dir.mkdir(parents=True, exist_ok=True)

# Save to parquet
output_path = output_dir / "top100_by_ic.parquet"
df.to_parquet(output_path)

print(f"Created verification input at {output_path}")
