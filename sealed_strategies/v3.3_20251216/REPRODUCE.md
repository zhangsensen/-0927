# Reproduce v3.3

To reproduce this release:

1. Ensure environment is set up (`uv sync`).
2. Run the full pipeline with Regime Gate ON:
   ```bash
   uv run python scripts/run_full_pipeline.py --top-n 200 --n-jobs 24 --regime-gate on
   ```
3. Select the portfolio:
   ```bash
   uv run python scripts/select_v3_3_portfolio.py
   ```
