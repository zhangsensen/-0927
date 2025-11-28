# Standalone ETF Rotation Strategy

This project contains the stable, production-ready implementation of the 18-factor ETF rotation strategy.
It is designed to be simple, robust, and free of the complex experimental features (like WFO or complex risk controls) that were causing instability.

## Key Features

- **18-Factor Model**: Uses a diverse set of 18 technical and fundamental factors.
- **Vectorized Backtester**: High-performance Numba-accelerated backtesting engine.
- **Light Timing**: Simple Moving Average + Momentum + Gold correlation timing model.
- **RORO (Risk-On/Risk-Off)**: Automatically switches to Gold (`518880`) when market timing is bearish, instead of holding cash.
- **No Complex Risk Control**: Pure rotation logic without stop-loss/take-profit interference.
- **Production Ready**: Clean code structure, ready for deployment.

## Project Structure

```
etf_rotation_standalone/
├── configs/               # Configuration files
│   ├── config.yaml        # Main strategy configuration
│   └── etf_pools.yaml     # ETF universe definition
├── core/                  # Core logic
│   ├── backtester_vectorized.py  # Backtest engine
│   ├── cross_section_processor.py # Factor processing
│   ├── data_contract.py   # Data validation
│   ├── data_loader.py     # Data loading
│   ├── market_timing.py   # Timing logic
│   ├── precise_factor_library.py # Factor definitions
│   └── utils/             # Helper functions
├── results/               # Backtest results
└── run_strategy.py        # Main execution script
```

## How to Run

1. **Install Dependencies** (if not already installed):
   ```bash
   uv sync --dev
   ```

2. **Step 1: Factor Selection (WFO)**
   Run the WFO engine to find the best factor combinations based on IC/ICIR:
   ```bash
   uv run python etf_rotation_standalone/run_wfo.py
   ```
   This will output the top ranked combinations to the console and `results_wfo/`.

3. **Step 2: Configure Strategy**
   Open `etf_rotation_standalone/configs/config.yaml` and update the `selected_factors` list with your chosen combination from Step 1.
   ```yaml
   strategy:
     selected_factors:
       - ADX_14D
       - CALMAR_RATIO_60D
       # ...
   ```

4. **Step 3: Run Backtest**
   Execute the strategy with the selected factors:
   ```bash
   uv run python etf_rotation_standalone/run_strategy.py
   ```

5. **Check Results**:
   Results will be printed to the console and saved to `etf_rotation_standalone/results/`.

## Configuration

- Edit `configs/config.yaml` to change backtest parameters (capital, commission, etc.) or data paths.
- Edit `configs/etf_pools.yaml` to modify the ETF universe.

## Strategy Logic

1. **Data Loading**: Loads OHLCV data for the specified ETF universe.
2. **Factor Calculation**: Computes 18 factors for each ETF.
3. **Processing**: Standardizes factors (Z-score) and handles outliers (Winsorization).
4. **Ranking**: Ranks ETFs based on the equal-weighted sum of all factors.
5. **Timing**: Applies a global market timing filter (Light Timing) to adjust position sizes.
6. **Execution**: Rebalances every 8 days, holding the top 3 ETFs.
