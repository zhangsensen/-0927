# v6.0 Reproduce Steps

```bash
cd /home/sensen/dev/projects/-0927

# 1. Precompute non-OHLCV factors
uv run python scripts/precompute_non_ohlcv_factors.py

# 2. WFO screening (hysteresis-aware)
uv run python src/etf_strategy/run_combo_wfo.py

# 3. VEC holdout validation (C2)
uv run python scripts/batch_vec_backtest.py

# 4. BT ground truth (C2)
uv run python scripts/batch_bt_backtest.py

# 5. Daily signal generation
uv run python scripts/generate_today_signal.py

# 6. Shadow signal generation
uv run python scripts/generate_today_signal.py --shadow-config configs/shadow_strategies.yaml
```

## Key Configuration

- Config: `configs/combo_wfo_config.yaml` (49 ETFs, 24 active factors, 7 bounded_factors)
- Frozen: `src/etf_strategy/core/frozen_params.py` (v6.0 with C2 strategy)
- Hysteresis: `src/etf_strategy/core/hysteresis.py` (dr=0.10, mh=9)
