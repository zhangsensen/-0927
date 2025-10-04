# HK Mid-Frequency Strategy Stack

This document explains the purpose, architecture, and day-to-day usage of the
`hk_midfreq` package. It is designed to help you wire the existing factor
screening outputs into a modular **signal → backtest → review** workflow powered
by `vectorbt`.

---

## 1. Package Goals

- Provide a thin integration layer between the professional factor screener and
  live strategy research.
- Encapsulate the baseline reversal strategy logic so you can iterate without
  modifying notebooks or ad-hoc scripts.
- Supply reproducible backtesting helpers and reporting tools for multi-symbol
  portfolios.

---

## 2. Module Map

| Module | Responsibility |
| --- | --- |
| [`config.py`](./config.py) | Dataclasses that store trading, execution, runtime toggles, and fusion rules. |
| [`factor_interface.py`](./factor_interface.py) | Utilities that read the screener output directory and build factor panels per symbol/timeframe. |
| [`fusion.py`](./fusion.py) | Helper that fuses factor panels into composite scores across timeframes. |
| [`strategy_core.py`](./strategy_core.py) | Candidate selection and signal generation (RSI + Bollinger + volume reversal) with multi-timeframe coordination. |
| [`backtest_engine.py`](./backtest_engine.py) | `vectorbt`-based engines for single-asset and portfolio-level simulations supporting multi-frequency inputs. |
| [`review_tools.py`](./review_tools.py) | Convenience helpers to inspect stats, trades, and charts. |
| [`__init__.py`](./__init__.py) | Public exports for downstream scripts and notebooks. |
| [`DEVELOPMENT_PLAN.md`](./DEVELOPMENT_PLAN.md) | Roadmap that tracks open tasks and milestones. |

All modules are importable via `import hk_midfreq as hkm` for a consistent API
surface.

---

## 3. Data & Directory Expectations

The loader assumes you have already run `ProfessionalFactorScreener` and that
results live under the directory defined by
`StrategyRuntimeConfig.base_output_dir` (defaults to `因子筛选/`). Each screening
session is expected to match the folder layout produced by
`EnhancedResultManager`:

```
因子筛选/
  └── <session_id>/
        ├── screening_statistics.json
        ├── executive_summary.txt
        └── timeframes/
              ├── 0700.HK_60min/
              │     └── top_factors_detailed.json
              └── ...
```

The strategy layer consumes `top_factors_detailed.json` to compute aggregate
scores for each symbol. `FactorScoreLoader.load_factor_panels` returns a
MultiIndex DataFrame keyed by `(symbol, timeframe, factor_name)` with flattened
factor metrics. Market OHLCV data should be supplied separately (e.g., via
`MultiTimeframeFactorStore`) as a dictionary of timeframes per symbol.

---

## 4. Configuration Overview

Instantiate a configuration to override defaults when needed:

```python
from hk_midfreq.config import TradingConfig, ExecutionConfig, StrategyRuntimeConfig

trading_cfg = TradingConfig(capital=2_000_000, max_positions=10)
execution_cfg = ExecutionConfig(transaction_cost=0.0045, stop_loss=0.02)
runtime_cfg = StrategyRuntimeConfig(base_output_dir=Path("/path/to/因子筛选"))
```

Pass these objects into other modules if you need bespoke behavior. Without
arguments, the defaults encoded in
`DEFAULT_TRADING_CONFIG` / `DEFAULT_EXECUTION_CONFIG` /
`DEFAULT_RUNTIME_CONFIG` are used.

---

## 5. Typical Workflow

1. **Discover scores** – Use `FactorScoreLoader.load_factor_panels` to read the
   latest screening session into per-symbol, per-timeframe factor panels.
2. **Fuse composites** – Run `FactorFusionEngine.fuse` (automatically invoked by
   `StrategyCore.select_candidates`) to produce composite scores that respect
   the runtime fusion rules (daily trend, hourly confirmation, intraday timing).
3. **Select candidates** – Call `StrategyCore.select_candidates` with the target
   universe; the method now pulls all required timeframes, fuses them, and ranks
   symbols by the composite score.
4. **Generate signals** – Provide nested OHLCV data to
   `StrategyCore.build_signal_universe`. The core checks daily trend and hourly
   confirmation before emitting intraday entry/exit signals.
5. **Backtest** – Call `run_portfolio_backtest` with a dictionary of
   `{symbol: {timeframe: ohlcv_df}}` plus the generated signals; obtain a
   `BacktestArtifacts` bundle containing the `vectorbt.Portfolio` plus metadata.
6. **Review** – Use functions from `review_tools.py` to print statistics, inspect
   trades, or render charts.

---

## 6. End-to-End Example

```python
from pathlib import Path

import pandas as pd
import vectorbt as vbt

from hk_midfreq import (
    BacktestArtifacts,
    FactorFusionEngine,
    FactorScoreLoader,
    StrategyCore,
    run_portfolio_backtest,
)

# 1) Load factor panels (auto-detects latest session under 因子筛选/)
score_loader = FactorScoreLoader()
panels = score_loader.load_factor_panels(
    symbols=["0700.HK", "9988.HK", "3690.HK"],
    timeframes=("daily", "60min", "15min", "5min"),
)

# 2) Fuse and rank (optional manual step; StrategyCore does this internally)
fusion = FactorFusionEngine()
fused = fusion.fuse(panels)

# 3) Prepare multi-timeframe price data (replace with your actual loader)
price_data: dict[str, dict[str, pd.DataFrame]] = {}
for symbol in fused.index:
    price_data[symbol] = {
        "daily": load_daily_ohlcv(symbol),
        "60min": load_hourly_ohlcv(symbol),
        "15min": load_intraday_ohlcv(symbol, "15min"),
        "5min": load_intraday_ohlcv(symbol, "5min"),
    }

# 4) Generate signals with multi-timeframe gating
core = StrategyCore()
signals = core.build_signal_universe(price_data)

# 5) Run portfolio backtest using multi-frequency prices
artifacts: BacktestArtifacts = run_portfolio_backtest(price_data, signals)
portfolio = artifacts.portfolio

# 6) Inspect results
portfolio.stats()
portfolio.trades.records_readable.head()
portfolio.plot().show()
```

Replace the `load_*` helpers with the appropriate data retrieval functions
(e.g., `MultiTimeframeFactorStore`). All portfolio sizing and friction rules
default to the values declared in `config.py`. The backtest engine now detects
the entry timeframe from each `StrategySignals` object and automatically pulls
the matching OHLCV DataFrame from the nested price dictionary.

---

## 7. CLI / Notebook Tips

- When iterating interactively, import the entire namespace via
  `import hk_midfreq as hkm` to access `hkm.FactorScoreLoader`,
  `hkm.StrategyCore`, and `hkm.run_portfolio_backtest`.
- Call `hkm.review_tools.print_summary(portfolio)` or
  `hkm.review_tools.plot_performance(portfolio)` for quick diagnostics.
- Use `StrategyCore.update_configs(...)` (if implemented in future extensions)
  or re-instantiate with custom configs to experiment with alternative position
  sizing or execution assumptions.

---

## 8. Extending the Stack

- **Additional factors**: Implement new aggregation schemes inside
  `FactorScoreLoader.load_symbol_scores` (e.g., IC-weighted average).
- **Alternative strategies**: Subclass `StrategyCore` or add new methods to
  generate different entry/exit rules while reusing the backtest engines.
- **Risk overlays**: Modify `run_portfolio_backtest` to supply custom weights,
  position limits, or stop overlays using the `vectorbt` API.
- **Reporting**: Expand `review_tools` with monthly breakdowns, factor exposure
  charts, or CSV export utilities.

---

## 9. Troubleshooting

| Symptom | Possible Cause | Resolution |
| --- | --- | --- |
| `FileNotFoundError: No screening session directory found.` | `因子筛选/` is empty or lives elsewhere. | Override `StrategyRuntimeConfig.base_output_dir` to point at the correct location. |
| Empty factor score series | Symbols not present in the selected session. | Confirm screener outputs exist or adjust the symbol universe. |
| `ValueError: Unsupported aggregation method` | Invalid `agg` parameter. | Choose `mean` or `max`, or extend the loader with your custom method. |
| Portfolio returns all zeros | Missing entries/exits or misaligned OHLCV indices. | Ensure price/volume series share the same frequency and timezone. |

---

## 10. Related Documents

- [`factor_system/factor_screening/README.md`](../factor_system/factor_screening/README.md)
  for details on the screener output layout.
- [`hk_midfreq/DEVELOPMENT_PLAN.md`](./DEVELOPMENT_PLAN.md) for outstanding tasks
  and roadmap discussions.

