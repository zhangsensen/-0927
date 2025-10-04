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
| [`config.py`](./config.py) | Dataclasses that store trading, execution, and runtime toggles. |
| [`factor_interface.py`](./factor_interface.py) | Utilities that read the screener output directory and aggregate factor scores per symbol/timeframe. |
| [`strategy_core.py`](./strategy_core.py) | Candidate selection and signal generation (RSI + Bollinger + volume reversal). |
| [`backtest_engine.py`](./backtest_engine.py) | `vectorbt`-based engines for single-asset and portfolio-level simulations. |
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
scores for each symbol. Market OHLCV data should be supplied separately (e.g.,
via `MultiTimeframeFactorStore` or another loader in your notebooks/scripts).

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

1. **Discover scores** – Use `FactorScoreLoader` to read the latest screening
   session and convert the scores into a ranked `Series`.
2. **Select candidates** – Feed the scores into `StrategyCore.select_candidates`
   (optionally with universe constraints) to choose the top symbols.
3. **Generate signals** – Run `StrategyCore.build_signals_for_symbol` for each
   symbol with the corresponding OHLCV data to obtain `entries`/`exits`.
4. **Backtest** – Call `run_portfolio_backtest` with a dictionary of per-symbol
   close prices and signals; obtain a `BacktestArtifacts` bundle containing the
   `vectorbt.Portfolio` plus metadata.
5. **Review** – Use functions from `review_tools.py` to print statistics, inspect
   trades, or render charts.

---

## 6. End-to-End Example

```python
from pathlib import Path

import pandas as pd
import vectorbt as vbt

from hk_midfreq import (
    BacktestArtifacts,
    FactorScoreLoader,
    StrategyCore,
    run_portfolio_backtest,
)

# 1) Load factor scores (auto-detects latest session under 因子筛选/)
score_loader = FactorScoreLoader()
factor_scores = score_loader.load_scores_as_series(["0700.HK", "9988.HK", "3690.HK"])

# 2) Choose candidates
core = StrategyCore()
selected = core.select_candidates(factor_scores, top_n=core.trading_config.max_positions)

# 3) Prepare price data (replace with your actual loader)
price_data: dict[str, pd.Series] = {
    symbol: load_close_series(symbol)  # user-defined helper
    for symbol in selected
}
volume_data: dict[str, pd.Series] = {
    symbol: load_volume_series(symbol)
    for symbol in selected
}

signals = {}
for symbol in selected:
    close = price_data[symbol]
    volume = volume_data[symbol]
    signals[symbol] = core.build_signals_for_symbol(close=close, volume=volume)

# 4) Run portfolio backtest
artifacts: BacktestArtifacts = run_portfolio_backtest(price_data, signals)
portfolio = artifacts.portfolio

# 5) Inspect results
portfolio.stats()
portfolio.trades.records_readable.head()
portfolio.plot().show()
```

Replace `load_close_series` / `load_volume_series` with the appropriate data
retrieval functions (e.g., `MultiTimeframeFactorStore`). All portfolio sizing
and friction rules default to the values declared in `config.py`.

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

