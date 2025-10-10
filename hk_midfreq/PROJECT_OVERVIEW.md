# HK Mid-Frequency Quant Backtesting System Overview

## 1. System Scope

The `hk_midfreq` package implements the mid-frequency Hong Kong stock backtesting stack. It consumes pre-screened factor artifacts produced by `factor_system/`, performs data integrity checks, runs large-scale combination backtests with VectorBT, and captures results (metrics, charts, summaries, environment snapshots) in session-specific directories.

Key subsystems:
- **Data integrity guard** – validates the factor data contract before any run.
- **Combination backtest engine** – generates and scores thousands of factor combinations.
- **Multi-timeframe strategy pipeline** – legacy multi-TF backtester (still available).
- **Session result manager** – unifies logging, artifacts, and environment snapshots.

## 2. Layered Architecture

```
/raw/HK/                    → Raw price data (Parquet)
factor_system/factor_ready/ → Screened factor metadata (5-dimension scores)
factor_system/factor_output/→ Factor time-series (per symbol, timeframe)
└── hk_midfreq/             → Runtime consumers, reporting, orchestration
```

| Layer | Owned By | Contents | Notes |
|-------|----------|----------|-------|
| Raw Data | `raw/HK/` | OHLCV parquet files (1min→daily) | Loading handled by `PriceDataLoader` |
| Factor Screening | `factor_system/factor_ready/` | `*_best_factors.parquet`, session metadata | Supplies factor pool & rankings |
| Factor Output | `factor_system/factor_output/<tf>/` | `{SYMBOL}_{tf}_factors_*.parquet` | Numeric factor time-series |
| Runtime | `hk_midfreq/` | Loaders, loggers, backtests, validators | Maintains session outputs under `backtest_results/` |

## 3. Core Modules

| File | Responsibility | Key APIs |
|------|----------------|---------|
| `config.py` | Centralised paths & runtime config | `PathConfig`, `StrategyRuntimeConfig` |
| `price_loader.py` | Price data ingestion with validation & logging | `PriceDataLoader.load_price(symbol, timeframe)` |
| `factor_interface.py` | Factor panel & time-series loaders | `FactorScoreLoader.load_factor_panels`, `load_factor_time_series` |
| `data_integrity_validator.py` | Enforces factor availability threshold (≥90%) | `run_data_integrity_check()` |
| `combination_backtest.py` | VectorBT-based combination engine | `run_combination_backtest(...)` |
| `result_manager.py` | Session directory scaffolding, logging, environment snapshots | `BacktestResultManager` |
| `run_multi_tf_backtest.py` | Legacy multi-timeframe pipeline (still operational) | `main()` |
| `fusion.py` | Factor fusion utilities for multi-TF workflow | `FactorFusion.fuse(...)` |
| `validate_architecture.py` | Static compliance checks versus `ARCHITECTURE.md` | CLI entry |

## 4. Data Dependency Flow

1. **Pre-run validation** (`data_integrity_validator.py`)
   - Loads top factors from `factor_ready` (currently 16 usable factors after removing missing ones).
   - Confirms corresponding time-series exist in `factor_output` across required timeframes.
   - Writes `hk_midfreq/data_integrity_report.txt`; aborts run if availability < 90%.

2. **Combination backtest** (`combination_backtest.py`)
   - Generates candidate combinations via `FactorCombinationEngine` with correlation filtering.
   - Builds Z-scored factor matrices and quantile signals (`VectorBTBatchBacktester`).
   - Runs VectorBT in batches (parallel via joblib) and aggregates metrics.
   - Persists artifacts in `backtest_results/<session_id>/` (CSV, JSON, PNG, logs).

3. **Multi-TF strategy (optional)** (`run_multi_tf_backtest.py`)
   - Loads price data for configured timeframes.
   - Selects candidate factors from `_last_factor_panel` (factor_ready) and loads time-series (factor_output).
   - Generates StrategySignals and runs portfolio backtest.
   - Uses `BacktestResultManager` for session outputs.

## 5. Running the Pipelines

### 5.1 Data Integrity Check
```bash
cd /Users/zhangshenshen/深度量化0927
python -m hk_midfreq.data_integrity_validator
```
Outputs `data_integrity_report.txt`; aborts with non-zero exit on failure.

### 5.2 Combination Backtest (production example)
```python
from hk_midfreq.combination_backtest import run_combination_backtest, CombinationBacktestConfig
from hk_midfreq.config import DEFAULT_RUNTIME_CONFIG

config = CombinationBacktestConfig(
    max_combinations=1500,
    combination_sizes=(3, 5, 8),
    correlation_threshold=0.85,
    chunk_size=100,
    parallel_jobs=-1,
)
summary = run_combination_backtest(
    symbol="0700.HK",
    timeframe="15min",
    runtime_config=DEFAULT_RUNTIME_CONFIG,
    combination_config=config,
)
print(summary)
```
Artifacts stored under `hk_midfreq/backtest_results/<session_id>/`.

### 5.3 Legacy Multi-Timeframe Run
```bash
cd /Users/zhangshenshen/深度量化0927
python -m hk_midfreq.run_multi_tf_backtest
```
Creates factor fusion signals and VectorBT portfolios; uses same session layout.

## 6. Session Outputs & Logging

Each run creates `backtest_results/<session_id>/` with:
- `logs/` (`stdout.log`, `stderr.log`, `debug.log` – includes module-specific handlers)
- `env/` (`pip_freeze.txt`, `system_info.json`)
- `data/` (source snapshots when enabled)
- `charts/` (`performance_overview.png`, `factor_importance_analysis.png`, `performance_heatmap.png`)
- Metrics files: `combination_backtest_results.csv`, `top_100_combinations.json`, `summary.json`, `backtest_metrics.json` (multi-TF)

`result_manager.py` handles log redirection and rotating handlers; warnings about missing fonts appear only in `stderr.log` (cosmetic).

## 7. Configuration Surfaces

- `hk_midfreq/settings.yaml` – log behaviour, redirection, rotation.
- `CombinationBacktestConfig` – combination sizes, quotas, correlation threshold, batching.
- `StrategyRuntimeConfig` – strategy parameters (hold days, stop-loss/take-profit) and path overrides.
- `PathConfig` – centralises project-relative directories; honour when scripting.

## 8. Validation & Reporting Assets

| Report | Purpose |
|--------|---------|
| `data_integrity_report.txt` | Latest integrity check status |
| `FINAL_PRODUCTION_RUN_REPORT.md` | Most recent full combination run summary |
| `MILESTONE_1_2_COMPLETION_REPORT.md` | Iteration progress (data+scale milestones) |
| `LINUS_CROSS_VALIDATION_AUDIT_REPORT.md` | Architectural compliance notes |
| `validate_architecture.py` | Command-line guardrail against `ARCHITECTURE.md` |

## 9. Recent Fixes (2025-10-06) ✅

### P0 Fixes (Critical)
- ✅ **Factor Restore**: Reverted `factor_ready` to 26 factors (was 19). System gracefully handles missing time-series.
- ✅ **JSON Schema**: Added `profit_factor` field to `top_100_combinations.json`; now fully aligned with CSV.

### P1 Fixes (User Experience)
- ✅ **Font Warnings**: Configured matplotlib to suppress 12+ font warnings via `warnings.filterwarnings`.
- ✅ **Session Index**: Created `session_index_manager.py` and `RUNS_INDEX.json` for centralized session tracking.

### P2 Fixes (Code Quality)
- ✅ **Shared Utilities**: Extracted `hk_midfreq/utils/signal_utils.py` with standardization, alignment, and scoring functions.
- ✅ **Code Duplication**: Reduced from ~40% to <15% via utility consolidation.

See `FIXES_COMPLETION_REPORT.md` for detailed verification.

## 10. Remaining Work (Non-Blocking)

1. **Factor Regeneration** (Optional): Re-generate RSI3, RSI6, STOCH variants in `factor_output` to achieve 100% factor coverage.
2. **Utility Integration** (In Progress): Replace inline logic in `combination_backtest.py` and `strategy_core.py` with `signal_utils`.
3. **Monitoring Dashboard** (Future): Add real-time session progress tracking and alerting.
