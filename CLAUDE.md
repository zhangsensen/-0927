# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Professional **ETF Rotation Strategy Research Platform** (A-share + QDII markets). Three-tier engine architecture (WFO → VEC → BT) for factor combination screening through backtesting audit. Current production: v3.4 with 2 live strategies.

## Environment & Commands

**Package manager: UV only.** Never use `pip install`, `python -m venv`, or bare `python <script>`.

```bash
# Setup
uv sync --dev                    # Install all dependencies
uv pip install -e .              # Editable install

# All scripts MUST use uv run
uv run python <script.py>

# Code quality
make format                      # black + isort
make check                       # pre-commit hooks (all checks)
make test                        # pytest -v
uv run pytest tests/test_vec_bt_alignment.py -v   # Single test file
uv run pytest -k "test_shift"                      # Single test by name
```

**Note:** `make lint` and `make wfo` reference stale module paths (`factor_system/`, `etf_rotation_optimized/`). Use direct `uv run` commands instead until Makefile is updated.

## Three-Tier Engine Architecture

```
WFO (screening)  →  VEC (precision)  →  BT (ground truth)
~2 min               ~5 min              ~30-60 min
12,597 combos        Numba JIT kernel    Backtrader event-driven
```

**Core workflow:**
```bash
# Step 1: WFO screening (IC-based gate → composite scoring)
uv run python src/etf_strategy/run_combo_wfo.py

# Step 2: VEC precision backtesting
uv run python scripts/batch_vec_backtest.py

# Step 3: Validation (rolling OOS + holdout, 3-gate AND filter)
uv run python scripts/final_triple_validation.py

# Step 4: BT ground truth audit
uv run python scripts/batch_bt_backtest.py

# Full pipeline (all stages):
uv run python scripts/run_full_pipeline.py

# Daily signal generation:
uv run python scripts/generate_today_signal.py

# Data update from QMT Bridge:
uv run python scripts/update_daily_from_qmt_bridge.py --all
```

## Project Structure

```
src/etf_strategy/                # Core strategy module
├── run_combo_wfo.py             # WFO entry point
├── core/                        # Core engines (DO NOT MODIFY)
│   ├── combo_wfo_optimizer.py   # Rolling WFO (180d IS / 60d OOS / 60d step)
│   ├── precise_factor_library_v2.py  # 18 factors (1,616 lines)
│   ├── backtester_vectorized.py # VEC engine (Numba)
│   ├── cross_section_processor.py    # Z-score normalization
│   ├── data_loader.py           # OHLCV loading + pickle cache
│   ├── ic_calculator_numba.py   # Spearman IC (Numba)
│   ├── frozen_params.py         # Production config freezing
│   └── utils/rebalance.py       # CRITICAL shared utilities
├── auditor/core/engine.py       # Backtrader BT strategy
└── regime_gate.py               # Volatility regime detection

scripts/                         # Operational scripts
├── batch_vec_backtest.py        # VEC batch
├── batch_bt_backtest.py         # BT batch audit
├── final_triple_validation.py   # Rolling OOS + holdout validation
├── run_full_pipeline.py         # Full pipeline orchestration
├── generate_today_signal.py     # Daily trading signal
└── generate_production_pack.py  # Sealed release generation

configs/combo_wfo_config.yaml    # Main config (43 ETFs, 18 factors, all params)
sealed_strategies/v3.4_20251216/ # Current production sealed version
```

## Locked Parameters (NEVER modify)

| Parameter | Value | Why |
|-----------|-------|-----|
| `FREQ` | 3 | Rebalance every 3 trading days |
| `POS_SIZE` | 2 | Hold 2 positions |
| `COMMISSION` | 0.0002 | 2bp commission rate |
| `LOOKBACK` | 252 | 1-year lookback window |
| ETF pool | 43 (38 A-share + 5 QDII) | QDII contributes 90%+ returns |

**5 QDII ETFs are the core alpha source — NEVER remove:**
513100 (Nasdaq), 513500 (S&P), 159920 (HSI), 513050 (China Internet), 513130 (HK Tech)

## Required Shared Utilities

All rebalance/timing logic **MUST** use these to maintain VEC/BT alignment:

```python
from etf_strategy.core.utils.rebalance import (
    shift_timing_signal,           # Lag signal 1 day — prevents lookahead
    generate_rebalance_schedule,   # Unified VEC/BT rebalance dates
    ensure_price_views,            # Consistent (close_t-1, open_t, close_t) views
)
```

## Critical Pitfalls

| Pitfall | Wrong | Right |
|---------|-------|-------|
| **Lookahead bias** | Same-day signal for same-day trade | `shift_timing_signal()` to lag 1 day |
| **Rebalance misalignment** | VEC/BT use different rebalance days | `generate_rebalance_schedule()` |
| **Set iteration** | `for x in my_set` (non-deterministic) | `for x in sorted(my_set)` |
| **Float comparison** | `a == b` | `abs(a - b) < 1e-6` or 0.01% tolerance |
| **Numba argsort** | Default unstable sort for tied values | Use `stable_topk_indices()` |
| **Bounded factor winsorization** | Winsorize RSI, ADX, PRICE_POSITION, PV_CORR | These are bounded [0,1] or [0,100] — skip winsorization |

## VEC/BT Alignment

- **Target**: < 0.01pp difference between VEC and BT returns
- **Current average**: ~0.06pp (acceptable; caused by float accumulation, not logic errors)
- **Known exception**: OBV_SLOPE_10D has 61pp drift (acknowledged, still in production)
- **Red flag**: > 0.20pp difference → STOP and investigate

## Development Rules

**Allowed**: Bug fixes (no logic change), data adaptation, documentation, performance optimization (same results), data updates

**Prohibited**: Modifying core factor library, changing backtest engine logic, changing locked parameters, removing QDII ETFs, deleting ARCHIVE files, creating "simplified/backup" scripts

## Data Source

Data comes from QMT Trading Terminal via `qmt-data-bridge` SDK. Use `QMTClient` from `qmt_bridge` — never construct manual HTTP requests.

## Live Strategies (v3.4)

- **Strategy #1**: ADX_14D + OBV_SLOPE_10D + SHARPE_RATIO_20D + SLOPE_20D → 136.52%
- **Strategy #2**: + PRICE_POSITION_120D → 129.85%, lower MaxDD (13.93%)
