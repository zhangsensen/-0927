# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ETF Rotation Strategy Research Platform** — A-share + QDII markets (A股 + QDII). Three-tier engine (WFO → VEC → BT) screens factor combinations, backtests them, and produces sealed production strategies. Current production: v3.4 with 2 live strategies.

## Environment & Commands

**Package manager: UV only.** Never use `pip install`, `python -m venv`, or bare `python <script>`.

```bash
uv sync --dev                    # Install all dependencies

# Makefile shortcuts (all functional)
make wfo                         # WFO screening (~2min)
make vec                         # VEC backtesting (~5min)
make bt                          # BT audit (~30-60min)
make pipeline                    # Full WFO → VEC → BT pipeline
make all                         # wfo + vec + bt
make format                      # black + isort
make lint                        # ruff + mypy
make check                       # pre-commit --all-files
make test                        # pytest -v
make test-cov                    # pytest with coverage

# Direct commands
uv run python src/etf_strategy/run_combo_wfo.py           # WFO screening
uv run python scripts/batch_vec_backtest.py                # VEC backtesting
uv run python scripts/final_triple_validation.py           # Rolling OOS + holdout validation
uv run python scripts/batch_bt_backtest.py                 # BT ground truth audit
uv run python scripts/run_full_pipeline.py                 # Full pipeline
uv run python scripts/generate_today_signal.py             # Daily trading signal
uv run python scripts/update_daily_from_qmt_bridge.py --all  # Data update from QMT

# Testing
uv run pytest tests/ -v                                    # All tests (55 cases)
uv run pytest tests/test_vec_bt_alignment.py -v            # Single file
uv run pytest -k "test_shift" -v                           # Single test by name
```

## Three-Tier Engine Architecture

```
WFO (screening)  →  VEC (precision)  →  BT (ground truth)
~2 min               ~5 min              ~30-60 min
IC gate + scoring    Numba JIT kernel    Backtrader event-driven
```

- **WFO**: Screens 12,597 factor combinations (sizes 2-7) using rolling IC as gate (≥0.05 or ≥55% positive rate), then ranks by composite score: Return(40%) + Sharpe(30%) + MaxDD(30%). IC alone has only 0.0319 correlation with actual returns — never rank by IC.
- **VEC**: Numba-accelerated vectorized backtest for top candidates. Fast but uses float shares.
- **BT**: Backtrader event-driven simulation with integer lots and capital constraints. Production ground truth.
- **Validation**: Rolling OOS (≥60% positive windows) + Holdout (return > 0) via `final_triple_validation.py`.

## Locked Parameters (NEVER modify)

| Parameter | Value | Enforced by |
|-----------|-------|-------------|
| `FREQ` | 3 | `frozen_params.py` |
| `POS_SIZE` | 2 | `frozen_params.py` |
| `COMMISSION` | 0.0002 (2bp) | `frozen_params.py` |
| `LOOKBACK` | 252 | `frozen_params.py` |
| ETF pool | 43 (38 A-share + 5 QDII) | `frozen_params.py` |

**5 QDII ETFs are the core alpha source (~90% of returns) — NEVER remove:**
513100 (Nasdaq), 513500 (S&P), 159920 (HSI), 513050 (China Internet), 513130 (HK Tech)

`frozen_params.py` validates config against hardcoded values at WFO/VEC/BT entry points. Override with `FROZEN_PARAMS_MODE=warn` env var (for A/B testing only).

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
| Lookahead bias | Same-day signal for same-day trade | `shift_timing_signal()` to lag 1 day |
| Rebalance misalignment | VEC/BT use different rebalance days | `generate_rebalance_schedule()` |
| Bounded factor winsorization | Winsorize RSI, ADX, PRICE_POSITION, CMF, CORR_MKT, PV_CORR | Skip — these are naturally bounded |
| Regime gate duplication | Apply via both timing_arr AND vol_regime | Apply via timing_arr only |
| IC lookahead | `.shift(-1)` on returns in WFO | IC calc handles t-1 internally |
| Set iteration | `for x in my_set` | `for x in sorted(my_set)` |
| Float comparison | `a == b` | `abs(a - b) < 1e-6` |
| Numba argsort | Default unstable sort for ties | Use `stable_topk_indices()` |
| Late-IPO ETF NaN | Crash on NaN prices beyond lookback | `.ffill().fillna(1.0)` — NaN factor scores prevent selection |

## Bounded Factors (NO winsorization)

Defined in `cross_section_processor.py`. These are naturally bounded and must NOT be winsorized:

```
ADX_14D [0,100], CMF_20D [-1,1], CORRELATION_TO_MARKET_20D [-1,1],
PRICE_POSITION_20D [0,1], PRICE_POSITION_120D [0,1], PV_CORR_20D [-1,1], RSI_14 [0,100]
```

## VEC/BT Alignment

- **Systemic gap**: Median ~4.8pp (execution model differences — float shares vs integer lots, not logic bugs)
- **Red flag**: > 20pp → STOP and investigate
- **OBV_SLOPE_10D**: Previously showed 61pp drift (fixed — was normalization difference, not computation error)

## Regime Gate

Volatility-based exposure scaling using 510300 (A-share proxy). Config: `regime_gate.enabled: false`.
- Thresholds 25/30/40 pct → exposures 1.0/0.7/0.4/0.1
- A/B tested (100k combos): Gate ON improves Sharpe for 71.5% of all strategies, reduces drawdown for 86.3%
- **Production decision: Gate OFF** — top candidates already contain risk-adjusted factors (CORR_MKT, MAX_DD, SHARPE), self-hedging makes external gate redundant. Gate is insurance for factor combos without built-in risk factors.

## Data Source

OHLCV data from QMT Trading Terminal via `qmt-data-bridge` SDK. Use `QMTClient` from `qmt_bridge` — never construct manual HTTP requests. Data cached in pickle with mtime-based invalidation (cache key includes file modification time).

## Module Independence

- `src/etf_data/` — standalone data download tool, NOT part of strategy pipeline
- `src/etf_strategy/` — core strategy module (depends on `raw/ETF/daily/` parquet files)
- `scripts/` — operational scripts that orchestrate the pipeline

## Development Rules

**Allowed**: Bug fixes (no logic change), data adaptation, documentation, performance optimization (same results)

**Prohibited**: Modifying core factor library, changing backtest engine logic, changing locked parameters, removing QDII ETFs, deleting ARCHIVE files, creating "simplified/backup" scripts

## Config

Single source of truth: `configs/combo_wfo_config.yaml` — 43 ETFs, 25 factors, all engine parameters.

## Code Style

- Python 3.11+, black (88 chars), isort (black profile), ruff, mypy strict
- pytest for testing (3 test files, 55 cases covering frozen_params, rebalance utilities, OBV alignment)
