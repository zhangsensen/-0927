# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ETF Rotation Strategy Research Platform** — A-share + QDII markets. Three-tier engine (WFO → VEC → BT) screens factor combinations, backtests them, and produces sealed production strategies. Current production: **v8.0** with FREQ=5, Exp4 hysteresis enabled, ICIR-weighted scoring, and stateful signal generation.

## Environment & Commands

**Package manager: UV only.** Never use `pip install`, `python -m venv`, or bare `python <script>`.

```bash
uv sync --dev                    # Install all dependencies

# Makefile shortcuts
make wfo                         # WFO screening (~2min)
make vec                         # VEC backtesting (~5min)
make bt                          # BT audit (~30-60min)
make pipeline                    # Full WFO → VEC → BT pipeline (~76s)
make all                         # wfo + vec + bt
make format                      # black + isort
make lint                        # ruff + mypy
make check                       # pre-commit --all-files
make test                        # pytest -v
make test-cov                    # pytest with coverage
make clean-numba                 # Clear Numba JIT cache (required after @njit signature changes)

# Direct commands
uv run python src/etf_strategy/run_combo_wfo.py           # WFO screening
uv run python scripts/batch_vec_backtest.py                # VEC backtesting
uv run python scripts/batch_bt_backtest.py                 # BT ground truth audit
uv run python scripts/run_full_pipeline.py                 # Full pipeline
uv run python scripts/generate_today_signal.py             # Daily trading signal (stateful)
uv run python scripts/update_daily_from_qmt_bridge.py --all  # Data update from QMT

# Testing
uv run pytest tests/ -v                                    # All tests (210 cases)
uv run pytest tests/test_frozen_params.py -v               # Single file
uv run pytest -k "test_shift" -v                           # Single test by name
```

## Three-Tier Engine Architecture

```
WFO (screening)  →  VEC (precision)  →  BT (ground truth)
~2 min               ~5 min              ~30-60 min
IC gate + scoring    Numba JIT kernel    Backtrader event-driven
```

- **WFO** (`src/etf_strategy/run_combo_wfo.py`): Screens 12,597 factor combinations (sizes 2-7) using rolling IC as gate (≥0.05 or ≥55% positive rate), then ranks by composite score: Return(40%) + Sharpe(30%) + MaxDD(30%). IC alone has only 0.0319 correlation with actual returns — never rank by IC.
- **VEC** (`scripts/batch_vec_backtest.py`): Numba-accelerated vectorized backtest for top candidates. Fast but uses float shares.
- **BT** (`scripts/batch_bt_backtest.py`): Backtrader event-driven simulation with integer lots and capital constraints. Production ground truth.
- **Validation**: Rolling OOS (≥60% positive windows) + Holdout (return > 0) via `final_triple_validation.py`.

## Production Parameters (v8.0)

Enforced by `src/etf_strategy/core/frozen_params.py` — validated at WFO/VEC/BT entry points. Override with `FROZEN_PARAMS_MODE=warn` for A/B testing only.

| Parameter | Value | Notes |
|-----------|-------|-------|
| `FREQ` | 5 | Rebalance every 5 trading days |
| `POS_SIZE` | 2 | Hold 2 ETFs |
| `COMMISSION` | 0.0002 (2bp) | |
| `LOOKBACK` | 252 | 1 year |
| `delta_rank` | 0.10 | Hysteresis: min rank01 gap for swap |
| `min_hold_days` | 9 | Hysteresis: min holding period |
| ETF pool | 49 (41 A-share + 8 QDII) | |
| Universe mode | `A_SHARE_ONLY` | QDII hard-blocked from live trading |

**Version registry**: v3.4/v4.0/v4.1 preserved with freq=3 and hysteresis disabled for rollback. `CURRENT_VERSION = "v8.0"`.

**v8.0 strategies**:
- Champion `composite_1`: `ADX_14D + BREAKOUT_20D + MARGIN_BUY_RATIO + PRICE_POSITION_120D + SHARE_CHG_5D` (5F)
- Fallback `core_4f`: `MARGIN_CHG_10D + PRICE_POSITION_120D + SHARE_CHG_20D + SLOPE_20D` (4F)

**8 QDII ETFs** (monitored but not traded in A_SHARE_ONLY mode):
159920 (恒生ETF), 513050 (中概互联网ETF), 513100 (纳指ETF), 513130 (恒生科技ETF), 513180 (恒生科技指数ETF), 513400 (道琼斯ETF), 513500 (标普500ETF), 513520 (日经ETF)

## Hysteresis State Machine

`src/etf_strategy/core/hysteresis.py` — `@njit` kernel shared by WFO/VEC/BT.

Rules:
1. Max 1 swap per rebalance (forced)
2. Swap only if rank_gap ≥ delta_rank (0.10)
3. Swap only if held_days ≥ min_hold_days (9)

**Signal generator state persistence**: `data/live/signal_state.json` stores per-strategy portfolios and hold_days between daily runs. Schema includes `version`, `freq`, `universe_mode` for env-mismatch detection (auto cold-start if validation fails). State only updates on rebalance days.

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
| Bounded factor sync | Hardcode bounded list in multiple files | Define once in `factor_registry.py` FACTOR_SPECS |
| Regime gate duplication | Apply via both timing_arr AND vol_regime | Apply via timing_arr only |
| IC lookahead | `.shift(-1)` on returns in WFO | IC calc handles t-1 internally |
| Set iteration | `for x in my_set` | `for x in sorted(my_set)` |
| Float comparison | `a == b` | `abs(a - b) < 1e-6` |
| Numba argsort | Default unstable sort for ties | Use `stable_topk_indices()` |
| Numba cache trap | Change `@njit` signature without clearing cache | Run `make clean-numba` |
| Late-IPO ETF NaN | Crash on NaN prices beyond lookback | `.ffill().fillna(1.0)` — NaN factor scores prevent selection |
| BT sizing commission | Use a_share rate for QDII positions | Pass `max(a_share, qdii)` rate |
| State file corruption | Silently use stale/wrong-env state | Validated on load; cold-start on mismatch |
| VEC hysteresis omission | VEC batch without `delta_rank`/`min_hold_days` | Always pass hysteresis params from config — without them results are meaningless |

## Bounded Factors (NO winsorization)

**Single source of truth**: `src/etf_strategy/core/factor_registry.py` → `FACTOR_SPECS` with `is_bounded=True`. Both `CrossSectionProcessor.BOUNDED_FACTORS` and `FrozenCrossSectionParams.bounded_factors` derive from this registry. The `cross_section.bounded_factors` key was removed from `combo_wfo_config.yaml` — it's now code-driven.

Currently bounded (rank-standardized to [-0.5, 0.5], NOT Winsorized):
```
ADX_14D [0,100], CMF_20D [-1,1], CORRELATION_TO_MARKET_20D [-1,1],
PRICE_POSITION_20D [0,1], PRICE_POSITION_120D [0,1], PV_CORR_20D [-1,1], RSI_14 [0,100]
```

To add a new bounded factor: add `FactorSpec(..., is_bounded=True, bounds=(...))` to `FACTOR_SPECS`.

## VEC/BT Alignment

- **Systemic gap**: ~1-2pp after Exp4.1→exec-side fix (float shares vs integer lots). Pre-fix gap was +25pp due to signal-side hysteresis feedback loop.
- **Red flag**: > 10pp → STOP and investigate
- **Hysteresis state**: BT MUST use execution-side state (`shadow_holdings`/`_hold_days`) for hmask, NOT signal-side (`_signal_portfolio`/`_signal_hold_days`). Signal-side creates self-referential feedback loop causing chain divergence.

## Regime Gate

Volatility-based exposure scaling using 510300 (A-share proxy). Config: `regime_gate.enabled: true`.
- Thresholds 25/30/40 pct → exposures 1.0/0.7/0.4/0.1
- A/B tested (100k combos): Gate ON improves Sharpe for 71.5%, reduces drawdown for 86.3%
- **Production: Gate ON** across all frozen versions (v3.4 through v8.0)

## Data Source

OHLCV data from QMT Trading Terminal via `qmt-data-bridge` SDK. Use `QMTClient` from `qmt_bridge` — never construct manual HTTP requests. Data cached in pickle with mtime-based invalidation.

## Module Independence

- `src/etf_data/` — standalone data download tool, NOT part of strategy pipeline
- `src/etf_strategy/` — core strategy module (depends on `raw/ETF/daily/` parquet files)
- `scripts/` — operational scripts that orchestrate the pipeline

## Development Rules

**Allowed**: Bug fixes (no logic change), data adaptation, documentation, performance optimization (same results)

**Prohibited**: Modifying core factor library, changing backtest engine logic, changing locked parameters, removing QDII ETFs, deleting ARCHIVE files, creating "simplified/backup" scripts

## Signal Evaluation Principle (v5.0+, applies to v8.0)

**Any new signal/factor must be evaluated under the production execution framework (FREQ=5 + Exp4 hysteresis + regime gate), otherwise it is not a valid production candidate.** The legacy pipeline mode (F3_OFF) is permitted only as a research reference and must never directly drive go-live decisions.

Rationale (verified 2026-02-11):
- Cross-sectional return PCA: PC1 median explained ratio = 59.8% — the A-share ETF universe is dominated by a single factor (sector/theme momentum)
- S1's 4 factors have moderate IC correlation (avg 0.31) and moderate cross-sectional overlap (avg 0.24) — they are not fully redundant but share the same dominant signal
- Factor space effective dimensionality (Kaiser): 5 out of 17 — most factors are redundant
- v3.4→v5.0 evidence: same signal (S1), execution-only improvement yielded +35.8pp holdout return; switching signals under same execution destroyed value
- Execution delivers 3.6x return multiplier vs signal improvement's 1.25x — always prioritize execution compatibility over signal quality
- Factor rank stability determines Exp4 compatibility: stable-rank factors (ADX, SLOPE, CMF) thrive; volatile-rank factors (PV_CORR, PP_20D) collapse
- OHLCV-derived factor space is near-saturated (Kaiser dimension 5/17) — genuine alpha requires new information sources (IOPV, FX), not recombinations

## Config

- **Engine params**: `configs/combo_wfo_config.yaml` — 49 ETFs, 23 active factors (17 OHLCV + 6 non-OHLCV), hysteresis, regime gate, cost model.
- **Factor metadata**: `src/etf_strategy/core/factor_registry.py` — bounded/unbounded classification, data source, bounds. Single source of truth for all factor metadata.
- **Non-OHLCV factors**: Can be computed inline (pass `loader` to `FactorCache.get_or_compute()`) or from pre-computed parquet (legacy fallback).

## Code Style

- Python 3.11+, black (88 chars), isort (black profile), ruff, mypy strict
- pytest for testing (6 test files, 157 cases covering frozen_params, rebalance utilities, OBV alignment, cost model, execution model, factor mining)
