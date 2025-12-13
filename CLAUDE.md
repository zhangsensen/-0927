# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a professional **ETF Rotation Strategy Research Platform** that uses a three-tier engine architecture (WFO → VEC → BT) for comprehensive factor mining to backtesting audit. The project implements a high-frequency rotation strategy achieving **237.45% returns** with v3.0 parameters.

## Key Commands

### Environment Setup
```bash
# Install dependencies using UV (required package manager)
uv sync --dev
uv pip install -e .  # Install project in editable mode

# All Python scripts MUST use 'uv run' prefix
uv run python <script.py>
```

### Core Production Workflow
```bash
# Step 1: WFO screening - Filter from 12,597 combinations
uv run python src/etf_strategy/run_combo_wfo.py

# Step 2: VEC recalculation - Full precise backtesting
uv run python scripts/batch_vec_backtest.py

# Step 3: BT audit (optional) - Event-driven audit verification
uv run python scripts/batch_bt_backtest.py

# Data update from QMT Bridge
uv run python scripts/update_daily_from_qmt_bridge.py --all
```

### Code Quality
```bash
make format        # black + isort formatting
make lint          # flake8 + mypy checks
make test          # run pytest tests
```

## Architecture

### Three-Tier Engine Architecture
```
┌─────────────────────────────────────────┐
│  WFO (Screening Layer)                  │
│  - Script: src/etf_strategy/run_combo_wfo.py
│  - Speed: ~2 min for 12,597 combos     │
│  - Output: Top-100 candidates (by IC)   │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  VEC (Recalculation Layer)             │
│  - Script: scripts/batch_vec_backtest.py
│  - Alignment: MUST match BT (< 0.01pp)  │
│  - Output: Precise returns, Sharpe, MDD │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  BT (Audit Layer) - GROUND TRUTH        │
│  - Script: scripts/batch_bt_backtest.py │
│  - Engine: Backtrader (event-driven)    │
│  - Output: Final audit report           │
└─────────────────────────────────────────┘
```

### Project Structure
```
src/                           # Main source code (standard src layout)
├── etf_strategy/              # Core strategy system
│   ├── run_combo_wfo.py       # WFO entry script
│   ├── core/                  # Core engines (🔒 DO NOT MODIFY)
│   │   ├── combo_wfo_optimizer.py    # Rolling WFO optimizer
│   │   ├── precise_factor_library_v2.py  # 18 factor library
│   │   ├── backtester_vectorized.py  # VEC backtesting engine
│   │   └── shared_types.py           # Shared utilities
│   └── auditor/               # BT audit module
│       └── core/engine.py     # Backtrader strategy
│
└── etf_data/                  # Data management module (independent)
    ├── core/                  # Downloader core
    └── config/                # Configuration management

scripts/                       # Operational scripts
├── batch_vec_backtest.py      # VEC batch backtesting
├── batch_bt_backtest.py       # BT batch auditing
└── update_daily_from_qmt_bridge.py  # Data updates

configs/                       # Global configurations
├── combo_wfo_config.yaml      # WFO configuration (43 ETFs)
└── etf_pools.yaml             # ETF pool definitions

results/                       # Backtesting results
└── ARCHIVE_*/                 # Archived best results
```

## Critical Parameters (v3.0 Production)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `FREQ` | 3 | Rebalance frequency (trading days) |
| `POS_SIZE` | 2 | Position count |
| `INITIAL_CAPITAL` | 1,000,000 | Initial capital |
| `COMMISSION` | 0.0002 | Commission rate (2bp) |
| `ETF_COUNT` | 43 | 38 A-shares + 5 QDII funds |
| `FACTOR_COUNT` | 18 | PreciseFactorLibrary factors |

**⚠️ CRITICAL**: 5 QDII ETFs contribute 90%+ returns and MUST NOT be removed:
- 513100 (Nasdaq 100): +22.03%
- 513500 (S&P 500): +25.37%
- 159920 (HSI): +17.13%
- 513050 (China Internet): +2.01%
- 513130 (HK Tech): +23.69%

## Development Rules

### ✅ Allowed Changes
- Bug fixes (without changing strategy logic)
- Data source adaptation
- Documentation improvements
- Performance optimizations (without changing results)
- Data updates (new date data)

### ❌ Prohibited Changes
- Modifying core factor library
- Changing backtesting engine logic
- **Modifying default parameter values (FREQ=3, POS=2)**
- **Modifying ETF pool definitions (especially 5 QDII funds)**
- Deleting ARCHIVE files
- Creating "simplified/backup" scripts - only use `src/etf_strategy/run_combo_wfo.py`

## Key Utilities

Critical shared utilities that MUST be used:
```python
from etf_strategy.core.utils.rebalance import (
    shift_timing_signal,           # Lag timing signals to avoid lookahead
    generate_rebalance_schedule,   # Unified rebalance scheduling
    ensure_price_views,            # Unified price views
)
```

## Common Pitfalls

| Issue | Problem | Solution |
|-------|---------|----------|
| Set iteration | Python set iteration order is non-deterministic | Use `sorted(set_obj)` |
| Lookahead bias | Using same-day signal for same-day execution | Use `shift_timing_signal` to lag 1 day |
| Rebalance day inconsistency | VEC/BT rebalance days differ | Use `generate_rebalance_schedule` |
| Float precision | Direct `==` comparison fails | Use 0.01% tolerance |
| Fund timing | BT fund calculation timing errors | Use post-sale cash calculation |

## Testing & Validation

- VEC/BT alignment MUST be < 0.01pp difference
- All tests: `uv run pytest tests/ -v`
- Critical alignment test: `scripts/full_vec_bt_comparison.py`

## Data Source

Data comes from QMT Trading Terminal via `qmt-data-bridge` SDK. **NEVER** construct manual HTTP requests - always use `QMTClient` from `qmt_bridge` package.

## Performance Characteristics

- WFO processes 12,597 combinations in ~2 minutes
- VEC uses Numba for high-speed vectorized backtesting
- BT uses Backtrader for event-driven audit verification
- All three tiers are aligned to < 0.01pp precision

## Strategy Status

- **Version**: v3.1 (Dec 1, 2025)
- **Status**: 🔒 Locked strategy with 237.45% returns verified
- **EPF Pool**: 43 ETFs (38 A-shares + 5 QDII)
- **Factors**: 18 factors in 5-factor optimal combination
- **Reproducibility**: ✅ Verified through Backtrader audit