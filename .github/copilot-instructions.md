# GitHub Copilot Instructions

## 🎯 Project Overview & Architecture
- **Core Domain**: Quantitative trading platform featuring `etf_rotation_optimized` (WFO production), `factor_system` (unified factor engine), and `real_backtest` (authoritative backtesting).
- **Key Components**:
  - `etf_rotation_optimized/`: Main ETF rotation strategy with Walk-Forward Optimization.
  - `factor_system/`: Unified engine for 154+ technical & 15+ money flow factors.
  - `real_backtest/`: **Authoritative** backtest implementation; overrides root-level scripts.
  - `etf_download_manager/`: Data acquisition with incremental updates.
- **Design Philosophy**: "Linus Quant" — stable APIs, short functions, 100% vectorization, no `.apply()`.

## 🛠️ Critical Workflows & Commands
- **Environment**:
  - Setup: `uv sync --dev && pre-commit install`.
  - Proxy: Run `bash setup_vscode_proxy.sh` if network issues arise (configures VS Code Server to use local Xray/proxy).
- **Production & Testing**:
  - **Smoke Test**: `python etf_rotation_optimized/run_combo_wfo.py --quick`
  - **Full Production**: `make run` or `python run_final_production.py`
  - **Backtest Verification**: `python etf_rotation_optimized/real_backtest/test_freq_no_lookahead.py` (Must pass!)
  - **Audit**: `make audit RUN_DIR=results/run_YYYYMMDD_HHMMSS`
- **Factor Generation**: `python scripts/production_run.py --set technical_only`

## 🔒 Guardrails & Conventions
- **Markdown Restrictions**:
  - **ONLY** create `.md` files in `docs/`.
  - **MUST** include `<!-- ALLOW-MD -->` in the first 10 lines of any new markdown file.
- **Data & Execution**:
  - **No Lookahead**: Strict T+1 execution; signals freeze at 14:30.
  - **Vectorization**: Use `numpy`/`pandas` vectorization. **Ban** `for` loops and `df.apply()` for data processing.
  - **Output Schema**: Adhere strictly to `docs/OUTPUT_SCHEMA.md`. Do not change output columns without updating docs.
- **Stability & Reproducibility**:
  - Use `RB_STABLE_RANK=1` and `RB_DAILY_IC_PRECOMP=1` for deterministic results.
  - Do not modify `etf_rotation_optimized/core/*` without explicit authorization.

## 🧩 Key Files & Directories
- `etf_rotation_optimized/run_combo_wfo.py`: Main entry point.
- `real_backtest/`: The source of truth for backtesting logic.
- `factor_system/factor_engine/api.py`: Unified API for factor calculation.
- `setup_vscode_proxy.sh`: Network configuration for VS Code Server.
- `Makefile`: Central hub for `lint`, `test`, `audit`, `monitor`.

## ⚠️ Common Pitfalls
- **Proxy Issues**: If Copilot/Extensions fail, check `setup_vscode_proxy.sh` matches your Xray port (default 10809).
- **Path Confusion**: Prefer `real_backtest/` scripts over root-level wrappers when debugging core logic.
- **Performance**: Always profile before optimizing. Use `RB_PROFILE_BACKTEST=1` to identify bottlenecks.
