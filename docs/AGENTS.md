<!-- ALLOW-MD -->
# Repository Guidelines

This repository hosts quantitative trading research, backtests, and production workflows. Use these guidelines when adding or modifying code, configs, or analysis notebooks.

## Project Structure & Module Organization

- Core research and strategies: `strategies/`, `features/`, `a_shares_strategy/`, `hk_midfreq/`, `etf_rotation_system/`.
- Configuration files: `config/`, `configs/`, and per‑project `*/config` folders.
- Scripts and tooling: `scripts/`, `check_wfo_progress.sh`, `VECTORIZATION_QUICK_START.sh`.
- Data and results (do not commit new large binaries): `raw/`, `output/`, `results/`, `results_combo_wfo/`, `logs/`, `cache/`.
- Documentation and reports: Markdown and text files in the repo root and subprojects.

## Build, Test, and Development Commands

- `make help` – list available tasks defined in `Makefile`.
- `make setup` – create or update the Python environment for local development.
- `make lint` – run linters and basic style checks.
- `make test` – execute the main automated test suite.
- `python profile_backtest.py` – run a profiling backtest (see `README.md` for arguments).

## Coding Style & Naming Conventions

- Language: Python 3; follow PEP 8 (4‑space indentation, 79–100 char lines).
- Use descriptive snake_case for functions/variables and PascalCase for classes.
- Prefer pure, testable functions in `strategies/` and `features/`; keep I/O and CLI logic in `scripts/`.
- Keep config-like values in YAML/JSON/TOML files under `config*/` rather than hard‑coding.

## Testing Guidelines

- Use `make test` for full runs; add focused tests near the module under test (e.g., `tests/` or project‑local `test_*.py` files).
- Name tests `test_<module>_<behavior>()` and keep them deterministic.
- When adding a new strategy or factor, include at least one regression or smoke test covering its main workflow.

## Commit & Pull Request Guidelines

- Write concise, imperative commit messages (e.g., `Add HK midfreq factor`, `Fix WFO grid search config`).
- Group related changes into a single commit; avoid mixing refactors with behavior changes.
- For PRs, include: purpose and context, key implementation notes, how you validated (commands run, data sets used), and any impact on production runs or stored results.

## Agent-Specific Instructions

- Treat this `AGENTS.md` as the source of truth for style and structure.
- When editing or creating files, prefer minimal, targeted changes consistent with existing patterns.
