# Repository Guidelines

## Project Structure & Module Organization
Core engine code lives in `factor_system/`, with `factor_engine/` handling factor computation, `factor_generation/` pipelines, and `utils/` shared helpers. YAML presets under `configs/` and `config/` drive factor screens and production jobs; example notebooks and scripts sit in `examples/` and `scripts/`. Production handoffs (scheduled jobs, packaging artifacts) are tracked in `production/`, while `docs/` captures research notes and release briefs. Place new tests in `tests/`, keeping fixtures or synthetic market data alongside each module they exercise.

## Build, Test, and Development Commands
Run `make install` (or `uv sync --dev`) before the first contribution to pull Python 3.11 dependencies and set up pre-commit hooks. Use `make format`, `make lint`, and `make check` before opening a PR to apply Black/isort, run Flake8 + mypy, and execute the full hook suite. `make test` delegates to `pytest -v` across the repository; `make test-cov` surfaces coverage gaps via an HTML report. To sanity-check feature work, run `make run-example`, which executes `factor_system/factor_screening/professional_factor_screener.py` with the default multi-timeframe config.

## Coding Style & Naming Conventions
Python files target Black’s 88-character line length and isort-organized alphabetical imports. Stick to snake_case for modules, functions, and variables; reserve PascalCase for classes and Config objects, mirroring existing engine components. Type hints and concise docstrings are required for new public APIs, especially under `factor_system/factor_engine/api.py`. Keep configuration filenames descriptive (`market_window_strategy.yaml`) and avoid spaces; prefer hyphen-separated CLI flags.

## Testing Guidelines
Author pytest modules with names like `test_factor_registry.py` and describe scenarios using `pytest.mark.parametrize` when checking multiple symbols or timeframes. Use `tmp_path` instead of writing to project directories, and keep deterministic data fixtures under `tests/fixtures/`. Maintain baseline coverage by running `pytest --cov=factor_system`, and add regression cases whenever a bug fix touches factor calculations or scheduling logic.

## Commit & Pull Request Guidelines
Follow the short, descriptive Conventional Commits pattern seen in history (`feat: add Hong Kong weekday trading strategy`, `refactor: clean up project structure`). Each PR should include a summary of the market scope impacted, validation commands (`make test`, `make run-example`), and links to any tracking tickets. Attach before/after metrics or screenshots when factor performance shifts. Rebase on `main`, ensure CI passes, and flag potential production or data migration impacts in the PR description.

## Security & Configuration Tips
Store credentials in environment variables loaded via `.env` files; never commit secrets or broker keys. Review YAML configs for unintended production toggles before merging, and sanitize any cached data under `cache/` or `output/` prior to sharing branches.
