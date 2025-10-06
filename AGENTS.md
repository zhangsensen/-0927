# Repository Guidelines

## Project Structure & Module Organization
Core factor logic and shared utilities live in `factor_system/`, with production screeners under `factor_system/factor_screening/`. Configuration defaults are in `factor_system/factor_screening/configs/`, while experimental templates stay in top-level `configs/`. Curated datasets belong in `data/`, vendor dumps in `raw/`, intermediate caches in `cache/`, and generated reports in `output/`. Place exploratory notebooks inside each component's `scripts/` subdirectory to keep the modules clean. Tests reside in `factor_system/factor_screening/tests/`.

## Build, Test, and Development Commands
Run `make install` once to sync dependencies via `uv` and install pre-commit hooks. Use `make format` before committing to apply Black and isort. Lint with `make lint`, execute unit tests via `make test`, and add coverage reporting with `make test-cov`. For an end-to-end smoke check, run `make run-example`, which invokes `professional_factor_screener.py` with the default multi-timeframe config.

## Coding Style & Naming Conventions
Target Python 3.11 with four-space indents and Black's 88-character line limit. Keep imports organized with isort's Black profile and avoid wildcard imports. Provide full type hints; missing annotations will fail mypy. Name modules and configs in snake_case (e.g., `enhanced_result_manager.py`, `0700_multi_timeframe_config.yaml`) and prefer descriptive suffixes such as `_pipeline`, `_manager`, or `_analyzer`.

## Testing Guidelines
pytest discovers files named `test_*.py` or `*_test.py` under `factor_system/factor_screening/tests/`. Isolate unit coverage for new factor calculators and tag slower paths with `@pytest.mark.slow` or `@pytest.mark.integration`. Use `make test-cov` to audit coverage and inspect the generated HTML report for signal-generation gaps.

## Commit & Pull Request Guidelines
Follow Conventional Commits (e.g., `feat: screening`, `fix: loader`) and include concise problem statements in messages. Pull requests should list changes as bullets, link relevant issues or tickets, and document the commands you ran (`make lint`, `make test`, etc.). Attach output snippets or figures when adding analytics so reviewers can reproduce results.

## Security & Configuration Tips
Never commit secrets or vendor data; load credentials through environment variables referenced in configs. When editing YAML configs, copy an existing template from `factor_system/factor_screening/configs/` to preserve defaults, and avoid mutating vendor dumps in `raw/`.
