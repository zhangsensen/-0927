# Repository Guidelines

## Project Structure & Module Organization
Core factor models and utilities reside in `factor_system/`, with the production screener under `factor_system/factor_screening/`. Shared configs live in `factor_system/factor_screening/configs/`, while experimental templates sit in top-level `configs/`. Data inputs are staged in `data/` (curated) and `raw/` (vendor dumps). Generated artifacts flow to `output/` and intermediate caches to `cache/`. Use `docs/` for internal reports, and keep exploratory notebooks inside each component's `scripts/` directory to avoid polluting the main modules.

## Build, Test, and Development Commands
Run `make install` once to sync dependencies via `uv` and install pre-commit hooks. Use `make format` before committing to enforce Black and isort. Lint with `make lint`, and execute `make test` for the unit suite. Add `make test-cov` when chasing coverage diffs. For a quick end-to-end check, run `make run-example`, which invokes `professional_factor_screener.py` using the default multi-timeframe config.

## Coding Style & Naming Conventions
Target Python 3.11 with 4-space indentation and Black’s 88-character limit. Organize imports with the isort Black profile; avoid relative wildcards and keep domain packages grouped. Type hints are mandatory; strict `mypy` gates: missing annotations will fail CI. Name modules and configs in snake_case, e.g. `enhanced_result_manager.py` or `0700_multi_timeframe_config.yaml`, and favor descriptive suffixes like `_pipeline`, `_manager`, or `_analyzer`.

## Testing Guidelines
Pytest discovers files named `test_*.py` or `*_test.py` under `factor_system/factor_screening/tests/`. Write focused unit tests for new factor calculators and mark slow paths with `@pytest.mark.slow` or `integration`. Monitor coverage with `make test-cov` and review the HTML report for gaps in critical signal generation.

## Commit & Pull Request Guidelines
Follow Conventional Commits (`feat: screening`, `fix: loader`). Each PR should include a problem statement, bullet summary of updates, linked issues or task IDs, and the commands you ran (`make lint`, `make test`, etc.). Attach relevant output snippets or figures when updating analytics so reviewers can replicate results.

## Security & Configuration Tips
Keep secrets and API keys outside the repo; reference them via environment variables consumed by configs. When editing YAML configs, duplicate existing templates in `factor_system/factor_screening/configs/` to preserve defaults, and never commit vendor data placed in `raw/`.
