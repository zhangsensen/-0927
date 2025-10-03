# Repository Guidelines

## Project Structure & Module Organization
Core logic lives in `factor_system/`; `factor_screening/` hosts the production screener, utilities, and the reference tests under `tests/`. Configuration templates sit in `factor_system/factor_screening/configs/` and top-level `configs/` for experimental runs. Market data inputs are staged in `data/` and `raw/`, while generated artifacts land in `output/` and tool-specific `cache/`. Use `docs/` for internal reports and keep notebooks or exploratory scripts inside component-level `scripts/` folders to maintain clarity.

## Build, Test, and Development Commands
Bootstrap the workspace with `make install` (runs `uv sync --dev` and installs hooks). Format consistently via `make format`, then lint with `make lint` before opening a pull request. Run the unit suite using `make test`; add `make test-cov` when you need coverage details. For an end-to-end smoke run, use `make run-example`, which triggers `professional_factor_screener.py` with the default multi-timeframe config.

## Coding Style & Naming Conventions
Target Python 3.11 and follow Black’s 88-character line budget with 4-space indentation. Imports should respect the isort Black profile; avoid relative wildcards. Type hints are expected, as `mypy` runs in strict mode. Name modules and configs in snake_case (`enhanced_result_manager.py`, `0700_multi_timeframe_config.yaml`), and prefer descriptive suffixes such as `_manager`, `_pipeline`, or `_analyzer` to reflect responsibilities.

## Testing Guidelines
Pytest discovers files named `test_*.py` or `*_test.py` inside `factor_system/factor_screening/tests/`; mirror that pattern for new suites. Mark long-running jobs with `@pytest.mark.slow` or `integration` so they can be deselected. Aim to keep coverage stable by checking `make test-cov` locally and watching the HTML report for gaps in critical factor calculators. Each new feature should include at least one focused unit test plus an integration check when data pipelines are touched.

## Commit & Pull Request Guidelines
History follows Conventional Commit prefixes (`feat`, `chore`, `fix`, `refactor`); include a concise scope and use English summaries even when domain terms are Chinese. For pull requests, provide: a problem statement, bullet summary of changes, linked issues or task IDs, and a short checklist of validation commands executed (tests, lint, example run). Attach relevant output snippets or screenshots for analytics updates so reviewers can reproduce the results.
