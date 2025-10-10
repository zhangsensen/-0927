# Repository Guidelines

## Project Structure & Module Organization
FactorEngine code lives under `factor_system/`. `factor_engine/` exposes the public API, caching, and provider integrations; `factor_generation/` handles batch factor creation; `shared/` houses reusable calculators; `factor_screening/` contains screening pipelines, configs, and archived results. Tests live in `tests/`, while integration suites such as `test_factor_engine_final.py` and `test_vectorbt_adapter_fixed.py` remain at repo root. Helper utilities belong in `scripts/`. Store raw inputs in `data/` or `raw/`, and route generated artifacts to `output/`, `cache/`, or module-level `screening_results/` directories.

## Build, Test, and Development Commands
Run `make install` to execute `uv sync --dev` and set up pre-commit hooks. Other staples: `make format` (Black + isort), `make lint` (flake8 at 88 chars plus mypy), `make test` (`pytest -v`), `make test-cov` (HTML coverage in `htmlcov/`), and `make run-example` to launch `professional_factor_screener.py --config factor_system/factor_screening/configs/multi_tf_config.yaml`. For quick iteration, call `pytest tests/test_factor_engine_accuracy.py -k rsi`.

## Coding Style & Naming Conventions
Target Python 3.11 with four-space indentation. Let `black` enforce the 88-character line limit and `isort` manage imports. Use `snake_case` for modules and functions (`factor_system/factor_engine/core/vectorbt_adapter.py`), `CapWords` for classes, and lower-hyphenated YAML filenames (e.g., `multi_tf_config.yaml`). Type hints are expected on public APIs, especially under `factor_system/factor_engine/core/`. Run `pre-commit run --all-files` before pushing.

## Testing Guidelines
`pytest` drives the suite. Place new unit coverage in `tests/` and add cross-package regression checks next to existing top-level suites. Name files `test_<feature>.py` and functions `test_<scenario>`. Prefer deterministic fixtures; seed vectorbt samples (`np.random.seed(42)`) and commit expected parity checks against shared calculators. Use `make test-cov` to watch coverage and review `htmlcov/index.html` for gaps.

## Commit & Pull Request Guidelines
Follow conventional commits (`feat:`, `fix:`, `refactor:`) with imperative subjects under ~60 characters. Capture ticket numbers or factor IDs in the body and note data or config impacts. Before opening a PR, confirm `make lint`, `make test`, and any targeted `pytest` jobs succeed, and attach logs or charts for screening updates so reviewers can reproduce.

## Security & Configuration Tips
Keep secrets in a local `.env` loaded by `python-dotenv`; do not commit them. Share reusable parameter sets via `factor_system/factor_screening/configs/templates/` and document overrides in PRs. Park large CSV/Parquet exports in `output/` or module `logs/` directories and extend `.gitignore` when new paths appear.
