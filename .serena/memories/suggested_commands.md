# Suggested Commands (v1.1)

## Environment Setup
- `uv sync --dev` – install all dependencies
- `uv pip install -e .` – install project in editable mode (required for imports)

## Production Workflow (reproduce 121% returns)
- `uv run python src/etf_strategy/run_combo_wfo.py` – WFO screening (12,597 combos)
- `uv run python scripts/batch_vec_backtest.py` – VEC batch backtest
- `uv run python scripts/batch_bt_backtest.py` – BT audit (optional)

## Testing & Quality
- `uv run pytest tests/ -v` – run all tests (20 tests)
- `make format` – format code (black + isort)
- `make lint` – lint code (flake8 + mypy)

## Utilities
- `uv run python tools/validate_combo_config.py` – validate configuration
- `uv run python scripts/full_vec_bt_comparison.py` – VEC/BT alignment check

## Important Notes
- All Python scripts MUST use `uv run python` prefix
- DO NOT use `python` or `python3` directly
