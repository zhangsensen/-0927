# Suggested Commands
- `cd etf_rotation_optimized && make install` – install dependencies for ETF rotation subsystem.
- `cd etf_rotation_optimized && make run` – execute the end-to-end ETF rotation pipeline.
- `pytest -v` – run repository-wide tests.
- `black path/to/module` & `isort path/to/module` – format Python code.
- `mypy path/to/module` – static type checks.
- `pre-commit run --all-files` – run repository lint suite.
- `python scripts/train_two_stage_ranker.py --help` – inspect ML ranking trainer options.
- `python real_backtest/run_profit_backtest.py --help` – review live backtest entrypoint options.