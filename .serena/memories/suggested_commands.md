# Suggested Commands

- Install deps: `make install`
- Run full pipeline: `make run-pipeline`
- Generate factors: `make generate-factors`
- Screen factors with latest panel: `make screen-factors`
- Run backtest (auto-detect latest panel/factors): `make backtest`
- Run pytest suite: `make test`
- Clean caches/artifacts: `make clean`
- Execute upgraded Top-N evaluation: `python scripts/compute_wfo_backtest_metrics.py <results_dir> --tx-cost-bps 5 --max-turnover 0.5 --target-vol 0.10`
- Negative IC diagnosis: `python scripts/analyze_negative_ic_windows.py --baseline <path> --enhanced <path>`
