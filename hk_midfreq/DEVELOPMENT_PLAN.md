# HK Mid-Frequency Strategy Module Development Plan

## 1. Project Setup
- [x] Create `hk_midfreq/` package to host strategy, backtest, and review tooling.
- [ ] Add module scaffolding files:
  - `config.py`
  - `factor_interface.py`
  - `strategy_core.py`
  - `backtest_engine.py`
  - `review_tools.py`
  - `__init__.py`
- [ ] Establish local data access adapters (e.g., wrappers around `MultiTimeframeFactorStore`).

## 2. Factor Integration Layer
- [ ] Implement `load_factor_scores` to pull top-ranked factors from screening outputs.
- [ ] Provide helper to fetch aligned OHLCV + factor data for target universe.
- [ ] Add validation routines (date coverage, liquidity filters).

## 3. Strategy Logic Layer
- [ ] Build reversal baseline (`hk_reversal_logic`) using vectorbt indicators.
- [ ] Implement portfolio-level candidate selection via factor scores.
- [ ] Parameterize strategy behaviors via `config.py` (capital, hold days, costs).
- [ ] Document extension points for future multi-factor blending.

## 4. Backtest Engine
- [ ] Deliver single-symbol backtest wrapper using `vbt.Portfolio.from_signals`.
- [ ] Implement multi-symbol portfolio aggregation and cash management.
- [ ] Support transaction cost modeling and dynamic position sizing hooks.
- [ ] Define standardized outputs (performance metrics, trades log, equity curve).

## 5. Review & Reporting Tools
- [ ] Build summary reporter (key metrics, drawdown, turnover).
- [ ] Add visualization helpers (equity curve, trade PnL histogram, heatmaps).
- [ ] Provide export utilities (CSV/JSON) for strategy diagnostics.

## 6. Automation & Testing
- [ ] Write smoke tests for each module (data loading, signal generation, backtest).
- [ ] Integrate with existing `make test` workflow (pytest collection inside `hk_midfreq/tests/`).
- [ ] Add pre-commit formatting hooks to cover new directory.

## 7. Documentation & Examples
- [ ] Draft README detailing module usage and configuration.
- [ ] Create `run_example.ipynb` demonstrating end-to-end workflow.
- [ ] Prepare onboarding checklist for new strategies (multi-symbol support, factor integration).

## 8. Timeline (Suggested)
| Week | Milestone |
|------|-----------|
| 1    | Complete module scaffolding, factor integration layer. |
| 2    | Implement strategy logic & baseline backtest, add unit tests. |
| 3    | Finalize review tools, documentation, and notebook demonstration. |

## 9. Dependencies & Coordination
- Align with `factor_system` outputs; reuse existing configs for data locations.
- Confirm vectorbt version compatibility with project lockfiles.
- Coordinate with data engineering to ensure timely Parquet updates for HK symbols.

## 10. Risk & Mitigation
- **Data latency**: Cache last successful load and expose diagnostics.
- **Parameter drift**: Centralize config constants and track changes via version control.
- **Performance bottlenecks**: Use lazy loading and chunked computations when iterating across symbols.

