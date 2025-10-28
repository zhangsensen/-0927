# Project Overview

- Name: etf_rotation_optimized
- Purpose: end-to-end ETF rotation research & execution pipeline with factor generation, cross-sectional standardization, walk-forward optimization, and backtesting.
- Key Components: core/ (factor library, selectors, optimizers), scripts/ (CLI utilities, diagnostics), configs/ (YAML experiment definitions), results/ (parquet outputs), docs/ (guides & audits).
- Workflow: generate factors → standardize cross-section → select factors under constraints → run walk-forward/backtest → evaluate & report.
- Data Formats: Parquet for panels & results, JSON for metadata, YAML for configs.
