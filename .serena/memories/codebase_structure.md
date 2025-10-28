# Codebase Structure Highlights

- `core/`: core engine modules (factor library, selectors, walk-forward optimizer, data validators, constants).
- `scripts/`: CLI utilities for analysis, reporting, batch runs.
- `configs/`: global configs, experiment definitions, constraint YAMLs.
- `results/`: generated artifacts (parquet panels, WFO outputs, metadata).
- `docs/`: user guides, audits (needs pruning per latest cleanup).
- `tests/`: pytest suites.
- `cache/`: intermediate factor calculations (parquet).
