# 🏗️ Project Consolidation & Integration Plan

**Date**: 2025-11-30
**Objective**: Unify the fragmented codebase into a standard, maintainable structure.

---

## 1. Current State Analysis (The "Fragmentation")

The project currently suffers from structural inconsistency:
*   **Split Logic**: Core strategy is in `etf_rotation_optimized`, but data management is in `etf_download_manager` (top-level), and auditing is in `strategy_auditor` (top-level).
*   **Scattered Scripts**: Entry points are found in `scripts/`, `etf_rotation_optimized/`, and `etf_download_manager/scripts/`.
*   **Config Duplication**: Configs exist in `configs/` and `etf_download_manager/config/`.
*   **Ghost Modules**: References to non-existent "Phase2" engines in tests.

## 2. Target Architecture (The "Unified" Vision)

We will move to a standard Python `src/` layout with centralized scripts and configs.

```
project_root/
├── src/                          # ✅ NEW: Single source of truth
│   ├── etf_strategy/             # (Renamed from etf_rotation_optimized)
│   │   ├── core/                 # Core logic (WFO, Factors)
│   │   └── auditor/              # (Moved from strategy_auditor)
│   └── etf_data/                 # (Renamed from etf_download_manager)
│       └── core/                 # Data logic
├── scripts/                      # ✅ NEW: Centralized Entry Points
│   ├── run_wfo.py                # (Was etf_rotation_optimized/run_combo_wfo.py)
│   ├── run_backtest.py           # (Was scripts/batch_vec_backtest.py)
│   ├── manage_data.py            # (Was etf_download_manager/scripts/download_etf_manager.py)
│   └── ...
├── configs/                      # ✅ NEW: Centralized Configuration
│   ├── strategy_config.yaml      # (Was combo_wfo_config.yaml)
│   ├── data_config.yaml          # (Was etf_download_manager/config/etf_config.yaml)
│   └── ...
├── results/                      # ✅ NEW: Centralized Output
└── tests/                        # Cleaned up tests
```

---

## 3. Execution Roadmap

### Phase 1: Cleanup (Immediate)
*   [ ] **Delete Dead Code**: Remove `tests/test_backtest_engine.py` (tests non-existent code).
*   [ ] **Remove Empty Dirs**: Remove `etf_rotation_optimized/results` (unused).

### Phase 2: Structural Migration
*   [ ] **Create `src/`**: Initialize the new source root.
*   [ ] **Move & Rename Strategy**:
    *   `etf_rotation_optimized` -> `src/etf_strategy`
*   [ ] **Move & Rename Data**:
    *   `etf_download_manager` -> `src/etf_data`
*   [ ] **Integrate Auditor**:
    *   `strategy_auditor` -> `src/etf_strategy/auditor`

### Phase 3: Script & Config Consolidation
*   [ ] **Centralize Scripts**: Move all entry scripts to `scripts/` and update imports to use `src.` prefix (e.g., `from etf_strategy.core import ...`).
*   [ ] **Centralize Configs**: Move `etf_download_manager/config/*.yaml` to `configs/`.
*   [ ] **Update Config Loaders**: Modify `etf_data` to read from `configs/`.

### Phase 4: Verification
*   [ ] **Run Full Pipeline**:
    1.  `python scripts/manage_data.py --action summary` (Verify Data)
    2.  `python scripts/run_wfo.py` (Verify Strategy)
    3.  `python scripts/run_backtest.py` (Verify Backtest)

---

## 4. Benefits
1.  **Single Import Path**: Everything is `from src.etf_...`. No more `sys.path.insert` hacks.
2.  **Clear Separation**: `scripts/` is for running, `src/` is for logic, `configs/` is for settings.
3.  **Professional Standard**: Matches industry standard Python project layout.

---

## 5. Next Steps
Shall I proceed with **Phase 1 (Cleanup)** and then **Phase 2 (Migration)**?
