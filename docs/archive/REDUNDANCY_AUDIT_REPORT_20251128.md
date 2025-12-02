# 🛡️ Redundancy & Safety Audit Report

**Date**: 2025-11-28
**Auditor**: GitHub Copilot (Gemini 3 Pro)
**Scope**: Global Codebase Audit for "Simplified" or "Mock" Engines

---

## 🚨 Executive Summary

The audit confirms that the **production pipeline (`etf_rotation_optimized`) is CLEAN** and uses the correct, verified engines. The "simplified" engines (`Phase2BacktestEngine`, `PositionOptimizer`) feared by the user **do not exist** in the production codebase. They are only referenced in broken test files, likely remnants of a previous iteration.

However, a structural redundancy was found in `etf_download_manager`, where configuration logic is duplicated.

---

## 🔍 Detailed Findings

### 1. `etf_rotation_optimized` (Core Strategy)
*   **Status**: ✅ **CLEAN**
*   **Verification**:
    *   No "simplified" or "mock" engines found in `core/`.
    *   `run_combo_wfo.py` uses the verified `ComboWFOOptimizer`.
    *   `scripts/batch_vec_backtest.py` uses the verified `PreciseFactorLibrary` and `CrossSectionProcessor`.

### 2. `strategy_auditor` (Audit Engine)
*   **Status**: ✅ **CLEAN**
*   **Verification**:
    *   Uses `GenericStrategy` in `core/engine.py`.
    *   Logic is explicitly aligned with VEC (e.g., `rebalance_schedule`, `shift_timing_signal`).
    *   No traces of `Phase2BacktestEngine`.

### 3. `etf_download_manager` (Data Acquisition)
*   **Status**: ⚠️ **REDUNDANT STRUCTURE**
*   **Issue**: Two competing configuration systems exist.
    1.  **Modern**: `etf_download_manager.core.config.ETFConfig` (Used by main logic).
    2.  **Legacy**: `etf_download_manager.config.etf_config_standalone.ETFConfig` (Used by `create_default_config` helper).
*   **Risk**: Low (functional), but confusing for maintenance.
*   **Recommendation**: Consolidate to `core/config.py`.

### 4. `tests/` (Unit Tests)
*   **Status**: ❌ **BROKEN / GHOST REFERENCES**
*   **Issue**: `tests/test_backtest_engine.py` imports `Phase2BacktestEngine` and `PositionOptimizer` from the root directory.
*   **Reality**: These files (`backtest_engine.py`, `position_optimizer.py`) **do not exist** in the codebase.
*   **Conclusion**: This test file is dead code testing non-existent modules.

### 5. `scripts/` (Execution)
*   **Status**: ✅ **CLEAN**
*   **Verification**:
    *   `batch_bt_backtest.py` correctly imports `strategy_auditor.core.engine`.
    *   `batch_vec_backtest.py` correctly imports `etf_rotation_optimized.core`.
    *   `archive/` contains legacy scripts, but they are isolated.

---

## 🛠️ Action Plan (To Be Executed)

The following actions are recommended to clean up the codebase. **No code has been changed yet.**

### Phase 1: Delete Dead Tests
*   [ ] Delete `tests/test_backtest_engine.py` (Tests non-existent code).

### Phase 2: Consolidate Download Manager
*   [ ] Update `etf_download_manager/scripts/download_etf_manager.py` to use `core.config` exclusively.
*   [ ] Delete `etf_download_manager/config/` folder (Legacy).

### Phase 3: Final Verification
*   [ ] Run `scripts/batch_vec_backtest.py` to ensure no regressions.

---

## 💬 Conclusion

The user's fear of "Phase2BacktestEngine" overriding production logic is **unfounded** in the current codebase state. The production system is safe. The references found are merely ghosts in a broken test file.
