# WFO Metric Mismatch Analysis & Upgrade Plan

## 1. The Core Discrepancy
**Observation**: The correlation between WFO Ranking (IC) and Actual Strategy Return (VEC) is near zero (~0.03).
**Root Cause**: Mismatch between the **Optimization Objective** and the **Strategy Payoff**.

| Feature | WFO (Current) | VEC (Strategy) |
| :--- | :--- | :--- |
| **Metric** | Spearman IC (Rank Correlation) | Annualized Return / Sharpe |
| **Scope** | Global (All 43 ETFs) | Local (Top 2 ETFs) |
| **Logic** | "How well do I rank *everyone*?" | "Did I pick the *winner*?" |
| **Impact** | Penalizes strategies that pick winners but rank losers poorly. | Only cares about the Top 2 slots. |

### Scenario Illustration
*   **Scenario A (High IC, Low Return)**:
    *   Prediction: Ranks 1..43 perfectly, but misses the magnitude of the top winner.
    *   Result: High IC, but if the Top 2 aren't the absolute best *performers* (just best ranked relative to others), return is average.
*   **Scenario B (Low IC, High Return)**:
    *   Prediction: Correctly identifies the #1 and #2 exploding assets. Ranks the other 41 randomly.
    *   Result: IC is near zero (noise in bottom 41 dilutes signal). **Return is Massive.**

**Conclusion**: The current WFO engine is optimizing for "Broad Market Understanding", while the strategy requires "Sniper-like Precision" for the Top 2.

## 2. Is the 237% Strategy Overfit?
**Verdict**: **High Risk.**
*   The strategy `ADX + MAX_DD + ...` was identified by scanning the *entire* parameter space with VEC (Full Sample).
*   It bypasses the WFO validation.
*   It is the "Survivor" of 12,597 combinations over the specific 2020-2025 history.

**However**:
*   The fact that it uses logical factors (Trend + Risk) suggests it's not *pure* noise.
*   To validate it, we must check if it performs well in **Rolling Out-of-Sample (OOS)** tests using the **Top-2 Return** metric, not IC.

## 3. The Solution: WFO 2.0
We need to upgrade `ComboWFOOptimizer` to support a new optimization target.

### Proposed Changes
1.  **New Metric**: `mean_oos_top_k_return` (instead of `mean_oos_ic`).
2.  **Logic**:
    *   In each OOS window, simulate the Top-2 holding.
    *   Calculate the return of that specific portfolio.
    *   Average these OOS returns across all windows.
3.  **Selection**: Rank combinations by this new metric.

### Expected Outcome
*   The "Best" WFO combos will align much closer to the VEC results.
*   We will see if the 237% strategy (or similar ones) consistently appears in the Top tier of this new WFO.
*   If it does, the strategy is robust. If not, it was a lucky outlier.

## 4. Action Plan
1.  **Modify** `src/etf_strategy/core/combo_wfo_optimizer.py` to add `_compute_top_k_return`.
2.  **Update** `run_combo_wfo.py` to allow selecting the metric (IC vs Return).
3.  **Re-run** WFO on the 12,597 combos using the new metric.
4.  **Compare** the new WFO rankings with the VEC results.
