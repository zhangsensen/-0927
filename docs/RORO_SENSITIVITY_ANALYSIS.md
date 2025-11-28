# 🔍 RORO Strategy Sensitivity Analysis: The "Gold Factor"

**Date**: 2025-11-28
**Subject**: Decomposing Alpha Sources for Rank 3 Strategy
**Strategy**: `ADX_14D + PRICE_POSITION_20D + SHARPE_RATIO_20D + SLOPE_20D + VOL_RATIO_20D`

## 1. Executive Summary

The user raised a critical concern: **"Is the 168% return just a proxy for the Gold Bull Market?"**

We performed a controlled experiment by isolating the "Risk-Off" asset class while keeping the Equity Selection logic identical.

| Scenario | Risk-Off Asset | Total Return | Max Drawdown | Alpha Source |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (Current)** | **Gold + Bond** | **166.0%** | **5.9%** | Equity Alpha + **Gold Beta** |
| **Stress Test A** | **Cash Only** | **89.7%** | **11.5%** | **Pure Equity Alpha** |
| **Stress Test B** | Bond Only | 76.0% | 11.0% | Equity Alpha + Bond Beta |

## 2. Key Findings

### A. Gold Contribution is Massive (~45% of Total Return)
- The jump from **89.7% (Cash)** to **166.0% (Gold)** confirms that **~76 percentage points** of the return came from holding Gold during A-share downturns.
- This validates the user's suspicion: The strategy effectively "timed the market" by switching to a booming asset (Gold).

### B. The Strategy is NOT "Meaningless" Without Gold
- Even in the **"Cash Only"** scenario (assuming Gold didn't exist or was flat), the strategy delivered:
    - **Return**: 89.7% (vs HS300 +22%)
    - **MaxDD**: 11.5% (vs HS300 -40%+)
- This proves the **Equity Selection + Market Timing** logic has genuine Alpha, independent of Gold.

### C. Bond Drag
- "Bond Only" performed worse than Cash (76% vs 90%).
- This suggests that during the specific Risk-Off periods (2022-2023), short-term bonds (511010) either underperformed or the **transaction costs** of switching in/out of bonds outweighed their meager yields.

## 3. Strategic Implication

The current "All-Weather" label is heavily reliant on **Gold's historical performance**. To build a *true* All-Weather system that survives a "Gold Bear Market", we must implement the **Refactoring Plan (Sub-pools)** immediately.

**Required Shift:**
- **From**: "Equity or Gold" (Binary)
- **To**: "Equity + Bond + Gold + Commodities + USD" (Diversified Risk-Off)

## 4. Conclusion

The strategy is **deployable** (89% base return is excellent), but the **expectations must be managed**. The 168% figure is a "Best Case" scenario driven by a Gold Supercycle. A realistic expectation for the future (without a Gold boom) is closer to the **90% return / 11% MaxDD** profile.
