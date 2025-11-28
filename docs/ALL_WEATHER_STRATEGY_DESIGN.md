# ❄️ All-Weather Snowball Strategy Design (1M Capital Edition)

**Objective**: Maximize Win Rate & Compound Growth ("Snowballing") for a 1M CNY account.
**Constraint**: 1M is "small" for institutional diversification but large enough for 5-8 positions.
**Core Philosophy**: **True Diversification** (Correlation Management) + **Aggressive Alpha** (Factor Rotation).

## 1. The "Sub-Pool" Architecture (Completed Phase 1)

We have successfully split the ETF universe into 7 distinct "Pillars":

| Pillar | Role | Target Allocation | Symbols |
| :--- | :--- | :--- | :--- |
| **1. EQUITY_BROAD** | **Anchor** (Beta) | 20% | HS300, CSI500, KC500 |
| **2. EQUITY_GROWTH** | **Attacker** (Alpha) | 15% | Chips, New Energy, Pharma |
| **3. EQUITY_CYCLICAL** | **Opportunist** | 10% | Securities, Bank, Resources |
| **4. EQUITY_DEFENSIVE**| **Shield** | 5% | Dividend, Consumer |
| **5. BOND** | **Safety** (Yield) | 20% | Treasury, Convertible |
| **6. COMMODITY** | **Hedge** (Inflation) | 15% | Gold, Silver |
| **7. QDII** | **Hedge** (Geographic) | 15% | Nasdaq, S&P 500 |

## 2. Why This Fits "1M Capital Snowballing"

For a 1M account, "All-Weather" doesn't mean holding everything all the time (which dilutes returns). It means **having access to everything** so you never miss a major trend.

### The "Snowball" Mechanism:
1.  **High Win Rate**: By diversifying across uncorrelated assets (Gold vs Stocks vs US Tech), we reduce the probability of a "Total Portfolio Drawdown".
2.  **Volatility Harvesting**: When A-shares drop, Gold/QDII often rise. We rebalance (sell high, buy low) to compound gains.
3.  **Cost Efficiency**: With 1M, we can comfortably hold ~6-8 ETFs (avg position 120k-160k). This is efficient for transaction costs and management.

## 3. Execution Strategy (Phase 2 Preview)

Instead of a static 20/15/10... split, we will implement a **Dynamic Regime Switching** model:

*   **Regime A: Bull Market (Risk-On)**
    *   Overweight: Growth + Cyclical + QDII
    *   Underweight: Bond + Gold
    *   *Goal: Aggressive Growth*

*   **Regime B: Bear Market / Stagflation (Risk-Off)**
    *   Overweight: Bond + Gold + Defensive
    *   Underweight: Growth + Broad
    *   *Goal: Capital Preservation (The "Snowball" must not melt)*

## 4. Next Steps

1.  **Factor Calibration (Phase 2)**:
    *   We need different factors for different pools.
    *   *Growth*: Momentum, Volatility.
    *   *Bond*: Yield curve, Macro signals.
    *   *Gold*: Dollar index, Real rates.
    
2.  **Portfolio Assembler (Phase 3)**:
    *   Build the engine that selects the "Top 6" assets from the 7 pools dynamically.

---
**Status**: Phase 1 (Pool Splitting) Complete. Data Verified.
**Ready for**: Phase 2 (Specific Factor Development).
