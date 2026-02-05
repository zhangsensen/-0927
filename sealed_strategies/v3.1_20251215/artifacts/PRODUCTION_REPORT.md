# 🚀 Production Strategy Pack v3.2 (Top 6)

**Generated**: 2025-12-15 03:19:20
**Source**: Leakage-Controlled Triple Validation (screening) + Backtrader Audit (ground truth)

## 1. Executive Summary
This pack is intended to be *audit-grade* (no ambiguous comparisons):
1.  **Screening (VEC + Rolling + Holdout)**: Candidate discovery + stability filtering (leakage-controlled).
2.  **Audit (BT Ground Truth)**: Final production metrics use BT-split returns (train/holdout) and BT risk stats.

**Key Principle**: When VEC and BT differ due to execution assumptions, BT wins.

### Data Splits
- **Train**: 2020-01-01 → 2025-04-30 (from manifest)
- **Holdout**: 2025-05-01 → 2025-12-12 (from manifest)

### 🏆 Best Strategy: `ADX_14D + MAX_DD_60D + PRICE_POSITION_120D + SHARPE_RATIO_20D + VOL_RATIO_60D`
- **Train Return (BT)**: 107.57%
- **Holdout Return (BT)**: 21.03%
- **Total Return (BT, full)**: 153.70%
- **Max Drawdown (BT, full)**: 17.83%
- **Calmar Ratio (BT, full)**: 0.99
- **Win Rate (trade)**: 51.40%
- **Profit Factor**: 1.88
- **Total Trades**: 286

## 2. Top Strategies List
Sorted by `prod_score_bt` (BT-ground-truth, OOS-first).

| Combo                                                                                                      |   ProdScore | BT Train Ret   | BT Holdout Ret   | BT Total Ret   | BT MDD   |   BT Calmar | BT WinRate   |   ProfitFactor |   Trades | Roll Win%   | Roll Worst   |
|:-----------------------------------------------------------------------------------------------------------|------------:|:---------------|:-----------------|:---------------|:---------|------------:|:-------------|---------------:|---------:|:------------|:-------------|
| ADX_14D + MAX_DD_60D + PRICE_POSITION_120D + SHARPE_RATIO_20D + VOL_RATIO_60D                              |       0.845 | 107.57%        | 21.03%           | 153.70%        | 17.83%   |        0.99 | 51.40%       |           1.88 |      286 | 78%         | -6.72%       |
| CORRELATION_TO_MARKET_20D + RELATIVE_STRENGTH_VS_MARKET_20D + SHARPE_RATIO_20D + SLOPE_20D + VOL_RATIO_20D |       0.725 | 96.51%         | 14.47%           | 129.21%        | 19.10%   |        0.82 | 50.17%       |           1.8  |      293 | 61%         | -6.69%       |
| ADX_14D + OBV_SLOPE_10D + SHARPE_RATIO_20D + SLOPE_20D                                                     |       0.605 | 109.45%        | 10.35%           | 134.19%        | 15.65%   |        1.02 | 50.00%       |           2.06 |      256 | 61%         | -5.92%       |
| ADX_14D + PRICE_POSITION_120D + SLOPE_20D + VOL_RATIO_20D                                                  |       0.57  | 51.84%         | 10.37%           | 70.76%         | 10.37%   |        0.94 | 52.51%       |           2.16 |      219 | 61%         | -4.25%       |
| ADX_14D + MAX_DD_60D + PV_CORR_20D + SHARPE_RATIO_20D                                                      |       0.565 | 90.34%         | 14.13%           | 120.11%        | 23.44%   |        0.63 | 53.85%       |           1.55 |      364 | 72%         | -7.91%       |
| ADX_14D + CALMAR_RATIO_60D + MAX_DD_60D + PV_CORR_20D + SHARPE_RATIO_20D + SLOPE_20D + VOL_RATIO_20D       |       0.51  | 69.61%         | 11.77%           | 93.17%         | 17.50%   |        0.7  | 50.00%       |           1.83 |      240 | 72%         | -6.15%       |

## 3. Audit Notes (No-Ambiguity)
- **Ground Truth**: All production returns shown in this report are from BT (event-driven).
- **VEC vs BT**: VEC is used for fast screening; it may diverge from BT due to execution modeling differences.
- **Leakage Control**: Rolling stability metrics are train-only (no holdout segment mixed).

## 4. Risk Disclosures
- **Holdout Length**: 2025-05 to 2025-10 is short; monitor live drawdowns and regime shifts.
- **QDII Exposure**: Many top strategies use QDII assets; US/HK regime shifts can impact OOS.
- **Execution**: BT assumes execution at bar prices; real slippage/taxes may reduce performance.

## 5. Implementation Guide
1.  **Select**: Choose 1-3 strategies (or a small basket) from the top list.
2.  **Monitor**: Track drawdown vs BT expectation and stop deploying if behavior breaks.
3.  **Rebalance**: Follow the fixed rule: FREQ=3 trading days, POS=2, no stop-loss.
