# Release Notes v3.3 (20251216)

## 🎯 Objective
Deliver a **robust, diverse strategy portfolio** optimized for compounding returns rather than single-shot high yield.
This release enables the **Regime Gate (Volatility)** mechanism to significantly reduce drawdown and improve Sharpe ratio.

## 🛡️ Regime Gate Impact (vs v3.1/v3.2)
- **Max Drawdown**: Reduced by ~3.2% (Median 16.0% -> 12.8%)
- **Sharpe Ratio**: Improved by ~0.1 (Median 0.88 -> 0.98)
- **Win Rate**: Improved by ~3% (Rolling 3M Win Rate 64% -> 67%)
- **Trade-off**: Total Return decreased by ~10% (Median 106% -> 96%), accepting lower absolute return for higher stability.

## 🧩 Portfolio Composition
Selected 5 strategies with **low factor overlap (Jaccard < 0.6)** to ensure diversity:

### Strategy #1
- **Combo**: `ADX_14D + OBV_SLOPE_10D + PRICE_POSITION_120D + PV_CORR_20D + SHARPE_RATIO_20D + SLOPE_20D`
- **Role**: Balanced / Quality
- **Composite Score**: 0.8032
- **Holdout Calmar**: 0.71
- **BT Total Return**: 129.50%
- **BT Max Drawdown**: 15.53%

### Strategy #2
- **Combo**: `ADX_14D + OBV_SLOPE_10D + PRICE_POSITION_120D + SLOPE_20D + VORTEX_14D`
- **Role**: Trend Follower
- **Composite Score**: 0.7000
- **Holdout Calmar**: 0.31
- **BT Total Return**: 83.44%
- **BT Max Drawdown**: 19.20%

### Strategy #3
- **Combo**: `ADX_14D + SHARPE_RATIO_20D + SLOPE_20D + VORTEX_14D`
- **Role**: Trend Follower
- **Composite Score**: 0.6355
- **Holdout Calmar**: 0.24
- **BT Total Return**: 118.49%
- **BT Max Drawdown**: 18.54%

### Strategy #4
- **Combo**: `ADX_14D + MAX_DD_60D + PRICE_POSITION_120D + SHARPE_RATIO_20D + VOL_RATIO_60D`
- **Role**: Defensive / Risk-Managed
- **Composite Score**: 0.5968
- **Holdout Calmar**: 1.16
- **BT Total Return**: 147.84%
- **BT Max Drawdown**: 17.83%

### Strategy #5
- **Combo**: `CALMAR_RATIO_60D + OBV_SLOPE_10D + PRICE_POSITION_20D + SHARPE_RATIO_20D + SLOPE_20D + VORTEX_14D`
- **Role**: Trend Follower
- **Composite Score**: 0.4935
- **Holdout Calmar**: 0.14
- **BT Total Return**: 87.26%
- **BT Max Drawdown**: 22.39%

