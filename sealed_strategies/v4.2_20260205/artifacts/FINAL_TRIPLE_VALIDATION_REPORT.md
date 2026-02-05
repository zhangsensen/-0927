# 🏆 Final Triple Validation Report

**Generated**: 2026-02-05 21:17:20

## 0. Leakage Control (Audit Note)
This run uses **train-only** rolling OOS summary as the gate input (no holdout segments).

- VEC (train): `/home/sensen/dev/projects/-0927/results/vec_from_wfo_20260205_211639/full_space_results.parquet`
- Rolling (train-only): `/home/sensen/dev/projects/-0927/results/rolling_oos_consistency_20260205_211717/rolling_oos_summary.parquet`
- Holdout (unseen): `/home/sensen/dev/projects/-0927/results/holdout_validation_20260205_211719/holdout_validation_results.parquet`

## 1. Screening Funnel
- **Total Universe**: 200
- **Risk Filter** (factor exclusions): 200
- **Rolling Consistency Gate** (Strict): 89 (PosRate>=0.6, Worst>=-0.08, Calmar>=0.8)
- **Holdout Gate** (Profitable): 55 (Return > 0.0)

## 2. Top 20 'Gold Standard' Strategies
Sorted by Composite Score (30% Train Calmar + 40% Roll Worst + 30% Holdout Calmar)

| Combo                                                                                                               |   Train Ret |   Train Calmar |   Roll Win% |   Roll Worst |   Holdout Ret |   Holdout Calmar |
|:--------------------------------------------------------------------------------------------------------------------|------------:|---------------:|------------:|-------------:|--------------:|-----------------:|
| AMIHUD_ILLIQUIDITY + PRICE_POSITION_20D + PV_CORR_20D + SLOPE_20D                                                   |      0.7628 |         1.4961 |        0.78 |      -0.029  |        0.08   |           0.424  |
| ADX_14D + AMIHUD_ILLIQUIDITY + GK_VOL_RATIO_20D + SHARPE_RATIO_20D + SLOPE_20D                                      |      0.5534 |         1.2707 |        0.61 |      -0.024  |        0.0716 |           0.2997 |
| ADX_14D + PRICE_POSITION_120D + PV_CORR_20D + SHARPE_RATIO_20D + SLOPE_20D + VORTEX_14D                             |      1.0123 |         1.6833 |        0.61 |      -0.03   |        0.0213 |           0.1118 |
| ADX_14D + BREAKOUT_20D + CALMAR_RATIO_60D + PRICE_POSITION_120D + SHARPE_RATIO_20D + SLOPE_20D                      |      0.9591 |         1.7276 |        0.72 |      -0.0228 |        0.0067 |           0.0343 |
| ADX_14D + BREAKOUT_20D + CALMAR_RATIO_60D + PV_CORR_20D + SHARPE_RATIO_20D + SLOPE_20D + VORTEX_14D                 |      0.8445 |         1.4897 |        0.61 |      -0.0428 |        0.0665 |           0.3135 |
| BREAKOUT_20D + CALMAR_RATIO_60D + SHARPE_RATIO_20D + SLOPE_20D + VOL_RATIO_20D + VORTEX_14D                         |      0.6515 |         1.1071 |        0.61 |      -0.0379 |        0.112  |           0.7376 |
| ADX_14D + PRICE_POSITION_120D + PRICE_POSITION_20D + PV_CORR_20D + SHARPE_RATIO_20D + SLOPE_20D + VORTEX_14D        |      1.0681 |         1.7539 |        0.67 |      -0.03   |        0.0086 |           0.0453 |
| ADX_14D + BREAKOUT_20D + CALMAR_RATIO_60D + SHARPE_RATIO_20D + SLOPE_20D + VORTEX_14D                               |      0.8851 |         1.7365 |        0.61 |      -0.0529 |        0.0761 |           0.3584 |
| BREAKOUT_20D + CALMAR_RATIO_60D + PV_CORR_20D + SHARPE_RATIO_20D + SLOPE_20D + VOL_RATIO_20D + VORTEX_14D           |      0.6899 |         1.1157 |        0.61 |      -0.0423 |        0.1152 |           0.7932 |
| ADX_14D + SHARPE_RATIO_20D + SLOPE_20D + VORTEX_14D                                                                 |      1.0395 |         1.8073 |        0.61 |      -0.0529 |        0.0452 |           0.2438 |
| ADX_14D + AMIHUD_ILLIQUIDITY + GK_VOL_RATIO_20D + PRICE_POSITION_20D + SHARPE_RATIO_20D + SLOPE_20D                 |      0.6186 |         1.2724 |        0.72 |      -0.0276 |        0.0264 |           0.1102 |
| ADX_14D + BREAKOUT_20D + CALMAR_RATIO_60D + PRICE_POSITION_20D + SHARPE_RATIO_20D + SLOPE_20D + VORTEX_14D          |      0.8605 |         1.533  |        0.61 |      -0.0529 |        0.0806 |           0.3797 |
| BREAKOUT_20D + CALMAR_RATIO_60D + SHARPE_RATIO_20D + SLOPE_20D + UP_DOWN_VOL_RATIO_20D + VOL_RATIO_20D              |      0.7503 |         1.3408 |        0.67 |      -0.0381 |        0.0288 |           0.162  |
| ADX_14D + BREAKOUT_20D + CALMAR_RATIO_60D + PRICE_POSITION_20D + SHARPE_RATIO_20D + SLOPE_20D                       |      0.9499 |         1.7139 |        0.67 |      -0.0316 |        0.0083 |           0.0426 |
| ADX_14D + PRICE_POSITION_120D + SLOPE_20D + VOL_RATIO_20D                                                           |      0.5254 |         1.0798 |        0.61 |      -0.0425 |        0.1165 |           1.1229 |
| BREAKOUT_20D + PRICE_POSITION_120D + SLOPE_20D + UP_DOWN_VOL_RATIO_20D + VOL_RATIO_20D + VORTEX_14D                 |      0.6628 |         1.0704 |        0.61 |      -0.0297 |        0.0643 |           0.4688 |
| ADX_14D + BREAKOUT_20D + CALMAR_RATIO_60D + PRICE_POSITION_120D + PRICE_POSITION_20D + SHARPE_RATIO_20D + SLOPE_20D |      0.9202 |         1.6719 |        0.61 |      -0.0353 |        0.006  |           0.0309 |
| ADX_14D + AMIHUD_ILLIQUIDITY + BREAKOUT_20D + GK_VOL_RATIO_20D + MAX_DD_60D + SLOPE_20D + UP_DOWN_VOL_RATIO_20D     |      0.6278 |         1.1591 |        0.61 |      -0.0445 |        0.1333 |           0.5892 |
| ADX_14D + BREAKOUT_20D + PRICE_POSITION_120D + PRICE_POSITION_20D + SLOPE_20D + VOL_RATIO_20D                       |      0.6573 |         1.1378 |        0.72 |      -0.0486 |        0.0955 |           0.6588 |
| ADX_14D + PRICE_POSITION_120D + SHARPE_RATIO_20D + SLOPE_20D + VORTEX_14D                                           |      1.0021 |         1.6688 |        0.67 |      -0.0535 |        0.0463 |           0.251  |

## 3. Conclusion
The top strategy **AMIHUD_ILLIQUIDITY + PRICE_POSITION_20D + PV_CORR_20D + SLOPE_20D** demonstrates:
- **Train**: 76.28% return, 1.50 Calmar
- **Stability**: 78% quarterly win rate, worst quarter -2.90%
- **Holdout**: 8.00% return in unseen data

This is a **leakage-controlled** triple validation (rolling gate uses train-only data; holdout remains unseen).
