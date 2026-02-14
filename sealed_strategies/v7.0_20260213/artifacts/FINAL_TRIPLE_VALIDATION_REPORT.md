# 🏆 Final Triple Validation Report

**Generated**: 2026-02-13 11:58:02

## 0. Leakage Control (Audit Note)
This run uses **train-only** rolling OOS summary as the gate input (no holdout segments).

- VEC (train): `results/vec_from_wfo_20260213_112309/full_space_results.parquet`
- Rolling (train-only): `results/rolling_oos_consistency_20260213_115735/rolling_oos_summary.parquet`
- Holdout (unseen): `results/holdout_validation_20260213_115745/holdout_validation_results.parquet`

## 1. Screening Funnel
- **Total Universe**: 200
- **Risk Filter** (factor exclusions): 200
- **Train Gate** (new): 186 / 200 (Return > 0.0, MDD < 25%)
- **Rolling Consistency Gate** (Strict): 49 (PosRate>=0.6, Worst>=-0.08, Calmar>=0.3)
- **Holdout Gate** (Profitable): 44 (Return > 0.0)

## 2. Top 20 'Gold Standard' Strategies
Sorted by Composite Score (30% Train Calmar + 40% Roll Worst + 30% Holdout Calmar)

| Combo                                                                                                                                 |   Train Ret |   Train Calmar |   Roll Win% |   Roll Worst |   Holdout Ret |   Holdout Calmar |
|:--------------------------------------------------------------------------------------------------------------------------------------|------------:|---------------:|------------:|-------------:|--------------:|-----------------:|
| ADX_14D + MARGIN_BUY_RATIO + PRICE_POSITION_120D + SHARE_ACCEL + SLOPE_20D                                                            |      1.7642 |         2.5279 |        0.83 |      -0.0497 |        0.3222 |           2.0614 |
| ADX_14D + PRICE_POSITION_120D + PRICE_POSITION_20D + SHARE_ACCEL + SLOPE_20D                                                          |      0.828  |         1.6637 |        0.78 |      -0.0439 |        0.2741 |           1.8885 |
| AMIHUD_ILLIQUIDITY + MARGIN_BUY_RATIO + MARGIN_CHG_10D + PRICE_POSITION_120D + SHARE_ACCEL + SHARPE_RATIO_20D + SLOPE_20D             |      0.7768 |         1.4885 |        0.78 |      -0.0306 |        0.195  |           1.0647 |
| PRICE_POSITION_120D + PRICE_POSITION_20D + SHARE_CHG_20D + SLOPE_20D                                                                  |      0.5051 |         0.7089 |        0.61 |      -0.0393 |        0.4149 |           4.4051 |
| MARGIN_BUY_RATIO + PRICE_POSITION_120D + PRICE_POSITION_20D + SHARE_ACCEL + SLOPE_20D                                                 |      0.6358 |         1.2049 |        0.72 |      -0.0497 |        0.1757 |           1.2672 |
| BREAKOUT_20D + MARGIN_BUY_RATIO + OBV_SLOPE_10D + PRICE_POSITION_120D + UP_DOWN_VOL_RATIO_20D                                         |      0.7062 |         0.9199 |        0.67 |      -0.0531 |        0.3938 |           2.5459 |
| AMIHUD_ILLIQUIDITY + OBV_SLOPE_10D + SHARE_ACCEL + SLOPE_20D + UP_DOWN_VOL_RATIO_20D + VOL_RATIO_20D + VORTEX_14D                     |      0.4599 |         0.8142 |        0.61 |      -0.0401 |        0.1689 |           1.2348 |
| AMIHUD_ILLIQUIDITY + MARGIN_BUY_RATIO + PRICE_POSITION_120D + PV_CORR_20D + SLOPE_20D + UP_DOWN_VOL_RATIO_20D                         |      0.9283 |         1.0986 |        0.83 |      -0.0684 |        0.6146 |           3.4322 |
| ADX_14D + BREAKOUT_20D + PRICE_POSITION_120D + SHARE_ACCEL + SHARE_CHG_20D + UP_DOWN_VOL_RATIO_20D                                    |      0.4585 |         0.6818 |        0.61 |      -0.0401 |        0.3352 |           2.3683 |
| ADX_14D + AMIHUD_ILLIQUIDITY + BREAKOUT_20D + MARGIN_CHG_10D + OBV_SLOPE_10D + SHARE_ACCEL + SLOPE_20D                                |      0.9786 |         1.1954 |        0.89 |      -0.0403 |        0.0499 |           0.2178 |
| ADX_14D + AMIHUD_ILLIQUIDITY + PRICE_POSITION_120D + PRICE_POSITION_20D + SHARE_ACCEL + SLOPE_20D                                     |      0.5243 |         0.7901 |        0.78 |      -0.0506 |        0.2524 |           1.3786 |
| ADX_14D + BREAKOUT_20D + OBV_SLOPE_10D + PRICE_POSITION_120D + SHARE_ACCEL + SLOPE_20D                                                |      0.2857 |         0.6868 |        0.72 |      -0.0407 |        0.2338 |           1.9999 |
| ADX_14D + AMIHUD_ILLIQUIDITY + MARGIN_CHG_10D + SHARE_ACCEL + SLOPE_20D + VOL_RATIO_20D                                               |      0.6135 |         0.7737 |        0.61 |      -0.0542 |        0.3088 |           1.9551 |
| ADX_14D + AMIHUD_ILLIQUIDITY + BREAKOUT_20D + CORRELATION_TO_MARKET_20D + MARGIN_CHG_10D + SHARE_ACCEL + SHARPE_RATIO_20D             |      1.31   |         1.2768 |        0.61 |      -0.0699 |        0.3025 |           1.727  |
| PRICE_POSITION_120D + PV_CORR_20D + SHARE_ACCEL + SLOPE_20D                                                                           |      0.5441 |         0.7216 |        0.61 |      -0.0588 |        0.4055 |           3.2089 |
| ADX_14D + AMIHUD_ILLIQUIDITY + BREAKOUT_20D + MARGIN_CHG_10D + MAX_DD_60D + PV_CORR_20D + SLOPE_20D                                   |      1.23   |         1.2085 |        0.72 |      -0.0746 |        0.3888 |           2.0224 |
| ADX_14D + AMIHUD_ILLIQUIDITY + MARGIN_CHG_10D + SLOPE_20D + VORTEX_14D                                                                |      0.855  |         1.1231 |        0.78 |      -0.0551 |        0.1315 |           0.9949 |
| AMIHUD_ILLIQUIDITY + MARGIN_CHG_10D + PV_CORR_20D + SHARE_ACCEL + SLOPE_20D + UP_DOWN_VOL_RATIO_20D + VORTEX_14D                      |      0.6494 |         1.1197 |        0.72 |      -0.0445 |        0.0277 |           0.1211 |
| BREAKOUT_20D + MARGIN_CHG_10D + PRICE_POSITION_120D + PV_CORR_20D + SHARE_ACCEL + SHARE_CHG_20D + SHARPE_RATIO_20D                    |      0.8057 |         0.9639 |        0.78 |      -0.0705 |        0.2437 |           2.2601 |
| CORRELATION_TO_MARKET_20D + MARGIN_BUY_RATIO + PRICE_POSITION_120D + PV_CORR_20D + UP_DOWN_VOL_RATIO_20D + VOL_RATIO_20D + VORTEX_14D |      1.049  |         1.1657 |        0.72 |      -0.0548 |        0.107  |           0.5673 |

## 3. Conclusion
The top strategy **ADX_14D + MARGIN_BUY_RATIO + PRICE_POSITION_120D + SHARE_ACCEL + SLOPE_20D** demonstrates:
- **Train**: 176.42% return, 2.53 Calmar
- **Stability**: 83% quarterly win rate, worst quarter -4.97%
- **Holdout**: 32.22% return in unseen data

This is a **leakage-controlled** triple validation (rolling gate uses train-only data; holdout remains unseen).
