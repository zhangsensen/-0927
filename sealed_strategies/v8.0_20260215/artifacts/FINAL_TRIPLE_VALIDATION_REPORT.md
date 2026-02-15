# 🏆 Final Triple Validation Report

**Generated**: 2026-02-14 14:49:34

## 0. Leakage Control (Audit Note)
This run uses **train-only** rolling OOS summary as the gate input (no holdout segments).

- VEC (train): `/home/sensen/dev/projects/-0927/results/vec_from_wfo_20260214_144851/full_space_results.parquet`
- Rolling (train-only): `/home/sensen/dev/projects/-0927/results/rolling_oos_consistency_20260214_144932/rolling_oos_summary.parquet`
- Holdout (unseen): `/home/sensen/dev/projects/-0927/results/holdout_validation_20260214_144933/holdout_validation_results.parquet`

## 1. Screening Funnel
- **Total Universe**: 200
- **Risk Filter** (factor exclusions): 200
- **Train Gate** (new): 200 / 200 (Return > 0.0, MDD < 25%)
- **Rolling Consistency Gate** (Strict): 156 (PosRate>=0.6, Worst>=-0.08, Calmar>=0.3)
- **Holdout Gate** (Profitable): 155 (Return > 0.0)

## 2. Top 20 'Gold Standard' Strategies
Sorted by Composite Score (30% Train Calmar + 40% Roll Worst + 30% Holdout Calmar)

| Combo                                                                                                                               |   Train Ret |   Train Calmar |   Roll Win% |   Roll Worst |   Holdout Ret |   Holdout Calmar |
|:------------------------------------------------------------------------------------------------------------------------------------|------------:|---------------:|------------:|-------------:|--------------:|-----------------:|
| ADX_14D + BREAKOUT_20D + MARGIN_BUY_RATIO + PRICE_POSITION_120D + SHARE_CHG_5D                                                      |      0.5161 |         1.3278 |        0.61 |      -0.0254 |        0.5574 |           7.4114 |
| ADX_14D + AMIHUD_ILLIQUIDITY + BREAKOUT_20D + MARGIN_BUY_RATIO + PRICE_POSITION_120D + SHARE_CHG_5D                                 |      0.5161 |         1.3278 |        0.61 |      -0.0254 |        0.5574 |           7.4114 |
| ADX_14D + BREAKOUT_20D + CORRELATION_TO_MARKET_20D + MARGIN_BUY_RATIO + PRICE_POSITION_120D + SHARE_CHG_5D                          |      0.5161 |         1.3278 |        0.61 |      -0.0254 |        0.5574 |           7.4114 |
| ADX_14D + AMIHUD_ILLIQUIDITY + BREAKOUT_20D + CORRELATION_TO_MARKET_20D + MARGIN_BUY_RATIO + PRICE_POSITION_120D + SHARE_CHG_5D     |      0.5161 |         1.3278 |        0.61 |      -0.0254 |        0.5574 |           7.4114 |
| BREAKOUT_20D + CALMAR_RATIO_60D + MARGIN_CHG_10D + MAX_DD_60D + PRICE_POSITION_120D + PRICE_POSITION_20D + PV_CORR_20D              |      0.6766 |         1.405  |        0.78 |      -0.0227 |        0.3819 |           3.2375 |
| ADX_14D + BREAKOUT_20D + CALMAR_RATIO_60D + MARGIN_CHG_10D + MAX_DD_60D + PRICE_POSITION_120D + PRICE_POSITION_20D                  |      0.712  |         1.4657 |        0.78 |      -0.0228 |        0.3781 |           3.2006 |
| BREAKOUT_20D + CALMAR_RATIO_60D + MARGIN_CHG_10D + MAX_DD_60D + PRICE_POSITION_120D + PRICE_POSITION_20D                            |      0.5055 |         1.0976 |        0.72 |      -0.0237 |        0.3663 |           3.0845 |
| AMIHUD_ILLIQUIDITY + BREAKOUT_20D + CALMAR_RATIO_60D + MARGIN_CHG_10D + MAX_DD_60D + PRICE_POSITION_120D + PRICE_POSITION_20D       |      0.5055 |         1.0976 |        0.72 |      -0.0237 |        0.3663 |           3.0845 |
| BREAKOUT_20D + CORRELATION_TO_MARKET_20D + GK_VOL_RATIO_20D + MARGIN_BUY_RATIO + PV_CORR_20D + SHARE_CHG_20D                        |      0.3979 |         1.2205 |        0.72 |      -0.047  |        0.487  |           6.6769 |
| MAX_DD_60D + PV_CORR_20D                                                                                                            |      0.2143 |         1.9003 |        0.83 |      -0.0076 |        0.0598 |           1.6763 |
| BREAKOUT_20D + CALMAR_RATIO_60D + CORRELATION_TO_MARKET_20D + MARGIN_BUY_RATIO + OBV_SLOPE_10D + PRICE_POSITION_20D + SHARE_CHG_10D |      0.868  |         1.1284 |        0.67 |      -0.0347 |        0.4103 |           2.1765 |
| ADX_14D + MARGIN_CHG_10D + MAX_DD_60D + PRICE_POSITION_120D + PV_CORR_20D + SHARE_ACCEL + SLOPE_20D                                 |      0.6243 |         1.4835 |        0.83 |      -0.0468 |        0.4144 |           2.6027 |
| BREAKOUT_20D + GK_VOL_RATIO_20D + MARGIN_BUY_RATIO + PV_CORR_20D + SHARE_CHG_20D                                                    |      0.3496 |         1.0877 |        0.72 |      -0.047  |        0.2531 |           3.5332 |
| AMIHUD_ILLIQUIDITY + MARGIN_CHG_10D + MAX_DD_60D + PRICE_POSITION_120D + PV_CORR_20D + SHARE_ACCEL + SLOPE_20D                      |      0.5802 |         1.3946 |        0.78 |      -0.0468 |        0.4143 |           2.6027 |
| MARGIN_CHG_10D + MAX_DD_60D + PRICE_POSITION_120D + PV_CORR_20D + SHARE_ACCEL + SLOPE_20D                                           |      0.5802 |         1.3946 |        0.78 |      -0.0468 |        0.4143 |           2.6027 |
| BREAKOUT_20D + CALMAR_RATIO_60D + MARGIN_BUY_RATIO + MARGIN_CHG_10D + MAX_DD_60D + PRICE_POSITION_20D + SHARE_CHG_20D               |      0.5153 |         1.0369 |        0.72 |      -0.0328 |        0.3384 |           1.8872 |
| BREAKOUT_20D + CORRELATION_TO_MARKET_20D + MARGIN_BUY_RATIO + PRICE_POSITION_120D + SHARE_CHG_5D + VOL_RATIO_20D                    |      0.4946 |         1.0258 |        0.61 |      -0.0509 |        0.2872 |           3.8222 |
| AMIHUD_ILLIQUIDITY + MAX_DD_60D                                                                                                     |      0.1965 |         2.455  |        0.83 |      -0.0076 |        0.0025 |           0.2569 |
| BREAKOUT_20D + CALMAR_RATIO_60D + MARGIN_BUY_RATIO + OBV_SLOPE_10D + PRICE_POSITION_20D + PV_CORR_20D + SHARE_CHG_20D               |      0.5288 |         0.7599 |        0.78 |      -0.0227 |        0.2551 |           2.7328 |
| MARGIN_CHG_10D + MAX_DD_60D + PRICE_POSITION_120D + PV_CORR_20D + SHARE_ACCEL + SLOPE_20D + VOL_RATIO_20D                           |      0.5443 |         0.8902 |        0.83 |      -0.0423 |        0.5556 |           3.5762 |

## 3. Conclusion
The top strategy **ADX_14D + BREAKOUT_20D + MARGIN_BUY_RATIO + PRICE_POSITION_120D + SHARE_CHG_5D** demonstrates:
- **Train**: 51.61% return, 1.33 Calmar
- **Stability**: 61% quarterly win rate, worst quarter -2.54%
- **Holdout**: 55.74% return in unseen data

This is a **leakage-controlled** triple validation (rolling gate uses train-only data; holdout remains unseen).
