# Algebraic Factor VEC Validation Results

**Date**: 2026-02-12
**Status**: Holdout analysis complete, 6 BT candidates identified
**Config**: F5 + Exp4 (dr=0.10, mh=9) + Regime Gate + T1_OPEN + med cost

## Critical Bug Found & Fixed

**`batch_vec_backtest.py` never passed hysteresis parameters to VEC kernel.**
- `delta_rank` defaulted to 0.0 (disabled), `min_hold_days` defaulted to 0 (disabled)
- All prior VEC batch runs were effectively F5_OFF (no hysteresis)
- Fix: read from `config["backtest"]["hysteresis"]` and pass to `run_vec_backtest()`
- Verified: S1 now matches sealed v5.0 exactly (Full +54.1%, HO +42.7%, 75 trades)

## Pipeline: Strategy B (Algebraic Factors → WFO → VEC)

1. Factor Mining: 78 algebraic survivors from 6-gate composability prefilter
2. WFO: 95-factor pool (17 base + 78 extra), 1.2M combos evaluated
3. VEC Selection: 27 diverse candidates across 4 groups
4. Holdout Split: Training ≤ 2025-04-30, Holdout: 2025-05 ~ 2026-02-10

## Candidate Selection Groups

- **Group A (Return-first)**: 12 top WFO `cum_oos_return`, diverse parents (max_parent_cnt ≤ 2)
- **Group B (Stability-first)**: 8 top WFO `positive_rate`, diverse parents
- **Group C (Non-CMF)**: 5 combos without CMF-family factors
- **Group D (Baselines)**: S1 (production) + C2 (shadow candidate)

## WFO Results (Pre-VEC)

- `CMF_20D__sub__GK_VOL_RATIO_20D` dominated 62% of top-1000 WFO combos
- 77% of top-1000 purely algebraic, 22% mixed
- S1 not in top 100k (expected: WFO doesn't apply hysteresis)

## VEC Results with Hysteresis (F5 + Exp4)

### Summary by Group

| Group | N | HO+ | PASS | Avg HO Ret | Avg HO MDD | Avg HO Sharpe | Avg WM |
|-------|---|-----|------|-----------|------------|--------------|--------|
| S1 baseline | 1 | 1/1 | 1/1 | +42.7% | 11.8% | +2.15 | -2.9% |
| C2 baseline | 1 | 1/1 | 1/1 | +63.9% | 10.4% | +2.63 | -5.5% |
| B (stability) | 5 | 5/5 | 3/5 | +12.5% | 13.4% | +0.99 | -3.9% |
| C (non-CMF) | 3 | 2/3 | 1/3 | +14.1% | 15.5% | +0.50 | -8.2% |
| A (return) | 10 | 9/10 | 4/10 | +26.9% | 14.1% | +1.67 | -6.3% |

### PASS criteria: HO return > 0 AND HO MDD < 15% AND worst month > -8%

### BT Candidates (Sharpe >= 1.5, MDD < 15%, WM > -8%)

| Rank | Combo | Full | HO Ret | HO MDD | HO Sharpe | WM | Group |
|------|-------|------|--------|--------|-----------|-----|-------|
| 1 | CMF_sub_GK + CMF_sub_VOL + CORR_mul_DOWNSIDE + OBV | +114.0% | +39.0% | 4.3% | +2.95 | -1.4% | A |
| 2 | AMIHUD + CALMAR + CORR_MKT (C2) | +22.6% | +63.9% | 10.4% | +2.63 | -5.5% | D |
| 3 | CALMAR_sub_TSMOM + CMF_mul_PP120 + DOWNSIDE_max_SLOPE + MDD_mul_SPREAD | +69.5% | +32.2% | 10.1% | +2.24 | -1.8% | B |
| 4 | BREAKOUT_sub_TSMOM + CMF_add_CORR + CMF_sub_GK + VOL_RATIO | +47.9% | +37.4% | 7.4% | +2.22 | -5.4% | A |
| 5 | ADX + OBV + SHARPE + SLOPE (S1) | +54.1% | +42.7% | 11.8% | +2.15 | -2.9% | D |
| 6 | CMF_min_TSMOM + CMF_sub_GK + CORR_mul_DOWNSIDE + PV_CORR | +60.1% | +24.9% | 9.7% | +1.99 | -4.3% | A |

## Key Findings

### 1. Hysteresis reverses group ranking
- Without hysteresis: Group B (stability-first) dominated, Group A suffered
- With hysteresis: Group A has best avg HO Sharpe (+1.67), Group B drops to +0.99
- **Conclusion**: factor-execution compatibility matters more than WFO selection method
- CMF-family factors work WELL with Exp4 hysteresis (stable ranks, like S1's ADX/OBV/SHARPE/SLOPE)

### 2. CMF family is NOT dead
- Previous non-hysteresis analysis falsely concluded CMF stacking = invalid direction
- With hysteresis: top BT candidate is CMF-based (HO Sharpe 2.95, MDD 4.3%)
- CMF factors have stable cross-sectional ranks → compatible with Exp4's rank-gap mechanism
- **Key insight**: factor suitability depends on execution framework, not just factor quality

### 3. New production-grade candidates found
- Rank 1 algebraic combo (CMF_sub_GK + CMF_sub_VOL + CORR_mul_DOWNSIDE + OBV) beats S1 on HO MDD (4.3% vs 11.8%) and matches on Sharpe
- C2 still strongest on raw HO return (+63.9%)
- B3 (CALMAR_sub_TSMOM combo) has excellent risk profile (MDD 10.1%, WM -1.8%)

### 4. Parent-factor dedup infrastructure validated
- `max_parent_occurrence` parameter added to WFO optimizer
- `check_parent_diversity()` function available for production screening
- Not needed for these specific candidates (diversity was selected manually)

## Recommended Next Steps

1. **BT ground truth**: Run top 6 candidates through Backtrader event-driven simulation
2. **Shadow deployment**: Winner(s) shadow alongside S1 for 8-12 weeks
3. **Research constraint**: Always run VEC with hysteresis enabled (F5+Exp4) for production-relevant results

## Anti-Pattern Documented

**batch_vec_backtest.py hysteresis omission**: The VEC batch script defaulted hysteresis params to 0 (disabled) since Exp4 was introduced. All historical VEC batch results were without hysteresis. This was fixed on 2026-02-12 by reading `config["backtest"]["hysteresis"]` and passing `delta_rank`/`min_hold_days` to `run_vec_backtest()`.
