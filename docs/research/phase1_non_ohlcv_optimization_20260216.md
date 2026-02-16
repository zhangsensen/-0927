# Phase 1: Non-OHLCV Factor Optimization Research

> Date: 2026-02-16
> Branch: `feat/factor-research-v9`
> Baseline: v8.0 sealed (composite_1 + core_4f)
> Pipeline: 200 combos × 4-gate validation (Train + Rolling + Holdout + BT)

## Overview

Phase 1 tests three hypotheses about optimizing existing non-OHLCV factors within the current 23-factor active set, benchmarked against v8.0 sealed strategies.

## Exp-A1: SHARE_CHG_10D vs SHARE_CHG_5D

**Hypothesis**: 10-day share change window may capture more stable flow signals than 5-day.

**Result**: REJECTED

| Metric | SHARE_CHG_5D (composite_1) | SHARE_CHG_10D variant |
|--------|---------------------------|----------------------|
| WFO Rank | #14 / 100k | #6,873 / 100k |
| WFO OOS Return | +56.7% | -10.7% |
| Delta | — | **-67.4pp** |

**Conclusion**: SHARE_CHG_5D vastly superior. 5-day window captures faster institutional outflow signals. SHARE_CHG_10D permanently excluded from composite_1 variants.

## Exp-A2: SHARE_ACCEL Integration

**Hypothesis**: SHARE_ACCEL (second derivative of fund share change) adds orthogonal alpha.

**Result**: PARTIAL — adds Rolling stability but not risk-adjusted improvement.

### Coverage

- 72/200 top combos contain SHARE_ACCEL
- 60/72 pass all 4 validation gates
- Best partners: MARGIN_CHG_10D (95% co-occurrence), SLOPE_20D (80%), PP120 (60%)

### Top SHARE_ACCEL Candidates vs v8.0

| Strategy | HO Ret | Sharpe | MDD | PF | Trades | Roll | HO Calmar |
|----------|--------|--------|-----|-----|--------|------|-----------|
| **composite_1 (v8.0)** | **53.9%** | **1.38** | **10.8%** | **4.88** | 77 | 61% | **7.41** |
| core_4f (v8.0) | 67.4% | 1.09 | 14.9% | 3.07 | 75 | 78% | 4.56 |
| ACCEL#1: MCH+MDD+PP120+PVC+SA+SL+VR (7F) | 54.6% | 1.12 | 15.5% | 4.34 | 63 | **83%** | 3.58 |
| ACCEL#2: ADX+MCH+PP20+SA+SL (5F) | 54.5% | 1.07 | 14.4% | 2.69 | 74 | 61% | 3.78 |
| ACCEL#3: BRK+MBR+MCH+OBV+SA+SC20+SH (7F) | 58.5% | 0.98 | 17.9% | 2.91 | 79 | 67% | — |

**Conclusion**: SHARE_ACCEL's best combo (ACCEL#1) achieves 83% Rolling stability — far better than composite_1's 61%. However, Sharpe (1.12 vs 1.38) and MDD (15.5% vs 10.8%) are worse. No SHARE_ACCEL combo dominates composite_1 on risk-adjusted metrics.

## Exp-A3: Full Non-OHLCV Grid Search

**Result**: v8.0 strategies confirmed as optimal within current factor space.

### Deduplication

200 combos → 146 unique strategies (54 duplicates from adding AMIHUD/CORR_MKT noise factors with near-zero ICIR weight — confirms ICIR weighting works correctly).

### Risk-Adjusted Composite Ranking

Score = HO_Calmar(30%) + Sharpe(25%) + Low_MDD(25%) + Roll_WinRate(20%)

| Rank | Strategy | HO | Sharpe | MDD | Roll | HO Calmar |
|------|----------|-----|--------|-----|------|-----------|
| #1 | BRK+CORR+GKV+MBR+PVC+SC20 (6F) | 47.0% | 1.25 | 10.6% | 72% | 6.68 |
| #2 | MCH+MDD+PP120+PVC+SC20+SL+VR (7F) | 51.0% | 1.04 | 12.5% | 83% | 4.80 |
| #7 | **core_4f (v8.0)** | **67.4%** | 1.09 | 14.9% | 78% | 4.56 |
| #8 | ACCEL#1 (best SHARE_ACCEL) | 54.6% | 1.12 | 15.5% | 83% | 3.58 |
| **#10** | **composite_1 (v8.0)** | 53.9% | **1.38** | **10.8%** | 61% | **7.41** |

### Key Findings

1. **composite_1 has BEST Sharpe (1.38) and BEST HO Calmar (7.41)** across all 200 candidates
2. Strategies ranked above it trade lower MDD for lower absolute return — not genuine outperformance
3. **Two distinct alpha families identified**:
   - **Family A** (composite_1): BREAKOUT + MARGIN_BUY + SHARE_CHG_5D → high Sharpe, low MDD
   - **Family B** (core_4f): MARGIN_CHG + PP120 + SLOPE → high absolute return, high Rolling stability
4. Non-OHLCV concentration is monotonically beneficial: 0-factor median OOS 7.1% → 3-factor 12.7%
5. Best non-OHLCV pair: MARGIN_BUY_RATIO + SHARE_CHG_5D (36/200 combos, 18%)

## Pipeline Health

| Metric | Value | Threshold |
|--------|-------|-----------|
| VEC-BT train gap (mean) | 0.07pp | < 5pp |
| VEC-BT train gap (median) | 0.00pp | < 5pp |
| Combos within 2pp | 197/200 (98.5%) | > 95% |
| BT margin failures | 0 | 0 |
| Rolling pass rate | 156/200 (78%) | — |
| Holdout pass rate | 155/156 (99.4%) | — |
| Final candidates | 155 | — |

## Conclusion

**v8.0 sealed strategies remain optimal within the current 23-factor active set.**

- No candidate simultaneously beats composite_1 on Sharpe + Calmar + MDD
- SHARE_ACCEL is a stability enhancer but not a risk-adjusted improver
- SHARE_CHG_5D >> SHARE_CHG_10D (permanent)
- Current factor space is near-saturated — genuine improvement requires new data sources (Phase 2: IOPV, FX, northbound flow, option IV)

## Results Archive

Pipeline results stored locally (not in git):
- WFO: `results/run_20260214_115216/` (reused, 100k combos)
- VEC: `results/vec_from_wfo_20260216_094250/`
- BT: `results/bt_backtest_top200_20260216_094331/`
- Rolling: `results/rolling_oos_consistency_20260216_094332/`
- Holdout: `results/holdout_validation_20260216_094334/`
- Final: `results/final_triple_validation_20260216_094334/`
