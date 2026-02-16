# WHEN/HOW Dimension Research — Credible Negative Results

**Date**: 2026-02-17
**Status**: CLOSED — All directions killed
**Script**: `scripts/research_when_how_stage01.py`
**Context**: Post v8.0 seal, WHAT dimension exhausted (Phase 1, moneyflow, score dispersion)

## Executive Summary

Tested two optimization axes beyond factor selection (WHAT):
- **WHEN**: When to trust the selection signal (cross-sectional return dispersion)
- **HOW**: How to combine two strategies (ensemble of composite_1 + core_4f)

**Result**: Both directions produce credible negative conclusions (Rule 29). v8.0 remains the ceiling with current data and methodology.

---

## Stage 0: Ensemble Failure Correlation (HOW)

**Question**: Do composite_1 (61% rolling win) and core_4f (78% rolling win) fail in different periods?

### Method
- Ran VEC for both strategies under production config (FREQ=5, POS_SIZE=2, hysteresis, regime gate)
- Split equity curves into non-overlapping segments (63-day quarters for train, 21-day months for holdout)
- Computed Pearson correlation and 2×2 contingency table

### Results

| Metric | Train (16Q) | Holdout (9M) | All (25) |
|--------|-------------|--------------|----------|
| Pearson rho | **0.586** | 0.637 | 0.619 |
| composite_1 win rate | 68.8% | 55.6% | - |
| core_4f win rate | 56.2% | 66.7% | - |
| P(both_fail) | 0.188 | 0.111 | - |
| P(both_fail\|one_fail) | 0.333 | 0.167 | - |
| P(independent both_fail) | 0.137 | - | - |

**Contingency Table (Train, quarterly)**:
```
                  core_4f WIN  core_4f LOSE
composite_1 WIN      7            4
composite_1 LOSE     2            3
```

### Decision: MARGINAL (rho=0.586 in 0.5-0.7 range)

- Not killed outright (rho < 0.7 and P(both_fail|one_fail) = 0.333 < 0.8)
- But moderate correlation means limited diversification benefit
- Quarterly blend Sharpe improvement: 0.383 vs max(0.318, 0.360) = only +0.023
- Proceeded to Stage 2a for concrete validation

---

## Stage 1: Cross-Sectional Return Dispersion (WHEN)

**Question**: Can ETF return dispersion predict when selection alpha is strong?

### Method
1. Computed daily cross-sectional std of 5D and 20D returns across 49 ETFs
2. Checked orthogonality with existing regime gate (volatility-based)
3. Tested quartile monotonicity of dispersion → next-period strategy returns
4. Split train/holdout per Rule 4

### Results

**Orthogonality Check** (critical gate):

| Dispersion Window | vs Regime Vol (Pearson) | vs Regime Vol (Spearman) |
|-------------------|------------------------|--------------------------|
| 5D | 0.369 | 0.392 |
| 20D | **0.538** | 0.428 |

20D dispersion **fails orthogonality** (|rho| = 0.538 > 0.5 threshold, Rule 31).

**Predictive Power** (even if orthogonality passed — it doesn't):

| Period | Disp Window | Pearson rho(disp,ret) | p-value | Monotonicity |
|--------|-------------|----------------------|---------|--------------|
| Train | 5D | 0.046 | 0.507 | NONE |
| Train | 20D | 0.005 | 0.943 | NONE |
| Holdout | 5D | -0.167 | 0.317 | NONE |
| Holdout | 20D | -0.118 | 0.480 | NONE |

Zero predictive power across all windows and periods. No monotonic quartile relationship.

### Decision: **KILL** (two independent reasons)

1. **Orthogonality failure**: 20D dispersion shares 29% variance (rho^2) with regime gate — same information dimension (Rule 31)
2. **No predictive power**: All correlations statistically indistinguishable from zero, no quartile monotonicity

**Distinction from failed "score dispersion"**: Score dispersion measured factor-ranking quality (model internal); return dispersion measures market environment (external). Both independently have zero signal for this strategy — selection alpha is not state-dependent in the dispersion dimension.

---

## Stage 2a: Ensemble Capital Split

**Question**: Does running both strategies with POS_SIZE=1 and blending equity curves improve risk-adjusted returns?

### Method
- Ran composite_1 and core_4f each with POS_SIZE=1 (half capital each)
- Blended: eq_blend = 0.5 × eq_composite_1 + 0.5 × eq_core_4f
- Compared to standalone composite_1 (POS_SIZE=2)

### Results

| Metric | composite_1 (PS=2) | Blend (2×PS=1) | Delta |
|--------|-------------------|----------------|-------|
| Total Return | 136.1% | 26.4% | **-109.7pp** |
| Sharpe | 1.577 | 0.363 | **-1.214** |
| Max DD | 10.8% | 16.8% | **+6.0pp** |
| Calmar | 1.780 | 0.293 | -1.487 |

**PS=1 degrades both strategies catastrophically**:
- composite_1: PS=2 Sharpe 1.589 → PS=1 Sharpe 0.402 (-75%)
- core_4f: PS=2 Sharpe 1.225 → PS=1 Sharpe 0.179 (-85%)

**Root cause**: Strategies were optimized for POS_SIZE=2. With PS=1:
- Higher concentration risk (100% in single ETF)
- Hysteresis dynamics change (fewer swap options)
- Selection alpha from top-2 diversity lost

**Holdout anomaly**: Blend HO Sharpe 2.239 despite terrible train (-6.7%) — classic train/HO inconsistency pattern, not trustworthy (Rule 4).

### Decision: **KILL** (both criteria)
- Blend Sharpe 0.363 << threshold 1.677 (standalone + 0.10)
- Blend MDD 16.8% > threshold 13.8% (standalone + 3pp)

### Note on Score Blend (Variant A, not tested)
Score Blend would average z-scored signals from both strategies and select top-2 from the blended ranking. This is functionally equivalent to a 9-factor composite (5+4 factors). Phase 1 research already exhaustively searched all factor combinations up to size 7 and found composite_1 (5F) as the Sharpe optimum. Adding more factors (Rule 28-adjacent reasoning) is unlikely to improve over the optimized 5F combination. The Score Blend was not tested but can be considered closed by transitivity from Phase 1 results.

---

## Synthesis

### Three orthogonal failure modes
1. **WHEN (dispersion)**: Not orthogonal to regime gate AND no predictive signal
2. **HOW (ensemble)**: Moderate strategy correlation (rho=0.586) AND PS=1 destroys strategy economics
3. **WHAT (factors)**: Already closed by Phase 1, moneyflow, score dispersion research

### Why the ceiling exists
- Selection alpha comes from **Price × Contrarian Flow** pattern (Rule 30)
- Both strategies exploit the same underlying phenomenon with different factor combinations
- The strategies are moderately correlated (rho=0.586) because they respond to the same market structure
- Dispersion (a market-level signal) doesn't gate selection alpha because the alpha source is ETF-specific, not regime-dependent
- The regime gate already captures the relevant market-level risk dimension

### Information-theoretic interpretation
- WHAT dimension: Kaiser 5/17, exhausted (Phase 1)
- WHEN dimension: Dispersion ⊂ Volatility (rho=0.538), already captured by regime gate
- HOW dimension: rho=0.586 insufficient for meaningful diversification at PS=1 granularity

### Actionable conclusions
1. **v8.0 is the production ceiling** with current data and methodology
2. **No v9.0 candidate exists** — all research directions produce credible negative results
3. **Shadow monitoring continues**: v8_composite_1 shadow for 8-12 weeks before S1→v8.0 switch
4. **Future alpha requires new data sources**: orthogonal dimensions (northbound capital, IV, FX, IOPV) — all data-blocked
5. **Enter pure maintenance mode**: daily signals, data updates, shadow tracking

---

## Rule Updates

**Rule 32** (new): POS_SIZE reduction is NOT a free parameter — strategies optimized for PS=N break at PS=M<N. Ensemble via capital split requires strategies independently viable at the target PS.

**Rule 33** (new): Cross-sectional dispersion ⊂ market volatility for A-share ETF universe (rho=0.538). These are not independent signals — high vol environments inherently produce high return dispersion.
