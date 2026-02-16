# Sector Constraint Research — NEGATIVE RESULT

**Date**: 2026-02-12
**Branch**: `feat/c2-shadow-commodity-constraint`
**Status**: CLOSED — no production change

## Hypothesis

S1 的同板块双持（尤其是 COMMODITY 金+银 518850/518880）是尾部回撤的放大器。
添加板块约束可以在不损失收益的前提下降低 MDD。

## Diagnostic Results

Script: `archive/scripts/negative_results/diagnose_sector_concentration.py`

| Metric | Value | Threshold |
|--------|-------|-----------|
| same_pool_rate | 33.3% (13/39) | — |
| same_pool_loss_share | 42.9% (3/7) | — |
| same_pool_dd_contrib | 57.1% (4/7) | — |
| **delta (loss_share - rate)** | **+9.5pp** | >15pp = build, <5pp = skip |

Verdict: Borderline — proceed to targeted A/B.

Main driver: COMMODITY pool (gold 518850 + silver 518880) appeared in 7/13 same-pool events.

## A/B Test Results

Script: `archive/scripts/negative_results/ab_commodity_max1.py`

Three VEC variants under S1_F5_ON (med cost, T1_OPEN):

| Metric | A: Baseline | B: Cmd max1 | C: 7-pool | B-A | C-A |
|--------|------------|-------------|-----------|-----|-----|
| HO Return | +42.7% | +42.0% | +30.1% | -0.7% | -12.7% |
| HO MDD | -11.8% | -15.3% | -10.5% | **-3.5%** | +1.3% |
| Worst Month | -2.7% | -2.7% | -2.9% | +0.0% | -0.2% |
| 10th pct Month | -2.3% | -2.3% | -2.2% | +0.0% | +0.1% |
| Oct-Dec MDD | -11.8% | -15.3% | -10.5% | **-3.5%** | +1.3% |
| Sharpe | 0.63 | 0.58 | 0.34 | -0.05 | -0.29 |
| Trades | 75 | 75 | 77 | +0 | +2 |

### Key Finding

COMMODITY max 1 **worsens** MDD by 3.5pp:
- Constraint fires in Dec 2025, forcing suboptimal substitution
- Monthly return: Dec +2.27% → -1.00% (B-A = -3.27pp)
- Drawdown trough extends from Dec 10 → Dec 17

Full 7-pool diversity marginally improves MDD (+1.3pp) but at unacceptable return cost (-12.7pp).

## Conclusion

1. **Same-pool double-hold is a feature, not a bug**: S1's factors naturally select correlated ETFs when they offer the best risk-adjusted opportunity
2. **Constraining optimal allocation hurts**: forcing diversification when the model sees concentrated value destroys alpha
3. **Borderline diagnostic correctly predicted borderline outcome**: +9.5pp delta was not strong enough signal to justify constraint
4. **Decision criterion validated**: the 15pp threshold would have correctly blocked this experiment; 9.5pp was borderline and confirmed negative

## Lessons

- Pool concentration diagnostic is useful as a **screening tool** — delta > 15pp threshold is well-calibrated
- For POS_SIZE=2, sector constraints are inherently high-cost (you're constraining 50% of the portfolio)
- Same-pool != same-risk: gold and silver may be in the same pool but can diverge on short horizons
- This research direction is **exhausted** for S1 under current POS_SIZE=2

## Scripts

- `archive/scripts/negative_results/diagnose_sector_concentration.py` — 3-metric diagnostic
- `archive/scripts/negative_results/ab_commodity_max1.py` — A/B test (baseline vs COMMODITY max 1 vs 7-pool)
- Results: `results/ab_commodity_max1/equity_curves.csv`
