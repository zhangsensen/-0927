# v5.0 DEPRECATED (2026-02-12)

**Status**: DEPRECATED — S1 holdout results were artifacts of a bounded_factors bug.

## What Happened

The v5.0 sealed S1 strategy reported HO +42.7% (Sharpe 2.15). This result was **invalid** due to:

1. **bounded_factors bug**: ADX_14D was incorrectly Winsorized instead of rank-standardized.
   - v5.0 sealed `bounded_factors` had only 4 entries: PP_20D, PP_120D, PV_CORR_20D, RSI_14
   - Correct set has 7 entries: +ADX_14D, +CMF_20D, +CORRELATION_TO_MARKET_20D
   - ADX_14D is one of S1's 4 constituent factors — Winsorization distorted its cross-sectional ranking

2. **Corrected S1 performance**: HO +30.4% (Sharpe 1.84) — a -12.3pp degradation
   - bounded_factors fix accounts for 97% of the drop
   - ETF pool expansion (43→49) contributes only -0.4pp

3. **43-ETF pool is now stale**: v6.0 uses 49 ETFs (41 A-share + 8 QDII)

## Superseded By

**v6.0_20260212**: C2 (AMIHUD_ILLIQUIDITY + CALMAR_RATIO_60D + CORRELATION_TO_MARKET_20D) as production strategy.

## Do NOT Use This Seal For

- Production trading decisions
- Performance benchmarking
- Strategy comparison baselines

The v5.0 sealed code/configs remain preserved for audit trail purposes only.

---
Deprecated by: Claude Code
Date: 2026-02-12
