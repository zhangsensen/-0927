# v7.0 DEPRECATED

**Superseded by**: v8.0_20260215
**Deprecated on**: 2026-02-15

## Reason

v7.0 was sealed on 2026-02-13, before three critical pipeline fixes applied on 2026-02-14:

1. **IC-sign factor direction fix**: WFO used abs(IC) discarding sign, causing 5/6 non-OHLCV factors to be systematically inverted in VEC/BT scoring.
2. **VEC metadata propagation**: `factor_signs` and `factor_icirs` were not passed from WFO output through VEC to downstream validation, causing VEC-BT holdout gap of 20.3pp.
3. **BT execution-side hysteresis**: `_compute_rebalance_targets()` used signal-side state (`_signal_portfolio`) creating a self-referential feedback loop, causing +25.7pp VEC-BT gap.

**v7.0's VEC-BT gap was +25.6pp**, exceeding the 10pp red flag threshold. v8.0 candidates all have gap < 2pp.
