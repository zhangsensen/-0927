# v4.2 Sealed Release — Post-Bugfix Clean Validation

**Version**: v4.2_20260205
**Sealed**: 2026-02-05
**Status**: Production Ready

## Key Changes vs v3.4

- **All P0 bugs fixed**: lookahead bias, BT defaults, double regime gate, bounded factors
- **16-factor orthogonal set** (was 25 with redundancy): 97.7% combo space reduction
- **2 new factors**: UP_DOWN_VOL_RATIO_20D, GK_VOL_RATIO_20D (passed 10-dim quality check)
- **Removed**: OBV_SLOPE_10D (normalization bug), SPREAD_PROXY (quality score 1.5)
- **First release with full Triple Validation** (Rolling OOS + Holdout)

## Production Strategies (5)

| # | Combo | BT | Holdout | MDD | Sharpe | Note |
|---|-------|-----|---------|-----|--------|------|
| 1 | ADX+AMIHUD+GK_VOL+PP20+PV_CORR+SHARPE | 134.8% | **23.0%** | 18.0% | 0.97 | All-around champion |
| 2 | AMIHUD+CALMAR+CORR_MKT+MAX_DD+PV_CORR+SHARPE+VOL_RATIO | **156.5%** | 8.2% | 22.9% | 0.88 | Highest BT return |
| 3 | ADX+SHARPE+SLOPE+VORTEX | 118.5% | 4.3% | 18.5% | **0.98** | Simplest (4 factors) |
| 4 | ADX+PP120+SLOPE+VOL_RATIO | 73.1% | 10.4% | **10.4%** | 0.86 | Lowest drawdown |
| 5 | BREAKOUT+CALMAR+PV_CORR+SHARPE+SLOPE+VOL_RATIO+VORTEX | 97.2% | 10.6% | 14.5% | 0.91 | Best risk-adjusted |

## Screening Funnel

```
26,316 combos (16C2..16C7)
  -> VEC backtest (all 26,316)
  -> BT audit (Top 200)
  -> Rolling OOS gate (89 passed, >=60% quarterly win)
  -> Holdout gate (55 passed, return > 0)
  -> Production selection (5 strategies)
```

## Quick Reproduce

```bash
cd /home/sensen/dev/projects/-0927
# Full pipeline (~5min, BT ~30min)
uv run python scripts/run_full_pipeline.py --top-n 200 --n-jobs 16
# Or just BT the sealed candidates
uv run python scripts/batch_bt_backtest.py \
  --combos sealed_strategies/v4.2_20260205/artifacts/production_candidates.parquet \
  --topk 5
```

## Directory Structure

```
v4.2_20260205/
├── README.md                    # This file
├── MANIFEST.json                # Version metadata
├── CHECKSUMS.sha256             # File integrity
├── artifacts/
│   ├── production_candidates.parquet   # Top 10 strategies
│   ├── production_candidates.csv       # Same, CSV format
│   ├── all_gold_standard.parquet       # All 55 validated strategies
│   ├── bt_results.parquet              # BT ground truth (200 combos)
│   ├── triple_validation_candidates.parquet
│   └── FINAL_TRIPLE_VALIDATION_REPORT.md
└── locked/
    ├── configs/combo_wfo_config.yaml
    ├── scripts/                 # Pipeline scripts
    ├── src/etf_strategy/        # Full source snapshot
    └── pyproject.toml
```
