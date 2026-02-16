# Conditional Factor Approaches for QDII Integration — Negative Results Archive

> **Date**: 2026-02-11
> **Status**: NEGATIVE — All hypotheses refuted, S1 baseline frozen
> **Scripts**: Archived to `archive/scripts/negative_results/` — `ablation_qdii_factors.py`, `ablation_qdii_deep_diag.py`, `ablation_adx_conditional_ic.py`, `ablation_diff_trade_attribution.py`

## 1. Problem Statement

QDII ETFs (513100 Nasdaq, 513500 S&P, 159920 HSI, 513050 China Internet, 513130 HK Tech) are structurally penalized in S1's unified cross-sectional Z-score due to different trading microstructure:
- **ADX_14D**: QDII gap-open frequently → narrow intraday range → low ADX → mean Z-score gap = **-0.30σ** vs A-share
- **OBV_SLOPE_10D**: QDII lower volume → OBV less responsive → gap = -0.02σ (minimal)

With `universe_mode = A_SHARE_ONLY` in production, QDII are monitored but not traded. This research tested whether conditional factor adjustments could unlock QDII alpha.

## 2. Production Chain (Baseline)

```
S1: ADX_14D + OBV_SLOPE_10D + SHARPE_RATIO_20D + SLOPE_20D
Scoring: unified 43-ETF cross-sectional Z-score → sum → topk(2)
Execution: FREQ=5, Exp4 hysteresis (dr=0.10, mh=9, max 1 swap)
Regime gate: ON (volatility, 510300, thresholds 25/30/40%)
Cost: med tier (A-share 20bp, QDII 50bp)
Holdout: 2025-05 ~ 2026-02-10
Baseline HO return: +42.7%, MDD -11.8%, worst month -2.7%
```

## 3. Experiments Conducted

### Exp A: Factor Ablation (4 variants)

| Variant | Factors | HO Return | HO Sharpe | HO MDD | QDII Rate |
|---------|---------|-----------|-----------|--------|-----------|
| Baseline (S1) | ADX+OBV+SHARPE+SLOPE | +42.7% | 2.28 | -11.8% | 8.7% |
| Drop ADX | OBV+SHARPE+SLOPE | +57.8% | 2.65 | -10.1% | 8.9% |
| Drop OBV | ADX+SHARPE+SLOPE | +28.1% | 1.52 | -14.3% | 9.1% |
| Drop Both | SHARPE+SLOPE | +29.0% | 1.44 | -13.7% | 8.5% |

**Initial read**: Drop ADX = +15.1pp improvement. But QDII selection barely changed (+0.2pp).

### Exp B: Deep Diagnostics on Drop ADX

Three diagnostics to evaluate robustness:

**(B1) Pre/post hysteresis QDII selection**
- Pre-hysteresis QDII selection: 12.2%
- Post-hysteresis QDII selection: 18.3%
- **Finding**: Hysteresis HELPS QDII (locks in positions via min_hold_days)

**(B2) Per-trade forward returns**
- Baseline per-BUY 10D return: higher than Drop ADX
- Baseline differential ETFs (gold, thematic equity): better ex-post returns
- **Finding**: Baseline has BETTER per-trade quality

**(B3) Temporal robustness**
- Half-year win rate: 6/11 (55%) — barely better than coin flip
- 82% of +15pp advantage concentrated in 2025 H2 single period (+12.5pp)
- Excluding 2025 H2: Drop ADX slightly underperforms baseline
- **Finding**: Improvement is NOT temporally stable

### Exp C: Regime-Conditional IC (Phase 0)

Spearman rank IC decomposed by volatility regime (low/mid/high/extreme):

| Factor | low_vol IC | mid_vol IC | high_vol IC | high_vol N |
|--------|-----------|-----------|------------|-----------|
| ADX_14D | +0.016 | +0.020 | -0.020 | **9** |
| OBV_SLOPE_10D | +0.019 | -0.025 | +0.040 | 9 |
| SHARPE_RATIO_20D | +0.030 | **-0.109** | +0.068 | 9 |
| SLOPE_20D | +0.022 | **-0.079** | +0.053 | 9 |

**Finding**: SHARPE and SLOPE degrade MORE than ADX in mid_vol. Sample size = 9 rebal days in high_vol — statistically meaningless. Any threshold-based switching would overfit to 29 non-low_vol events.

### Exp D: Differential Trade Attribution

Per-rebalance comparison of Baseline vs Drop ADX holdings, with state variable correlation:

| State Variable | Spearman r | p-value | Actionable? |
|----------------|-----------|---------|-------------|
| vol_20D | -0.071 | 0.367 | NO |
| ER_20D | +0.167 | **0.033** | Weak, wrong direction |
| dispersion_5D | +0.095 | 0.224 | NO |
| rank_autocorr_5D | -0.043 | 0.581 | NO |

**2025 H2 deep dive**: Per-trade differential return ≈ -0.04% (essentially zero). The +12.5pp aggregate comes from path-dependent compounding, not per-trade quality. Baseline's differential picks (gold, thematic equity) actually outperformed Drop ADX's choices (bonds, low-vol).

## 4. Key Results (5 Hypotheses Refuted)

| # | Hypothesis | Evidence | Verdict |
|---|-----------|----------|---------|
| 1 | ADX structurally blocks QDII selection | QDII rate: 8.7% → 8.9% after dropping ADX | **REFUTED** |
| 2 | Drop ADX is a robust improvement | 55% half-year WR, 82% concentrated in one period | **REFUTED** |
| 3 | Volatility regime gates ADX utility | r = -0.071, p = 0.367 | **REFUTED** |
| 4 | Efficiency Ratio (trendiness) gates ADX | r = +0.167 but WRONG direction (helps in trends, not chop) | **REFUTED** |
| 5 | Drop ADX wins via better per-trade selection | Per-trade diff ≈ 0%, advantage is path-dependent compounding | **REFUTED** |

## 5. Conclusions & Decisions

### Frozen decisions
1. **S1 baseline (4F) remains production standard** — no conditional factor modifications
2. **No regime-conditional switching** — sample size insufficient, no valid gating variable
3. **Conditional factor approaches within current 4F family are exhausted** — further tuning = overfitting risk

### Why this matters
- Constrains future search space: do NOT revisit conditional weighting of ADX/OBV/SHARPE/SLOPE
- Path-dependent compounding effects cannot be predicted or replicated — aggregate period returns can be misleading
- Per-trade differential analysis is the correct evaluation methodology, not period-level comparison

### Root cause identified
The +15pp Drop ADX advantage stems from a single favorable compounding path in 2025 H2, not from systematically better ETF selection. This is a **variance artifact**, not an alpha source.

## 6. Next Steps (Pivoting to Orthogonal Information)

Current factor family (trend/volume/return-pattern/hysteresis) is saturated. New signals must come from **orthogonal information dimensions**:

| Experiment | Signal Type | Data Required | Status |
|-----------|-------------|--------------|--------|
| **Exp7** (P1) | QDII premium/discount microstructure | IOPV or NAV daily | Data blocked — Tushare available |
| **Exp6** (P2) | FX beta-exposure models | USD/CNH, HKD/CNY daily | Data blocked — Tushare available |

Both designed as **QDII-specific sleeve strategies**, not forced into 43-ETF unified ranking.

## 7. Scripts & Artifacts

| Script | Purpose | Output |
|--------|---------|--------|
| `archive/scripts/negative_results/ablation_qdii_factors.py` | 4-variant factor ablation (GLOBAL, F5+Exp4) | `results/ablation_qdii_*/` |
| `archive/scripts/negative_results/ablation_qdii_deep_diag.py` | Pre/post hysteresis, trade quality, regime robustness | `results/ablation_diag_*/` |
| `archive/scripts/negative_results/ablation_adx_conditional_ic.py` | Rank IC decomposed by vol regime | `results/ablation_conditional_ic_*/` |
| `archive/scripts/negative_results/ablation_diff_trade_attribution.py` | Differential trade ledger + state variable correlation | `results/ablation_diff_attr_*/` |
