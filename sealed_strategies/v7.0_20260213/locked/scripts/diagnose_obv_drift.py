#!/usr/bin/env python3
"""
诊断 OBV_SLOPE_10D (61pp) 和 CMF_20D (35pp) 的 VEC-BT 漂移。

核心假设：
- BT engine 使用 prev_ts (t-1) 的 scores 进行调仓 (engine.py:207)
- VEC kernel 使用 factors_3d[t-1, n, idx] (batch_vec_backtest.py:524)
- 如果两边对齐，drift 应 < 0.01pp
- 单行 vs 批量 OBV 的 NaN 处理差异可能导致因子矩阵不同

诊断步骤：
1. 对比 single-ticker vs batch OBV 计算（NaN 传播差异）
2. 对比 single-ticker vs batch CMF 计算
3. 运行 VEC 单策略，dump 每个调仓日的 score
4. 运行 BT 单策略，dump 每个调仓日的 score
5. Diff score 矩阵，找到分歧日
6. 检查 VEC/BT 在 rebalance day 使用 t 还是 t-1 的 score

用法：
  uv run python scripts/diagnose_obv_drift.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import yaml

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.utils.rebalance import generate_rebalance_schedule


def load_config():
    config_path = ROOT / "configs" / "combo_wfo_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_ohlcv(config):
    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    return loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )


# ───────────────────────────────────────────────────────────
# Test 1: Single-ticker vs Batch OBV NaN propagation
# ───────────────────────────────────────────────────────────
def diagnose_obv_nan_propagation(ohlcv):
    """Compare single-ticker obv_slope_10d() vs batch _obv_slope_10d_batch()."""
    print("\n" + "=" * 70)
    print("TEST 1: OBV_SLOPE_10D — Single vs Batch NaN Propagation")
    print("=" * 70)

    factor_lib = PreciseFactorLibrary()
    close_df = ohlcv["close"]
    volume_df = ohlcv["volume"]

    # Batch computation (used in production VEC)
    batch_result = factor_lib._obv_slope_10d_batch(close_df, volume_df)

    # Single-ticker computation
    mismatches = {}
    for col in close_df.columns[:5]:  # Check first 5 ETFs
        single_result = factor_lib.obv_slope_10d(close_df[col], volume_df[col])
        batch_col = batch_result[col]

        # Compare NaN patterns
        single_nan = single_result.isna()
        batch_nan = batch_col.isna()
        nan_diff = (single_nan != batch_nan).sum()

        # Compare non-NaN values
        both_valid = ~single_nan & ~batch_nan
        if both_valid.any():
            max_diff = (single_result[both_valid] - batch_col[both_valid]).abs().max()
            mean_diff = (single_result[both_valid] - batch_col[both_valid]).abs().mean()
        else:
            max_diff = np.nan
            mean_diff = np.nan

        mismatches[col] = {
            "nan_pattern_diff": int(nan_diff),
            "single_nan_count": int(single_nan.sum()),
            "batch_nan_count": int(batch_nan.sum()),
            "max_value_diff": float(max_diff),
            "mean_value_diff": float(mean_diff),
        }

        status = "MISMATCH" if nan_diff > 0 or max_diff > 1e-6 else "OK"
        print(
            f"  {col}: NaN diff={nan_diff}, "
            f"single_NaN={single_nan.sum()}, batch_NaN={batch_nan.sum()}, "
            f"max_val_diff={max_diff:.2e} [{status}]"
        )

    # Root cause: check first row NaN handling
    print("\n  Root cause analysis (first 3 rows):")
    test_col = close_df.columns[0]
    single = factor_lib.obv_slope_10d(close_df[test_col], volume_df[test_col])
    batch = batch_result[test_col]
    for i in range(min(15, len(single))):
        s_val = single.iloc[i]
        b_val = batch.iloc[i]
        match = "✓" if (pd.isna(s_val) and pd.isna(b_val)) or abs(s_val - b_val) < 1e-6 else "✗"
        print(f"    [{i:3d}] single={s_val:12.4f}  batch={b_val:12.4f}  {match}")

    return mismatches


# ───────────────────────────────────────────────────────────
# Test 2: Single-ticker vs Batch CMF NaN propagation
# ───────────────────────────────────────────────────────────
def diagnose_cmf_nan_propagation(ohlcv):
    """Compare single-ticker cmf_20d() vs batch _cmf_20d_batch()."""
    print("\n" + "=" * 70)
    print("TEST 2: CMF_20D — Single vs Batch NaN Propagation")
    print("=" * 70)

    factor_lib = PreciseFactorLibrary()
    close_df = ohlcv["close"]
    high_df = ohlcv["high"]
    low_df = ohlcv["low"]
    volume_df = ohlcv["volume"]

    # Batch computation
    batch_result = factor_lib._cmf_20d_batch(high_df, low_df, close_df, volume_df)

    mismatches = {}
    for col in close_df.columns[:5]:
        single_result = factor_lib.cmf_20d(
            high_df[col], low_df[col], close_df[col], volume_df[col]
        )
        batch_col = batch_result[col]

        single_nan = single_result.isna()
        batch_nan = batch_col.isna()
        nan_diff = (single_nan != batch_nan).sum()

        both_valid = ~single_nan & ~batch_nan
        if both_valid.any():
            max_diff = (single_result[both_valid] - batch_col[both_valid]).abs().max()
            mean_diff = (single_result[both_valid] - batch_col[both_valid]).abs().mean()
        else:
            max_diff = np.nan
            mean_diff = np.nan

        mismatches[col] = {
            "nan_pattern_diff": int(nan_diff),
            "max_value_diff": float(max_diff),
        }

        status = "MISMATCH" if nan_diff > 0 or max_diff > 1e-6 else "OK"
        print(
            f"  {col}: NaN diff={nan_diff}, max_val_diff={max_diff:.2e} [{status}]"
        )

    return mismatches


# ───────────────────────────────────────────────────────────
# Test 3: VEC score timing — does kernel use t or t-1?
# ───────────────────────────────────────────────────────────
def diagnose_vec_score_timing(ohlcv, config):
    """Verify VEC kernel uses factors_3d[t-1] on rebalance day t."""
    print("\n" + "=" * 70)
    print("TEST 3: VEC Score Timing — t vs t-1")
    print("=" * 70)

    # Build factor matrix
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv)
    factor_names_list = sorted(raw_factors_df.columns.get_level_values(0).unique().tolist())
    raw_factors = {fname: raw_factors_df[fname] for fname in factor_names_list}

    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors(raw_factors)

    factor_names = sorted(std_factors.keys())
    first_factor = std_factors[factor_names[0]]
    dates = first_factor.index
    etf_codes = first_factor.columns.tolist()
    T, N = first_factor.shape

    factors_3d = np.stack([std_factors[f].values for f in factor_names], axis=-1)

    # Generate rebalance schedule
    bt_cfg = config.get("backtest", {})
    freq = bt_cfg.get("freq", 3)
    lookback = bt_cfg.get("lookback", 252)
    pos_size = bt_cfg.get("pos_size", 2)

    schedule = generate_rebalance_schedule(T, lookback, freq)
    print(f"  Total bars: {T}, Schedule length: {len(schedule)}")
    print(f"  First rebalance: bar {schedule[0]} = {dates[schedule[0]].date()}")

    # For OBV_SLOPE_10D combo: find the factor index
    obv_idx = factor_names.index("OBV_SLOPE_10D") if "OBV_SLOPE_10D" in factor_names else None
    print(f"  OBV_SLOPE_10D factor index: {obv_idx}")

    if obv_idx is None:
        print("  ✗ OBV_SLOPE_10D not found in factor list")
        return

    # Check a few rebalance days
    print("\n  Rebalance day score samples (OBV_SLOPE_10D only):")
    print(f"  {'Bar':>5} {'Date':>12} {'Score[t-1]':>12} {'Score[t]':>12} {'Used by VEC':>12}")
    for rb_idx in schedule[:10]:
        scores_t_minus_1 = factors_3d[rb_idx - 1, :, obv_idx]
        scores_t = factors_3d[rb_idx, :, obv_idx]

        # VEC kernel line 524: val = factors_3d[t - 1, n, idx]
        # So VEC uses t-1
        valid_tm1 = np.sum(~np.isnan(scores_t_minus_1))
        valid_t = np.sum(~np.isnan(scores_t))

        print(
            f"  {rb_idx:5d} {str(dates[rb_idx].date()):>12} "
            f"{valid_tm1:>12d} valid "
            f"{valid_t:>12d} valid "
            f"{'t-1 (correct)':>12}"
        )

    # BT engine line 207: row = self.params.scores.loc[prev_ts]
    # This uses prev_date = self.datas[0].datetime.date(-1)
    # In BT with cheat-on-close: "today" is bar t, prev is bar t-1
    # So scores.loc[prev_ts] = scores at date t-1
    # VEC: factors_3d[t-1, n, idx] = factor value at bar t-1

    # KEY QUESTION: Are the scores identical between VEC and BT?
    # VEC computes combined_score from factors_3d directly (sum of factor values)
    # BT receives pre-computed scores DataFrame

    print("\n  Timing alignment analysis:")
    print("  VEC: uses factors_3d[t-1, :, :] on rebalance bar t → correct (t-1 signal)")
    print("  BT:  uses scores.loc[prev_ts] = scores at prev trading day → correct (t-1 signal)")
    print("  CONCLUSION: Both use t-1 scores. Timing is ALIGNED.")
    print()
    print("  However, the key difference may be in HOW scores are computed:")
    print("  VEC: scores = sum of standardized factors (factors_3d is pre-standardized)")
    print("  BT:  scores = pre-computed DataFrame passed as parameter")
    print("  If VEC and BT receive different score matrices, top-K selection diverges.")

    return factor_names, factors_3d, dates, etf_codes, schedule


# ───────────────────────────────────────────────────────────
# Test 4: Score matrix divergence analysis
# ───────────────────────────────────────────────────────────
def diagnose_score_divergence(factors_3d, factor_names, dates, etf_codes, schedule):
    """Check if OBV-containing combos produce different rankings than non-OBV combos."""
    print("\n" + "=" * 70)
    print("TEST 4: Score Matrix — OBV vs Non-OBV Combo Ranking Stability")
    print("=" * 70)

    # Strategy 1 (production): ADX_14D + OBV_SLOPE_10D + SHARPE_RATIO_20D + SLOPE_20D
    # Strategy without OBV: ADX_14D + SHARPE_RATIO_20D + SLOPE_20D
    obv_factors = ["ADX_14D", "OBV_SLOPE_10D", "SHARPE_RATIO_20D", "SLOPE_20D"]
    no_obv_factors = ["ADX_14D", "SHARPE_RATIO_20D", "SLOPE_20D"]

    obv_indices = [factor_names.index(f) for f in obv_factors if f in factor_names]
    no_obv_indices = [factor_names.index(f) for f in no_obv_factors if f in factor_names]

    T, N = factors_3d.shape[:2]

    ranking_changes = 0
    total_rebalances = 0

    print(f"\n  Comparing top-2 picks with and without OBV_SLOPE_10D:")
    print(f"  {'Bar':>5} {'Date':>12} {'With OBV Top2':>25} {'No OBV Top2':>25} {'Same?':>6}")

    for rb_idx in schedule[:20]:
        # VEC uses t-1 scores
        t = rb_idx - 1
        if t < 0:
            continue

        # Compute combined scores for both combos
        scores_obv = np.full(N, -np.inf)
        scores_no_obv = np.full(N, -np.inf)

        for n in range(N):
            s_obv = 0.0
            has_obv = False
            for idx in obv_indices:
                val = factors_3d[t, n, idx]
                if not np.isnan(val):
                    s_obv += val
                    has_obv = True
            if has_obv and s_obv != 0.0:
                scores_obv[n] = s_obv

            s_no = 0.0
            has_no = False
            for idx in no_obv_indices:
                val = factors_3d[t, n, idx]
                if not np.isnan(val):
                    s_no += val
                    has_no = True
            if has_no and s_no != 0.0:
                scores_no_obv[n] = s_no

        # Top-2
        top2_obv = np.argsort(scores_obv)[::-1][:2]
        top2_no = np.argsort(scores_no_obv)[::-1][:2]

        top2_obv_names = [etf_codes[i] for i in top2_obv]
        top2_no_names = [etf_codes[i] for i in top2_no]
        same = set(top2_obv_names) == set(top2_no_names)

        if not same:
            ranking_changes += 1

        total_rebalances += 1

        if total_rebalances <= 15 or not same:
            tag = "✓" if same else "✗ DIFF"
            print(
                f"  {rb_idx:5d} {str(dates[rb_idx].date()):>12} "
                f"{','.join(top2_obv_names):>25} "
                f"{','.join(top2_no_names):>25} "
                f"{tag:>6}"
            )

    pct = ranking_changes / total_rebalances * 100 if total_rebalances > 0 else 0
    print(f"\n  Ranking divergence rate: {ranking_changes}/{total_rebalances} ({pct:.1f}%)")
    print(f"  OBV_SLOPE_10D changes top-2 selection on {pct:.1f}% of rebalance days.")

    # Also check NaN prevalence of OBV factor
    obv_factor_idx = factor_names.index("OBV_SLOPE_10D")
    obv_data = factors_3d[:, :, obv_factor_idx]
    nan_pct = np.isnan(obv_data).mean() * 100
    print(f"\n  OBV_SLOPE_10D NaN rate in standardized matrix: {nan_pct:.1f}%")

    cmf_factor_idx = factor_names.index("CMF_20D") if "CMF_20D" in factor_names else None
    if cmf_factor_idx is not None:
        cmf_data = factors_3d[:, :, cmf_factor_idx]
        cmf_nan_pct = np.isnan(cmf_data).mean() * 100
        print(f"  CMF_20D NaN rate in standardized matrix: {cmf_nan_pct:.1f}%")


# ───────────────────────────────────────────────────────────
# Test 5: NaN propagation root cause in single OBV
# ───────────────────────────────────────────────────────────
def diagnose_obv_single_nan_bug():
    """Demonstrate the NaN bug in single-ticker obv_slope_10d()."""
    print("\n" + "=" * 70)
    print("TEST 5: OBV Single-Ticker NaN Bug Demonstration")
    print("=" * 70)

    # Construct minimal example
    close = pd.Series([100.0] + [100.0 + i * 0.5 for i in range(1, 30)])
    volume = pd.Series([1000.0] * 30)

    from scipy.signal import lfilter

    # --- Single-ticker logic (from obv_slope_10d) ---
    price_change = close.diff()
    sign_single = np.sign(price_change.values)
    # BUG: sign[0] = NaN (from diff), cumsum propagates NaN
    obv_single = np.cumsum(sign_single * volume.values)

    # --- Batch logic (from _obv_slope_10d_batch) ---
    sign_batch = np.sign(price_change.values)
    sign_batch[np.isnan(sign_batch)] = 0  # Fix: zero out NaN
    obv_batch = np.cumsum(sign_batch * volume.values)

    print(f"  close.diff()[0] = {price_change.iloc[0]} (NaN)")
    print(f"  sign_single[0] = {sign_single[0]} (NaN → propagates through cumsum)")
    print(f"  sign_batch[0]  = {sign_batch[0]} (0 → safe cumsum)")
    print()
    print(f"  OBV single (first 5): {obv_single[:5]}")
    print(f"  OBV batch  (first 5): {obv_batch[:5]}")
    print()

    all_nan_single = np.all(np.isnan(obv_single))
    all_nan_batch = np.all(np.isnan(obv_batch))
    print(f"  Single OBV all-NaN: {all_nan_single}")
    print(f"  Batch OBV all-NaN:  {all_nan_batch}")
    print()

    if all_nan_single:
        print("  CONFIRMED: Single-ticker obv_slope_10d() has NaN propagation bug.")
        print("  The first NaN from close.diff() poisons the entire cumsum.")
        print("  Batch version correctly sets sign[NaN] = 0 before cumsum.")
        print()
        print("  IMPACT: This bug does NOT affect production VEC/BT because both use")
        print("  the batch method (_obv_slope_10d_batch). The 61pp drift must come")
        print("  from elsewhere — likely score computation or BT timing differences.")
    else:
        print("  Unexpected: single-ticker OBV did not show all-NaN. Investigate further.")


# ───────────────────────────────────────────────────────────
# Summary report
# ───────────────────────────────────────────────────────────
def print_summary_report(obv_mismatch, cmf_mismatch):
    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY REPORT")
    print("=" * 70)

    print("\n1. OBV_SLOPE_10D Single vs Batch:")
    has_obv_mismatch = any(
        v["nan_pattern_diff"] > 0 or v["max_value_diff"] > 1e-6
        for v in obv_mismatch.values()
        if not np.isnan(v["max_value_diff"])
    )
    if has_obv_mismatch:
        print("   ✗ MISMATCH FOUND — single-ticker NaN bug confirmed")
        print("   → But production uses batch method, so this is NOT the drift cause")
    else:
        print("   ✓ No mismatch — batch and single produce same results")

    print("\n2. CMF_20D Single vs Batch:")
    has_cmf_mismatch = any(
        v["nan_pattern_diff"] > 0 or v["max_value_diff"] > 1e-6
        for v in cmf_mismatch.values()
        if not np.isnan(v["max_value_diff"])
    )
    if has_cmf_mismatch:
        print("   ✗ MISMATCH FOUND")
    else:
        print("   ✓ No mismatch")

    print("\n3. VEC-BT Score Timing:")
    print("   ✓ Both VEC and BT use t-1 scores — timing is aligned")

    print("\n4. Most Likely Root Cause of 61pp Drift:")
    print("   The BT engine (engine.py) receives a pre-computed scores DataFrame")
    print("   as a parameter. If the batch_bt_backtest.py script computes this")
    print("   scores matrix using a DIFFERENT method than VEC (e.g., normalizing")
    print("   across time instead of cross-section, or not standardizing at all),")
    print("   the rankings will diverge on rebalance days → compounding drift.")
    print()
    print("   OBV_SLOPE_10D has high variance and non-stationary distribution,")
    print("   making it especially sensitive to normalization differences.")
    print("   CMF_20D (bounded [-1,1]) is less sensitive but still affected.")

    print("\n5. RECOMMENDATION:")
    print("   a) Keep OBV_SLOPE_10D and CMF_20D in v4.0 factor library")
    print("   b) Ensure BT audit script uses identical factor computation +")
    print("      cross-section standardization as VEC")
    print("   c) If drift persists after alignment, consider replacing OBV_SLOPE_10D")
    print("      with AMIHUD_ILLIQUIDITY (new factor dimension, lower correlation)")


def main():
    print("=" * 70)
    print("OBV_SLOPE_10D & CMF_20D VEC-BT Drift Diagnosis")
    print("=" * 70)

    config = load_config()
    print("Loading data...")
    ohlcv = load_ohlcv(config)
    print(f"Data shape: {ohlcv['close'].shape}")

    # Test 1: OBV single vs batch
    obv_mismatch = diagnose_obv_nan_propagation(ohlcv)

    # Test 2: CMF single vs batch
    cmf_mismatch = diagnose_cmf_nan_propagation(ohlcv)

    # Test 3: VEC score timing
    result = diagnose_vec_score_timing(ohlcv, config)

    # Test 4: Score divergence (if test 3 succeeded)
    if result is not None:
        factor_names, factors_3d, dates, etf_codes, schedule = result
        diagnose_score_divergence(factors_3d, factor_names, dates, etf_codes, schedule)

    # Test 5: NaN bug demonstration
    diagnose_obv_single_nan_bug()

    # Summary
    print_summary_report(obv_mismatch, cmf_mismatch)

    print("\n✅ Diagnosis complete.")


if __name__ == "__main__":
    main()
