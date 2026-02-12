#!/usr/bin/env python3
"""
P3: Smart Money Post-Ranking Filter — Binary Veto via Score Penalty

核心思路: 非OHLCV因子作为4F排名信号被Exp4稀释(25%权重→Δrank<0.10)
P3突破: 将连续信号转为BINARY GATE(0/-1000), 直接注入factors_3d绕过标准化,
使得veto触发时rank变化远超Exp4阈值, 实现"smart money一票否决"

测试矩阵:
- 信号源: SHARE_CHG_10D(最强IC), SHARE_ACCEL(加速度), MARGIN_BUY_RATIO(杠杆)
- 阈值: top 10%/20%/30% (IC<0信号→高值=差→veto), bottom同理(IC>0)
- 对照: C2 baseline (AMIHUD + CALMAR + CORR_MKT)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import yaml
import pandas as pd
import numpy as np
from datetime import datetime

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.factor_cache import FactorCache
from etf_strategy.core.frozen_params import load_frozen_config, FrozenETFPool
from etf_strategy.core.cost_model import load_cost_model, build_cost_array
from etf_strategy.core.execution_model import load_execution_model
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import shift_timing_signal
from etf_strategy.regime_gate import compute_regime_gate_arr

# Import run_vec_backtest from batch_vec_backtest
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "batch_vec", str(ROOT / "scripts/batch_vec_backtest.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
run_vec_backtest = _mod.run_vec_backtest

PENALTY = -1000.0  # Score penalty for vetoed ETFs (dominates ±3 z-score range)

# ── Veto configurations to test ──
VETO_CONFIGS = [
    # (name, factor_file, direction, threshold)
    # direction: "top" = veto highest ranked (IC < 0 signals), "bottom" = veto lowest
    ("SHARE_CHG_10D_top10", "SHARE_CHG_10D", "top", 0.90),
    ("SHARE_CHG_10D_top20", "SHARE_CHG_10D", "top", 0.80),
    ("SHARE_CHG_10D_top30", "SHARE_CHG_10D", "top", 0.70),
    ("SHARE_ACCEL_bot10", "SHARE_ACCEL", "bottom", 0.10),
    ("SHARE_ACCEL_bot20", "SHARE_ACCEL", "bottom", 0.20),
    ("SHARE_ACCEL_bot30", "SHARE_ACCEL", "bottom", 0.30),
    ("MARGIN_BUY_RATIO_top20", "MARGIN_BUY_RATIO", "top", 0.80),
]


def load_non_ohlcv_factor(factor_name: str, dates, etf_codes, factors_dir: Path) -> pd.DataFrame:
    """Load a non-OHLCV factor from parquet, align to dates/etf_codes."""
    fpath = factors_dir / f"{factor_name}.parquet"
    if not fpath.exists():
        raise FileNotFoundError(f"Factor file not found: {fpath}")
    df = pd.read_parquet(fpath)
    df.index = pd.to_datetime(df.index)
    df = df.reindex(index=dates, columns=etf_codes)
    return df


def compute_veto_column(factor_df: pd.DataFrame, direction: str, threshold: float) -> np.ndarray:
    """Compute binary veto column (T, N) with 0.0 or PENALTY.

    Args:
        factor_df: (T, N) DataFrame with raw factor values
        direction: "top" = veto highest cross-sectional rank, "bottom" = veto lowest
        threshold: percentile threshold (e.g. 0.80 = top 20% vetoed)

    Returns:
        (T, N) array with 0.0 (ok) or PENALTY (vetoed)
    """
    T, N = factor_df.shape
    result = np.zeros((T, N), dtype=np.float64)

    for t in range(T):
        row = factor_df.iloc[t].values
        valid_mask = np.isfinite(row)
        n_valid = valid_mask.sum()
        if n_valid < 5:  # Not enough data
            continue

        # Cross-sectional rank (0 to 1)
        ranks = np.full(N, np.nan)
        valid_vals = row[valid_mask]
        sorted_order = np.argsort(valid_vals)
        rank_pct = np.zeros(len(valid_vals))
        for i, idx in enumerate(sorted_order):
            rank_pct[idx] = i / (len(valid_vals) - 1) if len(valid_vals) > 1 else 0.5
        ranks[valid_mask] = rank_pct

        # Apply veto
        for n in range(N):
            if np.isnan(ranks[n]):
                continue
            if direction == "top" and ranks[n] >= threshold:
                result[t, n] = PENALTY
            elif direction == "bottom" and ranks[n] <= threshold:
                result[t, n] = PENALTY

    return result


def compute_holdout_metrics(equity_curve: np.ndarray, dates, holdout_start: str):
    """Compute holdout-period metrics from equity curve."""
    dates_list = [str(d.date()) if hasattr(d, 'date') else str(d) for d in dates]
    try:
        ho_idx = next(i for i, d in enumerate(dates_list) if d >= holdout_start)
    except StopIteration:
        return {}

    ho_eq = equity_curve[ho_idx:]
    if len(ho_eq) < 10:
        return {}

    ho_ret = ho_eq[-1] / ho_eq[0] - 1.0
    running_max = np.maximum.accumulate(ho_eq)
    drawdowns = (ho_eq - running_max) / running_max
    ho_mdd = -drawdowns.min()

    # Sharpe (annualized, daily)
    daily_rets = np.diff(ho_eq) / ho_eq[:-1]
    if len(daily_rets) > 1 and np.std(daily_rets) > 1e-10:
        ho_sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)
    else:
        ho_sharpe = 0.0

    # Worst month (approximate: 21 trading days)
    worst_month = 0.0
    for i in range(0, len(ho_eq) - 21, 5):
        m_ret = ho_eq[i + 21] / ho_eq[i] - 1.0
        if m_ret < worst_month:
            worst_month = m_ret

    return {
        "ho_return": ho_ret,
        "ho_mdd": ho_mdd,
        "ho_sharpe": ho_sharpe,
        "ho_worst_month": worst_month,
    }


def main():
    print("=" * 80)
    print("P3: Smart Money Post-Ranking Filter (Binary Veto)")
    print("=" * 80)

    # ── 1. Load config ──
    import os
    config_path = Path(
        os.environ.get("WFO_CONFIG_PATH", str(ROOT / "configs/combo_wfo_config.yaml"))
    )
    with open(config_path) as f:
        config = yaml.safe_load(f)

    frozen = load_frozen_config(config, config_path=str(config_path))
    print(f"🔒 参数冻结校验通过 (version={frozen.version})")

    backtest_config = config.get("backtest", {})
    FREQ = backtest_config["freq"]
    POS_SIZE = backtest_config["pos_size"]
    LOOKBACK = backtest_config.get("lookback") or backtest_config.get("lookback_window")
    INITIAL_CAPITAL = float(backtest_config["initial_capital"])
    COMMISSION_RATE = float(backtest_config["commission_rate"])
    print(f"  FREQ={FREQ}, POS_SIZE={POS_SIZE}, LOOKBACK={LOOKBACK}")

    # Execution model
    exec_model = load_execution_model(config)
    USE_T1_OPEN = exec_model.is_t1_open
    print(f"  EXECUTION: {exec_model.mode}")

    # Cost model
    cost_model = load_cost_model(config)
    qdii_set = set(FrozenETFPool().qdii_codes)

    # Hysteresis
    hyst_config = backtest_config.get("hysteresis", {})
    DELTA_RANK = float(hyst_config.get("delta_rank", 0.0))
    MIN_HOLD_DAYS = int(hyst_config.get("min_hold_days", 0))
    print(f"  HYSTERESIS: delta_rank={DELTA_RANK}, min_hold_days={MIN_HOLD_DAYS}")

    # ── 2. Load data & compute factors ──
    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )

    factor_cache = FactorCache(cache_dir=Path(config["data"].get("cache_dir") or ".cache"))
    cached = factor_cache.get_or_compute(ohlcv=ohlcv, config=config, data_dir=loader.data_dir)

    factor_names = list(cached["factor_names"])
    dates = cached["dates"]
    etf_codes = cached["etf_codes"]
    factors_3d_base = cached["factors_3d"].copy()  # (T, N, K) - already standardized
    T, N, K = factors_3d_base.shape
    print(f"  Data: T={T}, N={N}, K={K} factors")

    # ── 3. Load external non-OHLCV factors (same path as batch_vec) ──
    _ext_factors_dir = config.get("combo_wfo", {}).get("extra_factors", {}).get("factors_dir", "")
    if _ext_factors_dir:
        _ext_dir = Path(_ext_factors_dir)
        if not _ext_dir.is_absolute():
            _ext_dir = ROOT / _ext_dir
    else:
        _ext_dir = ROOT / "results" / "non_ohlcv_factors"

    # Load C2 factor names and ensure they're in factors_3d
    C2_FACTORS = ["AMIHUD_ILLIQUIDITY", "CALMAR_RATIO_60D", "CORRELATION_TO_MARKET_20D"]
    missing_c2 = [f for f in C2_FACTORS if f not in factor_names]
    if missing_c2:
        raise ValueError(f"C2 factors missing from factor cache: {missing_c2}")

    # Build factor_index_map
    factor_index_map = {name: idx for idx, name in enumerate(factor_names)}
    C2_INDICES = [factor_index_map[f] for f in C2_FACTORS]
    print(f"  C2 factor indices: {dict(zip(C2_FACTORS, C2_INDICES))}")

    # ── 4. Cost array ──
    COST_ARR = build_cost_array(cost_model, list(etf_codes), qdii_set)
    tier = cost_model.active_tier
    print(f"  COST: A股={tier.a_share*10000:.0f}bp, QDII={tier.qdii*10000:.0f}bp")

    # ── 5. Timing & regime gate ──
    close_prices = ohlcv["close"][etf_codes].ffill().fillna(1.0).values
    open_prices = ohlcv["open"][etf_codes].ffill().fillna(1.0).values
    high_prices = ohlcv["high"][etf_codes].ffill().fillna(1.0).values
    low_prices = ohlcv["low"][etf_codes].ffill().fillna(1.0).values

    timing_config = config.get("backtest", {}).get("timing", {})
    timing_type = timing_config.get("type", "light_timing")
    extreme_threshold = timing_config.get("extreme_threshold", -0.4)
    extreme_position = timing_config.get("extreme_position", 0.3)

    if timing_type == "light_timing":
        timing_module = LightTimingModule(
            extreme_threshold=extreme_threshold,
            extreme_position=extreme_position,
        )
        timing_arr_raw = (
            timing_module.compute_position_ratios(ohlcv["close"])
            .reindex(dates)
            .fillna(1.0)
            .values
        )
    else:
        timing_arr_raw = np.ones(T, dtype=np.float64)

    timing_arr = shift_timing_signal(timing_arr_raw)
    gate_arr = compute_regime_gate_arr(
        ohlcv["close"], dates, backtest_config=backtest_config
    )
    timing_arr = timing_arr * gate_arr
    print(f"  Timing: type={timing_type}, gate mean={gate_arr.mean():.3f}")

    # Risk control params (simplified - match batch_vec defaults)
    risk_config = backtest_config.get("risk_control", {})
    trailing_stop_pct = risk_config.get("trailing_stop_pct", 0.08)
    leverage_cap = risk_config.get("leverage_cap", 1.0)
    stop_on_rebalance_only = risk_config.get("stop_check_on_rebalance_only", False)

    # ── 6. Non-OHLCV factors directory ──
    factors_dir = ROOT / "results" / "non_ohlcv_factors"
    print(f"\n📦 Non-OHLCV factors dir: {factors_dir}")

    # Holdout config
    holdout_start = config["data"].get("holdout_start", "2025-05-01")
    print(f"  Holdout start: {holdout_start}")

    # ── 7. Build veto variants and run VEC ──
    results = []

    # 7a. C2 baseline (no veto)
    print(f"\n🔄 Running C2 baseline...")
    eq_curve_c2, ret, wr, pf, trades, rounding, risk = run_vec_backtest(
        factors_3d_base, close_prices, open_prices, high_prices, low_prices,
        timing_arr, C2_INDICES,
        freq=FREQ, pos_size=POS_SIZE, initial_capital=INITIAL_CAPITAL,
        commission_rate=COMMISSION_RATE, lookback=LOOKBACK,
        cost_arr=COST_ARR,
        trailing_stop_pct=trailing_stop_pct,
        stop_on_rebalance_only=stop_on_rebalance_only,
        leverage_cap=leverage_cap,
        use_t1_open=USE_T1_OPEN,
        delta_rank=DELTA_RANK,
        min_hold_days=MIN_HOLD_DAYS,
    )
    ho_metrics = compute_holdout_metrics(eq_curve_c2, dates, holdout_start)
    results.append({
        "config": "C2_baseline",
        "full_return": ret,
        "trades": trades,
        **ho_metrics,
    })
    print(f"  C2: full={ret*100:.1f}%, HO={ho_metrics.get('ho_return', 0)*100:.1f}%, "
          f"MDD={ho_metrics.get('ho_mdd', 0)*100:.1f}%, Sharpe={ho_metrics.get('ho_sharpe', 0):.2f}")

    # 7b. Veto variants
    for veto_name, factor_file, direction, threshold in VETO_CONFIGS:
        print(f"\n🔄 Running {veto_name}...")
        try:
            # Load the non-OHLCV factor
            factor_df = load_non_ohlcv_factor(factor_file, dates, etf_codes, factors_dir)
            valid_pct = factor_df.notna().sum().sum() / factor_df.size * 100
            print(f"  Factor {factor_file}: {valid_pct:.1f}% valid")

            # Compute veto column
            veto_col = compute_veto_column(factor_df, direction, threshold)
            n_vetoed = (veto_col == PENALTY).sum()
            veto_rate = n_vetoed / (T * N) * 100
            print(f"  Veto rate: {veto_rate:.1f}% ({n_vetoed}/{T*N})")

            # Append veto column to factors_3d (BYPASSES standardization)
            factors_3d_ext = np.concatenate(
                [factors_3d_base, veto_col[:, :, np.newaxis]], axis=2
            )
            veto_idx = K  # New column index

            # C2 + veto combo
            combo_indices = C2_INDICES + [veto_idx]

            eq_curve, ret, wr, pf, trades, rounding, risk = run_vec_backtest(
                factors_3d_ext, close_prices, open_prices, high_prices, low_prices,
                timing_arr, combo_indices,
                freq=FREQ, pos_size=POS_SIZE, initial_capital=INITIAL_CAPITAL,
                commission_rate=COMMISSION_RATE, lookback=LOOKBACK,
                cost_arr=COST_ARR,
                trailing_stop_pct=trailing_stop_pct,
                stop_on_rebalance_only=stop_on_rebalance_only,
                leverage_cap=leverage_cap,
                use_t1_open=USE_T1_OPEN,
                delta_rank=DELTA_RANK,
                min_hold_days=MIN_HOLD_DAYS,
            )
            ho_metrics = compute_holdout_metrics(eq_curve, dates, holdout_start)
            results.append({
                "config": f"C2+{veto_name}",
                "full_return": ret,
                "trades": trades,
                **ho_metrics,
            })
            print(f"  Result: full={ret*100:.1f}%, HO={ho_metrics.get('ho_return', 0)*100:.1f}%, "
                  f"MDD={ho_metrics.get('ho_mdd', 0)*100:.1f}%, Sharpe={ho_metrics.get('ho_sharpe', 0):.2f}")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "config": f"C2+{veto_name}",
                "full_return": np.nan,
                "trades": 0,
            })

    # ── 8. Results summary ──
    print("\n" + "=" * 100)
    print("P3 Smart Money Filter — Results Summary")
    print("=" * 100)
    print(f"{'Config':<35} {'Full%':>8} {'HO Ret%':>8} {'HO MDD%':>8} {'HO Sharpe':>10} {'HO WM%':>8} {'Trades':>7} {'Δ vs C2':>8}")
    print("-" * 100)

    c2_ho_ret = results[0].get("ho_return", 0)
    for r in results:
        ho_ret = r.get("ho_return", 0)
        ho_mdd = r.get("ho_mdd", 0)
        ho_sharpe = r.get("ho_sharpe", 0)
        ho_wm = r.get("ho_worst_month", 0)
        delta = (ho_ret - c2_ho_ret) * 100 if not np.isnan(ho_ret) else 0
        delta_str = f"{delta:+.1f}pp" if r["config"] != "C2_baseline" else "—"
        print(f"{r['config']:<35} {r.get('full_return', 0)*100:>7.1f}% "
              f"{ho_ret*100:>7.1f}% {ho_mdd*100:>7.1f}% {ho_sharpe:>9.2f} "
              f"{ho_wm*100:>7.1f}% {r.get('trades', 0):>6d}  {delta_str:>8}")

    # ── 9. Save results ──
    out_dir = ROOT / "results" / f"p3_smart_money_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_results = pd.DataFrame(results)
    df_results.to_csv(out_dir / "p3_results.csv", index=False)
    print(f"\n💾 Results saved to {out_dir}")

    # ── 10. Analyze veto effectiveness ──
    print("\n" + "=" * 80)
    print("Veto Effectiveness Analysis")
    print("=" * 80)
    for r in results[1:]:  # Skip baseline
        ho_ret = r.get("ho_return", 0)
        delta = (ho_ret - c2_ho_ret) * 100
        trades_delta = r.get("trades", 0) - results[0].get("trades", 0)
        verdict = "✅ IMPROVED" if delta > 1.0 else ("⚠️ MARGINAL" if delta > 0 else "❌ WORSE")
        print(f"  {r['config']:<35}: Δret={delta:+.1f}pp, Δtrades={trades_delta:+d}  {verdict}")


if __name__ == "__main__":
    main()
