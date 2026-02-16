#!/usr/bin/env python3
"""
P4: Aggregate Smart Money Timing — Portfolio-Level Exposure Scaling

核心思路: P3的per-ETF veto失败(破坏C2的近最优选择), 但fund_share/margin的
聚合信号可能作为PORTFOLIO-LEVEL timing提供额外信息:
- 当全市场资金大量涌入(SHARE_CHG_10D均值高) → IC说bearish → 降仓
- 当全市场资金流出(SHARE_CHG_10D均值低) → IC说bullish → 正常仓位

关键区别:
- regime gate用波动率(510300), P4用资金流(fund_share聚合)
- 两个信号源完全正交: 波动率≠资金流
- 不修改排名/选股, 只修改timing_arr

测试矩阵:
- 聚合信号: SHARE_CHG_10D/5D/20D均值, SHARE_ACCEL均值
- timing缩放: top 20/30%时 → 0.5/0.3/0.0 (降仓/空仓)
- 对照: C2 with standard timing only
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


def load_non_ohlcv_factor(factor_name: str, dates, etf_codes, factors_dir: Path) -> pd.DataFrame:
    """Load a non-OHLCV factor from parquet, align to dates/etf_codes."""
    fpath = factors_dir / f"{factor_name}.parquet"
    if not fpath.exists():
        raise FileNotFoundError(f"Factor file not found: {fpath}")
    df = pd.read_parquet(fpath)
    df.index = pd.to_datetime(df.index)
    df = df.reindex(index=dates, columns=etf_codes)
    return df


def compute_aggregate_signal(factor_df: pd.DataFrame, method: str = "mean") -> pd.Series:
    """Compute daily aggregate signal from cross-section of non-OHLCV factor.

    Args:
        factor_df: (T, N) DataFrame with factor values
        method: "mean" or "median"

    Returns:
        (T,) Series with daily aggregate values
    """
    if method == "mean":
        return factor_df.mean(axis=1)
    else:
        return factor_df.median(axis=1)


def compute_timing_overlay(
    agg_signal: pd.Series,
    direction: str,
    threshold_pct: float,
    scale_factor: float,
    lookback: int = 60,
) -> np.ndarray:
    """Compute timing overlay multiplier based on aggregate signal percentile.

    Args:
        agg_signal: Daily aggregate signal
        direction: "high_bearish" (high values → reduce), "low_bearish" (low values → reduce)
        threshold_pct: Percentile threshold for triggering (e.g. 0.80 = top 20%)
        scale_factor: Timing multiplier when triggered (e.g. 0.5 = half exposure)
        lookback: Rolling window for percentile computation

    Returns:
        (T,) array of multipliers (1.0 = normal, scale_factor when triggered)
    """
    T = len(agg_signal)
    overlay = np.ones(T, dtype=np.float64)
    values = agg_signal.values

    for t in range(lookback, T):
        window = values[max(0, t - lookback):t]
        valid = window[np.isfinite(window)]
        if len(valid) < 20:
            continue

        current = values[t]
        if np.isnan(current):
            continue

        # Compute percentile rank within rolling window
        pct_rank = np.mean(valid <= current)

        if direction == "high_bearish" and pct_rank >= threshold_pct:
            overlay[t] = scale_factor
        elif direction == "low_bearish" and pct_rank <= (1.0 - threshold_pct):
            overlay[t] = scale_factor

    return overlay


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

    daily_rets = np.diff(ho_eq) / ho_eq[:-1]
    if len(daily_rets) > 1 and np.std(daily_rets) > 1e-10:
        ho_sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)
    else:
        ho_sharpe = 0.0

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


# ── Timing overlay configurations ──
TIMING_CONFIGS = [
    # (name, factor_file, agg_method, direction, threshold_pct, scale_factor, lookback)
    # SHARE_CHG_10D: IC=-0.056 (high=bad), so high_bearish
    ("SCH10D_high80_s50", "SHARE_CHG_10D", "mean", "high_bearish", 0.80, 0.5, 60),
    ("SCH10D_high80_s30", "SHARE_CHG_10D", "mean", "high_bearish", 0.80, 0.3, 60),
    ("SCH10D_high80_s00", "SHARE_CHG_10D", "mean", "high_bearish", 0.80, 0.0, 60),
    ("SCH10D_high70_s50", "SHARE_CHG_10D", "mean", "high_bearish", 0.70, 0.5, 60),
    ("SCH10D_high70_s30", "SHARE_CHG_10D", "mean", "high_bearish", 0.70, 0.3, 60),
    # SHARE_CHG_5D: IC=-0.050 (high=bad)
    ("SCH5D_high80_s50", "SHARE_CHG_5D", "mean", "high_bearish", 0.80, 0.5, 60),
    ("SCH5D_high80_s30", "SHARE_CHG_5D", "mean", "high_bearish", 0.80, 0.3, 60),
    # SHARE_ACCEL: IC=+0.034 (low=bad, deceleration), so low_bearish
    ("ACCEL_low80_s50", "SHARE_ACCEL", "mean", "low_bearish", 0.80, 0.5, 60),
    ("ACCEL_low80_s30", "SHARE_ACCEL", "mean", "low_bearish", 0.80, 0.3, 60),
    # MARGIN_BUY_RATIO: IC=-0.031 (high=bad), so high_bearish
    ("MBR_high80_s50", "MARGIN_BUY_RATIO", "mean", "high_bearish", 0.80, 0.5, 60),
    # Wider lookback
    ("SCH10D_high80_s50_lb120", "SHARE_CHG_10D", "mean", "high_bearish", 0.80, 0.5, 120),
]


def main():
    print("=" * 80)
    print("P4: Aggregate Smart Money Timing (Portfolio-Level Exposure Scaling)")
    print("=" * 80)

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

    exec_model = load_execution_model(config)
    USE_T1_OPEN = exec_model.is_t1_open
    cost_model = load_cost_model(config)
    qdii_set = set(FrozenETFPool().qdii_codes)

    hyst_config = backtest_config.get("hysteresis", {})
    DELTA_RANK = float(hyst_config.get("delta_rank", 0.0))
    MIN_HOLD_DAYS = int(hyst_config.get("min_hold_days", 0))
    print(f"  HYSTERESIS: delta_rank={DELTA_RANK}, min_hold_days={MIN_HOLD_DAYS}")

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
    factors_3d = cached["factors_3d"]
    T, N, K = factors_3d.shape
    print(f"  Data: T={T}, N={N}, K={K}")

    factor_index_map = {name: idx for idx, name in enumerate(factor_names)}
    C2_FACTORS = ["AMIHUD_ILLIQUIDITY", "CALMAR_RATIO_60D", "CORRELATION_TO_MARKET_20D"]
    C2_INDICES = [factor_index_map[f] for f in C2_FACTORS]

    COST_ARR = build_cost_array(cost_model, list(etf_codes), qdii_set)

    close_prices = ohlcv["close"][etf_codes].ffill().fillna(1.0).values
    open_prices = ohlcv["open"][etf_codes].ffill().fillna(1.0).values
    high_prices = ohlcv["high"][etf_codes].ffill().fillna(1.0).values
    low_prices = ohlcv["low"][etf_codes].ffill().fillna(1.0).values

    # Timing
    timing_config = config.get("backtest", {}).get("timing", {})
    extreme_threshold = timing_config.get("extreme_threshold", -0.4)
    extreme_position = timing_config.get("extreme_position", 0.3)
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
    timing_arr_base = shift_timing_signal(timing_arr_raw)

    gate_arr = compute_regime_gate_arr(
        ohlcv["close"], dates, backtest_config=backtest_config
    )
    timing_arr_base = (timing_arr_base * gate_arr).astype(np.float64)

    risk_config = backtest_config.get("risk_control", {})
    trailing_stop_pct = risk_config.get("trailing_stop_pct", 0.08)
    leverage_cap = risk_config.get("leverage_cap", 1.0)
    stop_on_rebalance_only = risk_config.get("stop_check_on_rebalance_only", False)

    factors_dir = ROOT / "results" / "non_ohlcv_factors"
    holdout_start = config["data"].get("holdout_start", "2025-05-01")
    print(f"  Holdout start: {holdout_start}")

    results = []

    # ── Baseline: C2 with standard timing ──
    print(f"\n🔄 C2 baseline...")
    eq_c2, ret, _, _, trades, _, _ = run_vec_backtest(
        factors_3d, close_prices, open_prices, high_prices, low_prices,
        timing_arr_base, C2_INDICES,
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
    ho = compute_holdout_metrics(eq_c2, dates, holdout_start)
    results.append({"config": "C2_baseline", "full_return": ret, "trades": trades, **ho})
    print(f"  HO: ret={ho.get('ho_return',0)*100:.1f}%, MDD={ho.get('ho_mdd',0)*100:.1f}%, "
          f"Sharpe={ho.get('ho_sharpe',0):.2f}")

    # ── Test timing overlays ──
    for cfg_name, factor_file, agg_method, direction, threshold_pct, scale_factor, lookback_w in TIMING_CONFIGS:
        print(f"\n🔄 {cfg_name}...")
        try:
            factor_df = load_non_ohlcv_factor(factor_file, dates, etf_codes, factors_dir)
            agg_signal = compute_aggregate_signal(factor_df, agg_method)

            # Compute timing overlay with shifted signal (avoid lookahead)
            overlay = compute_timing_overlay(
                agg_signal, direction, threshold_pct, scale_factor, lookback_w
            )
            # Shift by 1 day to avoid lookahead (today's signal used tomorrow)
            overlay_shifted = np.ones(T, dtype=np.float64)
            overlay_shifted[1:] = overlay[:-1]

            # Apply overlay on top of base timing
            timing_arr_modified = timing_arr_base * overlay_shifted

            # Stats
            triggered_days = (overlay_shifted < 1.0).sum()
            triggered_pct = triggered_days / T * 100
            avg_scale_when_triggered = overlay_shifted[overlay_shifted < 1.0].mean() if triggered_days > 0 else 1.0
            print(f"  Triggered: {triggered_days}/{T} days ({triggered_pct:.1f}%), "
                  f"avg scale={avg_scale_when_triggered:.2f}")

            eq, ret, _, _, trades, _, _ = run_vec_backtest(
                factors_3d, close_prices, open_prices, high_prices, low_prices,
                timing_arr_modified, C2_INDICES,
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
            ho = compute_holdout_metrics(eq, dates, holdout_start)
            results.append({"config": f"C2+{cfg_name}", "full_return": ret, "trades": trades, **ho})
            print(f"  HO: ret={ho.get('ho_return',0)*100:.1f}%, MDD={ho.get('ho_mdd',0)*100:.1f}%, "
                  f"Sharpe={ho.get('ho_sharpe',0):.2f}")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({"config": f"C2+{cfg_name}", "full_return": np.nan, "trades": 0})

    # ── Results ──
    print("\n" + "=" * 105)
    print("P4 Aggregate Smart Money Timing — Results")
    print("=" * 105)
    print(f"{'Config':<35} {'Full%':>8} {'HO Ret%':>8} {'HO MDD%':>8} {'HO Sharpe':>10} {'HO WM%':>8} {'Trades':>7} {'Δ vs C2':>8}")
    print("-" * 105)

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

    # Highlight best
    if len(results) > 1:
        best = max(results[1:], key=lambda x: x.get("ho_return", -999))
        print(f"\n🏆 Best overlay: {best['config']} (HO {best.get('ho_return',0)*100:.1f}%)")
        best_delta = (best.get("ho_return", 0) - c2_ho_ret) * 100
        if best_delta > 1.0:
            print(f"  ✅ Improvement: {best_delta:+.1f}pp vs C2 baseline")
        elif best_delta > -1.0:
            print(f"  ⚠️ Marginal: {best_delta:+.1f}pp (within noise)")
        else:
            print(f"  ❌ All overlays worse than C2 baseline")

    # Save
    out_dir = ROOT / "results" / f"p4_agg_timing_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_dir / "p4_results.csv", index=False)
    print(f"\n💾 Saved to {out_dir}")


if __name__ == "__main__":
    main()
