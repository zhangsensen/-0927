
"""
WFO runner that scores combos using VEC backtest metrics (no lookahead):
Signal(T-1) -> Trade(T) -> Return(T+1).

Outputs:
- full_space_v3_no_lookahead_results.csv
- v3_top200_candidates_no_lookahead.csv

This runner coexists with the original IC-based flow and does not modify VEC/BT cores.
"""

import sys
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import combinations

# Project paths
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import shift_timing_signal
from etf_strategy.core.combo_wfo_optimizer import ComboWFOOptimizer

from scripts.batch_vec_backtest import run_vec_backtest, calculate_atr


def compute_period_metrics(equity_curve: np.ndarray) -> dict:
    """Compute ann_ret, max_dd, sharpe from an equity curve slice."""
    if equity_curve is None or len(equity_curve) < 2:
        return {"ann_ret": 0.0, "max_dd": 0.0, "sharpe": 0.0}
    eq = np.asarray(equity_curve, dtype=float)
    
    # 🛡️ Safety: Handle zeros or negative equity (bankruptcy)
    eq = np.maximum(eq, 1e-6)
    
    daily_ret = np.diff(eq) / eq[:-1]
    n = len(daily_ret)
    if n == 0:
        return {"ann_ret": 0.0, "max_dd": 0.0, "sharpe": 0.0}
    
    total_ret = eq[-1] / eq[0] - 1.0
    
    # 🛡️ Safety: Cap annual return to avoid overflow
    try:
        ann_ret = (1 + total_ret) ** (252 / max(1, n)) - 1
        if np.isinf(ann_ret) or np.isnan(ann_ret):
            ann_ret = -0.99
        elif ann_ret > 10.0: # Cap at 1000%
            ann_ret = 10.0
    except:
        ann_ret = -0.99

    # max drawdown
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = dd.min() if len(dd) else 0.0
    
    mu = daily_ret.mean()
    sigma = daily_ret.std(ddof=0)
    
    # 🛡️ Safety: Cap Sharpe
    sharpe = (mu / sigma * np.sqrt(252)) if sigma > 1e-12 else 0.0
    if np.isinf(sharpe) or np.isnan(sharpe):
        sharpe = 0.0
    elif sharpe > 20.0:
        sharpe = 20.0
    elif sharpe < -20.0:
        sharpe = -20.0
        
    return {"ann_ret": ann_ret, "max_dd": max_dd, "sharpe": sharpe}


def compute_yearly_returns(equity_curve: np.ndarray, dates: pd.DatetimeIndex) -> dict:
    if equity_curve is None or len(equity_curve) != len(dates):
        return {}
    eq = pd.Series(equity_curve, index=dates)
    yearly = {}
    for year, s in eq.groupby(eq.index.year):
        if len(s) < 2:
            yearly[year] = 0.0
            continue
        yearly[year] = s.iloc[-1] / s.iloc[0] - 1.0
    return yearly


def load_config():
    cfg_path = ROOT / "configs/combo_wfo_config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def prepare_data(config):
    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    data_end_date = config["data"].get("training_end_date") or config["data"]["end_date"]
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=data_end_date,
    )

    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv)
    factor_names_list = sorted(raw_factors_df.columns.get_level_values(0).unique().tolist())
    raw_factors = {fname: raw_factors_df[fname] for fname in factor_names_list}

    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors(raw_factors)

    dates = std_factors[factor_names_list[0]].index
    etf_codes = std_factors[factor_names_list[0]].columns.tolist()

    all_factors_stack = np.stack([std_factors[f].values for f in factor_names_list], axis=-1)

    close_prices = ohlcv["close"][etf_codes].ffill().bfill().values
    open_prices = ohlcv["open"][etf_codes].ffill().bfill().values
    high_prices = ohlcv["high"][etf_codes].ffill().bfill().values
    low_prices = ohlcv["low"][etf_codes].ffill().bfill().values

    # ATR (only if configured)
    stop_method = config["backtest"]["risk_control"].get("stop_method", "fixed")
    atr_arr = None
    if stop_method == "atr":
        atr_window = config["backtest"]["risk_control"].get("atr_window", 14)
        atr_arr = calculate_atr(high_prices, low_prices, close_prices, window=atr_window)

    # Timing
    timing_module = LightTimingModule(
        extreme_threshold=config["backtest"]["timing"]["extreme_threshold"],
        extreme_position=config["backtest"]["timing"]["extreme_position"],
    )
    timing_series = timing_module.compute_position_ratios(ohlcv["close"])
    timing_arr_raw = timing_series.reindex(dates).fillna(1.0).values
    timing_arr = shift_timing_signal(timing_arr_raw)

    # Vol regime (same as batch_bt_crosscheck)
    if "510300" in ohlcv["close"].columns:
        hs300 = ohlcv["close"]["510300"]
    else:
        hs300 = ohlcv["close"].iloc[:, 0]
    rets = hs300.pct_change()
    hv = rets.rolling(window=20, min_periods=20).std() * np.sqrt(252) * 100
    hv_5d = hv.shift(5)
    regime_vol = (hv + hv_5d) / 2
    exposure_s = pd.Series(1.0, index=regime_vol.index)
    exposure_s[regime_vol >= 25] = 0.7
    exposure_s[regime_vol >= 30] = 0.4
    exposure_s[regime_vol >= 40] = 0.1

    return {
        "dates": dates,
        "factor_names": factor_names_list,
        "all_factors_stack": all_factors_stack,
        "close_prices": close_prices,
        "open_prices": open_prices,
        "high_prices": high_prices,
        "low_prices": low_prices,
        "timing_arr": timing_arr,
        "vol_regime_arr": exposure_s.values,
        "etf_codes": etf_codes,
        "atr_arr": atr_arr,
    }


def generate_combos(factor_names, combo_sizes):
    combos = []
    for r in combo_sizes:
        for c in combinations(range(len(factor_names)), r):
            combos.append(tuple(c))
    return combos


def run_vec_on_window(combo_indices, data, config, start_idx, end_idx):
    lookback = config["backtest"]["lookback"]
    slice_start = max(0, start_idx - lookback)
    slice_end = end_idx

    # Slice data
    factors_slice = data["all_factors_stack"][slice_start:slice_end, :, :][..., list(combo_indices)]
    close_slice = data["close_prices"][slice_start:slice_end, :]
    open_slice = data["open_prices"][slice_start:slice_end, :]
    high_slice = data["high_prices"][slice_start:slice_end, :]
    low_slice = data["low_prices"][slice_start:slice_end, :]
    timing_slice = data["timing_arr"][slice_start:slice_end]
    vol_slice = data["vol_regime_arr"][slice_start:slice_end]

    atr_slice = data["atr_arr"][slice_start:slice_end] if data.get("atr_arr") is not None else None

    eq_curve, *_metrics = run_vec_backtest(
        factors_slice,
        close_slice,
        open_slice,
        high_slice,
        low_slice,
        timing_slice,
        list(range(len(combo_indices))),
        freq=config["backtest"]["freq"],
        pos_size=config["backtest"]["pos_size"],
        initial_capital=float(config["backtest"]["initial_capital"]),
        commission_rate=float(config["backtest"]["commission_rate"]),
        lookback=lookback,
        target_vol=config["backtest"]["risk_control"]["dynamic_leverage"]["target_vol"],
        vol_window=config["backtest"]["risk_control"]["dynamic_leverage"]["vol_window"],
        dynamic_leverage_enabled=config["backtest"]["risk_control"]["dynamic_leverage"]["enabled"],
        vol_regime_arr=vol_slice,
        use_atr_stop=config["backtest"]["risk_control"].get("stop_method", "fixed") == "atr",
        trailing_stop_pct=config["backtest"]["risk_control"].get("trailing_stop_pct", 0.0),
        atr_arr=atr_slice,
        atr_multiplier=config["backtest"]["risk_control"].get("atr_multiplier", 3.0),
        stop_on_rebalance_only=config["backtest"]["risk_control"].get("stop_check_on_rebalance_only", False),
        individual_trend_arr=None,
        individual_trend_enabled=config["backtest"]["timing"].get("individual_timing", {}).get("enabled", False),
        profit_ladders=config["backtest"]["risk_control"].get("profit_ladders", []),
        circuit_breaker_day=config["backtest"]["risk_control"].get("circuit_breaker", {}).get("max_drawdown_day", 0.0),
        circuit_breaker_total=config["backtest"]["risk_control"].get("circuit_breaker", {}).get("max_drawdown_total", 0.0),
        circuit_recovery_days=config["backtest"]["risk_control"].get("circuit_breaker", {}).get("recovery_days", 5),
        cooldown_days=config["backtest"]["risk_control"].get("cooldown_days", 0),
        leverage_cap=config["backtest"]["risk_control"].get("leverage_cap", 1.0),
    )

    # Focus on IS segment only
    rel_start = start_idx - slice_start
    if rel_start < 0:
        rel_start = 0
    eq_is = eq_curve[rel_start:]
    return compute_period_metrics(eq_is)


def run_full_vec(combo_indices, data, config):
    atr_arr = data["atr_arr"]

    eq_curve, total_ret, win_rate, profit_factor, num_trades, *_ = run_vec_backtest(
        data["all_factors_stack"][..., list(combo_indices)],
        data["close_prices"],
        data["open_prices"],
        data["high_prices"],
        data["low_prices"],
        data["timing_arr"],
        list(range(len(combo_indices))),
        freq=config["backtest"]["freq"],
        pos_size=config["backtest"]["pos_size"],
        initial_capital=float(config["backtest"]["initial_capital"]),
        commission_rate=float(config["backtest"]["commission_rate"]),
        lookback=config["backtest"]["lookback"],
        target_vol=config["backtest"]["risk_control"]["dynamic_leverage"]["target_vol"],
        vol_window=config["backtest"]["risk_control"]["dynamic_leverage"]["vol_window"],
        dynamic_leverage_enabled=config["backtest"]["risk_control"]["dynamic_leverage"]["enabled"],
        vol_regime_arr=data["vol_regime_arr"],
        use_atr_stop=config["backtest"]["risk_control"].get("stop_method", "fixed") == "atr",
        trailing_stop_pct=config["backtest"]["risk_control"].get("trailing_stop_pct", 0.0),
        atr_arr=atr_arr,
        atr_multiplier=config["backtest"]["risk_control"].get("atr_multiplier", 3.0),
        stop_on_rebalance_only=config["backtest"]["risk_control"].get("stop_check_on_rebalance_only", False),
        individual_trend_arr=None,
        individual_trend_enabled=config["backtest"]["timing"].get("individual_timing", {}).get("enabled", False),
        profit_ladders=config["backtest"]["risk_control"].get("profit_ladders", []),
        circuit_breaker_day=config["backtest"]["risk_control"].get("circuit_breaker", {}).get("max_drawdown_day", 0.0),
        circuit_breaker_total=config["backtest"]["risk_control"].get("circuit_breaker", {}).get("max_drawdown_total", 0.0),
        circuit_recovery_days=config["backtest"]["risk_control"].get("circuit_breaker", {}).get("recovery_days", 5),
        cooldown_days=config["backtest"]["risk_control"].get("cooldown_days", 0),
        leverage_cap=config["backtest"]["risk_control"].get("leverage_cap", 1.0),
    )
    metrics = compute_period_metrics(eq_curve)
    metrics['num_trades'] = num_trades
    metrics['win_rate'] = win_rate
    yearly = compute_yearly_returns(eq_curve, data["dates"])
    return metrics, yearly


def main():
    config = load_config()
    data = prepare_data(config)
    combo_sizes = config["combo_wfo"]["combo_sizes"]
    combos = generate_combos(data["factor_names"], combo_sizes)

    optimizer = ComboWFOOptimizer(
        combo_sizes=combo_sizes,
        is_period=config["combo_wfo"]["is_period"],
        oos_period=config["combo_wfo"]["oos_period"],
        step_size=config["combo_wfo"]["step_size"],
        n_jobs=config["combo_wfo"]["n_jobs"],
        verbose=1 if config["combo_wfo"].get("verbose", True) else 0,
        enable_fdr=config["combo_wfo"].get("enable_fdr", True),
        fdr_alpha=config["combo_wfo"].get("fdr_alpha", 0.05),
        complexity_penalty_lambda=config["combo_wfo"].get("scoring", {}).get("complexity_penalty_lambda", 0.0),
        rebalance_frequencies=config["combo_wfo"].get("rebalance_frequencies", [3]),
    )
    windows = optimizer._generate_windows(len(data["dates"]))
    print(f"Generated {len(windows)} WFO windows for VEC scoring")

    weight_cfg = config["combo_wfo"]["scoring"].get("vec_score_weights", {"ann_ret": 0.4, "sharpe": 0.3, "max_dd": 0.3})
    w_ann = weight_cfg.get("ann_ret", 0.4)
    w_sharpe = weight_cfg.get("sharpe", 0.3)
    w_dd = weight_cfg.get("max_dd", 0.3)

    window_records = []
    full_records = []

    for combo_idx, combo in enumerate(tqdm(combos, desc="VEC scoring per combo")):
        # Window metrics
        for w_idx, (is_range, _oos_range) in enumerate(windows):
            is_start, is_end = is_range
            win_metrics = run_vec_on_window(combo, data, config, is_start, is_end)
            window_records.append({
                "combo_id": combo_idx,
                "window": w_idx,
                "ann_ret": win_metrics["ann_ret"],
                "max_dd": win_metrics["max_dd"],
                "sharpe": win_metrics["sharpe"],
            })

        # Full-sample VEC metrics
        metrics_full, yearly = run_full_vec(combo, data, config)
        combo_name = " + ".join([data["factor_names"][i] for i in combo])
        rec = {
            "combo_id": combo_idx,
            "combo": combo_name,
            "ann_ret": metrics_full["ann_ret"],
            "max_dd": metrics_full["max_dd"],
            "sharpe": metrics_full["sharpe"],
            "num_trades": metrics_full.get("num_trades", 0),
            "win_rate": metrics_full.get("win_rate", 0.0),
        }
        for yr, val in yearly.items():
            rec[f"ret_{yr}"] = val
        full_records.append(rec)

    df_window = pd.DataFrame(window_records)
    # Ranking per window
    df_window["ann_rank"] = df_window.groupby("window")["ann_ret"].rank(ascending=False, method="average")
    df_window["sharpe_rank"] = df_window.groupby("window")["sharpe"].rank(ascending=False, method="average")
    df_window["dd_rank"] = df_window.groupby("window")["max_dd"].rank(ascending=True, method="average")
    df_window["score"] = w_ann * df_window["ann_rank"] + w_sharpe * df_window["sharpe_rank"] + w_dd * df_window["dd_rank"]

    agg = df_window.groupby("combo_id")["score"].mean().reset_index().rename(columns={"score": "wfo_vec_score"})

    df_full = pd.DataFrame(full_records)
    df_all = df_full.merge(agg, on="combo_id", how="left")

    # Save full space results
    full_path = ROOT / "results" / "full_space_v3_no_lookahead_results.csv"
    df_all.to_csv(full_path, index=False)
    print(f"Saved full space results to {full_path}")

    # Hard filters
    def pass_filters(row):
        if row["ann_ret"] <= 0.12:
            return False
        if row["max_dd"] <= -0.30:
            return False
        if row.get("ret_2022", 0.0) <= 0.0:
            return False
        if row.get("ret_2024", 0.0) <= 0.0:
            return False
        return True

    filtered = df_all[df_all.apply(pass_filters, axis=1)].copy()
    filtered = filtered.sort_values("wfo_vec_score", ascending=True)  # lower rank sum is better
    top200 = filtered.head(200)
    top_path = ROOT / "results" / "v3_top200_candidates_no_lookahead.csv"
    top200.to_csv(top_path, index=False)
    print(f"Saved Top200 to {top_path}")


if __name__ == "__main__":
    main()
