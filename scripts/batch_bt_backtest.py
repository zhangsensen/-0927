#!/usr/bin/env python3
"""
批量 BT 回测：遍历 WFO 输出的全部组合，逐个用 Backtrader GenericStrategy 回测并保存结果。
"""
import gc
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import yaml
import pandas as pd
import numpy as np
import backtrader as bt
from tqdm import tqdm
from datetime import datetime

from etf_strategy.core.utils.run_meta import write_step_meta
from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.factor_cache import FactorCache
from etf_strategy.core.cost_model import load_cost_model, CostModel
from etf_strategy.core.frozen_params import load_frozen_config, FrozenETFPool
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import (
    shift_timing_signal,
    generate_rebalance_schedule,
)
from etf_strategy.regime_gate import compute_regime_gate_arr, gate_stats
from etf_strategy.auditor.core.engine import GenericStrategy, PandasData
from aligned_metrics import compute_aligned_metrics

# ✅ P0: 删除硬编码 - 所有参数必须从配置文件读取
# FREQ = 8  # DELETED
# POS_SIZE = 3  # DELETED
# INITIAL_CAPITAL = 1_000_000.0  # DELETED
# COMMISSION_RATE = 0.0002  # DELETED
# LOOKBACK = 252  # DELETED


def run_bt_backtest(
    combined_score_df,
    timing_series,
    vol_regime_series,
    etf_codes,
    data_feeds,
    rebalance_schedule,
    freq,
    pos_size,
    initial_capital,
    commission_rate,
    target_vol=0.20,
    vol_window=20,
    dynamic_leverage_enabled=True,
    collect_daily_returns: bool = False,
    use_t1_open: bool = False,
    cost_model: CostModel | None = None,
    qdii_codes: set | None = None,
):
    """单组合 BT 回测引擎，返回收益和风险指标"""
    cerebro = bt.Cerebro(cheat_on_open=use_t1_open)
    cerebro.broker.setcash(initial_capital)
    # ✅ Exp2: 按标的设置佣金 (SPLIT_MARKET 模式)
    if cost_model is not None and cost_model.is_split_market and qdii_codes is not None:
        for ticker in data_feeds:
            rate = cost_model.get_cost(ticker, qdii_codes)
            cerebro.broker.setcommission(commission=rate, name=ticker, leverage=1.0)
    else:
        cerebro.broker.setcommission(commission=commission_rate, leverage=1.0)
    if use_t1_open:
        cerebro.broker.set_coc(False)
        cerebro.broker.set_coo(True)  # Cheat-On-Open: 订单在提交当 bar 的 open 成交
    else:
        cerebro.broker.set_coc(True)
    cerebro.broker.set_checksubmit(False)

    for ticker, df in data_feeds.items():
        data = PandasData(dataname=df, name=ticker)
        cerebro.adddata(data)

    cerebro.addstrategy(
        GenericStrategy,
        scores=combined_score_df,
        timing=timing_series,
        vol_regime=vol_regime_series,
        etf_codes=etf_codes,
        freq=freq,
        pos_size=pos_size,
        rebalance_schedule=rebalance_schedule,
        # ✅ P2: 动态降权参数
        target_vol=target_vol,
        vol_window=vol_window,
        dynamic_leverage_enabled=dynamic_leverage_enabled,
        # ✅ Exp1: T+1 Open
        use_t1_open=use_t1_open,
    )

    # ✅ P0: 添加 Analyzers 计算风险指标
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio,
        _name="sharpe",
        timeframe=bt.TimeFrame.Days,
        compression=1,
        riskfreerate=0.0,
        annualize=True,
    )
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name="annual_return")
    # ✅ P3: 添加 TradeAnalyzer 以获取交易详情
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    if collect_daily_returns:
        cerebro.addanalyzer(
            bt.analyzers.TimeReturn, _name="timereturn", timeframe=bt.TimeFrame.Days
        )

    start_val = cerebro.broker.getvalue()
    results = cerebro.run()
    end_val = cerebro.broker.getvalue()
    strat = results[0]

    bt_return = (end_val / start_val) - 1
    daily_returns = None
    if collect_daily_returns:
        try:
            tr_analysis = strat.analyzers.timereturn.get_analysis()
            if isinstance(tr_analysis, dict):
                # Preserve date index for slicing (train/holdout audit)
                daily_returns = pd.Series(tr_analysis).sort_index()
            else:
                daily_returns = pd.Series(list(tr_analysis))
        except Exception:
            daily_returns = None

    # ✅ P0: 提取风险指标
    # DrawDown Analyzer
    dd_analysis = strat.analyzers.drawdown.get_analysis()
    max_drawdown = dd_analysis.get("max", {}).get("drawdown", 0.0) / 100.0  # 转换为小数

    # Sharpe Analyzer
    sharpe_analysis = strat.analyzers.sharpe.get_analysis()
    sharpe_ratio = sharpe_analysis.get("sharperatio", 0.0)
    if sharpe_ratio is None:
        sharpe_ratio = 0.0

    # Returns Analyzer
    returns_analysis = strat.analyzers.returns.get_analysis()

    # Trade Analyzer
    trade_analysis = strat.analyzers.trades.get_analysis()
    total_trades = trade_analysis.get("total", {}).get("total", 0)
    win_trades = trade_analysis.get("won", {}).get("total", 0)
    loss_trades = trade_analysis.get("lost", {}).get("total", 0)
    win_rate = win_trades / total_trades if total_trades > 0 else 0.0

    # Avg Holding Period (in bars/days)
    len_stats = trade_analysis.get("len", {})
    avg_len = len_stats.get("average", 0.0)
    max_len = len_stats.get("max", 0)
    min_len = len_stats.get("min", 0)

    # PnL Stats
    pnl_stats = trade_analysis.get("pnl", {}).get("net", {})
    avg_pnl = pnl_stats.get("average", 0.0)
    total_pnl = pnl_stats.get("total", 0.0)

    # Profit Factor
    won_pnl = trade_analysis.get("won", {}).get("pnl", {}).get("total", 0.0)
    lost_pnl = abs(trade_analysis.get("lost", {}).get("pnl", {}).get("total", 0.0))
    profit_factor = won_pnl / lost_pnl if lost_pnl > 0 else float("inf")

    # 计算年化收益（与 VEC 一致的计算方式）
    trading_days = len(combined_score_df)
    years = trading_days / 252.0 if trading_days > 0 else 1.0
    annual_return = (1.0 + bt_return) ** (1.0 / years) - 1.0 if years > 0 else 0.0

    # 估算年化波动率：从 Sharpe 和年化收益反推
    # sharpe = annual_return / annual_vol => annual_vol = annual_return / sharpe
    if sharpe_ratio != 0.0 and abs(sharpe_ratio) > 0.0001:
        annual_volatility = abs(annual_return / sharpe_ratio)
    else:
        annual_volatility = 0.0

    start_idx = rebalance_schedule[0] if len(rebalance_schedule) > 0 else 0
    equity_curve = None
    if daily_returns is not None and len(daily_returns) > 0:
        dr_arr = np.asarray(daily_returns.values, dtype=np.float64)
        equity_curve = np.concatenate(
            [
                np.array([initial_capital], dtype=np.float64),
                initial_capital * np.cumprod(1.0 + dr_arr),
            ]
        )
    else:
        equity_curve = np.array([start_val, end_val], dtype=np.float64)
    aligned_metrics = compute_aligned_metrics(equity_curve, start_idx=start_idx)

    # Calmar Ratio
    calmar_ratio = annual_return / max_drawdown if max_drawdown > 0.0001 else 0.0

    risk_metrics = {
        "max_drawdown": max_drawdown,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": sharpe_ratio,
        "calmar_ratio": calmar_ratio,
        "aligned_return": aligned_metrics["aligned_return"],
        "aligned_sharpe": aligned_metrics["aligned_sharpe"],
        # ✅ P3: 添加交易详情
        "total_trades": total_trades,
        "win_rate": win_rate,
        "avg_len": avg_len,
        "max_len": max_len,
        "profit_factor": profit_factor,
        "avg_pnl": avg_pnl,
    }

    return bt_return, strat.margin_failures, risk_metrics, daily_returns


import multiprocessing as mp
from functools import partial

# 全局变量，用于子进程共享数据 (Copy-on-Write)
_shared_data = {}


def init_worker(
    data_feeds,
    std_factors,
    timing_series,
    vol_regime_series,
    etf_codes,
    target_vol,
    vol_window,
    dynamic_leverage_enabled,
    freq,
    pos_size,
    initial_capital,
    commission_rate,
    lookback,
    training_end_ts,
    use_t1_open,
    cost_model=None,
    qdii_codes=None,
):
    """子进程初始化：保存共享数据"""
    global _shared_data
    _shared_data["use_t1_open"] = use_t1_open
    _shared_data["data_feeds"] = data_feeds
    _shared_data["std_factors"] = std_factors
    _shared_data["timing_series"] = timing_series
    _shared_data["vol_regime_series"] = vol_regime_series
    _shared_data["etf_codes"] = etf_codes

    # ✅ P2: 动态降权参数
    _shared_data["target_vol"] = target_vol
    _shared_data["vol_window"] = vol_window
    _shared_data["dynamic_leverage_enabled"] = dynamic_leverage_enabled

    # ✅ P0: 保存配置参数
    _shared_data["freq"] = freq
    _shared_data["pos_size"] = pos_size
    _shared_data["initial_capital"] = initial_capital
    _shared_data["commission_rate"] = commission_rate
    _shared_data["training_end_date"] = training_end_ts

    # ✅ Exp2: 成本模型
    _shared_data["cost_model"] = cost_model
    _shared_data["qdii_codes"] = qdii_codes

    # ✅ 预计算调仓日程 (所有组合共享)
    T = len(timing_series)
    _shared_data["rebalance_schedule"] = generate_rebalance_schedule(
        total_periods=T,
        lookback_window=lookback,
        freq=freq,
    )


import numpy as np


def process_combo(row_data):
    """单个组合的处理函数"""
    # 禁用 GC 以提升性能（子进程生命周期短，无需 GC）
    gc.disable()

    combo_str = row_data["combo"]

    # 从全局变量获取数据
    data_feeds = _shared_data["data_feeds"]
    std_factors = _shared_data["std_factors"]
    timing_series = _shared_data["timing_series"]
    vol_regime_series = _shared_data["vol_regime_series"]
    etf_codes = _shared_data["etf_codes"]
    rebalance_schedule = _shared_data["rebalance_schedule"]
    training_end_ts = _shared_data.get("training_end_date")

    # ✅ P2: 动态降权参数
    target_vol = _shared_data["target_vol"]
    vol_window = _shared_data["vol_window"]
    dynamic_leverage_enabled = _shared_data["dynamic_leverage_enabled"]

    # ✅ P0: 获取配置参数
    freq = _shared_data["freq"]
    pos_size = _shared_data["pos_size"]
    initial_capital = _shared_data["initial_capital"]
    commission_rate = _shared_data["commission_rate"]
    use_t1_open = _shared_data.get("use_t1_open", False)
    cost_model = _shared_data.get("cost_model")
    qdii_codes = _shared_data.get("qdii_codes")

    factors = [f.strip() for f in combo_str.split(" + ")]
    dates = timing_series.index

    # 构造得分矩阵 (使用 DataFrame.add 保持 NaN 处理一致性)
    # ✅ 与 full_vec_bt_comparison.py 保持一致：fill_value=0 避免 NaN 传播
    combined_score_df = pd.DataFrame(0.0, index=dates, columns=etf_codes)
    for f in factors:
        combined_score_df = combined_score_df.add(std_factors[f], fill_value=0)

    # 运行回测
    bt_return, margin_failures, risk_metrics, daily_returns_s = run_bt_backtest(
        combined_score_df,
        timing_series,
        vol_regime_series,
        etf_codes,
        data_feeds,
        rebalance_schedule,
        freq,
        pos_size,
        initial_capital,
        commission_rate,
        target_vol,
        vol_window,
        dynamic_leverage_enabled,
        collect_daily_returns=True,
        use_t1_open=use_t1_open,
        cost_model=cost_model,
        qdii_codes=qdii_codes,
    )

    bt_train_return = np.nan
    bt_holdout_return = np.nan
    if isinstance(daily_returns_s, pd.Series) and len(daily_returns_s) > 0:
        eq = (1.0 + daily_returns_s.fillna(0.0)).cumprod() * float(initial_capital)
        eq = eq.sort_index()
        # Normalize index for safe comparisons
        try:
            eq.index = pd.to_datetime(eq.index)
        except Exception:
            pass

        if training_end_ts is not None:
            train_end = pd.to_datetime(training_end_ts)
            train_eq = eq.loc[eq.index <= train_end]
            hold_eq = eq.loc[eq.index > train_end]

            if len(train_eq) >= 2:
                bt_train_return = (train_eq.iloc[-1] / train_eq.iloc[0]) - 1.0
            elif len(train_eq) == 1:
                bt_train_return = 0.0

            if len(hold_eq) >= 2:
                bt_holdout_return = (hold_eq.iloc[-1] / hold_eq.iloc[0]) - 1.0
            elif len(hold_eq) == 1:
                bt_holdout_return = 0.0
        else:
            bt_train_return = bt_return

    return {
        "combo": combo_str,
        "bt_return": bt_return,
        "bt_train_return": bt_train_return,
        "bt_holdout_return": bt_holdout_return,
        "bt_margin_failures": margin_failures,
        # ✅ P0: 风险指标
        "bt_max_drawdown": risk_metrics["max_drawdown"],
        "bt_annual_return": risk_metrics["annual_return"],
        "bt_annual_volatility": risk_metrics["annual_volatility"],
        "bt_sharpe_ratio": risk_metrics["sharpe_ratio"],
        "bt_calmar_ratio": risk_metrics["calmar_ratio"],
        "bt_aligned_return": risk_metrics["aligned_return"],
        "bt_aligned_sharpe": risk_metrics["aligned_sharpe"],
        # ✅ P3: 交易详情
        "bt_total_trades": risk_metrics["total_trades"],
        "bt_win_rate": risk_metrics["win_rate"],
        "bt_avg_len": risk_metrics["avg_len"],
        "bt_max_len": risk_metrics["max_len"],
        "bt_profit_factor": risk_metrics["profit_factor"],
        "bt_avg_pnl": risk_metrics["avg_pnl"],
    }


def main():
    parser = argparse.ArgumentParser(description="批量 BT 回测 (支持 Top-K 筛选)")
    parser.add_argument(
        "--topk", type=int, default=None, help="仅回测 VEC 收益最高的 Top-K 个组合"
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="total_return",
        help="排序字段 (默认: total_return)",
    )
    parser.add_argument(
        "--combos", type=str, default=None, help="指定组合文件路径 (parquet)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径 (yaml)。默认使用 configs/combo_wfo_config.yaml",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("批量 BT 回测：多进程并行版 (Ryzen 9950X Optimized)")
    if args.combos:
        print(f"🎯 指定组合文件: {args.combos}")
    elif args.topk:
        print(f"🎯 筛选模式: Top {args.topk} (按 {args.sort_by} 排序)")
    else:
        print("⚙️ 全量模式: 回测所有组合")
    print("=" * 80)

    # 1. 加载 WFO 结果（优先 run_* 目录，兼容旧 unified_wfo_*）
    if args.combos:
        combos_path = Path(args.combos)
        if not combos_path.exists():
            print(f"❌ 错误: 文件不存在 {combos_path}")
            sys.exit(1)
        df_combos = pd.read_parquet(combos_path)
        # ✅ 支持 combos 模式下的 Top-K 筛选
        if args.topk:
            if args.sort_by not in df_combos.columns:
                print(
                    f"⚠️ 警告: 列 {args.sort_by} 不存在，无法排序。将使用原始顺序并截取 Top {args.topk}。"
                )
                df_combos = df_combos.head(args.topk)
            else:
                df_combos = df_combos.sort_values(args.sort_by, ascending=False).head(
                    args.topk
                )
                print(
                    f"✅ 已筛选 Top {len(df_combos)} 组合 (Min {args.sort_by}: {df_combos[args.sort_by].min():.4f})"
                )
    else:
        results_root = ROOT / "results"
        wfo_dirs = sorted(
            [d for d in results_root.glob("run_*") if d.is_dir() and not d.is_symlink()]
        )
        if not wfo_dirs:
            wfo_dirs = sorted(results_root.glob("unified_wfo_*"))

        if not wfo_dirs:
            print("❌ 未找到 WFO 结果目录")
            return

        latest_wfo = wfo_dirs[-1]
        combos_path = latest_wfo / "top100_by_ic.parquet"
        if not combos_path.exists():
            combos_path = latest_wfo / "all_combos.parquet"

        if not combos_path.exists():
            print(f"❌ 未找到 {combos_path}")
            return

        df_combos = pd.read_parquet(combos_path)
        print(f"✅ 加载 WFO 结果 ({latest_wfo.name})：{len(df_combos)} 个组合")

        # 筛选 Top-K
        if args.topk:
            if args.sort_by not in df_combos.columns:
                print(f"⚠️ 警告: 列 {args.sort_by} 不存在，无法排序。将使用原始顺序。")
            else:
                df_combos = df_combos.sort_values(args.sort_by, ascending=False).head(
                    args.topk
                )
                print(
                    f"✅ 已筛选 Top {len(df_combos)} 组合 (Min {args.sort_by}: {df_combos[args.sort_by].min():.4f})"
                )

    # 2. 加载数据
    config_path = (
        Path(args.config) if args.config else (ROOT / "configs/combo_wfo_config.yaml")
    )
    if not config_path.is_absolute():
        config_path = (ROOT / config_path).resolve()
    if not config_path.exists():
        print(f"❌ 错误: 配置文件不存在 {config_path}")
        sys.exit(1)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    frozen = load_frozen_config(config, config_path=str(config_path))
    print(f"🔒 参数冻结校验通过 (version={frozen.version})")

    # ✅ Exp1: 执行模型
    from etf_strategy.core.execution_model import load_execution_model
    exec_model = load_execution_model(config)
    USE_T1_OPEN = exec_model.is_t1_open
    print(f"   EXECUTION_MODEL: {exec_model.mode}")

    training_end_date = config.get("data", {}).get("training_end_date")
    training_end_ts = pd.to_datetime(training_end_date) if training_end_date else None

    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )

    # 3. 计算因子 (带缓存)
    factor_cache = FactorCache(cache_dir=Path(config["data"].get("cache_dir") or ".cache"))
    cached = factor_cache.get_or_compute(
        ohlcv=ohlcv,
        config=config,
        data_dir=loader.data_dir,
    )
    std_factors = cached["std_factors"]

    factor_names = sorted(std_factors.keys())
    first_factor = std_factors[factor_names[0]]
    dates = first_factor.index
    etf_codes = first_factor.columns.tolist()

    # ✅ P0: 从配置文件读取回测参数
    backtest_config = config.get("backtest", {})
    freq = backtest_config.get("freq", 8)
    pos_size = backtest_config.get("pos_size", 3)
    initial_capital = float(backtest_config.get("initial_capital", 1_000_000.0))
    commission_rate = float(backtest_config.get("commission_rate", 0.0002))
    lookback = backtest_config.get("lookback", 252)

    # ✅ Exp2: 加载成本模型
    cost_model = load_cost_model(config)
    qdii_codes = set(FrozenETFPool().qdii_codes)
    tier = cost_model.active_tier
    print(
        f"✅ 回测参数: FREQ={freq}, POS={pos_size}, Capital={initial_capital}, Comm={commission_rate}"
    )
    print(
        f"✅ 成本模型: mode={cost_model.mode}, tier={cost_model.tier}, "
        f"A股={tier.a_share*10000:.0f}bp, QDII={tier.qdii*10000:.0f}bp"
    )

    # ✅ P1: 从配置文件读取择时参数
    timing_config = config.get("backtest", {}).get("timing", {})
    extreme_threshold = timing_config.get("extreme_threshold", -0.4)
    extreme_position = timing_config.get("extreme_position", 0.3)
    print(f"✅ 择时参数: threshold={extreme_threshold}, position={extreme_position}")

    # ✅ P2: 从配置文件读取动态杠杆参数
    dl_config = (
        config.get("backtest", {}).get("risk_control", {}).get("dynamic_leverage", {})
    )
    dynamic_leverage_enabled = dl_config.get("enabled", False)
    target_vol = dl_config.get("target_vol", 0.20)
    vol_window = dl_config.get("vol_window", 20)
    print(
        f"✅ 动态杠杆: enabled={dynamic_leverage_enabled}, target_vol={target_vol}, vol_window={vol_window}"
    )

    timing_module = LightTimingModule(
        extreme_threshold=extreme_threshold,
        extreme_position=extreme_position,
    )
    timing_series_raw = timing_module.compute_position_ratios(ohlcv["close"])
    # ✅ 使用 shift_timing_signal 做 t-1 shift，避免未来函数
    timing_arr_shifted = shift_timing_signal(
        timing_series_raw.reindex(dates).fillna(1.0).values
    )
    timing_series = pd.Series(timing_arr_shifted, index=dates)

    # ✅ v3.2: Regime gate（可选），通过缩放 timing_series 实现统一降仓/停跑
    gate_arr = compute_regime_gate_arr(
        ohlcv["close"], dates, backtest_config=backtest_config
    )
    timing_series = timing_series * pd.Series(gate_arr, index=dates)
    if bool(backtest_config.get("regime_gate", {}).get("enabled", False)):
        s = gate_stats(gate_arr)
        print(
            f"✅ Regime gate enabled: mean={s['mean']:.2f}, min={s['min']:.2f}, max={s['max']:.2f}"
        )

    # ✅ v3.1: 波动率体制 (与 VEC 保持一致)
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
    vol_regime_series = exposure_s.reindex(dates).fillna(1.0)

    # 准备 data feeds
    data_feeds = {}
    for ticker in etf_codes:
        df = pd.DataFrame(
            {
                "open": ohlcv["open"][ticker],
                "high": ohlcv["high"][ticker],
                "low": ohlcv["low"][ticker],
                "close": ohlcv["close"][ticker],
                "volume": ohlcv["volume"][ticker],
            }
        )
        df = df.reindex(dates)
        df = df.ffill().fillna(0.01)
        data_feeds[ticker] = df

    print(f"✅ 数据加载完成：{len(dates)} 天 × {len(etf_codes)} 只 ETF")

    # 4. 多进程回测
    # 自动检测: 物理核心数 (16), compute-bound BT 任务 SMT 收益 <5%
    import os as _os
    num_workers = int(_os.environ.get("BT_NUM_WORKERS", min((_os.cpu_count() or 8) // 2, 16)))
    print(f"🚀 启动多进程回测 (Workers: {num_workers})...")

    # 准备任务列表 (转换为 dict 列表以便传递)
    tasks = [row.to_dict() for _, row in df_combos.iterrows()]

    print(f"🚀 准备回测 {len(tasks)} 个组合...")

    results = []
    with mp.Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(
            data_feeds,
            std_factors,
            timing_series,
            vol_regime_series,
            etf_codes,
            target_vol,
            vol_window,
            dynamic_leverage_enabled,
            freq,
            pos_size,
            initial_capital,
            commission_rate,
            lookback,
            training_end_ts,
            USE_T1_OPEN,
            cost_model,
            qdii_codes,
        ),
    ) as pool:
        # 使用 imap_unordered 获取实时进度
        for res in tqdm(
            pool.imap(process_combo, tasks), total=len(tasks), desc="BT 并行回测"
        ):
            results.append(res)

    # 5. 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_top{args.topk}" if args.topk else "_full"
    output_dir = ROOT / "results" / f"bt_backtest{suffix}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    df_results = pd.DataFrame(results)
    df_results.to_parquet(output_dir / "bt_results.parquet", index=False)

    write_step_meta(output_dir, step="bt", inputs={"combos": str(args.combos)}, config=str(args.config or "default"), extras={"combo_count": len(df_results), "topk": args.topk})

    print(f"\n✅ BT 批量回测完成")
    print(f"   输出目录: {output_dir}")
    print(f"   组合数: {len(df_results)}")
    print(f"   Margin 失败总数: {df_results['bt_margin_failures'].sum()}")


if __name__ == "__main__":
    main()
