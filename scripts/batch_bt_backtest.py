#!/usr/bin/env python3
"""
批量 BT 回测：遍历 WFO 输出的全部组合，逐个用 Backtrader GenericStrategy 回测并保存结果。
"""
import gc
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "etf_rotation_optimized"))
sys.path.insert(0, str(ROOT))

import yaml
import pandas as pd
import numpy as np
import backtrader as bt
from tqdm import tqdm
from datetime import datetime

from core.data_loader import DataLoader
from core.precise_factor_library_v2 import PreciseFactorLibrary
from core.cross_section_processor import CrossSectionProcessor
from core.market_timing import LightTimingModule
from core.utils.rebalance import shift_timing_signal, generate_rebalance_schedule
from strategy_auditor.core.engine import GenericStrategy, PandasData

FREQ = 8
POS_SIZE = 3
INITIAL_CAPITAL = 1_000_000.0
COMMISSION_RATE = 0.0002
LOOKBACK = 252


def run_bt_backtest(combined_score_df, timing_series, etf_codes, data_feeds, rebalance_schedule):
    """单组合 BT 回测引擎"""
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(INITIAL_CAPITAL)
    cerebro.broker.setcommission(commission=COMMISSION_RATE, leverage=1.0)
    cerebro.broker.set_coc(True)
    cerebro.broker.set_checksubmit(False)

    for ticker, df in data_feeds.items():
        data = PandasData(dataname=df, name=ticker)
        cerebro.adddata(data)

    cerebro.addstrategy(
        GenericStrategy, 
        scores=combined_score_df, 
        timing=timing_series, 
        etf_codes=etf_codes, 
        freq=FREQ, 
        pos_size=POS_SIZE,
        rebalance_schedule=rebalance_schedule
    )

    start_val = cerebro.broker.getvalue()
    results = cerebro.run()
    end_val = cerebro.broker.getvalue()
    strat = results[0]

    bt_return = (end_val / start_val) - 1

    return bt_return, strat.margin_failures


import multiprocessing as mp
from functools import partial

# 全局变量，用于子进程共享数据 (Copy-on-Write)
_shared_data = {}

def init_worker(data_feeds, std_factors, timing_series, etf_codes):
    """子进程初始化：保存共享数据"""
    global _shared_data
    _shared_data['data_feeds'] = data_feeds
    _shared_data['std_factors'] = std_factors
    _shared_data['timing_series'] = timing_series
    _shared_data['etf_codes'] = etf_codes
    
    # ✅ 预计算调仓日程 (所有组合共享)
    T = len(timing_series)
    _shared_data['rebalance_schedule'] = generate_rebalance_schedule(
        total_periods=T,
        lookback_window=LOOKBACK,
        freq=FREQ,
    )

import numpy as np

def process_combo(row_data):
    """单个组合的处理函数"""
    # 禁用 GC 以提升性能（子进程生命周期短，无需 GC）
    gc.disable()
    
    combo_str = row_data['combo']
    
    # 从全局变量获取数据
    data_feeds = _shared_data['data_feeds']
    std_factors = _shared_data['std_factors']
    timing_series = _shared_data['timing_series']
    etf_codes = _shared_data['etf_codes']
    rebalance_schedule = _shared_data['rebalance_schedule']
    
    factors = [f.strip() for f in combo_str.split(" + ")]
    dates = timing_series.index

    # 构造得分矩阵 (使用 DataFrame.add 保持 NaN 处理一致性)
    # ✅ 与 full_vec_bt_comparison.py 保持一致：fill_value=0 避免 NaN 传播
    combined_score_df = pd.DataFrame(0.0, index=dates, columns=etf_codes)
    for f in factors:
        combined_score_df = combined_score_df.add(std_factors[f], fill_value=0)

    # 运行回测
    bt_return, margin_failures = run_bt_backtest(
        combined_score_df, 
        timing_series, 
        etf_codes, 
        data_feeds,
        rebalance_schedule
    )
    
    return {
        "combo": combo_str,
        "bt_return": bt_return,
        "bt_margin_failures": margin_failures,
    }

def main():
    parser = argparse.ArgumentParser(description="批量 BT 回测 (支持 Top-K 筛选)")
    parser.add_argument("--topk", type=int, default=None, help="仅回测 VEC 收益最高的 Top-K 个组合")
    parser.add_argument("--sort-by", type=str, default="total_return", help="排序字段 (默认: total_return)")
    args = parser.parse_args()

    print("=" * 80)
    print("批量 BT 回测：多进程并行版 (Ryzen 9950X Optimized)")
    if args.topk:
        print(f"🎯 筛选模式: Top {args.topk} (按 {args.sort_by} 排序)")
    else:
        print("⚙️ 全量模式: 回测所有组合")
    print("=" * 80)

    # 1. 加载 WFO 结果
    wfo_dirs = sorted((ROOT / "results").glob("unified_wfo_*"))
    if not wfo_dirs:
        print("❌ 未找到 WFO 结果目录")
        return
    latest_wfo = wfo_dirs[-1]
    combos_path = latest_wfo / "all_combos.parquet"
    if not combos_path.exists():
        print(f"❌ 未找到 {combos_path}")
        return

    df_combos = pd.read_parquet(combos_path)
    print(f"✅ 加载 WFO 结果：{len(df_combos)} 个组合")

    # 筛选 Top-K
    if args.topk:
        if args.sort_by not in df_combos.columns:
            print(f"⚠️ 警告: 列 {args.sort_by} 不存在，无法排序。将使用原始顺序。")
        else:
            df_combos = df_combos.sort_values(args.sort_by, ascending=False).head(args.topk)
            print(f"✅ 已筛选 Top {len(df_combos)} 组合 (Min {args.sort_by}: {df_combos[args.sort_by].min():.4f})")

    # 2. 加载数据
    config_path = ROOT / "configs/combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )

    # 3. 计算因子
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv)

    factor_names_list = raw_factors_df.columns.get_level_values(0).unique().tolist()
    raw_factors = {fname: raw_factors_df[fname] for fname in factor_names_list}

    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors(raw_factors)

    factor_names = sorted(std_factors.keys())
    first_factor = std_factors[factor_names[0]]
    dates = first_factor.index
    etf_codes = first_factor.columns.tolist()

    timing_module = LightTimingModule()
    timing_series_raw = timing_module.compute_position_ratios(ohlcv["close"])
    # ✅ 使用 shift_timing_signal 做 t-1 shift，避免未来函数
    timing_arr_shifted = shift_timing_signal(timing_series_raw.reindex(dates).fillna(1.0).values)
    timing_series = pd.Series(timing_arr_shifted, index=dates)

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
    # Ryzen 9950X 有 16 核 32 线程。保留一点余量，使用 28-30 个进程。
    num_workers = 30
    print(f"🚀 启动多进程回测 (Workers: {num_workers})...")

    # 准备任务列表 (转换为 dict 列表以便传递)
    tasks = [row.to_dict() for _, row in df_combos.iterrows()]

    print(f"🚀 准备回测 {len(tasks)} 个组合...")

    results = []
    with mp.Pool(processes=num_workers, initializer=init_worker, initargs=(data_feeds, std_factors, timing_series, etf_codes)) as pool:
        # 使用 imap_unordered 获取实时进度
        for res in tqdm(pool.imap(process_combo, tasks), total=len(tasks), desc="BT 并行回测"):
            results.append(res)

    # 5. 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_top{args.topk}" if args.topk else "_full"
    output_dir = ROOT / "results" / f"bt_backtest{suffix}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    df_results = pd.DataFrame(results)
    df_results.to_parquet(output_dir / "bt_results.parquet", index=False)
    df_results.to_csv(output_dir / "bt_results.csv", index=False)

    print(f"\n✅ BT 批量回测完成")
    print(f"   输出目录: {output_dir}")
    print(f"   组合数: {len(df_results)}")
    print(f"   Margin 失败总数: {df_results['bt_margin_failures'].sum()}")



if __name__ == "__main__":
    main()
