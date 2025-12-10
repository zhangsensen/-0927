#!/usr/bin/env python3
"""
旗舰策略 BT 审计脚本
针对 v3.2 旗舰策略进行严格的 Backtrader 审计
"""
import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime
import argparse

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import shift_timing_signal, generate_rebalance_schedule
from etf_strategy.auditor.core.engine import GenericStrategy, PandasData
import etf_strategy.auditor.core.engine as engine_module

# 默认旗舰策略
DEFAULT_COMBO = "ADX_14D + CMF_20D + OBV_SLOPE_10D + VOL_RATIO_60D + VORTEX_14D"
TRAINING_END = "2025-05-31"
HOLDOUT_START = "2025-06-01"
HOLDOUT_END = "2025-12-08"

def run_bt_backtest(combined_score_df, timing_series, etf_codes, data_feeds, rebalance_schedule,
                    freq, pos_size, initial_capital, commission_rate,
                    target_vol=0.20, vol_window=20, dynamic_leverage_enabled=True, lookback=252, cheat_on_close=False):
    """单组合 BT 回测引擎"""
    # 动态修改 LOOKBACK
    engine_module.LOOKBACK = lookback
    
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_capital)
    cerebro.broker.setcommission(commission=commission_rate, leverage=1.0)
    cerebro.broker.set_coc(cheat_on_close)
    cerebro.broker.set_checksubmit(False)

    for ticker, df in data_feeds.items():
        data = PandasData(dataname=df, name=ticker)
        cerebro.adddata(data)

    cerebro.addstrategy(
        GenericStrategy, 
        scores=combined_score_df, 
        timing=timing_series, 
        etf_codes=etf_codes, 
        freq=freq, 
        pos_size=pos_size,
        rebalance_schedule=rebalance_schedule,
        target_vol=target_vol,
        vol_window=vol_window,
        dynamic_leverage_enabled=dynamic_leverage_enabled,
    )
    
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', 
                       timeframe=bt.TimeFrame.Days, compression=1,
                       riskfreerate=0.0, annualize=True)
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    start_val = cerebro.broker.getvalue()
    results = cerebro.run()
    end_val = cerebro.broker.getvalue()
    strat = results[0]

    bt_return = (end_val / start_val) - 1
    
    dd_analysis = strat.analyzers.drawdown.get_analysis()
    max_drawdown = dd_analysis.get('max', {}).get('drawdown', 0.0) / 100.0
    
    sharpe_analysis = strat.analyzers.sharpe.get_analysis()
    sharpe_ratio = sharpe_analysis.get('sharperatio', 0.0)
    if sharpe_ratio is None: sharpe_ratio = 0.0
    
    return {
        "return": bt_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "end_value": end_val
    }

def main():
    parser = argparse.ArgumentParser(description='Audit a strategy using Backtrader')
    parser.add_argument('--combo', type=str, default=DEFAULT_COMBO, help='Strategy combination string')
    args = parser.parse_args()
    
    TARGET_COMBO = args.combo
    
    print('='*80)
    print('🛡️  旗舰策略 BT 审计')
    print('='*80)
    print(f'策略: {TARGET_COMBO}')

    # 1. 加载配置
    config_path = ROOT / 'configs/combo_wfo_config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # 2. 加载数据
    print("\n📂 加载数据...")
    data_dir = Path(config["data"].get("data_dir"))
    loader = DataLoader(
        data_dir=data_dir,
        cache_dir=ROOT / '.cache',
    )
    
    etf_files = list(data_dir.glob("*.parquet"))
    etf_codes = [f.stem.split('_')[0].split('.')[0] for f in etf_files]
    etf_codes.sort() # 显式排序，确保与 DataLoader 一致
    
    ohlcv_full = loader.load_ohlcv(etf_codes=etf_codes, start_date='2020-01-01', end_date=HOLDOUT_END)
    
    # 兼容旧变量名
    ohlcv = ohlcv_full
    dates = ohlcv['close'].index
    
    # 3. 计算因子
    print("🔧 计算因子...")
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv)
    
    factor_names_list = raw_factors_df.columns.get_level_values(0).unique().tolist()
    raw_factors = {fname: raw_factors_df[fname] for fname in factor_names_list}
    
    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors(raw_factors)
    
    # 4. 准备回测数据
    factors = [f.strip() for f in TARGET_COMBO.split(" + ")]
    dates = std_factors[factors[0]].index
    etf_codes = std_factors[factors[0]].columns.tolist()
    
    # 组合得分
    combined_score_df = pd.DataFrame(0.0, index=dates, columns=etf_codes)
    for f in factors:
        # ⚠️ 关键修改：不使用 fill_value=0，保持 NaN 传播
        # 与 VEC 逻辑对齐：只要有一个因子是 NaN，总分就是 NaN
        combined_score_df = combined_score_df.add(std_factors[f])
        
    # 择时信号
    timing_module = LightTimingModule(extreme_threshold=-0.1, extreme_position=0.1)
    timing_series_raw = timing_module.compute_position_ratios(ohlcv["close"])
    timing_arr_shifted = shift_timing_signal(timing_series_raw.reindex(dates).fillna(1.0).values)
    timing_series = pd.Series(timing_arr_shifted, index=dates)
    
    # Data Feeds
    data_feeds = {}
    for ticker in etf_codes:
        df = pd.DataFrame({
            "open": ohlcv["open"][ticker],
            "high": ohlcv["high"][ticker],
            "low": ohlcv["low"][ticker],
            "close": ohlcv["close"][ticker],
            "volume": ohlcv["volume"][ticker],
        }).reindex(dates).ffill().fillna(0.01)
        data_feeds[ticker] = df
        
    # 5. 运行回测
    print("\n🚀 开始 BT 回测...")
    
    # 开启 Cheat-On-Close 以对齐 VEC (Signal t-1 -> Trade t Close)
    cerebro = bt.Cerebro()
    cerebro.broker.set_coc(True)
    
    # 参数
    freq = 3
    pos_size = 2
    initial_capital = 1_000_000.0
    commission_rate = 0.0002

    # 5.1 训练集
    print("   > 运行训练集 (2020-01-01 ~ 2025-05-31)...")
    train_mask = dates <= TRAINING_END
    train_dates = dates[train_mask]
    
    # 使用全量调仓日程，并截取训练集部分
    full_rebalance_idx = generate_rebalance_schedule(len(dates), 252, freq)
    full_rebalance_dates = dates[full_rebalance_idx]
    train_rebalance_dates = full_rebalance_dates[full_rebalance_dates <= TRAINING_END]
    
    train_feeds = {k: v.loc[train_dates] for k, v in data_feeds.items()}
    
    # 关键修改: Shift 1天，模拟在 t 时刻使用 t-1 信号
    train_scores = combined_score_df.shift(1).loc[train_dates]
    train_timing = timing_series.shift(1).loc[train_dates]

    res_train = run_bt_backtest(
        train_scores, train_timing, etf_codes, train_feeds, train_rebalance_dates,
        freq, pos_size, initial_capital, commission_rate,
        dynamic_leverage_enabled=False,
        lookback=252,
        cheat_on_close=True
    )
    
    # 5.2 Holdout集
    print("   > 运行 Holdout 集 (2025-06-01 ~ 2025-12-08)...")
    holdout_mask = (dates >= HOLDOUT_START) & (dates <= HOLDOUT_END)
    holdout_dates = dates[holdout_mask]
    
    # 使用全量调仓日程，并截取Holdout部分
    holdout_rebalance_dates = full_rebalance_dates[(full_rebalance_dates >= HOLDOUT_START) & (full_rebalance_dates <= HOLDOUT_END)]
    
    holdout_feeds = {k: v.loc[holdout_dates] for k, v in data_feeds.items()}
    
    # 同样 Shift 1天
    holdout_scores = combined_score_df.shift(1).loc[holdout_dates]
    holdout_timing = timing_series.shift(1).loc[holdout_dates]
    
    res_holdout = run_bt_backtest(
        holdout_scores, holdout_timing, etf_codes, holdout_feeds, holdout_rebalance_dates,
        freq, pos_size, initial_capital, commission_rate,
        dynamic_leverage_enabled=False,
        lookback=0,
        cheat_on_close=True
    )
    
    # 6. 输出报告
    print("\n" + "="*80)
    print("📊 BT 审计报告")
    print("="*80)
    print(f"策略: {TARGET_COMBO}")
    print("-" * 40)
    print(f"【训练集】")
    print(f"  收益率: {res_train['return']*100:.2f}%")
    print(f"  Sharpe: {res_train['sharpe_ratio']:.3f}")
    print(f"  最大回撤: {res_train['max_drawdown']*100:.2f}%")
    print("-" * 40)
    print(f"【Holdout集】")
    print(f"  收益率: {res_holdout['return']*100:.2f}%")
    print(f"  Sharpe: {res_holdout['sharpe_ratio']:.3f}")
    print(f"  最大回撤: {res_holdout['max_drawdown']*100:.2f}%")
    print("-" * 40)
    
    # VEC 对比 (硬编码 VEC 结果以便对比)
    vec_train_ret = 35.00
    vec_holdout_ret = 14.39
    
    print(f"【VEC vs BT 差异】")
    print(f"  训练集收益差异: {res_train['return']*100 - vec_train_ret:+.2f}pp")
    print(f"  Holdout收益差异: {res_holdout['return']*100 - vec_holdout_ret:+.2f}pp")
    
    if abs(res_holdout['return']*100 - vec_holdout_ret) < 1.0:
        print("\n✅ BT 审计通过！结果高度一致。")
    else:
        print("\n⚠️  注意：VEC与BT存在一定差异，请检查。")

if __name__ == "__main__":
    main()
