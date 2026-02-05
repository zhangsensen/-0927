#!/usr/bin/env python3
"""
诊断 Holdout 期为何所有策略失效
"""

import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import shift_timing_signal

# Load Configuration
config_path = ROOT / "configs/combo_wfo_config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

# Load Data
loader = DataLoader(
    data_dir=config["data"].get("data_dir"),
    cache_dir=config["data"].get("cache_dir"),
)

# 加载训练集数据
train_ohlcv = loader.load_ohlcv(
    etf_codes=config["data"]["symbols"],
    start_date=config["data"]["start_date"],
    end_date=config["data"]["training_end_date"],
)

# 加载 Holdout 数据
holdout_ohlcv = loader.load_ohlcv(
    etf_codes=config["data"]["symbols"],
    start_date=config["data"]["training_end_date"],
    end_date=config["data"]["end_date"],
)

print("=" * 80)
print("🔍 HOLDOUT PERIOD DIAGNOSIS")
print("=" * 80)

# Compute Timing Signal
EXTREME_THRESHOLD = -0.1
EXTREME_POSITION = 0.1

timing_module = LightTimingModule(
    extreme_threshold=EXTREME_THRESHOLD,
    extreme_position=EXTREME_POSITION,
)

train_timing = timing_module.compute_position_ratios(train_ohlcv["close"])
holdout_timing = timing_module.compute_position_ratios(holdout_ohlcv["close"])

# Shift timing signal
train_timing_shifted = shift_timing_signal(train_timing.values)
holdout_timing_shifted = shift_timing_signal(holdout_timing.values)

print(
    f"\n训练集择时信号统计 ({config['data']['start_date']} 至 {config['data']['training_end_date']}):"
)
print(f"  平均仓位: {train_timing_shifted.mean():.2%}")
print(
    f"  满仓天数: {(train_timing_shifted >= 0.95).sum()} / {len(train_timing_shifted)} ({(train_timing_shifted >= 0.95).mean():.1%})"
)
print(
    f"  低仓位天数 (<50%): {(train_timing_shifted < 0.5).sum()} / {len(train_timing_shifted)} ({(train_timing_shifted < 0.5).mean():.1%})"
)
print(
    f"  极端低仓 (<15%): {(train_timing_shifted <= 0.15).sum()} / {len(train_timing_shifted)} ({(train_timing_shifted <= 0.15).mean():.1%})"
)

print(
    f"\nHoldout择时信号统计 ({config['data']['training_end_date']} 至 {config['data']['end_date']}):"
)
print(f"  平均仓位: {holdout_timing_shifted.mean():.2%}")
print(
    f"  满仓天数: {(holdout_timing_shifted >= 0.95).sum()} / {len(holdout_timing_shifted)} ({(holdout_timing_shifted >= 0.95).mean():.1%})"
)
print(
    f"  低仓位天数 (<50%): {(holdout_timing_shifted < 0.5).sum()} / {len(holdout_timing_shifted)} ({(holdout_timing_shifted < 0.5).mean():.1%})"
)
print(
    f"  极端低仓 (<15%): {(holdout_timing_shifted <= 0.15).sum()} / {len(holdout_timing_shifted)} ({(holdout_timing_shifted <= 0.15).mean():.1%})"
)

# Market Return Analysis
train_market_ret = train_ohlcv["close"].mean(axis=1).pct_change().dropna()
holdout_market_ret = holdout_ohlcv["close"].mean(axis=1).pct_change().dropna()

print(f"\n市场收益统计:")
print(f"  训练集累计收益: {(1 + train_market_ret).prod() - 1:.2%}")
print(f"  Holdout累计收益: {(1 + holdout_market_ret).prod() - 1:.2%}")
print(f"  训练集年化波动: {train_market_ret.std() * np.sqrt(252):.2%}")
print(f"  Holdout年化波动: {holdout_market_ret.std() * np.sqrt(252):.2%}")

# ETF Return Distribution
train_etf_rets = train_ohlcv["close"].iloc[-1] / train_ohlcv["close"].iloc[0] - 1
holdout_etf_rets = holdout_ohlcv["close"].iloc[-1] / holdout_ohlcv["close"].iloc[0] - 1

print(f"\nETF收益分布:")
print(f"  训练集中位数收益: {train_etf_rets.median():.2%}")
print(f"  Holdout中位数收益: {holdout_etf_rets.median():.2%}")
print(
    f"  训练集正收益ETF: {(train_etf_rets > 0).sum()} / {len(train_etf_rets)} ({(train_etf_rets > 0).mean():.1%})"
)
print(
    f"  Holdout正收益ETF: {(holdout_etf_rets > 0).sum()} / {len(holdout_etf_rets)} ({(holdout_etf_rets > 0).mean():.1%})"
)

# Top/Bottom performers
print(f"\nHoldout期表现最佳ETF:")
for code, ret in holdout_etf_rets.nlargest(5).items():
    print(f"  {code}: {ret:.2%}")

print(f"\nHoldout期表现最差ETF:")
for code, ret in holdout_etf_rets.nsmallest(5).items():
    print(f"  {code}: {ret:.2%}")

# Check if any ETF has missing data in Holdout
print(f"\nHoldout期数据完整性:")
for code in config["data"]["symbols"]:
    if code in holdout_ohlcv["close"].columns:
        missing = holdout_ohlcv["close"][code].isna().sum()
        if missing > 0:
            print(f"  {code}: {missing} 缺失值")
    else:
        print(f"  {code}: ❌ 完全缺失")

print("\n" + "=" * 80)
