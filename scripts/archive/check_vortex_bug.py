"""
直接检查因子库的 VORTEX 计算 Bug
"""
import os
import sys
import numpy as np
import pandas as pd
import yaml

sys.path.append(os.getcwd())

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary

def check_vortex_bug():
    print("="*80)
    print("🐛 检查 VORTEX_14D 因子计算 Bug")
    print("="*80)
    
    # Load Data
    with open("configs/combo_wfo_config.yaml") as f:
        config = yaml.safe_load(f)
    
    loader = DataLoader(
        data_dir=config["data"]["data_dir"],
        cache_dir=config["data"]["cache_dir"]
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
        use_cache=True
    )
    
    high_df = ohlcv["high"]
    low_df = ohlcv["low"]
    close_df = ohlcv["close"]
    
    # 库的计算方式 (有 bug)
    vm_plus = (high_df - low_df.shift(1)).abs()
    vm_minus = (low_df - high_df.shift(1)).abs()

    prev_close = close_df.shift(1)
    tr1 = high_df - low_df
    tr2 = (high_df - prev_close).abs()
    tr3 = (low_df - prev_close).abs()
    
    # Bug: pd.concat 会把多个 DataFrame 横向拼接，导致 max(axis=1) 的结果是 Series
    tr_bug = (
        pd.concat([tr1, tr2, tr3], axis=1)
        .max(axis=1)
        .to_frame()
        .reindex(columns=close_df.columns, fill_value=0)
    )
    
    print(f"\n[1] Bug 版本 TR 检查:")
    print(f"   tr1 shape: {tr1.shape}")
    print(f"   pd.concat([tr1, tr2, tr3], axis=1) shape: {pd.concat([tr1, tr2, tr3], axis=1).shape}")
    print(f"   .max(axis=1) shape: {pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).shape}")
    print(f"   tr_bug shape: {tr_bug.shape}")
    print(f"   tr_bug 第一列样本: {tr_bug.iloc[20:25, 0].tolist()}")
    print(f"   tr_bug 是否全 0: {(tr_bug == 0).all().all()}")
    
    # 正确的计算方式 (逐列)
    print(f"\n[2] 正确版本 TR 检查 (逐列计算):")
    tr_correct = pd.DataFrame(index=close_df.index, columns=close_df.columns, dtype=float)
    for col in close_df.columns:
        tr1_col = high_df[col] - low_df[col]
        tr2_col = (high_df[col] - close_df[col].shift(1)).abs()
        tr3_col = (low_df[col] - close_df[col].shift(1)).abs()
        tr_correct[col] = pd.concat([tr1_col, tr2_col, tr3_col], axis=1).max(axis=1)
    
    print(f"   tr_correct shape: {tr_correct.shape}")
    print(f"   tr_correct 第一列样本: {tr_correct.iloc[20:25, 0].tolist()}")
    
    # 计算差异
    diff = (tr_bug - tr_correct).abs()
    print(f"\n[3] TR 差异:")
    print(f"   最大差异: {diff.max().max()}")
    print(f"   平均差异: {diff.mean().mean()}")
    
    # 正确计算 Vortex
    print(f"\n[4] 正确计算 VORTEX_14D...")
    vm_plus_sum = vm_plus.rolling(window=14, min_periods=14).sum()
    vm_minus_sum = vm_minus.rolling(window=14, min_periods=14).sum()
    tr_sum_correct = tr_correct.rolling(window=14, min_periods=14).sum()
    
    vi_plus = vm_plus_sum / (tr_sum_correct + 1e-10)
    vi_minus = vm_minus_sum / (tr_sum_correct + 1e-10)
    vortex_correct = vi_plus - vi_minus
    
    # 与库的结果对比
    lib = PreciseFactorLibrary()
    factors_df = lib.compute_all_factors(ohlcv)
    vortex_lib = factors_df["VORTEX_14D"]
    
    vortex_diff = (vortex_correct - vortex_lib).abs()
    print(f"   VORTEX 库 vs 正确计算 最大差异: {vortex_diff.max().max()}")
    
    # 结论
    print(f"\n🔍 结论:")
    if (tr_bug == 0).all().all():
        print("   ❌ 因子库的 _vortex_14d_batch 存在 Bug！")
        print("   原因: pd.concat([tr1,tr2,tr3], axis=1).max(axis=1) 返回 Series")
        print("         .to_frame().reindex(columns=..., fill_value=0) 导致所有列都是 0")
        print("         这意味着 tr_sum = 0，导致 vi_plus 和 vi_minus 爆炸或异常")
    else:
        print("   ✅ TR 计算正常")

if __name__ == "__main__":
    check_vortex_bug()
