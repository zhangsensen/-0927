#!/usr/bin/env python3
"""
验证T+1时序安全 - 确保只滞后1天
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from factor_system.factor_engine.providers.parquet_provider import ParquetDataProvider
from factor_system.factor_engine.providers.combined_provider import CombinedMoneyFlowProvider


def main():
    print("=" * 70)
    print("🔍 T+1时序安全验证")
    print("=" * 70)
    
    symbol = "600036.SH"
    
    # 1. 读取原始资金流文件
    print("\n1️⃣ 读取原始资金流文件...")
    raw_mf_file = Path("raw/SH/money_flow/600036.SH_moneyflow.parquet")
    raw_mf = pd.read_parquet(raw_mf_file)
    raw_mf['trade_date'] = pd.to_datetime(raw_mf['trade_date'])
    raw_mf = raw_mf.set_index('trade_date').sort_index()
    
    print(f"   原始数据: {raw_mf.shape}")
    print(f"   前3天 main_net:")
    # 计算main_net
    main_net_raw = (
        raw_mf['buy_lg_amount'] + raw_mf['buy_elg_amount'] -
        raw_mf['sell_lg_amount'] - raw_mf['sell_elg_amount']
    ) * 10000  # 转换为元
    
    for date, value in main_net_raw.head(3).items():
        print(f"     {date.date()}: {value:.2f}")
    
    # 2. 通过CombinedProvider加载
    print("\n2️⃣ 通过CombinedProvider加载...")
    price_provider = ParquetDataProvider(Path("raw"))
    combined = CombinedMoneyFlowProvider(
        price_provider=price_provider,
        money_flow_dir=Path("raw/SH/money_flow"),
        enforce_t_plus_1=True
    )
    
    data = combined.load_price_data(
        [symbol],
        'daily',
        datetime(2024, 8, 20),
        datetime(2024, 9, 10)
    )
    
    print(f"   合并数据: {data.shape}")
    
    # 3. 提取合并后的main_net
    symbol_data = data.xs(symbol, level='symbol')
    print(f"\n   合并后前5天 main_net:")
    for idx, value in symbol_data['main_net'].head(5).items():
        print(f"     {idx.date()}: {value if pd.notna(value) else 'NaN'}")
    
    # 4. 验证滞后
    print("\n3️⃣ 验证T+1滞后...")
    
    # 找到第一个非NaN值
    first_valid_idx = symbol_data['main_net'].first_valid_index()
    if first_valid_idx:
        first_valid_date = first_valid_idx
        first_valid_value = symbol_data.loc[first_valid_date, 'main_net']
        
        # 在原始数据中找前一天
        prev_date = first_valid_date - pd.Timedelta(days=1)
        
        # 向前查找最近的交易日
        while prev_date not in main_net_raw.index and prev_date >= main_net_raw.index.min():
            prev_date -= pd.Timedelta(days=1)
        
        if prev_date in main_net_raw.index:
            raw_value = main_net_raw.loc[prev_date]
            
            print(f"   合并数据第一个有效值:")
            print(f"     日期: {first_valid_date.date()}")
            print(f"     值: {first_valid_value:.2f}")
            print(f"\n   原始数据前一交易日:")
            print(f"     日期: {prev_date.date()}")
            print(f"     值: {raw_value:.2f}")
            print(f"\n   差异: {abs(first_valid_value - raw_value):.2f}")
            
            if abs(first_valid_value - raw_value) < 1.0:
                print("\n   ✅ T+1滞后正确：合并数据T日 = 原始数据T-1日")
            else:
                print("\n   ❌ T+1滞后异常：数值不匹配")
        else:
            print(f"   ⚠️ 无法找到原始数据中的 {prev_date.date()}")
    
    # 5. 检查是否有双重滞后
    print("\n4️⃣ 检查双重滞后...")
    
    # 如果是双重滞后，合并数据T日应该等于原始数据T-2日
    if first_valid_idx:
        two_days_before = first_valid_date - pd.Timedelta(days=2)
        while two_days_before not in main_net_raw.index and two_days_before >= main_net_raw.index.min():
            two_days_before -= pd.Timedelta(days=1)
        
        if two_days_before in main_net_raw.index:
            raw_value_t2 = main_net_raw.loc[two_days_before]
            
            if abs(first_valid_value - raw_value_t2) < 1.0:
                print(f"   ❌ 检测到双重滞后！合并数据T日 = 原始数据T-2日")
                print(f"      {first_valid_date.date()} = {two_days_before.date()}")
            else:
                print(f"   ✅ 无双重滞后")
    
    print("\n" + "=" * 70)
    print("✅ 验证完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
