#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速因子测试 - 使用真实ETF数据
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from factor_system.factor_engine import api

# 测试参数
symbols = ['510300.SH', '510500.SH', '159915.SZ']  # 3只主流ETF
end_date = datetime(2025, 10, 14)
start_date = end_date - timedelta(days=60)

print("="*80)
print("快速因子测试")
print("="*80)
print(f"标的: {symbols}")
print(f"时间范围: {start_date.date()} ~ {end_date.date()}")

# 测试几个常用因子
test_factors = [
    'RSI14',
    'MACD_12_26_9',
    'ATR14',
    'BB_20_2.0_Width',
    'CCI14'
]

print(f"\n测试因子: {test_factors}")
print("="*80)

for factor_id in test_factors:
    print(f"\n测试因子: {factor_id}")
    try:
        result = api.calculate_factors(
            factor_ids=[factor_id],
            symbols=symbols,
            timeframe='daily',
            start_date=start_date,
            end_date=end_date
        )
        
        if result is not None and not result.empty:
            print(f"✅ 成功: {result.shape}")
            print(f"   索引类型: {type(result.index)}")
            if hasattr(result.index, 'names'):
                print(f"   索引名称: {result.index.names}")
            print(f"   列名: {list(result.columns)[:5]}")
            print(f"   前3行:")
            print(result.head(3))
        else:
            print(f"❌ 失败: 返回空结果")
            
    except Exception as e:
        print(f"❌ 失败: {str(e)}")

print("\n" + "="*80)
print("测试完成")
