#!/usr/bin/env python3
"""检查因子系统统计"""

import sys
sys.path.append('/Users/zhangshenshen/深度量化0927')

from factor_system.factor_engine import api

def main():
    # 获取FactorEngine的所有因子
    factors = api.list_available_factors()
    print(f'FactorEngine总因子数: {len(factors)}')

    # 按类别统计
    categories = {}
    for f in factors:
        cat = f.split('_')[0]
        categories[cat] = categories.get(cat, 0) + 1

    print('\n因子分类统计:')
    for cat, count in sorted(categories.items()):
        print(f'{cat}: {count}个')

    # 显示具体因子列表
    print(f'\n所有因子列表:')
    for i, factor in enumerate(factors, 1):
        print(f'{i:3d}. {factor}')

if __name__ == "__main__":
    main()