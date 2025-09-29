#!/usr/bin/env python3
"""
存储概念股票筛选脚本
分析所有技术报告并筛选出前6只值得买入的股票
"""

import os
import re
import pandas as pd
from datetime import datetime
import json

# 股票评分权重定义
SCORE_WEIGHTS = {
    'recommendation': {
        '强烈买入': 10,
        '买入': 8,
        '持有': 5,
        '观望': 3,
        '卖出': 1,
        '强烈卖出': 0
    },
    'sharpe_ratio': 2.0,  # 夏普比率权重
    'total_return': 0.5,  # 总收益率权重
    'max_drawdown': -1.0,  # 最大回撤权重（负值）
    'volume_activity': 1.0,  # 成交量活跃度权重
    'rsi_position': 0.5,  # RSI位置权重
    'trend_strength': 1.0  # 趋势强度权重
}

def extract_score_from_report(report_file):
    """从技术分析报告中提取关键指标和评分"""

    with open(report_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取股票代码和名称
    stock_code = report_file.split('/')[-1].split('_')[0]

    # 提取关键指标
    metrics = {}

    # 提取建议
    recommendation_match = re.search(r'\*\*综合建议\*\*:\s*(.+)', content)
    if recommendation_match:
        metrics['recommendation'] = recommendation_match.group(1).strip()

    # 提取数值指标
    patterns = {
        'sharpe_ratio': r'\*\*夏普比率\*\*:\s*([-\d.]+)',
        'total_return': r'\*\*总收益率\*\*:\s*([-\d.]+)%',
        'max_drawdown': r'\*\*最大回撤\*\*:\s*([-\d.]+)%',
        'current_price': r'\*\*当前价格\*\*:\s*([\d.]+)元',
        'rsi': r'\*\*RSI指标\*\*:\s*([\d.]+)',
        'volatility': r'\*\*年化波动率\*\*:\s*([\d.]+)%'
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            metrics[key] = float(match.group(1))

    # 提取成交量活跃度
    volume_match = re.search(r'\*\*成交量活跃度\*\*:\s*(\w+)', content)
    if volume_match:
        volume_activity = volume_match.group(1)
        if volume_activity == '放量':
            metrics['volume_activity'] = 2.0
        elif volume_activity == '正常':
            metrics['volume_activity'] = 1.0
        else:
            metrics['volume_activity'] = 0.5

    # 提取趋势强度
    trend_match = re.search(r'\*\*趋势强度\*\*:\s*(\w+)', content)
    if trend_match:
        trend_strength = trend_match.group(1)
        if trend_strength == '强':
            metrics['trend_strength'] = 2.0
        elif trend_strength == '中等':
            metrics['trend_strength'] = 1.0
        else:
            metrics['trend_strength'] = 0.5

    # 计算RSI位置得分
    if 'rsi' in metrics:
        rsi = metrics['rsi']
        if 30 <= rsi <= 70:
            metrics['rsi_position'] = 1.5  # 正常区域
        elif rsi > 70:
            metrics['rsi_position'] = 1.0  # 超买但仍然强势
        else:
            metrics['rsi_position'] = 0.5  # 超卖

    return stock_code, metrics

def calculate_composite_score(metrics):
    """计算综合评分"""

    score = 0

    # 建议得分
    if 'recommendation' in metrics:
        score += SCORE_WEIGHTS['recommendation'].get(metrics['recommendation'], 0)

    # 夏普比率得分
    if 'sharpe_ratio' in metrics:
        score += metrics['sharpe_ratio'] * SCORE_WEIGHTS['sharpe_ratio']

    # 总收益率得分
    if 'total_return' in metrics:
        score += metrics['total_return'] * SCORE_WEIGHTS['total_return']

    # 最大回撤得分（负值，所以用绝对值）
    if 'max_drawdown' in metrics:
        score += abs(metrics['max_drawdown']) * SCORE_WEIGHTS['max_drawdown']

    # 成交量活跃度得分
    if 'volume_activity' in metrics:
        score += metrics['volume_activity'] * SCORE_WEIGHTS['volume_activity']

    # RSI位置得分
    if 'rsi_position' in metrics:
        score += metrics['rsi_position'] * SCORE_WEIGHTS['rsi_position']

    # 趋势强度得分
    if 'trend_strength' in metrics:
        score += metrics['trend_strength'] * SCORE_WEIGHTS['trend_strength']

    return score

def analyze_all_reports(report_dir):
    """分析所有报告并返回筛选结果"""

    stock_results = []

    # 遍历所有报告文件
    for filename in os.listdir(report_dir):
        if filename.endswith('_技术分析报告.md'):
            report_file = os.path.join(report_dir, filename)

            try:
                stock_code, metrics = extract_score_from_report(report_file)

                # 计算综合评分
                composite_score = calculate_composite_score(metrics)

                # 添加到结果列表
                stock_results.append({
                    'stock_code': stock_code,
                    'recommendation': metrics.get('recommendation', '未知'),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'total_return': metrics.get('total_return', 0),
                    'max_drawdown': metrics.get('max_drawdown', 0),
                    'current_price': metrics.get('current_price', 0),
                    'rsi': metrics.get('rsi', 0),
                    'volatility': metrics.get('volatility', 0),
                    'volume_activity': metrics.get('volume_activity', 0),
                    'trend_strength': metrics.get('trend_strength', 0),
                    'composite_score': composite_score
                })

            except Exception as e:
                print(f"❌ 分析文件 {filename} 时出错: {e}")

    # 转换为DataFrame并排序
    df = pd.DataFrame(stock_results)
    df = df.sort_values('composite_score', ascending=False)

    return df

def main():
    """主函数"""
    print("=" * 60)
    print("存储概念股票技术分析筛选")
    print("=" * 60)
    print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 分析报告目录
    report_dir = '/Users/zhangshenshen/深度量化0927/存储概念分析报告'

    # 分析所有报告
    print("🔍 正在分析技术报告...")
    results_df = analyze_all_reports(report_dir)

    # 显示结果
    print("\n" + "=" * 60)
    print("📊 综合评分排名")
    print("=" * 60)

    # 显示所有股票的评分
    for i, (_, row) in enumerate(results_df.iterrows(), 1):
        print(f"{i:2d}. {row['stock_code']:10s} | "
              f"评分: {row['composite_score']:6.2f} | "
              f"建议: {row['recommendation']:8s} | "
              f"收益率: {row['total_return']:7.2f}% | "
              f"夏普: {row['sharpe_ratio']:6.2f} | "
              f"价格: {row['current_price']:8.2f}元")

    # 筛选前6只值得买入的股票
    print("\n" + "=" * 60)
    print("🏆 前6只值得买入的股票（中短期）")
    print("=" * 60)

    # 过滤条件：建议为买入或强烈买入
    buy_candidates = results_df[
        results_df['recommendation'].isin(['强烈买入', '买入'])
    ].head(6)

    if len(buy_candidates) < 6:
        # 如果买入建议不足6只，补充持有建议中评分最高的
        hold_candidates = results_df[
            results_df['recommendation'] == '持有'
        ].head(6 - len(buy_candidates))
        buy_candidates = pd.concat([buy_candidates, hold_candidates])

    for i, (_, row) in enumerate(buy_candidates.iterrows(), 1):
        print(f"\n🥇 第{i}名: {row['stock_code']}")
        print(f"   💰 当前价格: {row['current_price']:.2f}元")
        print(f"   📊 综合评分: {row['composite_score']:.2f}")
        print(f"   🎯 投资建议: {row['recommendation']}")
        print(f"   📈 总收益率: {row['total_return']:.2f}%")
        print(f"   ⚡ 夏普比率: {row['sharpe_ratio']:.2f}")
        print(f"   📉 最大回撤: {row['max_drawdown']:.2f}%")
        print(f"   📊 年化波动率: {row['volatility']:.2f}%")
        print(f"   🎪 RSI指标: {row['rsi']:.1f}")

    # 保存筛选结果
    output_file = '/Users/zhangshenshen/深度量化0927/存储概念分析报告/筛选结果.md'

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"""# 存储概念股票技术分析筛选结果

**分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 分析概览
- **分析股票数量**: {len(results_df)}只
- **筛选方法**: 基于技术指标的综合评分系统
- **评分维度**: 投资建议、夏普比率、收益率、风险控制、成交量、趋势强度

## 🏆 前6只值得买入的股票（中短期）

""")

        for i, (_, row) in enumerate(buy_candidates.iterrows(), 1):
            f.write(f"""### {i}. {row['stock_code']}

- **💰 当前价格**: {row['current_price']:.2f}元
- **📊 综合评分**: {row['composite_score']:.2f}
- **🎯 投资建议**: {row['recommendation']}
- **📈 总收益率**: {row['total_return']:.2f}%
- **⚡ 夏普比率**: {row['sharpe_ratio']:.2f}
- **📉 最大回撤**: {row['max_drawdown']:.2f}%
- **📊 年化波动率**: {row['volatility']:.2f}%
- **🎪 RSI指标**: {row['rsi']:.1f}

""")

        f.write(f"""
## 📋 完整排名

| 排名 | 股票代码 | 综合评分 | 投资建议 | 总收益率 | 夏普比率 | 当前价格 |
|------|----------|----------|----------|----------|----------|----------|
""")

        for i, (_, row) in enumerate(results_df.iterrows(), 1):
            f.write(f"| {i} | {row['stock_code']} | {row['composite_score']:.2f} | {row['recommendation']} | {row['total_return']:.2f}% | {row['sharpe_ratio']:.2f} | {row['current_price']:.2f}元 |\n")

        f.write(f"""
## ⚠️ 风险提示
1. 技术分析仅供参考，不构成投资建议
2. 股票投资有风险，入市需谨慎
3. 建议结合基本面分析和风险管理
4. 关注市场整体趋势和行业政策变化

**分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")

    print(f"\n📄 详细筛选报告已保存至: {output_file}")
    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()