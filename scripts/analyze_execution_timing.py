#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
执行时机分析脚本

分析 14:50 执行 vs 次日开盘执行的价格偏差，验证哪种执行方式更接近回测假设。

核心问题：
- 回测假设以日收盘价成交
- 实盘无法在收盘后交易
- 必须选择: 14:50 执行 或 次日开盘执行

使用方法:
    uv run python scripts/analyze_execution_timing.py

依赖:
    - akshare (已添加到 pyproject.toml)
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import akshare as ak
except ImportError:
    print("❌ 请先安装 akshare: uv add akshare")
    exit(1)


# 43 ETF 池中的部分代表性 ETF
ETF_LIST = [
    # QDII (高跳空风险)
    ("513100", "纳指100ETF", "QDII"),
    ("513500", "标普500ETF", "QDII"),
    ("159920", "恒生ETF", "QDII"),
    ("513050", "中概互联", "QDII"),
    ("513130", "恒生科技", "QDII"),
    # A股宽基
    ("510300", "沪深300ETF", "宽基"),
    ("510500", "中证500ETF", "宽基"),
    ("510050", "上证50ETF", "宽基"),
    ("159915", "创业板ETF", "宽基"),
    ("512100", "中证1000ETF", "宽基"),
    # 行业ETF
    ("512880", "证券ETF", "行业"),
    ("515790", "光伏ETF", "行业"),
    ("512200", "房地产ETF", "行业"),
    ("512010", "医药ETF", "行业"),
    ("512480", "半导体ETF", "行业"),
]


def download_etf_min_data(symbol: str, period: str = "5") -> pd.DataFrame:
    """
    下载 ETF 分钟数据
    
    Args:
        symbol: ETF 代码 (如 "513100")
        period: 周期 ("1", "5", "15", "30", "60")
    
    Returns:
        DataFrame with columns: 时间, 开盘, 收盘, 最高, 最低, 成交量, ...
    """
    try:
        df = ak.fund_etf_hist_min_em(
            symbol=symbol,
            period=period,
            adjust=""  # 不复权，使用实际价格
        )
        if len(df) > 0:
            df['时间'] = pd.to_datetime(df['时间'])
        return df
    except Exception as e:
        print(f"  ⚠️ 下载 {symbol} 失败: {e}")
        return pd.DataFrame()


def analyze_execution_timing(etf_list: list = None) -> pd.DataFrame:
    """
    分析执行时机
    
    Returns:
        DataFrame with deviation analysis
    """
    if etf_list is None:
        etf_list = ETF_LIST
    
    all_results = []
    
    print("=" * 80)
    print("🔬 执行时机验证分析")
    print("=" * 80)
    
    for symbol, name, category in etf_list:
        print(f"\n📥 下载 {symbol} ({name})...")
        
        df = download_etf_min_data(symbol, period="5")
        
        if len(df) == 0:
            continue
        
        df['date'] = df['时间'].dt.date
        df['time'] = df['时间'].dt.time
        
        print(f"  ✅ 获取 {len(df)} 条数据, 日期范围: {df['date'].min()} ~ {df['date'].max()}")
        
        # 按日期分组分析
        for date, day_data in df.groupby('date'):
            day_data = day_data.sort_values('时间')
            
            # 关键时间点的价格
            close_bar = day_data[day_data['time'] == time(14, 55)]  # 收盘 K 线
            bar_1450 = day_data[day_data['time'] == time(14, 50)]   # 14:50
            bar_1445 = day_data[day_data['time'] == time(14, 45)]   # 14:45
            bar_1440 = day_data[day_data['time'] == time(14, 40)]   # 14:40
            open_bar = day_data[day_data['time'] == time(9, 30)]    # 开盘
            
            if len(close_bar) > 0 and len(bar_1450) > 0 and len(open_bar) > 0:
                all_results.append({
                    'symbol': symbol,
                    'name': name,
                    'category': category,
                    'date': date,
                    'open_price': open_bar['开盘'].values[0],
                    'price_1440': bar_1440['收盘'].values[0] if len(bar_1440) > 0 else np.nan,
                    'price_1445': bar_1445['收盘'].values[0] if len(bar_1445) > 0 else np.nan,
                    'price_1450': bar_1450['收盘'].values[0],
                    'close_price': close_bar['收盘'].values[0],
                })
    
    if len(all_results) == 0:
        print("\n❌ 没有获取到数据，请检查网络连接")
        return pd.DataFrame()
    
    # 转换为 DataFrame
    result_df = pd.DataFrame(all_results)
    print(f"\n📊 共收集 {len(result_df)} 天 × ETF 的数据")
    
    # 计算偏差 (百分比)
    result_df['dev_1440'] = (result_df['price_1440'] / result_df['close_price'] - 1) * 100
    result_df['dev_1445'] = (result_df['price_1445'] / result_df['close_price'] - 1) * 100
    result_df['dev_1450'] = (result_df['price_1450'] / result_df['close_price'] - 1) * 100
    
    # 计算次日开盘与前日收盘的跳空
    result_df = result_df.sort_values(['symbol', 'date'])
    result_df['prev_close'] = result_df.groupby('symbol')['close_price'].shift(1)
    result_df['gap_open'] = (result_df['open_price'] / result_df['prev_close'] - 1) * 100
    
    return result_df


def print_analysis_report(result_df: pd.DataFrame):
    """打印分析报告"""
    
    if len(result_df) == 0:
        return
    
    print("\n" + "=" * 80)
    print("📈 执行价格偏差分析 (相对于收盘价)")
    print("=" * 80)
    
    # 按类别汇总
    for category in result_df['category'].unique():
        cat_data = result_df[result_df['category'] == category]
        
        print(f"\n【{category}】({len(cat_data['symbol'].unique())} 只 ETF)")
        print("-" * 60)
        
        # 14:50 执行偏差
        dev_1450 = cat_data['dev_1450'].dropna()
        print(f"  14:50 执行 vs 收盘价:")
        print(f"    均值偏差:     {dev_1450.mean():+.4f}%")
        print(f"    标准差:       {dev_1450.std():.4f}%")
        print(f"    最大偏差:     {dev_1450.abs().max():.4f}%")
        print(f"    95%置信区间:  ±{dev_1450.std() * 1.96:.4f}%")
        
        # 次日开盘跳空
        gap = cat_data['gap_open'].dropna()
        if len(gap) > 0:
            print(f"\n  次日开盘 vs 前日收盘:")
            print(f"    均值跳空:     {gap.mean():+.4f}%")
            print(f"    标准差:       {gap.std():.4f}%")
            print(f"    最大跳空:     {gap.abs().max():.4f}%")
            print(f"    95%置信区间:  ±{gap.std() * 1.96:.4f}%")
    
    # 整体统计
    print("\n" + "=" * 80)
    print("🎯 整体统计")
    print("=" * 80)
    
    dev_all = result_df['dev_1450'].dropna()
    gap_all = result_df['gap_open'].dropna()
    
    print(f"\n【14:50 执行】(全部 ETF)")
    print(f"  样本量:       {len(dev_all)}")
    print(f"  平均偏差:     {dev_all.mean():+.5f}%")
    print(f"  标准差:       {dev_all.std():.5f}%")
    print(f"  95% 置信区间: ±{dev_all.std() * 1.96:.5f}%")
    
    print(f"\n【次日开盘执行】(全部 ETF)")
    print(f"  样本量:       {len(gap_all)}")
    print(f"  平均跳空:     {gap_all.mean():+.5f}%")
    print(f"  标准差:       {gap_all.std():.5f}%")
    print(f"  95% 置信区间: ±{gap_all.std() * 1.96:.5f}%")
    
    # QDII 单独分析
    print("\n" + "=" * 80)
    print("⚠️ QDII 特别分析 (跨境 ETF 隔夜风险)")
    print("=" * 80)
    
    qdii_data = result_df[result_df['category'] == 'QDII']
    if len(qdii_data) > 0:
        qdii_gap = qdii_data['gap_open'].dropna()
        non_qdii_gap = result_df[result_df['category'] != 'QDII']['gap_open'].dropna()
        
        print(f"\nQDII 次日开盘跳空:")
        print(f"  标准差: {qdii_gap.std():.4f}%")
        print(f"  最大跳空: {qdii_gap.abs().max():.4f}%")
        
        print(f"\n非 QDII 次日开盘跳空:")
        print(f"  标准差: {non_qdii_gap.std():.4f}%")
        print(f"  最大跳空: {non_qdii_gap.abs().max():.4f}%")
        
        print(f"\n💡 QDII 隔夜风险是非 QDII 的 {qdii_gap.std() / non_qdii_gap.std():.1f} 倍！")
    
    # 结论
    print("\n" + "=" * 80)
    print("💡 结论与建议")
    print("=" * 80)
    
    std_1450 = dev_all.std()
    std_gap = gap_all.std()
    
    print(f"""
    ┌────────────────────────────────────────────────────────────────┐
    │ 执行方式        │ 偏差标准差  │ 95% 置信区间  │ 推荐度      │
    ├────────────────────────────────────────────────────────────────┤
    │ 14:50 执行      │ {std_1450:.4f}%    │ ±{std_1450*1.96:.4f}%      │ {"⭐⭐⭐⭐⭐" if std_1450 < std_gap else "⭐⭐⭐"}        │
    │ 次日开盘执行    │ {std_gap:.4f}%    │ ±{std_gap*1.96:.4f}%      │ {"⭐⭐⭐⭐⭐" if std_gap < std_1450 else "⭐⭐⭐"}        │
    └────────────────────────────────────────────────────────────────┘
    """)
    
    if std_1450 < std_gap:
        print("""
    ✅ 推荐【14:50 执行】

    执行建议:
    1. 每天 14:30 运行信号生成脚本，获取预计调仓
    2. 14:45 确认信号，检查 QDII 溢价率
    3. 14:50-14:55 使用限价单执行
    4. 15:00 后核对，记录实际成交价

    限价单策略:
    - 买入: 挂 卖一价 或 现价 + 1 分
    - 卖出: 挂 买一价 或 现价 - 1 分
    - 5 分钟未成交则撤单，次日开盘补单
        """)
    else:
        print("""
    ✅ 推荐【次日开盘执行】

    执行建议:
    1. 每天 15:00 后运行信号生成脚本
    2. 生成次日调仓计划
    3. 次日 09:25-09:30 集合竞价下单
    4. 使用限价单，挂昨日收盘价附近
        """)


def main():
    """主函数"""
    # 分析执行时机
    result_df = analyze_execution_timing()
    
    if len(result_df) == 0:
        return
    
    # 打印分析报告
    print_analysis_report(result_df)
    
    # 保存详细数据
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"execution_timing_analysis_{datetime.now().strftime('%Y%m%d')}.csv"
    result_df.to_csv(output_file, index=False)
    print(f"\n📁 详细数据已保存: {output_file}")
    
    # 生成建议配置
    config_suggestion = """
# 执行配置建议 (基于数据分析)
execution:
  timing: "14:50"           # 执行时间
  order_type: "limit"       # 限价单
  price_offset_buy: 0.001   # 买入挂卖一 + 0.1%
  price_offset_sell: -0.001 # 卖出挂买一 - 0.1%
  timeout_seconds: 300      # 5 分钟超时
  qdii_premium_limit: 0.02  # QDII 溢价上限 2%
    """
    print(f"\n📋 建议执行配置:{config_suggestion}")


if __name__ == "__main__":
    main()
