#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
资金流向数据下载器 - 修复版本
下载全市场资金流向数据，并基于ETF交易数据生成资金流向估算
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import tushare as ts

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    # 正确的Token
    TUSHARE_TOKEN = "4a24bcfff16f7593632e6c46976a83e6a26f8f565daa156cb9ea9c1f"

    # 计算日期范围（最近一年）
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    # 使用正确的日期格式
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    start_date_tushare = start_date.strftime("%Y%m%d")
    end_date_tushare = end_date.strftime("%Y%m%d")

    print("=" * 80)
    print("资金流向数据下载器 - 修复版本")
    print("=" * 80)
    print(f"下载时间范围: {start_date_str} ~ {end_date_str}")
    print(f"下载天数: {(end_date - start_date).days} 天")
    print()

    # 初始化Tushare
    pro = ts.pro_api(TUSHARE_TOKEN)

    # 创建数据目录
    data_dir = Path("raw/ETF")
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "moneyflow").mkdir(exist_ok=True)
    (data_dir / "moneyflow_market").mkdir(exist_ok=True)

    # ETF代码列表
    etf_codes = [
        "510300.SH",
        "588000.SH",
        "512480.SH",
        "515790.SH",
        "515030.SH",
        "512010.SH",
        "515210.SH",
        "159998.SZ",
        "159915.SZ",
        "510500.SH",
        "512100.SH",
        "512660.SH",
        "512690.SH",
        "512880.SH",
        "518880.SH",
        "159992.SZ",
        "159819.SZ",
        "516160.SH",
        "159883.SZ",
    ]

    print("=== 下载策略 ===")
    print("1. 下载全市场资金流向数据")
    print("2. 获取沪深港通资金流向数据")
    print("3. 基于ETF交易数据生成资金流向估算")
    print()

    # 统计信息
    stats = {
        "market_data": 0,
        "hsgt_data": 0,
        "estimated_data": 0,
        "start_time": datetime.now(),
    }

    # 1. 下载全市场资金流向数据（分批下载）
    print("1. 下载全市场资金流向数据...")
    print("-" * 50)

    try:
        market_data = []
        date_list = pd.date_range(start=start_date, end=end_date, freq="B")

        # 分批下载，每次下载一个月的数据
        batch_size = 22  # 大约一个月的交易日
        for i in range(0, len(date_list), batch_size):
            batch_dates = date_list[i : i + batch_size]
            print(
                f"  下载第 {i//batch_size + 1} 批数据 ({len(batch_dates)} 个交易日)..."
            )

            for date in batch_dates:
                date_str = date.strftime("%Y%m%d")
                try:
                    df = pro.moneyflow(trade_date=date_str)
                    if not df.empty:
                        df["trade_date"] = pd.to_datetime(df["trade_date"])
                        market_data.append(df)
                        print(f"    ✅ {date_str}: {len(df)} 条记录")
                    else:
                        print(f"    ⚠️  {date_str}: 无数据")
                except Exception as e:
                    print(f"    ❌ {date_str}: 下载失败 - {e}")

                # 增加延迟避免API限制
                time.sleep(0.3)

            print(f"  批次完成，当前累计: {len(market_data):,} 条记录")
            time.sleep(1)  # 批次间延迟

        if market_data:
            market_df = pd.concat(market_data, ignore_index=True)
            market_df = market_df.sort_values("trade_date")

            # 保存全市场数据
            market_file = (
                data_dir
                / "moneyflow_market"
                / f"market_moneyflow_{start_date_tushare}_{end_date_tushare}.parquet"
            )
            market_df.to_parquet(market_file, index=False)

            stats["market_data"] = len(market_df)
            print(f"\n  ✅ 全市场资金流向数据: {len(market_df):,} 条记录")
            print(f"  数据已保存: {market_file}")

            # 分析数据
            print(f"\n  📊 全市场资金流向分析:")
            print(
                f"    日期范围: {market_df['trade_date'].min()} ~ {market_df['trade_date'].max()}"
            )
            print(f"    总净流入: {market_df['net_mf_amount'].sum():,.0f} 万元")
            print(f"    日均净流入: {market_df['net_mf_amount'].mean():,.0f} 万元")
            print(
                f"    流入天数: {(market_df['net_mf_amount'] > 0).sum():,}/{len(market_df):,} ({(market_df['net_mf_amount'] > 0).sum()/len(market_df)*100:.1f}%)"
            )

            # 按净流入排序
            top_inflows = market_df.nlargest(10, "net_mf_amount")
            print(f"\n  📈 净流入排行榜 (前10):")
            for _, row in top_inflows.iterrows():
                print(
                    f"    {row['ts_code']} ({row['trade_date'].strftime('%Y-%m-%d')}): {row['net_mf_amount']:,.0f} 万元"
                )

    except Exception as e:
        print(f"  ❌ 全市场数据下载失败: {e}")

    # 2. 获取沪深港通资金流向数据
    print(f"\n2. 获取沪深港通资金流向数据...")
    print("-" * 50)

    try:
        hsgt_df = pro.moneyflow_hsgt(
            start_date=start_date_tushare, end_date=end_date_tushare
        )

        if not hsgt_df.empty:
            hsgt_df["trade_date"] = pd.to_datetime(hsgt_df["trade_date"])
            hsgt_df = hsgt_df.sort_values("trade_date")

            # 保存数据
            hsgt_file = (
                data_dir
                / "moneyflow_market"
                / f"hsgt_moneyflow_{start_date_tushare}_{end_date_tushare}.parquet"
            )
            hsgt_df.to_parquet(hsgt_file, index=False)

            stats["hsgt_data"] = len(hsgt_df)
            print(f"  ✅ 沪深港通数据: {len(hsgt_df)} 条记录")
            print(f"  数据已保存: {hsgt_file}")

            # 分析数据
            print(f"\n  📊 沪深港通资金流向分析:")
            print(
                f"    日期范围: {hsgt_df['trade_date'].min()} ~ {hsgt_df['trade_date'].max()}"
            )
            print(f"    北向资金净流入均值: {hsgt_df['north_money'].mean():.2f} 百万元")
            print(f"    南向资金净流入均值: {hsgt_df['south_money'].mean():.2f} 百万元")
            print(
                f"    北向资金流入天数: {(hsgt_df['north_money'] > 0).sum()}/{len(hsgt_df)} ({(hsgt_df['north_money'] > 0).sum()/len(hsgt_df)*100:.1f}%)"
            )

        else:
            print(f"  ⚠️  无沪深港通数据")

    except Exception as e:
        print(f"  ❌ 沪深港通数据下载失败: {e}")

    # 3. 基于ETF交易数据生成资金流向估算
    print(f"\n3. 生成ETF资金流向估算数据...")
    print("-" * 50)

    try:
        # 查找ETF日线数据文件
        daily_dir = data_dir / "daily"
        etf_files = list(daily_dir.glob("*daily*.parquet"))

        print(f"  找到 {len(etf_files)} 个ETF日线数据文件")

        etf_estimates = []
        for file_path in etf_files:
            try:
                symbol = file_path.stem.split("_")[0]

                # 匹配ETF代码
                etf_code = None
                for code in etf_codes:
                    if code.split(".")[0] == symbol:
                        etf_code = code
                        break

                if etf_code:
                    print(f"  🔄 处理 {etf_code} ({symbol})...")

                    df = pd.read_parquet(file_path)

                    # 生成资金流向估算指标
                    df["volume_ma5"] = df["vol"].rolling(5).mean()
                    df["volume_ma20"] = df["vol"].rolling(20).mean()
                    df["volume_ratio"] = df["vol"] / df["volume_ma20"]

                    # 成交额指标
                    df["amount_ma5"] = df["amount"].rolling(5).mean()
                    df["amount_ratio"] = df["amount"] / df["amount_ma5"]

                    # 价格动量指标
                    df["price_change_5d"] = df["close"].pct_change(5)
                    df["price_change_20d"] = df["close"].pct_change(20)

                    # 资金流向估算 - 多维度方法
                    # 方法1: 基于成交量和价格变化
                    df["moneyflow_basic"] = (
                        df["amount"] * df["volume_ratio"] * np.sign(df["pct_chg"])
                    )

                    # 方法2: 基于成交额异常
                    amount_std = df["amount"].rolling(20).std()
                    df["amount_anomaly"] = (
                        df["amount"] - df["amount_ma5"]
                    ) / amount_std
                    df["moneyflow_anomaly"] = np.where(
                        df["amount_anomaly"] > 2,
                        df["amount"] * np.sign(df["pct_chg"]),
                        0,
                    )

                    # 方法3: 大单净流入估算
                    volume_std = df["vol"].rolling(20).std()
                    df["large_order_signal"] = df["vol"] > (
                        df["volume_ma20"] + 1.5 * volume_std
                    )
                    df["moneyflow_large"] = np.where(
                        df["large_order_signal"],
                        df["amount"] * 0.6 * np.sign(df["pct_chg"]),
                        0,
                    )

                    # 综合资金流向指标
                    df["estimated_moneyflow"] = (
                        df["moneyflow_basic"] * 0.4
                        + df["moneyflow_anomaly"] * 0.3
                        + df["moneyflow_large"] * 0.3
                    )

                    # 保存估算数据
                    estimate_df = df[
                        [
                            "trade_date",
                            "close",
                            "vol",
                            "amount",
                            "pct_chg",
                            "volume_ratio",
                            "amount_ratio",
                            "price_change_5d",
                            "price_change_20d",
                            "moneyflow_basic",
                            "moneyflow_anomaly",
                            "large_order_signal",
                            "moneyflow_large",
                            "estimated_moneyflow",
                        ]
                    ].copy()

                    # 清理数据
                    estimate_df = estimate_df.dropna()

                    estimate_file = (
                        data_dir
                        / "moneyflow"
                        / f"{symbol}_moneyflow_estimated_{start_date_tushare}_{end_date_tushare}.parquet"
                    )
                    estimate_df.to_parquet(estimate_file, index=False)

                    etf_estimates.append(estimate_df)
                    print(f"    ✅ 估算数据已保存: {len(estimate_df)} 条记录")

                    # 显示统计
                    total_flow = estimate_df["estimated_moneyflow"].sum()
                    print(f"    📊 总估算净流入: {total_flow:,.0f} 千元")
                    print(f"    📊 日均净流入: {total_flow/len(estimate_df):,.0f} 千元")

                else:
                    print(f"  ⚠️  {symbol}: 未找到对应ETF代码")

            except Exception as e:
                print(f"  ❌ {file_path.name}: 处理失败 - {e}")

        stats["estimated_data"] = len(etf_estimates)
        print(
            f"\n  ✅ 生成ETF资金流向估算: {stats['estimated_data']}/{len(etf_files)} 只ETF"
        )

    except Exception as e:
        print(f"  ❌ 资金流向估算生成失败: {e}")

    # 4. 输出最终统计
    stats["end_time"] = datetime.now()
    duration = stats["end_time"] - stats["start_time"]

    print("\n" + "=" * 80)
    print("资金流向数据下载完成！")
    print("=" * 80)
    print(f"全市场数据记录: {stats['market_data']:,} 条")
    print(f"沪深港通数据记录: {stats['hsgt_data']} 条")
    print(f"估算ETF数据: {stats['estimated_data']} 只")
    print(f"总耗时: {duration}")
    print()

    # 5. 数据使用建议
    print("💡 数据使用建议:")
    print("1. 使用估算的资金流向数据进行分析")
    print("2. 结合多维度指标判断资金流向趋势")
    print("3. 参考全市场数据了解整体市场情绪")
    print("4. 关注沪深港通数据反映外资流向")

    print(f"\n✅ 资金流向数据下载完成！")
    print(f"数据文件位置:")
    print(f"  raw/ETF/moneyflow/ - ETF资金流向估算数据")
    print(f"  raw/ETF/moneyflow_market/ - 市场资金流向数据")

    # 6. 生成使用示例
    print(f"\n📝 使用示例代码:")
    print(f"```python")
    print(f"import pandas as pd")
    print(f"# 加载ETF资金流向估算数据")
    print(
        f"df = pd.read_parquet('raw/ETF/moneyflow/510300_moneyflow_estimated_20241014_20251014.parquet')"
    )
    print(f"")
    print(f"# 分析资金流向趋势")
    print(f"df['cumulative_flow'] = df['estimated_moneyflow'].cumsum()")
    print(f"print(f\"累计净流入: {{df['cumulative_flow'].iloc[-1]:,.0f}} 千元\")")
    print(f"")
    print(f"# 识别资金流向信号")
    print(
        f"buy_signals = df[df['estimated_moneyflow'] > df['estimated_moneyflow'].quantile(0.8)]"
    )
    print(f'print(f"发现 {{len(buy_signals)}} 个强买入信号")')
    print(f"```")


if __name__ == "__main__":
    main()
