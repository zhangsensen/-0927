#!/usr/bin/env python3
"""
市场环境变化分析
===================================
检查训练期(2020-01~2025-04)与Holdout期(2025-05~2025-12)的市场环境差异

关键问题:
- 是因子失效？还是市场环境变化？
- Holdout期是牛市/熊市/震荡市？
- ETF池的收益分布有何变化？
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml

ROOT = Path(__file__).parent.parent


def load_etf_data():
    """加载ETF数据"""
    print("📊 加载ETF数据...")

    # 读取配置
    with open(ROOT / "configs/combo_wfo_config.yaml") as f:
        config = yaml.safe_load(f)

    etf_codes = config["data"]["symbols"]
    data_dir = Path(config["data"]["data_dir"])

    # 加载所有ETF
    etf_returns = {}

    for code in etf_codes:
        # 查找文件
        files = list(data_dir.glob(f"{code}*.parquet"))
        if not files:
            print(f"⚠️ 未找到 {code}")
            continue

        df = pd.read_parquet(files[0])
        df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        df = df.sort_values("date")

        # 使用复权收盘价
        df["close"] = df["adj_close"]

        # 计算日收益
        df["return"] = df["close"].pct_change()
        etf_returns[code] = df[["date", "close", "return"]].set_index("date")

    print(f"✅ 加载完成: {len(etf_returns)} 只ETF")
    return etf_returns


def analyze_period_stats(etf_returns, start_date, end_date, period_name):
    """分析特定时期的市场统计"""
    print(f"\n{'='*80}")
    print(f"📈 {period_name} ({start_date} ~ {end_date})")
    print(f"{'='*80}")

    period_stats = []

    for code, df in etf_returns.items():
        mask = (df.index >= start_date) & (df.index <= end_date)
        period_data = df[mask]

        if len(period_data) < 2:
            continue

        # 总收益
        total_return = period_data["close"].iloc[-1] / period_data["close"].iloc[0] - 1

        # 波动率
        volatility = period_data["return"].std() * np.sqrt(252)

        # Sharpe (假设无风险利率=0)
        mean_ret = period_data["return"].mean() * 252
        sharpe = mean_ret / volatility if volatility > 0 else 0

        # 最大回撤
        cummax = period_data["close"].cummax()
        drawdown = (period_data["close"] - cummax) / cummax
        max_dd = drawdown.min()

        period_stats.append(
            {
                "code": code,
                "total_return": total_return,
                "volatility": volatility,
                "sharpe": sharpe,
                "max_dd": max_dd,
                "days": len(period_data),
            }
        )

    df_stats = pd.DataFrame(period_stats)

    # 整体统计
    print(f"\n【整体市场表现】")
    print(f"  平均收益率: {df_stats['total_return'].mean():.2%}")
    print(f"  中位收益率: {df_stats['total_return'].median():.2%}")
    print(f"  收益率标准差: {df_stats['total_return'].std():.2%}")
    print(f"  正收益ETF占比: {(df_stats['total_return'] > 0).mean():.2%}")
    print(f"  收益>10%占比: {(df_stats['total_return'] > 0.10).mean():.2%}")
    print(f"  收益>20%占比: {(df_stats['total_return'] > 0.20).mean():.2%}")
    print(f"  平均波动率: {df_stats['volatility'].mean():.2%}")
    print(f"  平均Sharpe: {df_stats['sharpe'].mean():.4f}")
    print(f"  平均最大回撤: {df_stats['max_dd'].mean():.2%}")

    # Top10 和 Bottom10
    print(f"\n【Top10 表现最好的ETF】")
    top10 = df_stats.nlargest(10, "total_return")
    for idx, row in top10.iterrows():
        print(
            f"  {row['code']:12} {row['total_return']:+7.2%} | Sharpe={row['sharpe']:6.2f} | MaxDD={row['max_dd']:7.2%}"
        )

    print(f"\n【Bottom10 表现最差的ETF】")
    bottom10 = df_stats.nsmallest(10, "total_return")
    for idx, row in bottom10.iterrows():
        print(
            f"  {row['code']:12} {row['total_return']:+7.2%} | Sharpe={row['sharpe']:6.2f} | MaxDD={row['max_dd']:7.2%}"
        )

    return df_stats


def compare_periods(train_stats, hold_stats):
    """对比训练期和holdout期"""
    print(f"\n{'='*80}")
    print(f"🔍 训练期 vs Holdout期 对比")
    print(f"{'='*80}")

    # 合并
    merged = train_stats.merge(hold_stats, on="code", suffixes=("_train", "_hold"))

    print(f"\n【平均指标变化】")
    print(
        f"  收益率: 训练={merged['total_return_train'].mean():.2%} → Holdout={merged['total_return_hold'].mean():.2%}"
    )
    print(
        f"  波动率: 训练={merged['volatility_train'].mean():.2%} → Holdout={merged['volatility_hold'].mean():.2%}"
    )
    print(
        f"  Sharpe: 训练={merged['sharpe_train'].mean():.4f} → Holdout={merged['sharpe_hold'].mean():.4f}"
    )
    print(
        f"  最大回撤: 训练={merged['max_dd_train'].mean():.2%} → Holdout={merged['max_dd_hold'].mean():.2%}"
    )

    # 收益相关性
    corr = merged["total_return_train"].corr(merged["total_return_hold"])
    print(f"\n【ETF收益相关性】")
    print(f"  Pearson相关: {corr:.4f}")

    # 排序稳定性
    merged["rank_train"] = merged["total_return_train"].rank(ascending=False)
    merged["rank_hold"] = merged["total_return_hold"].rank(ascending=False)
    rank_corr = merged["rank_train"].corr(merged["rank_hold"])
    print(f"  Spearman秩相关: {rank_corr:.4f}")

    # 强弱互换
    print(f"\n【强弱互换ETF (训练Top20 vs Holdout表现)】")
    train_top20 = merged.nsmallest(20, "rank_train")  # rank越小越好
    print(
        f"  训练Top20在Holdout平均收益: {train_top20['total_return_hold'].mean():.2%}"
    )
    print(
        f"  训练Top20在Holdout正收益占比: {(train_top20['total_return_hold'] > 0).mean():.2%}"
    )

    # 显示训练Top20在Holdout的排名变化
    print(
        f"\n{'ETF':12} {'训练排名':>10} {'Hold排名':>10} {'排名变化':>10} {'Hold收益':>10}"
    )
    print("-" * 62)
    train_top20_sorted = train_top20.sort_values("rank_train")
    for idx, row in train_top20_sorted.iterrows():
        rank_change = int(row["rank_hold"] - row["rank_train"])
        print(
            f"{row['code']:12} {int(row['rank_train']):10d} {int(row['rank_hold']):10d} {rank_change:+10d} {row['total_return_hold']:+9.2%}"
        )


def main():
    """主函数"""
    print("=" * 80)
    print("🔬 市场环境变化分析")
    print("=" * 80)

    # 加载数据
    etf_returns = load_etf_data()

    # 定义时期
    train_start = "2020-01-01"
    train_end = "2025-04-30"
    hold_start = "2025-05-01"
    hold_end = "2025-12-08"

    # 分析训练期
    train_stats = analyze_period_stats(etf_returns, train_start, train_end, "训练期")

    # 分析Holdout期
    hold_stats = analyze_period_stats(etf_returns, hold_start, hold_end, "Holdout期")

    # 对比
    compare_periods(train_stats, hold_stats)

    print("\n" + "=" * 80)
    print("✅ 分析完成")
    print("=" * 80)

    print("\n💡 诊断建议:")
    print("  1. 如果Holdout期整体表现大幅低于训练期 → 市场进入熊市/震荡")
    print("  2. 如果ETF收益相关性低 → 风格轮动，需要因子调整")
    print("  3. 如果训练Top20在Holdout表现差 → WFO选出的因子过拟合")


if __name__ == "__main__":
    main()
