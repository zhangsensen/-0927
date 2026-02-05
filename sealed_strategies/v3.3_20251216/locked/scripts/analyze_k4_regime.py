import pandas as pd
import numpy as np
from pathlib import Path
import yaml

ROOT = Path("/home/sensen/dev/projects/-0927")


def load_etf_data():
    """加载ETF数据"""
    print("📊 加载ETF数据...")

    # 读取配置
    with open(ROOT / "configs/combo_wfo_config.yaml") as f:
        config = yaml.safe_load(f)

    etf_codes = config["data"]["symbols"]
    data_dir = Path(config["data"]["data_dir"])
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir

    print(f"📂 Data Dir: {data_dir}")

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
    print(f"  收益<-10%占比: {(df_stats['total_return'] < -0.10).mean():.2%}")
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


def main():
    etf_returns = load_etf_data()

    # Pre-K4 (Holdout Part 1)
    pre_k4_start = "2025-05-01"
    pre_k4_end = "2025-10-14"

    # K4 (Holdout Part 2 - The Problematic Period)
    k4_start = "2025-10-15"
    k4_end = "2025-12-12"

    pre_k4_stats = analyze_period_stats(
        etf_returns, pre_k4_start, pre_k4_end, "Pre-K4 (Holdout Part 1)"
    )
    k4_stats = analyze_period_stats(
        etf_returns, k4_start, k4_end, "K4 (Holdout Part 2)"
    )


if __name__ == "__main__":
    main()
