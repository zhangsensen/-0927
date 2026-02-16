#!/usr/bin/env python3
"""
快速评估ETF资金流向因子的Alpha价值

用爬取的大单数据构建因子，计算Rank IC与现有OHLCV因子对比，
判断是否值得纳入因子库。

候选因子:
1. MAIN_NET_FLOW_PCT: 主力净流入占比 (超大单+大单)
2. XL_NET_FLOW_PCT: 超大单净流入占比
3. SMART_MONEY_RATIO: 超大单 vs 小单比值
4. MAIN_FLOW_5D_SUM: 5日主力资金累计
5. FLOW_MOMENTUM: 主力流入动量 (5d vs 20d)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_fund_flow_data(etf_codes: list) -> dict:
    """加载所有ETF的资金流向数据"""
    moneyflow_dir = ROOT / "raw" / "ETF" / "moneyflow"
    data = {}
    for code in etf_codes:
        path = moneyflow_dir / f"fund_flow_{code}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            data[code] = df
    return data


def build_fund_flow_factors(flow_data: dict, etf_codes: list) -> dict:
    """从资金流数据构建因子矩阵"""
    # 获取所有日期的并集
    all_dates = set()
    for df in flow_data.values():
        all_dates.update(df.index)
    all_dates = sorted(all_dates)
    index = pd.DatetimeIndex(all_dates)

    factors = {}

    # 1. 主力净占比 (当日)
    main_pct = pd.DataFrame(index=index, columns=etf_codes, dtype=float)
    for code, df in flow_data.items():
        main_pct.loc[df.index, code] = df["main_net_pct"].values
    factors["MAIN_NET_FLOW_PCT"] = main_pct

    # 2. 超大单净占比
    xl_pct = pd.DataFrame(index=index, columns=etf_codes, dtype=float)
    for code, df in flow_data.items():
        xl_pct.loc[df.index, code] = df["xl_net_pct"].values
    factors["XL_NET_FLOW_PCT"] = xl_pct

    # 3. 聪明钱比率: 超大单净流入 / (|小单净流入| + ε)
    smart = pd.DataFrame(index=index, columns=etf_codes, dtype=float)
    for code, df in flow_data.items():
        ratio = df["xl_net"] / (df["s_net"].abs() + 1e6)
        smart.loc[df.index, code] = ratio.values
    factors["SMART_MONEY_RATIO"] = smart

    # 4. 5日主力资金累计占比
    main_5d = main_pct.rolling(5, min_periods=3).sum()
    factors["MAIN_FLOW_5D_SUM"] = main_5d

    # 5. 资金动量: 5日均值 / 20日均值
    ma5 = main_pct.rolling(5, min_periods=3).mean()
    ma20 = main_pct.rolling(20, min_periods=10).mean()
    factors["FLOW_MOMENTUM"] = (ma5 / (ma20.abs() + 0.01)) - 1

    # 6. 大单净占比 (区分于超大单)
    l_pct = pd.DataFrame(index=index, columns=etf_codes, dtype=float)
    for code, df in flow_data.items():
        l_pct.loc[df.index, code] = df["l_net_pct"].values
    factors["LARGE_ORDER_PCT"] = l_pct

    return factors


def compute_rank_ic(factor_df, fwd_ret_df):
    """计算Rank IC"""
    ic_list = []
    dates = factor_df.index.intersection(fwd_ret_df.index)

    for dt in dates:
        fv = factor_df.loc[dt].dropna()
        rv = fwd_ret_df.loc[dt].dropna()
        common = fv.index.intersection(rv.index)
        if len(common) < 5:
            continue
        corr = fv[common].rank().corr(rv[common].rank())
        if np.isfinite(corr):
            ic_list.append(corr)

    return np.array(ic_list) if ic_list else np.array([])


def main():
    import yaml

    print("=" * 80)
    print("🔍 ETF资金流向因子 Alpha评估")
    print("=" * 80)

    # 加载配置
    config_path = ROOT / "configs" / "combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    etf_codes = config["data"]["symbols"]

    # 加载OHLCV (用于计算前瞻收益)
    from etf_strategy.core.data_loader import DataLoader
    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    # 确保覆盖资金流向数据的时间段 (2025-08 ~ 2026-02)
    ohlcv = loader.load_ohlcv(
        etf_codes=etf_codes,
        start_date="2020-01-01",
        end_date="2026-02-28",
        use_cache=False,
    )
    close = ohlcv["close"]

    # 前瞻收益 (1日, 5日)
    fwd_1d = close.pct_change(1).shift(-1)
    fwd_5d = close.pct_change(5).shift(-5)

    print(f"📊 OHLCV: {len(close)}天, {len(close.columns)}个ETF")

    # 加载资金流向数据
    print(f"\n📊 加载资金流向数据...")
    flow_data = load_fund_flow_data(etf_codes)
    print(f"  成功加载: {len(flow_data)} 个ETF")

    if not flow_data:
        print("❌ 无数据!")
        return

    # 数据范围
    first_code = list(flow_data.keys())[0]
    print(f"  数据范围: {flow_data[first_code].index[0].date()} → {flow_data[first_code].index[-1].date()}")
    print(f"  天数: {len(flow_data[first_code])}")

    # 构建因子
    print(f"\n🔧 构建因子...")
    factors = build_fund_flow_factors(flow_data, etf_codes)
    print(f"  因子数: {len(factors)}")

    # IC评估
    print(f"\n🔍 Rank IC 评估 (1日 & 5日前瞻)")
    print("-" * 90)
    print(f"{'因子':<25} {'IC_1D':>8} {'IR_1D':>8} {'正样本_1D':>10} {'IC_5D':>8} {'IR_5D':>8} {'正样本_5D':>10}")
    print("-" * 90)

    results = []
    for fname, factor_df in factors.items():
        # 1日
        ic_1d = compute_rank_ic(factor_df, fwd_1d)
        # 5日
        ic_5d = compute_rank_ic(factor_df, fwd_5d)

        if len(ic_1d) < 10 or len(ic_5d) < 10:
            print(f"  {fname:<25} 样本不足 (1d={len(ic_1d)}, 5d={len(ic_5d)})")
            continue

        ic_1d_mean = ic_1d.mean()
        ir_1d = ic_1d_mean / ic_1d.std() if ic_1d.std() > 0 else 0
        pos_1d = (ic_1d > 0).mean()

        ic_5d_mean = ic_5d.mean()
        ir_5d = ic_5d_mean / ic_5d.std() if ic_5d.std() > 0 else 0
        pos_5d = (ic_5d > 0).mean()

        # 评级
        abs_ic = abs(ic_5d_mean)
        abs_ir = abs(ir_5d)
        if abs_ic >= 0.05 and abs_ir >= 0.3:
            rating = "🟢强"
        elif abs_ic >= 0.03 or abs_ir >= 0.2:
            rating = "🟡中"
        elif abs_ic >= 0.02 or abs_ir >= 0.1:
            rating = "🟠弱"
        else:
            rating = "🔴无效"

        results.append({
            "因子": fname,
            "IC_1D": ic_1d_mean,
            "IR_1D": ir_1d,
            "正样本_1D": pos_1d,
            "IC_5D": ic_5d_mean,
            "IR_5D": ir_5d,
            "正样本_5D": pos_5d,
            "评级": rating,
            "样本数": len(ic_5d),
        })

        print(f"  {fname:<25} {ic_1d_mean:>+.4f} {ir_1d:>+.3f} {pos_1d:>9.1%} "
              f"{ic_5d_mean:>+.4f} {ir_5d:>+.3f} {pos_5d:>9.1%}  {rating}")

    # 对比基线
    print("\n" + "=" * 90)
    print("📋 对比: 资金流因子 vs 现有OHLCV因子最佳(PRICE_POSITION_120D)")
    print("=" * 90)
    print(f"  PRICE_POSITION_120D (OHLCV):  IC_5D=+0.0375, IR_5D=+0.107")

    if results:
        best = max(results, key=lambda x: abs(x["IC_5D"]))
        print(f"  {best['因子']} (资金流): IC_5D={best['IC_5D']:+.4f}, IR_5D={best['IR_5D']:+.3f}")
        if abs(best["IC_5D"]) > 0.0375:
            print(f"\n  🎉 资金流因子 > OHLCV最佳因子! 有增量Alpha!")
        elif abs(best["IC_5D"]) > 0.02:
            print(f"\n  ⚠️ 资金流因子有一定信号, 可做补充因子")
        else:
            print(f"\n  ❌ 资金流因子信号与OHLCV因子相当, 增量有限")

    # 保存报告
    df_results = pd.DataFrame(results)
    report_path = ROOT / "results" / "fund_flow_factor_evaluation.md"
    with open(report_path, "w") as f:
        f.write("# ETF资金流向因子评估报告\n\n")
        f.write(f"**数据来源**: 东财资金流向 (push2his API)\n")
        f.write(f"**数据范围**: ~120天\n")
        f.write(f"**ETF数量**: {len(flow_data)}\n\n")
        f.write("## IC评估结果\n\n")
        f.write("| 因子 | IC_5D | IR_5D | 正样本率 | 评级 |\n")
        f.write("|------|-------|-------|---------|------|\n")
        for _, r in df_results.iterrows():
            f.write(f"| {r['因子']} | {r['IC_5D']:+.4f} | {r['IR_5D']:+.3f} | "
                    f"{r['正样本_5D']:.1%} | {r['评级']} |\n")

    print(f"\n📄 报告: {report_path}")


if __name__ == "__main__":
    main()
