#!/usr/bin/env python3
"""
策略筛选 v2.0 - 基于新开发思想

核心原则：
1. 锁死交易规则：FREQ, POS, 不止损, 不 cash (配置文件定义)
2. IC 只做"有无预测力"的门槛
3. 最终排序：OOS 收益 + Sharpe + 回撤 的综合得分

使用流程：
1. 先运行 WFO: uv run python src/etf_strategy/run_combo_wfo.py
2. 再运行全量 VEC: uv run python scripts/run_full_space_vec_backtest.py
3. 最后运行本脚本进行筛选: uv run python scripts/select_strategy_v2.py
"""

import sys
from pathlib import Path
from datetime import datetime
import yaml
import pandas as pd
import numpy as np

from etf_strategy.core.data_loader import DataLoader

ROOT = Path(__file__).parent.parent

# 近期震荡市窗口（Regime Fitness Filter）
RECENT_START = pd.Timestamp("2025-01-01")
RECENT_END = pd.Timestamp("2025-05-31")


def load_latest_results():
    """加载最新的 WFO 和 VEC 结果"""
    results_dir = ROOT / "results"
    
    # 查找最新 WFO 结果
    wfo_dirs = sorted([d for d in results_dir.glob("run_*") if d.is_dir() and not d.is_symlink()])
    if not wfo_dirs:
        raise FileNotFoundError("未找到 WFO 结果目录 (run_*)")
    latest_wfo = wfo_dirs[-1]
    
    # 查找最新 VEC 结果 (full space)
    vec_dirs = sorted([d for d in results_dir.glob("vec_full_space_*") if d.is_dir()])
    if not vec_dirs:
        raise FileNotFoundError("未找到全量 VEC 结果目录 (vec_full_space_*)")
    latest_vec = vec_dirs[-1]
    
    # 加载数据
    wfo_path = latest_wfo / "all_combos.parquet"
    if not wfo_path.exists():
        raise FileNotFoundError(f"WFO 结果文件不存在: {wfo_path}")
    
    vec_path = latest_vec / "full_space_results.parquet"
    if not vec_path.exists():
        # Backward-compat: older runs used CSV
        vec_path = latest_vec / "full_space_results.csv"
    if not vec_path.exists():
        raise FileNotFoundError(f"VEC 结果文件不存在: {vec_path}")
    
    wfo = pd.read_parquet(wfo_path)
    if vec_path.suffix.lower() in {".parquet", ".pq"}:
        vec = pd.read_parquet(vec_path)
    else:
        vec = pd.read_csv(vec_path)
    
    return wfo, vec, latest_wfo.name, latest_vec.name


# ============================================================================
# P0: 高风险因子过滤
# ============================================================================

# 高风险因子列表 (BT 审计发现 VEC/BT 差异过大)
# 2025-12-01 Top1000 BT 审计结果:
#   - OBV_SLOPE_10D: 平均差异 61pp, BT 收益仅 35% vs VEC 96%
#   - CMF_20D: 平均差异 35pp
#   - VOL_RATIO_60D: 平均差异 13.86pp (含 170 个策略), 58% 高差异策略含此因子
RISKY_FACTORS = ["OBV_SLOPE_10D", "CMF_20D", "VOL_RATIO_60D"]

# 禁止组合 (这些因子同时出现时风险极高)
# VOL_RATIO_20D + VOL_RATIO_60D: 平均差异 25.85pp, 31 个策略中 26 个差异 > 10pp
BANNED_FACTOR_COMBOS = [
    ("VOL_RATIO_20D", "VOL_RATIO_60D"),
]


def filter_risky_combos(df: pd.DataFrame, risky_factors: list = None, banned_combos: list = None) -> pd.DataFrame:
    """过滤包含高风险因子的组合
    
    参数:
        df: 包含 'combo' 列的 DataFrame
        risky_factors: 高风险因子列表，默认使用 RISKY_FACTORS
        banned_combos: 禁止的因子组合列表，默认使用 BANNED_FACTOR_COMBOS
        
    返回:
        过滤后的 DataFrame，并添加 'is_production_ready' 列
    """
    if risky_factors is None:
        risky_factors = RISKY_FACTORS
    if banned_combos is None:
        banned_combos = BANNED_FACTOR_COMBOS
    
    if not risky_factors and not banned_combos:
        df["is_production_ready"] = True
        df["risky_factors"] = ""
        return df
    
    def check_combo(combo: str) -> tuple[bool, str]:
        """检查组合是否包含高风险因子或禁止组合"""
        factors = set(f.strip() for f in combo.split(' + '))
        issues = []
        
        # 检查单个高风险因子
        risky_found = [f for f in factors if f in risky_factors]
        if risky_found:
            issues.extend(risky_found)
        
        # 检查禁止组合
        for banned in banned_combos:
            if all(f in factors for f in banned):
                issues.append(f"[禁止组合: {'+'.join(banned)}]")
        
        return len(issues) == 0, ', '.join(issues)
    
    results = df["combo"].apply(check_combo)
    df["is_production_ready"] = results.apply(lambda x: x[0])
    df["risky_factors"] = results.apply(lambda x: x[1])
    
    n_risky = (~df["is_production_ready"]).sum()
    n_safe = df["is_production_ready"].sum()
    
    print(f"\n🔒 高风险因子过滤")
    print(f"   高风险因子: {risky_factors}")
    print(f"   禁止组合: {banned_combos}")
    print(f"   可用于生产: {n_safe} 个")
    print(f"   仅限研究: {n_risky} 个 (含高风险因子/禁止组合)")
    
    # 返回过滤后的安全组合
    safe_df = df[df["is_production_ready"]].copy()
    return safe_df


def apply_ic_threshold(merged: pd.DataFrame, config: dict) -> pd.DataFrame:
    """应用 IC 门槛过滤 (有无预测力)
    
    门槛条件 (OR 关系):
    - mean_oos_ic > ic_threshold (默认 0.05)
    - positive_rate > pr_threshold (默认 55%)
    """
    selection_config = config.get("selection", {})
    ic_threshold = selection_config.get("ic_threshold", 0.05)
    pr_threshold = selection_config.get("positive_rate_threshold", 0.55)
    
    # 组合条件：IC > 门槛 OR positive_rate > 门槛
    mask = (merged["mean_oos_ic"] > ic_threshold) | (merged["positive_rate"] > pr_threshold)
    
    passed = merged[mask].copy()
    
    print(f"\n📊 IC 门槛过滤 (IC > {ic_threshold} OR PR > {pr_threshold*100:.0f}%)")
    print(f"   通过: {len(passed)} / {len(merged)} ({len(passed)/len(merged)*100:.1f}%)")
    print(f"   过滤: {len(merged) - len(passed)} 个策略")
    
    return passed


def _build_oos_windows(config: dict) -> list[pd.Interval]:
    """根据配置构建 WFO 的 OOS 窗口日期区间。

    返回值: list[pd.Interval]，长度应与 oos_return_list 对齐。
    """
    data_cfg = config.get("data", {})
    start_date = data_cfg.get("start_date")
    end_date = data_cfg.get("training_end_date") or data_cfg.get("end_date")

    # 加载交易日索引（仅需 close 价格索引即可）
    loader = DataLoader(
        data_dir=data_cfg.get("data_dir"),
        cache_dir=data_cfg.get("cache_dir"),
    )
    prices = loader.load_ohlcv(
        etf_codes=data_cfg.get("symbols"),
        start_date=start_date,
        end_date=end_date,
    )

    dates = pd.to_datetime(prices["close"].index)

    combo_cfg = config.get("combo_wfo", {})
    is_period = combo_cfg.get("is_period", 252)
    oos_period = combo_cfg.get("oos_period", 60)
    step_size = combo_cfg.get("step_size", 60)

    windows = []
    idx = is_period
    while idx + oos_period <= len(dates):
        oos_slice = dates[idx : idx + oos_period]
        if len(oos_slice) == 0:
            break
        windows.append(pd.Interval(left=oos_slice[0], right=oos_slice[-1], closed="both"))
        idx += step_size

    return windows


def apply_regime_filter(merged: pd.DataFrame, config: dict) -> pd.DataFrame:
    """近期震荡市适应性过滤。

    要求：
      - 仅保留近 5 个月 (2025-01-01 ~ 2025-05-31) 区间收益为正的组合
      - 仅保留该区间最大回撤不超过 8% 的组合
    """

    windows = _build_oos_windows(config)
    if not windows:
        print("⚠️ 未生成 OOS 窗口，跳过 Regime Fitness Filter")
        merged["Recent_Ret_5M"] = np.nan
        merged["Recent_MDD_5M"] = np.nan
        return merged

    # 找到落在目标区间内的窗口索引
    recent_idx = [i for i, w in enumerate(windows) if (w.left >= RECENT_START and w.right <= RECENT_END)]
    if not recent_idx:
        print("⚠️ 未找到位于 2025-01-01~2025-05-31 的 OOS 窗口，跳过 Regime Fitness Filter")
        merged["Recent_Ret_5M"] = np.nan
        merged["Recent_MDD_5M"] = np.nan
        return merged

    def compute_recent_metrics(oos_returns) -> tuple[float, float]:
        if oos_returns is None:
            return np.nan, np.nan
        if isinstance(oos_returns, str):
            oos_returns = np.fromstring(oos_returns.replace("[", " ").replace("]", " "), sep=" ")
        returns_arr = np.asarray(oos_returns, dtype=float)
        if returns_arr.size != len(windows):
            returns_arr = returns_arr[: len(windows)]
        if len(returns_arr) < max(recent_idx) + 1:
            return np.nan, np.nan
        recent = returns_arr[recent_idx]
        if recent.size == 0:
            return np.nan, np.nan
        equity = np.cumprod(1 + recent)
        period_ret = equity[-1] - 1
        peak = np.maximum.accumulate(equity)
        dd = np.max((peak - equity) / peak) if equity.size > 0 else np.nan
        return period_ret, dd

    metrics = merged["oos_return_list"].apply(compute_recent_metrics)
    merged["Recent_Ret_5M"] = metrics.apply(lambda x: x[0])
    merged["Recent_MDD_5M"] = metrics.apply(lambda x: x[1])

    # 硬性门槛
    filtered = merged[(merged["Recent_Ret_5M"] >= 0.0) & (merged["Recent_MDD_5M"] <= 0.08)].copy()

    print("\n🧭 Regime Fitness Filter (近期震荡市适应性)")
    print(f"   近期窗口: {RECENT_START.date()} → {RECENT_END.date()}")
    print(f"   门槛: 近期收益 >= 0%, 近期MaxDD <= 8%")
    print(f"   通过: {len(filtered)} / {len(merged)} ({len(filtered)/len(merged)*100:.1f}%)")
    print(f"   过滤: {len(merged) - len(filtered)} 个策略 (近期表现不佳)")

    return filtered


def compute_composite_score(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """计算综合得分
    
    综合得分 = w1 * return_rank + w2 * sharpe_rank + w3 * (1 - drawdown_rank)
    
    v2.1 更新: 使用滚动 OOS 收益作为排名依据，避免样本选择偏差
    
    收益排名优先级:
    1. mean_oos_return: 滚动 OOS 平均窗口收益 (每窗口 60 天的平均收益)
       - 优点: 真正的样本外收益，避免过拟合
       - 注意: 不使用 cum_oos_return 因为累乘会夸大差异
    2. vec_return: 全量样本收益 (回退选项)
    
    默认权重: 收益 40%, Sharpe 30%, 回撤 30%
    """
    selection_config = config.get("selection", {})
    weights = selection_config.get("composite_weights", {
        "return": 0.4,
        "sharpe": 0.3,
        "drawdown": 0.3,
    })
    
    w_return = weights.get("return", 0.4)
    w_sharpe = weights.get("sharpe", 0.3)
    w_drawdown = weights.get("drawdown", 0.3)
    
    # v2.1: 使用 mean_oos_return (滚动 OOS 平均窗口收益) 作为排名依据
    # 不使用 cum_oos_return 因为累乘会夸大收益差异
    if "mean_oos_return" in df.columns:
        return_col = "mean_oos_return"
        print("   使用 mean_oos_return (滚动 OOS 平均窗口收益) 作为排名依据")
    else:
        return_col = "vec_return"
        print("   ⚠️ 未找到 mean_oos_return，回退到 vec_return (全量样本)")
    
    # 计算各指标的百分位排名
    df["return_rank"] = df[return_col].rank(pct=True)
    df["sharpe_rank"] = df["vec_sharpe_ratio"].rank(pct=True)
    df["dd_rank"] = df["vec_max_drawdown"].rank(pct=True, ascending=True)  # 回撤越小越好
    
    # 计算综合得分
    df["composite_score"] = (
        w_return * df["return_rank"] +
        w_sharpe * df["sharpe_rank"] +
        w_drawdown * (1 - df["dd_rank"])
    )
    
    print(f"\n📊 综合得分计算 (权重: 收益{w_return*100:.0f}%, Sharpe{w_sharpe*100:.0f}%, 回撤{w_drawdown*100:.0f}%)")
    
    return df


def display_top_strategies(df: pd.DataFrame, top_n: int = 20):
    """显示 Top N 策略"""
    sorted_df = df.sort_values("composite_score", ascending=False)
    
    # 检查是否有 mean_oos_return
    has_oos_return = "mean_oos_return" in df.columns
    
    print(f"\n{'='*120}")
    print(f"🏆 Top {top_n} 策略 (按综合得分排序)")
    print(f"{'='*120}")
    
    if has_oos_return:
        print(f"{'排名':^4} | {'OOS收益':^8} | {'VEC收益':^8} | {'Sharpe':^7} | {'MaxDD':^7} | {'IC':^7} | {'PR':^6} | {'得分':^6} | 组合")
        print("-" * 120)
        
        for rank, (_, row) in enumerate(sorted_df.head(top_n).iterrows(), 1):
            combo_display = row["combo"][:40] + "..." if len(row["combo"]) > 40 else row["combo"]
            print(f"{rank:4d} | {row['mean_oos_return']:>7.2%} | {row['vec_return']:>7.2%} | {row['vec_sharpe_ratio']:>7.3f} | "
                  f"{row['vec_max_drawdown']:>6.1%} | "
                  f"{row['mean_oos_ic']:>6.4f} | {row['positive_rate']:>5.1%} | "
                  f"{row['composite_score']:>5.3f} | {combo_display}")
    else:
        print(f"{'排名':^4} | {'收益':^8} | {'Sharpe':^7} | {'MaxDD':^7} | {'Calmar':^7} | {'IC':^7} | {'PR':^6} | {'得分':^6} | 组合")
        print("-" * 120)
        
        for rank, (_, row) in enumerate(sorted_df.head(top_n).iterrows(), 1):
            combo_display = row["combo"][:40] + "..." if len(row["combo"]) > 40 else row["combo"]
            print(f"{rank:4d} | {row['vec_return']:>7.2%} | {row['vec_sharpe_ratio']:>7.3f} | "
                  f"{row['vec_max_drawdown']:>6.1%} | {row['vec_calmar_ratio']:>7.2f} | "
                  f"{row['mean_oos_ic']:>6.4f} | {row['positive_rate']:>5.1%} | "
                  f"{row['composite_score']:>5.3f} | {combo_display}")
    
    return sorted_df


def analyze_factor_frequency(df: pd.DataFrame, top_n: int = 20):
    """分析 Top N 策略中的因子出现频率"""
    sorted_df = df.sort_values("composite_score", ascending=False).head(top_n)
    
    factor_counts = {}
    for combo in sorted_df["combo"]:
        factors = [f.strip() for f in combo.split(" + ")]
        for f in factors:
            factor_counts[f] = factor_counts.get(f, 0) + 1
    
    print(f"\n📊 Top {top_n} 策略中的因子频率:")
    print("-" * 50)
    for factor, count in sorted(factor_counts.items(), key=lambda x: -x[1]):
        pct = count / top_n * 100
        bar = "█" * int(pct / 5)
        print(f"  {factor:35s} {count:3d} ({pct:5.1f}%) {bar}")


def compare_with_ic_ranking(merged: pd.DataFrame, selected: pd.DataFrame):
    """对比 IC 排序 vs 综合得分排序"""
    print(f"\n{'='*80}")
    print("📊 排序方法对比")
    print(f"{'='*80}")
    
    # IC 排序 Top1
    ic_top1 = merged.nlargest(1, "mean_oos_ic").iloc[0]
    
    # 综合得分 Top1
    score_top1 = selected.nlargest(1, "composite_score").iloc[0]
    
    # 检查是否有 mean_oos_return
    has_oos_return = "mean_oos_return" in merged.columns
    
    print("\n【原方法】按 IC 排序的 Top1:")
    print(f"  组合: {ic_top1['combo']}")
    if has_oos_return:
        print(f"  OOS收益: {ic_top1['mean_oos_return']:.2%} (平均窗口)")
    print(f"  VEC收益: {ic_top1['vec_return']:.2%}")
    print(f"  Sharpe: {ic_top1['vec_sharpe_ratio']:.3f}")
    print(f"  MaxDD: {ic_top1['vec_max_drawdown']:.1%}")
    print(f"  IC: {ic_top1['mean_oos_ic']:.4f}")
    
    print("\n【新方法】按综合得分排序的 Top1:")
    print(f"  组合: {score_top1['combo']}")
    if has_oos_return:
        print(f"  OOS收益: {score_top1['mean_oos_return']:.2%} (平均窗口)")
    print(f"  VEC收益: {score_top1['vec_return']:.2%}")
    print(f"  Sharpe: {score_top1['vec_sharpe_ratio']:.3f}")
    print(f"  MaxDD: {score_top1['vec_max_drawdown']:.1%}")
    print(f"  IC: {score_top1['mean_oos_ic']:.4f}")
    
    # 收益提升 (使用 OOS 收益如果可用)
    if has_oos_return:
        ic_return = ic_top1['mean_oos_return']
        score_return = score_top1['mean_oos_return']
        label = "OOS"
    else:
        ic_return = ic_top1['vec_return']
        score_return = score_top1['vec_return']
        label = "VEC"
    
    improvement = (score_return - ic_return) / max(abs(ic_return), 0.0001)
    print(f"\n📈 {label} 收益提升: {improvement*100:+.1f}%")


def save_results(df: pd.DataFrame, output_dir: Path, top_n: int = 100):
    """保存筛选结果"""
    sorted_df = df.sort_values("composite_score", ascending=False)
    
    # 保存 Top N
    top_df = sorted_df.head(top_n)
    top_df.to_parquet(output_dir / f"top{top_n}_by_composite.parquet", index=False)
    top_df.to_parquet(output_dir / f"top{top_n}_by_composite.parquet", index=False)
    
    # 保存完整结果
    sorted_df.to_parquet(output_dir / "all_combos_scored.parquet", index=False)
    sorted_df.to_parquet(output_dir / "all_combos_scored.parquet", index=False)
    
    print(f"\n✅ 结果已保存到: {output_dir}")
    print(f"   - top{top_n}_by_composite.csv/parquet")
    print(f"   - all_combos_scored.csv/parquet")
    
    return sorted_df


def main():
    print("=" * 80)
    print("策略筛选 v2.0 - 基于新开发思想")
    print("=" * 80)
    print()
    print("核心原则:")
    print("  1. 锁死交易规则: FREQ, POS, 不止损, 不 cash")
    print("  2. IC 只做门槛 (有无预测力)")
    print("  3. 最终排序: OOS收益 + Sharpe + 回撤 的综合得分")
    
    # 加载配置
    config_path = ROOT / "configs/combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # 添加默认的 selection 配置
    if "selection" not in config:
        config["selection"] = {
            "ic_threshold": 0.05,
            "positive_rate_threshold": 0.55,
            "composite_weights": {
                "return": 0.4,
                "sharpe": 0.3,
                "drawdown": 0.3,
            },
        }
    
    # 显示当前策略参数
    backtest_config = config.get("backtest", {})
    print(f"\n📋 策略参数 (已锁死):")
    print(f"   FREQ: {backtest_config.get('freq')}")
    print(f"   POS_SIZE: {backtest_config.get('pos_size')}")
    print(f"   止损: 禁用")
    print(f"   择时: {config.get('backtest', {}).get('timing', {}).get('type', 'light_timing')}")
    
    # 加载数据
    print("\n📂 加载数据...")
    try:
        wfo, vec, wfo_name, vec_name = load_latest_results()
        print(f"   WFO: {wfo_name} ({len(wfo)} 个组合)")
        print(f"   VEC: {vec_name} ({len(vec)} 个组合)")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("\n请先运行:")
        print("  1. uv run python src/etf_strategy/run_combo_wfo.py")
        print("  2. uv run python scripts/run_full_space_vec_backtest.py")
        return
    
    # 合并 WFO 和 VEC 结果
    merged = vec.merge(wfo, on="combo", how="left")
    print(f"\n📊 合并后: {len(merged)} 个组合")
    
    # Step 0 (P0): 过滤高风险因子组合
    safe_merged = filter_risky_combos(merged, RISKY_FACTORS)

    # Step 0.5: 近期震荡市适应性过滤
    regime_filtered = apply_regime_filter(safe_merged, config)
    
    # Step 1: IC 门槛过滤
    qualified = apply_ic_threshold(regime_filtered, config)
    
    # Step 2: 计算综合得分
    scored = compute_composite_score(qualified, config)
    
    # Step 3: 显示 Top 20
    sorted_df = display_top_strategies(scored, top_n=20)
    
    # 因子频率分析
    analyze_factor_frequency(scored, top_n=20)
    
    # 对比分析
    compare_with_ic_ranking(merged, scored)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT / "results" / f"selection_v2_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_results(scored, output_dir, top_n=100)
    
    # 生成报告
    report_path = output_dir / "SELECTION_REPORT.md"
    with open(report_path, "w") as f:
        f.write("# 策略筛选报告 v2.0\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 核心原则\n\n")
        f.write("1. **锁死交易规则**: FREQ, POS, 不止损, 不 cash\n")
        f.write("2. **IC 只做门槛**: 过滤无预测力的组合\n")
        f.write("3. **最终排序**: OOS收益 + Sharpe + 回撤 的综合得分\n\n")
        
        f.write("## 筛选参数\n\n")
        f.write(f"- IC 门槛: > {config['selection']['ic_threshold']}\n")
        f.write(f"- positive_rate 门槛: > {config['selection']['positive_rate_threshold']*100:.0f}%\n")
        f.write(f"- 综合得分权重: 收益{config['selection']['composite_weights']['return']*100:.0f}%, ")
        f.write(f"Sharpe{config['selection']['composite_weights']['sharpe']*100:.0f}%, ")
        f.write(f"回撤{config['selection']['composite_weights']['drawdown']*100:.0f}%\n\n")
        
        f.write("## 筛选结果\n\n")
        f.write(f"- 总组合数: {len(merged)}\n")
        f.write(f"- 通过门槛: {len(scored)}\n")
        f.write(f"- 过滤比例: {(len(merged) - len(scored)) / len(merged) * 100:.1f}%\n\n")
        
        f.write("## Top 10 策略\n\n")
        f.write("| 排名 | 收益 | Sharpe | MaxDD | IC | 组合 |\n")
        f.write("|------|------|--------|-------|-------|------|\n")
        
        for rank, (_, row) in enumerate(sorted_df.head(10).iterrows(), 1):
            f.write(f"| {rank} | {row['vec_return']:.2%} | {row['vec_sharpe_ratio']:.3f} | ")
            f.write(f"{row['vec_max_drawdown']:.1%} | {row['mean_oos_ic']:.4f} | {row['combo']} |\n")
    
    print(f"\n📝 报告已生成: {report_path}")
    
    print("\n" + "=" * 80)
    print("✅ 策略筛选完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
