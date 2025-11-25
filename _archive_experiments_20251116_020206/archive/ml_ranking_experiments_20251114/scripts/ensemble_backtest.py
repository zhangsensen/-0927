#!/usr/bin/env python3
"""
组合层回测器：多组合集成

输入: 
  - results/combo_daily_nav_top2000.pkl (所有组合的净值)
  - results/combo_clusters_top20.json (聚类结果)
输出:
  - results/ensemble_backtest_results.json (各种权重方案的回测结果)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def calculate_metrics(nav_series: np.ndarray, initial_capital: float = 1000000.0) -> Dict:
    """计算回测指标"""
    if len(nav_series) == 0:
        return {}
    
    # 基本收益指标
    final_value = float(nav_series[-1])
    total_ret = final_value / initial_capital - 1
    
    # 日收益率
    daily_returns = nav_series[1:] / nav_series[:-1] - 1
    days = len(daily_returns)
    
    # 年化收益
    annual_ret = (1 + total_ret) ** (252 / days) - 1 if days > 0 else 0
    
    # 波动率
    vol = float(np.std(daily_returns)) * np.sqrt(252)
    
    # Sharpe
    sharpe = annual_ret / vol if vol > 0 else 0
    
    # 最大回撤
    cummax = np.maximum.accumulate(nav_series)
    dd = (nav_series - cummax) / cummax
    max_dd = float(np.min(dd))
    
    # Sortino（下行波动）
    downside_returns = daily_returns[daily_returns < 0]
    downside_vol = float(np.std(downside_returns)) * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino = annual_ret / downside_vol if downside_vol > 0 else 0
    
    # Calmar
    calmar = annual_ret / abs(max_dd) if max_dd < 0 else 0
    
    # 胜率
    win_rate = float(np.sum(daily_returns > 0)) / len(daily_returns) if len(daily_returns) > 0 else 0
    
    return {
        "final_value": final_value,
        "total_ret": total_ret,
        "annual_ret": annual_ret,
        "vol": vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "sortino": sortino,
        "calmar": calmar,
        "win_rate": win_rate,
        "n_days": days,
    }


def equal_weight_ensemble(nav_df: pd.DataFrame, combo_list: List[str]) -> np.ndarray:
    """等权集成"""
    selected_navs = nav_df[combo_list].values
    ensemble_nav = np.mean(selected_navs, axis=1)
    return ensemble_nav


def risk_parity_ensemble(nav_df: pd.DataFrame, combo_list: List[str], lookback: int = 60) -> np.ndarray:
    """风险平价集成（按历史波动率倒数加权）"""
    selected_navs = nav_df[combo_list].values
    n_days, n_combos = selected_navs.shape
    
    # 计算每个组合的日收益率
    returns = selected_navs[1:] / selected_navs[:-1] - 1
    
    # 初始化集成净值
    ensemble_nav = np.zeros(n_days)
    ensemble_nav[0] = selected_navs[0, 0]  # 初始值
    
    # 逐日计算权重并更新净值
    for t in range(1, n_days):
        # 计算历史波动率（使用lookback窗口）
        start_idx = max(0, t - lookback)
        hist_returns = returns[start_idx:t, :]
        
        if len(hist_returns) > 0:
            vols = np.std(hist_returns, axis=0)
            # 避免除零
            vols = np.where(vols > 1e-8, vols, 1e-8)
            # 倒数权重
            inv_vols = 1.0 / vols
            weights = inv_vols / np.sum(inv_vols)
        else:
            # 如果没有历史数据，使用等权
            weights = np.ones(n_combos) / n_combos
        
        # 计算当日收益
        daily_ret = np.sum(weights * returns[t-1, :])
        ensemble_nav[t] = ensemble_nav[t-1] * (1 + daily_ret)
    
    return ensemble_nav


def sharpe_weighted_ensemble(nav_df: pd.DataFrame, combo_list: List[str], sharpe_dict: Dict[str, float]) -> np.ndarray:
    """Sharpe加权集成（Sharpe^2归一化）"""
    selected_navs = nav_df[combo_list].values
    
    # 获取Sharpe权重
    sharpes = np.array([sharpe_dict.get(combo, 0) for combo in combo_list])
    # 使用Sharpe^2作为权重
    sharpe_sq = sharpes ** 2
    weights = sharpe_sq / np.sum(sharpe_sq) if np.sum(sharpe_sq) > 0 else np.ones(len(combo_list)) / len(combo_list)
    
    # 加权平均
    ensemble_nav = np.dot(selected_navs, weights)
    return ensemble_nav


def volatility_target_ensemble(
    nav_df: pd.DataFrame, 
    combo_list: List[str], 
    base_weights: np.ndarray,
    target_vol: float = 0.12,
    lookback: int = 60
) -> np.ndarray:
    """波动目标集成（在基础权重上叠加波动控制）"""
    selected_navs = nav_df[combo_list].values
    n_days, n_combos = selected_navs.shape
    
    # 计算每个组合的日收益率
    returns = selected_navs[1:] / selected_navs[:-1] - 1
    
    # 初始化集成净值
    ensemble_nav = np.zeros(n_days)
    ensemble_nav[0] = selected_navs[0, 0]
    
    # 逐日计算杠杆并更新净值
    for t in range(1, n_days):
        # 计算历史波动率
        start_idx = max(0, t - lookback)
        hist_returns = returns[start_idx:t, :]
        
        if len(hist_returns) > 0:
            # 计算组合层的历史波动率
            portfolio_returns = np.dot(hist_returns, base_weights)
            realized_vol = np.std(portfolio_returns) * np.sqrt(252)
            
            # 计算杠杆（仅缩不放）
            leverage = min(1.0, target_vol / realized_vol) if realized_vol > 1e-8 else 1.0
        else:
            leverage = 1.0
        
        # 计算当日收益（应用杠杆）
        daily_ret = np.sum(base_weights * returns[t-1, :]) * leverage
        ensemble_nav[t] = ensemble_nav[t-1] * (1 + daily_ret)
    
    return ensemble_nav


def main():
    print("=" * 100)
    print("组合层回测 - 多组合集成")
    print("=" * 100)
    print()

    # 1. 加载数据
    here = Path(__file__).resolve()
    nav_path = here.parent.parent / "results" / "combo_daily_nav_top2000.pkl"
    cluster_path = here.parent.parent / "results" / "combo_clusters_top20.json"
    
    print("加载数据...")
    nav_df = pd.read_pickle(nav_path)
    with open(cluster_path, "r") as f:
        cluster_data = json.load(f)
    
    print(f"✓ 净值数据: {nav_df.shape[0]}天 × {nav_df.shape[1]}个组合")
    print(f"✓ 聚类数据: {cluster_data['n_clusters']}个簇")
    print()

    # 2. 提取代表组合列表和Sharpe
    representatives = []
    sharpe_dict = {}
    for cluster_info in cluster_data["clusters"].values():
        rep = cluster_info["representative"]
        representatives.append(rep)
        sharpe_dict[rep] = cluster_info["representative_sharpe"]
    
    print(f"✓ 代表组合数: {len(representatives)}")
    print(f"✓ 平均Sharpe: {np.mean(list(sharpe_dict.values())):.3f}")
    print()

    # 3. 读取原始回测结果以获取Top1基准
    results_dir = here.parent.parent / "results_combo_wfo"
    csv_files = sorted(results_dir.glob("*/top2000_profit_backtest_slip0bps_*.csv"))
    if not csv_files:
        raise FileNotFoundError("未找到Top2000回测CSV文件")
    
    backtest_df = pd.read_csv(csv_files[-1])
    top1_combo = f"combo_0"
    top1_nav = nav_df[top1_combo].values
    
    print(f"✓ Top1基准: {top1_combo}")
    print()

    # 4. 运行各种集成方案
    print("运行集成方案...")
    results = {}
    
    # 4.1 等权集成
    print("  - 等权集成...")
    eq_nav = equal_weight_ensemble(nav_df, representatives)
    results["equal_weight"] = {
        "name": "等权集成",
        "n_combos": len(representatives),
        "metrics": calculate_metrics(eq_nav),
        "nav": eq_nav.tolist(),
    }
    
    # 4.2 风险平价集成
    print("  - 风险平价集成...")
    rp_nav = risk_parity_ensemble(nav_df, representatives, lookback=60)
    results["risk_parity"] = {
        "name": "风险平价集成",
        "n_combos": len(representatives),
        "metrics": calculate_metrics(rp_nav),
        "nav": rp_nav.tolist(),
    }
    
    # 4.3 Sharpe加权集成
    print("  - Sharpe加权集成...")
    sharpe_nav = sharpe_weighted_ensemble(nav_df, representatives, sharpe_dict)
    results["sharpe_weighted"] = {
        "name": "Sharpe加权集成",
        "n_combos": len(representatives),
        "metrics": calculate_metrics(sharpe_nav),
        "nav": sharpe_nav.tolist(),
    }
    
    # 4.4 波动目标集成（基于等权）
    print("  - 波动目标集成（12%）...")
    eq_weights = np.ones(len(representatives)) / len(representatives)
    vol_target_nav = volatility_target_ensemble(
        nav_df, representatives, eq_weights, target_vol=0.12, lookback=60
    )
    results["volatility_target_12"] = {
        "name": "波动目标集成（12%）",
        "n_combos": len(representatives),
        "metrics": calculate_metrics(vol_target_nav),
        "nav": vol_target_nav.tolist(),
    }
    
    # 4.5 波动目标集成（基于等权，15%）
    print("  - 波动目标集成（15%）...")
    vol_target_nav_15 = volatility_target_ensemble(
        nav_df, representatives, eq_weights, target_vol=0.15, lookback=60
    )
    results["volatility_target_15"] = {
        "name": "波动目标集成（15%）",
        "n_combos": len(representatives),
        "metrics": calculate_metrics(vol_target_nav_15),
        "nav": vol_target_nav_15.tolist(),
    }
    
    # 4.6 Top1基准
    print("  - Top1基准...")
    results["top1_baseline"] = {
        "name": "Top1基准",
        "n_combos": 1,
        "metrics": calculate_metrics(top1_nav),
        "nav": top1_nav.tolist(),
    }
    
    print("✓ 集成方案完成")
    print()

    # 5. 保存结果
    output_path = here.parent.parent / "results" / "ensemble_backtest_results.json"
    
    # 准备输出（去掉nav以减小文件大小，单独保存）
    output_data = {
        "n_representatives": len(representatives),
        "representatives": representatives,
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "nav"} for k, v in results.items()},
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 已保存: {output_path}")
    
    # 保存净值序列
    nav_output_path = here.parent.parent / "results" / "ensemble_nav_series.pkl"
    nav_series_df = pd.DataFrame({
        k: v["nav"] for k, v in results.items()
    }, index=nav_df.index)
    nav_series_df.to_pickle(nav_output_path)
    print(f"✓ 已保存净值序列: {nav_output_path}")
    print()

    # 6. 打印对比结果
    print("=" * 100)
    print("回测结果对比")
    print("=" * 100)
    print(f"{'方案':<20} {'年化收益':>10} {'Sharpe':>8} {'最大回撤':>10} {'Sortino':>8} {'Calmar':>8} {'胜率':>8}")
    print("-" * 100)
    
    # 按Sharpe排序
    sorted_results = sorted(results.items(), key=lambda x: x[1]["metrics"].get("sharpe", 0), reverse=True)
    
    for key, data in sorted_results:
        m = data["metrics"]
        print(f"{data['name']:<20} {m['annual_ret']:>9.2%} {m['sharpe']:>8.3f} {m['max_dd']:>9.2%} "
              f"{m['sortino']:>8.3f} {m['calmar']:>8.3f} {m['win_rate']:>7.2%}")
    
    print()
    
    # 7. 计算改进幅度
    print("=" * 100)
    print("相对Top1基准的改进")
    print("=" * 100)
    
    top1_metrics = results["top1_baseline"]["metrics"]
    
    for key, data in sorted_results:
        if key == "top1_baseline":
            continue
        m = data["metrics"]
        
        annual_ret_improve = m["annual_ret"] - top1_metrics["annual_ret"]
        sharpe_improve = m["sharpe"] - top1_metrics["sharpe"]
        max_dd_improve = m["max_dd"] - top1_metrics["max_dd"]  # 负数表示回撤更小
        
        print(f"\n{data['name']}:")
        print(f"  年化收益: {m['annual_ret']:.2%} (vs {top1_metrics['annual_ret']:.2%}, {annual_ret_improve:+.2%})")
        print(f"  Sharpe: {m['sharpe']:.3f} (vs {top1_metrics['sharpe']:.3f}, {sharpe_improve:+.3f})")
        print(f"  最大回撤: {m['max_dd']:.2%} (vs {top1_metrics['max_dd']:.2%}, {max_dd_improve:+.2%})")
        print(f"  Sortino: {m['sortino']:.3f} (vs {top1_metrics['sortino']:.3f})")
        print(f"  Calmar: {m['calmar']:.3f} (vs {top1_metrics['calmar']:.3f})")
    
    print()
    print("=" * 100)
    print("✅ 完成！")
    print("=" * 100)


if __name__ == "__main__":
    main()

