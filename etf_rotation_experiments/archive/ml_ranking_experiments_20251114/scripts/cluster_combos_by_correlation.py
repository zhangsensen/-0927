#!/usr/bin/env python3
"""
基于日收益率相关性对组合进行层次聚类

输入: results/combo_daily_nav_top2000.pkl
输出: results/combo_clusters_top20.json (簇ID、代表组合、簇内平均相关性)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def main():
    print("=" * 100)
    print("组合相关性聚类分析")
    print("=" * 100)
    print()

    # 1. 加载净值数据
    here = Path(__file__).resolve()
    nav_path = here.parent.parent / "results" / "combo_daily_nav_top2000.pkl"
    
    if not nav_path.exists():
        raise FileNotFoundError(f"未找到净值文件: {nav_path}")
    
    print(f"加载净值数据: {nav_path}")
    nav_df = pd.read_pickle(nav_path)
    print(f"✓ 净值数据: {nav_df.shape[0]}天 × {nav_df.shape[1]}个组合")
    print()

    # 2. 计算日收益率
    print("计算日收益率...")
    returns_df = nav_df.pct_change().iloc[1:]  # 去掉第一行NaN
    print(f"✓ 收益率数据: {returns_df.shape[0]}天 × {returns_df.shape[1]}个组合")
    
    # 去除收益率全为NaN的组合
    valid_combos = returns_df.columns[returns_df.notna().sum() > 0]
    returns_df = returns_df[valid_combos]
    print(f"✓ 有效组合: {len(valid_combos)}个")
    print()

    # 3. 计算相关性矩阵
    print("计算相关性矩阵...")
    corr_matrix = returns_df.corr(method='pearson')
    print(f"✓ 相关性矩阵: {corr_matrix.shape}")
    print(f"  - 平均相关性: {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean():.4f}")
    print(f"  - 相关性范围: [{corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].min():.4f}, {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max():.4f}]")
    print()

    # 4. 转换为距离矩阵并进行层次聚类
    print("执行层次聚类...")
    # 距离 = 1 - 相关性（相关性越高，距离越小）
    distance_matrix = 1 - corr_matrix.values
    np.fill_diagonal(distance_matrix, 0)  # 对角线设为0
    
    # 转换为压缩距离矩阵（上三角）
    condensed_dist = squareform(distance_matrix, checks=False)
    
    # Ward linkage（最小化簇内方差）
    linkage_matrix = linkage(condensed_dist, method='ward')
    print("✓ 层次聚类完成")
    print()

    # 5. 切分为指定数量的簇
    target_clusters = 20
    print(f"切分为 {target_clusters} 个簇...")
    cluster_labels = fcluster(linkage_matrix, target_clusters, criterion='maxclust')
    print(f"✓ 簇分配完成")
    print()

    # 6. 为每个簇选择代表组合（Sharpe最高的）
    print("选择代表组合...")
    
    # 读取原始回测结果以获取Sharpe
    results_dir = here.parent.parent / "results_combo_wfo"
    csv_files = sorted(results_dir.glob("*/top2000_profit_backtest_slip0bps_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"未找到Top2000回测CSV文件")
    
    backtest_df = pd.read_csv(csv_files[-1])
    print(f"✓ 读取回测结果: {len(backtest_df)}个组合")
    
    # 构建combo_id到Sharpe的映射
    combo_to_sharpe = {}
    for idx, row in backtest_df.iterrows():
        combo_id = f"combo_{idx}"
        if combo_id in valid_combos:
            combo_to_sharpe[combo_id] = row['sharpe_net']
    
    # 为每个簇选择代表
    cluster_info = {}
    for cluster_id in range(1, target_clusters + 1):
        # 找到该簇的所有组合
        cluster_mask = cluster_labels == cluster_id
        cluster_combos = [valid_combos[i] for i in range(len(valid_combos)) if cluster_mask[i]]
        
        if not cluster_combos:
            continue
        
        # 选择Sharpe最高的作为代表
        cluster_sharpes = {combo: combo_to_sharpe.get(combo, 0) for combo in cluster_combos}
        representative = max(cluster_sharpes, key=cluster_sharpes.get)
        
        # 计算簇内平均相关性
        cluster_indices = [i for i in range(len(valid_combos)) if cluster_mask[i]]
        if len(cluster_indices) > 1:
            cluster_corr_values = []
            for i in range(len(cluster_indices)):
                for j in range(i + 1, len(cluster_indices)):
                    idx_i = cluster_indices[i]
                    idx_j = cluster_indices[j]
                    cluster_corr_values.append(corr_matrix.iloc[idx_i, idx_j])
            avg_intra_corr = np.mean(cluster_corr_values)
        else:
            avg_intra_corr = 1.0
        
        cluster_info[f"cluster_{cluster_id}"] = {
            "cluster_id": int(cluster_id),
            "n_combos": len(cluster_combos),
            "representative": representative,
            "representative_sharpe": float(combo_to_sharpe.get(representative, 0)),
            "avg_intra_correlation": float(avg_intra_corr),
            "all_combos": cluster_combos,
        }
    
    print(f"✓ 选择了 {len(cluster_info)} 个簇的代表组合")
    print()

    # 7. 计算代表组合之间的相关性
    print("计算代表组合间相关性...")
    representatives = [info["representative"] for info in cluster_info.values()]
    rep_corr_matrix = returns_df[representatives].corr(method='pearson')
    avg_inter_corr = rep_corr_matrix.values[np.triu_indices_from(rep_corr_matrix.values, k=1)].mean()
    print(f"✓ 代表组合间平均相关性: {avg_inter_corr:.4f}")
    print()

    # 8. 保存结果
    output_path = here.parent.parent / "results" / "combo_clusters_top20.json"
    
    summary = {
        "n_total_combos": len(valid_combos),
        "n_clusters": len(cluster_info),
        "avg_intra_correlation": float(np.mean([info["avg_intra_correlation"] for info in cluster_info.values()])),
        "avg_inter_correlation": float(avg_inter_corr),
        "clusters": cluster_info,
    }
    
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 已保存: {output_path}")
    print()

    # 9. 打印摘要
    print("=" * 100)
    print("聚类摘要")
    print("=" * 100)
    print(f"总组合数: {len(valid_combos)}")
    print(f"簇数量: {len(cluster_info)}")
    print(f"簇内平均相关性: {summary['avg_intra_correlation']:.4f}")
    print(f"簇间平均相关性: {avg_inter_corr:.4f}")
    print(f"相关性降低: {(summary['avg_intra_correlation'] - avg_inter_corr) / summary['avg_intra_correlation']:.2%}")
    print()
    
    print("Top 10 代表组合（按Sharpe排序）:")
    sorted_clusters = sorted(cluster_info.values(), key=lambda x: x["representative_sharpe"], reverse=True)
    for i, info in enumerate(sorted_clusters[:10], 1):
        print(f"{i:>2}. {info['representative']} | Sharpe={info['representative_sharpe']:.3f} | "
              f"簇大小={info['n_combos']} | 簇内相关={info['avg_intra_correlation']:.3f}")
    print()
    
    print("=" * 100)
    print("✅ 完成！")
    print("=" * 100)


if __name__ == "__main__":
    main()

