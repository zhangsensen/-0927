"""
数据加载模块: 加载WFO特征和真实回测结果
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd


def load_wfo_features(wfo_dir: str | Path) -> pd.DataFrame:
    """
    加载WFO结果作为特征表
    
    Args:
        wfo_dir: WFO结果目录路径，如 'results/run_20251114_155420'
        
    Returns:
        DataFrame包含所有策略的WFO指标
        
    Raises:
        FileNotFoundError: 如果all_combos.parquet不存在
    """
    wfo_dir = Path(wfo_dir)
    all_combos_path = wfo_dir / "all_combos.parquet"
    
    if not all_combos_path.exists():
        raise FileNotFoundError(
            f"WFO结果文件不存在: {all_combos_path}\n"
            f"请确保目录包含 all_combos.parquet"
        )
    
    df = pd.read_parquet(all_combos_path)
    
    print(f"✓ 加载WFO特征: {len(df)} 个策略组合")
    print(f"  特征维度: {df.shape}")
    print(f"  唯一combo: {df['combo'].nunique()}")
    
    return df


def load_real_backtest_results(backtest_dir: str | Path) -> pd.DataFrame:
    """
    加载真实回测结果作为标签表
    
    Args:
        backtest_dir: 回测结果目录路径，如 'results_combo_wfo/20251114_155420_20251114_161032'
        
    Returns:
        DataFrame包含所有策略的真实回测表现
        
    Raises:
        FileNotFoundError: 如果回测CSV文件不存在
    """
    backtest_dir = Path(backtest_dir)
    
    # 查找回测CSV文件 (匹配 top*_profit_backtest_*.csv 模式)
    csv_files = list(backtest_dir.glob("top*_profit_backtest_*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(
            f"回测结果文件不存在: {backtest_dir}\n"
            f"请确保目录包含 top*_profit_backtest_*.csv 文件"
        )
    
    # 使用最大的文件 (可能有多个，选最全的)
    backtest_file = max(csv_files, key=lambda p: p.stat().st_size)
    
    df = pd.read_csv(backtest_file)
    
    print(f"✓ 加载真实回测结果: {len(df)} 个策略")
    print(f"  数据文件: {backtest_file.name}")
    print(f"  目标列: annual_ret_net (均值={df['annual_ret_net'].mean():.4f}, std={df['annual_ret_net'].std():.4f})")
    
    return df


def build_training_dataset(
    wfo_df: pd.DataFrame,
    real_df: pd.DataFrame,
    target_col: str = "annual_ret_net",
    secondary_target: Optional[str] = "sharpe_net"
) -> Tuple[pd.DataFrame, pd.Series, dict]:
    """
    构建训练数据集：合并WFO特征和真实回测标签
    
    Args:
        wfo_df: WFO特征DataFrame
        real_df: 真实回测结果DataFrame
        target_col: 主目标列名 (用于排序学习)
        secondary_target: 次要目标列名 (用于验证)
        
    Returns:
        Tuple of:
        - merged_df: 合并后的完整DataFrame (包含特征和标签)
        - y: 目标变量Series (target_col)
        - metadata: 元信息dict包含次要目标、combo等
        
    Raises:
        ValueError: 如果匹配率过低
    """
    # 按combo字段合并
    merged = pd.merge(
        wfo_df, 
        real_df, 
        on="combo", 
        how="inner",
        suffixes=("_wfo", "_real")
    )
    
    # 验证匹配情况
    coverage = len(merged) / len(wfo_df) * 100
    
    print(f"\n构建训练数据集:")
    print(f"  WFO策略数: {len(wfo_df)}")
    print(f"  真实回测策略数: {len(real_df)}")
    print(f"  匹配成功: {len(merged)} ({coverage:.1f}%)")
    
    if coverage < 95:
        print(f"  ⚠️  警告: 匹配率低于95%，可能存在数据不一致")
    
    if coverage < 50:
        raise ValueError(
            f"匹配率过低 ({coverage:.1f}%)，请检查数据源是否一致"
        )
    
    # 提取目标变量
    if target_col not in merged.columns:
        raise ValueError(f"目标列 '{target_col}' 不存在于合并后的数据中")
    
    y = merged[target_col].copy()
    
    # 构建元信息
    metadata = {
        "combo": merged["combo"].values,
        "target_col": target_col,
        target_col: y.values,
    }
    
    if secondary_target and secondary_target in merged.columns:
        metadata[secondary_target] = merged[secondary_target].values
        print(f"  次要目标: {secondary_target} (均值={merged[secondary_target].mean():.4f})")
    
    # 检查目标分布
    print(f"\n目标变量 '{target_col}' 统计:")
    print(f"  均值: {y.mean():.6f}")
    print(f"  标准差: {y.std():.6f}")
    print(f"  最小值: {y.min():.6f}")
    print(f"  最大值: {y.max():.6f}")
    print(f"  缺失值: {y.isna().sum()}")
    
    return merged, y, metadata


def find_latest_wfo_run(base_dir: str | Path = "results") -> Path:
    """
    自动查找最新的WFO运行目录
    
    Args:
        base_dir: 结果根目录
        
    Returns:
        最新运行目录的Path对象
        
    Raises:
        FileNotFoundError: 如果没有找到任何运行目录
    """
    base_dir = Path(base_dir)
    
    # 查找所有run_*目录
    run_dirs = sorted([d for d in base_dir.glob("run_*") if d.is_dir()], reverse=True)
    
    if not run_dirs:
        raise FileNotFoundError(f"未找到任何WFO运行目录: {base_dir}/run_*")
    
    latest = run_dirs[0]
    print(f"✓ 自动发现最新WFO运行: {latest.name}")
    
    return latest


def find_latest_backtest_run(base_dir: str | Path = "results_combo_wfo") -> Path:
    """
    自动查找最新的回测运行目录
    
    Args:
        base_dir: 回测结果根目录
        
    Returns:
        最新运行目录的Path对象
        
    Raises:
        FileNotFoundError: 如果没有找到任何运行目录
    """
    base_dir = Path(base_dir)
    
    # 查找所有时间戳目录
    backtest_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()], reverse=True)
    
    if not backtest_dirs:
        raise FileNotFoundError(f"未找到任何回测运行目录: {base_dir}")
    
    latest = backtest_dirs[0]
    print(f"✓ 自动发现最新回测运行: {latest.name}")
    
    return latest


def load_multi_source_data(
    config,  # DatasetConfig类型,但避免循环导入
    add_source_id: bool = True,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.Series, dict]:
    """
    从多个数据源加载并合并训练数据
    
    支持多换仓周期的WFO实验数据聚合,用于训练更泛化的排序模型
    
    Args:
        config: DatasetConfig配置对象(包含多个DataSource)
        add_source_id: 是否添加rebalance_days和source_label列
        verbose: 是否打印详细日志
        
    Returns:
        Tuple of:
        - merged_df: 合并后的完整DataFrame(包含所有数据源)
        - y: 目标变量Series
        - metadata: 元信息dict(包含各数据源统计)
        
    Raises:
        ValueError: 如果数据源列表为空或数据质量问题
        
    Example:
        >>> from ml_ranker.config import DatasetConfig
        >>> config = DatasetConfig.from_yaml("configs/ranking_datasets.yaml")
        >>> merged_df, y, metadata = load_multi_source_data(config)
        >>> print(f"总样本数: {len(merged_df)}, 数据源数: {metadata['n_sources']}")
    """
    if not config.datasets:
        raise ValueError("配置中的datasets列表为空")
    
    all_merged = []
    source_stats = []
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"📦 加载多数据源训练集 (共{len(config.datasets)}个)")
        print(f"{'='*80}\n")
    
    for idx, ds in enumerate(config.datasets, 1):
        if verbose:
            print(f"[{idx}/{len(config.datasets)}] {ds.display_name}")
            print(f"  WFO目录: {ds.wfo_dir}")
            print(f"  回测目录: {ds.real_dir}")
        
        try:
            # 加载单个数据源
            wfo_df = load_wfo_features(ds.wfo_dir)
            real_df = load_real_backtest_results(ds.real_dir)
            merged, _, _ = build_training_dataset(
                wfo_df, 
                real_df, 
                config.target_col,
                config.secondary_target
            )
            
            # 添加元数据列标记来源
            if add_source_id:
                merged['rebalance_days'] = ds.rebalance_days
                merged['source_label'] = ds.label or f"source_{idx}"
                merged['source_id'] = idx
            
            all_merged.append(merged)
            
            # 统计信息
            source_stats.append({
                'source_id': idx,
                'rebalance_days': ds.rebalance_days,
                'label': ds.label or f"数据源{idx}",
                'n_samples': len(merged),
                'target_mean': merged[config.target_col].mean(),
                'target_std': merged[config.target_col].std(),
                'wfo_dir': ds.wfo_dir,
                'real_dir': ds.real_dir
            })
            
            if verbose:
                print(f"  ✓ 加载 {len(merged)} 个样本")
                print(f"  目标均值: {merged[config.target_col].mean():.4f}\n")
        
        except Exception as e:
            print(f"  ❌ 加载失败: {e}")
            raise ValueError(f"数据源{idx}加载失败: {e}")
    
    # 合并所有数据源
    if verbose:
        print(f"{'='*80}")
        print("🔗 合并所有数据源")
        print(f"{'='*80}")
    
    combined_df = pd.concat(all_merged, ignore_index=True)
    y = combined_df[config.target_col].copy()
    
    # 构建元信息
    metadata = {
        'combo': combined_df['combo'].values,
        'target_col': config.target_col,
        config.target_col: y.values,
        'n_sources': len(config.datasets),
        'source_stats': source_stats,
        'rebalance_days': combined_df['rebalance_days'].values if add_source_id else None,
        'source_label': combined_df['source_label'].values if add_source_id else None,
        'source_id': combined_df['source_id'].values if add_source_id else None
    }
    
    if config.secondary_target and config.secondary_target in combined_df.columns:
        metadata[config.secondary_target] = combined_df[config.secondary_target].values
    
    if verbose:
        print(f"  ✓ 合并完成: {len(combined_df)} 个样本")
        print(f"\n来源分布:")
        for stat in source_stats:
            print(f"  - {stat['rebalance_days']:2d}天: {stat['n_samples']:5d} 样本 "
                  f"(均值={stat['target_mean']:7.4f}, std={stat['target_std']:6.4f})")
        
        print(f"\n目标变量 '{config.target_col}' 统计:")
        print(f"  总样本数: {len(combined_df)}")
        print(f"  均值: {y.mean():.6f}")
        print(f"  标准差: {y.std():.6f}")
        print(f"  最小值: {y.min():.6f}")
        print(f"  最大值: {y.max():.6f}")
        print(f"  缺失值: {y.isna().sum()}")
        
        if add_source_id:
            print(f"\n换仓周期分布:")
            rebal_counts = combined_df['rebalance_days'].value_counts().sort_index()
            for days, count in rebal_counts.items():
                pct = count / len(combined_df) * 100
                print(f"  {days:2d}天: {count:5d} ({pct:5.1f}%)")
        
        print(f"{'='*80}\n")
    
    return combined_df, y, metadata
