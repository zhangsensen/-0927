"""
特征工程模块: 从WFO数据提取和构建模型特征
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats


def extract_scalar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    提取WFO的标量特征 (基础数值列)
    
    Args:
        df: 包含WFO结果的DataFrame
        
    Returns:
        只包含标量特征的DataFrame
    """
    # 标量特征列表 (排除object类型的序列特征)
    scalar_cols = [
        "combo_size",
        "mean_oos_ic",
        "oos_ic_std",
        "oos_ic_ir",
        "positive_rate",
        "best_rebalance_freq",
        "stability_score",
        "mean_oos_sharpe",
        "oos_sharpe_std",
        "mean_oos_sample_count",
        "oos_compound_sharpe",
        "oos_compound_mean",
        "oos_compound_std",
        "oos_compound_sample_count",
        "p_value",
        "q_value",
        "oos_sharpe_proxy",
    ]
    
    # 筛选存在的列
    available_cols = [col for col in scalar_cols if col in df.columns]
    
    features = df[available_cols].copy()
    
    # 处理布尔类型
    if "is_significant" in df.columns:
        features["is_significant"] = df["is_significant"].astype(int)
    
    print(f"  提取标量特征: {len(available_cols)} 列")
    
    return features


def expand_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    从序列特征中提取统计量
    
    Args:
        df: 包含WFO结果的DataFrame (含序列列如oos_ic_list)
        
    Returns:
        包含序列统计特征的DataFrame
    """
    seq_features = pd.DataFrame(index=df.index)
    
    # 1. 从 oos_ic_list 提取特征
    if "oos_ic_list" in df.columns:
        ic_arr = np.vstack(df["oos_ic_list"].values)  # shape: (n_samples, n_windows)
        
        seq_features["ic_seq_mean"] = np.nanmean(ic_arr, axis=1)
        seq_features["ic_seq_std"] = np.nanstd(ic_arr, axis=1)
        seq_features["ic_seq_min"] = np.nanmin(ic_arr, axis=1)
        seq_features["ic_seq_max"] = np.nanmax(ic_arr, axis=1)
        seq_features["ic_seq_median"] = np.nanmedian(ic_arr, axis=1)
        
        # 正值比例
        seq_features["ic_positive_ratio"] = (ic_arr > 0).mean(axis=1)
        
        # 趋势 (线性回归斜率)
        trends = []
        for i in range(len(ic_arr)):
            valid_mask = ~np.isnan(ic_arr[i])
            if valid_mask.sum() > 2:
                x = np.arange(valid_mask.sum())
                y = ic_arr[i][valid_mask]
                slope, _, _, _, _ = stats.linregress(x, y)
                trends.append(slope)
            else:
                trends.append(0.0)
        seq_features["ic_seq_trend"] = trends
        
        # CV (变异系数)
        seq_features["ic_seq_cv"] = seq_features["ic_seq_std"] / (seq_features["ic_seq_mean"].abs() + 1e-8)
    
    # 2. 从 oos_sharpe_list 提取特征
    if "oos_sharpe_list" in df.columns:
        sharpe_arr = np.vstack(df["oos_sharpe_list"].values)
        
        seq_features["sharpe_seq_mean"] = np.nanmean(sharpe_arr, axis=1)
        seq_features["sharpe_seq_std"] = np.nanstd(sharpe_arr, axis=1)
        seq_features["sharpe_seq_min"] = np.nanmin(sharpe_arr, axis=1)
        seq_features["sharpe_seq_max"] = np.nanmax(sharpe_arr, axis=1)
        seq_features["sharpe_seq_median"] = np.nanmedian(sharpe_arr, axis=1)
        
        # 正值比例
        seq_features["sharpe_positive_ratio"] = (sharpe_arr > 0).mean(axis=1)
        
        # CV
        seq_features["sharpe_seq_cv"] = seq_features["sharpe_seq_std"] / (seq_features["sharpe_seq_mean"].abs() + 1e-8)
    
    # 3. 从 oos_ir_list 提取特征
    if "oos_ir_list" in df.columns:
        ir_arr = np.vstack(df["oos_ir_list"].values)
        
        seq_features["ir_seq_mean"] = np.nanmean(ir_arr, axis=1)
        seq_features["ir_seq_std"] = np.nanstd(ir_arr, axis=1)
        seq_features["ir_positive_ratio"] = (ir_arr > 0).mean(axis=1)
    
    # 4. 从 positive_rate_list 提取特征
    if "positive_rate_list" in df.columns:
        pr_arr = np.vstack(df["positive_rate_list"].values)
        
        seq_features["posrate_seq_mean"] = np.nanmean(pr_arr, axis=1)
        seq_features["posrate_seq_std"] = np.nanstd(pr_arr, axis=1)
        seq_features["posrate_seq_min"] = np.nanmin(pr_arr, axis=1)
    
    print(f"  提取序列特征: {len(seq_features.columns)} 列")
    
    return seq_features


def create_cross_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    创建交叉特征 (特征间的乘积、比率等)
    
    Args:
        df: 包含基础特征的DataFrame
        
    Returns:
        包含交叉特征的DataFrame
    """
    cross_features = pd.DataFrame(index=df.index)
    
    # IC × Sharpe
    if "mean_oos_ic" in df.columns and "mean_oos_sharpe" in df.columns:
        cross_features["ic_x_sharpe"] = df["mean_oos_ic"] * df["mean_oos_sharpe"]
    
    # Stability × Positive Rate
    if "stability_score" in df.columns and "positive_rate" in df.columns:
        cross_features["stability_x_posrate"] = df["stability_score"] * df["positive_rate"]
    
    # IC IR × Sharpe Proxy
    if "oos_ic_ir" in df.columns and "oos_sharpe_proxy" in df.columns:
        cross_features["ic_ir_x_sharpe_proxy"] = df["oos_ic_ir"] * df["oos_sharpe_proxy"]
    
    # Compound Sharpe × IC
    if "oos_compound_sharpe" in df.columns and "mean_oos_ic" in df.columns:
        cross_features["compound_sharpe_x_ic"] = df["oos_compound_sharpe"] * df["mean_oos_ic"]
    
    # IC std / IC mean (相对波动)
    if "oos_ic_std" in df.columns and "mean_oos_ic" in df.columns:
        cross_features["ic_relative_std"] = df["oos_ic_std"] / (df["mean_oos_ic"].abs() + 1e-8)
    
    # Sharpe std / Sharpe mean
    if "oos_sharpe_std" in df.columns and "mean_oos_sharpe" in df.columns:
        cross_features["sharpe_relative_std"] = df["oos_sharpe_std"] / (df["mean_oos_sharpe"].abs() + 1e-8)
    
    print(f"  创建交叉特征: {len(cross_features.columns)} 列")
    
    return cross_features


def parse_combo_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    从combo字符串解析元信息特征
    
    Args:
        df: 包含combo列的DataFrame
        
    Returns:
        包含combo元特征的DataFrame
    """
    meta_features = pd.DataFrame(index=df.index)
    
    if "combo" not in df.columns:
        return meta_features
    
    # 因子数量 (通过'+'分隔符计数)
    meta_features["n_factors"] = df["combo"].str.count(r"\+") + 1
    
    # 是否包含特定因子类型 (可扩展)
    # 示例: 检查是否包含动量因子
    meta_features["has_momentum"] = df["combo"].str.contains("MOM_|SLOPE_", regex=True).astype(int)
    
    # 检查是否包含波动率因子
    meta_features["has_volatility"] = df["combo"].str.contains("VOL_|MAX_DD", regex=True).astype(int)
    
    # 检查是否包含技术指标
    meta_features["has_technical"] = df["combo"].str.contains("RSI|ADX|CMF|OBV|VORTEX", regex=True).astype(int)
    
    print(f"  解析combo特征: {len(meta_features.columns)} 列")
    
    return meta_features


def build_feature_matrix(
    merged_df: pd.DataFrame,
    include_cross_features: bool = True,
    include_combo_features: bool = False
) -> pd.DataFrame:
    """
    构建完整的特征矩阵
    
    Args:
        merged_df: 合并后的WFO和回测数据
        include_cross_features: 是否包含交叉特征
        include_combo_features: 是否解析combo字符串特征
        
    Returns:
        特征矩阵DataFrame (只包含数值型特征)
    """
    print("\n构建特征矩阵:")
    
    # 1. 提取标量特征
    scalar_feat = extract_scalar_features(merged_df)
    
    # 2. 展开序列特征
    seq_feat = expand_sequence_features(merged_df)
    
    # 3. 创建交叉特征
    feature_dfs = [scalar_feat, seq_feat]
    
    if include_cross_features:
        cross_feat = create_cross_features(
            pd.concat([scalar_feat, seq_feat], axis=1)
        )
        feature_dfs.append(cross_feat)
    
    # 4. 解析combo特征 (可选)
    if include_combo_features:
        combo_feat = parse_combo_features(merged_df)
        feature_dfs.append(combo_feat)
    
    # 合并所有特征
    X = pd.concat(feature_dfs, axis=1)
    
    # 处理缺失值
    n_missing_before = X.isna().sum().sum()
    if n_missing_before > 0:
        print(f"  ⚠️  发现缺失值: {n_missing_before} 个")
        # 用中位数填充
        X = X.fillna(X.median())
        print(f"  已用中位数填充")
    
    # 处理无穷值
    n_inf = np.isinf(X.values).sum()
    if n_inf > 0:
        print(f"  ⚠️  发现无穷值: {n_inf} 个")
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    
    print(f"\n✓ 特征矩阵构建完成:")
    print(f"  样本数: {len(X)}")
    print(f"  特征数: {len(X.columns)}")
    print(f"  特征列表 (前20): {list(X.columns[:20])}")
    
    return X


def get_feature_importance_names() -> List[str]:
    """返回特征的可读名称映射"""
    return {
        "mean_oos_ic": "平均OOS IC",
        "oos_ic_std": "OOS IC标准差",
        "oos_ic_ir": "OOS IC信息比率",
        "positive_rate": "正IC占比",
        "stability_score": "稳定性评分",
        "mean_oos_sharpe": "平均OOS Sharpe",
        "oos_sharpe_std": "OOS Sharpe标准差",
        "oos_sharpe_proxy": "OOS Sharpe代理",
        "ic_seq_trend": "IC趋势斜率",
        "ic_positive_ratio": "IC正值比例",
        "sharpe_positive_ratio": "Sharpe正值比例",
        "ic_x_sharpe": "IC×Sharpe交叉",
        "stability_x_posrate": "稳定性×正率",
    }
