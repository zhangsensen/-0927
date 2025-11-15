"""
Top-200 筛选核心逻辑

所有函数都是纯函数，便于测试和复用。
"""

import logging
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# ========== 配置常量 ==========

DEFAULT_CONFIG = {
    # 质量过滤阈值
    'quality_filter': {
        'standard': {
            'min_sharpe_net': 0.95,
            'max_dd_net': -0.28,
            'min_annual_ret_net': 0.12,
            'max_turnover': 1.6,
        },
        'relaxed': {
            'min_sharpe_net': 0.90,
            'max_dd_net': -0.30,
            'min_annual_ret_net': 0.10,
            'max_turnover': 1.8,
        },
        'tightened': {
            'min_sharpe_net': 1.0,
            'max_turnover': 1.4,
        },
        'adaptive_thresholds': {
            'min_samples': 300,
            'max_samples': 1500,
        }
    },
    
    # 因子分类关键词
    'factor_categories': {
        'trend': ['MOM', 'SLOPE', 'VORTEX', 'ADX', 'TREND', 'ROC'],
        'vol': ['VOL_RATIO', 'MAX_DD', 'RET_VOL', 'SHARPE', 'VAR', 'STD'],
        'volume_price': ['OBV', 'PV_CORR', 'CMF', 'MFI'],
        'relative': ['RSI', 'PRICE_POSITION', 'RELATIVE', 'CORRELATION', 'BETA'],
    },
    
    # 综合评分权重
    'scoring_weights': {
        'annual_ret_net': 0.25,
        'sharpe_net': 0.30,
        'calmar_ratio': 0.20,
        'win_rate': 0.15,
        'max_dd_net': -0.10,  # 负权重（回撤越小越好）
    },
    
    # 桶配额规则
    'bucket_quotas': {
        'size_thresholds': [100, 50, 20],  # >= 100, 50-99, 20-49, <20
        'quotas': [18, 12, 8, 5],
        'min_quota': 3,  # 每个桶最少分配
    },
    
    # combo_size 分布目标
    'combo_size_targets': {
        3: {'min': 40, 'max': 60},   # 20-30%
        4: {'min': 60, 'max': 80},   # 30-40%
        5: {'min': 70, 'max': 90},   # 35-45%
    },
    
    # 高换手控制
    'turnover_control': {
        'threshold': 1.4,
        'max_ratio': 0.3,  # 最多 30%
    },
    
    # 总配额
    'total_quota': 200,
}

# 关键字段（缺失时剔除）
REQUIRED_FIELDS = [
    'combo', 
    'combo_size', 
    'annual_ret_net', 
    'sharpe_net', 
    'max_dd_net', 
    'avg_turnover'
]

# 非关键字段的默认值
DEFAULT_VALUES = {
    'calmar_ratio': 0.0,
    'win_rate': 0.5,
    'sortino_ratio': 0.0,
}


# ========== 工具函数 ==========

def deep_merge_config(base: dict, override: dict) -> dict:
    """深度合并配置字典"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_config(result[key], value)
        else:
            result[key] = value
    return result


# ========== 数据预处理 ==========

def validate_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    验证并清理数据：检查关键字段，剔除缺失行
    
    参数:
        df: 原始 DataFrame
    
    返回:
        清理后的 DataFrame
    """
    initial_count = len(df)
    
    # 检查关键字段是否存在
    missing_fields = [f for f in REQUIRED_FIELDS if f not in df.columns]
    if missing_fields:
        raise ValueError(f"数据缺少关键字段: {missing_fields}")
    
    # 剔除关键字段有缺失值的行
    df_clean = df.dropna(subset=REQUIRED_FIELDS).copy()
    
    dropped = initial_count - len(df_clean)
    if dropped > 0:
        logging.warning(f"剔除了 {dropped} 行关键字段缺失的数据")
    
    logging.info(f"数据清理完成: {len(df_clean)} 行有效数据")
    
    return df_clean


def add_missing_fields_with_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """
    为非关键字段填充默认值
    
    参数:
        df: DataFrame
    
    返回:
        填充后的 DataFrame
    """
    df = df.copy()
    
    for field, default_value in DEFAULT_VALUES.items():
        if field in df.columns:
            # 填充缺失值
            missing_count = df[field].isna().sum()
            if missing_count > 0:
                df[field] = df[field].fillna(default_value)
                logging.info(f"字段 '{field}' 的 {missing_count} 个缺失值已填充为 {default_value}")
        else:
            # 字段不存在，创建并填充
            df[field] = default_value
            logging.info(f"字段 '{field}' 不存在，已创建并填充为 {default_value}")
    
    return df


# ========== 步骤 1：质量过滤 ==========

def apply_quality_filter(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """
    应用质量过滤条件
    
    参数:
        df: DataFrame
        thresholds: 阈值字典，如 {'min_sharpe_net': 0.95, 'max_dd_net': -0.28, ...}
    
    返回:
        过滤后的 DataFrame
    """
    mask = pd.Series(True, index=df.index)
    
    # 最低 Sharpe
    if 'min_sharpe_net' in thresholds:
        mask &= (df['sharpe_net'] >= thresholds['min_sharpe_net'])
    
    # 最大回撤（注意：max_dd_net 是负数，越大越好）
    if 'max_dd_net' in thresholds:
        mask &= (df['max_dd_net'] >= thresholds['max_dd_net'])
    
    # 最低年化收益
    if 'min_annual_ret_net' in thresholds:
        mask &= (df['annual_ret_net'] >= thresholds['min_annual_ret_net'])
    
    # 最大换手
    if 'max_turnover' in thresholds:
        mask &= (df['avg_turnover'] <= thresholds['max_turnover'])
    
    return df[mask].copy()


def adaptive_quality_filter(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    自适应质量过滤（标准 -> 放松/收紧）
    
    参数:
        df: DataFrame
        config: 配置字典
    
    返回:
        过滤后的 DataFrame
    """
    qf_config = config['quality_filter']
    
    # 第一轮：标准阈值
    filtered = apply_quality_filter(df, qf_config['standard'])
    n_filtered = len(filtered)
    
    logging.info(f"质量过滤（标准阈值）: {n_filtered} / {len(df)} 个组合通过")
    
    # 自适应调整
    min_samples = qf_config['adaptive_thresholds']['min_samples']
    max_samples = qf_config['adaptive_thresholds']['max_samples']
    
    if n_filtered < min_samples:
        logging.info(f"样本不足 (<{min_samples})，放松阈值重新过滤...")
        filtered = apply_quality_filter(df, qf_config['relaxed'])
        logging.info(f"质量过滤（放松阈值）: {len(filtered)} / {len(df)} 个组合通过")
        
    elif n_filtered > max_samples:
        logging.info(f"样本过多 (>{max_samples})，收紧阈值重新过滤...")
        tightened = {**qf_config['standard'], **qf_config['tightened']}
        filtered = apply_quality_filter(df, tightened)
        logging.info(f"质量过滤（收紧阈值）: {len(filtered)} / {len(df)} 个组合通过")
    
    return filtered


# ========== 步骤 2：因子结构解析 ==========

def parse_factor_list(combo: str) -> List[str]:
    """
    从 combo 字符串解析因子列表
    
    参数:
        combo: 如 "ADX_14D + CMF_20D + MAX_DD_60D"
    
    返回:
        因子列表: ['ADX_14D', 'CMF_20D', 'MAX_DD_60D']
    """
    return [f.strip() for f in combo.split('+')]


def classify_factor(factor_name: str, categories: dict) -> str:
    """
    将单个因子归类到 trend/vol/volume_price/relative/mixed
    
    参数:
        factor_name: 因子名称
        categories: 分类关键词字典
    
    返回:
        类别名称
    """
    factor_upper = factor_name.upper()
    
    for category, keywords in categories.items():
        for kw in keywords:
            if kw in factor_upper:
                return category
    
    return 'mixed'


def analyze_factor_structure(combo: str, categories: dict) -> dict:
    """
    分析因子结构
    
    参数:
        combo: 组合字符串
        categories: 分类关键词字典
    
    返回:
        {
            'factors': list,
            'dominant_factor': str,
            'factor_counts': dict,
            'bucket': str,
        }
    """
    factors = parse_factor_list(combo)
    
    factor_counts = {'trend': 0, 'vol': 0, 'volume_price': 0, 'relative': 0, 'mixed': 0}
    
    for factor in factors:
        category = classify_factor(factor, categories)
        factor_counts[category] += 1
    
    # 确定 dominant_factor（数量最多的类别）
    counted = {k: v for k, v in factor_counts.items() if v > 0}
    if not counted or all(v == 0 for v in factor_counts.values()):
        dominant_factor = 'mixed'
    else:
        dominant_factor = max(counted, key=counted.get)
    
    combo_size = len(factors)
    bucket = f"{combo_size}_{dominant_factor}"
    
    return {
        'factors': factors,
        'dominant_factor': dominant_factor,
        'factor_counts': factor_counts,
        'bucket': bucket,
    }


def add_factor_structure_columns(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    为 DataFrame 添加因子结构列
    
    参数:
        df: DataFrame
        config: 配置字典
    
    返回:
        添加了 dominant_factor 和 bucket 列的 DataFrame
    """
    df = df.copy()
    categories = config['factor_categories']
    
    structures = df['combo'].apply(lambda c: analyze_factor_structure(c, categories))
    
    df['dominant_factor'] = structures.apply(lambda s: s['dominant_factor'])
    df['bucket'] = structures.apply(lambda s: s['bucket'])
    
    logging.info(f"因子结构解析完成，生成了 {df['bucket'].nunique()} 个不同的桶")
    
    return df


# ========== 步骤 3：综合评分 ==========

def calculate_selection_score(df: pd.DataFrame, weights: dict) -> pd.Series:
    """
    计算综合评分
    
    参数:
        df: DataFrame
        weights: 权重字典
    
    返回:
        评分 Series
    """
    score = pd.Series(0.0, index=df.index)
    
    for field, weight in weights.items():
        if field in df.columns:
            # 处理可能的缺失值
            values = df[field].fillna(0.0)
            score += weight * values
    
    return score


# ========== 步骤 4：分桶配额分配 ==========

def calculate_bucket_quotas(bucket_sizes: pd.Series, config: dict) -> dict:
    """
    根据桶样本数计算原始配额
    
    参数:
        bucket_sizes: Series，index 为 bucket 名称，value 为样本数
        config: 配置字典
    
    返回:
        {bucket: raw_quota} 字典
    """
    bq_config = config['bucket_quotas']
    thresholds = bq_config['size_thresholds']
    quotas = bq_config['quotas']
    min_quota = bq_config['min_quota']
    
    raw_quotas = {}
    
    for bucket, size in bucket_sizes.items():
        # 根据样本数确定配额
        if size >= thresholds[0]:
            raw_quota = quotas[0]
        elif size >= thresholds[1]:
            raw_quota = quotas[1]
        elif size >= thresholds[2]:
            raw_quota = quotas[2]
        else:
            raw_quota = quotas[3]
        
        # 应用最小配额
        raw_quotas[bucket] = max(min_quota, raw_quota)
    
    return raw_quotas


def normalize_quotas(raw_quotas: dict, total_quota: int) -> dict:
    """
    归一化配额，确保总和 = total_quota
    
    参数:
        raw_quotas: 原始配额字典
        total_quota: 目标总配额
    
    返回:
        归一化后的配额字典
    """
    if not raw_quotas:
        return {}
    
    total_raw = sum(raw_quotas.values())
    
    # 比例缩放
    scaled = {}
    for bucket, raw_q in raw_quotas.items():
        scaled[bucket] = max(3, int(raw_q * total_quota / total_raw))
    
    # 调整差值
    diff = total_quota - sum(scaled.values())
    
    # 获取桶的样本数信息（假设 bucket 名称中不包含样本数，使用原始配额作为代理）
    sorted_buckets = sorted(raw_quotas.keys(), key=lambda b: raw_quotas[b], reverse=True)
    
    if diff > 0:
        # 需要补充：按原始配额从大到小依次 +1
        idx = 0
        while diff > 0 and idx < len(sorted_buckets):
            bucket = sorted_buckets[idx % len(sorted_buckets)]
            scaled[bucket] += 1
            diff -= 1
            idx += 1
    elif diff < 0:
        # 需要削减：按当前配额从大到小依次 -1
        sorted_by_quota = sorted(scaled.keys(), key=lambda b: scaled[b], reverse=True)
        idx = 0
        while diff < 0 and idx < len(sorted_by_quota):
            bucket = sorted_by_quota[idx]
            if scaled[bucket] > 3:  # 保证最少 3 个
                scaled[bucket] -= 1
                diff += 1
            idx += 1
    
    return scaled


def sample_from_buckets(df: pd.DataFrame, quotas: dict, score_col: str) -> pd.DataFrame:
    """
    从各桶按配额采样
    
    参数:
        df: DataFrame（包含 'bucket' 列）
        quotas: 配额字典 {bucket: quota}
        score_col: 评分列名
    
    返回:
        采样后的 DataFrame
    """
    sampled_dfs = []
    
    for bucket, quota in quotas.items():
        bucket_df = df[df['bucket'] == bucket]
        
        if len(bucket_df) == 0:
            continue
        
        # 取实际可采样数量（不超过桶大小）
        actual_quota = min(quota, len(bucket_df))
        
        # 按评分降序取 top N
        sampled = bucket_df.nlargest(actual_quota, score_col)
        sampled_dfs.append(sampled)
        
        logging.debug(f"桶 '{bucket}': {len(bucket_df)} 个组合，配额 {quota}，实际采样 {actual_quota}")
    
    result = pd.concat(sampled_dfs, ignore_index=True)
    
    logging.info(f"分桶采样完成: 从 {df['bucket'].nunique()} 个桶中采样了 {len(result)} 个组合")
    
    return result


# ========== 步骤 5：combo_size 分布调整 ==========

def check_combo_size_distribution(df: pd.DataFrame, targets: dict) -> dict:
    """
    检查当前 combo_size 分布
    
    参数:
        df: DataFrame
        targets: 目标配置 {size: {'min': x, 'max': y}}
    
    返回:
        {size: {'current': n, 'min': x, 'max': y, 'status': 'ok'|'insufficient'|'excess'}}
    """
    current_dist = df['combo_size'].value_counts().to_dict()
    
    result = {}
    for size, target in targets.items():
        current = current_dist.get(size, 0)
        
        if current < target['min']:
            status = 'insufficient'
        elif current > target['max']:
            status = 'excess'
        else:
            status = 'ok'
        
        result[size] = {
            'current': current,
            'min': target['min'],
            'max': target['max'],
            'status': status,
        }
    
    return result


def adjust_combo_size_distribution(
    candidate: pd.DataFrame,
    pool: pd.DataFrame,
    targets: dict,
    score_col: str
) -> pd.DataFrame:
    """
    调整 combo_size 分布：补充不足/删除超标
    
    参数:
        candidate: 当前候选集
        pool: 完整数据池（用于补充）
        targets: 目标配置
        score_col: 评分列名
    
    返回:
        调整后的 DataFrame
    """
    candidate = candidate.copy()
    dist_check = check_combo_size_distribution(candidate, targets)
    
    logging.info(f"combo_size 分布检查: {dist_check}")
    
    # 获取候选集中已入选的索引
    selected_indices = set(candidate.index)
    
    for size, info in dist_check.items():
        if info['status'] == 'insufficient':
            # 需要补充
            need = info['min'] - info['current']
            logging.info(f"combo_size={size} 不足，需要补充 {need} 个")
            
            # 从 pool 中找未入选的同 size 组合
            available = pool[
                (pool['combo_size'] == size) & 
                (~pool.index.isin(selected_indices))
            ]
            
            if len(available) > 0:
                # 按评分降序取 need 个
                to_add = available.nlargest(min(need, len(available)), score_col)
                candidate = pd.concat([candidate, to_add], ignore_index=False)
                selected_indices.update(to_add.index)
                logging.info(f"补充了 {len(to_add)} 个 combo_size={size} 的组合")
            else:
                logging.warning(f"无可用的 combo_size={size} 组合用于补充")
        
        elif info['status'] == 'excess':
            # 需要删除
            excess = info['current'] - info['max']
            logging.info(f"combo_size={size} 超标，需要删除 {excess} 个")
            
            # 从候选集中该 size 的组合里，按评分升序删除
            size_candidates = candidate[candidate['combo_size'] == size]
            to_remove = size_candidates.nsmallest(excess, score_col)
            candidate = candidate.drop(to_remove.index)
            selected_indices.difference_update(to_remove.index)
            logging.info(f"删除了 {len(to_remove)} 个 combo_size={size} 的组合")
    
    return candidate


# ========== 步骤 6：高换手控制 ==========

def check_high_turnover_ratio(df: pd.DataFrame, threshold: float) -> Tuple[int, float]:
    """
    计算高换手比例
    
    参数:
        df: DataFrame
        threshold: 高换手阈值
    
    返回:
        (高换手数量, 高换手比例)
    """
    high_turnover_count = (df['avg_turnover'] > threshold).sum()
    ratio = high_turnover_count / len(df) if len(df) > 0 else 0.0
    
    return high_turnover_count, ratio


def adjust_high_turnover_ratio(
    candidate: pd.DataFrame,
    pool: pd.DataFrame,
    threshold: float,
    max_ratio: float,
    max_count: int,
    score_col: str
) -> pd.DataFrame:
    """
    控制高换手比例不超过 max_ratio
    
    参数:
        candidate: 当前候选集
        pool: 完整数据池
        threshold: 高换手阈值
        max_ratio: 最大比例
        max_count: 最大数量（通常为 total_quota * max_ratio）
        score_col: 评分列名
    
    返回:
        调整后的 DataFrame
    """
    candidate = candidate.copy()
    high_count, ratio = check_high_turnover_ratio(candidate, threshold)
    
    logging.info(f"高换手（>{threshold}）检查: {high_count} 个组合 ({ratio:.1%})")
    
    if high_count > max_count:
        # 需要替换
        n_to_replace = high_count - max_count
        logging.info(f"高换手比例超标，需要替换 {n_to_replace} 个组合")
        
        # 从候选集中找高换手组合，按评分升序删除
        high_turnover = candidate[candidate['avg_turnover'] > threshold]
        to_remove = high_turnover.nsmallest(n_to_replace, score_col)
        
        candidate = candidate.drop(to_remove.index)
        
        # 从 pool 中找低换手且未入选的组合，按评分降序补充
        selected_indices = set(candidate.index)
        available = pool[
            (pool['avg_turnover'] <= threshold) &
            (~pool.index.isin(selected_indices))
        ]
        
        if len(available) > 0:
            to_add = available.nlargest(min(n_to_replace, len(available)), score_col)
            candidate = pd.concat([candidate, to_add], ignore_index=False)
            logging.info(f"替换了 {len(to_add)} 个低换手组合")
        else:
            logging.warning(f"无足够的低换手组合用于替换（需要 {n_to_replace}，可用 {len(available)}）")
    
    return candidate


# ========== 步骤 7：最终截断 ==========

def finalize_top200(
    df: pd.DataFrame,
    pool: pd.DataFrame,
    target_count: int,
    score_col: str
) -> pd.DataFrame:
    """
    最终截断/补足到目标数量
    
    参数:
        df: 当前候选集
        pool: 完整数据池
        target_count: 目标数量（200）
        score_col: 评分列名
    
    返回:
        最终 Top-N DataFrame（包含 final_rank 列）
    """
    df = df.copy()
    
    current_count = len(df)
    
    if current_count > target_count:
        # 截断
        logging.info(f"当前 {current_count} 个组合，截断到 {target_count}")
        df = df.nlargest(target_count, score_col)
    
    elif current_count < target_count:
        # 补足
        need = target_count - current_count
        logging.info(f"当前 {current_count} 个组合，需要补足 {need} 个")
        
        selected_indices = set(df.index)
        available = pool[~pool.index.isin(selected_indices)]
        
        if len(available) > 0:
            to_add = available.nlargest(min(need, len(available)), score_col)
            df = pd.concat([df, to_add], ignore_index=False)
            logging.info(f"从数据池补充了 {len(to_add)} 个组合")
        else:
            logging.warning(f"数据池无可用组合，最终只有 {len(df)} 个")
    
    # 按评分降序排序并添加 final_rank
    df = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    df['final_rank'] = range(1, len(df) + 1)
    
    logging.info(f"最终筛选完成: {len(df)} 个组合")
    
    return df


# ========== 主流程编排 ==========

def select_top200(df: pd.DataFrame, config: dict = None, verbose: bool = True) -> pd.DataFrame:
    """
    主流程：从 Top-2000 筛选 Top-200
    
    参数:
        df: 输入 DataFrame
        config: 配置字典（None 则使用 DEFAULT_CONFIG）
        verbose: 是否打印详细日志
    
    返回:
        Top-200 DataFrame（包含 selection_score、final_rank 等列）
    """
    # 配置日志级别
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    # 使用默认配置或合并用户配置
    if config is None:
        config = DEFAULT_CONFIG
    else:
        config = deep_merge_config(DEFAULT_CONFIG, config)
    
    logging.info("=" * 60)
    logging.info("开始 Top-200 筛选流程")
    logging.info("=" * 60)
    
    # 步骤 0：数据预处理
    logging.info("\n[步骤 0] 数据预处理")
    df_clean = validate_and_clean_data(df)
    df_clean = add_missing_fields_with_defaults(df_clean)
    
    # 步骤 1：质量过滤
    logging.info("\n[步骤 1] 质量过滤")
    df_filtered = adaptive_quality_filter(df_clean, config)
    
    # 保留完整数据池用于后续补充
    pool = df_filtered.copy()
    
    # 步骤 2：因子结构解析
    logging.info("\n[步骤 2] 因子结构解析")
    df_filtered = add_factor_structure_columns(df_filtered, config)
    
    # 步骤 3：综合评分
    logging.info("\n[步骤 3] 计算综合评分")
    df_filtered['selection_score'] = calculate_selection_score(
        df_filtered, 
        config['scoring_weights']
    )
    logging.info(f"评分范围: [{df_filtered['selection_score'].min():.4f}, {df_filtered['selection_score'].max():.4f}]")
    
    # 更新 pool 的评分和因子结构
    pool = df_filtered.copy()
    
    # 步骤 4：分桶配额采样
    logging.info("\n[步骤 4] 分桶配额采样")
    bucket_sizes = df_filtered['bucket'].value_counts()
    raw_quotas = calculate_bucket_quotas(bucket_sizes, config)
    normalized_quotas = normalize_quotas(raw_quotas, config['total_quota'])
    
    logging.info(f"桶配额分配: {normalized_quotas}")
    
    candidate = sample_from_buckets(df_filtered, normalized_quotas, 'selection_score')
    
    # 步骤 5：combo_size 分布调整
    logging.info("\n[步骤 5] combo_size 分布调整")
    candidate = adjust_combo_size_distribution(
        candidate,
        pool,
        config['combo_size_targets'],
        'selection_score'
    )
    
    # 步骤 6：高换手控制
    logging.info("\n[步骤 6] 高换手比例控制")
    turnover_config = config['turnover_control']
    max_high_turnover_count = int(config['total_quota'] * turnover_config['max_ratio'])
    
    candidate = adjust_high_turnover_ratio(
        candidate,
        pool,
        turnover_config['threshold'],
        turnover_config['max_ratio'],
        max_high_turnover_count,
        'selection_score'
    )
    
    # 步骤 7：最终截断
    logging.info("\n[步骤 7] 最终截断与排序")
    result = finalize_top200(
        candidate,
        pool,
        config['total_quota'],
        'selection_score'
    )
    
    # 打印最终统计
    logging.info("\n" + "=" * 60)
    logging.info("筛选完成！最终统计:")
    logging.info("=" * 60)
    
    size_dist = result['combo_size'].value_counts().sort_index()
    logging.info(f"combo_size 分布:\n{size_dist}")
    
    high_count, high_ratio = check_high_turnover_ratio(result, turnover_config['threshold'])
    logging.info(f"高换手（>{turnover_config['threshold']}）: {high_count} 个 ({high_ratio:.1%})")
    
    factor_dist = result['dominant_factor'].value_counts()
    logging.info(f"dominant_factor 分布:\n{factor_dist}")
    
    logging.info(f"\n平均指标:")
    logging.info(f"  annual_ret_net: {result['annual_ret_net'].mean():.4f}")
    logging.info(f"  sharpe_net: {result['sharpe_net'].mean():.4f}")
    logging.info(f"  max_dd_net: {result['max_dd_net'].mean():.4f}")
    logging.info(f"  avg_turnover: {result['avg_turnover'].mean():.4f}")
    
    return result
