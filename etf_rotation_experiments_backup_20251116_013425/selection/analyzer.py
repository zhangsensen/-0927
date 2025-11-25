"""
单组合分析模块

提供单个组合的策略画像分析功能。
"""

import pandas as pd
from typing import Union, Dict
from .core import DEFAULT_CONFIG, analyze_factor_structure


def analyze_single_combo(
    df: pd.DataFrame,
    combo_identifier: Union[str, int],
    config: dict = None
) -> dict:
    """
    分析单个组合，返回策略画像
    
    参数:
        df: 完整回测结果 DataFrame
        combo_identifier: 组合标识，支持：
            - str: combo 字符串（如 "ADX_14D + CMF_20D + ..."）
            - int: rank 或 final_rank 或行索引
        config: 配置字典（用于因子分类），None 则使用 DEFAULT_CONFIG
    
    返回:
        策略画像字典:
        {
            'combo': str,
            'combo_size': int,
            'rank': int,
            'final_rank': int (如果有),
            'performance': {
                'annual_ret_net': float,
                'sharpe_net': float,
                'max_dd_net': float,
                'calmar_ratio': float,
                'sortino_ratio': float,
            },
            'trading': {
                'avg_turnover': float,
                'avg_n_holdings': float,
                'n_rebalance': int,
                'win_rate': float,
            },
            'factor_structure': {
                'factors': list,
                'dominant_factor': str,
                'factor_counts': dict,
            },
            'selection_score': float (如果有),
        }
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    # 查找组合行
    if isinstance(combo_identifier, str):
        # 按 combo 字符串查找
        row = df[df['combo'] == combo_identifier]
        if len(row) == 0:
            raise ValueError(f"未找到 combo='{combo_identifier}' 的组合")
        row = row.iloc[0]
    elif isinstance(combo_identifier, int):
        # 尝试按 final_rank、rank 或索引查找
        if 'final_rank' in df.columns:
            row = df[df['final_rank'] == combo_identifier]
            if len(row) > 0:
                row = row.iloc[0]
            else:
                # 尝试 rank
                if 'rank' in df.columns:
                    row = df[df['rank'] == combo_identifier]
                    if len(row) > 0:
                        row = row.iloc[0]
                    else:
                        # 尝试索引
                        if combo_identifier in df.index:
                            row = df.loc[combo_identifier]
                        else:
                            raise ValueError(f"未找到 identifier={combo_identifier} 的组合")
                else:
                    if combo_identifier in df.index:
                        row = df.loc[combo_identifier]
                    else:
                        raise ValueError(f"未找到 identifier={combo_identifier} 的组合")
        elif 'rank' in df.columns:
            row = df[df['rank'] == combo_identifier]
            if len(row) > 0:
                row = row.iloc[0]
            else:
                if combo_identifier in df.index:
                    row = df.loc[combo_identifier]
                else:
                    raise ValueError(f"未找到 identifier={combo_identifier} 的组合")
        else:
            if combo_identifier in df.index:
                row = df.loc[combo_identifier]
            else:
                raise ValueError(f"未找到 identifier={combo_identifier} 的组合")
    else:
        raise TypeError("combo_identifier 必须是 str 或 int")
    
    # 解析因子结构
    factor_structure = analyze_factor_structure(row['combo'], config['factor_categories'])
    
    # 构建画像字典
    profile = {
        'combo': row['combo'],
        'combo_size': int(row['combo_size']),
        'rank': int(row['rank']) if 'rank' in row.index else None,
        'final_rank': int(row['final_rank']) if 'final_rank' in row.index else None,
        
        'performance': {
            'annual_ret_net': float(row['annual_ret_net']) if 'annual_ret_net' in row.index else None,
            'sharpe_net': float(row['sharpe_net']) if 'sharpe_net' in row.index else None,
            'max_dd_net': float(row['max_dd_net']) if 'max_dd_net' in row.index else None,
            'calmar_ratio': float(row['calmar_ratio']) if 'calmar_ratio' in row.index else None,
            'sortino_ratio': float(row['sortino_ratio']) if 'sortino_ratio' in row.index else None,
        },
        
        'trading': {
            'avg_turnover': float(row['avg_turnover']) if 'avg_turnover' in row.index else None,
            'avg_n_holdings': float(row['avg_n_holdings']) if 'avg_n_holdings' in row.index else None,
            'n_rebalance': int(row['n_rebalance']) if 'n_rebalance' in row.index else None,
            'win_rate': float(row['win_rate']) if 'win_rate' in row.index else None,
        },
        
        'factor_structure': {
            'factors': factor_structure['factors'],
            'dominant_factor': factor_structure['dominant_factor'],
            'factor_counts': factor_structure['factor_counts'],
        },
        
        'selection_score': float(row['selection_score']) if 'selection_score' in row.index else None,
    }
    
    return profile


def format_combo_profile(profile: dict) -> str:
    """
    格式化组合画像为可读字符串
    
    参数:
        profile: 由 analyze_single_combo() 返回的字典
    
    返回:
        格式化的字符串
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"组合: {profile['combo']}")
    lines.append("=" * 60)
    
    lines.append(f"\n基本信息:")
    lines.append(f"  因子数量: {profile['combo_size']}")
    if profile['rank']:
        lines.append(f"  原始排名: {profile['rank']}")
    if profile['final_rank']:
        lines.append(f"  最终排名: {profile['final_rank']}")
    
    lines.append(f"\n因子结构:")
    lines.append(f"  主导类别: {profile['factor_structure']['dominant_factor']}")
    lines.append(f"  因子列表: {', '.join(profile['factor_structure']['factors'])}")
    
    counts = profile['factor_structure']['factor_counts']
    non_zero = {k: v for k, v in counts.items() if v > 0}
    if non_zero:
        lines.append(f"  类别分布: {non_zero}")
    
    lines.append(f"\n性能指标:")
    perf = profile['performance']
    if perf['annual_ret_net'] is not None:
        lines.append(f"  年化收益: {perf['annual_ret_net']:.2%}")
    if perf['sharpe_net'] is not None:
        lines.append(f"  Sharpe 比率: {perf['sharpe_net']:.3f}")
    if perf['max_dd_net'] is not None:
        lines.append(f"  最大回撤: {perf['max_dd_net']:.2%}")
    if perf['calmar_ratio'] is not None:
        lines.append(f"  Calmar 比率: {perf['calmar_ratio']:.3f}")
    if perf['sortino_ratio'] is not None:
        lines.append(f"  Sortino 比率: {perf['sortino_ratio']:.3f}")
    
    lines.append(f"\n交易特征:")
    trade = profile['trading']
    if trade['avg_turnover'] is not None:
        lines.append(f"  平均换手: {trade['avg_turnover']:.3f}")
    if trade['avg_n_holdings'] is not None:
        lines.append(f"  平均持仓数: {trade['avg_n_holdings']:.1f}")
    if trade['n_rebalance'] is not None:
        lines.append(f"  调仓次数: {trade['n_rebalance']}")
    if trade['win_rate'] is not None:
        lines.append(f"  胜率: {trade['win_rate']:.2%}")
    
    if profile['selection_score'] is not None:
        lines.append(f"\n综合评分: {profile['selection_score']:.4f}")
    
    lines.append("=" * 60)
    
    return '\n'.join(lines)
