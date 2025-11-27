"""
信号强度过滤优化器

实现实验 1.1 和 1.2：
- 趋势强度阈值扫描
- 多因子方向一致性过滤
"""

import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from pathlib import Path


class SignalStrengthOptimizer:
    """
    信号强度优化器
    
    负责对因子信号进行过滤和增强，提升策略的信噪比。
    """
    
    def __init__(self, combo_factors: List[str], verbose: bool = True):
        """
        参数:
            combo_factors: 组合中的因子列表，如 ['OBV_SLOPE_10D', 'RSI_14', ...]
            verbose: 是否打印详细日志
        """
        self.combo_factors = combo_factors
        self.verbose = verbose
        
        # 因子分类
        self.trend_factors = []
        self.relative_factors = []
        self.vol_factors = []
        self.volume_price_factors = []
        
        self._classify_factors()
        
        if self.verbose:
            logging.info(f"组合因子: {combo_factors}")
            logging.info(f"趋势因子: {self.trend_factors}")
            logging.info(f"相对因子: {self.relative_factors}")
    
    def _classify_factors(self):
        """分类因子到不同类别"""
        trend_keywords = ['SLOPE', 'VORTEX', 'MOM', 'ADX', 'TREND', 'ROC']
        relative_keywords = ['RSI', 'PRICE_POSITION', 'RELATIVE', 'CORRELATION', 'BETA']
        vol_keywords = ['VOL_RATIO', 'MAX_DD', 'RET_VOL', 'SHARPE', 'VAR', 'STD']
        volume_price_keywords = ['OBV', 'PV_CORR', 'CMF', 'MFI']
        
        for factor in self.combo_factors:
            factor_upper = factor.upper()
            
            if any(kw in factor_upper for kw in trend_keywords):
                self.trend_factors.append(factor)
            elif any(kw in factor_upper for kw in relative_keywords):
                self.relative_factors.append(factor)
            elif any(kw in factor_upper for kw in vol_keywords):
                self.vol_factors.append(factor)
            elif any(kw in factor_upper for kw in volume_price_keywords):
                self.volume_price_factors.append(factor)
    
    def apply_trend_strength_filter(
        self,
        factor_data: pd.DataFrame,
        threshold_pct: float = 0.0
    ) -> pd.DataFrame:
        """
        实验 1.1: 趋势强度阈值过滤
        
        参数:
            factor_data: 因子数据 DataFrame，index 为日期，columns 为 [因子名_标的代码]
            threshold_pct: 百分位阈值 (0-100)，如 40 表示只保留排名前 60% 的标的
        
        返回:
            过滤后的因子数据（不满足条件的位置设为 NaN）
        """
        if threshold_pct == 0:
            return factor_data
        
        filtered_data = factor_data.copy()
        
        # 对每个趋势因子应用阈值
        for factor in self.trend_factors:
            # 找到该因子对应的所有列（不同标的）
            factor_cols = [col for col in factor_data.columns if col.startswith(factor + '_')]
            
            if not factor_cols:
                continue
            
            # 按横截面计算百分位
            for date in factor_data.index:
                values = factor_data.loc[date, factor_cols]
                
                # 计算阈值（需要根据因子方向调整）
                threshold = np.percentile(values.dropna(), 100 - threshold_pct)
                
                # 低于阈值的设为 NaN
                mask = values < threshold
                filtered_data.loc[date, factor_cols[mask]] = np.nan
        
        if self.verbose:
            original_valid = factor_data.notna().sum().sum()
            filtered_valid = filtered_data.notna().sum().sum()
            logging.info(f"趋势强度过滤 (阈值={threshold_pct}%): "
                        f"有效值从 {original_valid} 降至 {filtered_valid} "
                        f"({filtered_valid/original_valid:.1%})")
        
        return filtered_data
    
    def apply_direction_consistency_filter(
        self,
        factor_data: pd.DataFrame,
        min_consistent: int = 2
    ) -> pd.DataFrame:
        """
        实验 1.2: 多因子方向一致性过滤
        
        要求趋势因子的信号方向一致（都为正或都为负）
        
        参数:
            factor_data: 因子数据 DataFrame
            min_consistent: 最少需要一致的因子数（2 或 3）
        
        返回:
            过滤后的因子数据
        """
        if len(self.trend_factors) < 2:
            logging.warning("趋势因子少于2个，跳过方向一致性过滤")
            return factor_data
        
        filtered_data = factor_data.copy()
        
        # 获取所有标的代码（假设列名格式为 '因子名_标的代码'）
        symbols = set()
        for col in factor_data.columns:
            if '_' in col:
                symbol = col.split('_', 1)[1]
                symbols.add(symbol)
        
        # 对每个标的检查趋势因子的方向一致性
        for symbol in symbols:
            trend_cols = [f"{factor}_{symbol}" for factor in self.trend_factors 
                         if f"{factor}_{symbol}" in factor_data.columns]
            
            if len(trend_cols) < min_consistent:
                continue
            
            for date in factor_data.index:
                values = factor_data.loc[date, trend_cols].dropna()
                
                if len(values) < min_consistent:
                    # 缺失太多，保守起见设为 NaN
                    for col in trend_cols:
                        filtered_data.loc[date, col] = np.nan
                    continue
                
                # 检查方向一致性
                positive_count = (values > 0).sum()
                negative_count = (values < 0).sum()
                
                # 判断是否一致
                if min_consistent == 2:
                    # 至少2个方向一致
                    is_consistent = (positive_count >= 2) or (negative_count >= 2)
                else:
                    # 全部一致
                    is_consistent = (positive_count == len(values)) or (negative_count == len(values))
                
                if not is_consistent:
                    # 方向不一致，过滤掉
                    for col in trend_cols:
                        filtered_data.loc[date, col] = np.nan
        
        if self.verbose:
            original_valid = factor_data.notna().sum().sum()
            filtered_valid = filtered_data.notna().sum().sum()
            logging.info(f"方向一致性过滤 (min_consistent={min_consistent}): "
                        f"有效值从 {original_valid} 降至 {filtered_valid} "
                        f"({filtered_valid/original_valid:.1%})")
        
        return filtered_data
    
    def apply_rsi_extreme_filter(
        self,
        factor_data: pd.DataFrame,
        rsi_thresholds: Optional[Tuple[float, float]] = None
    ) -> pd.DataFrame:
        """
        实验 1.3: RSI 极端区域过滤
        
        只在 RSI 极端区域（超买/超卖）时才使用该信号
        
        参数:
            factor_data: 因子数据 DataFrame
            rsi_thresholds: (下限, 上限)，如 (35, 65)
        
        返回:
            过滤后的因子数据
        """
        if rsi_thresholds is None:
            return factor_data
        
        # 找到 RSI 因子列
        rsi_cols = [col for col in factor_data.columns if 'RSI' in col.upper()]
        
        if not rsi_cols:
            logging.warning("未找到 RSI 因子，跳过 RSI 过滤")
            return factor_data
        
        filtered_data = factor_data.copy()
        lower, upper = rsi_thresholds
        
        for col in rsi_cols:
            # RSI 在中性区域时降权（乘以 0.5）
            mask = (factor_data[col] >= lower) & (factor_data[col] <= upper)
            filtered_data.loc[mask, col] = filtered_data.loc[mask, col] * 0.5
        
        if self.verbose:
            affected_count = ((factor_data[rsi_cols[0]] >= lower) & 
                             (factor_data[rsi_cols[0]] <= upper)).sum()
            logging.info(f"RSI 极端过滤 (阈值={rsi_thresholds}): "
                        f"{affected_count} 个观测被降权")
        
        return filtered_data


def run_signal_optimization_experiments(
    combo_name: str,
    combo_factors: List[str],
    base_results: Dict,
    output_dir: Path
) -> pd.DataFrame:
    """
    运行信号优化实验套件
    
    参数:
        combo_name: 组合名称
        combo_factors: 因子列表
        base_results: 基线回测结果字典
        output_dir: 输出目录
    
    返回:
        实验结果汇总 DataFrame
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logging.info("=" * 60)
    logging.info(f"开始信号优化实验: {combo_name}")
    logging.info("=" * 60)
    
    optimizer = SignalStrengthOptimizer(combo_factors, verbose=True)
    
    # 实验配置
    experiments = []
    
    # 实验 1.1: 趋势强度阈值
    for threshold in [0, 20, 40, 60]:
        experiments.append({
            'exp_id': f'1.1_threshold_{threshold}',
            'exp_name': f'趋势强度阈值={threshold}%',
            'filter_type': 'trend_strength',
            'params': {'threshold_pct': threshold}
        })
    
    # 实验 1.2: 方向一致性
    for min_consistent in [2, 3]:
        experiments.append({
            'exp_id': f'1.2_consistency_{min_consistent}',
            'exp_name': f'方向一致性(min={min_consistent})',
            'filter_type': 'direction_consistency',
            'params': {'min_consistent': min_consistent}
        })
    
    # 实验 1.3: RSI 极端过滤
    for thresholds in [None, (30, 70), (35, 65), (40, 60)]:
        if thresholds is None:
            continue
        experiments.append({
            'exp_id': f'1.3_rsi_{thresholds[0]}_{thresholds[1]}',
            'exp_name': f'RSI极端过滤({thresholds})',
            'filter_type': 'rsi_extreme',
            'params': {'rsi_thresholds': thresholds}
        })
    
    # 实验结果记录
    results = []
    
    for exp in experiments:
        logging.info(f"\n运行实验: {exp['exp_name']}")
        logging.info("-" * 40)
        
        # TODO: 这里需要实际的回测逻辑
        # 由于完整回测需要原始数据和复杂的框架，这里先创建占位结果
        
        result = {
            'exp_id': exp['exp_id'],
            'exp_name': exp['exp_name'],
            'filter_type': exp['filter_type'],
            'params': str(exp['params']),
            # 占位指标（实际需要从回测中获取）
            'annual_ret_net': np.nan,
            'sharpe_net': np.nan,
            'max_dd_net': np.nan,
            'calmar_ratio': np.nan,
            'avg_turnover': np.nan,
            'win_rate': np.nan,
            'status': 'pending_backtest'
        }
        
        results.append(result)
    
    # 转换为 DataFrame
    results_df = pd.DataFrame(results)
    
    # 保存结果
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{combo_name}_signal_optimization_results.csv"
    results_df.to_csv(output_path, index=False)
    
    logging.info(f"\n实验配置已保存: {output_path}")
    logging.info(f"总计 {len(results)} 个实验待执行")
    
    return results_df


if __name__ == '__main__':
    # 测试代码
    combo_factors = ['OBV_SLOPE_10D', 'PRICE_POSITION_20D', 'RSI_14', 'SLOPE_20D', 'VORTEX_14D']
    
    optimizer = SignalStrengthOptimizer(combo_factors)
    
    # 创建模拟数据
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    symbols = ['ETF001', 'ETF002', 'ETF003']
    
    data = {}
    for factor in combo_factors:
        for symbol in symbols:
            data[f'{factor}_{symbol}'] = np.random.randn(100)
    
    factor_data = pd.DataFrame(data, index=dates)
    
    # 测试趋势强度过滤
    filtered1 = optimizer.apply_trend_strength_filter(factor_data, threshold_pct=40)
    
    # 测试方向一致性过滤
    filtered2 = optimizer.apply_direction_consistency_filter(factor_data, min_consistent=2)
    
    # 测试 RSI 过滤
    filtered3 = optimizer.apply_rsi_extreme_filter(factor_data, rsi_thresholds=(35, 65))
    
    print("信号优化器测试完成")
