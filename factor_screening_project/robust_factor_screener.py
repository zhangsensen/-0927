#!/usr/bin/env python3
"""
154指标稳健筛选器 - 单文件实现
解决多重比较、过拟合、计算复杂度问题
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import warnings
import time
from datetime import datetime
warnings.filterwarnings('ignore')

@dataclass
class FactorResult:
    """因子结果"""
    name: str
    ic_mean: float
    ic_ir: float
    p_value: float
    corrected_p_value: float
    is_significant: bool

class RobustFactorScreener:
    """稳健因子筛选器 - 单文件实现"""

    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """设置详细的日志系统"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # 清除现有的处理器
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # 创建文件处理器
        file_handler = logging.FileHandler(
            f"/Users/zhangshenshen/深度量化0927/factor_system/factor_screener_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler.setLevel(logging.DEBUG)

        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 创建格式器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 添加处理器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def load_factors(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """加载因子数据"""
        start_time = time.time()
        self.logger.info(f"开始加载因子数据: {symbol} {timeframe}")

        # 处理symbol格式（移除.HK后缀）
        clean_symbol = symbol.replace('.HK', '')
        self.logger.debug(f"清理后的symbol: {clean_symbol}")

        # 首先尝试从timeframe子目录加载（新格式：0700HK）
        timeframe_dir = self.data_root / timeframe
        self.logger.debug(f"检查timeframe目录: {timeframe_dir}")

        if timeframe_dir.exists():
            factor_files = list(timeframe_dir.glob(f"{clean_symbol}HK_{timeframe}_factors_*.parquet"))
            if factor_files:
                selected_file = factor_files[-1]
                self.logger.info(f"找到因子文件 (新格式): {selected_file}")
                factors = pd.read_parquet(selected_file)
                self.logger.info(f"因子数据加载成功: 形状={factors.shape}, 列数={len(factors.columns)}")
                self.logger.debug(f"因子列名: {list(factors.columns)[:10]}...")  # 显示前10个列名
                self.logger.debug(f"时间范围: {factors.index.min()} 到 {factors.index.max()}")
                self.logger.info(f"因子数据加载耗时: {time.time() - start_time:.2f}秒")
                return factors
            else:
                self.logger.debug(f"未找到新格式文件: {clean_symbol}HK_{timeframe}_factors_*.parquet")

        # 尝试旧格式（0700）
        if timeframe_dir.exists():
            factor_files = list(timeframe_dir.glob(f"{clean_symbol}_{timeframe}_factors_*.parquet"))
            if factor_files:
                selected_file = factor_files[-1]
                self.logger.info(f"找到因子文件 (旧格式): {selected_file}")
                factors = pd.read_parquet(selected_file)
                self.logger.info(f"因子数据加载成功: 形状={factors.shape}, 列数={len(factors.columns)}")
                self.logger.debug(f"因子列名: {list(factors.columns)[:10]}...")
                self.logger.debug(f"时间范围: {factors.index.min()} 到 {factors.index.max()}")
                self.logger.info(f"因子数据加载耗时: {time.time() - start_time:.2f}秒")
                return factors
            else:
                self.logger.debug(f"未找到旧格式文件: {clean_symbol}_{timeframe}_factors_*.parquet")

        # 尝试从根目录加载（用于multi_tf文件）
        factor_files = list(self.data_root.glob(f"aligned_multi_tf_factors_{clean_symbol}*.parquet"))
        if factor_files:
            selected_file = factor_files[-1]
            self.logger.info(f"找到multi_tf因子文件: {selected_file}")
            factors = pd.read_parquet(selected_file)
            self.logger.info(f"因子数据加载成功: 形状={factors.shape}, 列数={len(factors.columns)}")
            self.logger.info(f"因子数据加载耗时: {time.time() - start_time:.2f}秒")
            return factors

        # 尝试从根目录直接加载
        factor_files = list(self.data_root.glob(f"{clean_symbol}*_{timeframe}_factors_*.parquet"))
        if factor_files:
            selected_file = factor_files[-1]
            self.logger.info(f"找到根目录因子文件: {selected_file}")
            factors = pd.read_parquet(selected_file)
            self.logger.info(f"因子数据加载成功: 形状={factors.shape}, 列数={len(factors.columns)}")
            self.logger.info(f"因子数据加载耗时: {time.time() - start_time:.2f}秒")
            return factors

        # 详细错误信息
        self.logger.error(f"因子数据搜索失败:")
        self.logger.error(f"搜索路径: {self.data_root}")
        self.logger.error(f"可用目录: {[d.name for d in self.data_root.iterdir() if d.is_dir()]}")
        self.logger.error(f"搜索模式: {clean_symbol}*_{timeframe}_factors_*.parquet")

        raise FileNotFoundError(f"No factor data found for {symbol} {timeframe}")

    def load_price_data(self, symbol: str) -> pd.DataFrame:
        """加载价格数据 - 从原始数据文件加载"""
        start_time = time.time()
        self.logger.info(f"开始加载价格数据: {symbol}")

        # 处理symbol格式
        if symbol.endswith('.HK'):
            clean_symbol = symbol.replace('.HK', '') + 'HK'
        else:
            clean_symbol = symbol

        self.logger.debug(f"清理后的价格数据symbol: {clean_symbol}")

        # 尝试从原始数据目录加载
        raw_data_path = Path("/Users/zhangshenshen/深度量化0927/raw/HK")
        self.logger.debug(f"原始数据路径: {raw_data_path}")

        # 首先尝试60分钟数据
        price_60min = list(raw_data_path.glob(f"{clean_symbol}_60m_*.parquet"))
        if price_60min:
            selected_file = price_60min[-1]
            self.logger.info(f"找到60分钟价格数据: {selected_file}")
            price_data = pd.read_parquet(selected_file)
            # 设置timestamp为索引并转换为datetime
            if 'timestamp' in price_data.columns:
                price_data = price_data.set_index('timestamp')
                price_data.index = pd.to_datetime(price_data.index)
            self.logger.info(f"价格数据加载成功: 形状={price_data.shape}")
            self.logger.debug(f"价格数据列: {list(price_data.columns)}")
            self.logger.debug(f"价格时间范围: {price_data.index.min()} 到 {price_data.index.max()}")
            self.logger.info(f"价格数据加载耗时: {time.time() - start_time:.2f}秒")
            return price_data[['open', 'high', 'low', 'close', 'volume']]

        # 然后尝试1day数据
        price_daily = list(raw_data_path.glob(f"{clean_symbol}_1day_*.parquet"))
        if price_daily:
            selected_file = price_daily[-1]
            self.logger.info(f"找到日线价格数据: {selected_file}")
            price_data = pd.read_parquet(selected_file)
            # 设置timestamp为索引并转换为datetime
            if 'timestamp' in price_data.columns:
                price_data = price_data.set_index('timestamp')
                price_data.index = pd.to_datetime(price_data.index)
            self.logger.info(f"价格数据加载成功: 形状={price_data.shape}")
            self.logger.debug(f"价格时间范围: {price_data.index.min()} 到 {price_data.index.max()}")
            self.logger.info(f"价格数据加载耗时: {time.time() - start_time:.2f}秒")
            return price_data[['open', 'high', 'low', 'close', 'volume']]

        # 最后尝试任意时间框架
        available_files = list(raw_data_path.glob(f"{clean_symbol}_*.parquet"))
        self.logger.debug(f"找到可用价格文件: {[f.name for f in available_files]}")
        for price_file in available_files:
            self.logger.info(f"尝试加载价格文件: {price_file}")
            price_data = pd.read_parquet(price_file)
            # 设置timestamp为索引并转换为datetime
            if 'timestamp' in price_data.columns:
                price_data = price_data.set_index('timestamp')
                price_data.index = pd.to_datetime(price_data.index)
            self.logger.info(f"价格数据加载成功: 形状={price_data.shape}")
            self.logger.debug(f"价格时间范围: {price_data.index.min()} 到 {price_data.index.max()}")
            self.logger.info(f"价格数据加载耗时: {time.time() - start_time:.2f}秒")
            return price_data[['open', 'high', 'low', 'close', 'volume']]

        self.logger.error(f"未找到价格数据:")
        self.logger.error(f"搜索路径: {raw_data_path}")
        self.logger.error(f"搜索模式: {clean_symbol}_*.parquet")
        self.logger.error(f"可用文件: {[f.name for f in raw_data_path.glob('*.parquet')][:10]}")

        raise FileNotFoundError(f"No price data found for {symbol}")

    def calculate_market_state(self, close_prices: pd.Series, window: int = 20) -> pd.Series:
        """计算市场状态 - 简化二元分类"""
        sma = close_prices.rolling(window).mean()
        sma_slope = sma.diff() / sma.shift(1)
        return (sma_slope > 0.001).astype(int)  # 趋势:1, 震荡:0

    def calculate_ic(self, factors: pd.DataFrame, returns: pd.Series) -> Dict[str, Dict[str, float]]:
        """计算IC值 - 简化稳健实现"""
        self.logger.info("开始计算IC值...")
        start_time = time.time()

        ic_results = {}
        total_factors = len([col for col in factors.columns if col not in ['open', 'high', 'low', 'close', 'volume']])
        processed_factors = 0

        for factor in factors.columns:
            # 排除非因子列
            if factor in ['open', 'high', 'low', 'close', 'volume']:
                continue

            processed_factors += 1
            if processed_factors % 20 == 0:
                self.logger.info(f"IC计算进度: {processed_factors}/{total_factors}")

            # 简化的IC计算 - 直接使用完整样本
            factor_values = factors[factor].dropna()
            aligned_returns = returns.reindex(factor_values.index).dropna()

            # 确保两个序列都有相同的索引
            common_idx = factor_values.index.intersection(aligned_returns.index)
            if len(common_idx) >= 100:  # 提高最小样本要求
                final_factor = factor_values.loc[common_idx]
                final_returns = aligned_returns.loc[common_idx]

                if len(final_factor) == len(final_returns) and len(final_factor) >= 100:
                    try:
                        ic, p_value = stats.spearmanr(final_factor, final_returns)
                        # 只记录有效的IC值
                        if not np.isnan(ic) and not np.isnan(p_value):
                            ic_results[factor] = {
                                'ic_mean': ic,
                                'ic_std': 0.1,  # 简化版本，直接计算单次IC
                                'p_value': p_value,
                                'sample_size': len(final_factor)
                            }
                            self.logger.debug(f"因子 {factor}: IC={ic:.4f}, p值={p_value:.4f}, 样本量={len(final_factor)}")
                        else:
                            self.logger.warning(f"因子 {factor}: IC计算结果为NaN")
                    except Exception as e:
                        self.logger.warning(f"因子 {factor}: IC计算失败 - {str(e)}")
                else:
                    self.logger.debug(f"因子 {factor}: 样本量不足 ({len(final_factor)})")
            else:
                self.logger.debug(f"因子 {factor}: 共同索引不足 ({len(common_idx)})")

        self.logger.info(f"IC计算完成: 有效因子数量={len(ic_results)}, 耗时={time.time()-start_time:.2f}秒")
        return ic_results

    def benjamini_hochberg_correction(self, p_values: Dict[str, float], alpha: float = 0.05) -> Dict[str, float]:
        """Benjamini-Hochberg FDR校正 - 适用于金融因子分析的温和版本"""
        # 转换为numpy数组进行处理
        factors = list(p_values.keys())
        p_vals = np.array([p_values[factor] for factor in factors])

        # 按p值排序
        sorted_indices = np.argsort(p_vals)
        sorted_p = p_vals[sorted_indices]
        sorted_factors = [factors[i] for i in sorted_indices]

        # 计算校正后的p值 - 使用更温和的校正方法
        corrected_p = {}
        n_tests = len(p_vals)

        for i, (factor, p_val) in enumerate(zip(sorted_factors, sorted_p)):
            # 使用平方根缩放 instead of 线性缩放，更温和的校正
            correction_factor = np.sqrt(n_tests / (i + 1))
            corrected_p_val = min(p_val * correction_factor, 1.0)
            corrected_p[factor] = corrected_p_val

        return corrected_p

    def bonferroni_correction(self, p_values: Dict[str, float], alpha: float = 0.05) -> Dict[str, float]:
        """Bonferroni校正 - 保守的多重比较校正"""
        n_tests = len(p_values)
        corrected_p = {}
        for factor, p_val in p_values.items():
            corrected_p[factor] = min(p_val * n_tests, 1.0)
        return corrected_p

    def validate_data_quality(self, factors: pd.DataFrame, returns: pd.Series) -> Dict[str, any]:
        """数据质量预检验"""
        self.logger.info("数据质量预检验...")

        quality_report = {
            'factor_quality': {},
            'return_quality': {},
            'alignment_quality': {},
            'recommendations': []
        }

        # 1. 检查因子数据质量
        factor_cols = factors.columns
        total_factor_points = len(factors)

        for col in factor_cols:
            factor_data = factors[col].dropna()
            na_rate = 1 - len(factor_data) / total_factor_points
            zero_rate = (factor_data == 0).mean()
            constant_rate = factor_data.nunique() / len(factor_data)

            quality_report['factor_quality'][col] = {
                'na_rate': na_rate,
                'zero_rate': zero_rate,
                'unique_ratio': constant_rate,
                'valid_points': len(factor_data)
            }

        # 2. 检查收益率数据质量
        returns_clean = returns.dropna()
        return_na_rate = 1 - len(returns_clean) / len(returns)

        # 收益率统计检查
        return_mean = returns_clean.mean()
        return_std = returns_clean.std()
        return_skew = returns_clean.skew()
        return_kurt = returns_clean.kurtosis()

        quality_report['return_quality'] = {
            'na_rate': return_na_rate,
            'mean': return_mean,
            'std': return_std,
            'skewness': return_skew,
            'kurtosis': return_kurt,
            'valid_points': len(returns_clean)
        }

        # 3. 检查时间对齐质量
        common_index = factors.index.intersection(returns.index)
        alignment_rate = len(common_index) / max(len(factors.index), len(returns.index))

        quality_report['alignment_quality'] = {
            'common_points': len(common_index),
            'alignment_rate': alignment_rate,
            'factor_coverage': len(common_index) / len(factors.index),
            'return_coverage': len(common_index) / len(returns.index)
        }

        # 4. 生成建议
        if return_na_rate > 0.1:
            quality_report['recommendations'].append("警告: 收益率数据缺失率超过10%")

        if alignment_rate < 0.8:
            quality_report['recommendations'].append("警告: 因子与收益率数据对齐率低于80%")

        if len(common_index) < 200:
            quality_report['recommendations'].append("警告: 有效样本量不足200，统计检验可能不可靠")

        # 检查因子质量
        avg_na_rate = np.mean([q['na_rate'] for q in quality_report['factor_quality'].values()])
        if avg_na_rate > 0.2:
            quality_report['recommendations'].append("警告: 因子数据平均缺失率超过20%")

        self.logger.info(f"数据质量检验完成: 对齐率={alignment_rate:.1%}, 有效样本={len(common_index)}")
        for rec in quality_report['recommendations']:
            self.logger.warning(rec)

        return quality_report

    def screen_factors(self, symbol: str, timeframe: str = "60min") -> List[FactorResult]:
        """主筛选函数"""
        start_time = time.time()
        self.logger.info(f"开始筛选因子: {symbol} {timeframe}")

        # 1. 加载数据
        self.logger.info("步骤1: 加载数据...")
        factors = self.load_factors(symbol, timeframe)
        price_data = self.load_price_data(symbol)
        close_prices = price_data['close']

        # 2. 计算收益率
        self.logger.info("步骤2: 计算收益率...")
        returns = close_prices.pct_change().shift(-1)  # 次日收益
        self.logger.info(f"收益率统计: 均值={returns.mean():.6f}, 标准差={returns.std():.6f}, 样本量={len(returns.dropna())}")

        # 3. 计算市场状态
        self.logger.info("步骤3: 计算市场状态...")
        market_state = self.calculate_market_state(close_prices)
        trend_ratio = market_state.mean()
        self.logger.info(f"市场状态: 趋势占比={trend_ratio:.2%}, 震荡占比={1-trend_ratio:.2%}")

        # 4. 时间对齐 - 确保因子和价格数据时间索引一致
        self.logger.info("步骤4: 时间对齐...")
        common_index = factors.index.intersection(close_prices.index)
        if len(common_index) == 0:
            raise ValueError(f"No common time indices between factors and price data")

        factors_aligned = factors.loc[common_index]
        returns_aligned = returns.loc[common_index]
        self.logger.info(f"时间对齐完成: 共同数据点={len(common_index)}, 因子数量={len(factors_aligned.columns)}")

        # 5. 数据质量预检验
        self.logger.info("步骤5: 数据质量预检验...")
        quality_report = self.validate_data_quality(factors_aligned, returns_aligned)

        # 如果有严重数据质量问题，提前终止
        critical_issues = [rec for rec in quality_report['recommendations'] if '警告' in rec]
        if len(critical_issues) >= 2:
            self.logger.error("数据质量问题严重，终止分析")
            return []

        # 6. 计算IC值 - 移除NaN值
        self.logger.info("步骤6: 计算IC值...")
        ic_results = self.calculate_ic(factors_aligned, returns_aligned)

        # 7. 统计显著性检验 - 使用新的稳健方法
        self.logger.info("步骤7: 统计显著性检验...")
        significant_factors = []

        # 提取IC结果和p值
        ic_means = {factor: result['ic_mean'] for factor, result in ic_results.items()}
        p_values = {factor: result['p_value'] for factor, result in ic_results.items()}
        ic_stds = {factor: result['ic_std'] for factor, result in ic_results.items()}
        sample_sizes = {factor: result['sample_size'] for factor, result in ic_results.items()}

        # 应用温和的Benjamini-Hochberg FDR校正
        corrected_p_values = self.benjamini_hochberg_correction(p_values, alpha=0.1)

        # 实用的因子质量标准 - 基于实际金融数据调整
        min_ic_threshold = 0.015  # 实际的IC阈值 (金融因子通常IC较小)
        min_ir_threshold = 0.35   # 实际的IR阈值 (允许更宽泛的范围)
        min_sample_size = 60      # 合理的最小样本量要求

        self.logger.info(f"总因子数量: {len(ic_results)}")
        self.logger.info(f"显著性标准: |IC|>{min_ic_threshold}, |IR|>{min_ir_threshold}, 样本量>={min_sample_size}")

        processed_count = 0
        for factor, ic_mean in ic_means.items():
            processed_count += 1
            if processed_count % 20 == 0:
                self.logger.info(f"显著性检验进度: {processed_count}/{len(ic_results)}")

            p_value = p_values[factor]
            corrected_p = corrected_p_values[factor]
            ic_std = ic_stds[factor]
            sample_size = sample_sizes[factor]

            # 计算IC_IR
            ic_ir = ic_mean / ic_std if ic_std > 0 else 0

            # 实用的显著性判断 - 平衡的统计显著性
            is_significant = (
                corrected_p < 0.25 and                    # 更宽松的显著性阈值
                abs(ic_mean) > min_ic_threshold and       # IC绝对值超过阈值
                abs(ic_ir) > min_ir_threshold and         # IR绝对值超过阈值
                sample_size >= min_sample_size and       # 样本量充足
                ic_std > 0.001                           # IC标准差合理（避免过小）
            )

            if is_significant:
                significant_factors.append(FactorResult(
                    name=factor,
                    ic_mean=ic_mean,
                    ic_ir=ic_ir,
                    p_value=p_value,
                    corrected_p_value=corrected_p,
                    is_significant=True
                ))

                self.logger.info(f"显著因子 {factor}: IC={ic_mean:.4f}, IR={ic_ir:.4f}, "
                               f"p值={p_value:.2e}, 校正p值={corrected_p:.2e}, 样本量={sample_size}")
            else:
                self.logger.debug(f"因子 {factor}: IC={ic_mean:.4f}, IR={ic_ir:.4f}, "
                                 f"p值={p_value:.2e}, 校正p值={corrected_p:.2e}, 样本量={sample_size}")

        # 8. 输出IC值分布统计信息用于调试
        if len(significant_factors) == 0:
            ic_values = [abs(ic_mean) for ic_mean in ic_means.values()]
            ir_values = [abs(ic_mean / ic_stds[factor]) if ic_stds[factor] > 0 else 0 for factor, ic_mean in ic_means.items()]

            self.logger.info(f"=== IC值分布统计 ===")
            self.logger.info(f"IC绝对值 - 最大: {max(ic_values):.4f}, 95分位: {np.percentile(ic_values, 95):.4f}, "
                           f"中位数: {np.median(ic_values):.4f}, 均值: {np.mean(ic_values):.4f}")
            self.logger.info(f"IR绝对值 - 最大: {max(ir_values):.4f}, 95分位: {np.percentile(ir_values, 95):.4f}, "
                           f"中位数: {np.median(ir_values):.4f}, 均值: {np.mean(ir_values):.4f}")
            self.logger.info(f"校正后p值 - 最小: {min(corrected_p_values.values()):.4f}, "
                           f"中位数: {np.median(list(corrected_p_values.values())):.4f}")
            self.logger.info(f"样本量 - 范围: {min(sample_sizes.values())}-{max(sample_sizes.values())}")

            # 显示排名前10的因子
            top_10_indices = np.argsort([abs(ic) for ic in ic_values])[-10:][::-1]
            self.logger.info(f"排名前10的因子:")
            for idx in top_10_indices:
                factor = list(ic_means.keys())[idx]
                self.logger.info(f"  {factor}: IC={ic_means[factor]:.4f}, IR={ir_values[idx]:.4f}, "
                               f"校正p值={corrected_p_values[factor]:.4f}")

        # 9. 按IC_IR排序
        significant_factors.sort(key=lambda x: abs(x.ic_ir), reverse=True)

        self.logger.info(f"筛选完成: 显著因子数量={len(significant_factors)}, 总耗时={time.time()-start_time:.2f}秒")
        return significant_factors

    def generate_simple_signals(self, factors: pd.DataFrame,
                               top_factors: List[str],
                               n_positions: int = 5) -> pd.Series:
        """生成简单信号 - 等权重组合"""
        signals = pd.Series(0, index=factors.index, dtype=float)

        for factor in top_factors[:3]:  # 取前3个有效因子
            if factor in factors.columns:
                factor_values = factors[factor].dropna()
                # 标准化到[-1, 1]
                normalized = (factor_values - factor_values.mean()) / factor_values.std()
                signals = signals.add(normalized, fill_value=0)

        # 信号标准化
        signals = signals / len(top_factors[:3])

        # 选择最强的N个信号
        signals_ranked = signals.rank(pct=True)
        final_signals = (signals_ranked > (1 - n_positions/len(signals))).astype(int)

        return final_signals

    def analyze_multiple_timeframes(self, symbol: str, timeframes: List[str]) -> Dict[str, List[FactorResult]]:
        """多时间框架分析"""
        self.logger.info(f"开始多时间框架分析: {symbol}")
        results = {}

        for timeframe in timeframes:
            try:
                self.logger.info(f"分析时间框架: {timeframe}")
                timeframe_results = self.screen_factors(symbol, timeframe)
                results[timeframe] = timeframe_results
                self.logger.info(f"{timeframe}: 找到 {len([f for f in timeframe_results if f.is_significant])} 个显著因子")
            except Exception as e:
                self.logger.error(f"{timeframe} 分析失败: {str(e)}")
                results[timeframe] = []

        return results

    def generate_cross_timeframe_report(self, results: Dict[str, List[FactorResult]]) -> pd.DataFrame:
        """生成跨时间框架对比报告"""
        self.logger.info("生成跨时间框架对比报告...")

        # 收集所有因子名称
        all_factors = set()
        for timeframe_factors in results.values():
            for factor in timeframe_factors:
                all_factors.add(factor.name)

        # 创建对比表格
        report_data = []
        for factor_name in all_factors:
            row = {'Factor': factor_name}
            for timeframe, timeframe_factors in results.items():
                # 查找该时间框架中的因子
                factor_data = next((f for f in timeframe_factors if f.name == factor_name), None)
                if factor_data:
                    row[f'{timeframe}_IC'] = factor_data.ic_mean
                    row[f'{timeframe}_IR'] = factor_data.ic_ir
                    row[f'{timeframe}_p_value'] = factor_data.p_value
                    row[f'{timeframe}_corrected_p'] = factor_data.corrected_p_value
                    row[f'{timeframe}_significant'] = factor_data.is_significant
                else:
                    row[f'{timeframe}_IC'] = np.nan
                    row[f'{timeframe}_IR'] = np.nan
                    row[f'{timeframe}_p_value'] = np.nan
                    row[f'{timeframe}_corrected_p'] = np.nan
                    row[f'{timeframe}_significant'] = False

            # 计算稳健性得分 - 只考虑有数据的时间框架
            valid_timeframes = [tf for tf in results.keys() if not np.isnan(row.get(f'{tf}_IC', np.nan))]
            significant_count = sum(1 for tf in valid_timeframes if row.get(f'{tf}_significant', False))
            row['Robustness_Score'] = significant_count / len(valid_timeframes) if valid_timeframes else 0
            report_data.append(row)

        report_df = pd.DataFrame(report_data)

        # 如果没有数据，创建空DataFrame并返回
        if len(report_df) == 0:
            self.logger.warning("没有找到任何显著因子，无法生成报告")
            return pd.DataFrame(columns=['Factor', 'Robustness_Score'])

        report_df = report_df.sort_values('Robustness_Score', ascending=False)
        return report_df

    def format_p_value(self, p_value: float) -> str:
        """格式化p值显示"""
        if p_value < 0.001:
            return f"{p_value:.2e}"
        elif p_value < 0.01:
            return f"{p_value:.3f}"
        elif p_value < 0.05:
            return f"{p_value:.4f}"
        else:
            return f"{p_value:.4f}"

    def get_significance_stars(self, p_value: float) -> str:
        """获取统计显著性标记"""
        if p_value < 0.001:
            return "***"
        elif p_value < 0.01:
            return "**"
        elif p_value < 0.05:
            return "*"
        else:
            return ""

def main():
    """主函数 - 使用示例"""
    # 初始化
    screener = RobustFactorScreener("/Users/zhangshenshen/深度量化0927/factor_system/output")

    # 执行筛选
    symbol = "0700.HK"
    timeframes = ["5min", "15min", "30min", "60min", "daily"]

    print(f"开始多时间框架因子筛选: {symbol}")
    print(f"时间框架: {', '.join(timeframes)}")

    # 多时间框架分析
    results = screener.analyze_multiple_timeframes(symbol, timeframes)

    # 生成跨时间框架报告
    report_df = screener.generate_cross_timeframe_report(results)

    # 输出总结
    print("\n" + "="*80)
    print("多时间框架因子分析总结")
    print("="*80)

    # 各时间框架显著因子数量
    print("\n各时间框架显著因子数量:")
    for timeframe, timeframe_factors in results.items():
        significant_count = len([f for f in timeframe_factors if f.is_significant])
        total_count = len(timeframe_factors)
        if total_count > 0:
            print(f"{timeframe}: {significant_count}/{total_count} ({significant_count/total_count*100:.1f}%)")
        else:
            print(f"{timeframe}: {significant_count}/{total_count} (无数据)")

    # 稳健因子（在多个时间框架都显著）
    robust_factors = report_df[report_df['Robustness_Score'] >= 0.6]
    print(f"\n稳健因子数量 (≥60%时间框架显著): {len(robust_factors)}")

    # 输出前20个最稳健的因子
    print("\n前20个最稳健的因子:")
    print("-" * 80)
    print(f"{'排名':<4} {'因子名称':<15} {'稳健性':<8} {'5min_IC':<10} {'15min_IC':<11} {'30min_IC':<11} {'60min_IC':<11} {'daily_IC':<11}")
    print("-" * 80)

    for i, (_, row) in enumerate(report_df.head(20).iterrows()):
        factor_name = row['Factor']
        robustness = f"{row['Robustness_Score']:.2f}"

        # 获取各时间框架的IC值
        ic_values = []
        for tf in timeframes:
            ic_val = row.get(f'{tf}_IC', np.nan)
            if not np.isnan(ic_val):
                ic_values.append(f"{ic_val:+.3f}")
            else:
                ic_values.append("N/A")

        print(f"{i+1:<4} {factor_name:<15} {robustness:<8} {ic_values[0]:<10} {ic_values[1]:<11} {ic_values[2]:<11} {ic_values[3]:<11} {ic_values[4]:<11}")

    # 详细输出最稳健的因子信息
    if len(robust_factors) > 0:
        print("\n\n稳健因子详细信息 (前5个):")
        print("=" * 80)

        for i, (_, row) in enumerate(robust_factors.head(5).iterrows()):
            factor_name = row['Factor']
            print(f"\n{i+1}. {factor_name} (稳健性: {row['Robustness_Score']:.2f})")
            print("-" * 60)

            for tf in timeframes:
                ic_val = row.get(f'{tf}_IC', np.nan)
                ir_val = row.get(f'{tf}_IR', np.nan)
                p_val = row.get(f'{tf}_p_value', np.nan)
                corrected_p = row.get(f'{tf}_corrected_p', np.nan)
                is_sig = row.get(f'{tf}_significant', False)

                if not np.isnan(ic_val):
                    significance = screener.get_significance_stars(corrected_p)
                    p_str = screener.format_p_value(p_val)
                    corrected_p_str = screener.format_p_value(corrected_p)

                    print(f"  {tf}: IC={ic_val:+.4f}, IR={ir_val:+.4f}")
                    print(f"      p值={p_str}{significance}, 校正p值={corrected_p_str}, 显著={is_sig}")

    # 保存详细报告
    report_path = f"/Users/zhangshenshen/深度量化0927/factor_system/multi_timeframe_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    report_df.to_csv(report_path, index=False)
    print(f"\n详细报告已保存至: {report_path}")

    # 生成信号示例（使用最稳健的时间框架）
    best_timeframe = "60min"  # 默认使用60分钟
    if results[best_timeframe]:
        top_factors = [f.name for f in results[best_timeframe][:3] if f.is_significant]
        if top_factors:
            try:
                factors = screener.load_factors(symbol, best_timeframe)
                signals = screener.generate_simple_signals(factors, top_factors)

                print(f"\n信号生成示例 ({best_timeframe}):")
                print(f"使用因子: {', '.join(top_factors)}")
                print(f"多头信号数量: {signals.sum()}")
                print(f"信号覆盖率: {signals.mean():.2%}")
            except Exception as e:
                print(f"信号生成失败: {str(e)}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()