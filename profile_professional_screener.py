#!/usr/bin/env python3
"""
专业因子筛选器性能分析
分析 professional_factor_screener.py 中的关键函数性能瓶颈
"""

import sys
import time
import numpy as np
import pandas as pd
from line_profiler import profile
import logging

# 添加项目路径
sys.path.insert(0, '/Users/zhangshenshen/深度量化0927/factor_system/factor_screening')

# 导入要分析的类
from professional_factor_screener import ProfessionalFactorScreener, ScreeningConfig

# 设置日志级别
logging.basicConfig(level=logging.INFO)

class PerformanceTestScreener(ProfessionalFactorScreener):
    """性能测试版本的因子筛选器"""

    def __init__(self):
        # 简化配置用于测试
        config = ScreeningConfig()
        config.ic_horizons = [1, 3, 5, 10]  # 减少周期数
        super().__init__("/Users/zhangshenshen/深度量化0927/因子筛选/测试_data", config)

    @profile
    def load_factors_test(self, symbol: str = "0700.HK", timeframe: str = "60min") -> pd.DataFrame:
        """测试数据加载性能"""
        self.logger.info(f"测试加载因子数据: {symbol} {timeframe}")

        # 模拟数据加载
        dates = pd.date_range('2023-01-01', periods=1000, freq='D')
        np.random.seed(42)

        # 模拟20个因子
        factor_data = np.random.randn(1000, 20)
        factor_names = [f'factor_{i:02d}' for i in range(20)]

        factors_df = pd.DataFrame(
            factor_data,
            index=dates,
            columns=factor_names
        )

        # 添加一些OHLCV列（会被过滤掉）
        factors_df['open'] = np.random.randn(1000)
        factors_df['high'] = np.random.randn(1000)
        factors_df['low'] = np.random.randn(1000)
        factors_df['close'] = np.random.randn(1000)
        factors_df['volume'] = np.random.randint(100, 10000, 1000)

        return factors_df

    @profile
    def generate_returns_data(self) -> pd.Series:
        """生成收益率数据用于IC计算"""
        dates = pd.date_range('2023-01-01', periods=1000, freq='D')
        np.random.seed(42)

        # 模拟收益率序列
        base_returns = np.random.randn(1000) * 0.01
        returns = pd.Series(base_returns, index=dates, name='returns')

        return returns

    @profile
    def calculate_multi_horizon_ic_test(
        self, factors: pd.DataFrame, returns: pd.Series
    ) -> dict:
        """测试多周期IC计算性能"""
        self.logger.info("开始测试多周期IC计算...")
        start_time = time.time()

        ic_results = {}
        horizons = self.config.ic_horizons

        # 预计算所有周期的历史收益率（Linus模式：正确实现）
        historical_returns = {}
        for horizon in horizons:
            # 使用正向shift获取历史收益率，避免前视偏差
            historical_returns[horizon] = returns.shift(horizon)

        factor_cols = [
            col for col in factors.columns
            if col not in ["open", "high", "low", "close", "volume"]
        ]

        total_factors = len(factor_cols)
        processed = 0

        for factor in factor_cols:
            processed += 1
            if processed % 5 == 0:
                self.logger.info(f"处理进度: {processed}/{total_factors}")

            if factor not in factors or factors[factor].empty:
                continue

            factor_series = factors[factor].dropna()

            if len(factor_series) < 50:
                continue

            ic_results[factor] = {}

            for horizon in horizons:
                try:
                    aligned_factor = factor_series.reindex(historical_returns[horizon].index)
                    aligned_return = historical_returns[horizon].reindex(factor_series.index)

                    # 只使用有数据的部分
                    valid_mask = ~(aligned_factor.isna() | aligned_return.isna())

                    if valid_mask.sum() < 30:
                        continue

                    ic = aligned_factor[valid_mask].corr(aligned_return[valid_mask])
                    ic_results[factor][f'ic_{horizon}d'] = ic if not pd.isna(ic) else 0.0

                except Exception as e:
                    self.logger.warning(f"IC计算失败 {factor}-{horizon}d: {e}")
                    ic_results[factor][f'ic_{horizon}d'] = 0.0

        elapsed_time = time.time() - start_time
        self.logger.info(f"多周期IC计算完成，耗时: {elapsed_time:.2f}s")

        return ic_results

    @profile
    def calculate_rolling_ic_test(
        self, factors: pd.DataFrame, returns: pd.Series, window: int = 20
    ) -> dict:
        """测试滚动IC计算性能"""
        self.logger.info("开始测试滚动IC计算...")
        start_time = time.time()

        rolling_ic_results = {}
        factor_cols = [
            col for col in factors.columns
            if col not in ["open", "high", "low", "close", "volume"]
        ]

        for factor in factor_cols:
            if factor not in factors or factors[factor].empty:
                continue

            factor_series = factors[factor].dropna()

            # 模拟滚动IC计算
            rolling_ic_values = []

            for i in range(window, len(factor_series)):
                window_factor = factor_series.iloc[i-window:i]
                window_return = returns.reindex(window_factor.index).dropna()

                if len(window_factor) >= 10 and len(window_return) >= 10:
                    ic = window_factor.corr(window_return)
                    rolling_ic_values.append(ic if not pd.isna(ic) else 0.0)

            if rolling_ic_values:
                rolling_ic_results[factor] = {
                    'rolling_ic_mean': np.mean(rolling_ic_values),
                    'rolling_ic_std': np.std(rolling_ic_values),
                    'rolling_ic_samples': len(rolling_ic_values)
                }

        elapsed_time = time.time() - start_time
        self.logger.info(f"滚动IC计算完成，耗时: {elapsed_time:.2f}s")

        return rolling_ic_results

def run_performance_analysis():
    """运行性能分析主函数"""
    print("🚀 开始专业因子筛选器性能分析")
    print("=" * 60)

    # 创建测试实例
    screener = PerformanceTestScreener()

    print("\n📊 生成测试数据...")
    factors = screener.load_factors_test()
    returns = screener.generate_returns_data()

    print(f"✅ 数据生成完成:")
    print(f"   - 因子数据: {factors.shape}")
    print(f"   - 收益数据: {len(returns)} 个数据点")

    print("\n🔍 执行多周期IC计算性能分析...")
    ic_results = screener.calculate_multi_horizon_ic_test(factors, returns)

    print(f"✅ IC计算完成，分析了 {len(ic_results)} 个因子")

    print("\n📈 执行滚动IC计算性能分析...")
    rolling_results = screener.calculate_rolling_ic_test(factors, returns)

    print(f"✅ 滚动IC计算完成，分析了 {len(rolling_results)} 个因子")

    print("\n🎯 性能分析总结:")
    print("   - 详细的行级性能数据已保存到 .lprof 文件")
    print("   - 使用 'python -m line_profiler profile_professional_screener.py.lprof' 查看详细报告")
    print("   - 重点关注计算密集型操作的性能瓶颈")

if __name__ == "__main__":
    run_performance_analysis()