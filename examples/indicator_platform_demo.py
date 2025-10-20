#!/usr/bin/env python3
"""
指标平台集成示例 - 展示从因子工厂到指标平台的转变
"""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 导入新的指标平台组件
from factor_system.factor_engine.adapters.smart_indicator_adapter import (
    SmartIndicatorAdapter,
)
from factor_system.factor_engine.adapters.vbt_adapter_optimized import (
    VBTIndicatorAdapterOptimized,
)
from factor_system.factor_engine.prescreening.indicator_prescreener import (
    IndicatorPrescreener,
)


class IndicatorPlatformDemo:
    """指标平台演示 - 展示新的架构思路"""

    def __init__(self):
        self.smart_adapter = SmartIndicatorAdapter(
            lookback_period=252, forward_period=5, min_samples=60
        )

        self.optimized_adapter = VBTIndicatorAdapterOptimized(
            enable_smart_selection=True, ic_threshold=0.02
        )

        self.prescreener = IndicatorPrescreener(
            min_ic_threshold=0.01,
            min_ic_ir_threshold=0.1,
            max_missing_ratio=0.3,
            correlation_threshold=0.8,
        )

    def create_sample_data(self, n_days: int = 500, n_etfs: int = 5) -> pd.DataFrame:
        """创建示例数据"""
        np.random.seed(42)

        # 生成日期序列
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq="D")

        # 为每个ETF生成数据
        all_data = []

        for etf_id in range(n_etfs):
            etf_code = f"ETF{etf_id:03d}"

            # 生成价格序列（随机游走+趋势）
            returns = np.random.normal(0.0005, 0.02, n_days)  # 日收益率
            prices = [100.0]  # 起始价格

            for ret in returns:
                new_price = prices[-1] * (1 + ret)
                prices.append(new_price)

            prices = prices[1:]  # 去掉起始值

            # 生成OHLCV数据
            for i, date in enumerate(dates):
                close = prices[i]

                # 生成日内波动
                high_noise = np.random.uniform(0.005, 0.02)
                low_noise = np.random.uniform(0.005, 0.02)

                high = close * (1 + high_noise)
                low = close * (1 - low_noise)
                open_price = np.random.uniform(low, high)

                # 成交量（与价格变动相关）
                volume = int(
                    np.random.uniform(1000000, 5000000) * (1 + abs(returns[i]) * 10)
                )

                all_data.append(
                    {
                        "symbol": etf_code,
                        "date": date,
                        "open": open_price,
                        "high": high,
                        "low": low,
                        "close": close,
                        "volume": volume,
                    }
                )

        df = pd.DataFrame(all_data)
        return df

    def generate_target_variable(
        self, price_series: pd.Series, forward_days: int = 5
    ) -> pd.Series:
        """生成目标变量（未来收益率）"""
        # 未来n日收益率
        future_returns = price_series.pct_change(forward_days).shift(-forward_days)
        return future_returns

    def compare_approaches(self, data: pd.DataFrame):
        """对比传统方法和指标平台方法"""

        print("=" * 60)
        print("指标平台 vs 传统因子工厂 - 对比演示")
        print("=" * 60)

        # 为每个ETF单独处理
        etf_codes = data["symbol"].unique()
        results = []

        for etf_code in etf_codes:
            print(f"\n处理ETF: {etf_code}")

            # 提取单个ETF数据
            etf_data = data[data["symbol"] == etf_code].copy()
            etf_data = etf_data.sort_values("date").reset_index(drop=True)

            # 生成目标变量
            target = self.generate_target_variable(etf_data["close"])

            # ===== 方法1: 传统因子工厂（参数爆炸） =====
            print(f"方法1: 传统因子工厂 - {etf_code}")
            traditional_factors = self._traditional_factor_factory(etf_data)
            print(f"  生成因子数: {traditional_factors.shape[1]}")

            # ===== 方法2: 指标平台（智能选择） =====
            print(f"方法2: 指标平台 - {etf_code}")

            # 步骤1: 智能指标生成
            smart_indicators = self.optimized_adapter.compute_all_indicators(
                etf_data, target
            )
            print(f"  生成指标数: {smart_indicators.shape[1]}")

            # 步骤2: 指标预筛选
            screening_result = self.prescreener.prescreen_indicators(
                smart_indicators, target
            )
            final_indicators = screening_result["qualified_indicators"]
            print(f"  筛选后指标数: {len(final_indicators)}")
            print(f"  精简比例: {screening_result['reduction_ratio']:.1%}")

            # 存储结果
            results.append(
                {
                    "etf_code": etf_code,
                    "traditional_factors": traditional_factors.shape[1],
                    "smart_indicators": smart_indicators.shape[1],
                    "final_indicators": len(final_indicators),
                    "reduction_ratio": screening_result["reduction_ratio"],
                    "quality_metrics": screening_result["quality_metrics"],
                }
            )

        return results

    def _traditional_factor_factory(self, data: pd.DataFrame) -> pd.DataFrame:
        """模拟传统因子工厂 - 参数爆炸"""
        factors = {}
        close = data["close"].values

        # RSI - 8个周期（模仿原始VBTAdapter）
        for window in [6, 9, 12, 14, 20, 24, 30, 60]:
            import vectorbt as vbt

            rsi = vbt.RSI.run(close, window=window)
            factors[f"RSI_{window}"] = rsi.rsi.values

        # MA - 13个周期
        for window in [5, 10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 252]:
            ma = vbt.MA.run(close, window=window)
            factors[f"MA_{window}"] = ma.ma.values

        # BBANDS - 105个组合（7周期 × 3alpha × 5输出）
        for window in [10, 15, 20, 25, 30, 40, 50]:
            for alpha in [1.5, 2.0, 2.5]:
                bb = vbt.BBANDS.run(close, window=window, alpha=alpha)
                factors[f"BB_UPPER_{window}_{alpha}"] = bb.upper.values
                factors[f"BB_MIDDLE_{window}_{alpha}"] = bb.middle.values
                factors[f"BB_LOWER_{window}_{alpha}"] = bb.lower.values
                factors[f"BB_WIDTH_{window}_{alpha}"] = bb.bandwidth.values
                factors[f"BB_PERCENT_{window}_{alpha}"] = bb.percent.values

        return pd.DataFrame(factors, index=data.index)

    def demonstrate_optimization(self, data: pd.DataFrame):
        """演示优化效果"""

        print("\n" + "=" * 60)
        print("指标平台优化效果演示")
        print("=" * 60)

        # 使用第一个ETF演示
        etf_data = data[data["symbol"] == data["symbol"].iloc[0]].copy()
        target = self.generate_target_variable(etf_data["close"])

        print(f"\n原始数据: {len(etf_data)} 天")
        print(f"目标变量: 未来{self.smart_adapter.forward_period}日收益率")

        # 传统方法
        print(f"\n传统因子工厂:")
        traditional = self._traditional_factor_factory(etf_data)
        print(f"  生成因子: {traditional.shape[1]} 个")

        # 智能方法
        print(f"\n指标平台:")
        smart_result = self.optimized_adapter.compute_all_indicators(etf_data, target)
        print(f"  生成指标: {smart_result.shape[1]} 个")

        stats = self.optimized_adapter.get_indicator_stats()
        print(f"  精简比例: {stats['reduction_ratio']}")
        print(f"  智能选择: {'启用' if stats['smart_selection_enabled'] else '禁用'}")

        # 预筛选
        print(f"\n指标预筛选:")
        screening_result = self.prescreener.prescreen_indicators(smart_result, target)
        final_count = len(screening_result["qualified_indicators"])
        print(f"  筛选后指标: {final_count} 个")
        print(
            f"  总体精简比例: {(smart_result.shape[1] - final_count) / smart_result.shape[1]:.1%}"
        )

        # 质量分析
        quality_metrics = screening_result["quality_metrics"]
        if quality_metrics:
            avg_ic = np.mean([m.ic_mean for m in quality_metrics.values()])
            avg_ic_ir = np.mean([m.ic_ir for m in quality_metrics.values()])
            print(f"  平均IC: {avg_ic:.4f}")
            print(f"  平均IC_IR: {avg_ic_ir:.4f}")

        return {
            "traditional_count": traditional.shape[1],
            "smart_count": smart_result.shape[1],
            "final_count": final_count,
            "total_reduction": (traditional.shape[1] - final_count)
            / traditional.shape[1],
            "quality_report": screening_result["quality_report"],
        }


def main():
    """主函数"""

    print("🎯 指标平台架构演示")
    print("从'因子工厂'转向'指标平台' - 基于预测力的智能参数选择")

    # 创建演示实例
    demo = IndicatorPlatformDemo()

    # 生成示例数据
    print("\n📊 生成示例数据...")
    data = demo.create_sample_data(n_days=500, n_etfs=3)
    print(f"生成数据: {len(data)} 条记录，{data['symbol'].nunique()} 只ETF")

    # 对比两种方法
    print("\n🔍 对比传统方法 vs 指标平台...")
    comparison_results = demo.compare_approaches(data)

    # 显示对比结果
    print("\n📈 对比结果汇总:")
    print("-" * 40)

    total_traditional = 0
    total_final = 0

    for result in comparison_results:
        print(f"ETF {result['etf_code']}:")
        print(f"  传统因子: {result['traditional_factors']:3d} 个")
        print(f"  智能指标: {result['smart_indicators']:3d} 个")
        print(f"  最终指标: {result['final_indicators']:3d} 个")
        print(f"  精简比例: {result['reduction_ratio']:.1%}")
        print()

        total_traditional += result["traditional_factors"]
        total_final += result["final_indicators"]

    print(f"总计精简效果:")
    print(f"  传统方法: {total_traditional} 个因子")
    print(f"  指标平台: {total_final} 个指标")
    print(f"  整体精简: {(total_traditional - total_final) / total_traditional:.1%}")

    # 详细优化演示
    print("\n🔧 详细优化分析...")
    optimization_result = demo.demonstrate_optimization(data)

    print(f"\n🎯 核心优化点:")
    print(
        f"1. 参数空间压缩: {optimization_result['traditional_count']} → {optimization_result['smart_count']} (减少{(optimization_result['traditional_count'] - optimization_result['smart_count']) / optimization_result['traditional_count']:.1%})"
    )
    print(
        f"2. 质量预筛选: {optimization_result['smart_count']} → {optimization_result['final_count']} (再减少{(optimization_result['smart_count'] - optimization_result['final_count']) / optimization_result['smart_count']:.1%})"
    )
    print(f"3. 总体效果: 精简 {optimization_result['total_reduction']:.1%}")

    print(f"\n✅ 指标平台优势:")
    print("  ✓ 从参数爆炸转向智能选择")
    print("  ✓ 基于IC/IR的动态参数优化")
    print("  ✓ 指标生成阶段的质量控制")
    print("  ✓ 去除高相关性重复指标")
    print("  ✓ 保持预测力的同时大幅精简")

    print(f"\n🚀 架构升级完成！")


if __name__ == "__main__":
    main()
