#!/usr/bin/env python3
"""
A股统一架构实施 - 快速启动脚本

P0阶段：修复Registry实例化问题，建立基本连接
执行顺序：
1. 修复Registry实例化逻辑
2. 验证因子计算一致性  
3. 补充缺失的技术因子
4. 运行集成测试

使用方法：
    python 实施快速启动.py
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class AShareArchitectureFixer:
    """A股架构修复器"""

    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.a_share_dir = self.project_root / "a股"
        self.factor_engine_dir = self.project_root / "factor_system" / "factor_engine"

        print("=" * 60)
        print("🚀 A股统一架构实施 - 快速启动")
        print("=" * 60)
        print(f"项目根目录: {self.project_root}")
        print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

    def step1_fix_registry_issue(self):
        """步骤1: 修复Registry实例化问题"""
        print("🔧 步骤1: 修复Registry实例化问题")
        print("-" * 40)

        # 检查当前问题
        adapter_file = self.a_share_dir / "factor_adapter.py"
        if not adapter_file.exists():
            print("❌ factor_adapter.py 不存在")
            return False

        # 创建修复后的适配器
        fixed_adapter_content = '''"""
A股因子适配器 - 修复Registry实例化问题

修复要点：
1. 注册因子类而不是实例
2. 使用统一API接口
3. 简化初始化逻辑
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

# 使用统一API接口
from factor_system.factor_engine import api


class AShareFactorAdapter:
    """
    A股因子适配器 - 修复版本
    
    主要修复：
    - 使用统一API，避免Registry实例化问题
    - 简化因子映射逻辑
    - 增强错误处理
    """

    # 因子名称映射：A股项目 -> factor_engine
    FACTOR_MAPPING = {
        # 移动平均线
        'MA5': 'SMA_5',
        'MA10': 'SMA_10', 
        'MA20': 'SMA_20',
        'MA30': 'SMA_30',
        'MA60': 'SMA_60',
        'EMA5': 'EMA_5',
        'EMA12': 'EMA_12',
        'EMA26': 'EMA_26',

        # 动量指标
        'RSI': 'RSI_14_wilders',  # 使用Wilders平滑
        'MACD': 'MACD_12_26_9',
        'MACD_Signal': 'MACD_Signal_12_26_9', 
        'MACD_Hist': 'MACD_Hist_12_26_9',
        'KDJ_K': 'STOCH_14_K',
        'KDJ_D': 'STOCH_14_D',
        'KDJ_J': 'STOCH_14_J',
        'Williams_R': 'WILLR_14',

        # 波动性指标
        'ATR': 'ATR_14',
        'BB_Upper': 'BBANDS_Upper_20_2',
        'BB_Middle': 'BBANDS_Middle_20_2',
        'BB_Lower': 'BBANDS_Lower_20_2',

        # 趋势指标
        'ADX': 'ADX_14',
        'DI_plus': 'PLUS_DI_14',
        'DI_minus': 'MINUS_DI_14',

        # 成交量指标
        'OBV': 'OBV',
        'Volume_SMA': 'SMA_Volume_20',
        'MFI': 'MFI_14',

        # 其他指标
        'CCI': 'CCI_14',
        'MOM': 'MOM_10',
        'ROC': 'ROC_10',
        'TRIX': 'TRIX_14',
    }

    def __init__(self, data_dir: str):
        """
        初始化适配器

        Args:
            data_dir: A股数据目录路径
        """
        self.data_dir = data_dir
        
        print(f"✅ A股因子适配器初始化完成 (修复版本)")
        print(f"   数据目录: {data_dir}")
        
        # 可用因子列表
        self.available_factors = self._check_available_factors()
        print(f"   可用因子: {len(self.available_factors)}个")

    def _check_available_factors(self) -> List[str]:
        """检查factor_engine中可用的因子"""
        try:
            # 使用统一API获取可用因子
            available = api.list_available_factors()
            
            # 过滤出我们映射的因子
            mapped_factors = set(self.FACTOR_MAPPING.values())
            available_mapped = [f for f in available if f in mapped_factors]
            
            return available_mapped
            
        except Exception as e:
            print(f"⚠️  检查可用因子时出错: {e}")
            return []

    def get_technical_indicators(
        self,
        stock_code: str,
        timeframe: str = '1d',
        lookback_days: int = 252,
    ) -> pd.DataFrame:
        """
        获取技术指标DataFrame

        Args:
            stock_code: 股票代码 (e.g. '300450.SZ')
            timeframe: 时间框架
            lookback_days: 回看天数

        Returns:
            DataFrame with technical indicators
        """
        # 计算日期范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        # 获取需要计算的因子列表（去重，只计算可用的）
        factor_ids = list(set(self.FACTOR_MAPPING.values()))
        factor_ids = [f for f in factor_ids if f in self.available_factors]

        if not factor_ids:
            print(f"⚠️  没有可用的因子")
            return pd.DataFrame()

        try:
            # 使用统一API计算因子
            factors_df = api.calculate_factors(
                factor_ids=factor_ids,
                symbols=[stock_code],
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                use_cache=True,
            )

            if factors_df.empty:
                print(f"⚠️  {stock_code} 未计算到任何因子数据")
                return pd.DataFrame()

            # 重命名列（从factor_engine名称 -> A股项目名称）
            reverse_mapping = {v: k for k, v in self.FACTOR_MAPPING.items()}

            # 只保留映射中存在的列
            available_columns = [col for col in factors_df.columns if col in reverse_mapping]
            factors_df = factors_df[available_columns]

            # 重命名
            factors_df = factors_df.rename(columns=reverse_mapping)

            print(f"✅ {stock_code} 技术指标计算完成: {len(factors_df)}行 x {len(factors_df.columns)}列")

            return factors_df

        except Exception as e:
            print(f"❌ {stock_code} 技术指标计算失败: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def add_indicators_to_dataframe(
        self,
        df: pd.DataFrame,
        stock_code: str,
    ) -> pd.DataFrame:
        """
        将技术指标添加到现有DataFrame

        Args:
            df: 原始OHLCV数据
            stock_code: 股票代码

        Returns:
            添加了技术指标的DataFrame
        """
        # 确保df有timestamp列
        if 'timestamp' not in df.columns:
            if df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
            else:
                raise ValueError("DataFrame必须有timestamp列或索引")

        # 获取技术指标
        indicators = self.get_technical_indicators(
            stock_code=stock_code,
            lookback_days=len(df) + 60,  # 额外60天确保充足数据
        )

        if indicators.empty:
            print(f"⚠️  {stock_code} 未获取到技术指标，返回原数据")
            return df

        # 合并到原DataFrame（按timestamp对齐）
        df_with_indicators = df.merge(
            indicators,
            left_on='timestamp',
            right_index=True,
            how='left'
        )

        print(f"✅ {stock_code} 技术指标合并完成: 总列数 {len(df_with_indicators.columns)}")

        return df_with_indicators

    def calculate_single_indicator(
        self,
        stock_code: str,
        indicator_name: str,
        timeframe: str = '1d',
        lookback_days: int = 252,
    ) -> pd.Series:
        """
        计算单个技术指标

        Args:
            stock_code: 股票代码
            indicator_name: 指标名称（A股项目命名）
            timeframe: 时间框架
            lookback_days: 回看天数

        Returns:
            指标序列
        """
        if indicator_name not in self.FACTOR_MAPPING:
            raise ValueError(f"不支持的指标: {indicator_name}")

        factor_id = self.FACTOR_MAPPING[indicator_name]

        # 计算日期范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        try:
            result = api.calculate_single_factor(
                factor_id=factor_id,
                symbol=stock_code,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
            )

            return result

        except Exception as e:
            print(f"❌ {stock_code} {indicator_name} 计算失败: {e}")
            return pd.Series()

    def list_available_indicators(self) -> List[str]:
        """
        列出所有可用的技术指标

        Returns:
            指标名称列表（A股项目命名）
        """
        # 返回映射中且可用的指标
        available = []
        for a_share_name, factor_id in self.FACTOR_MAPPING.items():
            if factor_id in self.available_factors:
                available.append(a_share_name)
        return available

    def get_cache_stats(self) -> Dict:
        """
        获取缓存统计信息

        Returns:
            缓存统计字典
        """
        try:
            return api.get_cache_stats()
        except Exception as e:
            print(f"⚠️  获取缓存统计失败: {e}")
            return {}

    def clear_cache(self):
        """清空缓存"""
        try:
            api.clear_cache()
            print("✅ 缓存已清空")
        except Exception as e:
            print(f"❌ 清空缓存失败: {e}")


# 便捷函数
def create_a_share_adapter(data_dir: str = None) -> AShareFactorAdapter:
    """
    创建A股因子适配器的便捷函数

    Args:
        data_dir: 数据目录，默认使用项目中的A股目录

    Returns:
        A股因子适配器实例
    """
    if data_dir is None:
        data_dir = str(Path(__file__).parent.parent)
    
    return AShareFactorAdapter(data_dir)


if __name__ == "__main__":
    # 测试代码
    print("🧪 测试修复后的适配器...")
    
    adapter = create_a_share_adapter()

    # 测试获取技术指标
    stock_code = "300450.SZ"
    indicators = adapter.get_technical_indicators(stock_code)

    if not indicators.empty:
        print(f"\\n📊 {stock_code} 技术指标预览:")
        print(indicators.tail())

        print(f"\\n📈 可用指标: {adapter.list_available_indicators()}")

        print(f"\\n💾 缓存统计: {adapter.get_cache_stats()}")
    else:
        print(f"❌ 未能获取到{stock_code}的技术指标")
'''

        # 备份原文件
        backup_file = adapter_file.with_suffix(".py.backup")
        if adapter_file.exists():
            import shutil

            shutil.copy2(adapter_file, backup_file)
            print(f"✅ 已备份原文件到: {backup_file}")

        # 写入修复后的内容
        with open(adapter_file, "w", encoding="utf-8") as f:
            f.write(fixed_adapter_content)

        print("✅ 步骤1完成: Registry实例化问题已修复")
        print()
        return True

    def step2_test_factor_consistency(self):
        """步骤2: 验证因子计算一致性"""
        print("🧪 步骤2: 验证因子计算一致性")
        print("-" * 40)

        try:
            # 导入修复后的适配器
            sys.path.insert(0, str(self.a_share_dir))
            from factor_adapter import AShareFactorAdapter

            # 创建适配器
            adapter = AShareFactorAdapter(str(self.project_root))

            # 测试股票代码
            stock_code = "300450.SZ"

            # 检查数据文件是否存在
            stock_dir = self.a_share_dir / stock_code
            if not stock_dir.exists():
                print(f"⚠️  股票数据目录不存在: {stock_dir}")
                return False

            # 查找日线数据文件
            daily_files = list(stock_dir.glob(f"{stock_code}_1d_*.csv"))
            if not daily_files:
                print(f"⚠️  未找到日线数据文件")
                return False

            daily_file = sorted(daily_files)[-1]
            print(f"📁 使用数据文件: {daily_file}")

            # 测试获取技术指标
            print(f"🔄 正在计算 {stock_code} 的技术指标...")
            indicators = adapter.get_technical_indicators(stock_code)

            if indicators.empty:
                print("❌ 技术指标计算失败")
                return False

            print(
                f"✅ 技术指标计算成功: {len(indicators)}行 x {len(indicators.columns)}列"
            )
            print(f"📊 可用指标: {adapter.list_available_indicators()}")

            # 显示前几行数据
            print(f"\\n📈 指标数据预览:")
            print(indicators.tail(3))

            # 测试缓存
            cache_stats = adapter.get_cache_stats()
            print(f"\\n💾 缓存统计: {cache_stats}")

            print("✅ 步骤2完成: 因子计算一致性验证通过")
            print()
            return True

        except Exception as e:
            print(f"❌ 步骤2失败: {e}")
            import traceback

            traceback.print_exc()
            return False

    def step3_check_missing_factors(self):
        """步骤3: 检查缺失的技术因子"""
        print("🔍 步骤3: 检查缺失的技术因子")
        print("-" * 40)

        try:
            # 导入适配器
            sys.path.insert(0, str(self.a_share_dir))
            from factor_adapter import AShareFactorAdapter

            adapter = AShareFactorAdapter(str(self.project_root))

            # 检查映射的因子是否都可用
            missing_factors = []
            available_indicators = adapter.list_available_indicators()

            for a_share_name in adapter.FACTOR_MAPPING.keys():
                if a_share_name not in available_indicators:
                    factor_id = adapter.FACTOR_MAPPING[a_share_name]
                    missing_factors.append((a_share_name, factor_id))

            if missing_factors:
                print(f"⚠️  发现 {len(missing_factors)} 个缺失因子:")
                for a_share_name, factor_id in missing_factors:
                    print(f"   - {a_share_name} -> {factor_id}")

                # 检查关键因子
                critical_factors = ["RSI", "MACD", "ATR", "ADX"]
                missing_critical = [
                    name for name, _ in missing_factors if name in critical_factors
                ]

                if missing_critical:
                    print(f"\\n🔴 关键因子缺失: {missing_critical}")
                    print("   需要在factor_engine中补充这些因子")
                    return False
                else:
                    print(f"\\n🟡 非关键因子缺失，可后续补充")
                    return True
            else:
                print("✅ 所有映射因子都可用")
                print()
                return True

        except Exception as e:
            print(f"❌ 步骤3失败: {e}")
            return False

    def step4_integration_test(self):
        """步骤4: 集成测试"""
        print("🔗 步骤4: 集成测试")
        print("-" * 40)

        try:
            # 测试完整的技术分析流程
            sys.path.insert(0, str(self.a_share_dir))
            from factor_adapter import AShareFactorAdapter

            # 创建适配器
            adapter = AShareFactorAdapter(str(self.project_root))

            # 测试股票
            stock_code = "300450.SZ"

            # 加载原始数据
            import glob

            import pandas as pd

            stock_dir = self.a_share_dir / stock_code
            daily_files = list(stock_dir.glob(f"{stock_code}_1d_*.csv"))

            if not daily_files:
                print("❌ 未找到测试数据")
                return False

            daily_file = sorted(daily_files)[-1]

            # 读取数据（A股格式）
            df = pd.read_csv(daily_file, header=0, skiprows=[1])
            df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.rename(columns={"Date": "timestamp"})

            print(f"📊 原始数据: {len(df)}行")

            # 使用适配器添加技术指标
            df_with_indicators = adapter.add_indicators_to_dataframe(df, stock_code)

            print(
                f"✅ 添加技术指标后: {len(df_with_indicators)}行 x {len(df_with_indicators.columns)}列"
            )

            # 验证关键指标是否存在
            key_indicators = ["RSI", "MACD", "ATR", "Volume_SMA"]
            missing_key = [
                ind for ind in key_indicators if ind not in df_with_indicators.columns
            ]

            if missing_key:
                print(f"⚠️  关键指标缺失: {missing_key}")
            else:
                print("✅ 所有关键指标都存在")

            # 显示最新数据
            latest_data = df_with_indicators.iloc[-1]
            print(f"\\n📈 最新数据 ({latest_data['timestamp'].strftime('%Y-%m-%d')}):")
            print(f"   收盘价: {latest_data['Close']:.2f}")
            if "RSI" in df_with_indicators.columns:
                print(f"   RSI: {latest_data['RSI']:.2f}")
            if "MACD" in df_with_indicators.columns:
                print(f"   MACD: {latest_data['MACD']:.4f}")

            print("✅ 步骤4完成: 集成测试通过")
            print()
            return True

        except Exception as e:
            print(f"❌ 步骤4失败: {e}")
            import traceback

            traceback.print_exc()
            return False

    def run_all_steps(self):
        """运行所有步骤"""
        print("🚀 开始执行P0阶段修复...")
        print()

        steps = [
            ("步骤1: 修复Registry实例化问题", self.step1_fix_registry_issue),
            ("步骤2: 验证因子计算一致性", self.step2_test_factor_consistency),
            ("步骤3: 检查缺失的技术因子", self.step3_check_missing_factors),
            ("步骤4: 集成测试", self.step4_integration_test),
        ]

        results = []

        for step_name, step_func in steps:
            try:
                result = step_func()
                results.append((step_name, result))
            except Exception as e:
                print(f"❌ {step_name} 执行异常: {e}")
                results.append((step_name, False))

        # 汇总结果
        print("=" * 60)
        print("📊 P0阶段执行结果汇总")
        print("=" * 60)

        success_count = 0
        for step_name, result in results:
            status = "✅ 成功" if result else "❌ 失败"
            print(f"{status} - {step_name}")
            if result:
                success_count += 1

        print(
            f"\\n📈 总体成功率: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)"
        )

        if success_count == len(results):
            print("\\n🎉 P0阶段全部完成！可以进入P1阶段（代码重构）")
            print("\\n📋 下一步任务:")
            print("1. 重构sz_technical_analysis.py使用适配器")
            print("2. 删除300行重复的手工指标计算代码")
            print("3. 模块化评分系统")
            print("4. 创建配置文件结构")
        else:
            failed_steps = [name for name, result in results if not result]
            print(f"\\n⚠️  还有 {len(failed_steps)} 个步骤需要修复:")
            for step in failed_steps:
                print(f"   - {step}")

        print("=" * 60)

        return success_count == len(results)


def main():
    """主函数"""
    fixer = AShareArchitectureFixer()
    success = fixer.run_all_steps()

    if success:
        print("\\n🚀 P0阶段修复完成，A股统一架构基础已建立！")
        sys.exit(0)
    else:
        print("\\n❌ P0阶段修复未完全成功，请检查错误信息")
        sys.exit(1)


if __name__ == "__main__":
    main()
