"""
A股数据提供者集成测试
验证分钟转日线、资金流合并、T+1时序安全
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

from factor_system.factor_engine.providers.parquet_provider import ParquetDataProvider
from factor_system.factor_engine.providers.combined_provider import CombinedMoneyFlowProvider


class TestAShareProviderIntegration:
    """A股数据提供者集成测试"""
    
    @pytest.fixture
    def project_root(self):
        """项目根目录"""
        return Path(__file__).parent.parent
    
    def test_minute_to_daily_resample(self, project_root):
        """测试分钟数据转日线"""
        provider = ParquetDataProvider(project_root / "raw")
        
        # 请求日线数据（应自动从分钟转换）
        data = provider.load_price_data(
            ["600036.SH"],
            "daily",
            datetime(2024, 8, 1),
            datetime(2024, 12, 31)
        )
        
        assert not data.empty, "日线数据不应为空"
        assert len(data) > 50, "应至少有50个交易日"
        
        # 验证OHLC逻辑
        symbol_data = data.xs("600036.SH", level="symbol")
        assert (symbol_data['high'] >= symbol_data['low']).all(), "high应>=low"
        assert (symbol_data['high'] >= symbol_data['open']).all(), "high应>=open"
        assert (symbol_data['high'] >= symbol_data['close']).all(), "high应>=close"
    
    def test_money_flow_merge(self, project_root):
        """测试资金流合并"""
        price_provider = ParquetDataProvider(project_root / "raw")
        combined = CombinedMoneyFlowProvider(
            price_provider=price_provider,
            money_flow_dir=project_root / "raw/SH/money_flow",
            enforce_t_plus_1=True
        )
        
        data = combined.load_price_data(
            ["600036.SH"],
            "daily",
            datetime(2024, 8, 1),
            datetime(2024, 12, 31)
        )
        
        assert not data.empty, "合并数据不应为空"
        assert 'close' in data.columns, "应包含价格字段"
        assert 'main_net' in data.columns, "应包含资金流字段"
        
        # 验证资金流数据非全NaN
        assert data['main_net'].notna().sum() > 0, "资金流数据不应全为NaN"
    
    def test_t_plus_1_safety(self, project_root):
        """测试T+1时序安全"""
        price_provider = ParquetDataProvider(project_root / "raw")
        combined = CombinedMoneyFlowProvider(
            price_provider=price_provider,
            money_flow_dir=project_root / "raw/SH/money_flow",
            enforce_t_plus_1=True
        )
        
        data = combined.load_price_data(
            ["600036.SH"],
            "daily",
            datetime(2024, 8, 1),
            datetime(2024, 12, 31)
        )
        
        symbol_data = data.xs("600036.SH", level="symbol")
        
        # 第一天资金流应为NaN（T+1）
        first_mf_value = symbol_data['main_net'].iloc[0]
        assert pd.isna(first_mf_value), "第一天资金流应为NaN（T+1滞后）"
        
        # 验证temporal_safe标记
        if 'temporal_safe' in symbol_data.columns:
            assert symbol_data['temporal_safe'].iloc[1], "应标记为时序安全"
    
    def test_unit_consistency(self, project_root):
        """测试单位一致性"""
        price_provider = ParquetDataProvider(project_root / "raw")
        combined = CombinedMoneyFlowProvider(
            price_provider=price_provider,
            money_flow_dir=project_root / "raw/SH/money_flow",
            enforce_t_plus_1=True
        )
        
        data = combined.load_price_data(
            ["600036.SH"],
            "daily",
            datetime(2024, 8, 1),
            datetime(2024, 12, 31)
        )
        
        # 验证单位标记
        if 'value_unit' in data.columns:
            unit_values = data['value_unit'].dropna().unique()
            assert 'yuan' in unit_values, "单位应为'yuan'"
        
        # 验证数值量级（main_net应为亿级，即8-10位数）
        symbol_data = data.xs("600036.SH", level="symbol")
        main_net_values = symbol_data['main_net'].dropna()
        if len(main_net_values) > 0:
            avg_abs = main_net_values.abs().mean()
            assert 1e6 < avg_abs < 1e11, f"main_net量级异常: {avg_abs}"
    
    def test_date_normalization(self, project_root):
        """测试日期归一化"""
        price_provider = ParquetDataProvider(project_root / "raw")
        combined = CombinedMoneyFlowProvider(
            price_provider=price_provider,
            money_flow_dir=project_root / "raw/SH/money_flow",
            enforce_t_plus_1=True
        )
        
        data = combined.load_price_data(
            ["600036.SH"],
            "daily",
            datetime(2024, 8, 1),
            datetime(2024, 12, 31)
        )
        
        # 验证日期索引已归一化（时分秒为00:00:00）
        if isinstance(data.index, pd.MultiIndex):
            dates = data.index.get_level_values(1)
            for date in dates[:5]:  # 检查前5个
                assert date.hour == 0, "小时应为0"
                assert date.minute == 0, "分钟应为0"
                assert date.second == 0, "秒应为0"


if __name__ == "__main__":
    # 快速运行测试
    pytest.main([__file__, "-v", "-s"])
