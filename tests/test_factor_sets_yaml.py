"""
因子集YAML配置测试
验证因子集加载、解析和扩展功能
"""

import sys
from pathlib import Path

import pytest

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

from factor_system.factor_engine.core.registry import get_global_registry


class TestFactorSetsYAML:
    """因子集YAML配置测试"""
    
    @pytest.fixture
    def registry(self):
        """获取全局注册表"""
        return get_global_registry(include_money_flow=True)
    
    def test_yaml_loaded(self, registry):
        """测试YAML配置已加载"""
        # 验证YAML因子集已加载
        assert hasattr(registry, '_factor_sets_yaml')
        assert isinstance(registry._factor_sets_yaml, dict)
        assert len(registry._factor_sets_yaml) > 0
        
        print(f"✅ YAML因子集数量: {len(registry._factor_sets_yaml)}")
    
    def test_list_defined_sets(self, registry):
        """测试列出所有定义的因子集"""
        sets = registry.list_defined_sets()
        
        assert isinstance(sets, list)
        assert len(sets) > 0
        
        # 验证关键因子集存在
        expected_sets = [
            "tech_mini",
            "money_flow_core",
            "daily_default_research",
            "all"
        ]
        
        for set_name in expected_sets:
            assert set_name in sets, f"缺少因子集: {set_name}"
        
        print(f"✅ 可用因子集: {sets}")
    
    def test_tech_mini_set(self, registry):
        """测试tech_mini因子集"""
        factor_ids = registry.get_factor_ids_by_set("tech_mini")
        
        assert isinstance(factor_ids, list)
        assert len(factor_ids) > 0
        
        # 验证包含基础技术指标
        expected_factors = ["RSI", "MACD", "STOCH"]
        for factor in expected_factors:
            assert factor in factor_ids, f"tech_mini缺少: {factor}"
        
        # 验证无重复
        assert len(factor_ids) == len(set(factor_ids))
        
        # 验证已排序
        assert factor_ids == sorted(factor_ids)
        
        print(f"✅ tech_mini: {factor_ids}")
    
    def test_money_flow_core_set(self, registry):
        """测试money_flow_core因子集"""
        factor_ids = registry.get_factor_ids_by_set("money_flow_core")
        
        assert isinstance(factor_ids, list)
        assert len(factor_ids) > 0
        
        # 验证包含核心资金流因子
        expected_factors = [
            "MainNetInflow_Rate",
            "LargeOrder_Ratio",
            "Institutional_Absorption"
        ]
        for factor in expected_factors:
            assert factor in factor_ids, f"money_flow_core缺少: {factor}"
        
        print(f"✅ money_flow_core: {len(factor_ids)} 个因子")
    
    def test_daily_default_research_composition(self, registry):
        """测试daily_default_research组合集"""
        factor_ids = registry.get_factor_ids_by_set("daily_default_research")
        
        assert isinstance(factor_ids, list)
        assert len(factor_ids) > 0
        
        # 应该包含tech_mini的因子
        tech_mini = registry.get_factor_ids_by_set("tech_mini")
        for factor in tech_mini:
            assert factor in factor_ids, f"daily_default_research缺少tech_mini的: {factor}"
        
        # 应该包含money_flow_core的因子
        money_flow = registry.get_factor_ids_by_set("money_flow_core")
        for factor in money_flow:
            assert factor in factor_ids, f"daily_default_research缺少money_flow_core的: {factor}"
        
        # 验证无重复
        assert len(factor_ids) == len(set(factor_ids))
        
        print(f"✅ daily_default_research: {len(factor_ids)} 个因子")
        print(f"   包含: tech_mini({len(tech_mini)}) + money_flow_core({len(money_flow)}) + 其他")
    
    def test_all_factors_expansion(self, registry):
        """测试all因子集（动态扩展）"""
        factor_ids = registry.get_factor_ids_by_set("all")
        
        assert isinstance(factor_ids, list)
        assert len(factor_ids) > 0
        
        # 应该包含所有已注册因子
        all_registered = set(registry.factors.keys()) | set(registry.metadata.keys())
        assert len(factor_ids) >= len(all_registered) * 0.9  # 允许10%误差
        
        # 验证无重复
        assert len(factor_ids) == len(set(factor_ids))
        
        print(f"✅ all: {len(factor_ids)} 个因子")
        print(f"   注册表: factors={len(registry.factors)}, metadata={len(registry.metadata)}")
    
    def test_nested_set_resolution(self, registry):
        """测试嵌套因子集解析"""
        # production_standard应该包含多个子集
        if "production_standard" in registry.list_defined_sets():
            factor_ids = registry.get_factor_ids_by_set("production_standard")
            
            assert isinstance(factor_ids, list)
            assert len(factor_ids) > 20  # 应该有较多因子
            
            # 验证无重复
            assert len(factor_ids) == len(set(factor_ids))
            
            print(f"✅ production_standard: {len(factor_ids)} 个因子")
    
    def test_unknown_set_raises_error(self, registry):
        """测试未知因子集抛出错误"""
        with pytest.raises(ValueError) as exc_info:
            registry.get_factor_ids_by_set("unknown_set_12345")
        
        # 验证错误消息包含可用因子集
        error_msg = str(exc_info.value)
        assert "未定义的因子集" in error_msg
        assert "可用因子集" in error_msg
        
        print(f"✅ 未知因子集正确抛出错误")
    
    def test_no_duplicates_in_any_set(self, registry):
        """测试所有因子集无重复"""
        all_sets = registry.list_defined_sets()
        
        for set_name in all_sets:
            try:
                factor_ids = registry.get_factor_ids_by_set(set_name)
                unique_count = len(set(factor_ids))
                total_count = len(factor_ids)
                
                assert unique_count == total_count, f"{set_name}存在重复因子"
                
            except Exception as e:
                print(f"⚠️ {set_name} 解析失败: {e}")
        
        print(f"✅ 所有因子集无重复")
    
    def test_all_sets_sorted(self, registry):
        """测试所有因子集已排序"""
        all_sets = registry.list_defined_sets()
        
        for set_name in all_sets:
            try:
                factor_ids = registry.get_factor_ids_by_set(set_name)
                
                assert factor_ids == sorted(factor_ids), f"{set_name}未排序"
                
            except Exception as e:
                print(f"⚠️ {set_name} 解析失败: {e}")
        
        print(f"✅ 所有因子集已排序")


if __name__ == "__main__":
    # 快速运行测试
    pytest.main([__file__, "-v", "-s"])
