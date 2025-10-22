#!/bin/bash
# ETF项目清理和优化脚本 - Phase 1 & 2

set -e

echo "🚀 ETF项目清理和优化执行脚本"
echo "================================"

# Phase 1: 即时清理
echo ""
echo "Phase 1: 即时清理 (10分钟)"
echo "================================"

# 1. 删除孤立脚本
echo "1. 删除孤立脚本..."
rm -fv etf_rotation_system/run_professional_screener.py

# 2. 备份旧配置文件
echo "2. 备份旧配置文件..."
mkdir -p scripts/legacy_configs
cp etf_rotation_system/01_横截面建设/config/config_classes.py scripts/legacy_configs/
cp etf_rotation_system/03_vbt回测/config_loader_parallel.py scripts/legacy_configs/
cp etf_rotation_system/03_vbt回测/parallel_backtest_config.yaml scripts/legacy_configs/
cp etf_rotation_system/02_因子筛选/etf_cross_section_config.py scripts/legacy_configs/

echo "✅ Phase 1 Complete"

# Phase 2: ConfigManager 迁移准备
echo ""
echo "Phase 2: ConfigManager 迁移"
echo "================================"

echo "检查 ConfigManager..."
if [ ! -f "etf_rotation_system/config/config_manager.py" ]; then
    echo "❌ ConfigManager 不存在"
    exit 1
fi

echo "✓ ConfigManager 存在"

# 检查配置文件
echo "检查配置文件..."
required_configs=(
    "etf_rotation_system/config/backtest_config.yaml"
    "etf_rotation_system/config/screening_config.yaml"
    "etf_rotation_system/config/factor_panel_config.yaml"
)

for cfg in "${required_configs[@]}"; do
    if [ -f "$cfg" ]; then
        echo "  ✓ $cfg"
    else
        echo "  ✗ $cfg (缺失)"
    fi
done

echo ""
echo "📋 下一步建议:"
echo ""
echo "1. 迁移 generate_panel_refactored.py:"
echo "   - 替换硬编码配置为 ConfigManager 调用"
echo "   - 测试: python3 etf_rotation_system/01_横截面建设/generate_panel_refactored.py"
echo ""
echo "2. 迁移 run_etf_cross_section_configurable.py:"
echo "   - 替换硬编码配置为 ConfigManager 调用"
echo "   - 测试: python3 etf_rotation_system/02_因子筛选/run_etf_cross_section_configurable.py"
echo ""
echo "3. 迁移 parallel_backtest_configurable.py:"
echo "   - 替换硬编码配置为 ConfigManager 调用"
echo "   - 测试: python3 etf_rotation_system/03_vbt回测/parallel_backtest_configurable.py --help"
echo ""
echo "4. 验证完整流程"
echo "5. 提交清理"
echo ""
