#!/bin/bash
# 完整WFO流程 - 使用真实数据
# 修复后的portfolio_constructor验证

set -e

PROJECT_ROOT="/Users/zhangshenshen/深度量化0927/etf_rotation_optimized"
cd "$PROJECT_ROOT"

echo "========================================================================"
echo "WFO完整流程 - 真实数据验证（修复后的portfolio_constructor）"
echo "========================================================================"
echo ""
echo "修复内容:"
echo "  ✅ 信号T-1延迟 - 无前视偏差"
echo "  ✅ 成本归一化 - 避免成本爆炸"
echo "  ✅ 成本率稳定 - 避免分母崩溃"
echo ""
echo "========================================================================"
echo ""

# 使用默认配置运行完整流程
CONFIG_FILE="configs/default.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

echo "📋 使用配置: $CONFIG_FILE"
echo ""

# Step 1: 横截面加工
echo "========================================================================"
echo "Step 1/3: 横截面加工（加载数据 + 计算因子）"
echo "========================================================================"
python -c "
from core.pipeline import Pipeline
import sys

try:
    p = Pipeline.from_config('$CONFIG_FILE')
    p.run_step('cross_section')
    print('✅ Step 1 完成')
except Exception as e:
    print(f'❌ Step 1 失败: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ 横截面加工失败"
    exit 1
fi

echo ""

# Step 2: 因子筛选（标准化）
echo "========================================================================"
echo "Step 2/3: 因子筛选（标准化处理）"
echo "========================================================================"
python -c "
from core.pipeline import Pipeline
import sys

try:
    p = Pipeline.from_config('$CONFIG_FILE')
    p.run_step('factor_selection')
    print('✅ Step 2 完成')
except Exception as e:
    print(f'❌ Step 2 失败: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ 因子筛选失败"
    exit 1
fi

echo ""

# Step 3: WFO验证
echo "========================================================================"
echo "Step 3/3: WFO验证（使用修复后的portfolio_constructor）"
echo "========================================================================"
python -c "
from core.pipeline import Pipeline
import sys

try:
    p = Pipeline.from_config('$CONFIG_FILE')
    p.run_step('wfo')
    print('✅ Step 3 完成')
except Exception as e:
    print(f'❌ Step 3 失败: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ WFO验证失败"
    exit 1
fi

echo ""
echo "========================================================================"
echo "✅ 完整流程执行成功！"
echo "========================================================================"
echo ""
echo "结果目录:"
echo "  - 横截面: results/cross_section/"
echo "  - 因子筛选: results/factor_selection/"
echo "  - WFO结果: results/wfo/"
echo ""
echo "验证要点:"
echo "  1. 检查WFO结果中的IC和Sharpe"
echo "  2. 确认无前视偏差（信号T-1延迟）"
echo "  3. 确认成本计算合理（无爆炸）"
echo ""
