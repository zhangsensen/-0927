#!/bin/bash
# WFO修复验证 - 重新运行WFO测试修复后的portfolio_constructor

set -e

PROJECT_ROOT="/Users/zhangshenshen/深度量化0927/etf_rotation_optimized"
cd "$PROJECT_ROOT"

echo "=========================================================================="
echo "WFO修复验证 - 测试portfolio_constructor修复效果"
echo "=========================================================================="
echo ""

# 运行WFO pipeline（使用修复后的portfolio_constructor）
python -m pipeline from_config pipeline_config_wfo_test.yaml

echo ""
echo "=========================================================================="
echo "WFO运行完成！"
echo "=========================================================================="
echo ""
echo "检查结果目录: results/wfo/"
echo "验证修复效果: 确保没有前视偏差和成本爆炸错误"
