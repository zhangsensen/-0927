#!/bin/bash
# ETF轮动系统完整流程测试脚本
# 目的：验证横截面建设 → 因子筛选 → VBT回测的完整链路

set -e  # 遇到错误立即退出

PROJECT_ROOT="/Users/zhangshenshen/深度量化0927"
cd "$PROJECT_ROOT"

echo "======================================================================================================"
echo "ETF轮动系统完整流程测试"
echo "======================================================================================================"
echo ""

# ========== 步骤1: 横截面建设 ==========
echo "📊 步骤1/3: 横截面因子面板生成"
echo "------------------------------------------------------------------------------------------------------"
cd "$PROJECT_ROOT/etf_rotation_system/01_横截面建设"

python3 generate_panel_refactored.py \
    --data-dir ../../raw/ETF/daily \
    --output-dir ../data/results/panels \
    --workers 8

if [ $? -ne 0 ]; then
    echo "❌ 因子面板生成失败"
    exit 1
fi

# 获取最新生成的panel文件
LATEST_PANEL=$(ls -t ../data/results/panels/panel_*/panel.parquet | head -1)
echo "✅ 因子面板生成成功: $LATEST_PANEL"
echo ""

# ========== 步骤2: 因子筛选 ==========
echo "🎯 步骤2/3: 因子筛选（48→12因子）"
echo "------------------------------------------------------------------------------------------------------"
cd "$PROJECT_ROOT/etf_rotation_system/02_因子筛选"

python3 run_etf_cross_section_configurable.py --config optimized_screening_config.yaml

if [ $? -ne 0 ]; then
    echo "❌ 因子筛选失败"
    exit 1
fi

# 获取最新生成的筛选文件
LATEST_SCREENING=$(ls -t ../data/results/screening/screening_*/passed_factors.csv | head -1)
SCREENING_COUNT=$(wc -l < "$LATEST_SCREENING")
echo "✅ 因子筛选成功: $LATEST_SCREENING ($(($SCREENING_COUNT - 1))个因子)"
echo ""

# ========== 步骤3: VBT回测 ==========
echo "⚡ 步骤3/3: VBT回测（1万组合）"
echo "------------------------------------------------------------------------------------------------------"
cd "$PROJECT_ROOT/etf_rotation_system/03_vbt回测"

python3 large_scale_backtest_50k.py

if [ $? -ne 0 ]; then
    echo "❌ VBT回测失败"
    exit 1
fi

# 获取最新生成的回测文件
LATEST_BACKTEST=$(ls -t ../data/results/backtest/backtest_*/results.csv | head -1)
BACKTEST_DIR=$(dirname "$LATEST_BACKTEST")
BACKTEST_COUNT=$(wc -l < "$LATEST_BACKTEST")
echo "✅ VBT回测成功: $LATEST_BACKTEST ($(($BACKTEST_COUNT - 1))个策略)"
echo ""

# ========== 结果验证 ==========
echo "======================================================================================================"
echo "🎉 完整流程测试成功！"
echo "======================================================================================================"
echo ""
echo "📂 输出文件:"
echo "  1. 因子面板: $LATEST_PANEL"
echo "  2. 筛选结果: $LATEST_SCREENING"
echo "  3. 回测结果: $BACKTEST_DIR"
echo ""

# 快速统计
echo "📊 快速统计:"
python3 << EOF
import pandas as pd
import os

panel = pd.read_parquet("$LATEST_PANEL")
screening = pd.read_csv("$LATEST_SCREENING")
results = pd.read_csv("$LATEST_BACKTEST")

print(f"  • 因子面板: {panel.shape[0]:,}行 × {panel.shape[1]}个因子")
print(f"  • 筛选因子: {len(screening)}个核心因子")
print(f"  • 回测策略: {len(results):,}个")
print(f"  • Top Sharpe: {results['sharpe_ratio'].max():.4f}")
print(f"  • 平均Sharpe: {results['sharpe_ratio'].mean():.4f}")

# 检查轮动因子
top10 = results.nlargest(10, 'sharpe_ratio')
rotation_count = 0
cs_rank_count = 0
for _, row in top10.iterrows():
    weights = eval(row['weights'])
    if weights.get('ROTATION_SCORE', 0) > 0:
        rotation_count += 1
    if weights.get('CS_RANK_CHANGE_5D', 0) > 0:
        cs_rank_count += 1

print(f"\n  🎯 Top 10中轮动因子使用:")
print(f"    - ROTATION_SCORE: {rotation_count}/10")
print(f"    - CS_RANK_CHANGE_5D: {cs_rank_count}/10")
EOF

echo ""
echo "======================================================================================================"
echo "✅ 所有步骤完成，流程正常！"
echo "======================================================================================================"
