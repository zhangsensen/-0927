#!/bin/bash
#
# 执行延迟修复快速验证
# 使用 Platinum 策略测试 LAG=0 vs LAG=1 的收益差异
#

set -e

PROJECT_ROOT="/home/sensen/dev/projects/-0927/etf_rotation_experiments"
cd "$PROJECT_ROOT"

# 激活虚拟环境
source /home/sensen/dev/projects/-0927/.venv/bin/activate

echo "================================================================================"
echo "执行延迟修复验证测试"
echo "================================================================================"
echo ""
echo "策略配置:"
echo "  - Combo: OBV_SLOPE_10D, PRICE_POSITION_20D, RSI_14, SLOPE_20D, VORTEX_14D"
echo "  - Lookback: 120"
echo "  - Frequency: 2 天调仓"
echo "  - Position Size: 10"
echo ""

# 创建临时组合文件
COMBO_FILE="/tmp/platinum_combo.txt"
echo "OBV_SLOPE_10D,PRICE_POSITION_20D,RSI_14,SLOPE_20D,VORTEX_14D" > "$COMBO_FILE"

# 测试 1: LAG=0 (原始逻辑，Lag-1 IC)
echo "================================================================================"
echo "测试 1/2: RB_EXECUTION_LAG=0 (原始逻辑，存在前视偏差)"
echo "================================================================================"
echo ""

export RB_EXECUTION_LAG=0
export RB_DAILY_IC_PRECOMP=0

OUTPUT_LAG0="/tmp/wfo_lag0_output.txt"

python3 run_combo_wfo.py \
    --lookback 120 \
    --freq 2 \
    --position 10 \
    --combo-file "$COMBO_FILE" \
    --n-jobs 1 \
    2>&1 | tee "$OUTPUT_LAG0"

echo ""
echo "--- LAG=0 结果摘要 ---"
grep -E "(Annual Return|Max Drawdown|Sharpe)" "$OUTPUT_LAG0" || echo "未找到性能指标"
echo ""

# 测试 2: LAG=1 (修正逻辑，Lag-2 IC)
echo "================================================================================"
echo "测试 2/2: RB_EXECUTION_LAG=1 (修正逻辑，消除前视偏差)"
echo "================================================================================"
echo ""

export RB_EXECUTION_LAG=1
export RB_DAILY_IC_PRECOMP=0

OUTPUT_LAG1="/tmp/wfo_lag1_output.txt"

python3 run_combo_wfo.py \
    --lookback 120 \
    --freq 2 \
    --position 10 \
    --combo-file "$COMBO_FILE" \
    --n-jobs 1 \
    2>&1 | tee "$OUTPUT_LAG1"

echo ""
echo "--- LAG=1 结果摘要 ---"
grep -E "(Annual Return|Max Drawdown|Sharpe)" "$OUTPUT_LAG1" || echo "未找到性能指标"
echo ""

# 对比结果
echo "================================================================================"
echo "结果对比"
echo "================================================================================"
echo ""

echo "LAG=0 (原始逻辑，Lag-1 IC):"
grep -E "(Annual Return|Max Drawdown|Sharpe)" "$OUTPUT_LAG0" | head -3 || echo "  未找到指标"

echo ""
echo "LAG=1 (修正逻辑，Lag-2 IC):"
grep -E "(Annual Return|Max Drawdown|Sharpe)" "$OUTPUT_LAG1" | head -3 || echo "  未找到指标"

echo ""
echo "================================================================================"
echo "分析结论"
echo "================================================================================"
echo ""
echo "如果修复正确，应该观察到："
echo "  1. LAG=0 的年化收益明显高于 LAG=1（前视偏差导致高估）"
echo "  2. LAG=1 的收益接近 paper_trading 结果（-6% ~ 1%）"
echo "  3. LAG=0 的收益接近原始 WFO 结果（~20%）"
echo ""
echo "验证完成！"
echo "================================================================================"
