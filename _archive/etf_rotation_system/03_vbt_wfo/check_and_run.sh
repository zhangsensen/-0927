#!/bin/bash
# 快速检查配置并启动 WFO

echo "=========================================="
echo "WFO 回测配置检查"
echo "=========================================="
echo ""

CONFIG_FILE="simple_config.yaml"

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 显示关键配置
echo "📋 关键配置:"
echo "  Top-N: $(grep 'top_n_list:' $CONFIG_FILE)"
echo "  调仓频率: $(grep 'rebalance_freq_list:' $CONFIG_FILE)"
echo "  权重组合: $(grep 'max_combinations:' $CONFIG_FILE)"
echo "  IS回测: $(grep 'run_is:' $CONFIG_FILE)"
echo "  OOS回测: $(grep 'run_oos:' $CONFIG_FILE)"
echo ""

# 计算策略数
echo "📊 预计策略数:"
echo "  250 权重组合 × 2 Top-N × 2 调仓频率 = 1000 策略/Period"
echo "  IS + OOS = 2000 策略/Period"
echo ""

read -p "确认启动? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 启动 WFO 回测..."
    python3 production_runner_optimized.py
else
    echo "❌ 已取消"
    exit 0
fi
