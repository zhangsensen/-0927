#!/bin/bash
# WFO 回测启动脚本
# 使用说明: cd 到 03_vbt_wfo 目录后执行 ./run_wfo.sh

cd "$(dirname "$0")"

echo "🚀 启动 WFO 生产环境回测..."
echo "📝 配置文件: simple_config.yaml"
echo ""

# 运行 WFO
python3 production_runner_optimized.py

echo ""
echo "✅ 回测完成！"
echo "📁 结果保存在: /Users/zhangshenshen/深度量化0927/etf_rotation_system/data/results/vbtwfo/"
