#!/bin/bash
# 顺序运行剩余实验（Exp9-11）
# 简单直接，不使用复杂的shell特性

set -e

cd /Users/zhangshenshen/深度量化0927/etf_rotation_optimized

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  🧪 顺序运行剩余3个实验                                    ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# ========== Exp9 ==========
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [1/3] Exp9: threshold=0.90, beta=0.0"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python3 scripts/apply_experiment_config.py exp9
if [ $? -ne 0 ]; then
    echo "❌ Exp9配置失败"
    exit 1
fi

echo ""
echo "运行WFO..."
python3 scripts/step3_run_wfo.py
if [ $? -ne 0 ]; then
    echo "❌ Exp9 WFO失败"
    exit 1
fi

# 保存结果
latest=$(ls -t results/wfo/20*/wfo_results.pkl | head -n1)
cp "$latest" results/wfo/exp9.pkl
echo "✅ Exp9 完成！"
echo ""
sleep 2

# ========== Exp10 ==========
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [2/3] Exp10: threshold=0.88, beta=0.8"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python3 scripts/apply_experiment_config.py exp10
if [ $? -ne 0 ]; then
    echo "❌ Exp10配置失败"
    exit 1
fi

echo ""
echo "运行WFO..."
python3 scripts/step3_run_wfo.py
if [ $? -ne 0 ]; then
    echo "❌ Exp10 WFO失败"
    exit 1
fi

latest=$(ls -t results/wfo/20*/wfo_results.pkl | head -n1)
cp "$latest" results/wfo/exp10.pkl
echo "✅ Exp10 完成！"
echo ""
sleep 2

# ========== Exp11 ==========
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [3/3] Exp11: threshold=0.90, beta=0.8"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python3 scripts/apply_experiment_config.py exp11
if [ $? -ne 0 ]; then
    echo "❌ Exp11配置失败"
    exit 1
fi

echo ""
echo "运行WFO..."
python3 scripts/step3_run_wfo.py
if [ $? -ne 0 ]; then
    echo "❌ Exp11 WFO失败"
    exit 1
fi

latest=$(ls -t results/wfo/20*/wfo_results.pkl | head -n1)
cp "$latest" results/wfo/exp11.pkl
echo "✅ Exp11 完成！"
echo ""

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  🎉 全部实验完成！                                         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# 检查结果
echo "📊 实验结果文件："
for exp in 7 8 9 10 11; do
    if [ $exp -eq 7 ]; then
        file="results/wfo/exp7_max8_beta08_FIXED.pkl"
    else
        file="results/wfo/exp${exp}.pkl"
    fi
    
    if [ -f "$file" ]; then
        size=$(ls -lh "$file" | awk '{print $5}')
        echo "  ✅ Exp${exp}: ${size}"
    else
        echo "  ❌ Exp${exp}: 未找到"
    fi
done
