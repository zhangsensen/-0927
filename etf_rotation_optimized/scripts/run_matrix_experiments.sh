#!/bin/bash
# 小矩阵调参实验批量运行
# 4个实验: threshold × beta 矩阵

set -e

BASE_DIR="/Users/zhangshenshen/深度量化0927"
ETF_DIR="${BASE_DIR}/etf_rotation_optimized"
RESULTS_DIR="${ETF_DIR}/results/wfo"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  🧪 小矩阵调参实验 - 批量运行                             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# 实验列表
experiments=(8 9 10 11)
exp_names=(
    "Exp8: threshold=0.88, beta=0.0"
    "Exp9: threshold=0.90, beta=0.0"
    "Exp10: threshold=0.88, beta=0.8"
    "Exp11: threshold=0.90, beta=0.8"
)

total=${#experiments[@]}
current=0

for i in "${!experiments[@]}"; do
    exp_num="${experiments[$i]}"
    exp_desc="${exp_names[$i]}"
    current=$((current + 1))
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [$current/$total] ${exp_desc}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # 应用配置
    echo "1️⃣ 应用实验配置..."
    cd "${ETF_DIR}"
    python scripts/apply_experiment_config.py "exp${exp_num}"
    
    if [ $? -ne 0 ]; then
        echo "❌ 配置应用失败！跳过实验 ${exp_num}"
        continue
    fi
    
    echo ""
    echo "2️⃣ 运行WFO..."
    cd "${BASE_DIR}"
    python scripts/step3_run_wfo.py 2>&1 | tee "${ETF_DIR}/logs/exp${exp_num}_run.log"
    
    if [ $? -ne 0 ]; then
        echo "❌ WFO运行失败！跳过保存"
        continue
    fi
    
    echo ""
    echo "3️⃣ 保存实验结果..."
    
    # 找到最新的wfo结果文件
    latest_result=$(ls -t "${RESULTS_DIR}"/2*/wfo_results.pkl 2>/dev/null | head -n1)
    
    if [ -z "$latest_result" ]; then
        echo "❌ 未找到WFO结果文件！"
        continue
    fi
    
    # 保存到实验编号
    cp "$latest_result" "${RESULTS_DIR}/exp${exp_num}.pkl"
    echo "✅ 已保存: exp${exp_num}.pkl"
    echo ""
    
    # 短暂休息
    if [ $current -lt $total ]; then
        echo "⏸  休息2秒后继续..."
        sleep 2
        echo ""
    fi
done

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  ✅ 批量实验完成！                                         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 结果文件:"
for exp_num in "${experiments[@]}"; do
    if [ -f "${RESULTS_DIR}/exp${exp_num}.pkl" ]; then
        size=$(du -h "${RESULTS_DIR}/exp${exp_num}.pkl" | cut -f1)
        echo "  ✅ exp${exp_num}.pkl (${size})"
    else
        echo "  ❌ exp${exp_num}.pkl (未生成)"
    fi
done
echo ""
echo "🔍 下一步: 运行对比分析脚本"
echo "   python scripts/compare_matrix_results.py"
