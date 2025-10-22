#!/bin/bash
# ETF轮动系统完整生产流程
# 使用自然筛选（无强制保留）

set -e  # 遇到错误立即退出

echo "================================================================================"
echo "🚀 ETF轮动系统 - 完整生产流程"
echo "================================================================================"
echo ""
echo "配置："
echo "  • 筛选方式: 自然筛选（无强制保留）"
echo "  • 最大因子数: 15"
echo "  • 回测组合数: 10,000"
echo "  • 并行进程: 8"
echo ""
echo "================================================================================"

# 步骤1: 生成因子面板
echo ""
echo "📊 步骤1/3: 生成因子面板（48因子）"
echo "================================================================================"
cd etf_rotation_system/01_横截面建设
python3 generate_panel_refactored.py \
    --data-dir ../../raw/ETF/daily \
    --output-dir ../data/results/panels \
    --config config/factor_panel_config.yaml \
    --workers 8

if [ $? -ne 0 ]; then
    echo "❌ 因子面板生成失败"
    exit 1
fi

# 获取最新面板路径
LATEST_PANEL=$(ls -td ../data/results/panels/panel_* | head -1)
echo "✅ 面板生成完成: $LATEST_PANEL"

# 步骤2: 因子筛选
echo ""
echo "🔬 步骤2/3: 因子筛选（自然筛选，无强制保留）"
echo "================================================================================"
cd ../02_因子筛选
python3 run_etf_cross_section_configurable.py \
    --config optimized_screening_config.yaml

if [ $? -ne 0 ]; then
    echo "❌ 因子筛选失败"
    exit 1
fi

# 获取最新筛选结果
LATEST_SCREENING=$(ls -td ../data/results/screening/screening_* | head -1)
echo "✅ 因子筛选完成: $LATEST_SCREENING"

# 显示筛选结果
echo ""
echo "筛选结果预览:"
python3 << 'EOF'
import pandas as pd
from pathlib import Path
import glob

screening_dirs = sorted(glob.glob("../data/results/screening/screening_*"))
if screening_dirs:
    latest = screening_dirs[-1]
    passed = pd.read_csv(f"{latest}/passed_factors.csv")
    print(f"  通过因子数: {len(passed)}")
    print("\n  因子列表:")
    for idx, row in passed.iterrows():
        is_rotation = any(x in row['factor'] for x in ['ROTATION', 'CS_RANK', 'RELATIVE'])
        marker = "🎯" if is_rotation else "  "
        print(f"    {marker} {row['factor']:30s}  IC={row['ic_mean']:+.4f}  IR={row['ic_ir']:+.4f}")
EOF

# 步骤3: VBT回测
echo ""
echo "⚡ 步骤3/3: VBT大规模回测（10,000组合）"
echo "================================================================================"
cd ../03_vbt回测
python3 large_scale_backtest_50k.py

if [ $? -ne 0 ]; then
    echo "❌ 回测失败"
    exit 1
fi

# 获取最新回测结果
LATEST_BACKTEST=$(ls -td ../data/results/backtest/backtest_* | head -1)
echo "✅ 回测完成: $LATEST_BACKTEST"

# 显示回测结果
echo ""
echo "================================================================================"
echo "📊 完整流程执行完成"
echo "================================================================================"
echo ""
echo "结果目录:"
echo "  • 因子面板: $LATEST_PANEL"
echo "  • 因子筛选: $LATEST_SCREENING"
echo "  • 回测结果: $LATEST_BACKTEST"
echo ""

# 显示Top 5策略
echo "🏆 Top 5策略:"
python3 << 'EOF'
import pandas as pd
import glob

backtest_dirs = sorted(glob.glob("../data/results/backtest/backtest_*"))
if backtest_dirs:
    latest = backtest_dirs[-1]
    results = pd.read_csv(f"{latest}/results.csv")
    top5 = results.head(5)
    
    for idx, row in top5.iterrows():
        print(f"\n  #{idx+1}: Sharpe={row['sharpe_ratio']:.4f} | Return={row['total_return']:.2f}% | DD={row['max_drawdown']:.2f}% | Top_N={int(row['top_n'])}")
EOF

echo ""
echo "================================================================================"
echo "✅ 全部完成！"
echo "================================================================================"
