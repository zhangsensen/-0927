#!/bin/bash
# ETF轮动VBT回测引擎 - 配置化版本使用示例

# 设置工作目录
cd "$(dirname "$0")"

echo "=== ETF轮动VBT回测引擎 - 配置化版本使用示例 ==="
echo

# 示例1: 查看可用预设
echo "1. 查看可用预设:"
python backtest_engine_configurable.py --list-presets
echo

# 示例2: 显示预设配置
echo "2. 显示快速测试预设配置:"
python backtest_engine_configurable.py --preset quick_test --show-config
echo

# 示例3: 运行快速测试（使用实际数据）
echo "3. 运行快速测试预设:"
python backtest_engine_configurable.py \
    --preset quick_test \
    --panel /Users/zhangshenshen/深度量化0927/etf_rotation_system/data/results/panels/panel_20251020_104106/panel.parquet \
    --screening /Users/zhangshenshen/深度量化0927/etf_rotation_system/data/results/screening/screening_20251020_104628/passed_factors.csv \
    --price-dir /Users/zhangshenshen/深度量化0927/raw/ETF/daily
echo

# 示例4: 运行标准回测（自定义组合数）
echo "4. 运行标准回测（1000组合）:"
python backtest_engine_configurable.py \
    --preset standard \
    --max-combos 1000 \
    --panel /Users/zhangshenshen/深度量化0927/etf_rotation_system/data/results/panels/panel_20251020_104106/panel.parquet \
    --screening /Users/zhangshenshen/深度量化0927/etf_rotation_system/data/results/screening/screening_20251020_104628/passed_factors.csv \
    --price-dir /Users/zhangshenshen/深度量化0927/raw/ETF/daily
echo

# 示例5: 完全自定义参数
echo "5. 完全自定义参数示例:"
python backtest_engine_configurable.py \
    --panel /Users/zhangshenshen/深度量化0927/etf_rotation_system/data/results/panels/panel_20251020_104106/panel.parquet \
    --screening /Users/zhangshenshen/深度量化0927/etf_rotation_system/data/results/screening/screening_20251020_104628/passed_factors.csv \
    --price-dir /Users/zhangshenshen/深度量化0927/raw/ETF/daily \
    --max-combos 500 \
    --top-k 5
echo

# 示例6: 使用自定义配置文件（如果存在）
if [ -f "custom_config.yaml" ]; then
    echo "6. 使用自定义配置文件:"
    python backtest_engine_configurable.py --config custom_config.yaml
    echo
fi

echo "=== 所有示例完成 ==="
echo
echo "结果文件保存在 results/ 目录中："
echo "- backtest_results_*.csv: 回测结果表格"
echo "- best_strategy_*.json: 最优策略配置"
echo
echo "可以通过修改 backtest_config.yaml 来调整默认参数"