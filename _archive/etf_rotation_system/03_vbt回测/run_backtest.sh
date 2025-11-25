#!/bin/bash
# VBT回测启动脚本

echo "=== VBT回测配置 ==="
echo "工作目录: $(pwd)"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 进入正确的目录
cd "$(dirname "$0")"

# 显示配置
python3 << 'PYEOF'
import yaml
with open('parallel_backtest_config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
print("配置概览:")
print(f"  因子数: {config['factor_config']['top_k']}个")
print(f"  Top-N: {config['backtest_config']['top_n_list']}")
print(f"  调仓周期: {config['backtest_config']['rebalance_freq_list']}")
print(f"  费率: {config['backtest_config']['fees']*100:.2f}%")
print(f"  保存: Top {config['output_config']['save_top_results']}")
print("")
top_n = len(config['backtest_config']['top_n_list'])
rebal = len(config['backtest_config']['rebalance_freq_list'])
print(f"总策略数: {top_n * rebal * 10000:,}个")
print("="*60)
print("")
PYEOF

# 运行回测
echo "开始回测..."
python3 parallel_backtest_configurable.py

echo ""
echo "回测完成！"
