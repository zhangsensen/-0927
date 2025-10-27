#!/bin/bash
# 分批回测快速启动脚本

cd "$(dirname "$0")"

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║                   🚀 分批VBT回测执行器                                    ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "当前配置："
python3 -c "
import yaml
with open('parallel_backtest_config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
    
rebalance_freqs = config['backtest_config']['rebalance_freq_list']
top_n_list = config['backtest_config']['top_n_list']
max_combos = config['weight_grid'].get('max_combinations', 10000)

total = len(rebalance_freqs) * len(top_n_list) * max_combos

print(f'  调仓周期: {rebalance_freqs}')
print(f'  Top-N: {top_n_list}')
print(f'  权重组合: {max_combos:,}')
print(f'  总策略数: {total:,}')
print(f'')
print(f'分批方案:')
print(f'  总批次: {len(rebalance_freqs)} (每个调仓周期独立运行)')
print(f'  每批次策略数: ~{total//len(rebalance_freqs):,}')
print(f'  预计每批次耗时: 3-5分钟')
print(f'  总耗时估计: {len(rebalance_freqs)*4:.0f}分钟')
"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
read -p "确认开始分批回测？(y/N) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "⏳ 开始执行分批回测..."
    echo ""
    python3 batch_backtest_runner.py --batch-size 1
else
    echo ""
    echo "❌ 已取消"
fi
