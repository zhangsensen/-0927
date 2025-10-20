#!/bin/bash
# 验证实验管线设置

set -e

echo "🔍 验证实验管线设置..."
echo "================================"

# 1. 检查目录结构
echo ""
echo "📁 检查目录结构..."
for dir in \
    "strategies/experiments" \
    "strategies/experiments/experiment_configs" \
    "strategies/results/experiments"
do
    if [ -d "$dir" ]; then
        echo "  ✅ $dir"
    else
        echo "  ❌ $dir (不存在)"
        exit 1
    fi
done

# 2. 检查配置文件
echo ""
echo "📋 检查配置文件..."
for config in \
    "strategies/experiments/experiment_configs/p0_weight_grid_coarse.yaml" \
    "strategies/experiments/experiment_configs/p0_weight_grid_fine.yaml" \
    "strategies/experiments/experiment_configs/p0_topn_scan.yaml" \
    "strategies/experiments/experiment_configs/p0_cost_sensitivity.yaml"
do
    if [ -f "$config" ]; then
        echo "  ✅ $(basename $config)"
    else
        echo "  ❌ $(basename $config) (不存在)"
        exit 1
    fi
done

# 3. 检查脚本文件
echo ""
echo "🔧 检查脚本文件..."
for script in \
    "strategies/vectorbt_multifactor_grid.py" \
    "strategies/experiments/run_experiments.py" \
    "strategies/experiments/aggregate_results.py"
do
    if [ -f "$script" ]; then
        echo "  ✅ $(basename $script)"
    else
        echo "  ❌ $(basename $script) (不存在)"
        exit 1
    fi
done

# 4. 测试 YAML 配置加载
echo ""
echo "🧪 测试 YAML 配置加载..."
python3 -c "
import yaml
from pathlib import Path

config_path = Path('strategies/experiments/experiment_configs/p0_weight_grid_coarse.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

assert 'experiment' in config, 'Missing experiment section'
assert 'parameters' in config, 'Missing parameters section'
assert 'weight-grid' in config['parameters'], 'Missing weight-grid'
assert 'fees' in config['parameters'], 'Missing fees'

print('  ✅ YAML 配置格式正确')
"

# 5. 测试 Dry Run
echo ""
echo "🚀 测试 Dry Run..."
python3 strategies/experiments/run_experiments.py \
    --config strategies/experiments/experiment_configs/p0_weight_grid_coarse.yaml \
    --dry-run \
    | grep -q "DRY RUN"

if [ $? -eq 0 ]; then
    echo "  ✅ Dry Run 测试通过"
else
    echo "  ❌ Dry Run 测试失败"
    exit 1
fi

# 6. 检查依赖
echo ""
echo "📦 检查 Python 依赖..."
python3 -c "
import sys
missing = []

try:
    import yaml
except ImportError:
    missing.append('pyyaml')

try:
    import pandas
except ImportError:
    missing.append('pandas')

try:
    import numpy
except ImportError:
    missing.append('numpy')

try:
    import matplotlib
except ImportError:
    missing.append('matplotlib')

try:
    import seaborn
except ImportError:
    missing.append('seaborn')

if missing:
    print('  ❌ 缺少依赖: ' + ', '.join(missing))
    print('  安装命令: pip install ' + ' '.join(missing))
    sys.exit(1)
else:
    print('  ✅ 所有依赖已安装')
"

echo ""
echo "================================"
echo "✅ 验证完成！实验管线设置正确"
echo ""
echo "📚 下一步："
echo "  1. 运行单个实验："
echo "     python strategies/experiments/run_experiments.py \\"
echo "         --config strategies/experiments/experiment_configs/p0_weight_grid_coarse.yaml"
echo ""
echo "  2. 运行所有 P0 实验："
echo "     python strategies/experiments/run_experiments.py --pattern 'p0_*.yaml'"
echo ""
echo "  3. 查看 README："
echo "     cat strategies/experiments/README.md"
