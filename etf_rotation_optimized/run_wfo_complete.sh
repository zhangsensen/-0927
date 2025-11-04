#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

echo "[WFO] Running end-to-end pipeline (cross_section -> factor_selection -> wfo)"

python - <<'PY'
from pathlib import Path
from etf_rotation_optimized.core.pipeline import Pipeline

cfg_path = Path("configs/default.yaml")
pipe = Pipeline.from_config(cfg_path)
pipe.run_step("cross_section")
pipe.run_step("factor_selection")
pipe.run_step("wfo")
print("[WFO] Completed.")
PY

echo "[WFO] Outputs saved under results/ directories with current timestamp."
#!/bin/bash
# WFO完整流程 - 包含收益计算和Top-N策略排序
# 2025-11-03

set -e

PROJECT_ROOT="/Users/zhangshenshen/深度量化0927/etf_rotation_optimized"
cd "$PROJECT_ROOT"

echo "========================================================================"
echo "WFO完整流程 - 真实数据 + 收益计算 + Top-N策略"
echo "========================================================================"
echo ""

# 清除缓存
echo "Step 0: 清除缓存..."
rm -rf cache/ .cache/ __pycache__ */__pycache__ */*/__pycache__ 2>/dev/null || true
echo "✅ 缓存已清除"
echo ""

# 运行完整WFO
echo "Step 1-3: 运行完整WFO流程..."
python -c "
from core.pipeline import Pipeline

# 加载配置
p = Pipeline.from_config('configs/default.yaml')

# Step 1: 横截面
print('\n[Step 1/3] 横截面加工...')
p.run_step('cross_section')

# Step 2: 因子筛选
print('\n[Step 2/3] 因子筛选...')
p.run_step('factor_selection')

# Step 3: WFO + 收益计算 + Top-N排序
print('\n[Step 3/3] WFO + 收益计算 + Top-N排序...')
p.run_step('wfo')

print('\n✅ 完整流程执行成功！')
"

echo ""
echo "========================================================================"
echo "✅ WFO完整流程执行成功！"
echo "========================================================================"
echo ""
echo "结果目录:"
echo "  - WFO IC汇总: results/wfo/*/wfo_summary.csv"
echo "  - WFO KPI: results/wfo/*/wfo_kpi_*.csv"
echo "  - 净值曲线: results/wfo/*/wfo_equity_*.csv"
echo "  - Top-N策略: results/wfo/*/top5_strategies.csv"
echo ""
