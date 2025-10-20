#!/usr/bin/env bash
set -euo pipefail

# 仓库根目录
ROOT="/Users/zhangshenshen/深度量化0927"
cd "$ROOT"

echo "===> 校验真实数据可用性"

panel="factor_output/etf_rotation/panel_optimized_v2_20200102_20251014.parquet"
if [[ ! -f "$panel" ]]; then
    echo "❌ 缺少因子面板: $panel" >&2
    exit 1
fi

price_dir="raw/ETF/daily"
if [[ ! -d "$price_dir" ]]; then
    echo "❌ 缺少价格目录: $price_dir" >&2
    exit 1
fi

python3 - <<'PY'
import pandas as pd
from pathlib import Path

panel_path = Path("factor_output/etf_rotation/panel_optimized_v2_20200102_20251014.parquet")
panel = pd.read_parquet(panel_path)
dates = panel.index.get_level_values("date")
print(f"✅ 因子面板: {panel.shape}, 日期范围: {dates.min()} ~ {dates.max()}")

price_dir = Path("raw/ETF/daily")
files = sorted(price_dir.glob("*.parquet"))
if not files:
    raise SystemExit("❌ 没有任何ETF价格文件")
print(f"✅ 价格文件: {len(files)} 个")
PY

echo "===> 选择最新因子排序文件"
latest_factor=$(ls -1 production_factor_results/factor_screen_f5_*.json 2>/dev/null | sort | tail -n 1 || true)
if [[ -z "$latest_factor" ]]; then
    echo "❌ 未找到匹配的 factor_screen_f5_*.json" >&2
    exit 1
fi
echo "    使用因子文件: $latest_factor"

timestamp=$(date +%Y%m%d_%H%M%S)
output="strategies/results/real_backtest_${timestamp}.csv"

echo "===> 启动真实回测"
python3 strategies/vectorbt_multifactor_grid.py \
    --top-factors-json "$latest_factor" \
    --top-k 10 \
    --max-total-combos 10000 \
    --top-k-results 100 \
    --fees 0.00035 \
    --output "$output"

echo "===> 回测完成，摘要:"
python3 - <<PY
import pandas as pd
df = pd.read_csv("$output")
cols = ['combo_idx','annual_return','sharpe','max_drawdown','turnover','factor_count','concentration']
print(df[cols].head(10))
print("\n统计信息：")
print(df[['sharpe','turnover']].describe())
PY

echo "✅ 结果已保存: $output"