#!/bin/bash
# 批量执行Top1000和Top3000回测
# 使用方法: cd etf_rotation_experiments && bash scripts/run_large_scale_backtests.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(dirname "$SCRIPT_DIR")"
RUN_TS="20251113_145102"

cd "$EXP_ROOT"
mkdir -p logs

echo "=========================================="
echo "🚀 启动大规模回测任务"
echo "=========================================="
echo "实验根目录: $EXP_ROOT"
echo "WFO Run: $RUN_TS"
echo ""

# IC Top1000
echo "[1/4] IC Top1000 回测..."
python real_backtest/run_profit_backtest.py \
    --topk 1000 \
    --ranking-file results/run_${RUN_TS}/ranking_blends/ranking_baseline.parquet \
    --slippage-bps 2.0 \
    > logs/ic_top1000_$(date +%Y%m%d_%H%M%S).log 2>&1 &
PID1=$!
echo "  启动成功 (PID: $PID1)"

sleep 2

# Calibrated Top1000
echo "[2/4] Calibrated Top1000 回测..."
python real_backtest/run_profit_backtest.py \
    --topk 1000 \
    --ranking-file results/run_${RUN_TS}/ranking_blends/ranking_lightgbm.parquet \
    --slippage-bps 2.0 \
    > logs/calibrated_top1000_$(date +%Y%m%d_%H%M%S).log 2>&1 &
PID2=$!
echo "  启动成功 (PID: $PID2)"

sleep 2

# IC Top3000
echo "[3/4] IC Top3000 回测..."
python real_backtest/run_profit_backtest.py \
    --topk 3000 \
    --ranking-file results/run_${RUN_TS}/ranking_blends/ranking_baseline.parquet \
    --slippage-bps 2.0 \
    > logs/ic_top3000_$(date +%Y%m%d_%H%M%S).log 2>&1 &
PID3=$!
echo "  启动成功 (PID: $PID3)"

sleep 2

# Calibrated Top3000
echo "[4/4] Calibrated Top3000 回测..."
python real_backtest/run_profit_backtest.py \
    --topk 3000 \
    --ranking-file results/run_${RUN_TS}/ranking_blends/ranking_lightgbm.parquet \
    --slippage-bps 2.0 \
    > logs/calibrated_top3000_$(date +%Y%m%d_%H%M%S).log 2>&1 &
PID4=$!
echo "  启动成功 (PID: $PID4)"

echo ""
echo "=========================================="
echo "✅ 所有回测任务已启动"
echo "=========================================="
echo "进程ID:"
echo "  IC Top1000:         $PID1"
echo "  Calibrated Top1000: $PID2"
echo "  IC Top3000:         $PID3"
echo "  Calibrated Top3000: $PID4"
echo ""
echo "监控命令:"
echo "  tail -f logs/ic_top1000_*.log"
echo "  tail -f logs/calibrated_top1000_*.log"
echo "  tail -f logs/ic_top3000_*.log"
echo "  tail -f logs/calibrated_top3000_*.log"
echo ""
echo "检查进程:"
echo "  ps aux | grep run_profit_backtest"
echo ""
echo "等待完成后运行:"
echo "  python scripts/compare_topk_backtests.py --run-ts $RUN_TS --output results/run_${RUN_TS}/topk_comparison_report.md"
echo "=========================================="
