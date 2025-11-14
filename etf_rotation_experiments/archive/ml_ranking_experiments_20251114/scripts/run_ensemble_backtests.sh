#!/bin/bash
# Ensemble策略批量回测
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(dirname "$SCRIPT_DIR")"
ENSEMBLE_DIR="$EXP_ROOT/results/run_20251113_145102/ensemble_rankings"

cd "$EXP_ROOT"
mkdir -p logs

echo "=========================================="
echo "🚀 启动Ensemble策略回测"
echo "=========================================="
echo "实验根目录: $EXP_ROOT"
echo ""

# 策略1: 交集156组合
echo "[1/3] 交集策略 (IC ∩ Calibrator Top1000, 156组合)..."
python real_backtest/run_profit_backtest.py \
    --ranking-file "$ENSEMBLE_DIR/ranking_intersection_top1000.parquet" \
    --slippage-bps 2.0 \
    > logs/ensemble_intersection_$(date +%Y%m%d_%H%M%S).log 2>&1
echo "  ✅ 完成"

sleep 2

# 策略2: 并集913组合
echo "[2/3] 并集策略 (IC + Calibrator Top500, 913组合)..."
python real_backtest/run_profit_backtest.py \
    --ranking-file "$ENSEMBLE_DIR/ranking_union_top500.parquet" \
    --slippage-bps 2.0 \
    > logs/ensemble_union_$(date +%Y%m%d_%H%M%S).log 2>&1
echo "  ✅ 完成"

sleep 2

# 策略3: 加权ensemble Top1000
echo "[3/3] 加权Ensemble策略 (50%IC + 50%Cal, 1000组合)..."
python real_backtest/run_profit_backtest.py \
    --topk 1000 \
    --ranking-file "$ENSEMBLE_DIR/ranking_ensemble_50_50_top1000.parquet" \
    --slippage-bps 2.0 \
    > logs/ensemble_weighted_$(date +%Y%m%d_%H%M%S).log 2>&1
echo "  ✅ 完成"

echo ""
echo "=========================================="
echo "✅ 所有Ensemble回测完成"
echo "=========================================="
echo ""
echo "日志文件:"
echo "  logs/ensemble_intersection_*.log"
echo "  logs/ensemble_union_*.log"
echo "  logs/ensemble_weighted_*.log"
echo ""
echo "下一步: 运行对比分析脚本"
echo "=========================================="
