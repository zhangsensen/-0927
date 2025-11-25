#!/bin/bash
# 全量回测启动脚本
# 对WFO所有12,597组合执行8天频率真实回测，用于训练校准器

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# 配置环境变量
export RB_BACKTEST_ALL=1          # 🔥 全量回测模式
export RB_FORCE_FREQ=8            # 锁定8天频率
export RB_TEST_ALL_FREQS=0        # 禁用多频扫描
export RB_SKIP_PREV=1             # 跳过上一轮对比（加速）

# 线程限制（避免过载）
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# 生成时间戳
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="results/logs/rb_full_all_combos_8d_${TS}.log"

mkdir -p "$(dirname "$LOG_FILE")"

echo "========================================"
echo "全量回测启动（12,597组合 × 8天频率）"
echo "========================================"
echo "配置:"
echo "  RB_BACKTEST_ALL: $RB_BACKTEST_ALL"
echo "  RB_FORCE_FREQ: $RB_FORCE_FREQ"
echo "  RB_TEST_ALL_FREQS: $RB_TEST_ALL_FREQS"
echo "  并行度: 8核心"
echo "  预计耗时: ~2小时（12597组合 ÷ 8workers ≈ 1575组/worker × 4.5秒 ≈ 118分钟）"
echo "  日志: $LOG_FILE"
echo ""

# 后台执行并记录PID
nohup python -u -m real_backtest.run_production_backtest \
  > "$LOG_FILE" 2>&1 &

PID=$!
echo "[PID] $PID"
echo "$PID" > results/logs/rb_full_all_combos.pid

echo ""
echo "✅ 全量回测已启动（后台运行）"
echo "监控进度:"
echo "  tail -f $LOG_FILE"
echo ""
echo "检查状态:"
echo "  ps aux | grep $PID"
echo ""
echo "完成后，使用校准后的结果重新训练GBDT:"
echo "  python -m core.wfo_realbt_calibrator"
