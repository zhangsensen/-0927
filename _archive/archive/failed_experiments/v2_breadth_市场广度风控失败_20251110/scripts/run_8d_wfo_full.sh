#!/bin/bash
# 8天固定频率 WFO 全量扩容运行脚本 (修订版)
# 目标：执行组合级 WFO 全量搜索，生成 results/run_<ts>/all_combos.parquet 等核心产物
# 预计耗时：~1 小时（8核并行）
# 之前版本错误地直接调用了真实回测脚本 (real_backtest.run_production_backtest)，
# 会因为缺少最新 WFO 结果而报 "未找到WFO运行结果"。现改为正确执行 run_combo_wfo.py。

set -e

cd "$(dirname "$0")/.."
ROOT_DIR=$(pwd)

# 时间戳
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${ROOT_DIR}/results/logs"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/wfo_8d_full_${TS}.log"
echo "====== 8天 WFO 全量扩容 (组合级WFO生成) ======"
echo "时间戳: ${TS}"
echo "日志文件: ${LOG_FILE}"
echo "候选池: 12,597 组合（18因子，size=[2,3,4,5]）"
echo "============================="

# 并发与性能：限制数值库线程，避免与进程池争抢
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# 使 WFO 输出目录时间戳与本日志时间戳一致
export RB_RESULT_TS="${TS}"
# 频率固定为8天（已在config中锁定）

# PID 锁防止重复运行
LOCK_FILE="${ROOT_DIR}/results/.wfo_8d_active.pid"
if [ -f "$LOCK_FILE" ]; then
	OLD_PID=$(cat "$LOCK_FILE" 2>/dev/null || echo "")
	if [ -n "$OLD_PID" ] && ps -p "$OLD_PID" > /dev/null 2>&1; then
		echo "⚠️  已有运行中的 8d WFO 进程 (PID=$OLD_PID)，此次启动取消。"
		echo "如需强制重启：kill $OLD_PID && rm $LOCK_FILE 后再运行。"
		exit 1
	fi
fi

# 启动后台运行
echo "启动组合级 WFO 全量搜索 (run_combo_wfo.py)..."
nohup python -u run_combo_wfo.py > "${LOG_FILE}" 2>&1 &

PID=$!
echo "后台进程 PID: ${PID}"
echo "监控日志: tail -f ${LOG_FILE}"

# 保存 PID（时间戳+活动锁）
echo "${PID}" > "${ROOT_DIR}/results/.wfo_8d_full_${TS}.pid"
echo "${PID}" > "$LOCK_FILE"

echo ""
echo "运行已启动！"
echo "查看进度:"
echo "  tail -f ${LOG_FILE}"
echo "  或 ps -p ${PID}"
echo "完成后产物将位于: results/run_<动态时间戳>/all_combos.parquet 等"
echo "下一步 (示例): 生成学习排序 -> python real_backtest/scripts/learn_wfo_rank_formula.py --run_dir results/run_<ts> --target robust_score"
