#!/bin/bash
# 8天 WFO 监控脚本
# 检查 WFO 进程，完成后自动触发全频回测和分析

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
TS="20251108_022132"
PID_FILE="${ROOT_DIR}/results/.wfo_8d_full_${TS}.pid"
LOG_FILE="${ROOT_DIR}/results/logs/wfo_8d_full_${TS}.log"

if [ ! -f "${PID_FILE}" ]; then
    echo "❌ PID 文件不存在: ${PID_FILE}"
    exit 1
fi

PID=$(cat "${PID_FILE}")

echo "🔍 监控 8天 WFO 进程 (PID: ${PID})"
echo "   日志: ${LOG_FILE}"
echo "   检查间隔: 60秒"
echo ""

while kill -0 ${PID} 2>/dev/null; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WFO 仍在运行..."
    sleep 60
done

echo ""
echo "✅ 8天 WFO 已完成！"
echo ""

# 自动触发全频回测和分析
echo "🚀 启动自动化流水线（全频回测 + 分析）..."
python "${ROOT_DIR}/scripts/auto_8d_wfo_pipeline.py"

exit_code=$?
if [ ${exit_code} -eq 0 ]; then
    echo "✅ 流水线完成！"
else
    echo "❌ 流水线失败，退出码: ${exit_code}"
fi

exit ${exit_code}
