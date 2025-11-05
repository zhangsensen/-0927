#!/bin/bash
# WFO 1024测试进度监控脚本

LOG_FILE="/Users/zhangshenshen/深度量化0927/wfo_1024_results.log"
RESULT_DIR="/Users/zhangshenshen/深度量化0927/etf_rotation_optimized/results/wfo_parameter_grid_1024"

echo "========== WFO 1024测试进度监控 =========="
echo

# 检查进程
echo "1. 进程状态:"
if pgrep -f "test_wfo_parameter_grid" > /dev/null; then
    echo "  ✅ 进程运行中 (PID: $(pgrep -f 'test_wfo_parameter_grid'))"
    ps -p $(pgrep -f "test_wfo_parameter_grid") -o %cpu,%mem,etime,command | tail -1
else
    echo "  ❌ 进程未运行"
fi
echo

# 检查日志文件
echo "2. 日志文件:"
if [ -f "$LOG_FILE" ]; then
    FILE_SIZE=$(ls -lh "$LOG_FILE" | awk '{print $5}')
    LINE_COUNT=$(wc -l < "$LOG_FILE")
    echo "  文件大小: $FILE_SIZE"
    echo "  行数: $LINE_COUNT"
    echo
    echo "  最新10行:"
    tail -10 "$LOG_FILE" | sed 's/^/    /'
else
    echo "  日志文件不存在"
fi
echo

# 检查结果目录
echo "3. 结果文件:"
if [ -d "$RESULT_DIR" ]; then
    echo "  结果目录: $RESULT_DIR"
    ls -lh "$RESULT_DIR" 2>/dev/null | grep -v "^total" | sed 's/^/    /' || echo "    (空)"
else
    echo "  结果目录不存在"
fi

echo
echo "==================== 完成 ===================="
