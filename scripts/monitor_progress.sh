#!/bin/bash
# 监控全面生产进度

echo "================================"
echo "全面生产进度监控"
echo "================================"
echo ""

# 监控日志文件
LOG_FILE="full_production_all_factors.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "日志文件不存在，等待生成..."
    sleep 5
fi

# 实时显示关键信息
echo "最新进度:"
echo "--------"
tail -50 "$LOG_FILE" | grep -E "(批次|✅|❌|完成|因子)" | tail -20

echo ""
echo "统计信息:"
echo "--------"
echo -n "成功因子数: "
grep -c "✅" "$LOG_FILE" 2>/dev/null || echo "0"

echo -n "失败因子数: "
grep -c "❌" "$LOG_FILE" 2>/dev/null || echo "0"

echo -n "当前批次: "
grep "批次" "$LOG_FILE" | tail -1 | grep -oE "批次 [0-9]+/[0-9]+" || echo "未知"

echo ""
echo "最新5条日志:"
echo "--------"
tail -5 "$LOG_FILE"
