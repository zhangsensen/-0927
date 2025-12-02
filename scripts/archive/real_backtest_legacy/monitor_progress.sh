#!/bin/bash
# 持仓数网格搜索进度监控脚本

echo "========================================="
echo "持仓数网格搜索进度监控"
echo "========================================="
echo ""

# 检查进程是否运行
if ps aux | grep "run_position_grid_search.py" | grep -v grep > /dev/null; then
    echo "✅ 脚本正在运行中..."
    echo ""
else
    echo "❌ 脚本未在运行"
    exit 1
fi

# 检查日志文件大小
if [ -f grid_search_full.log ]; then
    LOG_SIZE=$(du -h grid_search_full.log | cut -f1)
    echo "📊 日志文件大小: $LOG_SIZE"
    echo ""
fi

# 显示关键进度信息
echo "📝 最新日志 (最后20行):"
echo "========================================="
tail -20 grid_search_full.log | grep -E "(Done|预计算IC权重|已完成|分析结果|最优持仓数)" || tail -20 grid_search_full.log
echo ""
echo "========================================="
echo ""

# 统计完成任务数
DONE_COUNT=$(grep "Done.*tasks" grid_search_full.log | tail -1 | grep -oE "[0-9]+ tasks" | grep -oE "[0-9]+")
if [ ! -z "$DONE_COUNT" ]; then
    echo "✅ 已完成任务数: $DONE_COUNT / 930"
    PROGRESS=$((DONE_COUNT * 100 / 930))
    echo "📈 完成进度: ${PROGRESS}%"
else
    echo "⏳ 正在启动..."
fi

echo ""
echo "运行: watch -n 30 './monitor_progress.sh' 实时监控"
