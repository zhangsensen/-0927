#!/bin/bash
# 全量回测进度监控脚本

LOG_FILE="results/logs/rb_full_all_combos_8d_20251108_201251.log"
TOTAL_COMBOS=12597

echo "========================================"
echo "全量回测进度监控"
echo "========================================"
echo ""

# 检查进程是否还在运行
PID_FILE="results/logs/rb_full_all_combos.pid"
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ 回测进程运行中 (PID: $PID)"
    else
        echo "⚠️  回测进程已结束 (PID: $PID)"
    fi
else
    echo "❌ 未找到PID文件"
fi

echo ""

# 提取最新进度
if [ -f "$LOG_FILE" ]; then
    LAST_DONE=$(grep "Done.*tasks" "$LOG_FILE" | tail -1)
    
    if [ -n "$LAST_DONE" ]; then
        # 提取已完成任务数
        COMPLETED=$(echo "$LAST_DONE" | grep -oE "Done +[0-9]+" | grep -oE "[0-9]+")
        ELAPSED=$(echo "$LAST_DONE" | grep -oE "elapsed: +[^ ]+" | awk '{print $2}')
        
        PROGRESS=$(echo "scale=1; $COMPLETED * 100 / $TOTAL_COMBOS" | bc)
        REMAINING=$((TOTAL_COMBOS - COMPLETED))
        
        echo "进度: $COMPLETED / $TOTAL_COMBOS ($PROGRESS%)"
        echo "已用时: $ELAPSED"
        echo "剩余: $REMAINING 组合"
        
        # 估算剩余时间
        if [[ "$ELAPSED" =~ ([0-9.]+)min ]]; then
            MINUTES="${BASH_REMATCH[1]}"
            MINUTES_INT=$(echo "$MINUTES" | cut -d. -f1)
            SPEED=$(echo "scale=2; $COMPLETED / $MINUTES" | bc)
            REMAINING_MINUTES=$(echo "scale=1; $REMAINING / $SPEED" | bc)
            echo "速度: ${SPEED} 组合/分钟"
            echo "预计剩余: ${REMAINING_MINUTES} 分钟"
        elif [[ "$ELAPSED" =~ ([0-9.]+)s ]]; then
            SECONDS="${BASH_REMATCH[1]}"
            SPEED=$(echo "scale=2; $COMPLETED / ($SECONDS / 60)" | bc)
            REMAINING_MINUTES=$(echo "scale=1; $REMAINING / $SPEED" | bc)
            echo "速度: ${SPEED} 组合/分钟"
            echo "预计剩余: ${REMAINING_MINUTES} 分钟"
        fi
    else
        echo "⏳ 回测刚启动，等待第一批结果..."
    fi
    
    echo ""
    echo "----------------------------------------"
    echo "最近日志 (最后10行):"
    echo "----------------------------------------"
    tail -10 "$LOG_FILE"
else
    echo "❌ 日志文件不存在: $LOG_FILE"
fi

echo ""
echo "========================================"
echo "监控命令:"
echo "  实时日志: tail -f $LOG_FILE"
echo "  完成后训练: python scripts/train_calibrator_full.py"
echo "========================================"
