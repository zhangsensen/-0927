#!/bin/bash
# 监控WFO流程进度

LOG_FILE="/tmp/wfo_pipeline_full.log"

echo "=============================================================================="
echo "🔍 WFO流程实时监控"
echo "=============================================================================="
echo ""

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ 日志文件不存在: $LOG_FILE"
    exit 1
fi

# 显示当前状态
echo "📊 当前状态:"
echo "-----------------------------------------------------------------------------"

if grep -q "步骤1/3: 生成因子面板" "$LOG_FILE"; then
    echo "  ✅ 步骤1: 因子面板生成"
    
    if grep -q "✅ 面板生成完成" "$LOG_FILE"; then
        PANEL=$(grep "✅ 面板生成完成" "$LOG_FILE" | tail -1 | cut -d: -f2)
        echo "     完成:$PANEL"
        
        # 提取因子数
        FACTOR_COUNT=$(grep "因子数:" "$LOG_FILE" | tail -1 | awk '{print $NF}')
        echo "     因子数: $FACTOR_COUNT"
    else
        echo "     进行中..."
    fi
else
    echo "  ⏳ 步骤1: 等待开始..."
fi

echo ""

if grep -q "步骤2/3: 因子筛选" "$LOG_FILE"; then
    echo "  ✅ 步骤2: 因子筛选"
    
    if grep -q "✅ 因子筛选完成" "$LOG_FILE"; then
        SCREENING=$(grep "✅ 因子筛选完成" "$LOG_FILE" | tail -1 | cut -d: -f2)
        echo "     完成:$SCREENING"
        
        # 提取通过因子数
        if grep -q "通过因子数:" "$LOG_FILE"; then
            PASSED=$(grep "通过因子数:" "$LOG_FILE" | tail -1 | awk '{print $NF}')
            echo "     通过因子: $PASSED"
        fi
    else
        echo "     进行中..."
    fi
else
    echo "  ⏳ 步骤2: 等待开始..."
fi

echo ""

if grep -q "步骤3/3: WFO大规模回测" "$LOG_FILE"; then
    echo "  ✅ 步骤3: WFO回测"
    
    if grep -q "✅ WFO回测完成" "$LOG_FILE"; then
        WFO=$(grep "✅ WFO回测完成" "$LOG_FILE" | tail -1 | cut -d: -f2)
        echo "     完成:$WFO"
    else
        # 检查回测进度
        if grep -q "折叠" "$LOG_FILE"; then
            PROGRESS=$(grep "折叠" "$LOG_FILE" | tail -1)
            echo "     进行中: $PROGRESS"
        else
            echo "     初始化..."
        fi
    fi
else
    echo "  ⏳ 步骤3: 等待开始..."
fi

echo ""
echo "=============================================================================="
echo "📝 最新日志 (最后30行):"
echo "------------------------------------------------------------------------------"
tail -n 30 "$LOG_FILE"

echo ""
echo "=============================================================================="
echo "💡 提示:"
echo "  • 查看完整日志: tail -f $LOG_FILE"
echo "  • 检查进程: ps aux | grep wfo"
echo "  • 重新监控: bash $(basename $0)"
echo "=============================================================================="
