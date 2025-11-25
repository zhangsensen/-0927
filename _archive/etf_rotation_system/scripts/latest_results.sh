#!/bin/bash
# 查看最新ETF轮动系统结果

echo "🔍 ETF轮动系统最新结果"
echo "========================================="

# 查找最新的因子面板
LATEST_PANEL=$(find data/panels -name "panel_*" -type d | sort | tail -1)
echo "📊 最新因子面板: $LATEST_PANEL"

if [ -d "$LATEST_PANEL" ]; then
    echo "📁 面板文件:"
    ls -la "$LATEST_PANEL/" | grep -E '\.(parquet|json|txt)$'

    # 显示执行日志的关键信息
    if [ -f "$LATEST_PANEL/execution_log.txt" ]; then
        echo ""
        echo "📋 执行日志摘要:"
        cat "$LATEST_PANEL/execution_log.txt" | head -10
    fi
fi

echo ""

# 查找最新的筛选结果
LATEST_SCREENING=$(find data/results/screening -name "screening_*" -type d | sort | tail -1)
echo "🔬 最新筛选结果: $LATEST_SCREENING"

if [ -d "$LATEST_SCREENING" ]; then
    echo "📁 筛选文件:"
    ls -la "$LATEST_SCREENING/" | grep -E '\.(csv|txt)$'

    # 显示筛选报告的关键信息
    if [ -f "$LATEST_SCREENING/screening_report.txt" ]; then
        echo ""
        echo "📋 筛选结果摘要:"
        cat "$LATEST_SCREENING/screening_report.txt" | grep -E "(筛选结果|🟢 核心|🟡 补充)"
    fi

    # 显示通过筛选的因子数量
    if [ -f "$LATEST_SCREENING/passed_factors.csv" ]; then
        PASSED_COUNT=$(tail -n +2 "$LATEST_SCREENING/passed_factors.csv" | wc -l)
        echo ""
        echo "✅ 通过筛选的因子数量: $PASSED_COUNT"
    fi
fi

echo ""
echo "📈 完整目录结构:"
echo "etf_rotation_system/data/"
echo "├── panels/           # 因子面板数据"
echo "├── screening/        # 因子筛选结果"
echo "└── results/          # 历史数据归档"