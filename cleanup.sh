#!/bin/bash
# 项目清理脚本 - 自动化执行清理操作
# 使用前请备份项目: git add -A && git commit -m "backup: before cleanup"

set -e

PROJECT_ROOT="/Users/zhangshenshen/深度量化0927"
cd "$PROJECT_ROOT"

echo "================================================================================"
echo "🧹 项目清理脚本 - 开始执行"
echo "================================================================================"
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 统计删除的文件
DELETED_COUNT=0
DELETED_SIZE=0

# 函数：删除文件并记录
delete_file() {
    if [ -f "$1" ]; then
        SIZE=$(stat -f%z "$1" 2>/dev/null || echo 0)
        rm -f "$1"
        DELETED_COUNT=$((DELETED_COUNT + 1))
        DELETED_SIZE=$((DELETED_SIZE + SIZE))
        echo -e "${GREEN}✓${NC} 删除: $1 ($(numfmt --to=iec-i --suffix=B $SIZE 2>/dev/null || echo $SIZE)B)"
    fi
}

# 函数：删除目录并记录
delete_dir() {
    if [ -d "$1" ]; then
        SIZE=$(du -sh "$1" 2>/dev/null | cut -f1)
        rm -rf "$1"
        DELETED_COUNT=$((DELETED_COUNT + 1))
        echo -e "${GREEN}✓${NC} 删除目录: $1 ($SIZE)"
    fi
}

echo "【第1步】删除根目录临时脚本..."
echo "-----------------------------------------------"
delete_file "test_engine_init.py"
delete_file "code_quality_mcp_check.py"
delete_file "verify_9factors_dataflow.py"
delete_file "launch_wfo_real_backtest.py"
delete_file "start_real_backtest.py"
delete_file "test_signal_threshold_impact.py"

echo ""
echo "【第2步】删除日志文件..."
echo "-----------------------------------------------"
delete_file "backtest_output.log"
delete_file "execution_20251025_193306.log"
delete_file "hk_factor_generation.log"
delete_file "production_run.log"
delete_file "run_optimized_220044.log"
delete_file "test_100_manual.log"
delete_file "test_minimal.log"
delete_file "wfo_full_run.log"

echo ""
echo "【第3步】删除无用目录..."
echo "-----------------------------------------------"
delete_dir "factor_ready"
delete_dir "etf_cross_section_results"
delete_dir "production_factor_results"

echo ""
echo "【第4步】删除过时报告..."
echo "-----------------------------------------------"
delete_file "ETF_CODE_MISMATCH_REPORT.md"

echo ""
echo "【第5步】删除过时Shell脚本..."
echo "-----------------------------------------------"
delete_file "monitor_wfo_backtest.sh"
delete_file "run_fixed_backtest.sh"
delete_file "run_real_backtest.sh"
delete_file "run_wfo_backtest.sh"

echo ""
echo "【第6步】清理scripts目录..."
echo "-----------------------------------------------"
cd scripts/
delete_file "analyze_100k_results.py"
delete_file "analyze_top1000_strategies.py"
delete_file "analyze_top1000_strategies_fixed.py"
delete_file "etf_rotation_backtest.py"
delete_file "generate_etf_rotation_factors.py"
delete_file "linus_reality_check_report.py"
delete_file "validate_candlestick_patterns.py"
delete_file "test_full_pipeline_with_configmanager.py"
cd "$PROJECT_ROOT"

echo ""
echo "【第7步】清理factor_screening结果文件..."
echo "-----------------------------------------------"
if [ -d "factor_system/factor_screening/screening_results" ]; then
    RESULT_COUNT=$(find factor_system/factor_screening/screening_results -type f | wc -l)
    if [ "$RESULT_COUNT" -gt 0 ]; then
        echo -e "${YELLOW}⚠${NC}  发现 $RESULT_COUNT 个结果文件"
        echo "    这些是过期的筛选结果，可以安全删除"
        read -p "    是否删除? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf factor_system/factor_screening/screening_results/*
            echo -e "${GREEN}✓${NC} 已清空结果目录"
        fi
    fi
fi

echo ""
echo "================================================================================"
echo "📊 清理统计"
echo "================================================================================"
echo -e "已删除文件数: ${GREEN}$DELETED_COUNT${NC}"
echo -e "已释放空间: ${GREEN}$(numfmt --to=iec-i --suffix=B $DELETED_SIZE 2>/dev/null || echo $DELETED_SIZE)B${NC}"
echo ""

echo "【验证步骤】运行测试..."
echo "-----------------------------------------------"
if command -v pytest &> /dev/null; then
    echo "运行pytest..."
    pytest -v --tb=short 2>&1 | head -20
else
    echo -e "${YELLOW}⚠${NC}  pytest未安装，跳过测试"
fi

echo ""
echo "================================================================================"
echo "✅ 清理完成！"
echo "================================================================================"
echo ""
echo "后续步骤:"
echo "  1. 验证功能: make test"
echo "  2. 代码检查: make lint"
echo "  3. 提交更改: git add -A && git commit -m 'cleanup: remove temporary files'"
echo ""
