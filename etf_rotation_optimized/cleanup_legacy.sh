#!/usr/bin/env bash
set -euo pipefail

TS=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR=".legacy_backup_${TS}"
mkdir -p "$BACKUP_DIR"

echo "[CLEANUP] Archiving legacy files to $BACKUP_DIR"

# Conservative: only move known .bak and obvious legacy experiments
for f in \
  core/ensemble_wfo_optimizer_v1_combo.py.bak \
  core/ensemble_wfo_optimizer.py.bak \
  core/pipeline_before_direct_wfo_refactor_20251029.py.bak \
  core/ensemble_wfo_optimizer_v1_combo.py.bak \
  core/ensemble_wfo_optimizer.py \
  core/ensemble_wfo_optimizer_v1_combo.py \
  ; do
  if [ -e "$f" ]; then
    mkdir -p "$(dirname "$BACKUP_DIR/$f")"
    mv "$f" "$BACKUP_DIR/$f" || true
  fi
done

echo "[CLEANUP] Done. Review backup before deleting permanently."
#!/bin/bash
# 清理历史垃圾代码和过期文档
# 2025-11-03

set -e

PROJECT_ROOT="/Users/zhangshenshen/深度量化0927/etf_rotation_optimized"
cd "$PROJECT_ROOT"

echo "========================================================================"
echo "清理历史垃圾代码和过期文档"
echo "========================================================================"
echo ""

# 创建备份目录
BACKUP_DIR=".legacy_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "备份目录: $BACKUP_DIR"
echo ""

# 1. 清理过期文档
echo "1. 清理过期文档..."
LEGACY_DOCS=(
    "LOOKAHEAD_BIAS_FIX_SUMMARY.md"
    "LOOKAHEAD_FIX_VALIDATION_REPORT.md"
    "CRITICAL_LOOKAHEAD_BIAS_AUDIT.md"
    "LINUS_DEEP_AUDIT_REPORT.md"
    "INTEGRATION_COMPLETION_REPORT.md"
    "FINAL_PRODUCTION_VALIDATION_REPORT.md"
    "EVENT_DRIVEN_DEVELOPMENT_REPORT.md"
    "FULL_VALIDATION_REPORT.md"
    "PORTFOLIO_CONSTRUCTOR_FIX.md"
)

for doc in "${LEGACY_DOCS[@]}"; do
    if [ -f "$doc" ]; then
        echo "  移除: $doc"
        mv "$doc" "$BACKUP_DIR/" 2>/dev/null || true
    fi
done

# 2. 清理过期脚本
echo ""
echo "2. 清理过期脚本..."
LEGACY_SCRIPTS=(
    "run_lookahead_fix_validation.sh"
    "run_full_production_pipeline.sh"
    "run_wfo_validation.sh"
    "run_full_wfo_real_data.sh"
    "test_portfolio_fix.py"
)

for script in "${LEGACY_SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        echo "  移除: $script"
        mv "$script" "$BACKUP_DIR/" 2>/dev/null || true
    fi
done

# 3. 清理过期配置
echo ""
echo "3. 清理过期配置..."
LEGACY_CONFIGS=(
    "pipeline_config_wfo_test.yaml"
    "vectorbt_backtest/configs/event_driven_test.yaml"
    "vectorbt_backtest/configs/backtest_config_event.yaml"
    "configs/experiments/exp_brutal.yaml"
)

for config in "${LEGACY_CONFIGS[@]}"; do
    if [ -f "$config" ]; then
        echo "  移除: $config"
        mv "$config" "$BACKUP_DIR/" 2>/dev/null || true
    fi
done

# 4. 清理过期测试
echo ""
echo "4. 清理过期测试..."
LEGACY_TESTS=(
    "vectorbt_backtest/tests/test_signal_quality.py"
    "vectorbt_backtest/tests/test_integration.py"
    "vectorbt_backtest/tests/test_engine_integration.py"
    "tests/test_portfolio_constructor_lookahead.py"
)

for test in "${LEGACY_TESTS[@]}"; do
    if [ -f "$test" ]; then
        echo "  移除: $test"
        mv "$test" "$BACKUP_DIR/" 2>/dev/null || true
    fi
done

# 5. 清理过期核心代码
echo ""
echo "5. 清理过期核心代码..."
LEGACY_CORE=(
    "vectorbt_backtest/core/signal_quality_evaluator.py"
    "vectorbt_backtest/core/event_driven_portfolio.py"
    "scripts/analyze_brutal_results.py"
    "scripts/run_ab_test.sh"
    "scripts/compare_ab_results.py"
)

for core in "${LEGACY_CORE[@]}"; do
    if [ -f "$core" ]; then
        echo "  移除: $core"
        mv "$core" "$BACKUP_DIR/" 2>/dev/null || true
    fi
done

# 6. 清理空目录
echo ""
echo "6. 清理空目录..."
find . -type d -empty -delete 2>/dev/null || true

# 7. 清理缓存和临时文件
echo ""
echo "7. 清理缓存和临时文件..."
rm -rf cache/ .cache/ __pycache__ */__pycache__ */*/__pycache__ 2>/dev/null || true
rm -rf htmlcov/ .coverage coverage.xml 2>/dev/null || true
rm -rf *.egg-info/ 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name ".DS_Store" -delete 2>/dev/null || true

echo ""
echo "========================================================================"
echo "✅ 清理完成！"
echo "========================================================================"
echo ""
echo "备份位置: $BACKUP_DIR"
echo ""
echo "保留的核心文件:"
echo "  - core/event_driven_portfolio_constructor.py (事件驱动构建器)"
echo "  - core/wfo_performance_evaluator.py (WFO性能评估)"
echo "  - core/wfo_strategy_ranker.py (Top-N策略排序)"
echo "  - WFO_COMPREHENSIVE_AUDIT.md (审计报告)"
echo "  - EVENT_DRIVEN_TRADING_GUIDE.md (使用指南)"
echo "  - run_wfo_complete.sh (完整运行脚本)"
echo ""
