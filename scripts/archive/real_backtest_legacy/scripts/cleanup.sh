#!/bin/bash
# ============================================================================
# ETF轮动系统 - 清理脚本
# ============================================================================
# 用途: 清理缓存、临时文件以及 Python 编译产物
# 使用: ./scripts/cleanup.sh
# ============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

printf '%s\n' "============================================================================"
printf '%s\n' "ETF轮动系统 - 开始清理"
printf '%s\n' "============================================================================"
printf '[info] 项目目录: %s\n\n' "$PROJECT_ROOT"

# 1. 清理 .cache 目录
printf '[1/5] 清理 .cache 目录...\n'
if [ -d ".cache" ]; then
    du -sh .cache 2>/dev/null | awk '{printf "  当前大小: %s\\n", $1}'
    rm -rf .cache
    printf '  [ok] .cache 目录已删除\n'
else
    printf '  [skip] 未发现 .cache 目录\n'
fi
printf '\n'

# 2. 清理 __pycache__ 目录
printf '[2/5] 清理 __pycache__ 目录...\n'
PY_DIR_COUNT=$(find . -type d -name "__pycache__" | wc -l | tr -d ' ')
if [ "$PY_DIR_COUNT" -gt 0 ]; then
    printf '  发现 %s 个 __pycache__ 目录\n' "$PY_DIR_COUNT"
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    printf '  [ok] __pycache__ 目录已删除\n'
else
    printf '  [skip] 未发现 __pycache__ 目录\n'
fi
printf '\n'

# 3. 清理 .pyc 文件
printf '[3/5] 清理 .pyc 文件...\n'
PYC_COUNT=$(find . -type f -name "*.pyc" | wc -l | tr -d ' ')
if [ "$PYC_COUNT" -gt 0 ]; then
    printf '  发现 %s 个 .pyc 文件\n' "$PYC_COUNT"
    find . -type f -name "*.pyc" -delete
    printf '  [ok] .pyc 文件已删除\n'
else
    printf '  [skip] 未发现 .pyc 文件\n'
fi
printf '\n'

# 4. 清理七天前的日志
printf '[4/5] 清理 results/logs 中超过 7 天的日志...\n'
if [ -d "results/logs" ]; then
    OLD_LOG_COUNT=$(find results/logs -name "*.log" -mtime +7 | wc -l | tr -d ' ')
    if [ "$OLD_LOG_COUNT" -gt 0 ]; then
        printf '  发现 %s 个过期日志文件\n' "$OLD_LOG_COUNT"
        find results/logs -name "*.log" -mtime +7 -delete
        printf '  [ok] 过期日志已删除\n'
    else
        printf '  [skip] 未发现过期日志\n'
    fi
else
    printf '  [skip] 未发现 results/logs 目录\n'
fi
printf '\n'

# 5. 清理 .DS_Store 文件
printf '[5/5] 清理 .DS_Store 文件...\n'
DS_COUNT=$(find . -name ".DS_Store" | wc -l | tr -d ' ')
if [ "$DS_COUNT" -gt 0 ]; then
    printf '  发现 %s 个 .DS_Store 文件\n' "$DS_COUNT"
    find . -name ".DS_Store" -delete
    printf '  [ok] .DS_Store 文件已删除\n'
else
    printf '  [skip] 未发现 .DS_Store 文件\n'
fi
printf '\n'

printf '%s\n' "============================================================================"
printf '%s\n' "[ok] 清理完成"
printf '%s\n\n' "============================================================================"
printf '已清理项目:\n'
printf '  - .cache 目录\n'
printf '  - __pycache__ 目录\n'
printf '  - .pyc 编译文件\n'
printf '  - results/logs 目录中过期日志\n'
printf '  - .DS_Store 系统文件\n'
printf '提示: 建议在发布前运行本脚本保持仓库整洁。\n'
