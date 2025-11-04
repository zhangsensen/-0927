#!/bin/bash

# 补充清理：移动更多WFO相关的旧报告

echo "【补充清理：etf_rotation_optimized WFO报告】"

ADDITIONAL_REPORTS=(
    "WFO_FINAL_VERIFICATION.md"
    "WFO_OPTIMIZED_CONFIG.md"
    "WFO_PERFORMANCE_ANALYSIS.md"
    "WFO_PERFORMANCE_BOTTLENECK_ANALYSIS.md"
    "WFO_REAL_DATA_VALIDATION.md"
    "WFO_STREAMING_REFACTOR.md"
    "WFO_TOP1000_UPDATE.md"
    "FILE_FORMAT_UPDATE.md"
)

count=0
for file in "${ADDITIONAL_REPORTS[@]}"; do
    if [ -f "$file" ]; then
        mv "$file" ../.archive_docs/etf_rotation_optimized_reports/
        echo "  ✓ $file"
        ((count++))
    fi
done

echo "✅ 补充清理完成: $count 个文件已归档"
echo ""
echo "保留的核心文档:"
ls -1 *.md
