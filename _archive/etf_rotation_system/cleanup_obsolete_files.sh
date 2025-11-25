#!/bin/bash
# ETF轮动系统 - 过期文件清理脚本
# 执行前已创建备份: etf_rotation_system_backup_20251021_235648.tar.gz

set -e

echo "🧹 开始清理过期文件..."
echo "备份已创建: etf_rotation_system_backup_20251021_235648.tar.gz"
echo ""

# 统计待清理文件
echo "📊 待清理文件统计:"
echo "  - deprecated目录: $(find 01_横截面建设/deprecated -type f | wc -l) 个文件"
echo "  - archive_docs目录: $(find 03_vbt回测/archive_docs -type f | wc -l) 个文件"
echo "  - archive_tests目录: $(find 03_vbt回测/archive_tests -type f | wc -l) 个文件"
echo "  - archive_tasks目录: $(find 03_vbt回测/archive_tasks -type f | wc -l) 个文件"
echo "  - 临时测试文件: 4 个"
echo ""

# 确认清理（安全措施）
read -p "确认清理以上文件? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "❌ 取消清理"
    exit 0
fi

echo ""
echo "🗑️  开始清理..."

# 1. 清理deprecated目录
echo "  [1/5] 清理deprecated目录..."
rm -rf 01_横截面建设/deprecated/

# 2. 清理archive_docs
echo "  [2/5] 清理archive_docs..."
rm -rf 03_vbt回测/archive_docs/

# 3. 清理archive_tests
echo "  [3/5] 清理archive_tests..."
rm -rf 03_vbt回测/archive_tests/

# 4. 清理archive_tasks
echo "  [4/5] 清理archive_tasks..."
rm -rf 03_vbt回测/archive_tasks/

# 5. 清理临时测试文件
echo "  [5/5] 清理临时测试文件..."
cd 03_vbt回测
rm -f verify_unstack_order.py verify_fix.py verify_deterministic.py test_optimization_debug.py
cd ..

echo ""
echo "✅ 清理完成！"
echo ""
echo "📦 保留的核心文件:"
echo "  - 01_横截面建设/generate_panel_refactored.py"
echo "  - 02_因子筛选/run_etf_cross_section_configurable.py"
echo "  - 03_vbt回测/parallel_backtest_configurable.py"
echo "  - 03_vbt回测/test_real_data.py (生产验收测试)"
echo "  - 04_精细策略/main.py"
echo ""
echo "若需恢复，解压备份:"
echo "  tar -xzf ../etf_rotation_system_backup_20251021_235648.tar.gz"
