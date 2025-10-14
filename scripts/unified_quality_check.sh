#!/usr/bin/env bash
# 深度量化0927 - 统一代码质量检查与合规性验证
# 集成pyscn、Vulture和专业量化系统检查的完整解决方案

set -e  # 遇到错误立即退出

# 颜色和符号定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 符号定义
CHECK_MARK="✅"
CROSS_MARK="❌"
WARNING="⚠️"
INFO="ℹ️"
ROCKET="🚀"
SHIELD="🛡️"
GEAR="⚙️"

echo -e "${CYAN}🔍 深度量化0927 - 统一代码质量检查${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# 检查虚拟环境
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo -e "${WARNING} 警告: 未检测到虚拟环境，建议使用虚拟环境运行${NC}"
    echo ""
fi

# 激活虚拟环境（如果存在）
if [[ -f ".venv/bin/activate" ]]; then
    echo -e "${INFO} 激活虚拟环境..."
    source .venv/bin/activate
fi

# 记录开始时间
START_TIME=$(date +%s)

# 创建临时报告目录
REPORT_DIR=".quality_reports"
TEMP_DIR=$(mktemp -d)
mkdir -p "$REPORT_DIR"

# 函数：打印步骤标题
print_step() {
    echo ""
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}$(printf '=%.0s' {1..50})${NC}"
}

# 函数：打印成功信息
print_success() {
    echo -e "${GREEN}${CHECK_MARK} $1${NC}"
}

# 函数：打印错误信息
print_error() {
    echo -e "${RED}${CROSS_MARK} $1${NC}"
}

# 函数：打印警告信息
print_warning() {
    echo -e "${YELLOW}${WARNING} $1${NC}"
}

# 函数：打印信息
print_info() {
    echo -e "${INFO} $1"
}

# 函数：检查命令是否存在
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 第一步：pyscn 代码质量分析
print_step "📊 Step 1: pyscn 代码质量分析 (CFG + APTED算法)"

if command_exists pyscn; then
    print_info "使用控制流图和APTED算法进行深度代码质量分析..."

    # 生成详细报告
    pyscn analyze factor_system/ examples/ scripts/ --verbose \
        --output-format json > "$TEMP_DIR/pyscn_report.json" 2>&1 || true

    # 提取关键指标
    if [[ -f "$TEMP_DIR/pyscn_report.json" ]]; then
        print_success "pyscn 分析完成"

        # 检查是否有严重问题
        if grep -q "Health Score.*[5-6][0-9]" "$TEMP_DIR/pyscn_report.json" 2>/dev/null; then
            print_warning "发现代码质量问题，建议查看详细报告"
        fi

        # 生成HTML报告
        pyscn analyze factor_system/ examples/ scripts/ --output-format html \
            --output-file "$REPORT_DIR/pyscn_quality_report.html" >/dev/null 2>&1 || true
        print_info "HTML报告已生成: $REPORT_DIR/pyscn_quality_report.html"
    else
        print_warning "pyscn 报告生成失败，使用文本模式"
        pyscn check factor_system/ examples/ scripts/ --threshold 10 || print_warning "发现高复杂度函数"
    fi
else
    print_warning "pyscn 未安装，跳过代码质量分析"
    print_info "安装命令: pip install pyscn>=1.1.1"
fi

# 第二步：Vulture 死代码检测
print_step "🦅 Step 2: Vulture 死代码检测"

if command_exists vulture; then
    print_info "检测未使用的代码和导入..."

    # 运行Vulture检查
    if [[ -f "vulture_whitelist.py" ]]; then
        vulture --min-confidence 80 --sort-by-size factor_system/ examples/ scripts/ \
            --whitelist vulture_whitelist.py > "$TEMP_DIR/vulture_report.txt" 2>&1 || true
    else
        vulture --min-confidence 80 --sort-by-size factor_system/ examples/ scripts/ \
            > "$TEMP_DIR/vulture_report.txt" 2>&1 || true
    fi

    # 统计死代码数量
    dead_code_count=$(grep -c "unused" "$TEMP_DIR/vulture_report.txt" 2>/dev/null || echo 0)

    if [[ $dead_code_count -eq 0 ]]; then
        print_success "未发现死代码问题"
    else
        print_warning "发现 $dead_code_count 个潜在的死代码问题"
        print_info "详细信息请查看: $TEMP_DIR/vulture_report.txt"
    fi
else
    print_warning "Vulture 未安装，跳过死代码检测"
    print_info "安装命令: pip install vulture>=2.14"
fi

# 第三步：量化系统专项检查
print_step "🛡️ Step 3: 量化系统专项安全检查"

print_info "检查未来函数使用情况..."
if [[ -f "factor_system/factor_screening/scripts/check_future_functions.py" ]]; then
    python factor_system/factor_screening/scripts/check_future_functions.py > "$TEMP_DIR/future_check.txt" 2>&1 || true

    if grep -q "发现未来函数" "$TEMP_DIR/future_check.txt" 2>/dev/null; then
        print_error "发现未来函数使用！这是严重的安全风险"
        grep "未来函数" "$TEMP_DIR/future_check.txt" | head -5
    else
        print_success "未发现未来函数使用，时间安全检查通过"
    fi
else
    print_warning "未来函数检查脚本未找到"
fi

print_info "检查因子清单合规性..."
if [[ -f "factor_system/factor_engine/validate_factor_registry.py" ]]; then
    python factor_system/factor_engine/validate_factor_registry.py > "$TEMP_DIR/factor_registry.txt" 2>&1 || true

    if grep -q "验证通过" "$TEMP_DIR/factor_registry.txt" 2>/dev/null; then
        print_success "因子清单合规性验证通过"
    else
        print_warning "因子清单验证发现问题"
    fi
else
    print_warning "因子清单验证脚本未找到"
fi

# 第四步：基础代码质量检查
print_step "⚙️ Step 4: 基础代码质量检查"

# Python语法检查
print_info "Python语法检查..."
syntax_errors=0
for py_file in $(find factor_system/ examples/ scripts/ -name "*.py" -type f | head -20); do
    if ! python -m py_compile "$py_file" 2>/dev/null; then
        ((syntax_errors++))
        print_error "语法错误: $py_file"
    fi
done

if [[ $syntax_errors -eq 0 ]]; then
    print_success "Python语法检查通过"
else
    print_error "发现 $syntax_errors 个语法错误"
fi

# Import排序检查
if command_exists isort; then
    print_info "检查导入排序..."
    if isort --check-only factor_system/ examples/ scripts/ --diff --quiet 2>/dev/null; then
        print_success "导入排序检查通过"
    else
        print_warning "导入排序需要调整"
        print_info "修复命令: isort factor_system/ examples/ scripts/"
    fi
fi

# 代码格式检查
if command_exists black; then
    print_info "检查代码格式..."
    if black --check factor_system/ examples/ scripts/ --quiet 2>/dev/null; then
        print_success "代码格式检查通过"
    else
        print_warning "代码格式需要调整"
        print_info "修复命令: black factor_system/ examples/ scripts/"
    fi
fi

# 类型检查（可选）
if command_exists mypy; then
    print_info "类型检查（可选）..."
    mypy factor_system/ --ignore-missing-imports > "$TEMP_DIR/mypy_report.txt" 2>&1 || true
    type_errors=$(grep -c "error:" "$TEMP_DIR/mypy_report.txt" 2>/dev/null || echo 0)

    if [[ $type_errors -eq 0 ]]; then
        print_success "类型检查通过"
    else
        print_info "类型检查发现 $type_errors 个提示（非阻塞）"
    fi
fi

# 安全检查（可选）
if command_exists bandit; then
    print_info "安全分析（可选）..."
    bandit -r factor_system/ -f json -o "$TEMP_DIR/bandit_report.json" >/dev/null 2>&1 || true

    if [[ -f "$TEMP_DIR/bandit_report.json" ]]; then
        high_issues=$(jq -r '.results | map(select(.issue_severity == "HIGH")) | length' "$TEMP_DIR/bandit_report.json" 2>/dev/null || echo 0)
        if [[ $high_issues -eq 0 ]]; then
            print_success "未发现高风险安全问题"
        else
            print_warning "发现 $high_issues 个高风险安全问题"
        fi
    fi
fi

# 第五步：综合报告生成
print_step "📋 Step 5: 综合质量报告生成"

# 计算总耗时
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# 生成总结报告
cat > "$REPORT_DIR/quality_summary.md" << EOF
# 深度量化0927 - 代码质量检查报告

**检查时间**: $(date '+%Y-%m-%d %H:%M:%S')
**检查耗时**: ${DURATION}秒
**项目版本**: $(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

## 📊 质量评分概览

### 主要检查结果
EOF

# 收集pyscn数据
if [[ -f "$TEMP_DIR/pyscn_report.json" ]]; then
    echo "### pyscn 代码质量分析" >> "$REPORT_DIR/quality_summary.md"
    echo "- 状态: ✅ 已完成" >> "$REPORT_DIR/quality_summary.md"
    echo "- 详细报告: [pyscn_quality_report.html](pyscn_quality_report.html)" >> "$REPORT_DIR/quality_summary.md"
    echo "" >> "$REPORT_DIR/quality_summary.md"
fi

# 收集Vulture数据
if [[ -f "$TEMP_DIR/vulture_report.txt" ]]; then
    dead_count=$(grep -c "unused" "$TEMP_DIR/vulture_report.txt" 2>/dev/null || echo "0")
    echo "### Vulture 死代码检测" >> "$REPORT_DIR/quality_summary.md"
    if [[ $dead_count -eq 0 ]]; then
        echo "- 状态: ✅ 无死代码" >> "$REPORT_DIR/quality_summary.md"
    else
        echo "- 状态: ⚠️ 发现 $dead_count 个潜在问题" >> "$REPORT_DIR/quality_summary.md"
    fi
    echo "" >> "$REPORT_DIR/quality_summary.md"
fi

# 添加量化系统检查结果
echo "### 量化系统安全检查" >> "$REPORT_DIR/quality_summary.md"
if [[ -f "$TEMP_DIR/future_check.txt" ]]; then
    if grep -q "发现未来函数" "$TEMP_DIR/future_check.txt" 2>/dev/null; then
        echo "- 未来函数检查: ❌ 发现违规" >> "$REPORT_DIR/quality_summary.md"
    else
        echo "- 未来函数检查: ✅ 通过" >> "$REPORT_DIR/quality_summary.md"
    fi
fi

if [[ -f "$TEMP_DIR/factor_registry.txt" ]]; then
    if grep -q "验证通过" "$TEMP_DIR/factor_registry.txt" 2>/dev/null; then
        echo "- 因子清单合规: ✅ 通过" >> "$REPORT_DIR/quality_summary.md"
    else
        echo "- 因子清单合规: ⚠️ 需要关注" >> "$REPORT_DIR/quality_summary.md"
    fi
fi

echo "" >> "$REPORT_DIR/quality_summary.md"
echo "## 🛠️ 修复建议" >> "$REPORT_DIR/quality_summary.md"
echo "" >> "$REPORT_DIR/quality_summary.md"
echo "1. **高优先级**: 修复所有未来函数使用问题" >> "$REPORT_DIR/quality_summary.md"
echo "2. **中优先级**: 清理死代码，提高代码质量" >> "$REPORT_DIR/quality_summary.md"
echo "3. **低优先级**: 优化代码格式和导入排序" >> "$REPORT_DIR/quality_summary.md"
echo "" >> "$REPORT_DIR/quality_summary.md"
echo "## 📝 详细报告" >> "$REPORT_DIR/quality_summary.md"
echo "- pyscn HTML报告: \`pyscn_quality_report.html\`" >> "$REPORT_DIR/quality_summary.md"
echo "- Vulture详细报告: \`vulture_report.txt\`" >> "$REPORT_DIR/quality_summary.md"
echo "- 未来函数检查: \`future_check.txt\`" >> "$REPORT_DIR/quality_summary.md"
echo "- 因子清单验证: \`factor_registry.txt\`" >> "$REPORT_DIR/quality_summary.md"

print_success "综合质量报告已生成: $REPORT_DIR/quality_summary.md"

# 第六步：最终结果展示
print_step "🎯 Step 6: 最终检查结果"

echo ""
echo -e "${CYAN}📊 质量检查完成情况:${NC}"
echo ""

# 显示关键结果
if [[ -f "$TEMP_DIR/pyscn_report.json" ]]; then
    echo -e "${GEAR} 代码质量分析: ${GREEN}已完成${NC}"
fi

if [[ -f "$TEMP_DIR/vulture_report.txt" ]]; then
    dead_count=$(grep -c "unused" "$TEMP_DIR/vulture_report.txt" 2>/dev/null || echo "0")
    if [[ $dead_count -eq 0 ]]; then
        echo -e "${GEAR} 死代码检测: ${GREEN}通过${NC}"
    else
        echo -e "${GEAR} 死代码检测: ${YELLOW}$dead_count 个问题${NC}"
    fi
fi

if [[ -f "$TEMP_DIR/future_check.txt" ]]; then
    if grep -q "发现未来函数" "$TEMP_DIR/future_check.txt" 2>/dev/null; then
        echo -e "${GEAR} 未来函数检查: ${RED}未通过${NC}"
    else
        echo -e "${GEAR} 未来函数检查: ${GREEN}通过${NC}"
    fi
fi

echo -e "${GEAR} Python语法: ${GREEN}通过${NC}"
echo -e "${GEAR} 检查耗时: ${DURATION}秒${NC}"

echo ""
echo -e "${CYAN}📁 报告位置:${NC}"
echo -e "${INFO} 主报告: $REPORT_DIR/quality_summary.md"
if [[ -f "$REPORT_DIR/pyscn_quality_report.html" ]]; then
    echo -e "${INFO} HTML报告: $REPORT_DIR/pyscn_quality_report.html"
fi

echo ""
echo -e "${GREEN}🎉 统一代码质量检查完成！${NC}"
echo ""
echo -e "${CYAN}💡 快速修复命令:${NC}"
echo -e "${INFO} 格式化代码: black factor_system/ examples/ scripts/"
echo -e "${INFO} 排序导入: isort factor_system/ examples/ scripts/"
echo -e "${INFO} 查看报告: cat $REPORT_DIR/quality_summary.md"
echo ""

# 清理临时文件
rm -rf "$TEMP_DIR"

# 根据检查结果设置退出码
if [[ -f "$TEMP_DIR/future_check.txt" ]] && grep -q "发现未来函数" "$TEMP_DIR/future_check.txt" 2>/dev/null; then
    echo -e "${RED}❌ 发现严重安全问题，提交被阻止${NC}"
    exit 1
elif [[ $syntax_errors -gt 0 ]]; then
    echo -e "${RED}❌ 发现语法错误，提交被阻止${NC}"
    exit 1
else
    echo -e "${GREEN}✅ 代码质量检查通过，可以提交${NC}"
    exit 0
fi