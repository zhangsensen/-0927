# ETF轮动量化交易系统 Makefile

.PHONY: help install panel screen backtest wfo pipeline clean test

# 默认目标：显示帮助
help:
	@echo "=================================="
	@echo "ETF轮动系统 - 可用命令"
	@echo "=================================="
	@echo ""
	@echo "环境管理:"
	@echo "  make install       - 安装依赖"
	@echo ""
	@echo "数据流程:"
	@echo "  make panel         - 生成因子面板"
	@echo "  make screen        - 因子筛选"
	@echo "  make backtest      - VBT回测"
	@echo "  make wfo           - WFO优化"
	@echo ""
	@echo "完整流程:"
	@echo "  make pipeline      - 运行完整流程"
	@echo ""
	@echo "维护:"
	@echo "  make clean         - 清理缓存"
	@echo "  make test          - 运行测试"
	@echo ""

# 安装依赖
install:
	@echo "📦 安装依赖..."
	pip install -r requirements.txt
	@echo "✅ 依赖安装完成"

# 生成因子面板
panel:
	@echo "📊 生成因子面板..."
	cd 01_横截面建设 && python3 generate_panel_refactored.py
	@echo "✅ 面板生成完成"

# 因子筛选
screen:
	@echo "🔬 因子筛选..."
	cd 02_因子筛选 && python3 run_etf_cross_section_configurable.py --config optimized_screening_config.yaml
	@echo "✅ 筛选完成"

# VBT回测
backtest:
	@echo "🚀 VBT回测..."
	cd 03_vbt回测 && python3 parallel_backtest_configurable.py --config-file parallel_backtest_config.yaml
	@echo "✅ 回测完成"

# WFO优化
wfo:
	@echo "🎯 WFO优化..."
	cd 03_vbt_wfo && python3 production_runner_optimized.py
	@echo "✅ WFO完成"

# 完整流程
pipeline: panel screen wfo
	@echo ""
	@echo "=================================="
	@echo "🎉 完整流程执行完成！"
	@echo "=================================="
	@echo ""
	@echo "结果位置:"
	@echo "  面板: data/results/panels/"
	@echo "  筛选: data/results/screening/"
	@echo "  WFO:  data/results/vbtwfo/"
	@echo ""

# 清理缓存
clean:
	@echo "🧹 清理缓存..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".numba_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "✅ 清理完成"

# 运行测试
test:
	@echo "🧪 运行测试..."
	python3 test_full_pipeline.py
	@echo "✅ 测试完成"

# 查看最新结果
latest:
	@echo "📁 最新结果:"
	@echo ""
	@echo "面板:"
	@ls -lt data/results/panels/ | head -3
	@echo ""
	@echo "筛选:"
	@ls -lt data/results/screening/ | head -3
	@echo ""
	@echo "WFO:"
	@ls -lt data/results/vbtwfo/ | head -3
