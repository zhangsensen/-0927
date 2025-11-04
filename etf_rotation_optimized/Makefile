# ============================================================================
# ETF轮动系统优化版 - Makefile
# ============================================================================
# 重构日期: 2025-10-28
# 架构: 横截面 -> 因子筛选 -> WFO -> VBT回测
# ============================================================================

.PHONY: install test clean run run-steps help

# ============================================================================
# 环境管理
# ============================================================================

install:
	@echo "📦 安装依赖..."
	pip install -r requirements.txt

# ============================================================================
# 测试
# ============================================================================

test:
	@echo "🧪 运行测试..."
	python -m pytest tests/ -v

# ============================================================================
# 清理
# ============================================================================

clean:
	@echo "🧹 清理缓存和输出..."
	rm -rf cache/*
	rm -rf results/*
	rm -rf output/*
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

# ============================================================================
# 运行流程
# ============================================================================

run:
	@echo "🚀 运行完整流程（横截面 -> 因子筛选 -> WFO -> 回测）..."
	python main.py run --config configs/default.yaml

# 运行单个步骤
run-cross-section:
	@echo "▶️  运行横截面加工..."
	python main.py run-steps --config configs/default.yaml --steps cross_section

run-factor-selection:
	@echo "▶️  运行因子筛选..."
	python main.py run-steps --config configs/default.yaml --steps factor_selection

run-wfo:
	@echo "▶️  运行WFO验证..."
	python main.py run-steps --config configs/default.yaml --steps wfo

run-backtest:
	@echo "▶️  运行VBT回测..."
	python main.py run-steps --config configs/default.yaml --steps backtest

# ============================================================================
# 帮助
# ============================================================================

help:
	@echo "============================================================================"
	@echo "ETF轮动系统优化版 v2.0 - 可用命令"
	@echo "============================================================================"
	@echo ""
	@echo "环境管理:"
	@echo "  make install              - 安装依赖"
	@echo ""
	@echo "测试:"
	@echo "  make test                 - 运行测试"
	@echo ""
	@echo "清理:"
	@echo "  make clean                - 清理缓存和输出"
	@echo ""
	@echo "运行流程:"
	@echo "  make run                  - 运行完整流程"
	@echo "  make run-cross-section    - 运行横截面加工"
	@echo "  make run-factor-selection - 运行因子筛选"
	@echo "  make run-wfo              - 运行WFO验证"
	@echo "  make run-backtest         - 运行VBT回测"
	@echo ""
	@echo "直接使用 main.py:"
	@echo "  python main.py run --config configs/default.yaml"
	@echo "  python main.py run-steps --config configs/default.yaml --steps cross_section"
	@echo ""
	@echo "============================================================================"
