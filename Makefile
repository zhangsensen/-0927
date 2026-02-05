# 量化交易开发环境 Makefile
#
# 🔒 强制使用 UV 包管理器（2026-02-05 更新）
# ❌ 禁止: pip install, python -m venv, source .venv/bin/activate
# ✅ 必须: uv run python <script>, uv sync, uv add/remove
# 📖 详见: AGENTS.md 顶部说明

.PHONY: help install format lint test clean wfo vec bt pipeline all

# ============ 帮助 ============
help:  ## 显示帮助信息
	@echo "ETF 轮动策略研究平台 - 命令列表"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ============ 环境安装 ============
install:  ## 安装所有依赖（包括开发依赖）
	uv sync --dev
	@echo "✅ 依赖安装完成"

install-prod:  ## 仅安装生产依赖
	uv sync
	@echo "✅ 生产依赖安装完成"

# ============ 核心工作流 ============
wfo:  ## 运行 WFO 筛选（12,597 组合）
	uv run python src/etf_strategy/run_combo_wfo.py

vec:  ## 运行 VEC 批量回测
	uv run python scripts/batch_vec_backtest.py

bt:  ## 运行 BT 批量审计
	uv run python scripts/batch_bt_backtest.py

pipeline:  ## 运行完整流水线（WFO → VEC → BT → 验证）
	uv run python scripts/run_full_pipeline.py

all: wfo vec bt  ## 运行核心三层：WFO → VEC → BT

# ============ 代码质量 ============
format:  ## 格式化代码（black + isort）
	uv run black .
	uv run isort .

lint:  ## 运行代码检查（ruff + mypy）
	uv run ruff check src/etf_strategy/
	uv run mypy src/etf_strategy/ --ignore-missing-imports --no-strict-optional || true

check:  ## 运行所有质量检查（pre-commit）
	uv run pre-commit run --all-files

# ============ 测试 ============
test:  ## 运行测试
	uv run pytest -v

test-cov:  ## 运行测试并生成覆盖率报告
	uv run pytest --cov=etf_strategy --cov-report=html --cov-report=term-missing

# ============ 清理 ============
clean:  ## 清理缓存和临时文件
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache
	@echo "✅ 缓存清理完成"

# ============ 依赖管理 ============
update-deps:  ## 更新所有依赖
	uv sync --upgrade
	uv lock --upgrade
	@echo "✅ 依赖更新完成"

export-requirements:  ## 导出 requirements.txt（兼容模式）
	uv pip compile pyproject.toml -o requirements.txt
	@echo "✅ requirements.txt 已导出"

# ============ 开发辅助 ============
setup-dev: install  ## 初始化开发环境
	uv run pre-commit install
	@echo "✅ 开发环境初始化完成"
