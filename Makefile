# 量化交易开发环境 Makefile
# 使用 UV 作为包管理器

.PHONY: help install format lint test clean run wfo vec bt verify all

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
	uv run python etf_rotation_optimized/run_unified_wfo.py

vec:  ## 运行 VEC 批量回测
	uv run python scripts/batch_vec_backtest.py

bt:  ## 运行 BT 批量审计
	uv run python scripts/batch_bt_backtest.py

verify:  ## 验证 VEC/BT 对齐（< 0.01pp）
	uv run python scripts/full_vec_bt_comparison.py

all: wfo vec bt verify  ## 运行完整工作流：WFO → VEC → BT → 验证

# ============ 代码质量 ============
format:  ## 格式化代码（black + isort）
	uv run black .
	uv run isort .

lint:  ## 运行代码检查（flake8 + mypy）
	uv run flake8 factor_system/ --max-line-length=88 --extend-ignore=E203,W503
	uv run mypy factor_system/ --ignore-missing-imports --no-strict-optional || true

check:  ## 运行所有质量检查（pre-commit）
	uv run pre-commit run --all-files

# ============ 测试 ============
test:  ## 运行测试
	uv run pytest -v

test-cov:  ## 运行测试并生成覆盖率报告
	uv run pytest --cov=factor_system --cov-report=html --cov-report=term-missing

# ============ 清理 ============
clean:  ## 清理缓存和临时文件
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache
	@echo "✅ 缓存清理完成"

clean-all: clean  ## 深度清理（包括因子缓存）
	uv run python scripts/cache_cleaner.py --all
	@echo "✅ 深度清理完成"

# ============ 依赖管理 ============
update-deps:  ## 更新所有依赖
	uv sync --upgrade
	uv lock --upgrade
	@echo "✅ 依赖更新完成"

export-requirements:  ## 导出 requirements.txt（兼容模式）
	uv pip compile pyproject.toml -o requirements.txt
	@echo "✅ requirements.txt 已导出"

# ============ 审计与监控 ============
audit:  ## 审计候选池（需要 RUN_DIR 参数）
	@if [ -z "$(RUN_DIR)" ]; then echo "Usage: make audit RUN_DIR=results/run_YYYYMMDD_HHMMSS"; exit 2; fi
	uv run python scripts/audit_candidate_pool.py --run-dir $(RUN_DIR)

monitor:  ## 监控 WFO 排名质量（需要 RUN_DIR 参数）
	@if [ -z "$(RUN_DIR)" ]; then echo "Usage: make monitor RUN_DIR=results/run_YYYYMMDD_HHMMSS"; exit 2; fi
	uv run python scripts/monitor_wfo_rank_quality.py --run-dir $(RUN_DIR) --thresholds config/monitor_thresholds.yaml

# ============ 开发辅助 ============
setup-dev: install  ## 初始化开发环境
	uv run pre-commit install
	@echo "✅ 开发环境初始化完成"