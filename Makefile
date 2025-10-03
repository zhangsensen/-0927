# 量化交易开发环境 Makefile

.PHONY: help install format lint test clean run

help:  ## 显示帮助信息
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## 安装开发依赖
	uv sync --dev
	pre-commit install

format:  ## 格式化代码
	black .
	isort .

lint:  ## 运行代码检查
	flake8 factor_system/ --max-line-length=88 --extend-ignore=E203,W503
	mypy factor_system/ --ignore-missing-imports --no-strict-optional || true

test:  ## 运行测试
	pytest -v

test-cov:  ## 运行测试并生成覆盖率报告
	pytest --cov=factor_system --cov-report=html --cov-report=term-missing

clean:  ## 清理缓存和临时文件
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf .mypy_cache

check:  ## 运行所有质量检查
	pre-commit run --all-files

run-example:  ## 运行示例筛选任务
	cd factor_system/factor_screening && python professional_factor_screener.py --config configs/0700_multi_timeframe_config.yaml

update-deps:  ## 更新依赖
	uv sync --upgrade
	pre-commit autoupdate

setup-dev:  ## 初始化开发环境
	make install
	make pre-commit-install