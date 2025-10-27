# ETF轮动系统优化版 Makefile

.PHONY: install test clean run-pipeline generate-factors screen-factors backtest signal monitor

# 安装依赖
install:
	pip install -r requirements.txt

# 运行测试
test:
	python -m pytest tests/

# 清理缓存
clean:
	rm -rf cache/*
	rm -rf output/*
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +

# 运行完整流程
run-pipeline:
	python main.py pipeline

# 生成因子
generate-factors:
	python main.py generate-factors

# 筛选因子
screen-factors:
	@latest_panel=$$(ls -t output/panel_*.parquet 2>/dev/null | head -1); \
	if [ -z "$$latest_panel" ]; then \
		echo "Error: No panel file found. Run 'make generate-factors' first."; \
		exit 1; \
	fi; \
	python main.py screen-factors --panel $$latest_panel

# 运行回测
backtest:
	@latest_panel=$$(ls -t output/panel_*.parquet 2>/dev/null | head -1); \
	latest_factors=$$(ls -t output/screening_*/selected_factors.txt 2>/dev/null | head -1); \
	if [ -z "$$latest_panel" ]; then \
		echo "Error: No panel file found. Run 'make generate-factors' first."; \
		exit 1; \
	fi; \
	if [ -n "$$latest_factors" ]; then \
		python main.py run-backtest --panel $$latest_panel --factors $$latest_factors --mode wfo; \
	else \
		python main.py run-backtest --panel $$latest_panel --mode backtest; \
	fi

# 生成交易信号
signal:
	@latest_panel=$$(ls -t output/panel_*.parquet 2>/dev/null | head -1); \
	latest_factors=$$(ls -t output/screening_*/selected_factors.txt 2>/dev/null | head -1); \
	if [ -z "$$latest_panel" ]; then \
		echo "Error: No panel file found. Run 'make generate-factors' first."; \
		exit 1; \
	fi; \
	if [ -n "$$latest_factors" ]; then \
		python main.py generate-signal --panel $$latest_panel --factors $$latest_factors --mode paper; \
	else \
		python main.py generate-signal --panel $$latest_panel --mode paper; \
	fi

# 监控模式
monitor:
	@latest_panel=$$(ls -t output/panel_*.parquet 2>/dev/null | head -1); \
	latest_factors=$$(ls -t output/screening_*/selected_factors.txt 2>/dev/null | head -1); \
	if [ -z "$$latest_panel" ]; then \
		echo "Error: No panel file found. Run 'make generate-factors' first."; \
		exit 1; \
	fi; \
	echo "Starting monitor mode (Ctrl+C to stop)..."; \
	if [ -n "$$latest_factors" ]; then \
		python -c "from main import *; from signal_generator import SignalGenerator; import pandas as pd; \
		cfg = load_config('config.yaml'); gen = SignalGenerator(cfg); \
		panel = pd.read_parquet('$$latest_panel'); \
		factors = open('$$latest_factors').read().strip().split('\n'); \
		gen.monitor(panel, factors)"; \
	else \
		python -c "from main import *; from signal_generator import SignalGenerator; import pandas as pd; \
		cfg = load_config('config.yaml'); gen = SignalGenerator(cfg); \
		panel = pd.read_parquet('$$latest_panel'); \
		gen.monitor(panel, panel.columns.tolist())"; \
	fi

# 帮助
help:
	@echo "ETF轮动系统优化版 - 可用命令:"
	@echo "  make install       - 安装依赖"
	@echo "  make test         - 运行测试"
	@echo "  make clean        - 清理缓存"
	@echo "  make run-pipeline - 运行完整流程"
	@echo "  make generate-factors - 生成因子面板"
	@echo "  make screen-factors   - 筛选因子"
	@echo "  make backtest        - 运行回测"
	@echo "  make signal          - 生成交易信号"
	@echo "  make monitor         - 启动监控模式"
