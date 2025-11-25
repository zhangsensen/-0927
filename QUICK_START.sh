#!/usr/bin/env bash
# 📋 项目快速使用指南 (Quick Start)

## 🚀 环境激活

# Bash/Zsh
source .venv/bin/activate

# Fish
source .venv/bin/activate.fish

## 📦 依赖管理 (UV)

# 同步所有依赖 (包括开发工具)
uv sync --dev

# 仅核心依赖 (轻量)
uv sync

# 安装特定的可选组
uv sync --extra web      # Web框架
uv sync --extra database # 数据库
uv sync --extra dev      # 开发工具

# 更新依赖
uv lock
uv sync

## ✨ 主要项目入口

# 1. ETF轮动优化 (成熟管线)
cd etf_rotation_optimized
python run_combo_wfo.py          # WFO优化主程序
python real_backtest/test_freq_no_lookahead.py  # 无未来函数回测

# 2. A股策略
cd a_shares_strategy
python generate_a_share_factors.py

# 3. ETF数据下载
cd etf_download_manager
python download_etf_with_custom_dates.py

## 🧹 代码质量检查

# 格式化
black .
isort .

# 检查
ruff check .
mypy .

# 测试
pytest tests/ -v

# 代码覆盖
pytest tests/ --cov --cov-report=html

## 📚 文档

# ETF轮动项目文档
cat etf_rotation_optimized/README.md
cat etf_rotation_optimized/docs/PROJECT_OVERVIEW.md

# 清理和整合报告
cat CLEANUP_&_DEPENDENCY_CONSOLIDATION_REPORT.md

## 📦 项目结构

根目录/
  ├── pyproject.toml              # 统一依赖配置 (UV)
  ├── uv.lock                     # 依赖锁定 (218 包)
  ├── README.md                   # 项目说明
  ├── Makefile                    # 构建配置
  ├── .venv/                      # 虚拟环境 (212 包)
  ├── etf_rotation_optimized/  ⭐ # 成熟管线项目
  ├── a_shares_strategy/          # 量化策略
  ├── etf_download_manager/       # 数据下载
  ├── scripts/                    # 工具脚本
  ├── configs/                    # 配置文件
  ├── real_backtest/              # 回测框架
  └── _archive/                   # 已过时的 10 个项目

## 🔧 配置文件位置

# ETF轮动配置
etf_rotation_optimized/configs/combo_wfo_config.yaml

# 风险控制
config/risk_control_rules.yaml

# 节假日
config/cn_holidays.txt

## 💡 常见任务

# 查看已安装的包
pip list

# 添加新依赖
# 1. 编辑 pyproject.toml 的 dependencies 或 optional-dependencies
# 2. 运行: uv lock && uv sync

# 运行特定模块
python -m etf_rotation_optimized.core.data_loader

# 交互式开发
ipython
jupyter lab

## 🐛 故障排除

# 清空缓存
rm -rf .pytest_cache __pycache__ .mypy_cache

# 重新创建虚拟环境
rm -rf .venv
uv venv
uv sync --dev

# 检查 Python 版本
python --version  # 需要 3.11+

# 验证依赖
python -c "import vectorbt, polars, numba; print('✓ 核心包完整')"
