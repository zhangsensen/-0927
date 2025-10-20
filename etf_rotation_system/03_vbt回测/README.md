# VBT量化回测模块

## 功能概述
本模块基于VectorBT提供高性能的量化回测功能，支持：
- 大规模策略优化 (50,000+策略组合)
- 多进程并行计算
- 完整的性能指标分析
- ETF轮动策略回测

## 核心文件
- `backtest_engine_full.py` - 专业VBT回测引擎，支持大规模策略优化
- `simple_backtest.py` - 简单回测工具
- `optimize_threshold_rebalancing.py` - 阈值优化和调仓分析
- `verify_migration.py` - 数据迁移验证工具
- `run.py` - 主运行脚本
- `strategies/` - 策略实现文件夹

## 使用方法

### 大规模策略优化
```bash
cd 03_vbt回测
python backtest_engine_full.py \
  --factor-panel ../../etf_cross_section_results/panel_20251018_021356.parquet \
  --data-dir ../../raw/ETF/daily \
  --factors MOMENTUM_20D RSI_14 VOLATILITY_60D \
  --weight-grid 0.0 0.2 0.4 0.6 0.8 1.0 \
  --max-total-combos 50000 \
  --top-n-list 3 5 8 10 12 \
  --fees 0.001
```

### 简单回测
```bash
python simple_backtest.py \
  --panel ../../etf_cross_section_results/panel_20251018_021356.parquet \
  --data-dir ../../raw/ETF/daily
```

## 输出结果
- 策略优化结果 (CSV格式)
- 性能指标：夏普比率、卡尔玛比率、最大回撤
- 最优策略配置 (JSON格式)
- 检查点文件 (用于增量计算)

## 性能特性
- 处理速度：1,200+ 策略/秒
- 支持策略：50,000+ 组合
- 多进程并行：4 workers
- 内存优化：分块处理

## 核心技术
- VectorBT向量化计算
- numpy.einsum张量运算
- 多进程并行处理
- 自适应批处理

## 回测指标
- 年化收益率
- 夏普比率
- 最大回撤
- 卡尔玛比率
- 胜率
- 换手率