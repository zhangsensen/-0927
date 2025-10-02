# 专业级因子筛选系统

## 🎯 项目概述

这是一个专业级的量化交易因子筛选系统，支持多时间框架、多市场的154项技术指标综合分析。系统采用5维度评估框架，提供统计严谨的因子筛选结果。

## 🏗️ 核心架构

### 主要模块
- **`professional_factor_screener.py`** - 核心筛选引擎
- **`enhanced_result_manager.py`** - 结果管理和报告生成
- **`config_manager.py`** - 配置管理
- **`utils.py`** - 工具函数（文件对齐等）

### 目录结构
```
factor_screening/
├── professional_factor_screener.py    # 主程序 ⭐
├── enhanced_result_manager.py         # 结果管理
├── config_manager.py                  # 配置管理
├── utils.py                          # 工具函数
├── configs/                          # 配置文件目录
│   └── 0700_multi_timeframe_config.yaml
├── cache/                            # 缓存目录
└── 因子筛选/                         # 输出结果目录
```

## 🚀 快速启动

### 环境准备
```bash
# 确保在项目根目录
cd /Users/zhangshenshen/深度量化0927/factor_system/factor_screening

# 激活虚拟环境（如果需要）
source ../../.venv/bin/activate
```

### 基础命令

#### 1. 单个时间框架分析
```bash
# 60分钟时间框架分析
uv run python professional_factor_screener.py --symbol 0700.HK --timeframe 60min

# 日线分析
uv run python professional_factor_screener.py --symbol 0700.HK --timeframe daily
```

#### 2. 批量多时间框架分析（推荐）
```bash
# 执行所有6个时间框架的完整分析
uv run python professional_factor_screener.py --config configs/0700_multi_timeframe_config.yaml
```

#### 3. 其他股票分析
```bash
# 替换股票代码，支持港股、美股、A股
uv run python professional_factor_screener.py --symbol 0005.HK --timeframe 60min  # 汇丰控股
uv run python professional_factor_screener.py --symbol AAPL --timeframe daily   # 苹果公司
```

## 📊 分析能力

### 154项技术指标
- **核心技术指标** (36项)：移动平均、MACD、RSI、布林带等
- **增强指标** (118项)：高级MA、震荡器、趋势、统计、周期指标

### 5维度评估框架
1. **预测能力** (35%) - 多周期IC分析、IC衰减、持续性
2. **稳定性** (25%) - 滚动IC、截面稳定性、一致性
3. **独立性** (20%) - VIF检测、相关性分析、信息增量
4. **实用性** (15%) - 交易成本、换手率、流动性需求
5. **短期适应性** (5%) - 反转效应、动量持续性、波动率敏感性

### 支持时间框架
- **分钟级**：1min, 2min, 3min, 5min, 15min, 30min, 60min
- **小时级**：4hour
- **日线级**：daily

## 📈 输出结果

### 结果文件

#### 单个时间框架分析
执行后在 `因子筛选/` 目录下生成：
```
因子筛选/
└── 0700.HK_60min_20251002_161254/    # 时间戳会话目录
    ├── screening_report.csv           # 筛选报告
    ├── session_info.json              # 会话信息
    ├── factor_metrics.json            # 因子指标详情
    └── comprehensive_scores.csv       # 综合评分
```

#### 批量多时间框架分析（推荐）
```
因子筛选/
└── 0700.HK_multi_timeframe_20251002_161254/  # 统一会话目录
    ├── 0700_multi_timeframe_summary_*.csv     # 多时间框架汇总报告
    ├── 0700_best_factors_overall_*.csv        # 跨时间框架最佳因子排行
    ├── 0700_batch_statistics_*.csv            # 批量处理统计摘要
    └── [各时间框架的详细报告文件]             # 所有时间框架的完整结果
```

**简化特性**：
- 🎯 **统一目录**：所有时间框架结果保存在同一会话目录
- 📊 **汇总报告**：自动生成跨时间框架的因子排行和统计
- 🗂️ **简化查找**：无需在多个子目录间切换查找结果

### 因子质量分级
- **🥇 Tier 1** (≥0.8) - 核心因子，强烈推荐
- **🥈 Tier 2** (0.6-0.8) - 重要因子，推荐使用
- **🥉 Tier 3** (0.4-0.6) - 备用因子，谨慎使用
- **❌ 不推荐** (<0.4) - 不建议使用

### 统计显著性
- ***** p < 0.001 - 高度显著
- **** p < 0.01 - 显著
- *** p < 0.05 - 边际显著

## ⚙️ 配置说明

### 批量配置文件
`configs/0700_multi_timeframe_config.yaml` 包含6个时间框架的完整配置：
- 5min, 15min, 30min, 60min, 4hour, daily
- 每个时间框架有独立的参数设置
- 支持并行处理和错误恢复

### 主要参数
```yaml
# 分析参数
ic_horizons: [1, 3, 5, 10, 20]        # IC预测周期
min_ic_threshold: 0.015                # 最小IC阈值
min_stability_threshold: 0.6           # 最小稳定性阈值

# 成本设置
commission_rate: 0.002                 # 佣金费率
slippage_bps: 5.0                      # 滑点(基点)

# 权重设置
weights:
  predictive_power: 0.35               # 预测能力权重
  stability: 0.25                       # 稳定性权重
  independence: 0.20                    # 独立性权重
  practicality: 0.10                    # 实用性权重
  short_term_fitness: 0.10              # 短期适应性权重
```

## 🔧 技术特性

### 性能优化
- **VectorBT集成** - 10-50x性能提升
- **向量化计算** - 消除Python循环
- **内存优化** - 40-60%内存使用减少
- **缓存机制** - 避免重复计算

### 统计严谨性
- **Benjamini-Hochberg FDR校正** - 多重比较校正
- **严格显著性检验** - α = 0.01, 0.05, 0.10
- **无前视偏差** - 严格的偏差预防
- **真实市场数据** - 无模拟数据

### 错误处理
- **智能数据验证** - 自动质量检查
- **优雅错误恢复** - 单个失败不影响整体
- **详细日志记录** - 完整的执行追踪

## 📝 使用示例

### 完整工作流程
```bash
# 1. 执行多时间框架分析
uv run python professional_factor_screener.py --config configs/0700_multi_timeframe_config.yaml

# 2. 查看结果
ls -la 因子筛选/
cat 因子筛选/*/screening_report.csv

# 3. 分析最佳因子
head -10 因子筛选/*/comprehensive_scores.csv
```

### 单股票深入研究
```bash
# 分析腾讯控股不同时间框架
for tf in 15min 30min 60min daily; do
    echo "=== 分析时间框架: $tf ==="
    uv run python professional_factor_screener.py --symbol 0700.HK --timeframe $tf
done
```

## 🎯 性能基准

### 处理速度
- **小规模** (500样本×20因子)：831+ 因子/秒
- **中规模** (1000样本×50因子)：864+ 因子/秒
- **大规模** (2000样本×100因子)：686+ 因子/秒

### 内存使用
- **中规模数据**：< 1MB
- **大规模数据**：< 50MB
- **缓存机制**：智能内存管理

## ⚠️ 注意事项

1. **数据依赖** - 确保因子数据在 `../因子输出/` 目录下
2. **时间框架匹配** - 确保因子文件时间框架与请求匹配
3. **磁盘空间** - 批量分析会生成较多结果文件
4. **计算时间** - 完整批量分析可能需要几分钟

## 🔍 故障排除

### 常见问题
```bash
# 检查因子数据是否存在
ls -la ../因子输出/60min/0700.HK_*.parquet

# 检查配置文件
cat configs/0700_multi_timeframe_config.yaml

# 查看详细日志
tail -f 因子筛选/*/professional_screener_*.log
```

### 清理缓存（如需要）
```bash
rm -rf cache/*
rm -rf logs/*
```

## 🎯 系统简化特性

### 统一架构设计
- **单一会话目录**：批量分析时所有时间框架共享一个会话目录
- **统一日志管理**：日志只保存在全局位置 `logs/screening/`，避免重复
- **汇总报告**：自动生成跨时间框架的因子对比和统计分析

### 简化前后对比
**简化前**：
- 每个时间框架创建独立会话目录
- 日志文件同时存在全局和会话目录
- 需要在多个目录间查找结果

**简化后**：
- ✅ 批量分析使用统一会话目录
- ✅ 日志只在全局位置保存
- ✅ 自动生成汇总报告和最佳因子排行
- ✅ 一站式查看所有分析结果

---

**系统版本**: 2.0.0 (简化架构版)
**最后更新**: 2025-10-02
**作者**: 量化首席工程师