# A股量化交易系统

## 🎯 系统概述

A股系统是深度量化框架的重要组成部分，专注于资金流因子计算和A股特有交易约束。系统完全集成到FactorEngine架构中，支持154个技术指标和15个专业资金流因子。

## ✅ 核心功能

### 1. **资金流因子系统**
- **8个核心因子**: MainNetInflow_Rate, LargeOrder_Ratio, SuperLargeOrder_Ratio等
- **4个增强因子**: Institutional_Absorption, Flow_Tier_Ratio_Delta等
- **3个约束因子**: Gap信号, 尾盘抢筹, 可交易性掩码
- **T+1执行约束**: 14:30信号冻结，符合A股交易规则

### 2. **数据管道**
- **统一数据加载**: 支持parquet格式，自动时区转换
- **质量验证**: OHLC逻辑检查，交易时间验证
- **标准化处理**: 字段映射，数据清洗

### 3. **多时间框架支持**
- 支持1min到monthly全时间框架
- 自动重采样和信号对齐
- A股交易时间识别（09:30-11:30, 13:00-15:00）

## 🚀 快速开始

### 环境准备
```bash
# 安装依赖
uv sync
source .venv/bin/activate
```

### 资金流因子计算
```bash
# 快速演示（资金流因子）
python examples/moneyflow_quickstart.py

# A股技术因子生成
python a_shares_strategy/generate_a_share_factors.py 600036.SH --timeframe 5min
```

### 系统测试
```bash
# A股集成测试
python tests/test_a_share_provider_integration.py -v

# 资金流综合测试
python tests/development/test_moneyflow_integration_comprehensive.py -v
```

## 📊 技术特性

- **向量化计算**: 100%向量化实现
- **无未来函数**: 严格时序对齐
- **T+1兼容**: 符合A股交易规则
- **可交易性过滤**: 自动识别停牌等不可交易时段

## 📁 数据格式

### 输入数据
```bash
# A股资金流数据路径
raw/SH/money_flow/{symbol}_money_flow.parquet

# 数据字段
columns: [timestamp, open, high, low, close, volume, turnover, ...]
```

### 输出因子
```bash
# 因子输出路径
factor_system/factor_output/A_SHARES/{timeframe}/{symbol}_{timeframe}_factors.parquet
```

## 📚 相关文档

- **核心指南**: `CLAUDE.md` - 完整系统说明
- **因子集**: `README_FACTOR_SETS.md` - 因子管理
- **快速开始**: `examples/moneyflow_quickstart.py` - 完整示例

## 🎯 下一步开发

- [ ] 因子筛选系统集成
- [ ] A股特色因子开发
- [ ] T+1策略框架
- [ ] 板块热度分析

---
**维护**: A股量化团队 | **更新**: 2025-10-14
