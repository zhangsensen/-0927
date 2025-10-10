# 深度量化0927项目概览

## 项目定位
专业级量化交易开发环境，统一因子计算引擎，支持多市场（A股、港股、美股）。

## 核心架构

### 1. FactorEngine 统一因子引擎
- **位置**: `factor_system/factor_engine/`
- **功能**: 统一因子计算核心，确保研究、回测、组合管理的一致性
- **因子数量**: 154个技术指标
- **性能**: VectorBT集成，10-50倍性能提升
- **缓存**: 双层缓存（内存+磁盘）

### 2. A股技术分析框架
- **位置**: `a股/`
- **特色**: 154个技术指标，中国市场专门分析
- **主要文件**:
  - `stock_analysis/sz_technical_analysis.py` - 主分析引擎
  - `data_download/simple_download.py` - 数据下载
  - `screen_top_stocks.py` - 股票筛选

### 3. 专业因子筛选系统
- **位置**: `factor_system/factor_screening/`
- **评价维度**: 5维度综合评价（预测能力35%、稳定性25%、独立性20%、实用性15%、短期适应性5%）
- **统计严谨性**: Benjamini-Hochberg FDR校正，VIF检测
- **性能**: 5.7因子/秒完整分析

### 4. 港股中频策略
- **位置**: `hk_midfreq/`
- **支持股票**: 276+港股
- **数据精度**: 分钟级
- **因子引擎适配**: 已与FactorEngine集成

## 技术栈
- **核心**: Python 3.11+, VectorBT, pandas, NumPy
- **技术分析**: TA-Lib, scikit-learn, scipy
- **数据存储**: PyArrow, Parquet
- **可视化**: matplotlib, seaborn, plotly

## 关键命令
```bash
# 环境管理
uv sync && source .venv/bin/activate

# A股分析
python a股/stock_analysis/sz_technical_analysis.py <STOCK_CODE>

# 因子计算
python factor_system/factor_generation/quick_start.py <STOCK_CODE>

# 专业筛选
python factor_system/factor_screening/cli.py screen <STOCK_CODE> <TIMEFRAME>

# FactorEngine一致性测试
python tests/test_factor_engine_consistency.py

# 批量处理
python factor_system/factor_screening/batch_screener.py
```

## 配置管理
- **FactorEngine**: 环境变量配置，预置开发/研究/生产环境
- **因子系统**: Python类+YAML配置管理
- **策略模板**: 长期、保守、高频、激进策略配置

## 质量标准
- **代码质量**: Black格式化，isort导入排序，mypy类型检查
- **测试覆盖**: 95%+覆盖率要求
- **性能标准**: 内存效率>70%，关键路径<1ms

## 市场覆盖
- **A股**: 中国股市专门分析
- **港股**: 276+股票，分钟级数据
- **美股**: 172+股票，多时间周期

## 当前状态
- **主分支**: master
- **最近提交**: 架构优化与代码清理 (9ab5e2b)
- **FactorEngine**: 已就绪
- **集成状态**: 各模块与FactorEngine集成完成

## 特色功能
- **多时间框架**: 1min到daily，自动对齐
- **统计严谨性**: 无前瞻偏差，处理幸存者偏差
- **风险管理**: VaR、最大回撤、夏普比率
- **实用指标**: 换手率、流动性要求评估