# 技术规格说明

## FactorEngine技术架构

### 核心组件
```
factor_system/factor_engine/
├── api.py                 # 统一API入口
├── core/
│   ├── engine.py         # 主计算引擎
│   ├── registry.py       # 因子注册管理
│   ├── cache.py          # 双层缓存系统
│   └── vectorbt_adapter.py # VectorBT集成
├── providers/            # 数据提供者
├── factors/             # 154个技术指标
│   ├── technical/       # 37个技术指标 (RSI, MACD, STOCH...)
│   ├── overlap/         # 12个重叠研究 (SMA, EMA, BBANDS...)
│   ├── statistic/       # 15个统计函数
│   └── pattern/         # 60+个蜡烛图模式
└── settings.py          # 环境配置
```

### 154指标分类
**核心技术指标 (36个)**
- 移动平均: MA5, MA10, MA20, MA30, MA60, EMA5, EMA12, EMA26
- 动量: MACD, RSI, Stochastic, Williams %R, CCI, MFI
- 波动率: 布林带, ATR, 标准差
- 成交量: OBV, Volume SMA, 成交量比率

**增强指标 (118个)**
- 高级MA: DEMA, TEMA, T3, KAMA, Hull MA
- 振荡器: TRIX, ROC, CMO, ADX, DI+, DI-
- 趋势: 抛物线SAR, Aroon, Chande动量
- 统计: Z-Score, 相关性, Beta, Alpha
- 周期: Hilbert变换, 正弦波, 趋势线

## 性能优化技术

### VectorBT集成
- **性能提升**: 10-50倍于传统pandas
- **向量化**: 消除Python循环
- **内存优化**: 40-60%内存使用减少
- **并行计算**: 符号级并行化

### 缓存策略
```
双层缓存架构:
├── 内存缓存: 快速访问，TTL配置
└── 磁盘缓存: 持久化存储，大容量
```

### 环境配置
```python
# 开发环境
- 缓存: 200MB, 2小时TTL
- 并行: 单线程，详细日志

# 研究环境
- 缓存: 512MB, 24小时TTL
- 并行: 4核心，信息日志

# 生产环境
- 缓存: 1GB, 7天TTL
- 并行: 全核心，警告日志
```

## 5维度筛选框架

### 权重分配
1. **预测能力 (35%)**: 多周期IC分析 (1,3,5,10,20天)
2. **稳定性 (25%)**: 滚动窗口IC分析，截面稳定性
3. **独立性 (20%)**: VIF检测，因子相关性分析
4. **实用性 (15%)**: 交易成本评估，换手率分析
5. **短期适应性 (5%)**: 反转效应检测，动量持续性

### 统计方法
- **多重比较**: Benjamini-Hochberg FDR校正
- **显著性水平**: α = 0.01, 0.05, 0.10
- **共线性检测**: 方差膨胀因子 (VIF)
- **稳定性验证**: 滚动窗口验证

## 数据结构规范

### 文件命名
```
A股: {SYMBOL_CODE}_1d_YYYY-MM-DD.csv
港股: {SYMBOL}_1min_YYYY-MM-DD_YYYY-MM-DD.parquet
输出: {SYMBOL}_{TIMEFRAME}_factors_YYYY-MM-DD_HH-MM-SS.parquet
```

### 支持时间周期
```
分钟级: 1min, 2min, 3min, 5min, 15min, 30min, 60min
日级别: daily
```

## 质量保证

### 代码质量
- **格式化**: Black (88字符行长度)
- **导入排序**: isort
- **类型检查**: mypy (disallow_untyped_defs)
- **测试**: pytest (95%+覆盖率)

### 性能标准
- **内存效率**: >70%
- **关键路径**: <1ms
- **向量化**: 所有向量化操作使用VectorBT
- **循环避免**: 禁用DataFrame.apply，使用内置函数

### 偏差预防
- **前瞻偏差**: 严格无前瞻计算
- **幸存者偏差**: 正确处理
- **数据对齐**: 跨时间周期正确对齐
- **真实数据**: 仅使用真实市场数据

## 集成接口

### QuantConnect MCP
```bash
./quantconnect-mcp-wrapper.sh [command]
```
- 项目管理和编译
- 回测执行和分析
- 实时算法部署
- 数据研究和因子测试

### API使用示例
```python
from factor_system.factor_engine import api
from datetime import datetime

# 单因子计算
rsi = api.calculate_single_factor(
    factor_id="RSI",
    symbol="0700.HK",
    timeframe="15min",
    start_date=datetime(2025, 9, 1),
    end_date=datetime(2025, 9, 30)
)

# 多因子计算
factors = api.calculate_factors(
    factor_ids=["RSI", "MACD", "STOCH"],
    symbols=["0700.HK", "0005.HK"],
    timeframe="15min",
    start_date=datetime(2025, 9, 1),
    end_date=datetime(2025, 9, 30)
)
```