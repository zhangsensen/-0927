# 269因子全NaN问题 - 快速诊断指南

## 🎯 问题现状

- **已验证有效**：8个因子（覆盖率80%+）
- **待修复**：269个因子全NaN
- **架构正确**：One Pass方案、时序安全、价格口径统一

## 🔍 诊断路径（30-60分钟）

### Step 1: 单因子复现（10分钟）

选择3个代表性因子进行诊断：

```bash
# 1. 简单移动平均（TA_SMA_20）
python scripts/debug_single_factor.py \
    --factor-id TA_SMA_20 \
    --symbol 510300.SH \
    --start 20200101 \
    --end 20251014

# 2. MACD信号线（MACD_SIGNAL）
python scripts/debug_single_factor.py \
    --factor-id MACD_SIGNAL \
    --symbol 510300.SH \
    --start 20200101 \
    --end 20251014

# 3. 布林带宽度（BB_WIDTH_20）
python scripts/debug_single_factor.py \
    --factor-id BB_WIDTH_20 \
    --symbol 510300.SH \
    --start 20200101 \
    --end 20251014
```

**检查点**：
- [ ] 是否整列全NaN？
- [ ] 还是冷启动期NaN后变正常？
- [ ] 覆盖率是多少？
- [ ] min_history是否合理？

### Step 2: 批量诊断（10分钟）

批量检查10个常见因子：

```bash
python scripts/debug_single_factor.py \
    --symbol 510300.SH \
    --batch \
        TA_SMA_20 TA_EMA_20 TA_RSI_14 \
        MACD_SIGNAL BB_WIDTH_20 ATR_14 \
        STOCH_K STOCH_D CCI_20 WILLR_14
```

**检查点**：
- [ ] 哪些因子成功？
- [ ] 哪些因子失败？
- [ ] 失败原因是什么？

### Step 3: 价格字段验证（5分钟）

检查数据文件的实际列名：

```bash
python -c "
import pandas as pd
from pathlib import Path

# 读取一个ETF文件
file = list(Path('raw/ETF/daily').glob('*.parquet'))[0]
df = pd.read_parquet(file)

print(f'文件: {file.name}')
print(f'列名: {df.columns.tolist()}')
print(f'形状: {df.shape}')
print(f'前5行:\n{df.head()}')
"
```

**检查点**：
- [ ] 是否有`close`列？
- [ ] 是否有`adj_close`列？
- [ ] 是否有`trade_date`列？
- [ ] OHLCV字段是否完整？

### Step 4: 索引与对齐检查（10分钟）

运行诊断模式的全量计算（只计算前10个因子）：

```bash
# 修改代码临时限制因子数量
python scripts/produce_full_etf_panel.py \
    --start-date 20240101 \
    --end-date 20241231 \
    --diagnose
```

**检查点**：
- [ ] MultiIndex是否正确？
- [ ] groupby后索引是否保持？
- [ ] 是否有对齐错位？

### Step 5: 缓存键验证（5分钟）

检查因子计算是否使用了错误的缓存：

```python
# 临时脚本
from factor_system.factor_engine.core.registry import FactorRegistry

registry = FactorRegistry()
factor = registry.get_factor('TA_SMA_20')()

print(f"因子类: {factor.__class__.__name__}")
print(f"min_history: {getattr(factor, 'min_history', 0)}")
print(f"参数: {getattr(factor, 'params', {})}")
```

## 🔧 常见根因与修复

### 根因1: 列名不匹配

**症状**：所有因子都报错"KeyError: 'close'"

**修复**：
```python
# 在load_etf_data中统一列名
if 'trade_date' in df.columns:
    df['date'] = pd.to_datetime(df['trade_date'])
if 'adj_close' not in df.columns and 'close' in df.columns:
    df['adj_close'] = df['close']
```

### 根因2: min_history过大

**症状**：覆盖率极低（<5%），但不是全NaN

**修复**：
```python
# 检查因子定义，确保min_history合理
# 例如：20日SMA应该是21（20+1 for shift），不是252+20
class TA_SMA_20(BaseFactor):
    min_history = 21  # 不是272
```

### 根因3: 索引错位

**症状**：计算后索引变成单层或日期错乱

**修复**：
```python
# 在groupby.apply后强制对齐
factor_series = data.groupby(level='symbol', group_keys=False).apply(
    calc_with_min_history
)
# 确保索引与原始data一致
factor_series = factor_series.reindex(data.index)
```

### 根因4: 价格字段混用

**症状**：部分因子用close，部分用adj_close

**修复**：
```python
# 统一从metadata读取
price_field = self.price_field  # 'close'
# 所有因子计算前重命名
input_data = data.rename(columns={price_field: 'close'})
```

### 根因5: 异常被吞没

**症状**：summary显示success，但实际全NaN

**修复**：
```python
# 在calc_with_min_history中不要捕获所有异常
try:
    result = factor.calculate(group_data)
    return result
except KeyError as e:
    # 字段缺失，记录并返回NaN
    logger.error(f"字段缺失: {e}")
    return pd.Series(np.nan, index=group_data.index)
except Exception as e:
    # 其他异常，重新抛出以便诊断
    logger.error(f"计算异常: {e}")
    raise
```

## ✅ 快速验证（修复后必须通过）

### 验证1: 随机抽样10个因子

```bash
python scripts/debug_single_factor.py \
    --symbol 510300.SH \
    --batch \
        TA_SMA_20 TA_EMA_20 TA_RSI_14 MACD_SIGNAL BB_WIDTH_20 \
        ATR_14 STOCH_K STOCH_D CCI_20 WILLR_14
```

**期望**：
- 覆盖率 ≥ 80%
- 冷启动期合理（20-60天）
- 无零方差

### 验证2: 面板一致性

```bash
python scripts/test_one_pass_panel.py
```

**期望**：
- MultiIndex正确
- price_field='close'
- 无索引错位

### 验证3: 5年全量面板

```bash
python scripts/produce_full_etf_panel.py \
    --start-date 20200102 \
    --end-date 20251014
```

**期望**：
- 成功因子数 ≥ 200
- 覆盖率分布合理
- factor_summary_5y.csv完整

## 📊 诊断检查清单

### 数据层
- [ ] ETF文件存在且可读
- [ ] 列名包含：trade_date, open, high, low, close, volume
- [ ] 日期格式正确（YYYYMMDD或datetime）
- [ ] 无大量缺失值（>20%）

### 因子层
- [ ] 所有因子已注册
- [ ] min_history设置合理
- [ ] 无循环依赖
- [ ] calculate方法返回Series

### 计算层
- [ ] MultiIndex保持(symbol, date)
- [ ] groupby不改变索引结构
- [ ] 价格字段统一
- [ ] 异常被正确捕获和记录

### 输出层
- [ ] panel形状正确
- [ ] 无全NaN列（除资金流类）
- [ ] coverage分布合理
- [ ] summary记录完整

## 🚀 建议执行顺序

1. **立即执行**（10分钟）：
   ```bash
   # 单因子诊断
   python scripts/debug_single_factor.py --factor-id TA_SMA_20 --symbol 510300.SH
   ```

2. **快速定位**（20分钟）：
   ```bash
   # 批量诊断
   python scripts/debug_single_factor.py --symbol 510300.SH --batch TA_SMA_20 TA_EMA_20 MACD_SIGNAL
   
   # 检查数据列名
   python -c "import pandas as pd; print(pd.read_parquet('raw/ETF/daily/510300.SH_daily_qfq.parquet').columns)"
   ```

3. **修复验证**（30分钟）：
   - 根据诊断结果修复代码
   - 运行验证脚本
   - 生成5年全量面板

## 📝 诊断日志模板

```
日期: 2025-01-15
诊断人: [您的名字]

### 问题描述
269个因子全NaN

### 诊断结果
1. 单因子测试:
   - TA_SMA_20: [成功/失败] 覆盖率: [X%]
   - MACD_SIGNAL: [成功/失败] 覆盖率: [X%]
   
2. 数据检查:
   - 列名: [列出实际列名]
   - 是否有close: [是/否]
   
3. 根因分析:
   - [具体根因]

### 修复方案
1. [修复步骤1]
2. [修复步骤2]

### 验证结果
- 单因子: [通过/失败]
- 批量: [通过/失败]
- 全量: [通过/失败]
```

---

**最后更新**: 2025-01-15  
**状态**: 🔍 诊断中
