# 269因子全NaN - 快速修复清单

## ⚡ 5分钟快速诊断

### 立即执行

```bash
# 1. 快速测试（5分钟）
python scripts/quick_factor_test.py
```

**这个脚本会告诉你**：
- ✅ 数据文件是否可读
- ✅ 列名是否正确
- ✅ 因子注册表是否正常
- ✅ 基础因子能否计算

### 根据结果采取行动

#### 场景A: 数据加载失败

**症状**：`test_data_loading` 失败

**修复**：
```bash
# 检查数据目录
ls -lh raw/ETF/daily/

# 查看文件内容
python -c "
import pandas as pd
df = pd.read_parquet('raw/ETF/daily/510300.SH_daily_qfq.parquet')
print('列名:', df.columns.tolist())
print('形状:', df.shape)
print(df.head())
"
```

#### 场景B: 因子注册表失败

**症状**：`test_factor_registry` 失败

**修复**：
```bash
# 检查因子引擎安装
python -c "from factor_system.factor_engine.core.registry import FactorRegistry; print('OK')"

# 重新安装
cd factor_system/factor_engine
pip install -e .
```

#### 场景C: 单因子计算失败

**症状**：`test_single_factor` 失败

**修复**：使用详细诊断
```bash
python scripts/debug_single_factor.py \
    --factor-id TA_SMA_20 \
    --symbol 510300.SH
```

---

## 🎯 30分钟完整诊断

### Step 1: 数据验证（5分钟）

```bash
# 检查ETF数据
python -c "
import pandas as pd
from pathlib import Path

files = list(Path('raw/ETF/daily').glob('*.parquet'))
print(f'文件数: {len(files)}')

# 检查第一个文件
df = pd.read_parquet(files[0])
print(f'列名: {df.columns.tolist()}')
print(f'必需字段检查:')
print(f'  close: {\"close\" in df.columns}')
print(f'  adj_close: {\"adj_close\" in df.columns}')
print(f'  trade_date: {\"trade_date\" in df.columns}')
print(f'  OHLCV: {all(c in df.columns for c in [\"open\", \"high\", \"low\", \"volume\"])}')
"
```

**期望输出**：
- 文件数 > 0
- 必需字段全部为True

### Step 2: 单因子诊断（10分钟）

```bash
# 测试3个代表性因子
python scripts/debug_single_factor.py --factor-id TA_SMA_20 --symbol 510300.SH
python scripts/debug_single_factor.py --factor-id MACD_SIGNAL --symbol 510300.SH
python scripts/debug_single_factor.py --factor-id BB_WIDTH_20 --symbol 510300.SH
```

**检查输出**：
- 覆盖率应该 > 80%
- 冷启动期应该 < 60天
- 不应该整列全NaN

### Step 3: 批量诊断（10分钟）

```bash
# 批量测试10个因子
python scripts/debug_single_factor.py \
    --symbol 510300.SH \
    --batch \
        TA_SMA_20 TA_EMA_20 TA_RSI_14 \
        MACD_SIGNAL BB_WIDTH_20 ATR_14 \
        STOCH_K STOCH_D CCI_20 WILLR_14
```

**检查输出**：
- 查看 `factor_output/debug/batch_summary.csv`
- 成功率应该 > 80%

### Step 4: 诊断模式全量计算（5分钟）

```bash
# 小范围测试（1年数据）
python scripts/produce_full_etf_panel.py \
    --start-date 20240101 \
    --end-date 20241231 \
    --diagnose
```

**检查输出**：
- 每个因子的详细计算信息
- 覆盖率分布
- 错误信息

---

## 🔧 常见问题快速修复

### 问题1: KeyError: 'close'

**原因**：数据文件没有close列，只有adj_close

**修复**：在 `produce_full_etf_panel.py` 的 `load_etf_data` 中添加：
```python
# 统一价格字段
if 'close' not in data.columns and 'adj_close' in data.columns:
    data['close'] = data['adj_close']
    self.price_field = 'adj_close'
elif 'close' in data.columns:
    self.price_field = 'close'
```

### 问题2: KeyError: 'date'

**原因**：数据文件使用trade_date而非date

**修复**：已在您的代码中修复
```python
data['date'] = pd.to_datetime(data['trade_date']).dt.normalize()
```

### 问题3: 整列全NaN但无报错

**原因**：min_history过大或索引错位

**诊断**：
```python
# 检查min_history
from factor_system.factor_engine.core.registry import FactorRegistry
registry = FactorRegistry()
factor = registry.get_factor('TA_SMA_20')()
print(f"min_history: {getattr(factor, 'min_history', 0)}")
```

**修复**：
- 如果min_history > 252，检查因子定义
- 如果索引错位，在groupby后添加 `factor_series = factor_series.reindex(data.index)`

### 问题4: 覆盖率极低（<10%）

**原因**：数据质量问题或窗口过长

**诊断**：
```bash
python scripts/debug_single_factor.py --factor-id [FACTOR_ID] --symbol 510300.SH
```

**修复**：
- 检查数据完整性
- 调整min_history
- 检查是否有大量NaN

---

## ✅ 验证清单

修复后必须通过以下验证：

### 基础验证
```bash
# 1. 快速测试
python scripts/quick_factor_test.py
# 期望：所有测试通过

# 2. 单因子验证
python scripts/debug_single_factor.py --factor-id TA_SMA_20 --symbol 510300.SH
# 期望：覆盖率 > 80%

# 3. 批量验证
python scripts/debug_single_factor.py --symbol 510300.SH --batch TA_SMA_20 TA_EMA_20 TA_RSI_14
# 期望：成功率 > 80%
```

### 全量验证
```bash
# 4. 1年全量面板
python scripts/produce_full_etf_panel.py --start-date 20240101 --end-date 20241231

# 5. 检查结果
python scripts/test_one_pass_panel.py
# 期望：面板结构正确，成功因子 > 200
```

### 生产验证
```bash
# 6. 5年全量面板
python scripts/produce_full_etf_panel.py --start-date 20200102 --end-date 20251014

# 7. 筛选高质量因子
python scripts/filter_factors_from_panel.py \
    --panel-file factor_output/etf_rotation/panel_FULL_20200102_20251014.parquet \
    --summary-file factor_output/etf_rotation/factor_summary_20200102_20251014.csv \
    --mode production

# 期望：筛选出 > 150 个高质量因子
```

---

## 📊 成功标准

### 最低标准（可接受）
- ✅ 快速测试全部通过
- ✅ 单因子覆盖率 > 80%
- ✅ 批量成功率 > 70%
- ✅ 全量面板成功因子 > 150

### 理想标准（生产就绪）
- ✅ 快速测试全部通过
- ✅ 单因子覆盖率 > 90%
- ✅ 批量成功率 > 90%
- ✅ 全量面板成功因子 > 200
- ✅ 无零方差因子（除常量类）
- ✅ 无重复因子组

---

## 🚀 执行顺序（推荐）

### 第一轮：快速定位（10分钟）
```bash
# 1. 快速测试
python scripts/quick_factor_test.py

# 2. 如果失败，查看具体错误并修复
```

### 第二轮：详细诊断（20分钟）
```bash
# 3. 单因子诊断
python scripts/debug_single_factor.py --factor-id TA_SMA_20 --symbol 510300.SH

# 4. 批量诊断
python scripts/debug_single_factor.py --symbol 510300.SH --batch TA_SMA_20 TA_EMA_20 TA_RSI_14

# 5. 根据诊断结果修复代码
```

### 第三轮：全量验证（30分钟）
```bash
# 6. 小范围测试
python scripts/produce_full_etf_panel.py --start-date 20240101 --end-date 20241231

# 7. 如果成功，运行5年全量
python scripts/produce_full_etf_panel.py --start-date 20200102 --end-date 20251014

# 8. 筛选高质量因子
python scripts/filter_factors_from_panel.py --mode production
```

---

## 📞 快速命令参考

```bash
# 快速测试（5分钟）
python scripts/quick_factor_test.py

# 单因子诊断
python scripts/debug_single_factor.py --factor-id TA_SMA_20 --symbol 510300.SH

# 批量诊断
python scripts/debug_single_factor.py --symbol 510300.SH --batch TA_SMA_20 TA_EMA_20 TA_RSI_14

# 诊断模式全量计算
python scripts/produce_full_etf_panel.py --start-date 20240101 --end-date 20241231 --diagnose

# 正常模式全量计算
python scripts/produce_full_etf_panel.py --start-date 20200102 --end-date 20251014

# 筛选因子
python scripts/filter_factors_from_panel.py --mode production
```

---

**最后更新**: 2025-01-15  
**预计修复时间**: 30-60分钟  
**状态**: 🔧 待执行
