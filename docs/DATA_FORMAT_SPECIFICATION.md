# ETF 数据格式规范

**版本**: v1.1
**日期**: 2026-02-12
**适用**: v3.2+（含 v5.0，所有封板版本）

---

## 📋 核心要求

### 数据格式
- **文件格式**: Parquet (`.parquet`)
- **命名规则**: `{ts_code}_daily_{start_date}_{end_date}.parquet`
  - 示例: `510300.SH_daily_20190211_20251212.parquet`
- **存储位置**: `raw/ETF/daily/`

### 索引要求
- **必须包含**: `trade_date` 列（整数格式，YYYYMMDD）
- **索引设置**: DataLoader 会自动将 `trade_date` 设为索引

---

## 📊 必需字段（16个）

### 1. 标识字段

| 字段名 | 类型 | 说明 | 示例 |
|--------|------|------|------|
| **ts_code** | `object` | 股票代码（Wind格式）| `510300.SH` |
| **trade_date** | `int64` | 交易日期（YYYYMMDD）| `20251212` |

> ⚠️ **CRITICAL**: `trade_date` 必须是 `int64` 类型，格式为 `YYYYMMDD`（如 20251212），不是字符串！

---

### 2. 价格字段（原始价格）

| 字段名 | 类型 | 说明 | 示例 | 用途 |
|--------|------|------|------|------|
| **open** | `float64` | 开盘价（原始）| 1.000 | 参考 |
| **high** | `float64` | 最高价（原始）| 1.020 | 参考 |
| **low** | `float64` | 最低价（原始）| 0.980 | 参考 |
| **close** | `float64` | 收盘价（原始）| 1.010 | 参考 |
| **pre_close** | `object` | 前收盘价 | `None` | ⚠️ 可为空 |

---

### 3. 复权价格字段（策略使用）⭐

| 字段名 | 类型 | 说明 | 示例 | 用途 |
|--------|------|------|------|------|
| **adj_open** | `float64` | 后复权开盘价 | 1.000 | **策略计算** ✅ |
| **adj_high** | `float64` | 后复权最高价 | 1.020 | **策略计算** ✅ |
| **adj_low** | `float64` | 后复权最低价 | 0.980 | **策略计算** ✅ |
| **adj_close** | `float64` | 后复权收盘价 | 1.010 | **策略计算** ✅ |
| **adj_factor** | `float64` | 复权因子 | 1.0 | 复权计算 |

> 🎯 **CRITICAL**: 策略**必须使用**复权价格（`adj_*`），而不是原始价格！  
> 原因：ETF 分红后，原始价格会跳空，影响收益率计算。

---

### 4. 成交量与金额字段

| 字段名 | 类型 | 说明 | 示例 | 用途 |
|--------|------|------|------|------|
| **vol** | `int64` | 成交量（手）| 123456 | **策略计算** ✅ |
| **amount** | `float64` | 成交金额（万元）| 1234.56 | 参考 |

> 📝 **Note**: DataLoader 会优先使用 `vol`，如果不存在则尝试 `volume`。

---

### 5. 涨跌字段

| 字段名 | 类型 | 说明 | 示例 | 用途 |
|--------|------|------|------|------|
| **change** | `object` | 涨跌额 | `None` | ⚠️ 可为空 |
| **pct_chg** | `object` | 涨跌幅(%) | `None` | ⚠️ 可为空 |

> ⚠️ **Warning**: 这两个字段通常为空（`None`），策略不使用。

---

## 📐 完整字段列表（按顺序）

```python
[
    'ts_code',      # 1. 股票代码
    'trade_date',   # 2. 交易日期 (YYYYMMDD, int64) ⚠️
    'pre_close',    # 3. 前收盘价 (可为空)
    'open',         # 4. 开盘价（原始）
    'high',         # 5. 最高价（原始）
    'low',          # 6. 最低价（原始）
    'close',        # 7. 收盘价（原始）
    'change',       # 8. 涨跌额 (可为空)
    'pct_chg',      # 9. 涨跌幅 (可为空)
    'vol',          # 10. 成交量（手）
    'amount',       # 11. 成交金额（万元）
    'adj_factor',   # 12. 复权因子
    'adj_open',     # 13. 后复权开盘价 ⚠️
    'adj_high',     # 14. 后复权最高价 ⚠️
    'adj_low',      # 15. 后复权最低价 ⚠️
    'adj_close',    # 16. 后复权收盘价 ⚠️
]
```

---

## 🔍 DataLoader 期望格式

### 输入格式（Parquet 文件）

```python
# 示例：510300.SH_daily_20190211_20251212.parquet

ts_code     object   # '510300.SH'
trade_date  int64    # 20251212 (YYYYMMDD)
pre_close   object   # None (可为空)
open        float64  # 1.000
high        float64  # 1.020
low         float64  # 0.980
close       float64  # 1.010
change      object   # None (可为空)
pct_chg     object   # None (可为空)
vol         int64    # 123456
amount      float64  # 1234.56
adj_factor  float64  # 1.0
adj_open    float64  # 1.000
adj_high    float64  # 1.020
adj_low     float64  # 0.980
adj_close   float64  # 1.010
```

### 处理流程

```python
# DataLoader 内部处理:

# 1. 读取 Parquet 文件
df = pd.read_parquet('raw/ETF/daily/510300.SH_daily_*.parquet')

# 2. 转换 trade_date 为 datetime 索引
df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
df.set_index('trade_date', inplace=True)

# 3. 映射字段
# close  <- adj_close
# high   <- adj_high
# low    <- adj_low
# open   <- adj_open
# volume <- vol (或 volume)

# 4. 输出格式：MultiIndex DataFrame
#    索引: (date, code)
#    列: [close, high, low, open, volume]
```

---

## ✅ 验证脚本

### 验证单个文件

```python
import pandas as pd

def verify_etf_data(file_path):
    """验证 ETF 数据格式是否符合规范"""
    
    # 读取数据
    df = pd.read_parquet(file_path)
    
    # 必需字段
    required_fields = [
        'ts_code',      # 股票代码
        'trade_date',   # 交易日期
        'adj_open',     # 后复权开盘价
        'adj_high',     # 后复权最高价
        'adj_low',      # 后复权最低价
        'adj_close',    # 后复权收盘价
        'vol',          # 成交量
    ]
    
    # 检查字段
    missing = [f for f in required_fields if f not in df.columns]
    if missing:
        return False, f"❌ 缺少字段: {missing}"
    
    # 检查 trade_date 类型
    if df['trade_date'].dtype != 'int64':
        return False, f"❌ trade_date 类型错误: {df['trade_date'].dtype}，应为 int64"
    
    # 检查 trade_date 格式 (YYYYMMDD)
    sample_date = df['trade_date'].iloc[0]
    if not (19900101 <= sample_date <= 20991231):
        return False, f"❌ trade_date 格式错误: {sample_date}，应为 YYYYMMDD"
    
    # 检查复权价格非空
    for col in ['adj_open', 'adj_high', 'adj_low', 'adj_close']:
        if df[col].isnull().any():
            return False, f"❌ {col} 存在空值"
    
    # 检查成交量非空
    if df['vol'].isnull().any():
        return False, f"❌ vol 存在空值"
    
    return True, f"✅ 数据格式正确（{len(df)} 行）"

# 使用示例
result, msg = verify_etf_data('raw/ETF/daily/510300.SH_daily_20190211_20251212.parquet')
print(msg)
```

### 批量验证

```bash
cd /home/sensen/dev/projects/-0927
uv run python << 'EOF'
import pandas as pd
import os

data_dir = 'raw/ETF/daily'
files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]

print(f"📊 检查 {len(files)} 个文件...\n")

errors = []
for file in files[:5]:  # 只检查前5个
    file_path = os.path.join(data_dir, file)
    try:
        df = pd.read_parquet(file_path)
        
        # 检查必需字段
        required = ['ts_code', 'trade_date', 'adj_close', 'vol']
        missing = [f for f in required if f not in df.columns]
        
        if missing:
            errors.append(f"{file}: 缺少字段 {missing}")
        elif df['trade_date'].dtype != 'int64':
            errors.append(f"{file}: trade_date 类型错误 {df['trade_date'].dtype}")
        else:
            print(f"✅ {file}: OK ({len(df)} 行)")
    except Exception as e:
        errors.append(f"{file}: {e}")

if errors:
    print("\n❌ 错误:")
    for err in errors:
        print(f"  - {err}")
else:
    print("\n✅ 所有文件格式正确")
EOF
```

---

## 🔧 数据转换（如果需要）

### 如果你的数据是 QMT 格式（索引是 `date` 而不是 `trade_date`）

```python
import pandas as pd

def convert_qmt_to_standard(qmt_file, output_file):
    """将 QMT 格式转换为标准格式"""
    
    # 读取 QMT 数据
    df = pd.read_parquet(qmt_file)
    
    # 1. 如果索引是 date，重置为 trade_date 列
    if df.index.name == 'date':
        df = df.reset_index()
        df.rename(columns={'date': 'trade_date'}, inplace=True)
    
    # 2. 转换 trade_date 为 int (YYYYMMDD)
    if df['trade_date'].dtype != 'int64':
        df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y%m%d').astype(int)
    
    # 3. 确保有复权价格
    if 'adj_close' not in df.columns:
        # 如果没有复权价格，使用原始价格（假设没有分红）
        df['adj_open'] = df['open']
        df['adj_high'] = df['high']
        df['adj_low'] = df['low']
        df['adj_close'] = df['close']
        df['adj_factor'] = 1.0
    
    # 4. 确保有 vol 列
    if 'vol' not in df.columns and 'volume' in df.columns:
        df['vol'] = df['volume']
    
    # 5. 保存
    df.to_parquet(output_file, index=False)
    print(f"✅ 转换完成: {output_file}")
    
    return df

# 使用示例
# convert_qmt_to_standard('qmt_format.parquet', 'standard_format.parquet')
```

---

## 🚨 常见错误

### 错误 1: 缺少 `trade_date` 列
```
ValueError: 510300.SH 缺少trade_date列
```
**原因**: 数据文件没有 `trade_date` 列  
**解决**: 
1. 检查数据文件是否有 `date` 或其他日期列
2. 使用转换脚本重命名为 `trade_date`

### 错误 2: `trade_date` 类型错误
```
ValueError: cannot convert object to datetime
```
**原因**: `trade_date` 不是 `int64` 类型  
**解决**: 
```python
df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y%m%d').astype(int)
```

### 错误 3: 缺少复权价格
```
ValueError: 510300.SH 缺少adj_close列
```
**原因**: 数据文件没有 `adj_close` 等复权字段  
**解决**: 
1. 如果数据源不提供复权价格，使用原始价格代替（假设无分红）
2. 或者使用专业数据源（如 Tushare, Wind）获取复权数据

### 错误 4: 缺少成交量
```
ValueError: 510300.SH 缺少vol或volume列
```
**原因**: 数据文件没有 `vol` 或 `volume` 列  
**解决**: 
1. 检查数据文件是否有成交量字段（可能叫其他名字）
2. 重命名为 `vol`

---

## 📝 数据来源建议

### 推荐数据源

| 数据源 | 优点 | 缺点 | 是否包含复权 |
|--------|------|------|--------------|
| **Tushare** | 免费，字段完整 | 需要积分 | ✅ |
| **Wind** | 专业，准确 | 收费 | ✅ |
| **QMT** | 免费，实时 | 需要客户端 | ⚠️ 部分 |
| **AKShare** | 开源，免费 | 数据可能延迟 | ✅ |

### 本项目使用的数据源
- **QMT Bridge API** (推荐)
  - Host: `192.168.122.132:8001`
  - SDK: `qmt-data-bridge`
  - 格式: 已转换为标准格式（包含复权价格）

---

## 🎯 总结

### 关键要点

1. **必须有 `trade_date` 列**（`int64` 类型，YYYYMMDD 格式）
2. **必须有复权价格**（`adj_open`, `adj_high`, `adj_low`, `adj_close`）
3. **必须有成交量**（`vol` 或 `volume`）
4. **文件格式**: Parquet
5. **存储位置**: `raw/ETF/daily/`

### 检查清单

- [ ] 文件格式为 Parquet
- [ ] 文件名格式: `{ts_code}_daily_{start}_{end}.parquet`
- [ ] 包含 `trade_date` 列（`int64`，YYYYMMDD）
- [ ] 包含 `adj_close`, `adj_high`, `adj_low`, `adj_open`
- [ ] 包含 `vol`（或 `volume`）
- [ ] 复权价格无空值
- [ ] 成交量无空值
- [ ] 数据存储在 `raw/ETF/daily/`

---

**最后更新**: 2026-02-12
**维护者**: Quant Team
**适用版本**: v3.2 ~ v5.0+
