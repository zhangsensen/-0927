# ETF 数据增量更新使用指南

## 功能特点

✅ **基于 QMT Bridge SDK** - 无需手动处理HTTP请求和参数  
✅ **增量更新** - 自动检测本地数据，只拉取新增部分  
✅ **批量处理** - 支持同时更新多个ETF  
✅ **数据去重** - 自动处理重复数据  
✅ **格式标准** - Parquet格式存储，便于数据分析  

## 快速开始

### 1. 确保已安装SDK

```bash
# 在虚拟环境中安装
pip install qmt-data-bridge
```

### 2. 更新单个ETF

```bash
python scripts/update_daily_from_qmt_bridge.py --symbols 510300
```

### 3. 更新多个ETF（逗号分隔）

```bash
python scripts/update_daily_from_qmt_bridge.py --symbols 510300,510500,512690
```

### 4. 使用配置文件批量更新

```bash
# 编辑 etf_list.json 添加你的ETF列表
python scripts/update_daily_from_qmt_bridge.py --config etf_list.json
```

### 5. 查看已下载的数据

```bash
# 列出所有已下载的ETF
python scripts/view_data.py

# 查看具体某个ETF的数据
python scripts/view_data.py 510300
```

## 命令行参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--symbols` | 指定ETF代码（逗号分隔） | `--symbols 510300,510500` |
| `--config` | 使用配置文件 | `--config etf_list.json` |
| `--all` | 更新配置文件中的所有ETF | `--all` |
| `--host` | QMT Bridge服务器地址 | `--host 192.168.122.132` |
| `--port` | QMT Bridge服务器端口 | `--port 8001` |
| `--data-dir` | 数据存储目录 | `--data-dir ./data/etf_daily` |
| `--exchange` | 交易所代码 (SH/SZ) | `--exchange SH` |
| `--force-days` | 强制获取最近N天 | `--force-days 30` |

## 使用场景

### 场景1: 每日增量更新

```bash
# 增量更新配置文件中的所有ETF（只拉取新数据）
python scripts/update_daily_from_qmt_bridge.py --all
```

**自动处理**：
- 检测本地最后日期：2025-12-12
- 只拉取 2025-12-13 及之后的数据
- 合并去重后保存

### 场景2: 首次下载或全量更新

```bash
# 强制获取最近30天数据（覆盖现有数据）
python scripts/update_daily_from_qmt_bridge.py --symbols 510300 --force-days 30
```

### 场景3: 添加新ETF到监控列表

```bash
# 1. 编辑 etf_list.json，添加新代码
# 2. 运行更新
python scripts/update_daily_from_qmt_bridge.py --all
```

### 场景4: 自定义QMT服务器

```bash
python scripts/update_daily_from_qmt_bridge.py \
    --symbols 510300 \
    --host 192.168.1.100 \
    --port 9000
```

## 配置文件格式

### etf_list.json

```json
{
  "symbols": [
    "510300",
    "510500",
    "512690"
  ],
  "description": "ETF监控列表"
}
```

### 或使用纯文本格式（etf_list.txt）

```
510300
510500
512690
```

## 数据格式

### 存储格式
- **文件名**: `{symbol}.parquet` (如 `510300.parquet`)
- **目录**: `./data/etf_daily/`

### 数据字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `trade_date` | int | 交易日期 (YYYYMMDD) |
| `open` | float | 开盘价 |
| `high` | float | 最高价 |
| `low` | float | 最低价 |
| `close` | float | 收盘价 |
| `volume` | float | 成交量 |
| `amount` | float | 成交额 |

### 读取数据示例

```python
import pandas as pd

# 读取单个ETF数据
df = pd.read_parquet("data/etf_daily/510300.parquet")

# 查看最近5天
print(df.tail(5))

# 计算日收益率
df['return'] = df['close'].pct_change()

# 过滤特定日期范围
df_2024 = df[df['trade_date'] >= 20240101]
```

## 定时任务设置

### Linux/Mac (crontab)

```bash
# 编辑定时任务
crontab -e

# 每个交易日 15:30 自动更新
30 15 * * 1-5 cd /home/sensen/dev/projects/-0927 && .venv/bin/python scripts/update_daily_from_qmt_bridge.py --all >> logs/etf_update.log 2>&1
```

### Windows (任务计划程序)

创建任务：
- 触发器：每天 15:30
- 操作：运行程序
  - 程序：`python.exe`
  - 参数：`scripts/update_daily_from_qmt_bridge.py --all`
  - 起始于：`C:\projects\-0927`

## 故障排查

### 1. SDK导入失败

```
❌ 请先安装 qmt-data-bridge:
   pip install qmt-data-bridge
```

**解决方法**：
```bash
pip install qmt-data-bridge
```

### 2. 无法连接QMT Bridge

```
❌ 510300.SH 获取数据失败: Connection refused
```

**检查项**：
- QMT Bridge服务是否运行？
- 服务器地址/端口是否正确？
- 网络是否可达？

```bash
# 测试连接
curl http://192.168.122.132:8001/api/v1/health
```

### 3. 数据为空

```
⚠️  510300.SH - 无新数据
```

**原因**：本地数据已是最新，无需更新

### 4. 深交所ETF

对于深交所ETF（如 159开头），需指定交易所：

```bash
python scripts/update_daily_from_qmt_bridge.py --symbols 159915 --exchange SZ
```

## 性能优化

### 1. 批量更新时控制速度

脚本已内置500ms延迟，避免请求过快。如需调整：

```python
# 在 update_batch 方法中修改
await asyncio.sleep(0.5)  # 改为其他值
```

### 2. 并行下载（高级用法）

如果ETF数量很多（>50个），可以使用并行下载：

```python
# 修改 update_batch 方法，使用 asyncio.gather
tasks = [self.update_symbol(s, exchange, force_days) for s in symbols]
results = await asyncio.gather(*tasks, return_exceptions=True)
```

## 与其他工具集成

### 1. 导出为CSV

```python
import pandas as pd

df = pd.read_parquet("data/etf_daily/510300.parquet")
df.to_csv("510300.csv", index=False)
```

### 2. 导入数据库

```python
import pandas as pd
from sqlalchemy import create_engine

df = pd.read_parquet("data/etf_daily/510300.parquet")
engine = create_engine("sqlite:///etf_data.db")
df.to_sql("etf_510300", engine, if_exists="replace", index=False)
```

### 3. 数据分析示例

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_parquet("data/etf_daily/510300.parquet")
df['date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')

# 绘制K线图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# 价格
ax1.plot(df['date'], df['close'], label='收盘价')
ax1.set_ylabel('价格')
ax1.legend()
ax1.grid(True)

# 成交量
ax2.bar(df['date'], df['volume'], alpha=0.5)
ax2.set_ylabel('成交量')
ax2.grid(True)

plt.tight_layout()
plt.show()
```

## 常见ETF代码

### 宽基指数

| 代码 | 名称 | 交易所 |
|------|------|--------|
| 510300 | 沪深300ETF | 上交所 |
| 510500 | 中证500ETF | 上交所 |
| 159915 | 创业板ETF | 深交所 |
| 159919 | 沪深300ETF | 深交所 |
| 512690 | 酒ETF | 上交所 |

### 行业主题

| 代码 | 名称 | 交易所 |
|------|------|--------|
| 512880 | 证券ETF | 上交所 |
| 512980 | 传媒ETF | 上交所 |
| 513050 | 中概互联 | 上交所 |
| 515000 | 科技ETF | 上交所 |
| 516160 | 新能源ETF | 上交所 |

## 技术支持

问题反馈：查看 QMT Bridge 项目文档

---

**提示**：此脚本完全基于 QMT Bridge SDK，无需处理底层HTTP请求细节。SDK会自动处理所有参数转换、错误重试等逻辑。
