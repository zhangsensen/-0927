# ETF下载管理器使用指南

本指南详细介绍ETF下载管理器的各种使用方式和配置选项。

## 📋 目录

1. [环境设置](#环境设置)
2. [基础使用](#基础使用)
3. [高级配置](#高级配置)
4. [编程接口](#编程接口)
5. [数据管理](#数据管理)
6. [故障排除](#故障排除)

## 🔧 环境设置

### 1. 安装依赖

```bash
pip install tushare pandas pyarrow PyYAML
```

### 2. 设置Token

```bash
# 方式1: 环境变量（推荐）
export TUSHARE_TOKEN="your_token_here"

# 方式2: 在配置文件中指定
# 编辑 config/etf_config.yaml
# tushare_token: "your_token_here"
```

### 3. 验证环境

```bash
cd 深度量化0927
python -c "
from etf_download_manager.config import setup_environment
setup_environment()
"
```

## 🚀 基础使用

### 快速开始

#### 方式1: 一键快速下载

最简单的方式，下载所有核心ETF：

```bash
python etf_download_manager/scripts/quick_download.py
```

#### 方式2: 使用主管理器

功能完整的管理器脚本：

```bash
# 显示ETF清单和统计
python etf_download_manager/scripts/download_etf_manager.py --action summary

# 列出所有ETF（显示前50个）
python etf_download_manager/scripts/download_etf_manager.py --action list

# 按分类列出ETF
python etf_download_manager/scripts/download_etf_manager.py --action list --category "宽基指数"

# 下载核心ETF（必配和核心级别）
python etf_download_manager/scripts/download_etf_manager.py --action download-core

# 下载高优先级ETF（核心、必配、高级别）
python etf_download_manager/scripts/download_etf_manager.py --action download-priority --priority high

# 下载指定ETF
python etf_download_manager/scripts/download_etf_manager.py --action download-specific --etf-codes 510300 510500 159915
```

### 交互式下载

```bash
python etf_download_manager/scripts/batch_download.py
```

提供菜单选择：
1. 下载所有ETF
2. 按分类下载
3. 下载高优先级ETF

## ⚙️ 高级配置

### 配置类型

系统提供三种预设配置：

#### 1. 默认配置（default）
- 下载最近2年数据
- 标准API设置
- 适合日常使用

#### 2. 快速配置（quick）
- 下载最近1年数据
- 较快的API设置
- 适合快速获取数据

#### 3. 完整配置（full）
- 下载最近3年数据
- 稳定的API设置
- 包含多种数据类型

### 使用不同配置

```bash
# 使用快速配置（下载1年数据）
python etf_download_manager/scripts/download_etf_manager.py --config quick --action download-core

# 使用完整配置（下载3年数据）
python etf_download_manager/scripts/download_etf_manager.py --config full --action download-core
```

### 自定义配置

#### 创建配置文件

```yaml
# my_config.yaml
source: "tushare"
tushare_token: "${TUSHARE_TOKEN}"

base_dir: "raw/ETF"
years_back: 2

max_retries: 3
retry_delay: 1.0
request_delay: 0.2

download_types:
  - "daily"

save_format: "parquet"
batch_size: 20
verbose: true
```

#### 使用自定义配置

```bash
python etf_download_manager/scripts/download_etf_manager.py \
  --config custom \
  --config-file my_config.yaml \
  --action download-core
```

### 配置参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `source` | string | "tushare" | 数据源 |
| `tushare_token` | string | "" | Tushare API Token |
| `base_dir` | string | "raw/ETF" | 数据存储目录 |
| `years_back` | int | 2 | 下载最近几年的数据 |
| `max_retries` | int | 3 | 最大重试次数 |
| `retry_delay` | float | 1.0 | 重试间隔（秒） |
| `request_delay` | float | 0.2 | API请求间隔（秒） |
| `batch_size` | int | 50 | 批处理大小 |
| `download_types` | list | ["daily"] | 下载的数据类型 |
| `save_format` | string | "parquet" | 数据保存格式 |
| `verbose` | bool | true | 是否显示详细信息 |

## 💻 编程接口

### 基础用法

```python
from etf_download_manager import ETFDownloadManager, ETFConfig, ETFListManager

# 1. 创建配置
config = ETFConfig(
    tushare_token="your_token",
    years_back=2,
    download_types=["daily"]
)

# 2. 创建下载器
downloader = ETFDownloadManager(config)

# 3. 获取ETF清单
list_manager = ETFListManager()
all_etfs = list_manager.get_all_etfs()
core_etfs = list_manager.get_must_have_etfs()

# 4. 下载数据
stats = downloader.download_multiple_etfs(core_etfs)
print(f"成功: {stats.success_count}, 失败: {stats.failed_count}")
```

### 高级用法

```python
from etf_download_manager import ETFDownloadType, ETFPriority

# 创建完整配置
config = ETFConfig(
    tushare_token="your_token",
    years_back=3,
    download_types=[ETFDownloadType.DAILY, ETFDownloadType.MONEYFLOW],
    save_format="parquet",
    batch_size=20,
    request_delay=0.3
)

# 创建下载器
downloader = ETFDownloadManager(config)

# 按优先级筛选ETF
list_manager = ETFListManager()
high_priority_etfs = list_manager.filter_etfs(
    priorities=[ETFPriority.CORE, ETFPriority.MUST_HAVE, ETFPriority.HIGH]
)

# 批量下载
stats = downloader.download_multiple_etfs(high_priority_etfs)

# 获取下载摘要
summary = stats.get_summary()
print(f"下载统计: {summary}")
```

### 数据管理

```python
# 加载已下载的数据
etf_info = list_manager.get_etf_by_code("510300")
daily_data = downloader.data_manager.load_daily_data(etf_info)

# 验证数据完整性
validation_result = downloader.data_manager.validate_data_integrity(etf_info)
print(f"数据完整性: {validation_result}")

# 获取数据摘要
data_summary = downloader.data_manager.get_data_summary()
print(f"数据摘要: {data_summary}")
```

## 📊 数据管理

### 数据目录结构

```
raw/ETF/
├── daily/              # 日线数据
├── moneyflow/          # 资金流向数据
├── minutes/            # 分钟数据
├── basic/              # 基础信息
└── summary/            # 下载摘要
```

### 数据验证

```bash
# 验证所有ETF数据完整性
python etf_download_manager/scripts/download_etf_manager.py --action validate
```

### 数据更新

```bash
# 更新单个ETF的最近30天数据
python etf_download_manager/scripts/download_etf_manager.py \
  --action update \
  --etf-code 510300 \
  --days-back 30
```

### 编程方式更新

```python
# 更新指定ETF
etf_info = list_manager.get_etf_by_code("510300")
result = downloader.update_etf_data(etf_info, days_back=30)

if result.success:
    print(f"更新成功: 日线数据 {result.daily_records} 条")
else:
    print(f"更新失败: {result.error_message}")
```

## 🔍 ETF筛选

### 按优先级筛选

```python
# 获取不同优先级的ETF
core_etfs = list_manager.filter_etfs(priorities=[ETFPriority.CORE])
must_have_etfs = list_manager.filter_etfs(priorities=[ETFPriority.MUST_HAVE])
high_priority_etfs = list_manager.filter_etfs(
    priorities=[ETFPriority.CORE, ETFPriority.MUST_HAVE, ETFPriority.HIGH]
)
```

### 按分类筛选

```python
# 获取特定分类的ETF
tech_etfs = list_manager.get_etfs_by_category("科技半导体")
finance_etfs = list_manager.get_etfs_by_category("金融")
```

### 按交易所筛选

```python
from etf_download_manager import ETFExchange

# 获取上交所ETF
sh_etfs = list_manager.filter_etfs(exchanges=[ETFExchange.SH])

# 获取深交所ETF
sz_etfs = list_manager.filter_etfs(exchanges=[ETFExchange.SZ])
```

### 复合筛选

```python
# 组合筛选条件
filtered_etfs = list_manager.filter_etfs(
    priorities=[ETFPriority.MUST_HAVE, ETFPriority.HIGH],
    categories=["科技半导体", "新能源"],
    exclude_codes=["510300"]  # 排除特定ETF
)
```

## 📈 性能优化

### API调用优化

```python
# 网络较好时的配置
fast_config = ETFConfig(
    request_delay=0.1,      # 减少延迟
    max_retries=2,          # 减少重试
    batch_size=50           # 增大批次
)

# 网络较差时的配置
stable_config = ETFConfig(
    request_delay=0.5,      # 增加延迟
    max_retries=5,          # 增加重试
    retry_delay=2.0,        # 增加重试间隔
    batch_size=10           # 减小批次
)
```

### 存储优化

```python
# 使用parquet格式（推荐）
config = ETFConfig(save_format="parquet")

# 使用CSV格式
config = ETFConfig(save_format="csv")
```

### 内存优化

```python
# 分批下载大量ETF
large_etf_list = list_manager.get_all_etfs()
batch_size = 20

for i in range(0, len(large_etf_list), batch_size):
    batch = large_etf_list[i:i + batch_size]
    stats = downloader.download_multiple_etfs(batch)
    print(f"批次 {i//batch_size + 1} 完成")
```

## 🚨 故障排除

### 常见错误及解决方案

#### 1. Token错误

```
错误: Tushare Token未设置
```

**解决方案**:
```bash
export TUSHARE_TOKEN="your_token_here"
```

#### 2. 网络超时

```
错误: API请求超时
```

**解决方案**:
```python
config = ETFConfig(
    timeout=60,           # 增加超时时间
    max_retries=5,        # 增加重试次数
    retry_delay=2.0       # 增加重试间隔
)
```

#### 3. 权限不足

```
错误: 获取ETF基础信息失败
```

**解决方案**: 升级Tushare账户权限。

#### 4. 磁盘空间不足

```
错误: No space left on device
```

**解决方案**: 清理磁盘空间或更改数据目录。

### 调试模式

```bash
# 启用详细日志
python etf_download_manager/scripts/download_etf_manager.py --verbose --action download-core
```

### 查看日志

```bash
# 查看下载日志
tail -f etf_download.log

# 查看最近的错误
grep "ERROR" etf_download.log | tail -10
```

## 📝 最佳实践

1. **设置合理的延迟**: 根据API限制调整`request_delay`
2. **使用适当的批次大小**: 平衡性能和稳定性
3. **定期验证数据**: 使用验证功能确保数据完整性
4. **备份配置文件**: 保存自定义配置文件
5. **监控日志文件**: 及时发现和解决问题
6. **选择合适的数据格式**: 推荐使用parquet格式
7. **合理规划存储空间**: 大量ETF数据需要足够空间

## 🔗 相关链接

- [Tushare官网](https://tushare.pro/)
- [PyArrow文档](https://arrow.apache.org/docs/python/)
- [Pandas文档](https://pandas.pydata.org/docs/)