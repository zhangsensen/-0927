# ETF下载管理器

统一的ETF数据下载管理系统，消除重复代码，提供简洁高效的ETF数据下载解决方案。

## 🚀 特性

- **统一管理**: 整合所有ETF下载功能到一个统一的管理器
- **消除重复**: 去除原有脚本中的重复代码和逻辑
- **灵活配置**: 支持多种配置模式和自定义配置
- **模块化设计**: 清晰的模块分离，易于维护和扩展
- **完整的错误处理**: 健壮的错误处理和重试机制
- **数据完整性**: 自动验证下载数据的完整性
- **进度跟踪**: 详细的下载进度和统计信息

## 📁 目录结构

```
etf_download_manager/
├── core/                   # 核心模块
│   ├── models.py          # 数据模型
│   ├── config.py          # 配置管理
│   ├── downloader.py      # 下载器核心
│   ├── data_manager.py    # 数据管理器
│   └── etf_list.py        # ETF清单管理
├── scripts/               # 脚本工具
│   ├── download_etf_manager.py  # 主下载脚本
│   ├── quick_download.py         # 快速下载脚本
│   └── batch_download.py         # 批量下载脚本
├── config/                # 配置文件
│   ├── etf_config.yaml           # 默认配置
│   ├── quick_config.yaml         # 快速配置
│   ├── full_config.yaml          # 完整配置
│   └── etf_config.py             # 配置管理工具
├── docs/                  # 文档
│   ├── README.md                  # 本文档
│   ├── usage.md                   # 使用指南
│   └── api.md                     # API文档
└── tests/                 # 测试文件
```

## 🛠️ 安装和设置

### 1. 环境要求

- Python 3.8+
- tushare
- pandas
- pyarrow (用于parquet格式)
- PyYAML (用于配置文件)

### 2. 安装依赖

```bash
pip install tushare pandas pyarrow PyYAML
```

### 3. 设置Tushare Token

```bash
export TUSHARE_TOKEN="your_tushare_token_here"
```

或者在配置文件中指定Token。

## 📖 快速开始

### 方式1: 快速下载（推荐新手）

最简单的ETF数据下载方式：

```bash
cd 深度量化0927
python etf_download_manager/scripts/quick_download.py
```

这将下载所有核心ETF（必配ETF）的最近2年日线数据。

### 方式2: 使用主下载脚本

功能完整的下载脚本：

```bash
# 查看ETF清单
python etf_download_manager/scripts/download_etf_manager.py --action list

# 下载核心ETF
python etf_download_manager/scripts/download_etf_manager.py --action download-core

# 按优先级下载
python etf_download_manager/scripts/download_etf_manager.py --action download-priority --priority high

# 下载指定ETF
python etf_download_manager/scripts/download_etf_manager.py --action download-specific --etf-codes 510300 510500

# 更新单个ETF
python etf_download_manager/scripts/download_etf_manager.py --action update --etf-code 510300

# 验证数据完整性
python etf_download_manager/scripts/download_etf_manager.py --action validate

# 使用自定义配置
python etf_download_manager/scripts/download_etf_manager.py --config custom --config-file my_config.yaml
```

### 方式3: 交互式批量下载

```bash
python etf_download_manager/scripts/batch_download.py
```

提供交互式菜单选择下载方式。

## ⚙️ 配置管理

### 预设配置

系统提供三种预设配置：

1. **default**: 标准配置，适合日常使用
2. **quick**: 快速配置，适合快速下载
3. **full**: 完整配置，适合完整数据下载

```bash
# 使用快速配置
python etf_download_manager/scripts/download_etf_manager.py --config quick

# 使用完整配置
python etf_download_manager/scripts/download_etf_manager.py --config full
```

### 自定义配置

创建自己的配置文件：

```yaml
# my_config.yaml
source: "tushare"
tushare_token: "${TUSHARE_TOKEN}"
base_dir: "raw/ETF"
years_back: 2
download_types:
  - "daily"
save_format: "parquet"
batch_size: 20
```

然后使用：

```bash
python etf_download_manager/scripts/download_etf_manager.py --config custom --config-file my_config.yaml
```

## 📊 数据格式

### 目录结构

```
raw/ETF/
├── daily/                  # 日线数据
│   ├── 510300_daily_20230101_20231231.parquet
│   └── 510500_daily_20230101_20231231.parquet
├── moneyflow/             # 资金流向数据（如果可用）
│   ├── 510300_moneyflow_20230101_20231231.parquet
│   └── ...
├── minutes/               # 分钟数据（如果启用）
│   ├── 510300/
│   │   ├── 510300_20231201_1min.parquet
│   │   └── ...
│   └── ...
├── basic/                 # 基础信息
│   ├── etf_basic_info_20231201.parquet
│   └── etf_basic_latest.parquet
└── summary/               # 下载摘要
    └── download_summary_20231201_120000.json
```

### 数据字段

**日线数据字段**:
- `trade_date`: 交易日期
- `open`: 开盘价
- `high`: 最高价
- `low`: 最低价
- `close`: 收盘价
- `pre_close`: 前收盘价
- `change`: 涨跌额
- `pct_chg`: 涨跌幅
- `vol`: 成交量（手）
- `amount`: 成交额（千元）

## 🔧 高级用法

### 编程方式使用

```python
from etf_download_manager import ETFDownloadManager, ETFConfig, ETFListManager

# 创建配置
config = ETFConfig(
    tushare_token="your_token",
    years_back=2,
    download_types=["daily"]
)

# 创建下载器
downloader = ETFDownloadManager(config)

# 获取ETF清单
list_manager = ETFListManager()
core_etfs = list_manager.get_must_have_etfs()

# 下载数据
stats = downloader.download_multiple_etfs(core_etfs)
print(f"下载完成: 成功 {stats.success_count}, 失败 {stats.failed_count}")
```

### 数据验证

```python
# 验证下载数据的完整性
validation_results = downloader.validate_downloaded_data(core_etfs)

for etf_code, result in validation_results.items():
    if result['overall_valid']:
        print(f"✅ {etf_code}: 数据完整")
    else:
        print(f"❌ {etf_code}: 数据有问题")
```

### 更新数据

```python
# 更新最近30天的数据
result = downloader.update_etf_data(etf_info, days_back=30)
```

## 📈 性能优化

1. **合理设置延迟**: 根据API限制调整`request_delay`
2. **批处理大小**: 根据网络情况调整`batch_size`
3. **选择数据格式**: parquet格式比CSV更高效
4. **重试机制**: 根据网络稳定性调整`max_retries`

## 🚨 注意事项

1. **API限制**: Tushare有API调用频率限制，请合理设置延迟
2. **Token权限**: 某些数据可能需要更高级别的Tushare权限
3. **ETF资金流向**: Tushare标准接口不提供ETF资金流向数据
4. **存储空间**: 大量ETF数据可能占用较多存储空间
5. **网络稳定**: 下载过程需要稳定的网络连接

## 🐛 故障排除

### 常见问题

1. **Token错误**: 检查Tushare Token是否正确设置
2. **网络超时**: 增加超时时间和重试次数
3. **权限不足**: 升级Tushare账户权限
4. **磁盘空间**: 确保有足够的存储空间

### 日志文件

下载过程中的详细信息会记录在`etf_download.log`文件中。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 📄 许可证

本项目遵循MIT许可证。