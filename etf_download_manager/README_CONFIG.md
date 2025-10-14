# ETF配置管理系统使用指南

## 📖 概述

ETF配置管理系统提供模块化的ETF清单管理功能，支持：
- ✅ 完整的39只ETF清单配置
- ✅ 按优先级和类别分类管理
- ✅ 便捷的ETF添加/删除操作
- ✅ 自动生成下载列表
- ✅ 配置验证和报告导出

## 🏗️ 系统架构

```
etf_download_manager/
├── config/
│   ├── etf_config.yaml          # 主配置文件（包含ETF清单）
│   ├── etf_config_manager.py    # 配置管理器核心类
│   ├── etf_config_standalone.py # 独立配置类
│   └── etf_config.py            # 配置加载工具
├── scripts/
│   └── maintain_etf_config.py   # 配置维护命令行工具
└── README_CONFIG.md            # 本使用指南
```

## 📋 当前ETF配置概览

### 总体统计
- **总ETF数量**: 39只
- **分组数量**: 6个主要分类
- **类别数量**: 12个投资类别
- **优先级分布**:
  - 最高优先级 (1): 3只
  - 高优先级 (2): 15只
  - 一般优先级 (3): 19只

### 分类结构

#### 1. 核心必配ETF (3只)
市场主要指数，投资组合核心配置
- 510300.SH - 沪深300ETF
- 510050.SH - 上证50ETF
- 159949.SZ - 创业板50ETF

#### 2. 市场规模ETF (3只)
覆盖大、中、小盘各级别市场指数
- 159915.SZ - 创业板ETF
- 510500.SH - 中证500ETF
- 512100.SZ - 中证1000ETF

#### 3. 行业主题ETF (22只)
覆盖新能源、科技、消费、医药等主要行业
- **新能源系列** (4只): 159819, 515030, 515180, 515790
- **科技系列** (6只): 512480, 516160, 159995, 159998, 515650, 516520
- **消费系列** (4只): 159928, 513500, 512690, 512720
- **医药健康** (2只): 159883, 512010
- **金融地产** (3只): 512800, 512880, 511380
- **其他行业** (3只): 515210, 518880, 159992

#### 4. 创新成长ETF (4只)
聚焦创新、成长风格的投资主题
- 159801.SZ - 创成长ETF
- 516090.SZ - 双创50ETF
- 588000.SH - 科创50ETF
- 588200.SH - 科创板ETF

#### 5. 固定收益ETF (2只)
债券类ETF，提供稳健收益选择
- 511010.SH - 国债ETF
- 511260.SH - 十年国债ETF

#### 6. 海外市场ETF (3只)
投资海外市场的ETF产品
- 159920.SZ - 恒生科技ETF
- 513100.SZ - 纳指ETF
- 513130.SZ - 恒生科技ETF港元

## 🛠️ 使用方法

### 1. 命令行工具使用

#### 查看配置摘要
```bash
cd etf_download_manager
python scripts/maintain_etf_config.py summary
```

#### 按条件筛选ETF
```bash
# 按优先级筛选
python scripts/maintain_etf_config.py list --priority 1
python scripts/maintain_etf_config.py list --priority 2

# 按类别筛选
python scripts/maintain_etf_config.py list --category "新能源"
python scripts/maintain_etf_config.py list --category "科技"

# 按分组筛选
python scripts/maintain_etf_config.py list --group sector_etfs
python scripts/maintain_etf_config.py list --group core_etfs
```

#### 生成下载列表
```bash
# 生成高优先级ETF下载列表
python scripts/maintain_etf_config.py download-list --priorities 1 2 --output high_priority.txt

# 生成所有ETF下载列表
python scripts/maintain_etf_config.py download-list --output all_etfs.txt
```

#### 添加新ETF
```bash
python scripts/maintain_etf_config.py add \
    --code "159999.SZ" \
    --name "新ETF名称" \
    --category "科技" \
    --priority 2 \
    --group sector_etfs \
    --description "ETF描述说明"
```

#### 删除ETF
```bash
python scripts/maintain_etf_config.py remove --code "159999.SZ"
```

#### 验证配置
```bash
python scripts/maintain_etf_config.py validate
```

#### 导出配置报告
```bash
python scripts/maintain_etf_config.py export --output etf_config_report.md
```

### 2. Python代码使用

#### 基本用法
```python
from config import ETFConfigManager, ETFInfo

# 加载配置管理器
config_manager = ETFConfigManager()

# 获取所有ETF
all_etfs = config_manager.get_all_etfs()
print(f"总ETF数量: {len(all_etfs)}")

# 按优先级获取ETF
priority_1_etfs = config_manager.get_etfs_by_priority(1)
print(f"最高优先级ETF: {len(priority_1_etfs)}只")

# 按类别获取ETF
tech_etfs = config_manager.get_etfs_by_category("科技")
print(f"科技类ETF: {len(tech_etfs)}只")
```

#### 添加新ETF
```python
# 创建ETF信息
new_etf = ETFInfo(
    code="159999.SZ",
    name="新科技ETF",
    category="科技",
    priority=2,
    description="跟踪科技指数"
)

# 添加到指定分组
config_manager.add_etf("sector_etfs", new_etf)

# 保存配置
config_manager.save_config()
```

#### 生成下载列表
```python
# 获取高优先级ETF下载列表
download_list = config_manager.get_download_list(priorities=[1, 2])

# 保存到文件
with open("download_list.txt", "w") as f:
    for etf_code in download_list:
        f.write(f"{etf_code}\n")
```

#### 配置验证
```python
# 验证配置
issues = config_manager.validate_config()
if issues:
    print("发现配置问题:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("配置验证通过")
```

## 📝 配置文件结构

### 主配置文件 (etf_config.yaml)
```yaml
# 基础配置
source: "tushare"
tushare_token: "your_token_here"
base_dir: "raw/ETF"
years_back: 2

# ETF清单配置
etf_list:
  core_etfs:
    name: "核心必配ETF"
    description: "市场主要指数，投资组合核心配置"
    etfs:
      - code: "510300.SH"
        name: "沪深300ETF"
        category: "市场规模"
        priority: 1
        description: "跟踪沪深300指数，A股市场核心基准"
```

### ETF信息字段说明
- **code**: ETF代码（含交易所后缀，如 510300.SH）
- **name**: ETF中文名称
- **category**: 投资类别（如 科技、新能源、消费等）
- **priority**: 下载优先级（1=最高，2=高，3=一般）
- **description**: ETF详细描述

## 🚀 最佳实践

### 1. 新增ETF的最佳流程

1. **确定分类**: 根据ETF特性选择合适的分组和类别
2. **设置合理优先级**:
   - 1: 市场核心指数、必需配置
   - 2: 重要行业ETF、高关注度
   - 3: 细分主题ETF、可选配置
3. **验证配置**: 添加后运行验证确保无冲突
4. **测试下载**: 生成下载列表并测试下载功能

### 2. 配置维护建议

- **定期验证**: 每月运行一次配置验证
- **版本控制**: 配置文件纳入版本控制管理
- **备份重要**: 重大修改前备份配置文件
- **文档同步**: 添加新ETF时同步更新相关文档

### 3. 下载策略建议

- **分批下载**: 按优先级分批下载，优先下载核心ETF
- **定期更新**: 建议每日更新最高优先级ETF数据
- **存储管理**: 定期清理过期数据，保持存储空间

## 🔧 故障排除

### 常见问题

#### 1. 配置文件加载失败
```
FileNotFoundError: 配置文件不存在
```
**解决方案**: 检查配置文件路径，确保文件存在且可读

#### 2. ETF代码重复
```
发现重复ETF代码: 510300.SH
```
**解决方案**: 运行配置验证，删除重复的ETF条目

#### 3. 优先级无效
```
优先级必须是1、2或3，当前为: 4
```
**解决方案**: 修改ETF的priority字段为有效值(1-3)

### 调试技巧

1. **启用详细输出**: 设置 `verbose: true` 查看详细日志
2. **验证配置**: 定期运行 `validate` 命令检查配置
3. **查看统计**: 使用 `summary` 命令了解配置概况
4. **导出报告**: 使用 `export` 命令生成详细报告

## 📚 扩展开发

### 添加新的ETF分组

1. 在 `etf_config.yaml` 中添加新分组
```yaml
etf_list:
  new_category_etfs:
    name: "新分类ETF"
    description: "新分类的ETF产品"
    etfs: []
```

2. 在代码中使用新分组
```python
config_manager.get_etfs_by_category_group("new_category_etfs")
```

### 自定义ETF字段

可以在ETF信息中添加自定义字段，如：
- `market`: 市场类型（A股、港股、美股）
- `expense_ratio`: 费率
- `tracking_index`: 跟踪指数
- `launch_date`: 成立日期

## 📞 技术支持

如遇到问题或需要功能增强，请：
1. 查看本指南的故障排除部分
2. 运行配置验证工具检查问题
3. 查看生成的配置报告了解详情
4. 检查日志文件获取详细错误信息

---

**版本**: 1.0
**更新时间**: 2025-10-14
**维护状态**: 活跃维护中