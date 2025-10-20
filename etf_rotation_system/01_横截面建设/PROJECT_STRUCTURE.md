# ETF因子面板项目结构

## 📁 目录结构

```
/Users/zhangshenshen/深度量化0927/etf_rotation_system/01_横截面建设/
├── 📄 QUICK_REFERENCE.md           # 快速参考指南
├── 📄 README.md                   # 项目说明文档
├── 📁 __pycache__/                 # Python缓存文件
├── 📁 config/                      # 配置文件目录
│   ├── 📄 factor_panel_config.yaml  # 主配置文件
│   └── 📄 config_classes.py          # 配置类定义
├── 📁 deprecated/                  # 遗留代码目录
│   ├── 📄 README.md                 # 遗留代码说明
│   ├── 📄 generate_panel.py        # 原始版本(已废弃)
│   └── 📄 generate_panel_original.py # 原始备份(已废弃)
├── 📁 docs/                        # 文档目录
│   └── 📄 configuration_guide.md   # 详细配置指南
├── 📁 etf_rotation_system/         # 输出目录
│   └── 📁 data/results/panels/     # 面板结果目录
├── 📁 examples/                    # 示例目录
│   ├── 📄 basic_usage.md          # 基本使用示例
│   ├── 📄 config_modification.md  # 配置修改示例
│   ├── 📄 custom_config.md       # 自定义配置示例
│   └── 📄 migration_commands.md  # 迁移命令示例
├── 📄 generate_panel_refactored.py # 重构版本主程序 ✅
├── 📄 migrate_to_config.py          # 迁移工具
└── 📄 test_equivalence.py          # 功能等价性测试
```

## 🚀 核心文件说明

### 主要程序
- **`generate_panel_refactored.py`**: 配置驱动的ETF因子面板生成程序
- **`config/factor_panel_config.yaml`**: 完整的配置文件
- **`config/config_classes.py`**: 类型安全的配置类定义

### 工具文件
- **`migrate_to_config.py`**: 从原版本迁移到配置版本的辅助工具
- **`test_equivalence.py`**: 验证重构版本与原版本功能等价性的测试套件

### 文档文件
- **`QUICK_REFERENCE.md`**: 快速参考手册
- **`docs/configuration_guide.md`**: 详细的配置使用指南
- **`examples/`**: 各种使用场景的示例文件

### 遗留代码
- **`deprecated/`**: 包含原始版本和备份文件，仅用于参考对比

## ✅ 当前使用的文件

要运行ETF因子面板生成，请使用：

```bash
# 使用配置文件运行
python generate_panel_refactored.py --config config/factor_panel_config.yaml

# 或使用默认配置
python generate_panel_refactored.py
```

## 🗑️ 不再使用的文件

以下文件已移动到 `deprecated/` 目录：
- `generate_panel.py` (原版本)
- `generate_panel_original.py` (备份文件)

这些文件仅用于参考和对比，不建议在生产环境中使用。

## 📁 输出目录

执行结果保存在：
```
/Users/zhangshenshen/深度量化0927/etf_rotation_system/01_横截面建设/etf_rotation_system/data/results/panels/
└── panel_YYYYMMDD_HHMMSS/    # 时间戳文件夹
    ├── panel.parquet         # 因子数据
    ├── metadata.json         # 元数据
    └── execution_log.txt     # 执行日志
```

---

**维护者**: Claude Code  
**更新时间**: 2025-10-20  
**版本**: 1.0
