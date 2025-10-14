# Scripts 目录说明

## 📁 目录结构

```
scripts/
├── production_run.py          # 🚀 主生产脚本（唯一入口）
├── tools/                     # 🛠️ 开发工具
│   └── audit_indicator_coverage.py
└── archive/                   # 📦 历史脚本（已废弃）
    ├── migrate_factor_ids.py
    ├── migrate_parquet_schema.py
    ├── old_generate_moneyflow_factors.py
    ├── old_produce_money_flow_factors.py
    ├── old_produce_money_flow_factors_v2.py
    ├── old_run_money_flow_only.py
    └── run_factor_production_simple.py
```

## 🚀 生产脚本

### `production_run.py` - A股因子生产主脚本

**唯一的生产环境入口**，功能完整、配置驱动。

#### 核心特性
- ✅ **配置驱动**: 通过 `factor_system/config/money_flow_config.yaml` 配置
- ✅ **多时间框架**: 支持 daily, 240min, 120min, 60min, 30min, 15min, 5min, 1min
- ✅ **A股会话感知**: 严格遵守 9:30-11:30, 13:00-15:00 交易时间
- ✅ **因子覆盖**: 150+技术指标 + 11个资金流因子
- ✅ **质量校验**: 自动验证每日K线数，生成质量报告
- ✅ **独立存储**: 每个股票独立保存 parquet 文件

#### 使用方式

```bash
# 基本用法（使用配置文件）
python scripts/production_run.py

# 指定因子集
python scripts/production_run.py --set all
python scripts/production_run.py --set technical_only
```

#### 配置文件
- 路径: `factor_system/config/money_flow_config.yaml`
- 配置项: 股票列表、时间范围、因子集、数据路径等

#### 输出
- 数据文件: `factor_system/factor_output/production/{timeframe}/{symbol}_{timeframe}_{timestamp}.parquet`
- 质量报告: `factor_system/factor_output/production/{timeframe}/report_{timestamp}.md`

## 🛠️ 开发工具

### `tools/audit_indicator_coverage.py` - 指标覆盖率审计

审计 VectorBT 可用指标与当前引擎实际使用的指标对比。

```bash
python scripts/tools/audit_indicator_coverage.py
```

## 📦 历史脚本 (archive/)

**已废弃的脚本**，保留用于历史参考，不应在生产环境使用。

### 迁移脚本（一次性使用）
- `migrate_factor_ids.py` - 因子ID标准化迁移
- `migrate_parquet_schema.py` - Parquet Schema统一迁移

### 旧版生产脚本（已被 production_run.py 替代）
- `run_factor_production_simple.py` - 简化版生产脚本（硬编码配置）
- `old_generate_moneyflow_factors.py` - 旧版资金流因子生成
- `old_produce_money_flow_factors.py` - 旧版资金流因子生产 v1
- `old_produce_money_flow_factors_v2.py` - 旧版资金流因子生产 v2
- `old_run_money_flow_only.py` - 旧版纯资金流因子脚本

## 🧪 测试脚本

测试和验证脚本已移至 `tests/` 目录：

- `tests/verify_intraday_resample_cn.py` - A股重采样验证
- `tests/verify_t_plus_1.py` - T+1时序验证
- `tests/test_session_resample.py` - 会话感知重采样回归测试
- `tests/development/` - 开发阶段的集成测试

## 📋 最佳实践

### 生产环境
1. **只使用** `production_run.py`
2. 通过 YAML 配置文件修改参数
3. 定期检查质量报告

### 开发调试
1. 使用 `tests/` 目录下的验证脚本
2. 新增验证脚本应放在 `tests/development/`
3. 工具类脚本放在 `scripts/tools/`

### 代码维护
1. 废弃的脚本移到 `scripts/archive/`
2. 保持 `scripts/` 根目录整洁
3. 更新本 README 文档

## 🎯 快速开始

```bash
# 1. 修改配置文件
vim factor_system/config/money_flow_config.yaml

# 2. 运行生产脚本
python scripts/production_run.py

# 3. 查看输出
ls -lh factor_system/factor_output/production/60min/

# 4. 查看质量报告
cat factor_system/factor_output/production/60min/report_*.md
```

## 📞 联系与支持

如有问题，请查阅：
- 项目文档: `docs/`
- 配置说明: `factor_system/config/money_flow_config.yaml`
- 测试用例: `tests/`

---

**最后更新**: 2025-10-14  
**维护者**: 量化首席工程师
