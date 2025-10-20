# 废弃脚本归档

本目录包含已废弃的旧版脚本，仅供历史参考。

## 废弃原因

这些脚本已被 `etf_factor_engine_production/` 下的新版本替代：

### 已归档脚本

| 旧脚本 | 替代方案 | 废弃原因 |
|--------|----------|----------|
| `production_full_cross_section.py` | `etf_factor_engine_production/scripts/produce_full_etf_panel.py` | 功能重复，新版本支持配置驱动 |
| `capacity_constraints.py` | 已集成到新版回测框架 | 独立脚本已无必要 |
| `pool_management.py` | 已集成到新版回测框架 | 独立脚本已无必要 |
| `aggregate_pool_metrics.py` | 已集成到新版回测框架 | 独立脚本已无必要 |

## 迁移指南

如果你之前使用这些脚本，请参考以下迁移路径：

### `production_full_cross_section.py` → `produce_full_etf_panel.py`

**旧用法**:
```bash
python scripts/production_full_cross_section.py --start-date 20200102
```

**新用法**:
```bash
python etf_factor_engine_production/scripts/produce_full_etf_panel.py --start-date 20200102
```

**改进**:
- ✅ 配置驱动（`etf_config.yaml`）
- ✅ 完整元数据生成
- ✅ 更好的错误处理

---

**归档时间**: 2025-10-16  
**维护状态**: 不再维护
