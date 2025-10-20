# scripts/ 目录脚本状态说明

**更新时间**: 2025-10-16 22:00  
**目的**: 明确所有脚本的使用状态，避免审查混淆  

---

## 📋 活跃脚本清单

以下脚本均为**活跃使用**，不应删除：

| 脚本 | 用途 | 状态 | 最后更新 |
|------|------|------|----------|
| `cache_cleaner.py` | 清理缓存文件 | ✅ 活跃 | 2025-10 |
| `ci_checks.py` | CI/CD质量检查 | ✅ 活跃 | 2025-10 |
| `etf_rotation_backtest.py` | ETF轮动回测 | ✅ 活跃 | 2025-10 |
| `linus_reality_check_report.py` | Linus式审查报告 | ✅ 活跃 | 2025-10 |
| `notification_handler.py` | 通知处理 | ✅ 活跃 | 2025-10 |
| `path_utils.py` | 路径工具函数 | ✅ 活跃 | 2025-10 |
| `production_cross_section_validation.py` | 横截面验证 | ✅ 活跃 | 2025-10 |
| `production_pipeline.py` | 生产流水线 | ✅ 活跃 | 2025-10 |
| `validate_candlestick_patterns.py` | K线形态验证 | ✅ 活跃 | 2025-10 |

**总计**: 9个活跃脚本

---

## 🗑️ 已删除/归档脚本

以下脚本已被删除或归档：

| 脚本 | 原用途 | 删除原因 | 替代方案 |
|------|--------|----------|----------|
| `production_full_cross_section.py` | 全量横截面生产 | 功能重复 | `etf_factor_engine_production/scripts/produce_full_etf_panel.py` |
| `capacity_constraints.py` | 容量约束 | 已集成 | 集成到回测框架 |
| `pool_management.py` | 池管理 | 已集成 | 集成到回测框架 |
| `aggregate_pool_metrics.py` | 池指标聚合 | 已集成 | 集成到回测框架 |

---

## 🔄 脚本迁移映射

### 旧版 → 新版

```
scripts/production_full_cross_section.py (已删除)
  ↓
etf_factor_engine_production/scripts/produce_full_etf_panel.py (新版)

改进:
  ✅ 配置驱动 (etf_config.yaml)
  ✅ 完整元数据生成 (MetadataGenerator)
  ✅ 更好的错误处理
```

---

## 📊 目录结构对比

### 顶层 scripts/ (工具脚本)
```
scripts/
├── cache_cleaner.py          # 缓存清理
├── ci_checks.py              # CI检查
├── etf_rotation_backtest.py  # 回测
├── production_pipeline.py    # 生产流水线
└── ... (其他工具脚本)
```

**用途**: 通用工具和辅助脚本

### etf_factor_engine_production/scripts/ (因子生产)
```
etf_factor_engine_production/scripts/
├── produce_full_etf_panel.py      # 全量因子生产
├── produce_optimized_panel.py     # 优化因子生产
├── factor_optimizer.py            # 因子优化器
├── metadata_generator.py          # 元数据生成器
└── filter_factors_from_panel.py   # 因子筛选
```

**用途**: ETF因子系统专用生产脚本

---

## ✅ 验证方法

### 检查活跃脚本
```bash
cd /Users/zhangshenshen/深度量化0927
ls -1 scripts/*.py | wc -l
# 预期输出: 9
```

### 检查已删除脚本
```bash
ls scripts/production_full_cross_section.py 2>&1
# 预期输出: No such file or directory
```

### 检查新版脚本
```bash
ls etf_factor_engine_production/scripts/produce_full_etf_panel.py
# 预期输出: 文件存在
```

---

## 🎯 使用建议

### 何时使用顶层 scripts/
- 通用工具功能（缓存清理、CI检查等）
- 跨项目的辅助脚本
- 系统级别的维护脚本

### 何时使用 etf_factor_engine_production/scripts/
- ETF因子生产相关
- 因子优化和筛选
- 元数据生成

---

## 📝 维护规则

1. **新增脚本**: 明确用途，更新本文档
2. **删除脚本**: 记录原因，提供替代方案
3. **重构脚本**: 保持向后兼容，或提供迁移指南
4. **定期审查**: 每季度检查一次，清理废弃脚本

---

**维护人**: 开发团队  
**审查周期**: 每季度  
**最后审查**: 2025-10-16  

🪓 **代码要干净、逻辑要可证、系统要能跑通**
