# 🗑️ 死代码候选清单

**扫描日期**: 2025-10-15  
**原则**: 未被生产链路引用、功能已被替代、仅用于一次性调试

---

## 📋 候选列表

### 1. 调试与验证脚本（17 个）

| 文件 | 理由 | 替代物 | 操作 |
|------|------|--------|------|
| `scripts/alert_and_snapshot.py` | 一次性告警测试 | `production/notification_handler.py` | 归档 |
| `scripts/debug_single_factor.py` | 单因子调试工具 | 手动调试 | 归档 |
| `scripts/filter_factors_from_panel.py` | 因子过滤工具 | `produce_full_etf_panel.py` 内置 | 归档 |
| `scripts/fix_panel_mismatch.py` | 一次性修复脚本 | 已修复 | 归档 |
| `scripts/generate_correlation_heatmap.py` | 相关性热图生成 | 非生产必需 | 归档 |
| `scripts/generate_funnel_report.py` | 漏斗报告生成 | 非生产必需 | 归档 |
| `scripts/quality_check.py` | 质量检查（旧版） | `ci_checks.py` | 归档 |
| `scripts/quality_dashboard.py` | 质量仪表板 | 非生产必需 | 归档 |
| `scripts/quality_monitor.py` | 质量监控（使用 print） | `ci_checks.py` | 归档 |
| `scripts/quick_temporal_check.py` | 时序快速检查（使用 print） | `ci_checks.py` | 归档 |
| `scripts/quick_verify.py` | 快速验证 | `ci_checks.py` | 归档 |
| `scripts/regression_test.py` | 回归测试 | 非生产必需 | 归档 |
| `scripts/test_extended_scoring.py` | 评分测试 | 非生产必需 | 归档 |
| `scripts/verify_index_alignment.py` | 索引对齐验证 | `ci_checks.py` | 归档 |
| `scripts/verify_pool_separation.py` | 分池验证 | `ci_checks.py` | 归档 |
| `scripts/verify_t1_safety.py` | T+1 安全验证 | `ci_checks.py` | 归档 |
| `scripts/produce_etf_panel.py` | 旧版面板生产 | `produce_full_etf_panel.py` | 归档 |

### 2. 重复文档（35+ 个）

| 文件 | 理由 | 替代物 | 操作 |
|------|------|--------|------|
| `PRODUCTION_READY.md` | 旧版总结 | 新版 `PRODUCTION_READY.md` | 保留（更新） |
| `FINAL_PRODUCTION_REPORT.md` | 临时报告 | 合并到 `PRODUCTION_READY.md` | 归档 |
| `PRODUCTION_COMPLETE.md` | 临时报告 | 合并到 `PRODUCTION_READY.md` | 归档 |
| `FINAL_FIX_REPORT.md` | 临时报告 | 合并到 `CHANGELOG.md` | 归档 |
| `ALL_CLEAR.md` | 临时报告 | 合并到 `PRODUCTION_READY.md` | 归档 |
| 其他 30+ 临时 MD | 过程文档 | 合并或归档 | 归档 |

### 3. production/ 重复脚本

| 文件 | 理由 | 操作 |
|------|------|------|
| `production/*.py` | 与 `scripts/` 重复 | 删除，改为引用 `scripts/` |

---

## 🎯 核心保留（8 个生产脚本）

| 文件 | 功能 | 状态 |
|------|------|------|
| `scripts/produce_full_etf_panel.py` | 因子面板生产 | ✅ 保留 |
| `scripts/pool_management.py` | 分池管理 | ✅ 保留 |
| `scripts/etf_rotation_backtest.py` | 回测引擎 | ✅ 保留 |
| `scripts/capacity_constraints.py` | 容量检查 | ✅ 保留 |
| `scripts/ci_checks.py` | CI 保险丝 | ✅ 保留 |
| `scripts/aggregate_pool_metrics.py` | 指标汇总 | ✅ 保留 |
| `scripts/notification_handler.py` | 通知处理 | ✅ 保留 |
| `scripts/production_pipeline.py` | 主调度 | ✅ 保留 |

---

## 📁 目标目录结构

```
/Users/zhangshenshen/深度量化0927/
├── scripts/                           # 核心生产脚本（8 个）
│   ├── produce_full_etf_panel.py
│   ├── pool_management.py
│   ├── etf_rotation_backtest.py
│   ├── capacity_constraints.py
│   ├── ci_checks.py
│   ├── aggregate_pool_metrics.py
│   ├── notification_handler.py
│   └── production_pipeline.py
├── configs/                           # 配置文件
│   └── etf_pools.yaml
├── production/                        # 生产运维
│   ├── cron_daily.sh
│   ├── README.md
│   ├── DEPLOYMENT_SUMMARY.md
│   └── VERIFICATION_REPORT.md
├── factor_output/                     # 产出与快照
├── snapshots/                         # 快照目录
├── archive/                           # 归档目录
│   └── 20251015_deprecated/           # 本次归档
├── PRODUCTION_READY.md                # 项目入口文档
├── CHANGELOG.md                       # 变更日志
└── README.md                          # 项目说明
```

---

## 🔄 清理操作

### 归档（保留 2 周回滚窗口）

```bash
# 创建归档目录
mkdir -p archive/20251015_deprecated/{scripts,docs}

# 归档调试脚本
mv scripts/alert_and_snapshot.py archive/20251015_deprecated/scripts/
mv scripts/debug_single_factor.py archive/20251015_deprecated/scripts/
mv scripts/filter_factors_from_panel.py archive/20251015_deprecated/scripts/
mv scripts/fix_panel_mismatch.py archive/20251015_deprecated/scripts/
mv scripts/generate_correlation_heatmap.py archive/20251015_deprecated/scripts/
mv scripts/generate_funnel_report.py archive/20251015_deprecated/scripts/
mv scripts/produce_etf_panel.py archive/20251015_deprecated/scripts/
mv scripts/quality_check.py archive/20251015_deprecated/scripts/
mv scripts/quality_dashboard.py archive/20251015_deprecated/scripts/
mv scripts/quality_monitor.py archive/20251015_deprecated/scripts/
mv scripts/quick_temporal_check.py archive/20251015_deprecated/scripts/
mv scripts/quick_verify.py archive/20251015_deprecated/scripts/
mv scripts/regression_test.py archive/20251015_deprecated/scripts/
mv scripts/test_extended_scoring.py archive/20251015_deprecated/scripts/
mv scripts/verify_index_alignment.py archive/20251015_deprecated/scripts/
mv scripts/verify_pool_separation.py archive/20251015_deprecated/scripts/
mv scripts/verify_t1_safety.py archive/20251015_deprecated/scripts/

# 归档临时文档
mv FINAL_PRODUCTION_REPORT.md archive/20251015_deprecated/docs/
mv PRODUCTION_COMPLETE.md archive/20251015_deprecated/docs/
mv FINAL_FIX_REPORT.md archive/20251015_deprecated/docs/
mv ALL_CLEAR.md archive/20251015_deprecated/docs/
```

### 删除 production/ 重复脚本

```bash
# 删除重复脚本（改为引用 scripts/）
rm production/produce_full_etf_panel.py
rm production/pool_management.py
rm production/etf_rotation_backtest.py
rm production/capacity_constraints.py
rm production/ci_checks.py
rm production/aggregate_pool_metrics.py
rm production/notification_handler.py
rm production/production_pipeline.py
```

---

## ✅ 验证清单

- [ ] 归档完成（archive/20251015_deprecated/）
- [ ] 核心 8 脚本保留
- [ ] production/ 清理完成
- [ ] E2E 测试通过（pool_management → backtest → capacity → CI → aggregate）
- [ ] 文档更新完成（PRODUCTION_READY.md, CHANGELOG.md）

---

**🪓 清理原则**: 保留核心，归档冗余，单一入口
