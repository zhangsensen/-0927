# 📝 变更日志

所有重要变更都将记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

---

## [1.2.0] - 2025-10-15

### 🪓 清理
- 归档 37 个非核心脚本到 `archive/20251015_deprecated/scripts/`
- 归档 35+ 个临时文档到 `archive/20251015_deprecated/docs/`
- 移除 `production/` 重复脚本，改为引用 `scripts/`
- 清理 `__pycache__` 目录

### ✨ 新增
- 单一入口脚本：`production/run_production.sh`
- 死代码清单：`DEAD_CODE_CANDIDATES.md`
- 变更日志：`CHANGELOG.md`（本文件）

### 📝 文档
- 更新 `PRODUCTION_READY.md`（项目入口文档）
- 更新 `production/cron_daily.sh`（引用 `scripts/`）
- 更新 `production/README.md`（运维文档）

### 🎯 核心保留（8 个生产脚本）
- `scripts/produce_full_etf_panel.py` - 因子面板生产
- `scripts/pool_management.py` - 分池管理
- `scripts/etf_rotation_backtest.py` - 回测引擎
- `scripts/capacity_constraints.py` - 容量检查
- `scripts/ci_checks.py` - CI 保险丝
- `scripts/aggregate_pool_metrics.py` - 指标汇总
- `scripts/notification_handler.py` - 通知处理
- `scripts/production_pipeline.py` - 主调度

---

## [1.1.1] - 2025-10-15

### 🐛 修复
- 修复容量检查路径错误（`backtest_result_*.json` → `backtest_metrics.json`）
- 修复回测引擎组合估值错误（清算后现金 → 收盘价估值）
- 修复 Pandas FutureWarning（`resample('M')` → `resample('ME')`）
- 移除未使用变量（`portfolio_value`）
- 修复无占位符 f-string（3 处）
- 移除未使用导入（`numpy`）

### ✅ 验证
- 三池回测指标正常
- 组合指标全部达标
- CI 检查全部通过
- 容量检查正常运行
- 0 lint 警告，0 FutureWarning

---

## [1.1.0] - 2025-10-15

### 🔧 修复
- 修复调仓逻辑（先清算旧持仓 → 转为现金 → 按目标权重建仓）
- 日频权益曲线真实化（逐日标价持仓，Σ(份额 × 当日收盘价)）
- CI 检查真实化（从 `backtest_metrics.json` 读取真实指标）
- 累计成本追踪（每期累加交易成本）

### ✨ 新增
- 回测结果记录持仓快照（`positions`, `execution_prices`）
- 日频权益曲线输出（`daily_equity.parquet`）
- CI 真实指标校验（年化收益、最大回撤、夏普、月胜率）

### 📊 验证
- A_SHARE: 年化 28.71%, 回撤 -19.40%, 夏普 1.09, 月胜率 52.38%
- QDII: 年化 26.50%, 回撤 -15.71%, 夏普 1.38, 月胜率 80.95%
- OTHER: 年化 11.02%, 回撤 -10.48%, 夏普 0.68, 月胜率 66.67%
- PORTFOLIO: 年化 28.05%, 回撤 -18.29%, 夏普 1.18, 月胜率 60.95%

---

## [1.0.0] - 2025-10-15

### ✨ 新增
- 分池 E2E 隔离（A_SHARE, QDII, OTHER）
- T+1 shift 精确化（移除 `np.roll` 环回）
- CI 保险丝（8 项检查）
- 分池指标汇总（按权重合并）
- 通知与快照（钉钉/邮件）
- 配置化资金约束（`configs/etf_pools.yaml`）

### 📁 目录结构
- `scripts/` - 核心生产脚本
- `configs/` - 配置文件
- `production/` - 生产运维
- `factor_output/` - 产出与快照
- `snapshots/` - 快照目录

### 🎯 核心功能
- 因子面板生产（209 因子）
- 分池回测（真实持仓回放）
- 容量检查（ADV% 约束）
- CI 检查（8 项检查）
- 指标汇总（组合指标）
- 通知与快照（保留 10 次）

---

## 格式说明

- `Added` - 新增功能
- `Changed` - 功能变更
- `Deprecated` - 即将废弃的功能
- `Removed` - 已移除的功能
- `Fixed` - 问题修复
- `Security` - 安全相关

---

**🪓 遵循 Linus 哲学：保留核心，归档冗余，单一入口**
