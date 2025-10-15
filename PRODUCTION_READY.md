# 🎯 ETF 轮动策略 - 生产环境文档

**项目**: FactorEngine - ETF 轮动策略  
**状态**: ✅ 生产就绪  
**版本**: v1.2.0  
**日期**: 2025-10-15

---

## 🚀 快速启动

### 一键运行完整流水线

```bash
cd <PROJECT_ROOT>
bash production/run_production.sh
```

或直接调用：

```bash
cd <PROJECT_ROOT>
python3 scripts/production_pipeline.py
```

---

## 📁 目录结构

```
<PROJECT_ROOT>/
├── scripts/                           # 核心生产脚本（8 个）
│   ├── produce_full_etf_panel.py      # 因子面板生产
│   ├── pool_management.py             # 分池管理
│   ├── etf_rotation_backtest.py       # 回测引擎
│   ├── capacity_constraints.py        # 容量检查
│   ├── ci_checks.py                   # CI 保险丝
│   ├── aggregate_pool_metrics.py      # 指标汇总
│   ├── notification_handler.py        # 通知处理
│   └── production_pipeline.py         # 主调度（单一入口）
├── configs/                           # 配置文件
│   └── etf_pools.yaml                 # 分池配置、资金约束、ETF分类
├── production/                        # 生产运维
│   ├── run_production.sh              # 统一入口脚本
│   ├── cron_daily.sh                  # 定时任务
│   ├── README.md                      # 运维文档
│   ├── DEPLOYMENT_SUMMARY.md          # 部署总结
│   └── VERIFICATION_REPORT.md         # 验证报告
├── factor_output/                     # 产出与快照
│   └── etf_rotation_production/
│       ├── panel_A_SHARE/
│       ├── panel_QDII/
│       ├── panel_OTHER/
│       └── pool_metrics_summary.csv
├── snapshots/                         # 快照目录
├── archive/                           # 归档目录
│   └── 20251015_deprecated/           # 本次归档
├── PRODUCTION_READY.md                # 本文档（项目入口）
├── CHANGELOG.md                       # 变更日志
├── DEAD_CODE_CANDIDATES.md            # 死代码清单
└── README.md                          # 项目说明
```

---

## ✅ 核心功能

### 1. 分池 E2E 隔离
- **A_SHARE**: 16 个 A 股 ETF，209 因子，覆盖率 90.8%
- **QDII**: 4 个 QDII ETF，209 因子，覆盖率 90.8%
- **OTHER**: 23 个其他 ETF，209 因子，覆盖率 90.8%

### 2. CI 保险丝（8 项检查）
- ✅ T+1 shift 静态扫描
- ✅ 覆盖率骤降检查（≥80%）
- ✅ 有效因子数检查（≥8）
- ✅ 回测指标阈值检查（真实数据）
- ✅ 索引规范检查
- ✅ 零方差检查
- ✅ 元数据完整性

### 3. 分池指标汇总
- 合并三池回测指标
- 按权重计算组合指标（A_SHARE 70%, QDII 30%）
- CI 阈值自动校验

### 4. 容量检查
- ADV% 约束检查（可配置阈值）
- 持仓权重约束
- 从真实回测结果解析末期持仓

### 5. 通知与告警
- 钉钉 Webhook 通知（可选）
- 邮件备用通知（可选）
- 快照管理（保留最近 10 次）

### 6. 配置化资金约束
```yaml
capital_constraints:
  A_SHARE:
    target_capital: 7000000
    max_single_weight: 0.25
    max_adv_pct: 0.05
  QDII:
    target_capital: 3000000
    max_single_weight: 0.30
    max_adv_pct: 0.03
```

---

## 📊 生产指标（真实数据）

### 分池回测

| 池 | 年化收益 | 最大回撤 | 夏普比率 | 月胜率 | CI 状态 |
|----|----------|----------|----------|--------|---------|
| A_SHARE | 28.71% | -19.40% | 1.09 | 52.38% | ✅ |
| QDII | 26.50% | -15.71% | 1.38 | 80.95% | ✅ |
| OTHER | 11.02% | -10.48% | 0.68 | 66.67% | ✅ |

### 组合指标（加权）

| 指标 | 实际值 | 阈值 | 状态 |
|------|--------|------|------|
| 年化收益 | 28.05% | ≥8% | ✅ |
| 最大回撤 | -18.29% | ≥-30% | ✅ |
| 夏普比率 | 1.18 | ≥0.5 | ✅ |
| 月胜率 | 60.95% | ≥45% | ✅ |
| 年化换手 | 0.02 | ≤10.0 | ✅ |

---

## 🔧 环境配置

### 必需环境变量（可选）

```bash
# 钉钉通知
export DINGTALK_WEBHOOK="https://oapi.dingtalk.com/robot/send?access_token=YOUR_TOKEN"

# 邮件通知
export SMTP_SERVER="smtp.example.com"
export SMTP_PORT="587"
export EMAIL_SENDER="your_email@example.com"
export EMAIL_PASSWORD="your_password"
export EMAIL_RECIPIENTS="recipient1@example.com"
```

### 定时任务

```bash
# 编辑 crontab
crontab -e

# 添加每日 18:00 运行（替换 <PROJECT_ROOT> 为实际路径）
0 18 * * * /bin/bash -lc 'cd <PROJECT_ROOT> && bash production/cron_daily.sh'
```

---

## 📋 每日巡检清单

### 1. 面板质量检查

```bash
# 检查三池面板覆盖率
python3 scripts/ci_checks.py --output-dir factor_output/etf_rotation_production/panel_A_SHARE
python3 scripts/ci_checks.py --output-dir factor_output/etf_rotation_production/panel_QDII
python3 scripts/ci_checks.py --output-dir factor_output/etf_rotation_production/panel_OTHER
```

**预期**:
- 覆盖率 ≥ 80%
- 有效因子数 ≥ 8
- 零方差因子数 = 0

### 2. 回测指标检查

```bash
# 查看组合指标
cat factor_output/etf_rotation_production/pool_metrics_summary.csv
```

**预期**:
- 年化收益 ≥ 8%
- 最大回撤 ≥ -30%
- 夏普比率 ≥ 0.5
- 月胜率 ≥ 45%

### 3. 容量报告检查

```bash
# 查看容量违规
cat factor_output/etf_rotation_production/panel_A_SHARE/capacity_constraints_report.json
```

**预期**:
- ADV% 超限数量 ≤ 5
- 单只权重超限数量 = 0

### 4. 快照检查

```bash
# 查看最近快照
ls -lt snapshots/ | head -5
```

**预期**:
- 保留最近 10 次快照
- 快照包含完整配置与指标

---

## 🔄 版本管理与回滚

### 快照结构

```
snapshots/snapshot_production_YYYYMMDD_HHMMSS/
├── configs/                           # 配置快照
├── factor_output/                     # 产出快照
├── backtest_metrics.json              # 回测指标
└── snapshot_meta.json                 # 快照元数据
```

### 回滚方案

```bash
# 1. 查看可用快照
ls -lt snapshots/

# 2. 回滚到指定快照
SNAPSHOT_ID="snapshot_production_20251015_163143"
cp -r snapshots/$SNAPSHOT_ID/configs/* configs/
cp -r snapshots/$SNAPSHOT_ID/factor_output/* factor_output/

# 3. 验证回滚
python3 scripts/ci_checks.py --output-dir factor_output/etf_rotation_production/panel_A_SHARE
```

---

## ⚠️ 常见排错

### 1. 回测指标异常

**症状**: 年化收益 < 0% 或 > 100%

**排查**:
```bash
# 检查日频权益曲线
python3 -c "import pandas as pd; df = pd.read_parquet('factor_output/etf_rotation_production/panel_A_SHARE/daily_equity.parquet'); print(df.head()); print(df.tail())"
```

**解决**: 检查持仓份额与价格数据

### 2. 容量检查失败

**症状**: "未找到回测结果，跳过容量检查"

**排查**:
```bash
# 检查回测结果文件
ls factor_output/etf_rotation_production/panel_A_SHARE/backtest_*.{json,parquet}
```

**解决**: 确保回测已运行并生成 `backtest_metrics.json`

### 3. CI 检查失败

**症状**: "❌ 年化收益未达标"

**排查**:
```bash
# 查看回测指标
cat factor_output/etf_rotation_production/panel_A_SHARE/backtest_metrics.json
```

**解决**: 
- 检查因子质量
- 调整 CI 阈值（`--min-annual-return`）
- 优化因子筛选

### 4. numba 缓存报错

**症状**: "numba cache error"

**解决**:
```bash
# 跳过面板重算，直接运行回测
python3 scripts/etf_rotation_backtest.py \
  --panel-file factor_output/etf_rotation_production/panel_A_SHARE/panel_FULL_*.parquet \
  --production-factors factor_output/etf_rotation_production/panel_A_SHARE/production_factors.txt \
  --price-dir raw/ETF/daily \
  --output-dir factor_output/etf_rotation_production/panel_A_SHARE
```

---

## 🎓 技术亮点

### Linus 哲学实践

1. **消灭特殊情况**: 配置化替代硬编码
2. **API 稳定性**: 参数化路径，向后兼容
3. **简洁即武器**: 单一职责，模块化
4. **代码即真理**: CI 自动验证，快照可追溯
5. **无冗余代码**: 归档死代码，保留核心

### 量化工程纪律

1. **T+1 强制**: 精确控制 NaN，移除环回
2. **分池隔离**: 避免时区/节假日错窗
3. **真实回测**: 逐日标价持仓，真实权益曲线
4. **CI 保险丝**: 8 项检查，真实指标校验
5. **容量约束**: ADV% 检查，发现超限违规
6. **单一入口**: `scripts/production_pipeline.py` 统一调度

---

## 📞 联系方式

- **项目负责人**: 张深深
- **部署日期**: 2025-10-15
- **版本**: v1.2.0
- **状态**: ✅ 生产就绪

---

## 🔄 版本历史

### v1.2.0 (2025-10-15) - 代码清理与结构化
- ✅ 归档 37 个非核心脚本
- ✅ 归档 35+ 个临时文档
- ✅ production/ 引用 scripts/，移除重复
- ✅ 单一入口：`scripts/production_pipeline.py`
- ✅ 更新文档：PRODUCTION_READY.md, CHANGELOG.md

### v1.1.1 (2025-10-15) - 异常修复版
- ✅ 修复容量检查路径错误
- ✅ 修复回测引擎组合估值错误
- ✅ 修复 Pandas FutureWarning
- ✅ 移除未使用变量与导入

### v1.1.0 (2025-10-15) - 全面真实化
- ✅ 修复调仓逻辑（先清算后建仓）
- ✅ 日频权益曲线真实化（逐日标价持仓）
- ✅ CI 检查真实化（读取真实指标）

### v1.0.0 (2025-10-15) - 初始生产版本
- ✅ 分池 E2E 隔离
- ✅ T+1 shift 精确化
- ✅ CI 保险丝（8 项检查）

---

**🚀 系统已就绪，可投入生产！**
