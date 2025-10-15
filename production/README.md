# 🚀 ETF 轮动策略生产环境

## 📋 目录结构

```
production/
├── produce_full_etf_panel.py      # 因子面板生产（支持分池）
├── pool_management.py             # 分池管理主控
├── etf_rotation_backtest.py       # 回测引擎
├── capacity_constraints.py        # 容量与约束检查
├── ci_checks.py                   # CI 保险丝
├── aggregate_pool_metrics.py      # 分池指标汇总
├── notification_handler.py        # 通知处理（钉钉/邮件）
├── production_pipeline.py         # 主调度流水线
└── README.md                      # 本文档
```

---

## 🎯 核心功能

### 1. 分池生产
按 A_SHARE / QDII / OTHER 分池计算因子，避免时区/节假日错窗。

```bash
python3 production_pipeline.py
```

### 2. 回测与容量
- 日频权益曲线
- 极端月归因（top3/bottom3）
- ADV% 约束检查

### 3. CI 保险丝
8 项检查：
- T+1 shift 静态扫描
- 覆盖率骤降（≥80%）
- 有效因子数（≥8）
- 索引规范
- 零方差检查
- 元数据完整性

### 4. 通知与快照
- 失败自动通知（钉钉/邮件）
- 保留最近 10 次快照

---

## 🔧 配置

### 环境变量

```bash
# 钉钉通知
export DINGTALK_WEBHOOK="https://oapi.dingtalk.com/robot/send?access_token=YOUR_TOKEN"

# 邮件通知
export SMTP_SERVER="smtp.example.com"
export SMTP_PORT="587"
export EMAIL_SENDER="your_email@example.com"
export EMAIL_PASSWORD="your_password"
export EMAIL_RECIPIENTS="recipient1@example.com,recipient2@example.com"
```

### 资金约束配置

编辑 `configs/etf_pools.yaml`:

```yaml
capital_constraints:
  A_SHARE:
    target_capital: 7000000  # 700万
    max_single_weight: 0.25
    max_adv_pct: 0.05
  
  QDII:
    target_capital: 3000000  # 300万
    max_single_weight: 0.30
    max_adv_pct: 0.03
```

---

## 📊 运行流程

### 完整流水线

```bash
cd /path/to/repo
python3 production/production_pipeline.py
```

**执行步骤**：
1. 分池面板生产（A_SHARE, QDII, OTHER）
2. 分池回测（可选）
3. 容量检查（可选）
4. CI 检查（所有池）
5. 指标汇总（合并三池 metrics）
6. 创建快照
7. 发送通知

### 单独运行

```bash
# 仅生产面板
python3 production/pool_management.py

# 仅 CI 检查
python3 production/ci_checks.py --output-dir factor_output/etf_rotation_production/panel_A_SHARE

# 仅指标汇总
python3 production/aggregate_pool_metrics.py
```

---

## 📈 输出结构

```
factor_output/etf_rotation_production/
├── panel_A_SHARE/
│   ├── panel_FULL_*.parquet          # 因子面板
│   ├── factor_summary_*.csv          # 因子概要
│   ├── panel_meta.json               # 元数据
│   ├── production_factors.txt        # 生产因子列表
│   ├── backtest_results.parquet      # 回测结果
│   └── backtest_metrics.json         # 回测指标
├── panel_QDII/
│   └── ...
├── panel_OTHER/
│   └── ...
└── pool_metrics_summary.csv          # 分池指标汇总
```

---

## 🔔 告警规则

### CI 失败
- 覆盖率 < 70%
- 有效因子 < 8
- 零方差因子 > 10

### 容量超限
- 单只权重 > max_single_weight
- ADV% > max_adv_pct

### 回测指标
- 年化收益 < 8%
- 最大回撤 < -30%
- 夏普比率 < 0.5
- 月胜率 < 45%

---

## 🕐 定时任务

### Cron 示例

```bash
# 每日 18:00 运行
0 18 * * * /path/to/repo/production/cron_daily.sh
```

---

## 🧪 测试运行

```bash
# 测试通知
python3 production/notification_handler.py

# 测试单池生产（A_SHARE，16个ETF）
python3 production/produce_full_etf_panel.py \
  --symbols "510050.SH,510300.SH,510500.SH,159915.SZ,159949.SZ" \
  --pool-name A_SHARE \
  --output-dir factor_output/test_run
```

---

## 📝 维护日志

| 日期 | 版本 | 说明 |
|------|------|------|
| 2025-10-15 | 1.0.0 | 初始生产版本 |
| | | - 分池 E2E 隔离 |
| | | - T+1 shift 精确化 |
| | | - 日频权益曲线 |
| | | - 通知与快照 |

---

## 🆘 故障排查

### 问题：面板生产失败
- 检查数据目录：`raw/ETF/daily/*.parquet`
- 检查 symbols 白名单：`configs/etf_pools.yaml`

### 问题：CI 检查失败
- 查看具体失败项
- 检查适配器：`factor_system/factor_engine/adapters/vbt_adapter_production.py`

### 问题：通知未发送
- 检查环境变量：`echo $DINGTALK_WEBHOOK`
- 测试网络连通性

---

## 📞 联系方式

- 项目负责人：张深深
- 技术支持：[GitHub Issues](https://github.com/your-repo/issues)
