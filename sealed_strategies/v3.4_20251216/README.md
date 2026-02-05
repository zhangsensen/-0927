# v3.4 Production Strategy - README

**Status**: 🔒 **SEALED** (2025-12-16)  
**Version**: v3.4_20251216  
**Strategies**: 2 (震荡市精选双策略)

---

## 📌 What is This?

本封板版本包含**基于最近 60 天交易记录深度分析**精选出的**震荡市最优双策略**，专为 2025Q4 震荡磨底环境优化。

### 核心策略
1. **Strategy #1** (4因子): `ADX_14D + OBV_SLOPE_10D + SHARPE_RATIO_20D + SLOPE_20D`
2. **Strategy #2** (5因子): `ADX_14D + OBV_SLOPE_10D + PRICE_POSITION_120D + SHARPE_RATIO_20D + SLOPE_20D`

### 为什么是这两个？
- **近 60 天收益**: -0.23%（其他策略 -5% ~ -9%）
- **成功避开军工坑** (2025-10-20)：其他策略踩坑亏 1.1%，本策略买纳指赚 6.5%
- **避免假突破陷阱**: 简洁因子组合不会在黄金震荡中反复止损

---

## 🚀 Quick Start (5分钟上手)

### 1. 环境准备
```bash
cd /home/sensen/dev/projects/-0927/sealed_strategies/v3.4_20251216/locked
uv sync --dev
```

### 2. 验证封板完整性
```bash
sha256sum -c ../CHECKSUMS.sha256
```

### 3. 查看策略详情
```bash
cat ../artifacts/production_candidates.csv
```

### 4. 运行今日信号（模拟）
```bash
# 确保数据是最新的
uv run python scripts/update_daily_from_qmt_bridge.py --all

# Strategy #1 信号
uv run python scripts/generate_today_signal.py \
  --combo "ADX_14D + OBV_SLOPE_10D + SHARPE_RATIO_20D + SLOPE_20D" \
  --output signals/strategy1_$(date +%Y%m%d).json

# Strategy #2 信号
uv run python scripts/generate_today_signal.py \
  --combo "ADX_14D + OBV_SLOPE_10D + PRICE_POSITION_120D + SHARPE_RATIO_20D + SLOPE_20D" \
  --output signals/strategy2_$(date +%Y%m%d).json
```

### 5. 审计回测（可选）
```bash
uv run python scripts/batch_bt_backtest.py \
  --candidates ../artifacts/production_candidates.csv \
  --output results/bt_audit_$(date +%Y%m%d).parquet
```

---

## 📁 Directory Structure

```
v3.4_20251216/
├── CHECKSUMS.sha256            # 所有文件校验和（防篡改）
├── MANIFEST.json               # 封板清单与元数据
├── README.md                   # 本文件
├── REPRODUCE.md                # 详细复现步骤
├── RELEASE_NOTES.md            # 版本变更记录
│
├── artifacts/                  # 生产制品
│   ├── production_candidates.csv       # 2 策略完整指标
│   ├── PRODUCTION_REPORT.md            # 详细报告
│   ├── QUICK_REFERENCE.md              # 快速参考
│   └── DEPLOYMENT_GUIDE.md             # 部署指南
│
└── locked/                     # 锁定代码（不可变）
    ├── configs/                # 配置文件
    │   ├── combo_wfo_config.yaml
    │   ├── etf_pools.yaml
    │   └── etf_config.yaml
    ├── scripts/                # 核心脚本
    │   ├── batch_bt_backtest.py
    │   ├── generate_today_signal.py
    │   └── update_daily_from_qmt_bridge.py
    ├── src/                    # 源码模块
    │   ├── etf_strategy/
    │   └── etf_data/
    ├── pyproject.toml          # 依赖定义
    └── Makefile                # 常用命令
```

---

## 🎯 Core Performance

| Metric | Strategy #1 | Strategy #2 | Average |
|:-------|:------------|:------------|:--------|
| **Total Return** | 136.52% | 129.85% | **133.19%** |
| **Sharpe Ratio** | 1.03 | 1.04 | **1.04** |
| **Max Drawdown** | 15.47% | 13.93% | **14.70%** |
| **Win Rate** | **50.79%** ✅ | **49.42%** ✅ | **50.11%** |
| **Profit Factor** | **2.11** ✅ | **2.11** ✅ | **2.11** |
| **Trades (5.95y)** | 252 | 257 | **~255** |
| **Avg Hold** | 9.2 days | 9.0 days | **9.1 days** |
| **Recent 63d** | -0.67% | -0.67% | **-0.67%** |
| **Recent 120d** | +6.78% | +6.78% | **+6.78%** |

**关键指标解读**：
- ✅ **胜率 ~50%**：健康水平！不过拟合（>60%）也不靠运气（<40%）
- ✅ **盈亏比 2.11**：每赚 ¥2.11，只亏 ¥1（平均盈利 = 亏损 × 2.1）
- ✅ **期望收益**：50% × 2.11 - 50% × 1 = **+0.55** (正期望！)

**Backtest Period**: 2020-01-01 至 2025-12-12 (5.95 年)  
**Rebalance**: 每 3 交易日  
**Position**: 2 只 ETF

---

## ⚠️ Risk Warnings

1. **高度同质化**: 两策略因子重合 80%，持仓重合 >80%，不是真实分散
2. **QDII 依赖**: 近期 40% 交易为海外科技 ETF（513100/513500），美股暴跌会同步重创
3. **震荡市优化**: 在强趋势市（如 2020H2）可能跑输单策略
4. **短周期高频**: 平均持有 9 天，交易成本敏感

---

## 📖 Documentation

| 文档 | 说明 |
|:-----|:-----|
| **REPRODUCE.md** | 从零复现完整回测（数据下载 → 信号生成 → 审计） |
| **PRODUCTION_REPORT.md** | 详细性能报告 + 交易分析 + 风控建议 |
| **TRADING_STATISTICS.md** | **交易统计详细报告**（胜率/盈亏比/分段表现） ⭐ |
| **HOLDOUT_VALIDATION_REPORT.md** | **过拟合风险评估**（5/5 满分 ✅）⭐ |
| **QUICK_REFERENCE.md** | 因子速查 + 持仓特征 + 监控指标 |
| **DEPLOYMENT_GUIDE.md** | 生产部署流程 + 熔断机制 + 再平衡规则 |
| **RELEASE_NOTES.md** | v3.4 变更记录 + 与 v3.3 对比 |

---

## 🛠️ Common Tasks

### 查看最新持仓
```bash
uv run python -c "
from etf_strategy.core.data_loader import load_factor_data
df = load_factor_data()
print(df[df.index.get_level_values('date') == df.index.get_level_values('date').max()])
"
```

### 测试单策略
```bash
uv run python scripts/batch_bt_backtest.py \
  --combo "ADX_14D + OBV_SLOPE_10D + SHARPE_RATIO_20D + SLOPE_20D" \
  --start 2024-01-01 \
  --end 2025-12-12
```

### 更新数据
```bash
# 增量更新（推荐）
uv run python scripts/update_daily_from_qmt_bridge.py --all

# 验证数据完整性
uv run python scripts/verify_data_integrity.py
```

---

## 🔍 Key Insights (最近 60 天交易分析)

### 2025-10-20: 避开军工坑 ✅
- ❌ **拖油瓶策略**: 买 512400 (军工) @ 1.671 → 止损 @ 1.652 (-1.1%)
- ✅ **本策略**: 买 513100 (纳指) @ 1.807 → 卖出 @ 1.924 (+6.5%)
- **收益差**: 7.6pp

### 2025-11-10: 抄底国内大盘
- 买入 511010 (上证50) + 511260 (科创50)
- 震荡持平，12-01 止损退出

### 2025-12-01: 轮动黄金/有色
- 买入 518880 (黄金) + 515180 (有色)
- 持有至 12-12，小幅盈利 +0.5%

### 2025-12-12: 当前持仓
- 159949 (创业板)
- 159915 (科技ETF)
- **方向**: 国内科技成长

---

## 📞 Support & Feedback

**问题反馈**:
- 检查 `locked/src/etf_strategy/` 源码注释
- 运行 `uv run pytest tests/ -v` 验证环境

**性能监控**:
- 日频监控: 见 `artifacts/DEPLOYMENT_GUIDE.md`
- 周度审计: 持仓重合度 + QDII占比

**紧急熔断**:
- 单日跌 > 2%: 次日减仓 30%
- 连续 3 天跌 > 1%: 暂停开新仓

---

## 🎓 Design Philosophy

本策略遵循以下原则：
1. **数据驱动**: 基于实际交易记录选策略，不靠拍脑袋
2. **简洁优先**: 4-5 因子组合，避免过拟合
3. **可复现**: 锁定代码 + 配置 + 校验和，任何人可独立验证
4. **风险透明**: 明确告知同质化风险 + QDII 依赖

---

**Last Updated**: 2025-12-16 16:00 CST  
**Checksum**: See `CHECKSUMS.sha256`  
**Reproducibility**: ✅ **100% (Configs + Scripts + Src Locked)**
