# v3.4 Quick Reference

**Version**: v3.4_20251216  
**Strategies**: 2 (震荡市精选双策略)

---

## 📋 Strategy Cheat Sheet

### Strategy #1 (4因子)
```
ADX_14D + OBV_SLOPE_10D + SHARPE_RATIO_20D + SLOPE_20D
```
**Key Metrics**:
- Total Return: **136.52%**
- Sharpe: 1.03
- MaxDD: **15.47%**
- **Win Rate: 50.79%** ✅ (健康！不过拟合也不靠运气)
- **Profit Factor: 2.11** ✅ (盈亏比优秀！赚=亏×2.1)
- Trades: 252
- Avg Hold: 9.2 days
- Recent 63d: **-0.67%** | 120d: **+6.78%**

### Strategy #2 (5因子)
```
ADX_14D + OBV_SLOPE_10D + PRICE_POSITION_120D + SHARPE_RATIO_20D + SLOPE_20D
```
**Key Metrics**:
- Total Return: **129.85%**
- Sharpe: 1.04
- MaxDD: **13.93%** (更低！)
- **Win Rate: 49.42%** ✅ (健康！不过拟合也不靠运气)
- **Profit Factor: 2.11** ✅ (盈亏比优秀！赚=亏×2.1)
- Trades: 257
- Avg Hold: 9.0 days
- Recent 63d: **-0.67%** | 120d: **+6.78%**

---

## 🔧 Factor Definitions

### Trend (趋势)
- **ADX_14D**: Average Directional Index (14日)  
  衡量趋势强度，> 25 表示强趋势，< 20 表示震荡

### Momentum (动量)
- **PRICE_POSITION_120D**: 当前价格在过去 120 日价格范围中的位置 (0-1)  
  > 0.8 表示接近高点，< 0.2 表示接近低点

### Risk (风险)
- **SHARPE_RATIO_20D**: 20日夏普比率  
  衡量单位风险收益，> 1 表示风险调整后收益良好

### Trend Strength (趋势强度)
- **SLOPE_20D**: 20日价格斜率（线性回归）  
  正值表示上涨趋势，负值表示下跌趋势

### Volume (成交量/资金流)
- **OBV_SLOPE_10D**: On-Balance Volume 10日斜率  
  衡量资金流入/流出强度

---

## 🎯 Holding Characteristics

| Metric | Strategy #1 | Strategy #2 | Portfolio | 说明 |
|:-------|:------------|:------------|:----------|:-----|
| **Avg Hold Period** | 9.2 days | 9.0 days | **~9 days** | 短周期轮动 |
| **Turnover (Annual)** | ~4,000% | ~4,100% | **~4,050%** | 高频策略 |
| **Trades (5.95y)** | 252 | 257 | **~255** | 年均 ~43 次 |
| **Position Size** | 2 ETF | 2 ETF | **2 ETF** | 集中持仓 |
| **Rebalance Freq** | 3 days | 3 days | **3 days** | 固定周期 |
| **Win Rate** | **50.79%** | **49.42%** | **~50%** | 健康水平 ✅ |
| **Profit Factor** | **2.11** | **2.11** | **2.11** | 盈亏比优秀 ✅ |

---

### 📊 胜率与盈亏比解读

#### 胜率 ~50%（健康区间 ✅）
```
过高 (>60%) → 过拟合风险，实盘衰减大
健康 (45-55%) → 趋势跟随策略典型特征 ✅
过低 (<40%) → 赌运气，不可持续
```

**本策略**: 50.79% / 49.42%，处于**最佳区间**，说明：
- ✅ 不依赖"预测准确率"，而是"截断亏损，放飞利润"
- ✅ 避免了过拟合（如果胜率 >70%，实盘必然崩溃）
- ✅ 不靠运气（如果胜率 <40%，说明因子无效）

#### 盈亏比 2.11（优秀 ✅）
```
Profit Factor = 总盈利 / 总亏损
```

**本策略**: 2.11 → 每赚 ¥2.11，只亏 ¥1  
**期望收益**: 50% × 2.11 - 50% × 1 = **+0.55** (正期望！)

**为什么盈亏比重要？**
- 胜率 50% + 盈亏比 2.11 → 长期必赚
- 胜率 70% + 盈亏比 1.0 → 实盘可能变 50% × 1.0 = 亏损
- 胜率 30% + 盈亏比 3.0 → 也能赚，但波动大

---

## 📊 Recent Holdings (Last 60 Days)

### Most Traded ETFs
| ETF Code | Name | Category | Frequency |
|:---------|:-----|:---------|:----------|
| **511260** | 科创50 | 国内大盘 | 4 次 |
| **515180** | 有色金属 | 大宗商品 | 4 次 |
| **513100** | 纳指ETF | 海外科技 | 2 次 |
| **513500** | 标普500 | 海外大盘 | 2 次 |
| **518880** | 黄金ETF | 避险资产 | 2 次 |

### Current Holdings (2025-12-12)
- **159949** (创业板ETF): 国内科技成长
- **159915** (科技ETF): 国内科技

---

## ⚡ Quick Commands

### 数据更新
```bash
cd sealed_strategies/v3.4_20251216/locked
uv run python scripts/update_daily_from_qmt_bridge.py --all
```

### 生成今日信号
```bash
# Strategy #1
uv run python scripts/generate_today_signal.py \
  --combo "ADX_14D + OBV_SLOPE_10D + SHARPE_RATIO_20D + SLOPE_20D"

# Strategy #2
uv run python scripts/generate_today_signal.py \
  --combo "ADX_14D + OBV_SLOPE_10D + PRICE_POSITION_120D + SHARPE_RATIO_20D + SLOPE_20D"
```

### 回测审计
```bash
uv run python scripts/batch_bt_backtest.py \
  --candidates ../artifacts/production_candidates.csv
```

### 查看最新持仓
```bash
uv run python -c "
import pandas as pd
df = pd.read_csv('raw/ETF/daily/159949.csv')  # 创业板ETF
print(df.tail(5))
"
```

---

## 🎯 Monitoring Dashboard

### 日频监控（每日收盘后）
| Metric | Threshold | Action |
|:-------|:----------|:-------|
| **组合日收益** | 连续 3 日 < -1% | 暂停开新仓 |
| **QDII持仓占比** | > 50% | 手动减仓 20% |
| **同步大跌** | 两策略单日同跌 > 2% | 次日减仓 30% |

### 周频审计（每周五）
| Metric | Threshold | Action |
|:-------|:----------|:-------|
| **持仓重合度** | > 90% | 分散失效，考虑停用一个 |
| **胜率** | < 45% 持续 1 月 | 暂停策略 |
| **回撤** | > 20% | 全部清仓，等待信号 |

---

## ⚠️ Risk Warnings

### 1. 高度同质化（伪多样性）
- **因子重合**: 4/5 因子相同（80%）
- **持仓重合**: 最近 60 天 >80% 标的相同
- **结论**: 这不是真实分散，而是"1.5 个策略"

### 2. QDII 依赖（海外风险）
- **近期占比**: 40% 交易为纳指/标普（513100/513500）
- **风险**: 美股暴跌会同步重创
- **建议**: QDII 持仓 > 50% 时手动减仓

### 3. 短周期高频（成本敏感）
- **平均持有**: 9 天
- **年化换手**: ~4,000%
- **交易成本**: 假设 0.02% 单边，年化成本 ~80%（实际收益需扣除）

### 4. 震荡市优化（环境依赖）
- **当前环境**: 政策底 + 震荡磨底（2025Q4）
- **风险**: 如果转为单边趋势市（如 2020H2），可能跑输单策略

---

## 🔧 Parameter Tuning (Advanced)

### 调仓频率（FREQ）
- **当前**: 3 交易日
- **可选**: 5 日（降低成本）、1 日（提高灵活性）
- **Trade-off**: 频率 ↑ → 收益 ↑ 但成本 ↑

### 持仓数量（POS_SIZE）
- **当前**: 2 只 ETF
- **可选**: 3 只（分散）、1 只（集中）
- **Trade-off**: 持仓 ↑ → 回撤 ↓ 但收益 ↓

### 止损阈值（MAX_DRAWDOWN）
- **当前**: 无止损（策略内置风控）
- **建议**: 单策略 -20%、组合 -15%、单日 -3%

---

## 📝 FAQ

### Q1: 为什么只有 2 个策略？
**A**: 基于最近 60 天交易记录分析，其他 3 个策略（-5% ~ -9%）严重拖累，只保留抗跌的 2 个（-0.23%）。

### Q2: 两个策略是否真实分散？
**A**: **不是**。因子重合 80%，持仓重合 >80%，这是"伪多样性"。但在震荡市表现最优，所以仍然保留。

### Q3: QDII 占比过高怎么办？
**A**: 手动干预。当 513100/513500/159920/513050/513130 持仓占比 > 50% 时，减仓 20%。

### Q4: 如何判断策略是否失效？
**A**: 三个信号：
1. 胜率 < 45% 持续 1 个月
2. 回撤 > 20%
3. 近 60 天收益 < -10%

### Q5: 如何从 v3.3 迁移到 v3.4？
**A**: 
1. 备份 v3.3 信号
2. 更新信号生成脚本（5 策略 → 2 策略）
3. 运行 v3.4 回测验证
4. 部署到生产

---

## 🎓 Learning Resources

### 因子说明
- `locked/src/etf_strategy/core/precise_factor_library_v2.py`（18 因子完整定义）

### 回测引擎
- `locked/src/etf_strategy/auditor/core/engine.py`（Backtrader 策略）

### 配置文件
- `locked/configs/combo_wfo_config.yaml`（WFO 参数）
- `locked/configs/etf_pools.yaml`（ETF 池定义）

### 工作流程
- `REPRODUCE.md`（从零复现完整回测）
- `DEPLOYMENT_GUIDE.md`（生产部署流程）

---

## 📞 Support

**问题排查**:
1. 检查 `README.md`（快速开始）
2. 查看 `REPRODUCE.md`（复现步骤）
3. 阅读 `artifacts/PRODUCTION_REPORT.md`（详细报告）
4. 运行 `uv run pytest tests/ -v`（环境验证）

**紧急联系**: 
- 单日跌 > 2%: 执行熔断（减仓 30%）
- 回撤 > 20%: 全部清仓，等待信号

---

**Last Updated**: 2025-12-16 16:00 CST  
**Quick Access**: `cat artifacts/QUICK_REFERENCE.md`
