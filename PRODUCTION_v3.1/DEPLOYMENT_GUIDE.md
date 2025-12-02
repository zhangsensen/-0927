# 🚀 部署指南

> 本指南帮助你将策略部署到实盘环境

---

## 1️⃣ 环境准备

### Python 环境
```bash
# 推荐使用 UV 包管理器
uv sync --dev
uv pip install -e .
```

### 数据准备
```bash
# 下载 ETF 数据 (需要 Tushare Token)
uv run python src/etf_data/scripts/download_etfs.py --config configs/full_config.yaml
```

---

## 2️⃣ 策略运行

### 生成交易信号
```bash
# 运行 WFO 优化器生成信号
uv run python src/etf_strategy/run_combo_wfo.py
```

### 输出文件
- `results/run_latest/` - 最新运行结果
- `results/run_latest/signals.csv` - 交易信号

---

## 3️⃣ 实盘执行

### 每日流程
1. **收盘后** (15:00 后): 运行策略生成明日信号
2. **开盘前** (9:15 前): 检查 QDII 溢价率
3. **开盘时** (9:30): 挂限价单执行交易

### QDII 溢价监控

| 溢价率 | 操作 |
|--------|------|
| < 1% | 正常交易 |
| 1-2% | 谨慎，可小仓位 |
| > 2% | **暂停交易**，等待回落 |

### 查看 IOPV (实时净值)
- 集思录: https://www.jisilu.cn/data/qdii/
- 东方财富: ETF 详情页

---

## 4️⃣ 风险控制

### 硬性规则
- 单只 ETF 最大仓位: 50%
- QDII 整体上限: 根据个人外汇额度
- 单日最大交易次数: 2 次 (FREQ=3 决定)

### 软性规则
- 大盘暴跌 > 5%: 观望一天
- 连续亏损 3 次: 检查是否系统问题
- 月度回撤 > 10%: 减仓至 50%

---

## 5️⃣ 常见问题

### Q: 为什么实盘收益和回测差距大？
A: 主要原因:
1. QDII 溢价 (最大杀手)
2. 滑点 (市价单)
3. 成交时点偏差

### Q: 可以调整 FREQ 或 POS 吗？
A: **不建议**。当前参数是局部最优，调整 = 过拟合。

### Q: 资金量多大合适？
A: 
- 最小: 10 万 (流动性够)
- 建议: 50-100 万 (滑点可控)
- 上限: 500 万 (需考虑冲击成本)

---

## 6️⃣ 核心参数 (只读)

```yaml
# 锁定，禁止修改
FREQ: 3           # 调仓频率
POS: 2            # 持仓数量
INITIAL_CAPITAL: 1000000
COMMISSION: 0.0002  # 2bp

# 因子组合 (锁定)
FACTORS:
  - ADX_14D
  - MAX_DD_60D
  - PRICE_POSITION_120D
  - PRICE_POSITION_20D
  - SHARPE_RATIO_20D
```

---

## 7️⃣ 联系方式

如遇问题，请检查:
1. 数据是否最新
2. 配置文件是否被修改
3. 运行日志中的错误信息

---

**🔒 本策略已封板，专注执行而非优化。**
