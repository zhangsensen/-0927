# ETF Data Directory — 数据字典

> 最后更新: 2026-02-12 | 维护者: Claude Code
>
> **目的**: 确保所有模型（策略引擎、研究脚本、AI助手）对每份数据的口径、价值、陷阱有一致理解。

---

## 信息层次总览

```
raw/ETF/
│
│  第1层: 价格 ──────────────────────────────────────────────
├── daily/           49 files   前复权OHLCV (策略引擎唯一数据源)    ✅ 生产级
│   └── 全部18个OHLCV因子从这里计算，Kaiser有效维度 5/17，已接近饱和
│
│  第2层: 价值锚 ────────────────────────────────────────────
├── fund_daily/      49 files   未复权OHLCV (真实成交价)            ✅ 生产级
├── fund_nav/        49 files   每日单位净值 (ETF内在价值)          ✅ 生产级
├── factors/         49+1       折溢价 = fund_daily.close / NAV - 1 ✅ 已修正
│   └── 折溢价IC=-0.046, 训练vs OOS几乎无衰减, 有套利锚定的均值回复性
│
│  第3层: 资金行为 ──────────────────────────────────────────
├── fund_share/      49 files   每日基金份额 (机构申赎代理)         ✅ 生产级
├── margin/           1 file    融资融券 (杠杆情绪)                 ✅ 可用
│   └── fund_share IC最强(-0.055→HO -0.113), margin待挖掘, 均与价格正交
│
│  第4层: 宏观环境 ──────────────────────────────────────────
├── fx/               2 files   外汇 (USDCNH离岸 + BOC中间价)     ✅ 生产级
│   └── QDII归因: 收益 = 底层标的 + 汇率变动 + 折溢价变动
│
│  第5层: 微观结构 ──────────────────────────────────────────
├── moneyflow/       49 files   资金流向 (5档)                     ❌ 数据损坏+仅120天
│
│  辅助 ─────────────────────────────────────────────────────
├── snapshots/        1 file    全市场ETF快照 (含IOPV)             📷 实盘用
├── realtime/         1 file    实时行情快照                        📷 实盘用
├── etf_list.parquet            100只ETF广义清单
└── tushare_extra/              IC研究中间结果
```

**核心判断**: 第2层(折溢价)和第3层(份额+融资融券)是突破OHLCV饱和天花板的最大alpha来源。

### 为什么第3层 > 第5层？数据生成机制决定一切

ETF数据按**生成机制**分为三类，质量天差地别：

| 类型 | 来源 | 代表 | 特点 |
|------|------|------|------|
| **结算级** | 中登/交易所清算系统 | fund_share, margin | 法定披露，经清算验证，**不可能错**——错了意味着钱对不上 |
| **基础行情** | 交易所K线接口 | daily, fund_daily, fund_nav | 炒股软件的命根子，结构固定，绝对准确 |
| **推算数据** | 数据商自行估算 | moneyflow | 东财/同花顺按成交金额大小**猜测**资金分类，规则人为设定，各家结果不同 |

**fund_share 为什么比 moneyflow 有效？**

机构大量买入ETF时，**不走二级市场**——走一级市场申购（拿一篮子股票换ETF份额）：
- **moneyflow**: 完全看不见（没有二级市场成交记录）
- **fund_share**: 精确捕捉（份额增加 = 有人申购了）

此外，ETF二级市场成交以做市商为主，做市商的大额报价是"提供流动性"而非"主力建仓"——moneyflow把两者混为一谈。

这就是 fund_share IC=-0.055（HO增强到-0.113）而 moneyflow 基本无用的根本原因：**真正的大钱走的通道，moneyflow根本监测不到。**

---

## ⚠️ 全局注意事项

### 日期格式不一致

| 数据集 | trade_date 类型 | 示例 | 统一方法 |
|--------|----------------|------|----------|
| daily/ | **int64** | `20260210` | 策略引擎原生格式 |
| fund_daily/, factors/, fx/, margin/ | **str** | `"20260210"` | `int(x)` 或 `pd.to_datetime(x)` |
| fund_nav/, fund_share/, moneyflow/ | **datetime64** | `2026-02-10` | `.dt.strftime('%Y%m%d').astype(int)` |

**跨表join前务必统一。** 策略引擎用int64，Tushare返回str，Pandas原生用datetime64。

### 覆盖差异

| 范围 | 文件数 | 说明 |
|------|--------|------|
| daily/, fund_daily/, factors/ | 49 | 策略池全量 |
| fund_nav/, fund_share/ | 49 | 策略池全量 |
| margin/ | 1 (stacked) | 49只ETF合并在一个文件，需pivot |
| moneyflow/ | 49 | 策略池全量，但数据损坏 |

---

## 1. daily/ — 策略引擎主数据 (第1层: 价格)

**代表什么**: 市场对ETF的定价共识。前复权价消除了分红/拆分断裂，保证收益率连续可比。

| 属性 | 值 |
|------|---|
| 来源 | QMT Trading Terminal via `qmt-data-bridge` SDK |
| 覆盖 | 49只 ETF (49策略池 + 3观察) |
| 历史 | 各自上市日 ~ 2026-02-10, ~1455行/只 |
| 更新 | `scripts/update_daily_from_qmt_bridge.py --all` |
| 消费者 | `src/etf_strategy/core/data_loader.py` → WFO/VEC/BT 三层引擎 |

**Schema:**

| 字段 | 类型 | 说明 | 示例 |
|------|------|------|------|
| trade_date | **int64** | 交易日 YYYYMMDD | 20260210 |
| adj_open | float64 | 前复权开盘价 | 4.707 |
| adj_high | float64 | 前复权最高价 | 4.753 |
| adj_low | float64 | 前复权最低价 | 4.649 |
| adj_close | float64 | 前复权收盘价 | 4.733 |
| vol | float64 | 成交量 (股) | 3795550.0 |

**文件命名**: `{ts_code}_daily_{start}_{end}.parquet`

**价值**: 全部18个OHLCV因子从这里计算 (SLOPE, SHARPE, ADX, OBV, CMF, CALMAR...)

**局限**:
- 前复权 = 历史价格被追溯调整。**不能**与NAV比较，**不能**算真实折溢价
- Kaiser有效维度仅5/17 — OHLCV因子空间已接近饱和
- 不含amount(成交额)字段

---

## 2. fund_daily/ — 真实成交价 (第2层: 价值锚)

**代表什么**: 当天投资者实际买卖的价格，没有任何调整。close=4.733就是收盘时的真实价。

| 属性 | 值 |
|------|---|
| 来源 | Tushare Pro `fund_daily` 接口 |
| 覆盖 | 49只 ETF |
| 历史 | 各自上市日 ~ 2026-02-12, ~1457行/只 |
| 更新 | `scripts/download_supplementary_data.py --only fund_daily` |

**Schema:**

| 字段 | 类型 | 说明 | 示例 |
|------|------|------|------|
| ts_code | str | ETF代码 | "510300.SH" |
| trade_date | **str** | 交易日 YYYYMMDD | "20260210" |
| pre_close | float64 | 昨收 (原始) | 4.727 |
| open | float64 | 开盘 (原始未复权) | 4.733 |
| high | float64 | 最高 (原始) | 4.753 |
| low | float64 | 最低 (原始) | 4.722 |
| close | float64 | **收盘 (原始未复权)** | 4.733 |
| change | float64 | 涨跌额 | 0.006 |
| pct_chg | float64 | 涨跌幅 (%) | 0.1269 |
| vol | float64 | 成交量 (手) | 3795550.0 |
| amount | float64 | 成交额 (千元) | 1537472.863 |

**文件命名**: `fund_daily_{code}_{market}.parquet` (如 `fund_daily_510300_SH.parquet`)

**价值**:
- **折溢价计算的分子** — `premium = (close / unit_nav - 1) × 100`
- 验证前复权数据是否有复权错误
- 含amount字段，daily/没有

**与daily/的区别**: daily给策略算因子用(需要连续收益率)，fund_daily给折溢价和实盘对账用。

---

## 3. fund_nav/ — 基金内在价值 (第2层: 价值锚)

**代表什么**: 基金管理人每天收盘后按持仓市值算出的"内在价值"。ETF的NAV是底层一篮子股票/资产的加权净值。

| 属性 | 值 |
|------|---|
| 来源 | Tushare Pro `fund_nav` 接口 |
| 覆盖 | 49只 ETF (策略池) |
| 历史 | 2020-01-20 ~ 2026-02-11, ~1467行/只 |
| 更新 | `scripts/update_tushare_funddata.py` |

**Schema:**

| 字段 | 类型 | 说明 | 可用性 |
|------|------|------|--------|
| ts_code | str | ETF代码 | ✅ |
| trade_date | **datetime64** | 净值计算日 (T日收盘) | ✅ |
| ann_date | **str** YYYYMMDD | 公告日 (**T+1日**) | ✅ |
| unit_nav | float64 | **单位净值** (未复权) | ✅ 核心字段 |
| accum_nav | float64 | 累计净值 (含历史分红) | ✅ |
| adj_nav | float64 | 复权净值 | ✅ |
| net_asset | float64 | 资产净值 (元) | ⚠️ 多数NaN |
| total_netasset | float64 | 合计资产净值 | ❌ 99%为NaN |
| accum_div | str | 累计分红 | ❌ 全部NaN |
| update_flag | str | 更新标记 | 0或1 |

**文件命名**: `fund_nav_{code6}.parquet` (如 `fund_nav_510300.parquet`)

**核心口径**:
- **`ann_date = trade_date + 1`**: NAV是T日收盘后计算，T+1公告。回测中用T日NAV算T+1信号，**没有前视偏差**
- `unit_nav` 是未复权净值，与 `fund_daily.close` 直接可比
- `total_netasset` 大面积缺失，不可用

**价值**:
- **折溢价因子的分母** — close > NAV = 二级市场给了溢价
- QDII溢价特别有意义: 纳指ETF溢价+2.6% = 国内投资者愿意多付2.6%买美股敞口(额度稀缺+汇率预期+情绪)
- A股ETF折溢价 ≈ 0%: 有申赎套利机制纠正偏离，但偏离方向和速度仍有信息量

---

## 4. fund_share/ — 机构在投票 (第3层: 资金行为)

**代表什么**: ETF总份额变动 = 净申购 - 净赎回。份额增加说明有人在一级市场申购(通常是机构)，减少说明在赎回。

| 属性 | 值 |
|------|---|
| 来源 | Tushare Pro `fund_share` 接口 |
| 覆盖 | 49只 ETF (策略池) |
| 历史 | 2020-01-20 ~ 2026-02-12, ~1465行/只 |
| 更新 | `scripts/update_tushare_funddata.py` |

**Schema:**

| 字段 | 类型 | 说明 | 示例 |
|------|------|------|------|
| ts_code | str | ETF代码 | "510300.SH" |
| trade_date | **datetime64** | 交易日 | 2026-02-12 |
| fd_share | float64 | **基金份额 (万份)** | 1004178.77 |
| fund_type | str | 基金类型 | "ETF" (常量) |
| market | str | 市场 | "SH" / "SZ" |

**文件命名**: `fund_share_{code6}.parquet` (如 `fund_share_510300.parquet`)

**口径**: **万份**。510300 约100万万份 = 100亿份，合理。

**价值 — 突破OHLCV天花板的关键**:
- **结算级数据**: 来自中登/交易所清算系统，法定披露义务，经清算验证，不可能错
- **捕捉一级市场大钱**: 机构大量买入ETF时走一级市场申购（拿一篮子股票换份额），这个行为在二级市场成交记录(moneyflow)里**完全不可见**，但fund_share精确捕捉
- **IC已验证**: SHARE_CHG_10D IC = -0.055 (HO -0.113)，**最强的非OHLCV因子**，且HO不衰减反而增强——说明信号是结构性的，不是过拟合
- **与价格正交**: 完全独立于OHLCV的信息源。Kaiser分析说17个OHLCV因子有效维度只有5，份额变动来自完全不同的信息通道
- **经济学解释清晰**: 份额增加 → 后续收益差 = 机构"接盘"信号（散户不走申赎通道，门槛高、需一篮子股票）

**衍生因子方向** (IC研究结果见 `tushare_extra/ic_results_fundshare.csv`):
- SHARE_CHG_5D/10D/20D: 份额变动率
- SHARE_ACCEL: 份额加速度
- SHARE_RANK_CHG: 截面排名变动

**局限**: 不区分申购和赎回，只有净结果。日内份额波动看不到。

---

## 5. factors/ — 折溢价因子 (第2层衍生)

**代表什么**: 二级市场价格相对NAV的偏离程度。溢价=投资者愿意多付钱买，折价=投资者想卖但没人接。

| 属性 | 值 |
|------|---|
| 来源 | 本地计算: `fund_daily.close / fund_nav.unit_nav - 1` |
| 覆盖 | 46只 (fund_nav已全量覆盖) |
| 历史 | 各自上市日 ~ 2026-02-11, ~1456行/只 |
| 状态 | ✅ 已修正 (旧版错用adj_close导致系统性偏差，备份为.bak) |

**Schema:**

| 字段 | 类型 | 说明 | 示例 |
|------|------|------|------|
| trade_date | **str** YYYYMMDD | 交易日 | "20260210" |
| premium_rate | float64 | 折溢价率 (%) | +2.575 |

**文件命名**: `premium_rate_{code6}.parquet`

**附带**: `premium_rate_summary.csv` (46只, 含mean/std/min/max统计)

**口径验证**:
- A股ETF: 均值 ≈ 0% (±0.05%) — 套利机制使偏离快速收敛 ✅
- QDII ETF: 纳指+2.6%, 中概+2.2%, 标普+1.5% — 反映跨境溢价 ✅
- 旧数据(adj_close): 510300均值-4.5%, 513100均值-25% ← **已作废，.bak文件**

**价值**:
- **PREMIUM_DEV_20D IC = -0.046**: 最稳定因子 (训练-0.046 vs OOS-0.047, 几乎无衰减)
- QDII溢价 = 额度稀缺度的温度计: 从+1%飙到+10%时往往是短期顶部
- 均值回复性: 有天然套利锚定，不像动量因子regime-dependent

---

## 6. fx/ — QDII的隐藏损益 (第4层: 宏观环境)

**代表什么**: 人民币兑美元/港元的汇率变动。QDII ETF的收益 = 底层标的表现 + 汇率变动 + 折溢价变动。

### usdcnh_daily.parquet — 离岸做因子用

| 属性 | 值 |
|------|---|
| 来源 | Tushare Pro `fx_daily` (FXCM离岸市场) |
| 历史 | 2019-01-02 ~ 2026-02-11, 2204行 |

| 字段 | 类型 | 说明 |
|------|------|------|
| ts_code | str | "USDCNH.FXCM" (常量) |
| trade_date | **str** YYYYMMDD | 交易日 |
| bid_open / bid_close / bid_high / bid_low | float64 | 买入报价 |
| ask_open / ask_close / ask_high / ask_low | float64 | 卖出报价 |
| tick_qty | int64 | 成交笔数 |

**口径**: 离岸人民币市场真实交易价，波动大于在岸。`bid_close` 范围 ≈ 6.3~7.4。

### boc_fx_daily.parquet — 央行做归因用

| 属性 | 值 |
|------|---|
| 来源 | AkShare `currency_boc_safe` (国家外汇管理局) |
| 历史 | 2019-01-02 ~ 2026-02-12, 1729行 |

| 字段 | 类型 | 说明 | 单位陷阱 |
|------|------|------|----------|
| trade_date | **str** YYYYMMDD | 交易日 | |
| 美元 | float64 | USD/CNY 中间价 | **100美元兑人民币** (≈725, 不是7.25!) |
| 港元 | float64 | HKD/CNY 中间价 | **100港元兑人民币** (≈93) |

**价值**:
- **Exp6数据基础**: FX beta因子 — USDCNH上涨(人民币贬值)时QDII自动受益
- **归因必备**: 不分解FX，无法知道赚的是美股alpha还是人民币贬值beta
- **两个源互补**: FXCM做因子(市场驱动、波动大)，BOC做归因(官方基准、稳定)

**为什么重要**: 策略目前100% A股、零QDII持仓。但如果未来QDII入选，不理解FX暴露就是裸奔。

---

## 7. margin/ — 杠杆情绪 (第3层: 资金行为)

**代表什么**: 谁在借钱买(融资)和谁在借券卖空(融券)。结算级数据，来自券商向交易所的法定申报。

### margin_pool43_2020_now.parquet ✅

| 属性 | 值 |
|------|---|
| 来源 | Tushare Pro `margin_detail` (原始数据来自沪深交易所) |
| 数据级别 | **结算级** — 券商法定申报，经交易所验证 |
| 格式 | **长表** (49只ETF合并, 50806行) |
| 历史 | 2020-01-02 ~ 2026-02-11 |
| 注意 | 需按ts_code pivot后才能做截面因子 |

| 字段 | 类型 | 说明 | 因子方向 |
|------|------|------|----------|
| trade_date | **str** YYYYMMDD | 交易日 | |
| ts_code | str | ETF代码 | |
| rzye | float64 | **融资余额** (元) | 杠杆多头存量 |
| rqye | float64 | 融券余额 (元) | 卖空存量 |
| rzmre | float64 | **融资买入额** | 当日新增杠杆，比存量更灵敏 |
| rzche | float64 | 融资偿还额 | |
| rqyl | float64 | 融券余量 (股) | |
| rqchl | float64 | 融券偿还量 | |
| rqmcl | float64 | 融券卖出量 | |
| rzrqye | float64 | 融资融券余额合计 | |

**潜在因子方向**:
- `rzye / 流通市值` = 杠杆密度
- `rzmre / amount` = 杠杆占比 (当日新增杠杆占总成交的比例)
- `rzye / rqye` = 融资融券比 = 多空情绪
- 以上均需标准化为截面rank

**覆盖**: 41/43只ETF有>500天数据。小ETF(如516520)数据较少。

**局限**: 不是所有ETF都是两融标的。

---

## 8. moneyflow/ — 资金流向 ❌ 数据损坏 + 结构性缺陷

**理论价值**: 按订单大小分的资金流向 — 超大单(>100万)=机构, 大单(20-100万)=大户, 中/小单=散户。

**结构性缺陷 (即使数据修好也存在)**:
- **推算数据，非结算数据**: 东财/同花顺按成交金额大小自行分类，规则人为设定，各家结果不同
- **ETF做市商污染**: ETF二级市场成交以做市商为主，做市商的大额报价是"提供流动性"而非"主力建仓"，moneyflow把两者混为一谈
- **看不到一级市场**: 机构大额申赎走一级市场，不经过二级市场成交——moneyflow对真正的大钱完全盲区
- **Tushare `moneyflow` 接口不支持ETF**: 只支持个股。`TushareFlowCrawler` 是死代码，调用返回空

| 属性 | 值 |
|------|---|
| 来源 | 东方财富 Web API (非标准化"特色数据"接口，非K线级基础行情) |
| 覆盖 | 49只, 各~120行 |
| 历史 | 2025-08 ~ 2026-02 (120天硬限制，无法获取更早) |
| 状态 | ❌ **列数据错位 + 历史不足 + 无替代源** |

**Schema (标称 vs 实际)**:

| 字段 | 标称含义 | **实际存储** |
|------|----------|-------------|
| date | datetime64 | ✅ 正确 |
| main_net | 主力净流入(元) | ✅ 大致正确 |
| main_net_pct | 主力净流入占比 | ✅ |
| xl_net | 超大单净流入 | ✅ |
| xl_net_pct | 超大单占比 | ✅ |
| l_net | 大单净流入 | ✅ |
| l_net_pct | 大单占比 | ✅ |
| m_net | 中单净流入 | ❌ **实际是收盘价** (4.18~4.80) |
| m_net_pct | 中单占比 | ❌ **实际是涨跌幅** |
| s_net | 小单净流入 | ❌ **实际是涨跌幅** (-17~11) |
| s_net_pct | 小单占比 | ❌ **全为0** |

**结论**: 仅main/xl/l三档可信，m/s两档数据错位。不入因子管道。可作实盘监控参考。

**无长历史替代源**: 已确认三个渠道全部不行：
1. Tushare `moneyflow` — 接口不支持ETF（仅个股）
2. 东财 Web API — 120天硬限制 + 列错位
3. AkShare — 同样无ETF长历史moneyflow

**替代方案**: 不需要修复moneyflow。fund_share（结算级，IC=-0.055）和margin（结算级）已经从更可靠的通道捕捉了机构行为，且覆盖了moneyflow看不到的一级市场申赎。

---

## 9. 辅助数据

### snapshots/ — 全市场ETF快照 📷

| 属性 | 值 |
|------|---|
| 来源 | AkShare `fund_etf_spot_em` (东方财富) |
| 覆盖 | **1383只ETF** (全市场, 远超策略池43只) |
| 字段数 | 37列 |
| 存储 | `snapshot_{YYYYMMDD}.parquet`, 单日快照 |

**关键字段**: `IOPV实时估值`, `基金折价率`, 5档资金流(主力/超大/大/中/小), `委比`, `量比`, `换手率`, `最新份额`, `流通市值`

**价值**: 盘中IOPV是实时参考净值(比收盘NAV更及时)。1383只ETF全覆盖可用于universe扩展研究。

### realtime/ — 实时行情 📷

`quotes_{YYYYMMDD}.parquet`, 100只ETF, 8列 (code, name, price, change_pct, volume, amount, iopv, premium_rate)。
IOPV列多数为None。

### etf_list.parquet — ETF清单

100只ETF的code/name/market/ts_code。广义池(策略池43只是其子集)。

### tushare_extra/ — 研究中间产物

`ic_results_fundshare.csv`: 8个fund_share衍生因子的IC测试结果。
最强: SHARE_CHG_10D IC=-0.055, HO IC=-0.113。

---

## 因子构建路线图

### 已在生产的因子 (18个, 全部来自daily/)

```
ADX_14D, OBV_SLOPE_10D, SHARPE_RATIO_20D, SLOPE_20D          ← S1 (生产)
AMIHUD_ILLIQUIDITY, CALMAR_RATIO_60D, CORRELATION_TO_MARKET_20D ← C2 (shadow)
CMF_20D, DOWNSIDE_VOL_20D, GK_VOLATILITY_20D, MDD_60D,
PRICE_POSITION_20D, PRICE_POSITION_120D, PV_CORR_20D,
RSI_14, SPREAD_20D, TSMOM_20D, VOL_RATIO_5_20              ← 候选池
```

### 已验证待入管道的因子

| 因子 | 数据源 | IC | 状态 |
|------|--------|-----|------|
| PREMIUM_DEV_20D | factors/ (折溢价) | -0.046 (无衰减) | 已在config的18因子中 |
| SHARE_CHG_10D | fund_share/ | -0.055 (HO -0.113) | IC验证通过，待WFO |
| 78个代数因子 | daily/ (组合) | varies | 6个BT候选，见algebraic_factor_vec_validation.md |

### 待开发的因子方向

| 方向 | 数据源 | 实验编号 | 数据就绪? |
|------|--------|----------|-----------|
| FX beta | fx/ | Exp6 | ✅ 数据已有 |
| QDII折溢价归因 | fund_nav/ + fx/ | Exp7 | ✅ 数据已有 |
| 杠杆情绪 | margin/ | 待定 | ✅ 数据已有 |
| 份额变动 | fund_share/ | 待定 | ✅ 数据已有 |
| 北向资金 | 未采集 | 待定 | ❌ |
| 期权隐含波动率 | 未采集 | 待定 | ❌ |

---

## 清理记录 (2026-02-12 已完成)

| 状态 | 对象 | 原因 |
|------|------|------|
| ✅ 已删 | `margin/margin_2025.parquet` | 两源混拼, 52% NaN, 被pool43替代 |
| ✅ 已删 | `factors/*.bak.parquet` (42个) | 旧版用adj_close算折溢价, 系统性偏差 |
| ✅ 已删 | `shares/` 整个目录 | 被fund_share/替代 |
| ✅ 已删 | `holders/` | 3行残片数据 |
| ✅ 已删 | `summary/` | 空目录 |
| ✅ 已删 | `download_report.md` | 旧报告 |
| 保留 | `moneyflow/` | 损坏但120天实盘可参考 |
| 保留 | `snapshots/`, `realtime/` | 实盘监控用 |
