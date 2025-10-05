我将用 Tavily 全网检索并行搜集与“港股中频（持仓2-3天）、小资金（~100万港币）、2025年适用策略”相关的权威资料与最新规则，再给出可落地的策略清单与要点。随后提炼要点并给出参考链接。

[7 tools called]

我先并行检索港股关键交易规则与可验证研究来源（印花税与费用、卖空规则、收市竞价时段、A-H溢价、跨市场ADR-H价差），随后汇总出适合2-3天持仓、小资金（~100万港币）的可落地策略清单与执行细节。

[5 tools called]

我已并行检索并核对了港股交易规则与近年可用研究（印花税/费用、卖空制度、CAS 收市竞价、A-H 溢价、ETF 流动性、ADR-H 价差）。下面给出面向小资金（~100万港币）、周内持仓2-3天的可落地策略清单与执行要点（均附可查证来源）。

### 策略清单（2-3天持仓，适配小资金）
- A-H 溢价均值回归（单边版）
  - 核心：当 A 股相对 H 股溢价指数偏高（如 HSAHP 指数偏离历史均值）时，优先做多折价明显的 H 股龙头，博取2-3天的溢价回归与情绪修复。
  - 标的池：恒生国企/大盘龙头（如 0700.HK、9988.HK、2318.HK 等）。
  - 入场/出场：用溢价Z分数或分位（如>+1σ触发观察，回落至均值附近减仓）；叠加南向成交/换手确认。
  - 依据：恒生A-H溢价指数方法与最新样本说明（含2025年6月版事实表）。参考：[Hang Seng A-H Premium Index（方法与成分）](https://www.hsi.com.hk/static/uploads/contents/en/dl_centre/factsheets/ahpremiume.pdf)

- ADR-H 价差对齐（开盘信息跟随）
  - 核心：以美股收盘ADR相对前日港股收盘的超额变动为信号，港股开盘择向入场，持有1-2天观察延续/回归（不做“无风险套利”，而是用作方向性/强度信号）。
  - 风险点：汇率变化、ADR成分与H股差异、隔夜消息；严控持仓与止损。
  - 依据：跨市场存托凭证存在价差与流动性耦合的证据（学术综述）。参考（方法论）：[Depositary Receipts Arbitrage综述](https://www.researchgate.net/publication/4959640_Arbitrage_opportunities_in_the_depositary_receipts_market_Myth_or_reality)

- 财报/电话会后短期动量（PEAD轻量化）
  - 核心：财报电话会披露的经营管理（OM）信息偏积极，往往带来后续超额收益；港股可缩短为2-3天“确认+持有”窗口。
  - 执行：筛选港股中/大盘披露后首日强势个股，叠加成交/南向放量确认；2-3天滚动止盈。
  - 依据：OM相关信息对后续收益有预测性（1-3个月层面），短窗可做战术化简化。参考：[Management OM 内容与收益（MnSc, 2023/2025）](https://pubsonline.informs.org/doi/10.1287/mnsc.2023.02221)

- 收市竞价（CAS）失衡信号的隔日/两日交易
  - 核心：CAS 阶段若出现显著买/卖盘失衡与异常价格偏移，次日早盘常见延续或回撤机会；结合盘后公告与南向流量判断方向。
  - 执行：记录 CAS 参考价与随机收盘区间，配合次日开盘缺口与前30分钟量能选择做多/做空（仅做多更友好）。
  - 依据：CAS 时段与细分阶段官方说明（2025年文件）。参考：[HKEX CAS 时段结构（2025资料包）](https://www.hkex.com.hk/-/media/HKEX-Market/Services/Circulars-and-Notices/Participant-and-Members-Circulars/SEHK/2025/CT10925E1.pdf)

- A-H 溢价/估值主题的ETF波段替代
  - 核心：对个股不熟悉时，用 2800.HK（恒指）、2828.HK（国企）等ETF做2-3天波段，择时依据A-H溢价、南向净流、政策/数据节奏。
  - 理由：ETF ADT与市场做市提升，滑点与个股尾部事件风险较低，更适合小资金周内交易。
  - 依据：香港ETF市场流动性与做市改善（2025年报告）；ETF入门与品种。参考：[HK ETF 市场研究（2025）](https://cms.hangsenginvestment.com/cms/ivp/hsvm/document/ETF_Research_Paper.pdf)，[ETF 指南（2025）](https://www.stashaway.hk/r/complete-guide-investing-hk-etf-us-etf)

- A-H 溢价主题的择时背景
  - 观察项：HSAHP 指数（>120/130说明A溢价较高，H股相对便宜，利好做多H的回归交易）。
  - 参考：[HSAHP 指数介绍/方法（2025年6月）](https://www.hsi.com.hk/static/uploads/contents/en/dl_centre/factsheets/ahpremiume.pdf)

### 成本、规则与可交易性（决定策略胜率的“地板”）
- 成本（2025年常见水平，官方/大行文件一致）：
  - 印花税：0.10%/边（买0.1%+卖0.1%≈往返0.2%）。参考：[费用表（2025-07，Mizuho）](https://www.mizuhogroup.com/binaries/content/assets/pdf/securities/asia-limited/market-fees-schedule.pdf)
  - 监管征费/交易费/处理费：SFC 0.0027%、AFRC 0.00015%、交易费 0.00565%、处理费 0.00341%/边 等；清算结算费率2025-06起上调（CCASS结算费 0.0042%/边，无最低最高）。参考：[长江证券公告（2025-06-16）](https://www.cjsc.com.hk/en/company-announcement-en/2937)、[HSBC 费用（2025-01-02）](https://www.hsbc.com.hk/content/dam/hsbc/hk/docs/broking/investments-products/fees-and-charges-schedule-2jan2025e.pdf)
  - 含滑点/佣金后，往返“门槛”常在≥0.60%附近（以流动性好、点差窄个股为前提）。2-3天策略需将目标/止损与胜率匹配该成本“地板”。

- 卖空与披露（如做双边/对冲）：
  - 仅“指定证券”可做卖空，且须合规借券，严禁裸卖空；卖空披露阈值为HK$3,000万或已发行额0.02%孰低。参考：[IFEC 卖空说明（2025-07）](https://www.ifec.org.hk/web/en/investment/investment-products/stock/stock-trading/short-selling.page)，[HKEX 交易所规则与卖空规章](https://www.hkex.com.hk/Services/Rules-and-Forms-and-Fees/Rules/SEHK/Rules-of-the-Exchange/Rules?sc_lang=en)
  - 新纳入“指定证券”公告可作为可做空/对冲的可交易性改善信号（例：CATL 2025年纳入）。参考：[HKEX 新闻稿（2025-05-16）](https://www.hkex.com.hk/News/News-Release/2025/250516news?sc_lang=en)

- CAS 时间与撮合阶段（影响收盘价形成方式与策略可行性）
  - 结构/阶段划分与随机收盘窗口详见官方文件。参考：[HKEX CAS 资料（2025）](https://www.hkex.com.hk/-/media/HKEX-Market/Services/Circulars-and-Notices/Participant-and-Members-Circulars/SEHK/2025/CT10925E1.pdf)

### 执行与风控（1,000,000 HKD 账户示例）
- 资金分配：单笔头寸 10万–20万HKD；单笔风险 0.4%–0.6%（即每笔亏损4k–6k HKD封顶）。
- 选股/流动性：优先恒指/国企成份与高ADT标的；尽量用限价单；避开除权除息与停牌风险。
- 止盈止损：目标≥成本“地板”2–3倍；CAS/开盘缺口出现预期外方向时减仓。
- 仅做多版本：如无合规借券条件，避免空头腿；用反向/杠杆产品需小仓位、短持有。

如需，我可以基于你现成的数据管线（因子/IC/成交/南向）把以上4类策略做成可回测的最小可行组合，并加上交易成本模型（印花税/结算费2025新规）与滑点假设，输出近两年（2024–2025）的实证统计与风控参数。