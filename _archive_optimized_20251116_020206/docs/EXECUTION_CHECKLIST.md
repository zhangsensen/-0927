<!-- ALLOW-MD -->
# 🎯 ETF轮动策略上线执行清单

## ✅ 已完成（2025-11-10）

### 1. WFO分析与策略筛选
- [x] WFO扫描12,597个组合完成
- [x] Top100性能验证通过（无未来函数，NAV diff=0）
- [x] 因子频率分析：RSI_14(90%), MAX_DD_60D(70%)
- [x] 筛选6个候选组合（Sharpe>0.9, MaxDD>-22%, 因子多样化）
- [x] 稳健性评估完成（基于WFO多窗口结果）

### 2. 生产配置生成
- [x] `production/strategy_config_v1.json` - 5个策略详细配置
- [x] `production/allocation_config_v1.json` - 权重分配与相关性矩阵
- [x] `production/DEPLOYMENT_GUIDE.md` - 完整部署文档
- [x] `production/strategy_candidates_selected.csv` - 候选清单

### 3. 风控规则设计
- [x] 策略级：60天滚动Sharpe<0.3/回撤<-30%触发降权50%
- [x] 组合级：总回撤<-28%, 单仓>12%, 行业集中度>40%
- [x] 紧急止损：连续10天亏损或单日-5%

---

## 📋 待执行（按优先级）

### Phase 1: 数据与基础设施（1-2天）

#### 1.1 数据接入
- [ ] **ETF数据源确认**
  - 43只ETF代码清单验证（对照`etf_download_manager/高优先级ETF下载清单_43只.txt`）
  - 历史数据完整性检查（2020-01-01至今，无缺失）
  - 日线OHLCV数据质量验证（复权方式、极值检测）
  
- [ ] **因子计算模块测试**
  - 18个因子计算函数单元测试
  - 与WFO结果对比验证（抽样3个策略，因子值diff<1e-6）
  - 性能测试（43只ETF×18因子×1400天<5秒）

#### 1.2 代码部署
- [ ] **生产环境准备**
  - Git分支：`git checkout -b production-v1.0`
  - 依赖安装：`pip install -r requirements.txt`（Python 3.11+, pandas, numpy, numba）
  - 配置文件路径映射（dev→prod）
  
- [ ] **核心模块部署**
  - `etf_rotation_optimized/core/` → `/prod/core/`
  - `etf_rotation_optimized/real_backtest/` → `/prod/backtest/`
  - `production/strategy_config_v1.json` → `/prod/config/active_strategy.json`

#### 1.3 风控系统集成
- [ ] **实时监控模块**
  ```python
  # 每日盘后执行
  def daily_risk_check():
      for strategy in active_strategies:
          dd_60d = calculate_rolling_drawdown(strategy, 60)
          sharpe_60d = calculate_rolling_sharpe(strategy, 60)
          if dd_60d < -0.30 or sharpe_60d < 0.30:
              trigger_alert(strategy, "RISK_BREACH")
              reduce_weight(strategy, factor=0.5)
  ```
  
- [ ] **预警通知配置**
  - Slack/钉钉webhook设置
  - 邮件报警列表（负责人+风控+技术）
  - 短信紧急通知（单日亏损>5%）

---

### Phase 2: 回测验证（2-3天）

#### 2.1 完整回测复现
- [ ] **运行生产回测脚本**
  ```bash
  cd /Users/zhangshenshen/深度量化0927/etf_rotation_optimized
  python real_backtest/run_production_backtest.py \
      --config ../production/strategy_config_v1.json \
      --start-date 2020-01-01 \
      --end-date 2025-10-14 \
      --output results/prod_validation/
  ```
  
- [ ] **对比WFO结果**
  - 5个策略的Sharpe/MaxDD/年化收益差异<2%
  - 调仓日期一致性检查
  - 成本模型验证（佣金+滑点）

#### 2.2 极端场景测试
- [ ] **历史压力测试**
  - 2020年3月（新冠暴跌）：组合最大回撤是否<-28%
  - 2021年春节后（抱团瓦解）：策略是否触发降权
  - 2022年全年（熊市）：累计收益是否为正
  
- [ ] **边界条件测试**
  - 数据缺失1天：是否自动跳过调仓
  - ETF停牌：是否剔除该标的并重新计算权重
  - 因子计算异常：是否回退至默认策略

---

### Phase 3: 模拟盘测试（7天）

#### 3.1 模拟环境搭建
- [ ] **券商API接入**
  - 模拟盘账户开通（华泰/中信/国泰君安）
  - 交易接口测试（下单/撤单/查询持仓）
  - 行情接口测试（实时价格/成交量）
  
- [ ] **自动化交易脚本**
  ```python
  def execute_rebalance(target_positions):
      current_positions = get_current_holdings()
      orders = calculate_trades(current_positions, target_positions)
      
      for order in orders:
          if order['action'] == 'BUY':
              place_order(order['symbol'], order['quantity'], 'LIMIT', order['price'])
          elif order['action'] == 'SELL':
              place_order(order['symbol'], order['quantity'], 'LIMIT', order['price'])
  ```

#### 3.2 模拟盘运行（11-11至11-17）
- [ ] **Day 1-3**: 冷启动
  - 初始化持仓（5个策略×5只ETF=25个头寸）
  - 记录初始NAV=1.0
  - 验证第一次调仓执行（预计Day 8）
  
- [ ] **Day 4-7**: 正常运行
  - 每日盘后风控检查
  - 对比理论NAV vs 实际NAV（允许偏差<0.5%）
  - 记录交易成本（实际佣金+滑点 vs 回测假设）

#### 3.3 问题排查
- [ ] **常见问题预案**
  | 问题 | 检查项 | 解决方案 |
  |------|--------|----------|
  | NAV偏离>1% | 成交价vs收盘价差异 | 调整执行时间至14:50 |
  | 订单未成交 | 流动性/涨跌停 | 改用市价单或次日补单 |
  | 因子计算错误 | 数据更新延迟 | 增加数据校验步骤 |
  | 风控误触发 | 滚动窗口计算bug | 修复逻辑并回测验证 |

---

### Phase 4: 小资金实盘（30天）

#### 4.1 实盘启动（11-18）
- [ ] **资金配置**
  - 初始资金：10万元（测试账户）
  - 策略权重：严格按`allocation_config_v1.json`执行
  - 单只ETF最小交易：100股（约500-1000元）
  
- [ ] **首次建仓**
  - 09:35执行首次下单（使用VWAP算法单）
  - 记录成交明细（价格/数量/时间）
  - 计算实际持仓权重 vs 目标权重偏差

#### 4.2 日常运行（11-18至12-17）
- [ ] **每日流程**（工作日）
  ```
  16:00 - 数据更新（ETF价格、因子计算）
  16:30 - 风控检查（60天滚动指标）
  17:00 - 生成日报（发送至Slack）
  
  次日09:30 - 若到调仓日，执行交易
  次日10:00 - 确认订单成交，更新持仓
  ```
  
- [ ] **每周复盘**（周五17:00）
  - 5个策略的周度收益/Sharpe/回撤
  - 与基准对比（沪深300、中证500）
  - 交易成本占比分析
  - 因子IC衰减监控

#### 4.3 异常处理记录
- [ ] **建立问题日志**
  | 日期 | 问题描述 | 影响策略 | 损失/偏差 | 解决措施 | 状态 |
  |------|----------|----------|-----------|----------|------|
  | 2025-11-20 | 某ETF停牌导致无法买入 | strat_002 | NAV偏离0.3% | 次日补单 | 已解决 |
  | ... | ... | ... | ... | ... | ... |

---

### Phase 5: 扩大规模（12-18后）

#### 5.1 绩效评估
- [ ] **30天实盘数据分析**
  - 实际Sharpe vs 回测Sharpe偏差
  - 实际成本 vs 回测成本假设
  - 滑点率统计（实际成交价 vs VWAP）
  
- [ ] **决策条件**
  - 实际Sharpe > 0.8 → 扩大至50万
  - 实际Sharpe 0.6-0.8 → 维持10万，继续观察
  - 实际Sharpe < 0.6 → 停止策略，分析原因

#### 5.2 资金扩容方案
- [ ] **50万→100万→500万**
  - 每次扩容前重新评估流动性
  - 单只ETF持仓不超过该ETF日均成交额的2%
  - 若遇容量瓶颈，考虑扩充ETF池（43只→100只）

---

## 🔧 可选优化（不阻塞上线）

### 后续增强（1个月后）
- [ ] 单因子WFO（combo_sizes=[1]）生成风险清单
- [ ] 校准器 vs IC排序A/B测试
- [ ] Rank-Sharpe负相关根因分析（当前-0.717）
- [ ] Top30鲁棒性测试（变化OOS长度/步长）

### 长期规划（3个月+）
- [ ] 机器学习因子权重（GBDT/XGBoost）
- [ ] 高频信号融合（30分钟/小时线）
- [ ] 跨资产扩展（A股/港股/商品ETF）
- [ ] 自适应风控（动态调整降权阈值）

---

## 📊 关键指标监控

### 每日监控（自动化）
- 组合NAV（5条子策略 + 1条总组合）
- 60天滚动Sharpe（每个策略）
- 当前回撤深度（与历史最高比）
- 调仓执行偏差（目标权重 vs 实际权重）

### 每周监控（人工复盘）
- 胜率趋势（最近20个交易日）
- 换手率与成本占比
- 因子IC衰减（Top5因子的滚动IC）
- 策略相关性变化（是否>0.7需要降权）

### 每月监控（深度分析）
- 完整绩效vs基准
- 因子重要性排序
- 收益归因（哪些因子贡献最大）
- 策略调整建议（剔除/替换/权重再分配）

---

## 📞 联系方式

**项目负责人**: [待填写]  
**风控负责人**: [待填写]  
**技术负责人**: [待填写]  
**紧急联系**: [手机]  

**文档路径**: `/production/`  
**Git Repo**: [待填写]  
**Slack频道**: `#etf-rotation-prod`

---

**最后更新**: 2025-11-10  
**预计上线**: 2025-11-18（小资金实盘）  
**下次审核**: 2025-12-18（30天后复盘）
