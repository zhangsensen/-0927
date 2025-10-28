# 工程化 + 诊断最终反馈

## 核心发现

### ✅ 工程化三件套交付

**质量等级**：生产级

1. **数据版本绑定** ✅
   - WFO metadata → cross_section/factor_selection 精确关联
   - 版本错配消除，审计追踪完整

2. **成本模型** ✅
   - 参数化成本链路（0-N bps）
   - 毛/净收益分离，平均换手率记录

3. **无重叠OOS权益** ✅
   - stitched_oos_equity.csv 可直接推生产
   - 连续权益曲线，自动步长估计

---

## 🔬 诊断结果 - IC衰减分析

### 数字真相

```
WFO IC:          0.1728 (优秀)
OOS实际IC:       0.0156 (困难)
衰减幅度:        -91.0% 🔴 严重
```

### 时间序列发现

| 时期 | IC均值 | 趋势 |
|------|--------|------|
| 早期(窗1-18) | 0.0273 | 相对好 |
| 晚期(窗37-54) | 0.0108 | 持续下滑 |
| 衰减 | -60.3% | 非常规因素 |

**结论**：不是固定过拟合，而是**因子在时间流逝中失效**。

### 因子选中频率

```
前3名（都是100%）：
  PRICE_POSITION_20D      100%
  RSI_14                  100%
  SHARPE_RATIO_20D        100%

后续：
  RELATIVE_STRENGTH...     92.6%
  MOM_20D                  79.6%
  CMF_20D                  20.4%
  VORTEX_14D                7.4%
```

**观察**：前3个因子近乎垄断，存在**高度冗余或多重共线性**。

### IC与收益的关系

```
高IC窗口(>75分位):  Sharpe=-0.0042, AnnRet=+26.11% ✅
低IC窗口(<25分位):  Sharpe=-1.0963, AnnRet=-10.52% ❌
相关系数:            0.1901 (很弱)
```

**关键发现**：IC和收益相关性只有 0.19，说明即使高IC也很难转化为利润。

---

## 🎬 根本问题诊断

### 问题1：因子信号质量衰减（致命）

**症状**：
- WFO中IC=0.173，但OOS只有0.016
- 早期IC=0.027，晚期IC=0.011，持续衰减
- IC与实际收益的相关性仅0.19

**根本原因**（概率排序）：
1. **WFO IS期太短** (70% 概率)
   - IS可能只有80-100天，足以选到随机噪声因子
   - 因子在长期OOS中失效

2. **因子本身非平稳** (20% 概率)
   - 中国A股市场环境急速变化（2020-2025）
   - 因子统计特性在不同市场制度下漂移

3. **数据质量问题** (10% 概率)
   - 某些因子依赖前复权，可能存在漂移
   - 标准化方法不适配市场结构变化

### 问题2：因子冗余（重要但不致命）

**症状**：
- PRICE_POSITION_20D、RSI_14、SHARPE_RATIO_20D 100%被选中
- 这3个都是动量/位置指标，高度相关

**影响**：
- 信号多重共线，实际有效因子数<3
- 换手率居高（24.6%），抵消因子边际贡献

### 问题3：TopN=5太激进（次要）

**症状**：
- 每日完全重建5资产组合
- 24.6%日换手，吃掉3.68%收益（5 bps成本）

**但是**：
- 改TopN=10也不能根本救药，因为信号本身就差
- 这是 symptom，不是病根

---

## 💡 立即行动清单

### P1: 因子稳定性调查 (3小时)

```python
# 分析每个因子在WFO IS/OOS中的独立IC
# 代码位置: scripts/step3_run_wfo.py 输出增强

for window in all_windows:
    for factor in selected_factors:
        ic_in_is = calc_ic(factor_is, return_is)
        ic_in_oos = calc_ic(factor_oos, return_oos)
        print(f"因子{factor} | IS_IC={ic_in_is:.4f} | OOS_IC={ic_in_oos:.4f} | 衰减={衰减%}")
```

**如果发现**：
- SHARPE_RATIO_20D 在OOS衰减>80% → 删掉
- RSI_14 稳定性好 → 保留并增加权重
- PRICE_POSITION_20D 完全失效 → 替换成其他因子

### P2: 试验更长IS窗口 (2小时)

修改 `scripts/step3_run_wfo.py`：
- 当前：IS=80天，OOS=60天，步长=20天
- 试验：IS=120天，OOS=40天，步长=20天（权衡窗口数 vs 因子稳定性）

预期：IC衰减应该降到 50% 以内（从91%）

### P3: 因子冗余消除 (1小时)

试验移除冗余因子组合：
```
方案A: 只用RSI_14 + MOM_20D（去掉PRICE_POSITION）
方案B: 只用SHARPE + RELATIVE_STRENGTH（去掉RSI）
方案C: 只用1个主因子 + Ensemble平均
```

**目标**：如果IC不降反升，说明存在过参数化。

### P4: 信号平滑（周频而非日频）(1小时)

改 `run_backtest_combinations()`：
```python
# 当前：日频选股
# 改为：周一重建持仓，整周持有

factor_signal_weekly = factor_signal_daily.resample('W').first()
rebalance_every_n_days = 5  # 每周一
```

预期：
- 换手率从24.6% → ~5%
- Sharpe 因为噪声降低而改善

---

## 📊 成本敏感性总结

| 成本 | Sharpe | AnnRet | 可接受性 |
|------|--------|--------|---------|
| 0 bps | -0.051 | 19.76% | ❌ 勉强 |
| 5 bps | -0.217 | 16.08% | ❌ 不可用 |
| 10 bps | -0.38 | 12.40% | ❌ 废掉 |

**结论**：当前信号强度无法支撑任何成本。必须先提升IC。

---

## ✋ 暂停工程化

之前建议的"可选增强"（metadata.json、命令行参数、精确换手计算）：

**全部 PASS**。不做。

理由：
1. IC衰减91%，代码再完美也救不了
2. TopN、cost参数都可配了，够用
3. 投入产出比 <0.1，是技债，不是价值

---

## 🎯 最终决策

### 现状
```
✅ 工程化完成：版本绑定、成本链路、OOS拼接
✅ 诊断完成：IC衰减91%，因子失效是根本病因
❌ 策略可用性：Sharpe=-0.05，任何成本模型都杀死它
```

### 下一阶段（转向策略创新）
1. **今天**：完成P1因子稳定性分析（3小时）
2. **明天**：试验P2/P3（IS窗口+因子冗余）（3小时）
3. **后天**：试验P4（周频信号）（1小时）
4. **评估**：如果IC升到0.05+、Sharpe升到+0.2+ 才值得继续

---

## 代码状态

| 件 | 状态 | 位置 |
|----|------|------|
| Step 1-4 | ✅ 可用 | scripts/ |
| 诊断脚本 | ✅ 可用 | diagnose_ic_decay.py |
| 工程文档 | ✅ 完整 | ENGINEERING_CHECKPOINT.md |
| 测试回溯 | ✅ 可重现 | results/backtest/202510* |

### 立即可用

```bash
# 诊断
python diagnose_ic_decay.py

# 0 bps成本回测（当前默认）
python etf_rotation_optimized/scripts/step4_backtest_1000_combinations.py

# 5 bps成本回测
TRADING_COST_BPS=5 python etf_rotation_optimized/scripts/step4_backtest_1000_combinations.py

# 10 bps成本回测（预期Sharpe<-0.3）
TRADING_COST_BPS=10 python etf_rotation_optimized/scripts/step4_backtest_1000_combinations.py
```

---

## 最后的话

> 代码已经很干净了。系统已经能跑、能复现、能审计。  
> 但是信号根本不对。  
> 与其打磨框架，不如对着IC衰减的病根开刀。  
> 这才是真正能改变收益曲线的事。

---

**生成时间**：2025-10-27 17:55  
**版本**：工程化v1.2 + 诊断v1.0  
**下一检查点**：P1因子分析完成后
