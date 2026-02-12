# v5.0-prod1 上线审批摘要

> **日期**: 2026-02-10
> **版本**: v5.0-prod1 (git tag, pushed)
> **性质**: 自运维参考，非对外文档

---

## 1. 策略概况

| 项目 | 值 |
|------|-----|
| 策略类型 | ETF 轮动 (A股 + QDII) |
| 持仓数量 | 2 只 ETF |
| 调仓频率 | 每 5 个交易日 |
| ETF 池 | 49 只 (41 A股 + 8 QDII 监控) |
| 宇宙模式 | A_SHARE_ONLY (QDII 硬屏蔽) |
| 迟滞控制 | delta_rank=0.10, min_hold_days=9 |
| 择时/风控 | 轻择时 + 波动率 regime gate (510300 proxy) |

**封存策略**:
- **S1 (4F)**: ADX_14D + OBV_SLOPE_10D + SHARPE_RATIO_20D + SLOPE_20D
- **S2 (5F)**: S1 + PRICE_POSITION_120D

---

## 2. BT Regression 基线 (2026-02-10)

成本档: med (A股 20bp, QDII 50bp), 执行: T1_OPEN

| Config | BT Full | BT Holdout | Trades | VEC Full | VEC-BT Gap(ho) |
|--------|---------|-----------|--------|----------|----------------|
| **S1_F5_ON** | +39.4% | **+21.4%** | 116 | +42.5% | -8.8pp |
| **S2_F5_ON** | +31.9% | **+21.3%** | 113 | +25.2% | -2.5pp |
| S1_F20_OFF | +45.9% | +17.9% | 87 | +42.8% | -1.6pp |
| S2_F20_OFF | +59.0% | +17.9% | 88 | +54.6% | -1.6pp |

**数据区间**: Train 2020-01 ~ 2025-04, Holdout 2025-05 ~ 2025-12

---

## 3. 验收标准 (全部 PASS)

| 标准 | 阈值 | 结果 | 判定 |
|------|------|------|------|
| 方向一致性 | VEC 正 → BT 正 | 全部 4 条一致 | PASS |
| Holdout gap | < 10pp | S1 -8.8, S2 -2.5, F20 -1.6 | PASS |
| Full gap | < 15pp | S1 -3.2, S2 +6.6, F20 +3~4 | PASS |
| 基线可复现 (±5pp) | 与历史记录对比 | S1: -3.9pp, S2: +4.4pp, F20: ~0pp | PASS |

---

## 4. 实盘验证 (6 周)

| 项目 | 值 |
|------|-----|
| 区间 | 2025-12-18 ~ 2026-02-09 |
| 收益 | +6.37% (+49,178 CNY) |
| 交易 | 22 笔, 胜率 83.3%, PL 2.33 |
| 持仓构成 | 100% A股, 0% QDII |
| 实际成本 | ~10-15bp (纯 A股) |

---

## 5. 已知风险与监控

### 5.1 F5_ON Trade Count Inflation (+63%)

**现象**: BT 交易笔数 (116/113) 比 VEC (71/70) 多约 63%

**原因**: 整手约束导致 lot-rounding re-buy (执行层摩擦，非决策逻辑错误)

**监控方式**:
- 每 20 个交易日统计: 成交笔数、成交额/净值、显性费用占比
- 对比 BT 预期: ~116 笔 / 252 交易日 ≈ 9.2 笔/20日

**降级阈值**: 20 日成交笔数 > 15 笔 或 显性费用率突变

### 5.2 成本敏感性

| QDII 占比 | 有效成本 | 风险等级 |
|-----------|---------|---------|
| ~0% (当前) | 10-15bp | 健康 |
| ~40% | 25-30bp | 承压 |
| ~70%+ | 40bp+ | 高危 |

**触发观察**: 当 QDII 进入 Top2 排名且连续 2 次调仓选入 QDII 时，开始监控成本占比

---

## 6. 降级方案: F20_OFF

| 项目 | F5_ON (主策略) | F20_OFF (降级) |
|------|---------------|----------------|
| 调仓频率 | 5 日 | 20 日 |
| 迟滞 | ON (dr=0.10, mh=9) | OFF |
| BT Holdout | +21% | +18% |
| 年化换手 | ~15x | ~12x |
| VEC-BT gap | 2~9pp | <2pp |
| Trade inflation | +63% | <3% |

**何时降级**:
1. 交易笔数异常飙升 (> 15 笔/20日)
2. 迟滞 state 文件损坏或环境不匹配冷启动后信心不足
3. 成本档从 low 升到 high (QDII 主导)

**降级操作**:
```yaml
# configs/combo_wfo_config.yaml
backtest:
  freq: 20
  hysteresis:
    delta_rank: 0.0
    min_hold_days: 0
```
同步更新 `frozen_params.py` 中 `CURRENT_VERSION` 或使用 `FROZEN_PARAMS_MODE=warn` 临时切换。

**恢复**: 排查原因 → 清理 signal_state.json → 恢复 FREQ=5 + Exp4 参数 → 跑一轮 VEC 验证 → 恢复

---

## 7. 每日运维 Checklist

### 交易日 (T日收盘后)
```bash
# 1. 更新数据
uv run python scripts/update_daily_from_qmt_bridge.py --all

# 2. 生成信号 (asof=T, trade_date=T+1)
uv run python scripts/generate_today_signal.py \
  --candidates results/production_pack_*/production_candidates.parquet \
  --asof $(date +%Y-%m-%d) \
  --trade-date <下一交易日>

# 3. 查看输出
cat data/live/signal_*.md

# 4. (调仓日) 确认 state 文件更新
cat data/live/signal_state.json | python -m json.tool | head -5
```

### 每 20 交易日
- 统计实盘交易笔数，对比 BT 预期 (~9 笔/20日)
- 检查显性费用占比
- 查看 QDII 排名趋势 (是否接近 Top2)

### 异常处理
| 场景 | 处理 |
|------|------|
| signal_state.json 校验失败 | 自动冷启动，无需手动干预；检查日志确认原因 |
| 连续 2 次调仓无换仓 | 正常 (迟滞在保护)，不需要干预 |
| BT/VEC 方向翻转 | STOP，跑完整 BT audit 后再决策 |
| Numba 报错 | `make clean-numba` 后重跑 |

---

## 8. 快速参考

```
# 回归验证
uv run python scripts/run_v5_validation.py          # VEC 12 runs (~3min)
uv run python scripts/run_bt_exp4_audit.py           # BT 4 runs (~30min)

# 冻结参数测试
uv run pytest tests/test_frozen_params.py -v

# 全管线冒烟
FROZEN_PARAMS_MODE=warn uv run python scripts/run_full_pipeline.py
```

---

*v5.0-prod1 | 回归通过 | 2026-02-10*
