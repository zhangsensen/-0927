# 高优先级任务执行清单

**日期**: 2025-10-15  
**状态**: 待执行  
**优先级**: 🔴 高

---

## 📋 必须补齐（6项）

### 1. 全周期回测与归因 ⚠️
**状态**: 框架完成，需价格数据

**需要**:
- 价格数据源（open, high, low, close, volume）
- 路径: `raw/ETF/daily/*.parquet`

**输出**:
- 年度收益/回撤
- 极端月归因（单票/赛道/因子贡献）
- 月胜率、换手率

**脚本**: `scripts/etf_rotation_backtest.py`

**执行命令**:
```bash
# 需要先准备价格数据
python3 scripts/etf_rotation_backtest.py \
    --start-date 20200102 \
    --end-date 20251014 \
    --top-n 5 \
    --rebalance-freq M
```

---

### 2. 容量与ADV数据 ⚠️
**状态**: 框架完成，需成交量数据

**需要**:
- 20日ADV数据源
- ETF成交量历史数据

**约束**:
- 单标成交额 < 5% ADV
- 月度统计落盘

**脚本**: `scripts/capacity_constraints.py`

**执行命令**:
```bash
python3 scripts/capacity_constraints.py \
    --volume-data raw/ETF/daily \
    --target-capital 1000000
```

---

### 3. A股/QDII分池 🔴
**状态**: 待实现

**要求**:
- 面板按池分开生产
- 回测按池分开执行
- 禁止混池评分
- 策略侧顶层权重整合

**实现方案**:
```python
# 1. ETF分类配置
etf_pools = {
    'A_SHARE': ['510050.SH', '510300.SH', ...],  # A股ETF
    'QDII': ['513100.SH', '513500.SH', ...],      # QDII
}

# 2. 分池生产面板
for pool_name, symbols in etf_pools.items():
    produce_panel(symbols, output_dir=f'factor_output/{pool_name}')

# 3. 分池回测
for pool_name in etf_pools.keys():
    backtest(pool_name, calendar=get_calendar(pool_name))

# 4. 顶层整合
combine_pools(pools=['A_SHARE', 'QDII'], weights={'A_SHARE': 0.7, 'QDII': 0.3})
```

**新增脚本**: `scripts/pool_management.py`

---

### 4. 生产因子清单治理 ✅
**状态**: 已完成基础，需增强

**当前**:
- ✅ 12个生产因子
- ✅ 月度快照
- ✅ 漏斗报告

**需增强**:
- 因子权重记录
- 有效因子<8自动告警
- 回退方案

**实现**:
```python
# 增强alert_and_snapshot.py
def check_factor_count(factors):
    if len(factors) < 8:
        # 触发告警
        send_alert("因子数不足")
        # 执行回退
        fallback_to_previous_snapshot()
```

---

### 5. CI与泄露防线 ✅
**状态**: 基础完成，需集成

**当前CI检查**:
- ✅ 静态扫描shift(1)
- ✅ 索引规范检查
- ✅ 覆盖率阈值
- ✅ 有效因子数阈值

**需集成**:
- CI失败阻断面板放行
- 自动化pipeline

**实现**:
```bash
# .github/workflows/ci.yml
name: Factor Panel CI
on: [push, pull_request]
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - name: Run CI Checks
        run: python3 scripts/ci_checks.py
      - name: Block if Failed
        if: failure()
        run: exit 1
```

---

### 6. 价格口径与元数据 ✅
**状态**: 已实现，需完善元数据

**当前**:
- ✅ 统一price_field='close'
- ✅ 适配器记录
- ⚠️ panel_meta.json需完善

**需完善**:
```json
{
  "price_field": "close",
  "price_field_priority": ["adj_close", "close"],
  "engine_version": "1.0.1",
  "generated_at": "2025-10-15T14:00:00",
  "data_range": {
    "start_date": "2020-01-02",
    "end_date": "2025-10-14"
  },
  "factors": {
    "TA_RSI_14": {
      "min_history": 15,
      "family": "TA-Lib",
      "bucket": "momentum",
      "price_field": "close",
      "cache_key": "a3f2b1c4d5e6f7g8"
    }
  }
}
```

**脚本**: 修改`scripts/produce_full_etf_panel.py`

---

## 🎯 执行优先级

### 立即执行（今天）
1. ✅ **价格口径元数据完善** - 修改produce脚本
2. ✅ **生产因子治理增强** - 增强alert系统
3. 🔴 **A股/QDII分池** - 创建pool_management.py

### 本周执行
4. ⚠️ **全周期回测** - 需要价格数据
5. ⚠️ **容量ADV数据** - 需要成交量数据
6. ✅ **CI集成** - 创建GitHub Actions配置

---

## 📊 当前状态总结

| 任务 | 状态 | 阻塞因素 | 优先级 |
|------|------|----------|--------|
| 全周期回测 | ⚠️ 框架完成 | 需价格数据 | 🔴 高 |
| 容量ADV | ⚠️ 框架完成 | 需成交量数据 | 🔴 高 |
| A股QDII分池 | 🔴 待实现 | 无 | 🔴 高 |
| 因子清单治理 | ✅ 基础完成 | 需增强 | 🟡 中 |
| CI泄露防线 | ✅ 基础完成 | 需集成 | 🟡 中 |
| 价格口径元数据 | ✅ 已实现 | 需完善 | 🟡 中 |

---

## 🚀 下一步行动

### 今天完成
```bash
# 1. 完善元数据
python3 scripts/produce_full_etf_panel.py --enhance-metadata

# 2. 创建分池管理
python3 scripts/pool_management.py --create-pools

# 3. 增强告警系统
python3 scripts/alert_and_snapshot.py --enable-fallback
```

### 本周完成
- 补充价格数据源
- 补充成交量数据源
- 完成全周期回测
- 完成容量约束验证

---

**更新时间**: 2025-10-15 14:30  
**负责人**: AI工程师  
**审核**: Linus式标准
