# 🚀 全天候策略 WFO 快速参考卡

> **阅读时间**: 2 分钟  
> **完整文档**: `docs/ALLWEATHER_WFO_SPEC.md`

---

## ⚡ 核心原则 (必读)

```
┌────────────────────────────────────────────────────────┐
│  ❌ 禁止: 手工硬编码任何参数/权重/规则                   │
│  ✅ 正确: 一切皆因子，数据驱动，WFO 优化                │
└────────────────────────────────────────────────────────┘
```

### 错误示例
```python
# ❌ 绝对禁止
if is_bull:
    weight = 1.6  # 哪来的 1.6？
else:
    weight = 0.3  # 哪来的 0.3？
```

### 正确示例
```python
# ✅ 因子化处理
regime_factor = compute_regime_factor(prices)  # 返回连续值
# WFO 决定这个因子的使用方式和权重
```

---

## 🔄 标准流程

```
因子构建 → WFO 优化 → VEC 验证 → BT 审计
    │          │          │          │
    ↓          ↓          ↓          ↓
 32个因子   最优组合   daily_values  差异<0.01pp
```

---

## 📊 因子体系

| 类别 | 数量 | 适用池 |
|------|------|--------|
| 基础因子 | 18 | 所有 |
| 债券因子 | 5 | BOND |
| 商品因子 | 5 | COMMODITY |
| QDII因子 | 5 | QDII |
| 体制因子 | 4 | 全局权重调节 |

---

## 🏗️ 关键文件

| 文件 | 用途 | 状态 |
|------|------|------|
| `core/regime_factors.py` | 体制因子 | 待创建 |
| `configs/allweather_wfo_config.yaml` | WFO配置 | 待创建 |
| `run_allweather_wfo.py` | WFO入口 | 待创建 |

---

## ✅ 验收标准

| 指标 | 标准 |
|------|------|
| VEC/BT 差异 | < 0.01pp |
| OOS Sharpe | > 0.3 |
| 最大回撤 | < 30% |
| WFO 耗时 | < 5 分钟 |

---

## 🛠️ 命令速查

```bash
# WFO 优化
uv run python etf_rotation_optimized/run_allweather_wfo.py

# VEC 回测
uv run python scripts/run_allweather_vec.py --config <wfo_output>

# BT 审计
uv run python scripts/run_allweather_bt.py --config <wfo_output>

# 对齐验证
make verify
```

---

## ⚠️ 开发注意

1. **因子无前视偏差**: t 时刻只能用 t-1 及之前的数据
2. **分池独立 WFO**: 每个池有不同的有效因子
3. **体制因子也是因子**: 不是 if-else 规则
4. **必须运行代码**: 永不提交未运行的代码

---

**详细规格请阅读**: `docs/ALLWEATHER_WFO_SPEC.md`
