# 因子健康监控系统

## 核心哲学

**没有永恒的alpha。因子会失效，必须监控。**

---

## 为什么因子会失效？

### 1. 套利机会被榨干
- Alpha被发现 → 资金涌入 → 收益磨平
- 例：动量因子在2010年代有效，现在拥挤

### 2. 市场结构变化
- 2020年有效 ≠ 2024年有效
- 例：疫情后市场结构性变化

### 3. 过度拟合历史数据
- 所谓"有效因子"可能只是过拟合
- 样本外失效是常态

---

## 系统架构

```
数据输入 → 因子计算 → 健康监控 → 动态筛选 → WFO优化 → 策略执行
                         ↑
                    每季度重新平衡
```

---

## 使用方法

### 1. 基础监控

```python
from core.factor_health_monitor import FactorHealthMonitor

# 创建监控器
monitor = FactorHealthMonitor(
    lookback_recent=252,      # 最近1年
    lookback_historical=504,  # 历史2年
    health_threshold_dying=0.7,
    health_threshold_dead=0.5,
    verbose=True
)

# 监控因子池
healthy, dying, dead, reports = monitor.monitor_factor_pool(
    factors_dict,  # {factor_name: (T, N) array}
    returns        # (T, N) array
)

# 查看报告
for name, report in reports.items():
    print(report)
```

### 2. 动态因子池

```python
# 获取动态因子池（自动淘汰失效因子）
filtered_factors, factor_weights = monitor.get_dynamic_factor_pool(
    factors_dict,
    returns,
    min_healthy_factors=3  # 最少保留3个健康因子
)

# filtered_factors: 只包含健康+濒死因子
# factor_weights: 基于健康评分的权重
```

### 3. 集成到WFO流程

```python
# 在WFO前先过滤因子池
filtered_factors, weights = monitor.get_dynamic_factor_pool(
    original_factors,
    returns
)

# 使用过滤后的因子池进行WFO
wfo_optimizer.run(filtered_factors, returns)
```

---

## 健康评分标准

### 衰减比率
```
decay_ratio = recent_ic / historical_ic
```

### 健康评分
```
health_score = decay_ratio * (1.0 if recent_ic > min_ic else 0.5)
```

### 状态判断

| 健康评分 | 状态 | 操作 |
|---------|------|------|
| ≥ 0.7 | ✅ healthy | 保持使用 |
| 0.5-0.7 | ⚠️ dying | 降低权重，密切监控 |
| < 0.5 | ❌ dead | 立即淘汰 |
| IC < 0 | ❌ dead | 立即淘汰 |

---

## 监控频率

### 建议频率
- **每月**: 运行健康检查，生成报告
- **每季度**: 重新平衡因子池，淘汰失效因子
- **每年**: 全面审查因子库，引入新因子

### 自动化脚本

```bash
# 每月1号运行
0 0 1 * * cd /path/to/project && python scripts/monthly_factor_health_check.py

# 每季度1号运行
0 0 1 1,4,7,10 * cd /path/to/project && python scripts/quarterly_factor_rebalance.py
```

---

## 实战案例

### 案例1：检测因子衰减

```python
# 假设你有13个因子
factors_dict = {
    "MOM_20D": mom_20d_data,
    "SLOPE_20D": slope_20d_data,
    # ... 其他11个因子
}

# 运行监控
monitor = FactorHealthMonitor(verbose=True)
healthy, dying, dead, reports = monitor.monitor_factor_pool(
    factors_dict, returns
)

# 输出示例：
# ✅ MOM_20D: 健康评分 0.92 (保持使用)
# ⚠️ SLOPE_20D: 健康评分 0.65 (降低权重)
# ❌ VOL_RATIO_20D: 健康评分 0.32 (立即淘汰)
```

### 案例2：动态因子池

```python
# 原始13个因子
original_factors = load_all_factors()

# 每季度重新平衡
filtered_factors, weights = monitor.get_dynamic_factor_pool(
    original_factors,
    returns,
    min_healthy_factors=5  # 至少保留5个
)

# 结果：可能只剩8个健康因子
# 权重自动根据健康评分分配
```

---

## 配置说明

### default.yaml配置

```yaml
factor_selection:
  health_monitor:
    enabled: true                     # 启用健康监控
    lookback_recent: 252              # 最近窗口（1年）
    lookback_historical: 504          # 历史窗口（2年）
    health_threshold_dying: 0.7       # 濒死阈值
    health_threshold_dead: 0.5        # 死亡阈值
    min_healthy_factors: 3            # 最少健康因子数
    rebalance_frequency: "quarterly"  # 重新平衡频率
```

---

## 警告信号

### 立即检查的情况

1. **健康因子 < 3个**
   - 系统风险极高
   - 立即停止交易，重新审查因子库

2. **所有因子IC < 0.05**
   - 因子池整体失效
   - 需要引入新因子或暂停策略

3. **衰减比率 < 0.5**
   - 因子快速失效
   - 检查市场结构是否变化

---

## Linus式检查清单

### 每月检查
- [ ] 运行健康监控报告
- [ ] 检查健康因子数量
- [ ] 记录衰减趋势

### 每季度检查
- [ ] 重新平衡因子池
- [ ] 淘汰失效因子
- [ ] 评估是否需要新因子

### 每年检查
- [ ] 全面审查因子库
- [ ] 回测历史表现
- [ ] 更新因子计算逻辑

---

## 常见问题

### Q: 为什么不直接用IC阈值筛选？
A: IC阈值是静态的，无法检测因子衰减。健康监控是动态的，能及时发现失效趋势。

### Q: 如果所有因子都失效怎么办？
A: 
1. 立即停止交易
2. 检查数据质量
3. 审查市场环境
4. 引入新因子或暂停策略

### Q: 多久重新平衡一次因子池？
A: 建议每季度。太频繁会过度交易，太慢会错过失效信号。

---

## 总结

**核心原则**：
1. 接受因子会失效的现实
2. 建立持续监控体系
3. 动态轮换因子池
4. 像选股一样选因子

**Linus哲学**：
> "量化交易是工程问题，不是学术研究。  
> 能赚钱的因子就是好因子，不管它理论上多优美。"

---

## 参考资料

- `core/factor_health_monitor.py` - 核心实现
- `configs/default.yaml` - 配置文件
- `tests/test_factor_health_monitor.py` - 测试用例
