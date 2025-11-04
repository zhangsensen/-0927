# 先验加权实验（研究分支）

**状态**: 🔴 **研究级** - 不建议生产使用  
**ROI**: 低（IC提升13.2%但p=0.49不显著）  
**日期**: 2025-10-29

---

## 结论

### 性能
- IC提升: +0.0021 (+13.2%)
- 统计显著性: p=0.49 ❌
- 效应量: Cohen's d=0.12（小效应）
- 负IC窗口改善率: 77.8% ✅

### 定位
**防御性工具**，不是进攻性工具：
- 在IS信号失效时（负IC窗口）有改善
- 在IS信号有效时反而干扰IC加权
- 更像"稳健性增强器"

### 决策
- ❌ 不建议生产使用（ROI太低）
- ✅ 保留研究分支并行跟踪
- ⏳ 等待更多窗口数据（需1138窗口达80%功效，当前仅36窗口）

---

## 文件清单

### 脚本
- `generate_prior_contributions.py` - 生成先验贡献数据
- `generate_stability_prior.py` - 生成纯稳定性先验
- `validate_prior_weighted.py` - 基础统计验证
- `deep_analysis_prior.py` - 深度分析
- `robust_statistical_tests.py` - 稳健统计检验
- `go_nogo_decision.py` - Go/No-Go决策

### 配置
- `prior_contributions.yaml` - 先验贡献数据
- `prior_weighted.yaml` - 先验加权配置

### 报告
- `STAGE3_VALIDATION_REPORT.md` - 完整验证报告
- `PRIOR_WEIGHTING_OPTIMIZATION_ROADMAP.md` - 优化路线图
- `STAGE3_FINAL_EXECUTIVE_SUMMARY.md` - 执行摘要

### 结果
- `prior_weighted_validation.csv` - 36窗口详细对比
- `prior_weighted_analysis.png` - 可视化图表

---

## 核心发现

### 1. 先验加权的本质
- **防御性工具**: 在负IC窗口改善率77.8%
- **条件有效性**: 仅在市场regime变化时有用
- **最优策略**: 自适应混合（IS质量低时用先验，高时用IC）

### 2. 当前问题
- **方差过大**: 增加20.9%，降低统计功效
- **样本不足**: 需1138窗口，当前仅36窗口
- **先验构造**: 强度带来噪声，应改为纯稳定性

### 3. 优化方向（如继续投入）
1. **家族收缩先验**: 因子按家族聚类，降噪
2. **纯稳定性先验**: 去掉强度，只用稳定性
3. **自适应混合**: 基于IS质量动态切换
4. **正则融合**: final_w = (1-λ)·prior + λ·ic_weighted

---

## 上线门槛（未达标）

### 统计门槛 ❌
- Wilcoxon p<0.10 (当前0.387)
- 正向时期≥3/4 (当前1/4)

### 稳健门槛 ❌
- 胜率≥60% (当前52.8%)
- 损失不对称≤0.50 (当前0.79)

### 实盘门槛 ⏳
- 成本后回测待验证

---

## 使用说明

### 生成先验
```bash
cd research/prior_weighting_experiment
python generate_prior_contributions.py
```

### 验证
```bash
python validate_prior_weighted.py
python robust_statistical_tests.py
python go_nogo_decision.py
```

### 深度分析
```bash
python deep_analysis_prior.py
```

---

## 不建议生产的原因

1. **统计不显著**: 所有检验p>0.10
2. **ROI太低**: 投入产出比不划算
3. **复杂度增加**: 引入额外维护成本
4. **风险未知**: 成本后回测未验证

---

## 保留价值

1. **研究线索**: 负IC窗口改善率高，值得深挖
2. **方法论**: 验证流程完整，可复用
3. **未来潜力**: 样本增加或优化后可能有效

---

**建议**: 生产继续用ic_weighted，此分支并行跟踪，每月更新验证报告。
