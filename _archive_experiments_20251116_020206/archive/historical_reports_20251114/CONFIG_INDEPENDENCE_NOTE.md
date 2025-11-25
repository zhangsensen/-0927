# 🔒 配置文件独立性说明

**创建时间**: 2025-11-11  
**原因**: 防止 experiments 项目修改影响稳定的 optimized 项目

---

## 🔴 问题背景

### 原始设计（符号链接）
```bash
etf_rotation_experiments/configs/combo_wfo_config.yaml 
  -> ../../etf_rotation_optimized/configs/combo_wfo_config.yaml
```

**问题**: 
- experiments 项目的任何配置修改都会影响稳定项目
- 违反了"迭代实验不破坏稳定管线"的原则
- 用户设置了只读保护，阻止了错误修改

---

## ✅ 修正方案

### 1. 删除符号链接，创建独立配置
```bash
cd /Users/zhangshenshen/深度量化0927/etf_rotation_experiments/configs
rm combo_wfo_config.yaml
cp ../../etf_rotation_optimized/configs/combo_wfo_config.yaml combo_wfo_config.yaml
```

### 2. 修改 experiments 项目配置
```yaml
# experiments/configs/combo_wfo_config.yaml
backtest:
  commission_rate: 0.000005  # 万0.5（真实ETF场内交易费率）
  initial_capital: 1000000
```

### 3. 稳定项目配置保持不变
```yaml
# optimized/configs/combo_wfo_config.yaml
backtest:
  commission_rate: 0.00005  # 原始配置（万五，用于历史对比）
  initial_capital: 1000000
```

---

## 📊 当前状态

| 项目 | 配置文件 | 佣金率 | 状态 |
|------|---------|--------|------|
| **experiments** | 独立文件 | 0.000005 (万0.5) | ✅ 可修改 |
| **optimized** | 原始文件 | 0.00005 (万五) | 🔒 只读保护 |

---

## 🎯 设计原则

### 1. 稳定项目保护
- `etf_rotation_optimized/` 应设置为**只读**或**受保护分支**
- 任何修改必须经过严格审查
- 配置文件不应被 experiments 项目引用

### 2. 实验项目自由
- `etf_rotation_experiments/` 拥有**独立配置**
- 可以自由修改参数进行实验
- 不影响稳定项目的运行

### 3. 配置同步策略
- 稳定项目的配置作为**基线**
- experiments 项目从基线**复制**后独立修改
- 实验成功后，手动将配置合并回稳定项目

---

## 🔧 验证命令

### 检查配置独立性
```bash
# 查看 experiments 配置
head -3 /Users/zhangshenshen/深度量化0927/etf_rotation_experiments/configs/combo_wfo_config.yaml

# 查看 optimized 配置
head -3 /Users/zhangshenshen/深度量化0927/etf_rotation_optimized/configs/combo_wfo_config.yaml

# 确认不是符号链接
ls -la /Users/zhangshenshen/深度量化0927/etf_rotation_experiments/configs/combo_wfo_config.yaml
```

### 预期输出
```
# experiments: commission_rate: 0.000005
# optimized:   commission_rate: 0.00005
# 文件类型: -rw-r--r--（普通文件，非符号链接）
```

---

## 📝 经验教训

### AI 助手的错误
1. **未注意工作目录**: 应该在 experiments 项目工作，却修改了 optimized 项目
2. **未考虑符号链接**: 符号链接会导致修改传播到源文件
3. **未验证文件独立性**: 应该先检查文件是否为符号链接

### 用户的正确保护机制
1. **只读保护**: 对稳定项目设置只读权限
2. **及时发现**: 立即发现并阻止了错误操作
3. **清晰提醒**: 明确指出了问题所在

---

## 🚀 后续建议

### 1. 强化项目隔离
```bash
# 为稳定项目设置只读保护
chmod -R 444 /Users/zhangshenshen/深度量化0927/etf_rotation_optimized/configs/

# 或使用 Git 分支保护
git branch --set-upstream-to=origin/main optimized
git config branch.main.pushRemote no_push
```

### 2. 配置管理策略
- 稳定项目配置纳入版本控制
- experiments 项目配置添加到 `.gitignore`
- 使用配置模板 + 环境变量区分不同环境

### 3. 自动化检查
```bash
# 在 experiments 项目的 pre-commit hook 中检查
if [ -L configs/combo_wfo_config.yaml ]; then
    echo "❌ 错误: 配置文件不应为符号链接"
    exit 1
fi
```

---

**修正完成**: 2025-11-11  
**责任人**: AI 助手（错误操作） + 用户（及时纠正）  
**状态**: ✅ 已修复，两个项目配置已独立

