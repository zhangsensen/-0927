# ETF轮动系统 - 文档索引

**最后更新**: 2025-10-27  
**版本**: v2.0-optimized

---

## 📚 文档导航

### 🚀 快速开始

| 文档 | 用途 | 阅读时间 |
|------|------|---------|
| **README.md** | 项目概览和快速开始 | 5分钟 |
| **QUICK_START_GUIDE.md** | 详细快速开始指南 | 15分钟 |

### 📖 深度理解

| 文档 | 用途 | 阅读时间 |
|------|------|---------|
| **PROJECT_SUMMARY.md** | 项目总结（一句话+核心机制） | 10分钟 |
| **PROJECT_ARCHITECTURE.md** | 完整架构和设计说明 | 30分钟 |
| **CORE_ALGORITHMS.md** | 核心算法详解（IC、标准化、WFO等） | 45分钟 |

### 🔧 实践指南

| 文档 | 用途 | 阅读时间 |
|------|------|---------|
| **PROJECT_GUIDELINES.md** | 项目规范和开发指南 | 20分钟 |
| **WFO_EXPERIMENTS_GUIDE.md** | WFO实验和参数调优 | 30分钟 |

---

## 📋 按用途分类

### 对于新用户

**目标**: 快速理解项目并运行

**推荐阅读顺序**:
1. README.md (5分钟)
2. QUICK_START_GUIDE.md (15分钟)
3. PROJECT_SUMMARY.md (10分钟)

**总耗时**: 30分钟

### 对于开发者

**目标**: 深入理解架构并进行开发

**推荐阅读顺序**:
1. PROJECT_SUMMARY.md (10分钟)
2. PROJECT_ARCHITECTURE.md (30分钟)
3. CORE_ALGORITHMS.md (45分钟)
4. PROJECT_GUIDELINES.md (20分钟)

**总耗时**: 105分钟

### 对于研究员

**目标**: 理解因子和回测逻辑

**推荐阅读顺序**:
1. PROJECT_SUMMARY.md (10分钟)
2. CORE_ALGORITHMS.md (45分钟)
3. WFO_EXPERIMENTS_GUIDE.md (30分钟)

**总耗时**: 85分钟

### 对于大模型

**目标**: 完整理解项目用于代码生成

**推荐阅读顺序**:
1. PROJECT_SUMMARY.md (10分钟)
2. PROJECT_ARCHITECTURE.md (30分钟)
3. CORE_ALGORITHMS.md (45分钟)
4. PROJECT_GUIDELINES.md (20分钟)

**总耗时**: 105分钟

---

## 📄 文档详解

### README.md

**内容**:
- 项目概览
- 核心特性
- 快速开始
- 项目结构
- 配置说明
- 性能指标
- CLI命令
- 风险提示

**适合**: 所有用户

---

### QUICK_START_GUIDE.md

**内容**:
- 环境配置
- 安装步骤
- 数据准备
- 运行流程
- 结果解读
- 常见问题
- 调试技巧

**适合**: 新用户

---

### PROJECT_SUMMARY.md

**内容**:
- 项目一句话总结
- 核心价值
- 系统架构
- 数据流
- 12个精选因子
- 6大核心机制
- 3步工作流程
- 关键参数
- 性能指标
- 与原版本对比
- 质量保证
- 设计原则

**适合**: 所有用户

---

### PROJECT_ARCHITECTURE.md

**内容**:
- 项目概览（规模、哲学）
- 核心架构（分层、模块）
- 数据流（完整流程）
- 6大核心机制详解
- 7个模块详解
- 3步工作流程
- 配置系统
- 性能优化
- 质量保证
- 快速参考

**适合**: 开发者、大模型

---

### CORE_ALGORITHMS.md

**内容**:
- IC计算算法
- 标准化算法
- 极值截断算法
- FDR校正算法
- WFO窗口划分
- 因子选择算法
- 性能优化
- 关键参数参考

**适合**: 研究员、开发者、大模型

---

### PROJECT_GUIDELINES.md

**内容**:
- 开发规范
- 代码风格
- 测试要求
- 文档要求
- 提交规范
- 性能要求
- 安全要求

**适合**: 开发者

---

### WFO_EXPERIMENTS_GUIDE.md

**内容**:
- WFO框架原理
- 参数说明
- 实验设计
- 结果分析
- 常见问题
- 最佳实践

**适合**: 研究员、开发者

---

## 🎯 常见问题对应文档

### Q: 如何快速开始?
**答**: 阅读 README.md 和 QUICK_START_GUIDE.md

### Q: 项目的整体架构是什么?
**答**: 阅读 PROJECT_ARCHITECTURE.md

### Q: 12个因子是如何计算的?
**答**: 阅读 PROJECT_SUMMARY.md 的因子清单，详细见 CORE_ALGORITHMS.md

### Q: IC是如何计算的?
**答**: 阅读 CORE_ALGORITHMS.md 的 IC计算算法

### Q: WFO是如何工作的?
**答**: 阅读 CORE_ALGORITHMS.md 的 WFO窗口划分，详细见 WFO_EXPERIMENTS_GUIDE.md

### Q: 如何修改参数?
**答**: 阅读 PROJECT_SUMMARY.md 的关键参数，详细见 WFO_EXPERIMENTS_GUIDE.md

### Q: 如何添加新因子?
**答**: 阅读 PROJECT_GUIDELINES.md

### Q: 如何调试问题?
**答**: 阅读 QUICK_START_GUIDE.md 的调试技巧

### Q: 代码规范是什么?
**答**: 阅读 PROJECT_GUIDELINES.md

### Q: 性能如何优化?
**答**: 阅读 PROJECT_ARCHITECTURE.md 的性能优化

---

## 📊 文档关系图

```
README.md (入口)
    ↓
QUICK_START_GUIDE.md (快速开始)
    ↓
PROJECT_SUMMARY.md (总体理解)
    ↙         ↓         ↘
  开发者    研究员    大模型
    ↓         ↓         ↓
PROJECT_   CORE_    PROJECT_
GUIDELINES ALGORITHMS ARCHITECTURE
    ↓         ↓         ↓
    └─────────┴─────────┘
         ↓
WFO_EXPERIMENTS_GUIDE.md (深度实践)
```

---

## 🔍 按主题索引

### 架构设计

- PROJECT_ARCHITECTURE.md - 完整架构
- PROJECT_SUMMARY.md - 系统架构图

### 数据处理

- CORE_ALGORITHMS.md - 标准化、极值截断
- PROJECT_ARCHITECTURE.md - 数据流

### 因子系统

- PROJECT_SUMMARY.md - 12个精选因子
- CORE_ALGORITHMS.md - 因子选择算法
- PROJECT_GUIDELINES.md - 因子开发规范

### 回测框架

- CORE_ALGORITHMS.md - WFO窗口划分
- WFO_EXPERIMENTS_GUIDE.md - WFO实验指南
- PROJECT_ARCHITECTURE.md - WFO机制

### 性能优化

- PROJECT_ARCHITECTURE.md - 性能优化策略
- PROJECT_GUIDELINES.md - 性能要求

### 质量保证

- PROJECT_ARCHITECTURE.md - 质量保证
- PROJECT_GUIDELINES.md - 测试要求

### 开发指南

- PROJECT_GUIDELINES.md - 开发规范
- QUICK_START_GUIDE.md - 调试技巧

---

## 📈 学习路径

### 初级 (新用户)

```
README.md
    ↓
QUICK_START_GUIDE.md
    ↓
PROJECT_SUMMARY.md
```

**目标**: 能够运行系统，理解基本概念

---

### 中级 (开发者)

```
PROJECT_SUMMARY.md
    ↓
PROJECT_ARCHITECTURE.md
    ↓
PROJECT_GUIDELINES.md
```

**目标**: 能够修改代码，添加新功能

---

### 高级 (研究员/大模型)

```
PROJECT_SUMMARY.md
    ↓
PROJECT_ARCHITECTURE.md
    ↓
CORE_ALGORITHMS.md
    ↓
WFO_EXPERIMENTS_GUIDE.md
```

**目标**: 深入理解算法，进行高级研究

---

## 🎓 推荐阅读时间

| 角色 | 初级 | 中级 | 高级 |
|------|------|------|------|
| 新用户 | 30分钟 | - | - |
| 开发者 | 30分钟 | 70分钟 | 120分钟 |
| 研究员 | 30分钟 | 60分钟 | 120分钟 |
| 大模型 | 30分钟 | 70分钟 | 120分钟 |

---

## 📞 获取帮助

### 常见问题

查看各文档的 "常见问题" 部分

### 调试技巧

见 QUICK_START_GUIDE.md

### 开发指南

见 PROJECT_GUIDELINES.md

### 实验指南

见 WFO_EXPERIMENTS_GUIDE.md

---

## ✅ 文档检查清单

- ✅ README.md - 快速开始
- ✅ QUICK_START_GUIDE.md - 详细指南
- ✅ PROJECT_SUMMARY.md - 项目总结
- ✅ PROJECT_ARCHITECTURE.md - 完整架构
- ✅ CORE_ALGORITHMS.md - 核心算法
- ✅ PROJECT_GUIDELINES.md - 项目规范
- ✅ WFO_EXPERIMENTS_GUIDE.md - 实验指南
- ✅ INDEX.md - 文档索引

---

**版本**: v2.0-optimized  
**最后更新**: 2025-10-27  
**维护者**: ETF Rotation System Team
