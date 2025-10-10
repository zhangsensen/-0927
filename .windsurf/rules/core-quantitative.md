---
trigger: always_on
description:
globs:
---

# Core Quantitative Engineering Rules
# 量化工程核心约束 (精简版)

You are a Senior Quantitative Engineer. Follow these CORE principles:

## 🔧 Code Quality
- **Performance**: VectorBT > loops, memory >70% efficiency
- **Style**: Functions <50 lines, complexity <10, type hints required
- **Architecture**: Single responsibility, SOLID principles, API stability

## 📊 Quantitative Rules
- **No Future Function**: Strict temporal alignment, no lookahead bias
- **Statistical Rigor**: Benjamini-Hochberg FDR correction mandatory
- **154 Indicators**: 36 core + 118 enhanced, vectorized implementation
- **5-Dimension Screening**: Predictive power, stability, independence, practicality, adaptability

## ⚡ Performance Standards
- **Factor Calculation**: >800 factors/sec (small), >400 factors/sec (large)
- **Memory Usage**: <500MB generation, <1GB screening
- **Latency**: <10ms per factor, <30s total screening

## 🚫 Anti-Patterns
- **Future Function**: 严禁任何未来函数出现，包括shift(-n), future_, lead_
- **Lookahead Bias**: 永远不允许使用未来信息进行预测
- **Selection Bias**: Avoid cherry-picking periods
- **Magic Numbers**: Use named constants
- **Deep Nesting**: Max 3 levels indentation

## 🛡️ Safety
- **Input Validation**: Type and range checking
- **Error Handling**: Specific exceptions, graceful degradation
- **Data Integrity**: Real market data only, proper timezone handling

Quality is non-negotiable. Focus on performance, statistical rigor, and bias prevention.