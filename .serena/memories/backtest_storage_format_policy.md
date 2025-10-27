# Backtest Results Storage Format Policy

## 核心原则
**所有回测结果必须使用Parquet格式存储，禁止使用Pickle (.pkl)**

## 理由
1. **跨平台兼容性**: Parquet是列式存储格式，跨语言、跨平台通用
2. **可读性**: 支持DuckDB直接查询，无需反序列化
3. **压缩效率**: 更好的压缩比，节省存储空间
4. **安全性**: 避免Pickle的代码执行漏洞
5. **性能**: 列式存储，查询特定列更快

## 实施规范
```python
# ✅ 正确做法
results_df.to_parquet('results.parquet', compression='zstd', engine='pyarrow')

# ❌ 禁止做法
import pickle
with open('results.pkl', 'wb') as f:
    pickle.dump(results, f)
```

## 文件命名规范
- 回测结果: `results_YYYYMMDD_HHMMSS.parquet`
- WFO结果: `wfo_YYYYMMDD_HHMMSS/results.parquet`
- Summary: `summary.json` (元数据可用JSON)

## 迁移清单
- [x] production_runner_optimized.py - 修改结果保存逻辑
- [ ] 已有.pkl文件转换脚本 (可选，历史数据)
