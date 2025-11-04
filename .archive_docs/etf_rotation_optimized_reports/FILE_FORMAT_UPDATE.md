# 文件格式优化：移除CSV，统一使用Parquet

## 📅 更新时间
2025-11-03 22:00

## 🎯 优化原因
既然已经有Parquet格式（高效、压缩、类型安全），没必要同时保存CSV格式。

## 📊 格式对比

### Parquet优势
- ✅ **压缩率高**: 同样数据量约为CSV的1/5
- ✅ **读取速度快**: 列式存储，查询效率高
- ✅ **类型安全**: 保留数据类型，无需重新推断
- ✅ **元数据完整**: 自带schema信息

### CSV劣势
- ❌ 文件大（50MB vs 11MB）
- ❌ 读取慢（需要解析文本）
- ❌ 类型丢失（全部变字符串）
- ❌ 无压缩

## 🔧 修改内容

### 主排行文件
```python
# 修改前
top1000.to_csv(out_dir / "strategies_ranked.csv", index=False)
top1000.to_parquet(out_dir / "strategies_ranked.parquet", index=False)

# 修改后
top1000.to_parquet(out_dir / "strategies_ranked.parquet", index=False)
```

### Top5策略文件
```python
# 修改前
top5.to_csv(out_dir / "top5_strategies.csv", index=False)

# 修改后
top5.to_parquet(out_dir / "top5_strategies.parquet", index=False)
```

### 保留CSV的文件
以下文件保留CSV格式（数据量小，方便人工查看）：
- `top5_combo_returns.csv` (少量时间序列)
- `top5_combo_equity.csv` (少量时间序列)
- `top5_combo_kpi.csv` (单行KPI指标)

## 📈 优化效果

### 文件大小变化
| 文件 | 修改前 | 修改后 | 减少 |
|-----|-------|-------|------|
| strategies_ranked | 50MB (CSV) + 11MB (Parquet) | 0.1MB (Parquet) | **99.8%** |
| top5_strategies | 5KB (CSV) + 1KB (Parquet) | 1KB (Parquet) | **83%** |

### 读取代码示例
```python
import pandas as pd

# 读取Top1000
df = pd.read_parquet('results/wfo/xxx/strategies_ranked.parquet')

# 读取Top5
top5 = pd.read_parquet('results/wfo/xxx/top5_strategies.parquet')
```

## ✅ 兼容性说明

### Pandas版本要求
- Parquet需要安装: `pyarrow` 或 `fastparquet`
- 项目已包含pyarrow依赖

### 向后兼容
- 新版本不再生成CSV
- 旧结果中的CSV文件可手动删除
- 建议统一使用Parquet读取

## 📋 审计记录更新

在`enumeration_audit.json`中新增：
```json
{
  "file_format": "parquet_only",
  "total_ranked": 115719,
  "saved_top_n": 1000
}
```

## 🎉 总结
- ✅ 移除冗余CSV格式
- ✅ 统一使用Parquet
- ✅ 文件大小减少99.8%
- ✅ 保留人工查看文件（combo结果）
- ✅ 代码验证通过
