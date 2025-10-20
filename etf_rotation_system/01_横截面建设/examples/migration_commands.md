# 迁移命令示例

## 概述

从原始硬编码版本迁移到配置驱动版本的完整流程。

## 迁移步骤

### 1. 环境准备
```bash
# 检查当前目录结构
ls -la generate_panel*.py

# 确保配置文件存在
ls -la config/factor_panel_config.yaml
```

### 2. 备份原始文件
```bash
# 创建备份
python migrate_to_config.py --backup

# 手动备份（可选）
cp generate_panel.py generate_panel_original_backup.py
cp -r results results_backup
```

### 3. 验证新系统
```bash
# 验证配置文件语法
python migrate_to_config.py --validate

# 测试配置加载
python migrate_to_config.py --test

# 运行完整验证
python migrate_to_config.py --all
```

### 4. 功能等价性测试
```bash
# 运行等价性测试
python test_equivalence.py

# 查看测试结果
cat test_results.txt
```

### 5. 实际运行测试
```bash
# 使用新配置系统运行
python generate_panel_refactored.py --config config/factor_panel_config.yaml

# 检查输出结果
ls -la etf_rotation_system/data/results/panels/
```

### 6. 性能对比测试
```bash
# 测试原版本性能（如果还保留）
time python generate_panel.py

# 测试新版本性能
time python generate_panel_refactored.py --config config/factor_panel_config.yaml
```

## 高级迁移选项

### 自定义迁移配置
```bash
# 使用自定义配置文件进行迁移
python migrate_to_config.py --config config/custom_migration_config.yaml

# 指定不同的输出目录
python migrate_to_config.py --output-dir custom_migration_results
```

### 批量验证
```bash
# 验证多个配置文件
for config in config/*.yaml; do
    echo "验证配置文件: $config"
    python migrate_to_config.py --config "$config" --validate
done
```

## 迁移验证清单

### 文件完整性检查
```bash
# 检查必要文件是否存在
files=(
    "generate_panel_refactored.py"
    "config/factor_panel_config.yaml"
    "config/config_classes.py"
    "migrate_to_config.py"
    "test_equivalence.py"
)

for file in "${files[@]}"; do
    if [[ -f "$file" ]]; then
        echo "✅ $file 存在"
    else
        echo "❌ $file 缺失"
    fi
done
```

### 配置语法验证
```bash
# YAML语法检查
python -c "
import yaml
try:
    with open('config/factor_panel_config.yaml', 'r') as f:
        yaml.safe_load(f)
    print('✅ YAML语法正确')
except yaml.YAMLError as e:
    print(f'❌ YAML语法错误: {e}')
"
```

### 功能验证
```bash
# 运行快速功能测试
python -c "
from config.config_classes import FactorPanelConfig
try:
    config = FactorPanelConfig.from_yaml('config/factor_panel_config.yaml')
    print('✅ 配置加载成功')
    print(f'✅ 因子数量: {len([k for k, v in config.factor_enable.__dict__.items() if v])}')
    print(f'✅ 工作进程数: {config.processing.max_workers}')
except Exception as e:
    print(f'❌ 配置加载失败: {e}')
"
```

## 迁移结果对比

### 输出文件结构对比
```bash
# 原版本输出结构（如果存在）
tree results/ -L 2

# 新版本输出结构
tree etf_rotation_system/data/results/panels/ -L 2
```

### 结果文件对比
```bash
# 如果有原版本结果文件，可以进行对比
if [[ -f "results/panel.parquet" ]] && [[ -f "etf_rotation_system/data/results/panels/panel_"*"/panel.parquet" ]]; then
    echo "发现结果文件，可以进行对比分析"

    # 使用Python脚本对比结果
    python -c "
import pandas as pd
import glob

# 加载原版本结果
try:
    original = pd.read_parquet('results/panel.parquet')
    print(f'原版本数据形状: {original.shape}')
except:
    print('原版本结果文件不存在或无法读取')

# 加载新版本结果
new_files = glob.glob('etf_rotation_system/data/results/panels/*/panel.parquet')
if new_files:
    latest = max(new_files)
    try:
        new_result = pd.read_parquet(latest)
        print(f'新版本数据形状: {new_result.shape}')

        # 比较因子数量
        if 'original' in locals() and 'new_result' in locals():
            print(f'原版本因子数: {len(original.columns)}')
            print(f'新版本因子数: {len(new_result.columns)}')
    except Exception as e:
        print(f'新版本结果读取失败: {e}')
"
fi
```

## 常见迁移问题

### 配置文件问题
```bash
# 问题: 配置文件路径错误
# 解决: 检查配置文件是否存在
ls -la config/factor_panel_config.yaml

# 问题: 配置文件权限问题
# 解决: 修改文件权限
chmod 644 config/factor_panel_config.yaml
```

### 依赖问题
```bash
# 问题: 缺少必要的Python包
# 解决: 安装依赖
pip install pyyaml pandas numpy

# 问题: Python版本不兼容
# 解决: 检查Python版本
python --version  # 需要3.7+
```

### 数据路径问题
```bash
# 问题: 数据目录不存在
# 解决: 检查数据目录
ls -la "/Users/zhangshenshen/深度量化0927/raw/ETF/daily"

# 问题: 输出目录权限不足
# 解决: 创建输出目录
mkdir -p "/Users/zhangshenshen/深度量化0927/etf_rotation_system/data/results/panels"
```

## 迁移完成确认

### 最终验证
```bash
# 运行完整测试套件
python test_equivalence.py

# 检查测试输出
if [[ $? -eq 0 ]]; then
    echo "✅ 迁移成功！"
else
    echo "❌ 迁移过程中发现问题，请检查日志"
fi
```

### 清理临时文件
```bash
# 清理迁移过程中产生的临时文件
rm -f *.tmp
rm -f test_results.txt
rm -f validation_log.txt

# 保留备份文件（可选）
echo "备份文件保存在 deprecated/ 目录中"
```

## 回滚方案

如果迁移过程中遇到问题，需要回滚到原版本：

```bash
# 1. 恢复原始文件
cp deprecated/generate_panel.py .

# 2. 恢复原始配置（如果有）
# cp deprecated/old_config.yaml config/

# 3. 验证原版本功能
python generate_panel.py

# 4. 检查结果一致性
ls -la results/
```

## 迁移后续工作

### 文档更新
```bash
# 更新项目文档
echo "请更新以下文档："
echo "- README.md"
echo "- 使用说明文档"
echo "- 配置说明文档"
```

### 培训和交接
```bash
# 为团队成员提供培训材料
echo "培训要点："
echo "1. 配置文件的使用方法"
echo "2. 新命令行参数"
echo "3. 故障排除流程"
echo "4. 性能优化建议"
```

### 监控和维护
```bash
# 设置监控脚本
echo "建议创建以下监控脚本："
echo "- 配置文件变更监控"
echo "- 定期功能验证脚本"
echo "- 性能监控脚本"
echo "- 数据质量检查脚本"
```

通过以上完整的迁移流程，可以确保从硬编码版本到配置驱动版本的平滑过渡。
