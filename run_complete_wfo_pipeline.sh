#!/bin/bash
# 完整WFO生产流程：因子生成→筛选→WFO回测
# 使用修复后的窗口配置

set -e  # 遇到错误立即退出

PROJECT_ROOT="/Users/zhangshenshen/深度量化0927"
cd "$PROJECT_ROOT"

echo "================================================================================"
echo "🚀 完整WFO生产流程 - 窗口配置已修复"
echo "================================================================================"
echo ""
echo "修复内容:"
echo "  ✅ TopN参数bug已修复（纯排名筛选）"
echo "  ✅ 窗口参数已扩展（短中长期窗口）"
echo ""
echo "流程配置:"
echo "  • 步骤1: 因子面板生成（~50+因子）"
echo "  • 步骤2: 因子筛选（取前15个）"
echo "  • 步骤3: WFO回测（2.3M组合）"
echo ""
echo "窗口配置更新:"
echo "  • price_position: [20, 60, 120] (原[60])"
echo "  • momentum: [20, 63, 126, 252] (原[63, 252])"
echo "  • volume_ratio: [5, 20, 60] (原[20])"
echo "  • volatility: [20, 60, 120] (原[120])"
echo "  • drawdown: [63, 126] (原[126])"
echo ""
echo "================================================================================"

# 步骤1: 生成因子面板
echo ""
echo "📊 步骤1/3: 生成因子面板（使用新窗口配置）"
echo "================================================================================"
cd "$PROJECT_ROOT/etf_rotation_system/01_横截面建设"

python3 generate_panel_refactored.py \
    --data-dir ../../raw/ETF/daily \
    --output-dir ../data/results/panels \
    --config config/factor_panel_config.yaml \
    --workers 8

if [ $? -ne 0 ]; then
    echo "❌ 因子面板生成失败"
    exit 1
fi

# 获取最新面板路径
LATEST_PANEL=$(ls -td ../data/results/panels/panel_* | head -1)
echo "✅ 面板生成完成: $LATEST_PANEL"

# 显示因子数量
python3 << 'EOF'
import pandas as pd
from pathlib import Path
import json

panel_dirs = sorted(Path("../data/results/panels").glob("panel_*"))
if panel_dirs:
    latest = panel_dirs[-1]
    meta_file = latest / "panel_meta.json"
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
        print(f"\n  生成因子数: {len(meta.get('factors', []))}")
        print(f"  ETF数量: {len(meta.get('symbols', []))}")
        print(f"  数据期间: {meta.get('date_range', {}).get('start')} ~ {meta.get('date_range', {}).get('end')}")
        
        # 检查新增的短窗口因子
        factors = meta.get('factors', [])
        short_window_factors = [f for f in factors if any(w in f for w in ['_20D', '_5D'])]
        if short_window_factors:
            print(f"\n  新增短窗口因子 ({len(short_window_factors)}个):")
            for f in sorted(short_window_factors)[:10]:  # 显示前10个
                print(f"    • {f}")
EOF

# 步骤2: 因子筛选
echo ""
echo "🔬 步骤2/3: 因子筛选（自然筛选，无强制保留）"
echo "================================================================================"
cd "$PROJECT_ROOT/etf_rotation_system/02_因子筛选"

# 更新筛选配置使用最新面板
echo "  更新筛选配置..."
python3 << 'EOF'
import yaml
from pathlib import Path

# 获取最新面板路径（绝对路径）
PROJECT_ROOT = Path("/Users/zhangshenshen/深度量化0927")
panel_dirs = sorted((PROJECT_ROOT / "etf_rotation_system/data/results/panels").glob("panel_*"))

if panel_dirs:
    latest_panel = panel_dirs[-1]
    panel_file = latest_panel / "panel.parquet"
    
    # 读取筛选配置
    config_file = Path("optimized_screening_config.yaml")
    with open(config_file) as f:
        config = yaml.safe_load(f)
    
    # 更新为绝对路径
    config['data_source']['panel_file'] = str(panel_file)
    
    # 保存更新后的配置
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"  ✅ 筛选配置已更新: {panel_file}")
else:
    print("  ❌ 未找到面板文件")
    exit(1)
EOF

python3 run_etf_cross_section_configurable.py \
    --config optimized_screening_config.yaml

if [ $? -ne 0 ]; then
    echo "❌ 因子筛选失败"
    exit 1
fi

# 获取最新筛选结果
LATEST_SCREENING=$(ls -td ../data/results/screening/screening_* | head -1)
echo "✅ 因子筛选完成: $LATEST_SCREENING"

# 显示筛选结果
echo ""
echo "筛选结果预览:"
python3 << 'EOF'
import pandas as pd
from pathlib import Path

screening_dirs = sorted(Path("../data/results/screening").glob("screening_*"))
if screening_dirs:
    latest = screening_dirs[-1]
    passed = pd.read_csv(latest / "passed_factors.csv")
    print(f"  通过因子数: {len(passed)}")
    print("\n  Top 10因子:")
    for idx, row in passed.head(10).iterrows():
        print(f"    {idx+1:2d}. {row['factor']:35s}  IC={row['ic_mean']:+.4f}  IR={row['ic_ir']:+.4f}  t={row['ic_t_stat']:+.2f}")
    
    # 检查是否包含短窗口因子
    short_window = passed[passed['factor'].str.contains('_20D|_5D', regex=True)]
    if not short_window.empty:
        print(f"\n  ✅ 包含短窗口因子 ({len(short_window)}个):")
        for idx, row in short_window.head(5).iterrows():
            print(f"    • {row['factor']:35s}  IC={row['ic_mean']:+.4f}")
    else:
        print(f"\n  ⚠️  未包含短窗口因子")
EOF

# 步骤3: WFO回测
echo ""
echo "⚡ 步骤3/3: WFO大规模回测（2.3M组合）"
echo "================================================================================"
cd "$PROJECT_ROOT/etf_rotation_system/03_vbt_wfo"

# 更新WFO配置使用最新筛选结果
echo "  更新WFO配置..."
python3 << 'EOF'
import yaml
from pathlib import Path

# 读取当前配置
config_file = Path("simple_config.yaml")
with open(config_file) as f:
    config = yaml.safe_load(f)

# 获取最新筛选结果
screening_dirs = sorted(Path("../data/results/screening").glob("screening_*"))
if screening_dirs:
    latest = screening_dirs[-1]
    screening_file = str(latest / "passed_factors.csv")
    config['screening_file'] = screening_file
    
    # 保存更新后的配置
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"  ✅ WFO配置已更新: {screening_file}")
EOF

# 运行WFO回测
python3 production_runner_optimized.py

if [ $? -ne 0 ]; then
    echo "❌ WFO回测失败"
    exit 1
fi

# 获取最新WFO结果
LATEST_WFO=$(ls -td ../data/results/vbtwfo/wfo_* | head -1)
echo "✅ WFO回测完成: $LATEST_WFO"

# 分析结果
echo ""
echo "================================================================================"
echo "📊 完整流程执行完成"
echo "================================================================================"
echo ""
echo "结果目录:"
echo "  • 因子面板: $LATEST_PANEL"
echo "  • 因子筛选: $LATEST_SCREENING"
echo "  • WFO回测: $LATEST_WFO"
echo ""

echo "🏆 WFO结果分析:"
python3 << 'EOF'
import pandas as pd
from pathlib import Path

wfo_dirs = sorted(Path("../data/results/vbtwfo").glob("wfo_*"))
if wfo_dirs:
    latest = wfo_dirs[-1]
    results_file = latest / "results.parquet"
    
    if results_file.exists():
        df = pd.read_parquet(results_file)
        
        # 整体统计
        print(f"\n  总策略数: {len(df):,}")
        print(f"  IS阶段: Sharpe={df['is_sharpe'].mean():.4f} ± {df['is_sharpe'].std():.4f}")
        print(f"  OOS阶段: Sharpe={df['oos_sharpe'].mean():.4f} ± {df['oos_sharpe'].std():.4f}")
        
        # TopN效果分析
        if 'top_n' in df.columns:
            print(f"\n  TopN参数效果:")
            topn_stats = df.groupby('top_n').agg({
                'is_sharpe': 'mean',
                'oos_sharpe': 'mean'
            }).round(4)
            for top_n, row in topn_stats.iterrows():
                print(f"    TopN={top_n}: IS={row['is_sharpe']:+.4f}, OOS={row['oos_sharpe']:+.4f}")
        
        # Top 5策略
        print(f"\n  🥇 Top 5策略（按IS Sharpe排序）:")
        top5 = df.nlargest(5, 'is_sharpe')
        for idx, (_, row) in enumerate(top5.iterrows(), 1):
            print(f"    {idx}. IS={row['is_sharpe']:+.4f} OOS={row['oos_sharpe']:+.4f} "
                  f"TopN={int(row.get('top_n', 0))} Reb={int(row.get('rebalance_freq', 0))}d")
        
        # 检查过拟合
        decay = ((df['is_sharpe'] - df['oos_sharpe']) / (df['is_sharpe'].abs() + 1e-6) * 100)
        severe_overfit = (decay > 50).sum()
        print(f"\n  ⚠️  过拟合检查:")
        print(f"    严重过拟合(衰减>50%): {severe_overfit:,}个 ({severe_overfit/len(df)*100:.1f}%)")
        print(f"    平均衰减: {decay.mean():.1f}%")
    else:
        print(f"\n  ⚠️  未找到results.parquet文件")
EOF

echo ""
echo "================================================================================"
echo "✅ 全部完成！"
echo "================================================================================"
