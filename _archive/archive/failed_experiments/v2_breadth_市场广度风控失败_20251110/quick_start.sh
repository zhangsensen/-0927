#!/bin/bash
# ETF Rotation V2 - 快速开始脚本
# 用于系统性对比 baseline vs 市场广度 vs 综合版

set -e  # 遇到错误立即退出

PROJECT_ROOT="/Users/zhangshenshen/深度量化0927/etf_rotation_v2_breadth"
cd "$PROJECT_ROOT"

echo "========================================="
echo "  ETF Rotation V2 - 风控层实验"
echo "========================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. 验证环境
echo -e "${YELLOW}[1/5]${NC} 验证环境..."
if ! python3 -c "import numpy, pandas, yaml" 2>/dev/null; then
    echo -e "${RED}✗${NC} 依赖缺失，请先安装: pip install numpy pandas pyyaml"
    exit 1
fi
echo -e "${GREEN}✓${NC} 依赖检查通过"
echo ""

# 2. 集成测试
echo -e "${YELLOW}[2/5]${NC} 运行集成测试..."
if python3 test_risk_control.py 2>&1 | grep -q "测试总结.*✅ 通过"; then
    echo -e "${GREEN}✓${NC} 集成测试通过"
else
    echo -e "${RED}✗${NC} 集成测试失败，请检查 test_risk_control.py 输出"
    exit 1
fi
echo ""

# 3. 准备配置文件
echo -e "${YELLOW}[3/5]${NC} 准备配置文件..."

# 3.1 Baseline（无风控）
cat > configs/run_baseline.yaml << 'EOF'
run_id: "BASELINE_NO_RC"
output_root: "results/baseline_no_rc"

data:
  factor_root: "factor_output"
  etf_pool_name: "etf_pool_mid40"
  start: "2018-01-01"
  end: "2024-12-31"

cross_section:
  winsorize_quantiles: [0.025, 0.975]
  normalize_method: "z-score"

wfo:
  is_period: 252
  oos_period: 60
  step_size: 20
  warmup: 20
  factor_weighting: "ic_weighted"
  min_factor_ic: 0.012

# 无风控配置
EOF
echo -e "${GREEN}✓${NC} configs/run_baseline.yaml（无风控）"

# 3.2 市场广度版
cat > configs/run_market_breadth.yaml << 'EOF'
run_id: "V2_MARKET_BREADTH"
output_root: "results/v2_market_breadth"

data:
  factor_root: "factor_output"
  etf_pool_name: "etf_pool_mid40"
  start: "2018-01-01"
  end: "2024-12-31"

cross_section:
  winsorize_quantiles: [0.025, 0.975]
  normalize_method: "z-score"

wfo:
  is_period: 252
  oos_period: 60
  step_size: 20
  warmup: 20
  factor_weighting: "ic_weighted"
  min_factor_ic: 0.012

risk_control:
  market_breadth:
    enabled: true
    breadth_floor: 0.25
    score_threshold: 0.0
    defensive_scale: 0.5
    verbose: true
  
  volatility_target:
    enabled: false
  
  correlation_monitor:
    enabled: false
  
  combine_strategy: "min"
EOF
echo -e "${GREEN}✓${NC} configs/run_market_breadth.yaml（仅市场广度）"

# 3.3 综合版
cat > configs/run_comprehensive.yaml << 'EOF'
run_id: "V2_COMPREHENSIVE"
output_root: "results/v2_comprehensive"

data:
  factor_root: "factor_output"
  etf_pool_name: "etf_pool_mid40"
  start: "2018-01-01"
  end: "2024-12-31"

cross_section:
  winsorize_quantiles: [0.025, 0.975]
  normalize_method: "z-score"

wfo:
  is_period: 252
  oos_period: 60
  step_size: 20
  warmup: 20
  factor_weighting: "ic_weighted"
  min_factor_ic: 0.012

risk_control:
  market_breadth:
    enabled: true
    breadth_floor: 0.25
    score_threshold: 0.0
    defensive_scale: 0.5
    verbose: true
  
  volatility_target:
    enabled: true
    target_vol: 0.30
    min_window: 20
    max_scale: 1.0
    min_scale: 0.3
    verbose: true
  
  correlation_monitor:
    enabled: true
    corr_threshold: 0.65
    window: 20
    min_penalty: 0.5
    verbose: true
  
  combine_strategy: "multiply"
EOF
echo -e "${GREEN}✓${NC} configs/run_comprehensive.yaml（三模块全开）"
echo ""

# 4. 询问用户
echo -e "${YELLOW}[4/5]${NC} 选择运行模式:"
echo "  1) 仅测试（跳过完整回测）"
echo "  2) Baseline（无风控）"
echo "  3) 市场广度版（推荐）"
echo "  4) 综合版（三模块）"
echo "  5) 全部运行（baseline + 市场广度 + 综合）"
echo ""
read -p "请选择 [1-5]: " choice

case $choice in
    1)
        echo -e "${GREEN}✓${NC} 测试已完成，退出"
        exit 0
        ;;
    2)
        echo -e "${YELLOW}→${NC} 运行 Baseline..."
        python3 run_combo_wfo.py --config configs/run_baseline.yaml
        ;;
    3)
        echo -e "${YELLOW}→${NC} 运行市场广度版..."
        python3 run_combo_wfo.py --config configs/run_market_breadth.yaml
        ;;
    4)
        echo -e "${YELLOW}→${NC} 运行综合版..."
        python3 run_combo_wfo.py --config configs/run_comprehensive.yaml
        ;;
    5)
        echo -e "${YELLOW}→${NC} 运行全部版本（预计耗时15-30分钟）..."
        echo ""
        
        echo -e "${YELLOW}[1/3]${NC} Baseline..."
        python3 run_combo_wfo.py --config configs/run_baseline.yaml
        
        echo ""
        echo -e "${YELLOW}[2/3]${NC} 市场广度版..."
        python3 run_combo_wfo.py --config configs/run_market_breadth.yaml
        
        echo ""
        echo -e "${YELLOW}[3/3]${NC} 综合版..."
        python3 run_combo_wfo.py --config configs/run_comprehensive.yaml
        ;;
    *)
        echo -e "${RED}✗${NC} 无效选择"
        exit 1
        ;;
esac

echo ""
echo "========================================="
echo "  运行完成！"
echo "========================================="
echo ""

# 5. 生成对比报告
echo -e "${YELLOW}[5/5]${NC} 生成对比报告..."

python3 - << 'PYTHON_CODE'
import pandas as pd
from pathlib import Path

results_dir = Path("results")
versions = [
    ("baseline_no_rc", "Baseline（无风控）"),
    ("v2_market_breadth", "市场广度版"),
    ("v2_comprehensive", "综合版"),
]

print("\n" + "=" * 80)
print("  风控层对比报告")
print("=" * 80)
print()

for folder, name in versions:
    log_path = results_dir / folder / "wfo" / "risk_control_log.csv"
    wfo_path = results_dir / folder / "wfo" / "wfo_summary.csv"
    
    if not log_path.exists() and not wfo_path.exists():
        print(f"⏭️  {name}: 未运行")
        continue
    
    print(f"📊 {name}")
    print("-" * 80)
    
    # WFO指标
    if wfo_path.exists():
        wfo_df = pd.read_csv(wfo_path)
        print(f"  平均OOS IC: {wfo_df['oos_ic'].mean():.4f}")
        print(f"  平均IR: {wfo_df['oos_ir'].mean():.3f}")
        print(f"  正IC率: {wfo_df['positive_rate'].mean()*100:.1f}%")
    
    # 风控日志
    if log_path.exists():
        rc_df = pd.read_csv(log_path)
        triggered = rc_df[rc_df['final_scale'] < 1.0]
        
        if len(triggered) > 0:
            print(f"  触发防守: {len(triggered)}/{len(rc_df)} ({len(triggered)/len(rc_df)*100:.1f}%)")
            print(f"  平均缩仓: {(1 - triggered['final_scale'].mean())*100:.1f}%")
            print(f"  最低仓位: {triggered['final_scale'].min()*100:.0f}%")
            
            # 2020年危机期
            rc_df['date'] = pd.to_datetime(rc_df['date'])
            crisis = rc_df[(rc_df['date'] >= '2020-02-01') & (rc_df['date'] <= '2020-04-30')]
            if len(crisis) > 0:
                crisis_triggered = crisis[crisis['final_scale'] < 1.0]
                print(f"  2020危机触发: {len(crisis_triggered)}/{len(crisis)} ({len(crisis_triggered)/len(crisis)*100:.0f}%)")
                print(f"  危机平均仓位: {crisis['final_scale'].mean()*100:.0f}%")
        else:
            print("  无风控触发")
    else:
        print("  无风控日志")
    
    print()

print("=" * 80)
print("详细日志:")
for folder, name in versions:
    log_path = results_dir / folder / "wfo" / "risk_control_log.csv"
    if log_path.exists():
        print(f"  {name}: {log_path}")
print("=" * 80)

PYTHON_CODE

echo ""
echo -e "${GREEN}✓${NC} 全部完成！"
echo ""
echo "下一步:"
echo "  1. 查看风控日志: results/*/wfo/risk_control_log.csv"
echo "  2. 对比WFO结果: results/*/wfo/wfo_summary.csv"
echo "  3. 阅读指南: RISK_CONTROL_V2_GUIDE.md"
echo "  4. 如果市场广度效果好，考虑合并到主项目"
echo ""
