#!/usr/bin/env python3
"""
验证执行延迟修复的正确性

测试场景：
1. RB_EXECUTION_LAG=0：应该与原始回测结果一致（Lag-1 IC）
2. RB_EXECUTION_LAG=1：应该与 paper_trading 结果一致（Lag-2 IC）

简化版：直接运行 Platinum 策略的回测
"""

import os
import sys
from pathlib import Path
import subprocess

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_wfo_with_lag(execution_lag: int, label: str):
    """使用指定的 execution_lag 运行 WFO"""
    print(f"\n{'='*80}")
    print(f"测试场景: {label} (RB_EXECUTION_LAG={execution_lag})")
    print(f"{'='*80}\n")
    
    # 设置环境变量
    env = os.environ.copy()
    env["RB_EXECUTION_LAG"] = str(execution_lag)
    env["RB_DAILY_IC_PRECOMP"] = "0"  # 关闭预计算以简化
    
    # 运行 Platinum 策略的最小测试
    # Combo ID: 10813 (从 LOOKAHEAD_BIAS_DIAGNOSIS.md)
    # Factors: OBV_SLOPE_10D, PRICE_POSITION_20D, RSI_14, SLOPE_20D, VORTEX_14D
    # Lookback: 120, Rebalance: 2 (freq=2 天)
    
    cmd = [
        "python3",
        str(project_root / "run_combo_wfo.py"),
        "--lookback", "120",
        "--freq", "2",
        "--position", "10",
        "--combo-file", "/tmp/test_combo.txt",
        "--n-jobs", "1",
    ]
    
    # 创建临时组合文件
    test_combo_file = Path("/tmp/test_combo.txt")
    test_combo_file.write_text("OBV_SLOPE_10D,PRICE_POSITION_20D,RSI_14,SLOPE_20D,VORTEX_14D\n")
    
    print(f"运行命令: {' '.join(cmd)}")
    print(f"环境变量: RB_EXECUTION_LAG={execution_lag}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        
        print("\n--- 标准输出 ---")
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        
        if result.stderr:
            print("\n--- 标准错误 ---")
            print(result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)
        
        if result.returncode != 0:
            print(f"\n❌ 运行失败，退出码: {result.returncode}")
            return None
        
        # 解析结果（从输出中提取）
        lines = result.stdout.split("\n")
        annual_ret = None
        max_dd = None
        sharpe = None
        
        for line in lines:
            if "Annual Return" in line or "年化收益" in line:
                try:
                    annual_ret = float(line.split(":")[-1].strip().rstrip("%"))
                except:
                    pass
            if "Max Drawdown" in line or "最大回撤" in line:
                try:
                    max_dd = float(line.split(":")[-1].strip().rstrip("%"))
                except:
                    pass
            if "Sharpe" in line or "夏普" in line:
                try:
                    sharpe = float(line.split(":")[-1].strip())
                except:
                    pass
        
        return {
            "execution_lag": execution_lag,
            "label": label,
            "annual_return": annual_ret,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
        }
        
    except subprocess.TimeoutExpired:
        print("\n❌ 运行超时（5 分钟）")
        return None
    except Exception as e:
        print(f"\n❌ 运行错误: {e}")
        return None

def main():
    """主函数"""
    print("\n" + "="*80)
    print("执行延迟修复验证测试（使用 Platinum 策略）")
    print("="*80)
    
    # 测试 1: 原始逻辑（Lag-1 IC）
    print("\n🔍 测试 1/2: 原始逻辑（预期：高估收益，~20%）")
    result_lag0 = run_wfo_with_lag(execution_lag=0, label="原始逻辑 (Lag-1 IC，存在前视偏差)")
    
    # 测试 2: 延迟执行（Lag-2 IC）
    print("\n🔍 测试 2/2: 延迟执行（预期：真实收益，~-6% 到 1%）")
    result_lag1 = run_wfo_with_lag(execution_lag=1, label="延迟执行 (Lag-2 IC，无前视偏差)")
    
    # 对比结果
    print("\n" + "="*80)
    print("结果对比")
    print("="*80)
    
    if result_lag0 and result_lag1:
        print(f"\n{'场景':<30} {'年化收益':<15} {'夏普比率':<15} {'最大回撤':<15}")
        print("-" * 80)
        
        results = [result_lag0, result_lag1]
        for r in results:
            ann = f"{r['annual_return']:.2f}%" if r['annual_return'] is not None else "N/A"
            shp = f"{r['sharpe']:.4f}" if r['sharpe'] is not None else "N/A"
            dd = f"{r['max_drawdown']:.2f}%" if r['max_drawdown'] is not None else "N/A"
            print(f"{r['label']:<30} {ann:>12}  {shp:>12}  {dd:>12}")
        
        # 分析
        lag0_ret = result_lag0["annual_return"]
        lag1_ret = result_lag1["annual_return"]
        
        if lag0_ret is not None and lag1_ret is not None:
            print("\n" + "="*80)
            print("分析结论")
            print("="*80)
            
            print(f"\n1. 性能差异:")
            print(f"   - Lag-1 IC (原始): {lag0_ret:.2f}%")
            print(f"   - Lag-2 IC (修正): {lag1_ret:.2f}%")
            print(f"   - 收益差距: {lag0_ret - lag1_ret:.2f}% (Lag-1 高估)")
            
            if lag0_ret > lag1_ret + 2:  # 至少 2% 差距
                print(f"\n2. ✅ 验证成功:")
                print(f"   - Lag-1 IC 明显高估收益（存在前视偏差）")
                print(f"   - Lag-2 IC 反映真实执行延迟")
                print(f"   - 修复逻辑正确！")
            else:
                print(f"\n2. ⚠️  异常结果:")
                print(f"   - 预期 Lag-1 显著高于 Lag-2，但实际差距较小")
                print(f"   - 可能因子组合对延迟不敏感，或存在其他问题")
    else:
        print("\n❌ 无法完成对比：部分测试失败")
    
    print(f"\n3. 下一步:")
    print(f"   - 如验证成功，使用 RB_EXECUTION_LAG=1 重新训练完整 WFO")
    print(f"   - 所有新策略将基于 Lag-2 IC（无前视偏差）")
    print(f"   - 预期平均收益率会降低，但真实可交易")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
