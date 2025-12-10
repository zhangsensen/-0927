#!/usr/bin/env python3
"""
QMT 数据验证与修复脚本

功能:
1. 验证 QMT 数据质量
2. 检测与原始数据的差异
3. 修复复权因子问题（使用固定复权因子）
4. 生成对齐后的数据

使用方法:
    uv run python scripts/validate_qmt_data.py --check    # 仅检查
    uv run python scripts/validate_qmt_data.py --fix      # 检查并修复
    uv run python scripts/validate_qmt_data.py --compare  # 对比原始数据
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
QMT_DIR = PROJECT_ROOT / "raw/ETF/daily_qmt"
DAILY_DIR = PROJECT_ROOT / "raw/ETF/daily"
OUTPUT_DIR = PROJECT_ROOT / "raw/ETF/daily_qmt_fixed"


def load_etf_pool():
    """从配置文件加载策略使用的 ETF 池"""
    import yaml
    config_path = PROJECT_ROOT / "configs/etf_pools.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    etfs = []
    for pool in config.get("pools", {}).values():
        if "symbols" in pool:
            etfs.extend(pool["symbols"])
    return list(set(etfs))


def check_qmt_data():
    """检查 QMT 数据质量"""
    print("=" * 70)
    print("QMT 数据质量检查")
    print("=" * 70)
    
    qmt_files = list(QMT_DIR.glob("*.parquet"))
    print(f"找到 {len(qmt_files)} 个 QMT 数据文件")
    
    # 加载策略 ETF 池
    strategy_etfs = load_etf_pool()
    print(f"策略使用 {len(strategy_etfs)} 个 ETF")
    
    # 检查覆盖率
    qmt_codes = set()
    for f in qmt_files:
        code = f.stem.split("_")[0].split(".")[0]
        qmt_codes.add(code)
    
    missing = set(strategy_etfs) - qmt_codes
    if missing:
        print(f"⚠️ 缺少 {len(missing)} 个 ETF: {sorted(missing)}")
    else:
        print(f"✅ ETF 覆盖率 100%")
    
    # 检查每个文件的数据质量
    issues = []
    for f in qmt_files:
        code = f.stem.split("_")[0].split(".")[0]
        df = pd.read_parquet(f)
        
        # 检查必需列
        required_cols = ["trade_date", "open", "high", "low", "close", "vol", 
                        "adj_open", "adj_high", "adj_low", "adj_close", "adj_factor"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            issues.append(f"{code}: 缺少列 {missing_cols}")
        
        # 检查 ts_code
        if "ts_code" in df.columns:
            null_count = df["ts_code"].isnull().sum()
            if null_count > 0:
                issues.append(f"{code}: ts_code 有 {null_count}/{len(df)} 个空值")
        
        # 检查缺失值
        for col in ["close", "adj_close"]:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    issues.append(f"{code}: {col} 有 {null_count} 个缺失值")
    
    if issues:
        print(f"\n发现 {len(issues)} 个问题:")
        for issue in issues[:20]:
            print(f"  - {issue}")
        if len(issues) > 20:
            print(f"  ... 还有 {len(issues) - 20} 个问题")
    else:
        print(f"✅ 数据质量检查通过")
    
    return len(issues) == 0


def compare_with_original():
    """对比 QMT 数据与原始数据"""
    print("\n" + "=" * 70)
    print("QMT vs 原始数据对比")
    print("=" * 70)
    
    results = []
    strategy_etfs = load_etf_pool()
    
    for code in strategy_etfs:
        try:
            qmt_file = list(QMT_DIR.glob(f"{code}*_daily_*.parquet"))[0]
            daily_file = list(DAILY_DIR.glob(f"{code}*_daily_*.parquet"))[0]
        except IndexError:
            results.append({"ETF": code, "状态": "缺少文件"})
            continue
        
        qmt_df = pd.read_parquet(qmt_file)
        daily_df = pd.read_parquet(daily_file)
        
        # 设置日期索引
        qmt_df["trade_date"] = pd.to_datetime(qmt_df["trade_date"])
        daily_df["trade_date"] = pd.to_datetime(daily_df["trade_date"])
        qmt_df = qmt_df.set_index("trade_date").sort_index()
        daily_df = daily_df.set_index("trade_date").sort_index()
        
        # 共同日期
        common_idx = qmt_df.index.intersection(daily_df.index)
        
        if len(common_idx) == 0:
            results.append({"ETF": code, "状态": "无共同日期"})
            continue
        
        # 计算收益率差异
        qmt_ret = qmt_df.loc[common_idx, "adj_close"].pct_change().dropna()
        daily_ret = daily_df.loc[common_idx, "adj_close"].pct_change().dropna()
        
        ret_diff = (qmt_ret - daily_ret).abs()
        max_diff_bp = ret_diff.max() * 10000
        mean_diff_bp = ret_diff.mean() * 10000
        
        # 累计收益差异
        qmt_cum = (1 + qmt_ret).prod() - 1
        daily_cum = (1 + daily_ret).prod() - 1
        cum_diff_pct = (qmt_cum - daily_cum) * 100
        
        results.append({
            "ETF": code,
            "共同天数": len(common_idx),
            "日收益max差(bp)": f"{max_diff_bp:.1f}",
            "日收益mean差(bp)": f"{mean_diff_bp:.2f}",
            "累计差(%)": f"{cum_diff_pct:.3f}",
            "相关性": f"{qmt_ret.corr(daily_ret):.4f}",
            "状态": "OK" if max_diff_bp < 1 else "有差异"
        })
    
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    
    # 统计
    ok_count = len([r for r in results if r.get("状态") == "OK"])
    diff_count = len([r for r in results if r.get("状态") == "有差异"])
    print(f"\n统计: {ok_count} 个完全一致, {diff_count} 个有差异")
    
    return results


def fix_adj_factor(output_dir: Path = OUTPUT_DIR):
    """
    修复 QMT 数据的 ts_code 问题
    
    QMT 原始数据的复权因子是正确的（后复权），只需要修复 ts_code 为 None 的问题。
    
    注意：
    - QMT 数据使用后复权，adj_factor 在分红/拆股日会突变
    - 这是正确的行为，不需要修改
    - 只修复 ts_code 列
    """
    print("\n" + "=" * 70)
    print("修复 QMT ts_code 问题")
    print("=" * 70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    qmt_files = list(QMT_DIR.glob("*.parquet"))
    fixed_count = 0
    
    for f in qmt_files:
        df = pd.read_parquet(f)
        
        # 只修复 ts_code
        full_code = f.stem.split("_")[0]  # e.g., 510300.SH
        df["ts_code"] = full_code
        
        # 保存（不改变其他数据）
        output_file = output_dir / f.name
        df.to_parquet(output_file, index=False)
        fixed_count += 1
    
    print(f"\n✅ 修复完成: {fixed_count}/{len(qmt_files)} 个文件")
    print(f"输出目录: {output_dir}")
    print(f"\n注意：只修复了 ts_code，复权数据保持原样")
    
    return output_dir


def validate_fixed_data(fixed_dir: Path = OUTPUT_DIR):
    """验证修复后的数据"""
    print("\n" + "=" * 70)
    print("验证修复后的数据")
    print("=" * 70)
    
    if not fixed_dir.exists():
        print(f"⚠️ 修复目录不存在: {fixed_dir}")
        return False
    
    files = list(fixed_dir.glob("*.parquet"))
    if not files:
        print(f"⚠️ 修复目录为空")
        return False
    
    all_ok = True
    for f in files[:5]:  # 抽查前5个
        code = f.stem.split("_")[0]
        df = pd.read_parquet(f)
        
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.set_index("trade_date").sort_index()
        
        # 验证收益率一致性
        ret_close = df["close"].pct_change()
        ret_adj = df["adj_close"].pct_change()
        ret_diff = (ret_close - ret_adj).abs().max()
        
        if ret_diff > 1e-10:
            print(f"  ❌ {code}: 收益率差异 {ret_diff:.2e}")
            all_ok = False
        else:
            print(f"  ✅ {code}: 收益率完全一致")
        
        # 验证 ts_code
        if df["ts_code"].isnull().any():
            print(f"  ❌ {code}: ts_code 有空值")
            all_ok = False
    
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="QMT 数据验证与修复")
    parser.add_argument("--check", action="store_true", help="检查数据质量")
    parser.add_argument("--compare", action="store_true", help="对比原始数据")
    parser.add_argument("--fix", action="store_true", help="修复复权因子")
    parser.add_argument("--validate", action="store_true", help="验证修复后的数据")
    parser.add_argument("--all", action="store_true", help="执行所有步骤")
    
    args = parser.parse_args()
    
    if not any([args.check, args.compare, args.fix, args.validate, args.all]):
        args.all = True
    
    if args.check or args.all:
        check_qmt_data()
    
    if args.compare or args.all:
        compare_with_original()
    
    if args.fix or args.all:
        fix_adj_factor()
    
    if args.validate or args.all:
        validate_fixed_data()


if __name__ == "__main__":
    main()
