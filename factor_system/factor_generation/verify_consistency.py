#!/usr/bin/env python3
"""
因子生成一致性验证脚本
验证生成的因子文件与 FactorEngine 注册列表是否完全一致
"""

import sys
from pathlib import Path

import pandas as pd

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from factor_system.factor_engine.api import get_engine


def verify_factor_consistency(parquet_file: Path) -> dict:
    """
    验证Parquet文件中的因子列与FactorEngine注册列表的一致性

    Args:
        parquet_file: 因子Parquet文件路径

    Returns:
        验证结果字典
    """
    # 加载FactorEngine注册列表
    engine = get_engine()
    engine_factors = set(sorted(engine.registry.factors.keys()))

    # 读取Parquet文件
    df = pd.read_parquet(parquet_file)
    price_cols = {"open", "high", "low", "close", "volume"}
    parquet_factors = set([c for c in df.columns if c not in price_cols])

    # 计算差异
    missing = sorted(engine_factors - parquet_factors)
    extra = sorted(parquet_factors - engine_factors)

    return {
        "file": str(parquet_file),
        "engine_count": len(engine_factors),
        "parquet_count": len(parquet_factors),
        "missing": missing,
        "extra": extra,
        "consistent": len(missing) == 0 and len(extra) == 0,
    }


def main():
    """主函数"""
    print("🔍 因子生成一致性验证")
    print("=" * 80)

    # 检查输出目录
    output_dir = project_root / "factor_system" / "factor_output" / "HK"
    if not output_dir.exists():
        print(f"❌ 输出目录不存在: {output_dir}")
        return False

    # 查找所有因子文件
    factor_files = list(output_dir.glob("*/*_factors.parquet"))
    if not factor_files:
        print(f"❌ 未找到因子文件: {output_dir}")
        return False

    print(f"📁 找到 {len(factor_files)} 个因子文件\n")

    # 验证每个文件
    all_consistent = True
    results = []

    for file_path in sorted(factor_files):
        result = verify_factor_consistency(file_path)
        results.append(result)

        timeframe = file_path.parent.name
        symbol = file_path.stem.split("_")[0]

        if result["consistent"]:
            print(
                f"✅ {symbol:10s} {timeframe:10s}: {result['parquet_count']} 因子 (完美匹配)"
            )
        else:
            print(
                f"❌ {symbol:10s} {timeframe:10s}: {result['parquet_count']} 因子 (不一致)"
            )
            all_consistent = False

            if result["missing"]:
                print(
                    f"   缺失 {len(result['missing'])} 个: {result['missing'][:5]}..."
                )
            if result["extra"]:
                print(f"   多余 {len(result['extra'])} 个: {result['extra'][:5]}...")

    # 总结
    print("\n" + "=" * 80)
    if all_consistent:
        print("🎉 所有文件验证通过！生成的因子与 FactorEngine 注册列表完全一致！")
        print(f"✅ 验证文件数: {len(results)}")
        print(f"✅ 因子总数: {results[0]['engine_count']} (Engine)")
        print(f"✅ 一致性: 100%")
        return True
    else:
        print("❌ 发现不一致！部分文件的因子列与 FactorEngine 注册列表不匹配。")
        inconsistent = [r for r in results if not r["consistent"]]
        print(f"❌ 不一致文件数: {len(inconsistent)}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
