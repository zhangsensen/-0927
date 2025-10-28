#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速批量运行剩余实验 (Exp9-11)
"""
import subprocess
import sys
from pathlib import Path

ETF_DIR = Path("/Users/zhangshenshen/深度量化0927/etf_rotation_optimized")

experiments = [
    (9, "threshold=0.90, beta=0.0"),
    (10, "threshold=0.88, beta=0.8"),
    (11, "threshold=0.90, beta=0.8"),
]

for exp_num, exp_desc in experiments:
    print(f"\n{'='*70}")
    print(f"  🧪 运行 Exp{exp_num}: {exp_desc}")
    print(f"{'='*70}\n")

    # 1. 应用配置
    print(f"1️⃣ 应用配置...")
    result = subprocess.run(
        ["python3", "scripts/apply_experiment_config.py", f"exp{exp_num}"],
        cwd=ETF_DIR,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"❌ 配置失败: {result.stderr}")
        continue

    print(result.stdout)

    # 2. 运行WFO
    print(f"2️⃣ 运行WFO...")
    result = subprocess.run(["python", "scripts/step3_run_wfo.py"], cwd=ETF_DIR)

    if result.returncode != 0:
        print(f"❌ WFO运行失败")
        continue

    # 3. 保存结果
    print(f"\n3️⃣ 保存结果...")
    result_files = list((ETF_DIR / "results" / "wfo").glob("20*/wfo_results.pkl"))
    if result_files:
        latest = max(result_files, key=lambda p: p.stat().st_mtime)
        dest = ETF_DIR / "results" / "wfo" / f"exp{exp_num}.pkl"

        import shutil

        shutil.copy2(latest, dest)
        print(f"✅ Exp{exp_num} 完成！结果已保存到 {dest.name}")
    else:
        print(f"❌ 未找到结果文件")

print(f"\n{'='*70}")
print("🎉 所有实验运行完成！")
print(f"{'='*70}\n")

# 检查结果
results_dir = ETF_DIR / "results" / "wfo"
for exp in [7, 8, 9, 10, 11]:
    pkl_file = (
        results_dir / f"exp{exp}.pkl"
        if exp > 7
        else results_dir / f"exp{exp}_max8_beta08_FIXED.pkl" if exp == 7 else None
    )
    if pkl_file and pkl_file.exists():
        size = pkl_file.stat().st_size // 1024
        print(f"  ✅ Exp{exp}: {pkl_file.name} ({size}KB)")
    else:
        print(f"  ❌ Exp{exp}: 未找到结果文件")
