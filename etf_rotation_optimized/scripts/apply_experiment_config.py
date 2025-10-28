#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验配置应用工具
功能: 将实验配置合并到主配置文件，避免手动编辑出错
"""

import sys
from pathlib import Path

import yaml


def apply_experiment_config(exp_name: str):
    """
    应用实验配置到主配置

    Args:
        exp_name: A/B/C/D 或配置文件名
    """
    # 路径配置
    base_dir = Path(__file__).parent.parent
    main_config = base_dir / "configs" / "FACTOR_SELECTION_CONSTRAINTS.yaml"
    exp_dir = base_dir / "configs" / "experiments"

    # 确定实验配置文件
    if exp_name in ["A", "B", "C", "D"]:
        config_map = {
            "A": "config_A_baseline.yaml",
            "B": "config_B_tiebreak.yaml",
            "C": "config_C_meta.yaml",
            "D": "config_D_full.yaml",
        }
        exp_file = exp_dir / config_map[exp_name]
    elif exp_name in ["A_opt", "C_opt"]:
        config_map = {
            "A_opt": "config_A_optimized.yaml",
            "C_opt": "config_C_optimized.yaml",
        }
        exp_file = exp_dir / config_map[exp_name]
    elif exp_name.startswith("exp"):
        # 支持exp8, exp9等编号配置
        exp_file = exp_dir / f"{exp_name}.yaml"
    else:
        exp_file = exp_dir / exp_name

    if not exp_file.exists():
        print(f"❌ 实验配置文件不存在: {exp_file}")
        return False

    # 加载配置
    with open(main_config, "r", encoding="utf-8") as f:
        main_cfg = yaml.safe_load(f)

    with open(exp_file, "r", encoding="utf-8") as f:
        exp_cfg = yaml.safe_load(f)

    # 合并配置（实验配置覆盖主配置）
    if "meta_factor_weighting" in exp_cfg:
        main_cfg["meta_factor_weighting"] = exp_cfg["meta_factor_weighting"]

    if "correlation_deduplication" in exp_cfg:
        # 覆盖strategy、threshold和icir参数
        if "strategy" in exp_cfg["correlation_deduplication"]:
            main_cfg["correlation_deduplication"]["strategy"] = exp_cfg[
                "correlation_deduplication"
            ]["strategy"]
        if "threshold" in exp_cfg["correlation_deduplication"]:
            main_cfg["correlation_deduplication"]["threshold"] = exp_cfg[
                "correlation_deduplication"
            ]["threshold"]
        if "icir_min_windows" in exp_cfg["correlation_deduplication"]:
            main_cfg["correlation_deduplication"]["icir_min_windows"] = exp_cfg[
                "correlation_deduplication"
            ]["icir_min_windows"]
        if "icir_std_floor" in exp_cfg["correlation_deduplication"]:
            main_cfg["correlation_deduplication"]["icir_std_floor"] = exp_cfg[
                "correlation_deduplication"
            ]["icir_std_floor"]

    if "minimum_ic" in exp_cfg:
        # 覆盖minimum_ic阈值
        if "global_minimum" in exp_cfg["minimum_ic"]:
            main_cfg["minimum_ic"]["global_minimum"] = exp_cfg["minimum_ic"][
                "global_minimum"
            ]

    # 🆕 支持 family_quota 的合并（A方案新增）
    if "family_quota" in exp_cfg:
        # 更新或添加实验配置中的family定义
        if "family_quota" not in main_cfg:
            main_cfg["family_quota"] = {}
        for family_name, family_config in exp_cfg["family_quota"].items():
            main_cfg["family_quota"][family_name] = family_config

    # 备份原配置
    backup = main_config.parent / f"FACTOR_SELECTION_CONSTRAINTS.backup_{exp_name}.yaml"
    with open(backup, "w", encoding="utf-8") as f:
        yaml.dump(main_cfg, f, allow_unicode=True, default_flow_style=False)

    # 写入新配置
    with open(main_config, "w", encoding="utf-8") as f:
        yaml.dump(main_cfg, f, allow_unicode=True, default_flow_style=False)

    print(f"✅ 已应用实验配置 {exp_name}")
    print(f"   Meta enabled: {exp_cfg.get('meta_factor_weighting', {}).get('enabled')}")
    print(
        f"   Meta beta: {exp_cfg.get('meta_factor_weighting', {}).get('beta', 'N/A')}"
    )
    print(
        f"   Dedup threshold: {exp_cfg.get('correlation_deduplication', {}).get('threshold', 'N/A')}"
    )
    print(
        f"   Strategy: {exp_cfg.get('correlation_deduplication', {}).get('strategy')}"
    )
    print(
        f"   Minimum IC: {exp_cfg.get('minimum_ic', {}).get('global_minimum', 'N/A')}"
    )
    print(f"   备份: {backup.name}")

    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python apply_experiment_config.py <A|B|C|D|A_opt|C_opt>")
        print("\n原始配置:")
        print("  A - Baseline (meta=off, strategy=keep_higher_ic, IC=0.02)")
        print(
            "  B - ICIR Tie-break Only (meta=off, strategy=keep_higher_icir, IC=0.02)"
        )
        print("  C - Meta Factor (meta=on beta=0.3, strategy=keep_higher_ic, IC=0.02)")
        print("  D - Full (meta=on beta=0.3, strategy=keep_higher_icir, IC=0.02)")
        print("\n优化配置 (基于深度数据分析):")
        print("  A_opt - Optimized Baseline (meta=off, IC=0.012, beta=0.6)")
        print("  C_opt - Optimized Meta Factor (meta=on beta=0.6, IC=0.012)")
        print("\n优化说明:")
        print("  - IC阈值: 0.02 → 0.012 (IS IC中位数, 预期2.5+因子/窗口)")
        print("  - Meta Beta: 0.3 → 0.6 (翻倍权重, ICIR加权可达60%)")
        print("  - 基于990个IC值的统计分析优化")
        sys.exit(1)

    exp = sys.argv[1]
    valid_configs = [
        "A",
        "B",
        "C",
        "D",
        "A_opt",
        "C_opt",
        "exp8",
        "exp9",
        "exp10",
        "exp11",
        "exp_baseline",
        "exp_new_factors",
    ]  # 🆕 A方案新配置
    if exp not in valid_configs:
        print(f"❌ 无效的配置: {exp}, 请使用 {'/'.join(valid_configs)}")
        sys.exit(1)

    apply_experiment_config(exp)
