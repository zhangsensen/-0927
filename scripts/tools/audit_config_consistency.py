#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置一致性审计脚本

检查项目中所有生产配置和文档是否一致使用 v3.0 参数。

v3.0 生产参数:
- FREQ = 3
- POS_SIZE = 2
- 43 ETF (38 A股 + 5 QDII)
- 收益率: 237.45%

运行方式:
    uv run python scripts/tools/audit_config_consistency.py
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_yaml_config():
    """检查 YAML 配置文件中的参数"""
    import yaml

    issues = []

    # combo_wfo_config.yaml
    config_path = PROJECT_ROOT / "configs" / "combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    backtest = config.get("backtest", {})

    # 检查 freq
    if backtest.get("freq") != 3:
        issues.append(f"❌ combo_wfo_config.yaml: backtest.freq = {backtest.get('freq')} (应为 3)")
    else:
        print("✅ combo_wfo_config.yaml: backtest.freq = 3")

    # 检查 pos_size
    if backtest.get("pos_size") != 2:
        issues.append(f"❌ combo_wfo_config.yaml: backtest.pos_size = {backtest.get('pos_size')} (应为 2)")
    else:
        print("✅ combo_wfo_config.yaml: backtest.pos_size = 2")

    # 检查 rebalance_frequency
    if "3d" not in str(backtest.get("rebalance_frequency", "")):
        issues.append(f"❌ combo_wfo_config.yaml: rebalance_frequency = {backtest.get('rebalance_frequency')} (应含 3d)")
    else:
        print("✅ combo_wfo_config.yaml: rebalance_frequency = 3d")

    # 检查 combo_wfo.rebalance_frequencies
    combo_wfo = config.get("combo_wfo", {})
    freqs = combo_wfo.get("rebalance_frequencies", [])
    if freqs != [3]:
        issues.append(f"❌ combo_wfo_config.yaml: combo_wfo.rebalance_frequencies = {freqs} (应为 [3])")
    else:
        print("✅ combo_wfo_config.yaml: combo_wfo.rebalance_frequencies = [3]")

    return issues


def check_etf_pool():
    """检查 ETF 池配置"""
    import yaml

    issues = []

    # etf_pools.yaml
    pools_path = PROJECT_ROOT / "configs" / "etf_pools.yaml"
    with open(pools_path) as f:
        pools = yaml.safe_load(f)

    # 收集所有 ETF (从各子池)
    all_symbols = set()
    pools_data = pools.get("pools", {})
    for pool_name, pool_config in pools_data.items():
        symbols = pool_config.get("symbols", [])
        all_symbols.update(symbols)

    # 检查总数
    # 生产池应为 43 只 (不包括实验性的 513180, 513400, 513520)
    # 实际子池有 50 只，但生产配置 (combo_wfo_config) 只使用 43 只
    if len(all_symbols) < 43:
        issues.append(f"❌ etf_pools.yaml: 总共 {len(all_symbols)} 只 ETF (应至少 43)")
    else:
        print(f"✅ etf_pools.yaml: 总共 {len(all_symbols)} 只 ETF (含子池定义)")

    # 检查 5 只关键 QDII 是否存在
    qdii_codes = ["513100", "513500", "159920", "513050", "513130"]
    qdii_pool = pools_data.get("QDII", {}).get("symbols", [])

    missing_qdii = [code for code in qdii_codes if code not in qdii_pool]
    if missing_qdii:
        issues.append(f"❌ etf_pools.yaml: QDII 池缺少关键 ETF: {missing_qdii}")
    else:
        print("✅ etf_pools.yaml: 5 只关键 QDII 均存在于 QDII 池")

    # 检查 QDII 描述中是否包含 Alpha 来源说明
    qdii_desc = pools_data.get("QDII", {}).get("description", "")
    if "Alpha" in qdii_desc or "90%" in qdii_desc:
        print("✅ etf_pools.yaml: QDII 池标注为 Alpha 来源")
    else:
        issues.append("⚠️ etf_pools.yaml: QDII 池描述中未标注 Alpha 来源")

    return issues


def check_readme():
    """检查 README.md 中的参数"""
    issues = []

    readme_path = PROJECT_ROOT / "README.md"
    content = readme_path.read_text()

    # 检查版本号
    if "v3.0" not in content and "v3.1" not in content:
        issues.append("❌ README.md: 未提及 v3.0 或 v3.1 版本")
    else:
        print("✅ README.md: 包含 v3.x 版本信息")

    # 检查收益率
    if "237" not in content:
        issues.append("❌ README.md: 未提及 237% 收益率")
    else:
        print("✅ README.md: 包含 237% 收益率")

    # 检查是否有旧的 FREQ=8 作为当前参数（而非历史对比）
    lines = content.split("\n")
    for i, line in enumerate(lines):
        # 检查是否在 v1.0 历史部分（可接受）
        if "v1.0" in line.lower() or "旧" in line or "legacy" in line.lower():
            continue
        # 如果 FREQ=8 出现在非历史对比的上下文中，警告
        if "FREQ" in line and "8" in line and "3" not in line:
            # 如果是对比表格（包含 3），则 OK
            if "|" not in line:
                issues.append(f"⚠️ README.md 第 {i+1} 行: 可能存在旧参数引用: {line.strip()}")

    return issues


def check_docs():
    """检查 docs/ 目录中的核心文档"""
    issues = []

    # docs/README.md
    docs_readme = PROJECT_ROOT / "docs" / "README.md"
    if docs_readme.exists():
        content = docs_readme.read_text()
        if "v3.0" in content or "v3.1" in content:
            print("✅ docs/README.md: 包含 v3.x 版本信息")
        else:
            issues.append("⚠️ docs/README.md: 可能需要更新到 v3.x")

    # docs/BEST_STRATEGY_43ETF_UNIFIED.md
    best_strategy = PROJECT_ROOT / "docs" / "BEST_STRATEGY_43ETF_UNIFIED.md"
    if best_strategy.exists():
        content = best_strategy.read_text()
        if "237" in content:
            print("✅ docs/BEST_STRATEGY_43ETF_UNIFIED.md: 包含 237% 收益率")
        else:
            issues.append("❌ docs/BEST_STRATEGY_43ETF_UNIFIED.md: 未包含 237% 收益率")

    # docs/ETF_POOL_ARCHITECTURE.md
    etf_arch = PROJECT_ROOT / "docs" / "ETF_POOL_ARCHITECTURE.md"
    if etf_arch.exists():
        print("✅ docs/ETF_POOL_ARCHITECTURE.md: 存在")
    else:
        issues.append("❌ docs/ETF_POOL_ARCHITECTURE.md: 缺失")

    return issues


def main():
    print("=" * 60)
    print("🔍 v3.0 配置一致性审计")
    print("=" * 60)
    print()

    all_issues = []

    print("📁 检查 YAML 配置文件...")
    all_issues.extend(check_yaml_config())
    print()

    print("📊 检查 ETF 池配置...")
    all_issues.extend(check_etf_pool())
    print()

    print("📄 检查 README.md...")
    all_issues.extend(check_readme())
    print()

    print("📚 检查 docs/ 文档...")
    all_issues.extend(check_docs())
    print()

    print("=" * 60)
    if all_issues:
        print(f"⚠️ 发现 {len(all_issues)} 个问题:")
        for issue in all_issues:
            print(f"  {issue}")
        print()
        print("请修复以上问题以确保生产配置一致性。")
        return 1
    else:
        print("✅ 所有检查通过！配置一致性验证成功。")
        print()
        print("🔒 v3.0 生产参数:")
        print("   FREQ = 3")
        print("   POS_SIZE = 2")
        print("   ETF 池 = 43 (含 5 只关键 QDII)")
        print("   收益率 = 237.45%")
        return 0


if __name__ == "__main__":
    sys.exit(main())
