#!/usr/bin/env python3
"""
深入调查FactorEngine与factor_generation因子数量差异的原因

分析：
1. factor_generation实际启用的配置
2. 配置对因子数量的影响
3. 154指标的声明与实际实现的差距
4. VectorBT可用性与实际使用的差距
"""

import yaml
from pathlib import Path

def analyze_factor_generation_config():
    """分析factor_generation的配置"""
    print("🔍 分析factor_generation配置...")

    config_file = Path("/Users/zhangshenshen/深度量化0927/factor_system/factor_generation/config.yaml")

    if not config_file.exists():
        print("❌ factor_generation配置文件不存在")
        return {}

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print(f"📊 factor_generation配置:")
    indicators = config.get('indicators', {})
    print(f"  - 启用的指标类型:")
    for key, value in indicators.items():
        if key.startswith('enable_') and value:
            indicator_name = key.replace('enable_', '')
            print(f"    ✅ {indicator_name}: 已启用")
        elif key.startswith('enable_') and not value:
            indicator_name = key.replace('enable_', '')
            print(f"    ❌ {indicator_name}: 未启用")

    timeframes = config.get('timeframes', {})
    enabled_timeframes = timeframes.get('enabled', [])
    print(f"  - 启用的时间框架: {enabled_timeframes}")

    return config

def analyze_vectorbt_availability():
    """分析VectorBT指标的实际可用性"""
    print("\n🔍 分析VectorBT指标可用性...")

    try:
        import vectorbt as vbt
    except ImportError as e:
        print(f"❌ VectorBT不可用: {e}")
        return []

    # VectorBT核心指标列表（从代码中提取）
    vbt_core_indicators = [
        "MA", "MACD", "RSI", "BBANDS", "STOCH", "ATR", "OBV", "MSTD",
        "BOLB", "FIXLB", "FMAX", "FMEAN", "FMIN", "FSTD", "LEXLB",
        "MEANLB", "OHLCSTCX", "OHLCSTX", "RAND", "RANDNX", "RANDX",
        "RPROB", "RPROBCX", "RPROBNX", "RPROBX", "STCX", "STX", "TRENDLB"
    ]

    available_indicators = []
    unavailable_indicators = []

    for indicator in vbt_core_indicators:
        if hasattr(vbt, indicator):
            available_indicators.append(indicator)
        else:
            unavailable_indicators.append(indicator)

    print(f"📊 VectorBT指标可用性:")
    print(f"  - VectorBT核心指标总数: {len(vbt_core_indicators)}")
    print(f"  - 可用指标: {len(available_indicators)}")
    print(f"  - 不可用指标: {len(unavailable_indicators)}")

    if unavailable_indicators:
        print(f"  - 不可用的指标: {unavailable_indicators[:10]}")  # 显示前10个

    return available_indicators

def analyze_talib_availability():
    """分析TA-Lib指标可用性"""
    print("\n🔍 分析TA-Lib指标可用性...")

    try:
        import vectorbt as vbt
        if hasattr(vbt, "talib"):
            # TA-Lib指标列表
            talib_indicators = [
                "SMA", "EMA", "WMA", "DEMA", "TEMA", "TRIMA", "KAMA", "MAMA", "T3",
                "RSI", "STOCH", "STOCHF", "STOCHRSI", "MACD", "MACDEXT", "BBANDS",
                "MIDPOINT", "SAR", "SAREXT", "ADX", "ADXR", "APO"
            ]

            available_talib = []
            unavailable_talib = []

            for indicator in talib_indicators:
                try:
                    vbt.talib(indicator)
                    available_talib.append(f"TA_{indicator}")
                except Exception:
                    unavailable_talib.append(indicator)

            print(f"📊 TA-Lib指标可用性:")
            print(f"  - TA-Lib指标总数: {len(talib_indicators)}")
            print(f"  - 可用TA-Lib指标: {len(available_talib)}")
            print(f"  - 不可用TA-Lib指标: {len(unavailable_talib)}")

            return available_talib
        else:
            print("❌ VectorBT.TA-Lib不可用")
            return []
    except ImportError:
        print("❌ VectorBT不可用")
        return []

def estimate_actual_factor_count():
    """估算factor_generation实际能生成的因子数量"""
    print("\n🔍 估算factor_generation实际因子数量...")

    config = analyze_factor_generation_config()
    vbt_indicators = analyze_vectorbt_availability()
    talib_indicators = analyze_talib_availability()

    # 根据配置估算
    enabled_configs = config.get('indicators', {})

    estimated_factors = []

    # 1. 移动平均类
    if enabled_configs.get('enable_ma', False):
        # 假设支持多个窗口期
        ma_windows = [5, 10, 20, 30, 60]  # 常用窗口期
        for window in ma_windows:
            estimated_factors.append(f"MA{window}")
        estimated_factors.append(f"SMA{window}")

    if enabled_configs.get('enable_ema', False):
        ema_spans = [5, 12, 26]  # 常用EMA跨度
        for span in ema_spans:
            estimated_factors.append(f"EMA{span}")

    # 2. MACD类
    if enabled_configs.get('enable_macd', False):
        estimated_factors.extend(["MACD", "MACD_Signal", "MACD_Hist"])

    # 3. RSI类
    if enabled_configs.get('enable_rsi', False):
        estimated_factors.append("RSI")

    # 4. 布林带类
    if enabled_configs.get('enable_bbands', False):
        estimated_factors.extend(["BBANDS_upper", "BBANDS_middle", "BBANDS_lower"])

    # 5. 随机指标类
    if enabled_configs.get('enable_stoch', False):
        estimated_factors.extend(["STOCH_K", "STOCH_D"])

    # 6. ATR类
    if enabled_configs.get('enable_atr', False):
        estimated_factors.append("ATR")

    # 7. OBV类
    if enabled_configs.get('enable_obv', False):
        estimated_factors.append("OBV")
        if enabled_configs.get('enable_all_periods', False):
            # OBV的移动平均
            obv_ma_windows = [5, 10, 20]
            for window in obv_ma_windows:
                estimated_factors.append(f"OBV_SMA{window}")

    # 8. MSTD类
    if enabled_configs.get('enable_mstd', False):
        estimated_factors.append("MSTD")

    # 9. BOLB类（VectorBT特有）
    if enabled_configs.get('enable_manual_indicators', False) and "BOLB" in vbt_indicators:
        estimated_factors.append("BOLB_20")

    # 10. 波动率类
    if enabled_configs.get('enable_manual_indicators', False):
        estimated_factors.append("VOLATILITY_20")

    # 去重
    unique_factors = list(set(estimated_factors))

    print(f"📊 估算结果:")
    print(f"  - 基于配置估算的因子数量: {len(unique_factors)}个")
    print(f"  - 估算的因子列表: {sorted(unique_factors)}")

    return unique_factors

def analyze_154_indicators_discrepancy():
    """分析154指标声明与实际实现的差距"""
    print("\n🔍 分析154指标声明与实现的差距...")

    calc_file = Path("/Users/zhangshenshen/深度量化0927/factor_system/factor_generation/enhanced_factor_calculator.py")

    with open(calc_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 查找TODO注释，了解哪些指标未实现
    todo_patterns = [
        r'# TODO:.*?暂未启用',
        r'# TODO:.*?暂未实现',
        r'# TODO:.*?暂未完成'
    ]

    todos = []
    for pattern in todo_patterns:
        matches = re.findall(pattern, content, re.DOTALL)
        todos.extend(matches)

    print(f"📊 154指标实现状态:")
    print(f"  - 文件声明: 154个技术指标")
    print(f"  - TODO注释发现: {len(todos)}个")

    for todo in todos[:5]:  # 显示前5个TODO
        print(f"    - {todo.strip()}")

    if len(todos) > 5:
        print(f"    ... 还有{len(todos)-5}个TODO注释")

    # 查找实际的实现
    implemented_count = content.count('factor_data[') + content.count('factor_data["')
    print(f"  - 实际factor_data赋值: {implemented_count}处")

    return {
        'declared_count': 154,
        'todo_count': len(todos),
        'implemented_assignments': implemented_count
    }

def main():
    """主函数"""
    print("🎯 FactorEngine vs factor_generation 差异深度调查")
    print("=" * 60)

    # 1. 分析配置
    config = analyze_factor_generation_config()

    # 2. 分析VectorBT可用性
    vbt_indicators = analyze_vectorbt_availability()

    # 3. 分析TA-Lib可用性
    talib_indicators = analyze_talib_availability()

    # 4. 估算实际因子数量
    estimated_factors = estimate_actual_factor_count()

    # 5. 分析154指标声明差距
    discrepancy_info = analyze_154_indicators_discrepancy()

    print(f"\n📋 关键发现:")
    print(f"  1. 配置限制: factor_generation通过配置文件控制因子启用")
    print(f"  2. 可用性限制: VectorBT和TA-Lib指标并非全部可用")
    print(f"  3. 实现差距: 声明154指标，但实际实现约{len(estimated_factors)}个")
    print(f"  4. 配置影响: 不同配置会产生不同数量的因子")

    # 与FactorEngine对比
    fe_count = 102  # 从之前的分析得到
    fg_count = len(estimated_factors)

    print(f"\n🎯 最终对比:")
    print(f"  - FactorEngine注册因子: {fe_count}个")
    print(f"  - factor_generation估算实现: {fg_count}个")
    print(f"  - 实现差距: {fe_count - fg_count}个")
    print(f"  - 一致性比率: {fg_count/fe_count*100:.1f}%")

    if fg_count < fe_count * 0.5:
        print(f"\n🚨 重大发现: factor_generation实际实现的因子数量不到FactorEngine的50%!")
        print(f"  这可能是配置、可用性或实现完成度的问题")

if __name__ == "__main__":
    import re
    main()