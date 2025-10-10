#!/usr/bin/env python3
"""
从factor_generation模块中提取154个因子的完整清单
"""

import re


def extract_factors_from_enhanced_calculator():
    """从enhanced_factor_calculator.py中提取所有因子"""

    # 读取enhanced_factor_calculator.py
    with open(
        "/Users/zhangshenshen/深度量化0927/factor_system/factor_generation/enhanced_factor_calculator.py",
        "r",
        encoding="utf-8",
    ) as f:
        content = f.read()

    factors = set()

    # 1. 提取factor_calculations.append中的因子名
    calculation_pattern = r'factor_calculations\.append\(\s*\(?["\']([^"\']+)["\']'
    calc_matches = re.findall(calculation_pattern, content)
    factors.update(calc_matches)

    # 2. 提取MA因子
    ma_pattern = r'f"MA(\w+)"'
    ma_matches = re.findall(ma_pattern, content)
    for w in ma_matches:
        factors.add(f"MA{w}")

    # 3. 提取EMA因子
    ema_pattern = r'f"EMA(\w+)"'
    ema_matches = re.findall(ema_pattern, content)
    for s in ema_matches:
        factors.add(f"EMA{s}")

    # 4. 提取MACD因子
    macd_pattern = r'f"MACD_(\d+)_(\d+)_(\d+)"'
    macd_matches = re.findall(macd_pattern, content)
    for f, s, sig in macd_matches:
        factors.add(f"MACD_{f}_{s}_{sig}")

    # 5. 提取RSI因子
    rsi_pattern = r'f"RSI(\w+)"'
    rsi_matches = re.findall(rsi_pattern, content)
    for w in rsi_matches:
        factors.add(f"RSI{w}")

    # 6. 提取布林带因子
    bb_pattern = r'f"BB_(\w+)_([\d.]+)"'
    bb_matches = re.findall(bb_pattern, content)
    for w, alpha in bb_matches:
        factors.add(f"BB_{w}_{alpha}")

    # 7. 提取STOCH因子
    stoch_pattern = r'f"STOCH_(\d+)_(\d+)"'
    stoch_matches = re.findall(stoch_pattern, content)
    for k, d in stoch_matches:
        factors.add(f"STOCH_{k}_{d}")

    # 8. 提取ATR因子
    atr_pattern = r'f"ATR(\w+)"'
    atr_matches = re.findall(atr_pattern, content)
    for w in atr_matches:
        factors.add(f"ATR{w}")

    # 9. 提取MSTD因子
    mstd_pattern = r'f"MSTD(\w+)"'
    mstd_matches = re.findall(mstd_pattern, content)
    for w in mstd_matches:
        factors.add(f"MSTD{w}")

    # 10. 提取OBV因子
    obv_pattern = r'f"OBV_SMA(\w+)"'
    obv_matches = re.findall(obv_pattern, content)
    for w in obv_matches:
        factors.add(f"OBV_SMA{w}")

    # 11. 提取手动计算的指标
    manual_patterns = [
        (r'f"WILLR(\w+)"', "WILLR{}"),
        (r'f"CCI(\w+)"', "CCI{}"),
        (r'f"Momentum(\w+)"', "Momentum{}"),
        (r'f"Position(\w+)"', "Position{}"),
        (r'f"Trend(\w+)"', "Trend{}"),
        (r'f"Volume_Ratio(\w+)"', "Volume_Ratio{}"),
        (r'f"Volume_Momentum(\w+)"', "Volume_Momentum{}"),
        (r'f"VWAP(\w+)"', "VWAP{}"),
    ]

    for pattern, template in manual_patterns:
        matches = re.findall(pattern, content)
        for param in matches:
            factors.add(template.format(param))

    # 12. 添加特殊因子
    special_factors = ["OBV", "BOLB_20"]
    factors.update(special_factors)

    # 13. 提取TA-Lib指标
    ta_pattern = r'f"TA_(\w+[^"]*)"'
    ta_matches = re.findall(ta_pattern, content)
    for ta_name in ta_matches:
        # 清理TA名称
        clean_ta = ta_name.rstrip("_")
        factors.add(f"TA_{clean_ta}")

    return sorted(list(factors))


def categorize_factors(factors):
    """对因子进行分类"""
    categories = {
        "移动平均线": [],
        "MACD指标": [],
        "RSI指标": [],
        "布林带": [],
        "随机指标": [],
        "ATR指标": [],
        "波动率指标": [],
        "成交量指标": [],
        "威廉指标": [],
        "商品通道": [],
        "动量指标": [],
        "位置指标": [],
        "趋势强度": [],
        "TA-Lib指标": [],
        "其他": [],
    }

    for factor in factors:
        if factor.startswith("MA") or factor.startswith("EMA"):
            categories["移动平均线"].append(factor)
        elif factor.startswith("MACD"):
            categories["MACD指标"].append(factor)
        elif factor.startswith("RSI"):
            categories["RSI指标"].append(factor)
        elif factor.startswith("BB"):
            categories["布林带"].append(factor)
        elif factor.startswith("STOCH"):
            categories["随机指标"].append(factor)
        elif factor.startswith("ATR"):
            categories["ATR指标"].append(factor)
        elif factor.startswith("MSTD"):
            categories["波动率指标"].append(factor)
        elif (
            factor.startswith("OBV")
            or factor.startswith("Volume")
            or factor.startswith("VWAP")
        ):
            categories["成交量指标"].append(factor)
        elif factor.startswith("WILLR"):
            categories["威廉指标"].append(factor)
        elif factor.startswith("CCI"):
            categories["商品通道"].append(factor)
        elif factor.startswith("Momentum"):
            categories["动量指标"].append(factor)
        elif factor.startswith("Position"):
            categories["位置指标"].append(factor)
        elif factor.startswith("Trend"):
            categories["趋势强度"].append(factor)
        elif factor.startswith("TA_"):
            categories["TA-Lib指标"].append(factor)
        else:
            categories["其他"].append(factor)

    return categories


def main():
    """主函数"""
    print("🔍 分析factor_generation模块中的因子...")

    factors = extract_factors_from_enhanced_calculator()
    categories = categorize_factors(factors)

    print(f"\n📊 总计发现 {len(factors)} 个因子")

    # 打印分类结果
    for category, factor_list in categories.items():
        if factor_list:  # 只显示非空类别
            print(f"\n📈 {category} ({len(factor_list)}个):")
            for factor in factor_list:
                print(f"  - {factor}")

    # 保存到文件
    with open(
        "/Users/zhangshenshen/深度量化0927/factor_generation_factors_list.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write("factor_generation模块中的154个因子清单\n")
        f.write("=" * 50 + "\n\n")

        total_count = 0
        for category, factor_list in categories.items():
            if factor_list:
                f.write(f"{category} ({len(factor_list)}个):\n")
                for factor in factor_list:
                    f.write(f"  {factor}\n")
                    total_count += 1
                f.write("\n")

        f.write(f"总计: {total_count} 个因子\n")

    print(f"\n✅ 因子清单已保存至: factor_generation_factors_list.txt")
    print(f"📈 总计: {len(factors)} 个因子")

    return factors, categories


if __name__ == "__main__":
    factors, categories = main()
