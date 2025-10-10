#!/usr/bin/env python3
"""
从factor_generation模块中提取154个因子的完整清单
"""

import re


def extract_all_factors():
    """从enhanced_factor_calculator.py中提取所有因子"""

    # 读取enhanced_factor_calculator.py
    with open('/Users/zhangshenshen/深度量化0927/factor_system/factor_generation/enhanced_factor_calculator.py', 'r', encoding='utf-8') as f:
        content = f.read()

    factors = set()

    # 1. 提取所有MA因子 - 查找MA模式
    ma_patterns = [
        r'factor_data\[f"MA(\w+)"\]',
        r'f"MA(\w+)"',
        r'"MA(\w+)"'
    ]

    for pattern in ma_patterns:
        ma_matches = re.findall(pattern, content)
        for w in ma_matches:
            if w.isdigit():
                factors.add(f"MA{w}")

    # 2. 提取所有EMA因子
    ema_patterns = [
        r'factor_data\[f"EMA(\w+)"\]',
        r'f"EMA(\w+)"',
        r'"EMA(\w+)"'
    ]

    for pattern in ema_patterns:
        ema_matches = re.findall(pattern, content)
        for s in ema_matches:
            if s.isdigit():
                factors.add(f"EMA{s}")

    # 3. 提取MACD因子
    macd_patterns = [
        r'factor_data\[f"MACD_(\d+)_(\d+)_(\d+)_(\w+)"\]',
        r'f"MACD_(\d+)_(\d+)_(\d+)"',
        r'"MACD_(\d+)_(\d+)_(\d+)"'
    ]

    for pattern in macd_patterns:
        macd_matches = re.findall(pattern, content)
        for match in macd_matches:
            if len(match) == 4:  # MACD_f_s_sig_component
                f, s, sig, comp = match
                factors.add(f"MACD_{f}_{s}_{sig}_{comp}")
            elif len(match) == 3:  # MACD_f_s_sig
                f, s, sig = match
                factors.add(f"MACD_{f}_{s}_{sig}_MACD")
                factors.add(f"MACD_{f}_{s}_{sig}_Signal")
                factors.add(f"MACD_{f}_{s}_{sig}_Hist")

    # 4. 提取RSI因子
    rsi_patterns = [
        r'factor_data\[f"RSI(\w+)"\]',
        r'f"RSI(\w+)"',
        r'"RSI(\w+)"'
    ]

    for pattern in rsi_patterns:
        rsi_matches = re.findall(pattern, content)
        for w in rsi_matches:
            if w.isdigit():
                factors.add(f"RSI{w}")

    # 5. 提取布林带因子
    bb_patterns = [
        r'factor_data\[f"BB_(\d+)_([\d.]+)_(\w+)"\]',
        r'f"BB_(\d+)_([\d.]+)"',
        r'"BB_(\d+)_([\d.]+)"'
    ]

    for pattern in bb_patterns:
        bb_matches = re.findall(pattern, content)
        for match in bb_matches:
            if len(match) == 3:  # BB_window_alpha_component
                w, alpha, comp = match
                factors.add(f"BB_{w}_{alpha}_{comp}")
            elif len(match) == 2:  # BB_window_alpha
                w, alpha = match
                factors.add(f"BB_{w}_{alpha}_Upper")
                factors.add(f"BB_{w}_{alpha}_Middle")
                factors.add(f"BB_{w}_{alpha}_Lower")
                factors.add(f"BB_{w}_{alpha}_Width")

    # 6. 提取STOCH因子
    stoch_patterns = [
        r'factor_data\[f"STOCH_(\d+)_(\d+)_(\w+)"\]',
        r'f"STOCH_(\d+)_(\d+)"',
        r'"STOCH_(\d+)_(\d+)"'
    ]

    for pattern in stoch_patterns:
        stoch_matches = re.findall(pattern, content)
        for match in stoch_matches:
            if len(match) == 3:  # STOCH_k_d_component
                k, d, comp = match
                factors.add(f"STOCH_{k}_{d}_{comp}")
            elif len(match) == 2:  # STOCH_k_d
                k, d = match
                factors.add(f"STOCH_{k}_{d}_K")
                factors.add(f"STOCH_{k}_{d}_D")

    # 7. 提取ATR因子
    atr_patterns = [
        r'factor_data\[f"ATR(\w+)"\]',
        r'f"ATR(\w+)"',
        r'"ATR(\w+)"'
    ]

    for pattern in atr_patterns:
        atr_matches = re.findall(pattern, content)
        for w in atr_matches:
            if w.isdigit():
                factors.add(f"ATR{w}")

    # 8. 提取MSTD因子
    mstd_patterns = [
        r'factor_data\[f"MSTD(\w+)"\]',
        r'f"MSTD(\w+)"',
        r'"MSTD(\w+)"'
    ]

    for pattern in mstd_patterns:
        mstd_matches = re.findall(pattern, content)
        for w in mstd_matches:
            if w.isdigit():
                factors.add(f"MSTD{w}")

    # 9. 提取OBV因子
    obv_patterns = [
        r'factor_data\[f"OBV(\w*)"\]',
        r'f"OBV(\w*)"',
        r'"OBV(\w*)"'
    ]

    for pattern in obv_patterns:
        obv_matches = re.findall(pattern, content)
        for suffix in obv_matches:
            if suffix:
                factors.add(f"OBV{suffix}")
            else:
                factors.add("OBV")

    # 10. 提取手动计算的指标
    manual_patterns = [
        (r'factor_data\[f"WILLR(\d+)"\]', "WILLR{}"),
        (r'f"WILLR(\d+)"', "WILLR{}"),
        (r'factor_data\[f"CCI(\d+)"\]', "CCI{}"),
        (r'f"CCI(\d+)"', "CCI{}"),
        (r'factor_data\[f"Momentum(\d+)"\]', "Momentum{}"),
        (r'f"Momentum(\d+)"', "Momentum{}"),
        (r'factor_data\[f"Position(\d+)"\]', "Position{}"),
        (r'f"Position(\d+)"', "Position{}"),
        (r'factor_data\[f"Trend(\d+)"\]', "Trend{}"),
        (r'f"Trend(\d+)"', "Trend{}"),
        (r'factor_data\[f"Volume_Ratio(\d+)"\]', "Volume_Ratio{}"),
        (r'f"Volume_Ratio(\d+)"', "Volume_Ratio{}"),
        (r'factor_data\[f"Volume_Momentum(\d+)"\]', "Volume_Momentum{}"),
        (r'f"Volume_Momentum(\d+)"', "Volume_Momentum{}"),
        (r'factor_data\[f"VWAP(\d+)"\]', "VWAP{}"),
        (r'f"VWAP(\d+)"', "VWAP{}"),
    ]

    for pattern, template in manual_patterns:
        matches = re.findall(pattern, content)
        for param in matches:
            if param.isdigit():
                factors.add(template.format(param))

    # 11. 添加特殊因子
    special_patterns = [
        r'"OBV"',
        r'"BOLB_(\d+)"',
        r'TA_(\w+)'
    ]

    for pattern in special_patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            if pattern.startswith('"BOLB'):
                factors.add(f"BOLB_{match}")
            elif pattern.startswith('"OBV'):
                factors.add("OBV")
            elif pattern.startswith('TA_'):
                factors.add(f"TA_{match}")

    # 12. 提取factor_data赋值的所有因子
    factor_data_pattern = r'factor_data\[f"([^"]+)"\]'
    fd_matches = re.findall(factor_data_pattern, content)
    for factor in fd_matches:
        factors.add(factor)

    # 13. 提取字符串形式的因子名
    string_pattern = r'["\']([A-Z][A-Z0-9_]+)["\']'
    string_matches = re.findall(string_pattern, content)

    # 过滤掉明显的非因子字符串
    for item in string_matches:
        if (len(item) > 2 and
            not item.startswith('def ') and
            not item.startswith('class ') and
            not item.startswith('import ') and
            not item.startswith('from ') and
            not any(keyword in item.lower() for keyword in ['logger', 'error', 'warning', 'info', 'debug', 'time', 'data', 'result', 'value'])):
            factors.add(item)

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
        "其他": []
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
        elif any(x in factor for x in ["OBV", "Volume", "VWAP"]):
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
    print("🔍 深度分析factor_generation模块中的因子...")

    factors = extract_all_factors()
    categories = categorize_factors(factors)

    print(f"\n📊 总计发现 {len(factors)} 个因子")

    # 打印分类结果
    for category, factor_list in categories.items():
        if factor_list:  # 只显示非空类别
            print(f"\n📈 {category} ({len(factor_list)}个):")
            for factor in factor_list:
                print(f"  - {factor}")

    # 保存到文件
    with open('/Users/zhangshenshen/深度量化0927/factor_generation_factors_list.txt', 'w', encoding='utf-8') as f:
        f.write("factor_generation模块中的因子清单\n")
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