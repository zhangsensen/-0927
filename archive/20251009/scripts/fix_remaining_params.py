#!/usr/bin/env python3
"""
修复剩余的参数初始化问题
"""
import re

def fix_remaining_params():
    """修复overlap_generated.py中的参数初始化问题"""

    file_path = "/Users/zhangshenshen/深度量化0927/factor_system/factor_engine/factors/overlap_generated.py"

    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 修复MACD_12_26_9的参数初始化
    macd_pattern = r'(class MACD_12_26_9\(BaseFactor\):.*?def __init__\(self, \*\*kwargs\):.*?default_params = \{[^}]+\}\s*default_params\.update\(kwargs\)\s*)(super\(\).__init__\(\*\*default_params\))'
    macd_replacement = r'\1self.fastperiod = default_params.get(\'fastperiod\', 12)\n        self.slowperiod = default_params.get(\'slowperiod\', 26)\n        self.signalperiod = default_params.get(\'signalperiod\', 9)\n        \2'

    if re.search(macd_pattern, content, re.DOTALL):
        content = re.sub(macd_pattern, macd_replacement, content, flags=re.DOTALL)
        print("✅ 修复MACD_12_26_9参数初始化")
    else:
        print("⚠️ 未找到MACD_12_26_9初始化模式")

    # 修复BB_10_2_0_Upper的参数初始化
    bb_pattern = r'(class BB_10_2_0_Upper\(BaseFactor\):.*?def __init__\(self, \*\*kwargs\):.*?default_params = \{[^}]+\}\s*default_params\.update\(kwargs\)\s*)(super\(\).__init__\(\*\*default_params\))'
    bb_replacement = r'\1self.timeperiod = default_params.get(\'timeperiod\', 10)\n        self.nbdevup = default_params.get(\'nbdevup\', 2.0)\n        self.nbdevdn = default_params.get(\'nbdevdn\', 2.0)\n        \2'

    if re.search(bb_pattern, content, re.DOTALL):
        content = re.sub(bb_pattern, bb_replacement, content, flags=re.DOTALL)
        print("✅ 修复BB_10_2_0_Upper参数初始化")
    else:
        print("⚠️ 未找到BB_10_2_0_Upper初始化模式")

    # 修复计算方法中缺少close变量定义的问题
    # 在MACD计算方法中添加close变量
    macd_calc_pattern = r'(def calculate\(self, data: pd\.DataFrame\) -> pd\.Series:.*?open_price = data\["open"\]\.astype\("float64"\))(\s*high = data\["high"\]\.astype\("float64"\))'
    macd_calc_replacement = r'\1\n        close = data["close"].astype("float64")\2'

    content = re.sub(macd_calc_pattern, macd_calc_replacement, content, flags=re.DOTALL)

    # 在BB计算方法中添加close变量
    bb_calc_pattern = r'(def calculate\(self, data: pd\.DataFrame\) -> pd\.Series:.*?open_price = data\["open"\]\.astype\("float64"\))(\s*high = data\["high"\]\.astype\("float64"\))'
    bb_calc_replacement = r'\1\n        close = data["close"].astype("float64")\2'

    content = re.sub(bb_calc_pattern, bb_calc_replacement, content, flags=re.DOTALL)

    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print("✅ 参数初始化修复完成！")

if __name__ == "__main__":
    fix_remaining_params()