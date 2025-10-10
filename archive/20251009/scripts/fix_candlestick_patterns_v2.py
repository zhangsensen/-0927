#!/usr/bin/env python3
"""
精确修复K线模式识别实现
"""
import re


def fix_candlestick_patterns_v2():
    """修复technical_generated.py中的K线模式实现问题"""

    file_path = "/Users/zhangshenshen/深度量化0927/factor_system/factor_engine/factors/technical_generated.py"

    # 读取文件
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 修复问题1：添加open_price定义
    # 查找所有K线模式的calculate方法，并添加open_price定义
    pattern = r'(class TA_CDL[A-Z0-9_]+\(BaseFactor\):.*?def calculate\(self, data: pd\.DataFrame\) -> pd\.Series:.*?)(price = data\["close"\]\.astype\("float64"\))'

    def add_open_price(match):
        class_header = match.group(1)
        price_line = match.group(2)
        return f'{class_header}open_price = data["open"].astype("float64")\n        {price_line}'

    content = re.sub(pattern, add_open_price, content, flags=re.DOTALL)

    # 修复问题2：修复factor_name未定义的问题
    content = content.replace(
        'logger.error(f"计算{factor_name}失败: {{e}}")',
        'logger.error(f"计算失败: {e}")',
    )
    content = content.replace('name="{factor_name}"', "name=self.factor_id")

    # 修复问题3：修复ImportError中的返回值使用正确的变量
    content = content.replace(
        'result = pd.Series([0] * len(price), index=price.index, name="',
        'result = pd.Series([0] * len(close), index=close.index, name="',
    )

    # 修复问题4：确保异常处理返回正确的Series
    content = content.replace(
        'return pd.Series([np.nan] * len(price), index=price.index, name="{factor_name}")',
        "return pd.Series([np.nan] * len(close), index=close.index, name=self.factor_id)",
    )

    # 写回文件
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    print("K线模式识别V2修复完成！")
    print("修复内容：")
    print("1. ✅ 添加了open_price定义")
    print("2. ✅ 修复了factor_name未定义问题")
    print("3. ✅ 修复了变量引用问题")
    print("4. ✅ 修复了异常处理返回值")


if __name__ == "__main__":
    fix_candlestick_patterns_v2()
