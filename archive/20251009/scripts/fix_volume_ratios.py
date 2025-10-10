#!/usr/bin/env python3
"""
修复成交量比率参数化问题
"""
import re


def fix_volume_ratios():
    """修复volume_generated.py中的成交量比率参数化问题"""

    file_path = "/Users/zhangshenshen/深度量化0927/factor_system/factor_engine/factors/volume_generated.py"

    # 读取文件
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 修复每个成交量比率类的窗口参数
    volume_ratios = [
        ("Volume_Ratio10", 10),
        ("Volume_Ratio15", 15),
        ("Volume_Ratio20", 20),
        ("Volume_Ratio25", 25),
        ("Volume_Ratio30", 30),
    ]

    for class_name, period in volume_ratios:
        print(f"修复 {class_name} 使用窗口 {period}...")

        # 修复类注释中的参数
        old_param_comment = f"参数: {{}}"
        new_param_comment = f"参数: {{'timeperiod': {period}}}"
        content = content.replace(old_param_comment, new_param_comment)

        # 修复初始化参数
        old_init_params = "default_params = {}"
        new_init_params = f"default_params = {{'timeperiod': {period}}}"
        content = content.replace(old_init_params, new_init_params)

        # 修复硬编码的窗口大小
        old_window = "volume_sma = volume.rolling(window=20).mean()"
        new_window = f"volume_sma = volume.rolling(window={period}).mean()"
        content = content.replace(old_window, new_window)

        # 修复factor_name未定义问题
        old_error = f'logger.error(f"计算{{factor_name}}失败: {{e}}")'
        new_error = 'logger.error(f"计算失败: {e}")'
        content = content.replace(old_error, new_error)

        # 修复返回名称中的factor_name
        old_return = 'name="{factor_name}"'
        new_return = "name=self.factor_id"
        content = content.replace(old_return, new_return)

    # 写回文件
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    print("✅ 成交量比率参数化修复完成！")
    print("修复内容：")
    for class_name, period in volume_ratios:
        print(f"  ✅ {class_name}: 使用正确的窗口 {period}")


if __name__ == "__main__":
    fix_volume_ratios()
