#!/usr/bin/env python3
"""
精确修复成交量比率参数化问题
"""
import re


def fix_volume_ratios_v2():
    """精确修复volume_generated.py中的成交量比率参数化问题"""

    file_path = "/Users/zhangshenshen/深度量化0927/factor_system/factor_engine/factors/volume_generated.py"

    # 读取文件
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 精确修复每个成交量比率类
    fixes = [
        ("Volume_Ratio10", "class Volume_Ratio10", 10),
        ("Volume_Ratio15", "class Volume_Ratio15", 15),
        ("Volume_Ratio20", "class Volume_Ratio20", 20),
        ("Volume_Ratio25", "class Volume_Ratio25", 25),
        ("Volume_Ratio30", "class Volume_Ratio30", 30),
    ]

    for class_name, class_marker, period in fixes:
        print(f"修复 {class_name} 使用窗口 {period}...")

        # 查找类的位置
        class_start = content.find(class_marker)
        if class_start == -1:
            print(f"❌ 未找到 {class_name}")
            continue

        # 找到下一个类的开始位置
        next_class_start = len(content)
        for next_class, _, _ in fixes:
            if next_class != class_name:
                next_marker = f"class {next_class}"
                pos = content.find(next_marker, class_start + 1)
                if pos != -1 and pos < next_class_start:
                    next_class_start = pos

        # 提取类的完整内容
        class_content = content[class_start:next_class_start]

        # 修复参数注释
        class_content = re.sub(
            r"参数: \{\}", f"参数: {{'timeperiod': {period}}}", class_content
        )

        # 修复初始化参数
        class_content = re.sub(
            r"default_params = \{\}",
            f"default_params = {{'timeperiod': {period}}}",
            class_content,
        )

        # 修复硬编码的窗口大小 - 精确匹配
        class_content = re.sub(
            r"volume_sma = volume\.rolling\(window=\d+\)\.mean\(\)",
            f"volume_sma = volume.rolling(window={period}).mean()",
            class_content,
        )

        # 修复factor_name问题
        class_content = re.sub(
            r'logger\.error\(f"计算\{factor_name\}失败: \{e\}"\)',
            'logger.error(f"计算失败: {e}")',
            class_content,
        )
        class_content = re.sub(
            r'name="\{factor_name\}"', "name=self.factor_id", class_content
        )

        # 替换回原内容
        content = content[:class_start] + class_content + content[next_class_start:]

    # 写回文件
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    print("✅ 成交量比率参数化精确修复完成！")

    # 验证修复结果
    print("\n验证修复结果:")
    for class_name, _, period in fixes:
        # 检查参数注释
        param_pattern = f"{class_name}.*参数:.*'timeperiod': {period}"
        if re.search(param_pattern, content):
            print(f"  ✅ {class_name} 参数注释正确")
        else:
            print(f"  ❌ {class_name} 参数注释错误")

        # 检查初始化参数
        init_pattern = f"{class_name}.*default_params = .*'timeperiod': {period}"
        if re.search(init_pattern, content, re.DOTALL):
            print(f"  ✅ {class_name} 初始化参数正确")
        else:
            print(f"  ❌ {class_name} 初始化参数错误")

        # 检查窗口大小
        window_pattern = (
            f"{class_name}.*volume_sma = volume\.rolling\(window={period}\)\.mean\(\)"
        )
        if re.search(window_pattern, content, re.DOTALL):
            print(f"  ✅ {class_name} 窗口大小正确")
        else:
            print(f"  ❌ {class_name} 窗口大小错误")


if __name__ == "__main__":
    fix_volume_ratios_v2()
