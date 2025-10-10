"""
Matplotlib中文字体配置

统一配置中文显示，消除字体告警
"""

import logging

import matplotlib
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def setup_chinese_fonts():
    """
    配置Matplotlib中文字体

    解决中文显示乱码和字体告警问题
    """
    try:
        # 尝试常见的CJK字体（注意大小写和完整名称）
        font_candidates = [
            "PingFang SC",  # macOS (常用)
            "Heiti SC",  # macOS
            "STHeiti",  # macOS
            "Songti SC",  # macOS
            "Arial Unicode MS",  # macOS fallback
            "Hiragino Sans GB",  # macOS
            "Microsoft YaHei",  # Windows
            "SimHei",  # Windows
            "Noto Sans CJK SC",  # Linux
            "Source Han Sans SC",  # Adobe开源字体
            "WenQuanYi Micro Hei",  # Linux
        ]

        # 获取系统可用字体
        from matplotlib.font_manager import FontManager

        fm = FontManager()
        available_fonts = {f.name for f in fm.ttflist}

        # 找到第一个可用字体
        selected_font = None
        for font in font_candidates:
            # 尝试精确匹配和模糊匹配
            if font in available_fonts:
                selected_font = font
                break
            # 尝试小写匹配
            lowercase_match = [f for f in available_fonts if font.lower() == f.lower()]
            if lowercase_match:
                selected_font = lowercase_match[0]
                break

        if selected_font:
            # 设置字体
            plt.rcParams["font.sans-serif"] = [
                selected_font,
                "DejaVu Sans",
                "sans-serif",
            ]
            plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

            logger.debug(f"Matplotlib中文字体已配置: {selected_font}")
        else:
            # 关闭unicode_minus警告，使用系统默认字体
            plt.rcParams["axes.unicode_minus"] = False

            # 不打印警告（避免日志噪音）
            # 如果确实需要中文，用户会看到方框并自行处理
            logger.debug(
                f"未找到中文字体，使用系统默认。"
                f"如需中文支持，请安装: PingFang SC (macOS), SimHei (Windows), 或 Noto Sans CJK SC (Linux)"
            )

    except Exception as e:
        # 至少关闭unicode_minus警告
        plt.rcParams["axes.unicode_minus"] = False
        logger.debug(f"配置Matplotlib字体时出现异常，使用默认配置: {e}")


# 模块导入时自动配置
setup_chinese_fonts()
