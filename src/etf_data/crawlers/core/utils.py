"""
爬虫工具函数
"""

import json
import re
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd


def extract_json_from_js(js_text: str, var_name: str) -> Optional[Dict]:
    """
    从JavaScript代码中提取JSON数据

    例如：var data = {...} 中提取 {...}

    Args:
        js_text: JavaScript代码文本
        var_name: 变量名

    Returns:
        解析后的字典或None
    """
    pattern = rf"var\s+{re.escape(var_name)}\s*=\s*(.*?);"
    match = re.search(pattern, js_text, re.DOTALL)

    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    return None


def parse_eastmoney_date(date_str: str) -> Optional[datetime]:
    """
    解析东财日期格式

    支持格式：
    - 2024-01-01
    - 2024/01/01
    - 20240101

    Args:
        date_str: 日期字符串

    Returns:
        datetime对象或None
    """
    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y%m%d",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    return None


def validate_etf_data(df: pd.DataFrame, required_cols: list) -> bool:
    """
    验证ETF数据完整性

    Args:
        df: 数据DataFrame
        required_cols: 必需的列名列表

    Returns:
        是否有效
    """
    if df.empty:
        return False

    # 检查必需列
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        return False

    # 检查数据有效性
    if df[required_cols].isnull().all().any():
        print("Some required columns are all null")
        return False

    return True
