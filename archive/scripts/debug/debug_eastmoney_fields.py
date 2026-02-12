#!/usr/bin/env python3
"""
东财接口字段逆向分析脚本

对比 fflow 接口（只有120天）和 kline 接口（有历史数据）的数据，
试图找到 kline 接口中对应的资金流向字段。
"""

import requests
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

HEADERS = {"User-Agent": "Mozilla/5.0"}

def get_fflow_data(secid):
    """获取标准的资金流向数据 (120天)"""
    url = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
    params = {
        "secid": secid,
        "fields1": "f1,f2,f3,f7",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
        "klt": "101",
        "lmt": "5", # 只取最近5天用于对比
    }
    r = requests.get(url, params=params, headers=HEADERS)
    data = r.json()
    rows = []
    if data.get("data") and data["data"].get("klines"):
        for line in data["data"]["klines"]:
            parts = line.split(",")
            rows.append({
                "date": parts[0],
                "main_net": float(parts[1]),
                "xl_net": float(parts[5]),
                "l_net": float(parts[3]),
                "m_net": float(parts[11]),
                "s_net": float(parts[9])
            })
    return pd.DataFrame(rows)

def get_kline_data(secid):
    """获取K线数据，请求所有可能的扩展字段"""
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    # f51-f57: OHLCV, f62-f65: 只有部分接口有
    # 尝试请求 f51(日期) 和 f62-f70 (可能的资金流)
    fields = "f51,f62,f63,f64,f65,f66,f69,f70"
    params = {
        "secid": secid,
        "fields1": "f1,f2,f3,f7",
        "fields2": fields,
        "klt": "101",
        "lmt": "5", 
    }
    
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=10)
        data = r.json()
    except Exception as e:
        logger.error(f"K线接口请求失败: {e}")
        return pd.DataFrame()

    rows = []
    if data.get("data") and data["data"].get("klines"):
        for line in data["data"]["klines"]:
            parts = line.split(",")
            # f51, f62, f63...
            row = {"date": parts[0]} 
            field_names = fields.split(",")
            for i, val in enumerate(parts):
                fname = field_names[i]
                try:
                    row[fname] = float(val) if val != "-" else 0
                except:
                    row[fname] = val
            rows.append(row)
    return pd.DataFrame(rows)

logger.info("开始对比 510300 数据...")
df_fflow = get_fflow_data("1.510300")
df_kline = get_kline_data("1.510300")

if df_fflow.empty or df_kline.empty:
    logger.error("数据获取失败")
else:
    # 合并对比
    target_date = df_fflow.iloc[-1]["date"]
    logger.info(f"目标日期: {target_date}")
    
    row_f = df_fflow[df_fflow["date"] == target_date].iloc[0]
    row_k = df_kline[df_kline["f51"] == target_date].iloc[0]
    
    logger.info("\n资金流向数据 (fflow):")
    logger.info(f"主力净流入 (main_net): {row_f['main_net']}")
    logger.info(f"超大单净流入 (xl_net): {row_f['xl_net']}")
    
    logger.info("\nK线数据 (kline) 匹配字段:")
    # 遍历K线字段寻找近似值
    targets = {
        "main_net": row_f['main_net'],
        "xl_net": row_f['xl_net']
    }
    
    for k_col, k_val in row_k.items():
        if isinstance(k_val, (int, float)):
            for t_name, t_val in targets.items():
                if abs(k_val - t_val) < 1.0: # 精确匹配
                    logger.info(f"✅ 找到 {t_name} -> {k_col} (值: {k_val})")
                elif abs(k_val - t_val) / (abs(t_val)+1) < 0.01: # 误差1%
                     logger.info(f"⚠️ 近似 {t_name} -> {k_col} (值: {k_val})")

