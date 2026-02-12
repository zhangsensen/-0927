#!/usr/bin/env python3
"""
ETF资金流向替代数据源调研脚本

目标：寻找能提供超过120天历史数据的免费源
候选：
1. 网易财经 (163) CSV下载
2. 新浪财经 (Sina) API
3. 问财 (iWencai) / pywencai
"""

import requests
import pandas as pd
import logging
from io import StringIO
import traceback

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

HEADERS = {"User-Agent": "Mozilla/5.0"}

def test_netease(symbol="510300"):
    """
    测试网易财经CSV下载
    URL: http://quotes.money.163.com/service/chddata.html
    参数: code=0600030 (0=沪, 1=深), start=20200101, end=20251231, fields=...
    """
    logger.info(f"\n[1/3] 测试网易财经 (163)...")
    
    # 网易代码格式: 0+代码(沪), 1+代码(深)
    # 510300是沪市 => 0510300
    netease_code = f"0{symbol}"
    
    url = "http://quotes.money.163.com/service/chddata.html"
    params = {
        "code": netease_code,
        "start": "20200101",
        "end": "20251231",
        "fields": "TCLOSE;HIGH;LOW;TCAP;MCAP", # 基础字段，看有没有隐藏的
    }
    
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=5)
        if r.status_code == 200 and "日期" in r.text:
            df = pd.read_csv(StringIO(r.text), encoding="gbk")
            logger.info(f"✅ 下载成功! 共 {len(df)} 行")
            logger.info(f"  列: {df.columns.tolist()}")
            
            # 检查是否有资金流相关字段
            flow_keywords = ["流入", "流出", "资金", "大单", "主力"]
            found = [col for col in df.columns if any(k in col for k in flow_keywords)]
            if found:
                logger.info(f"  🎉 发现资金流字段: {found}")
            else:
                logger.info(f"  ❌ 未发现资金流字段 (仅有基础行情)")
        else:
            logger.info(f"❌ 下载失败 (Status {r.status_code})")
    except Exception as e:
        logger.info(f"❌ 请求异常: {e}")


def test_sina(symbol="sh510300"):
    """
    测试新浪财经资金流向接口
    URL: http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/MoneyFlow.ssl_qsfx_lscjflow
    """
    logger.info(f"\n[2/3] 测试新浪财经 (Sina)...")
    
    # 接口1: 个股资金流向（历史）
    url = "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/MoneyFlow.ssl_qsfx_lscjflow"
    params = {
        "page": "1",
        "num": "20",
        "sort": "opendate",
        "asc": "0",
        "daima": symbol,
    }
    
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=5)
        if r.status_code == 200 and len(r.text) > 10:
            # 新浪返回的是非标准JSON (key没有引号)，需要处理
            # 简单判断是否包含有效数据
            if "r0_net" in r.text or "r0" in r.text: # r0通常是大单
                 logger.info(f"✅ 发现数据! 响应片段: {r.text[:100]}...")
                 logger.info(f"  可能是有效的历史资金流接口!")
            else:
                 logger.info(f"⚠️ 响应内容不包含预期字段: {r.text[:100]}")
        else:
            logger.info(f"❌ 请求失败或无数据")
    except Exception as e:
        logger.info(f"❌ 请求异常: {e}")


def test_pywencai():
    """测试 pywencai 库是否存在"""
    logger.info(f"\n[3/3] 测试问财 (pywencai)...")
    try:
        import pywencai
        logger.info(f"✅ pywencai 已安装")
    except ImportError:
        logger.info(f"❌ pywencai 未安装 (需要 Node.js 环境)")

if __name__ == "__main__":
    test_netease("510300")
    test_sina("sh510300")
    test_pywencai()
