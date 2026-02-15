#!/usr/bin/env python3
"""
东方财富ETF数据爬虫

数据来源：东方财富网
- ETF实时行情
- ETF份额规模（申赎数据）
- ETF折溢价率
- ETF资金流入流出
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class EastmoneyETFCrawler:
    """东方财富ETF数据爬虫"""

    # ETF列表API
    ETF_LIST_URL = "http://push2.eastmoney.com/api/qt/clist/get"
    # ETF历史份额API
    ETF_SHARE_URL = "http://datainterface.eastmoney.com/EM_DataCenter/JS.aspx"
    # ETF实时行情API
    ETF_QUOTE_URL = "http://push2.eastmoney.com/api/qt/ulist.np/get"

    def __init__(self):
        self.session = requests.Session()

        # 配置重试
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # 设置headers
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Referer": "http://fund.eastmoney.com/",
            }
        )

    def get_etf_list(self) -> pd.DataFrame:
        """
        获取ETF列表

        Returns:
            DataFrame包含ETF代码、名称等信息
        """
        params = {
            "pn": 1,
            "pz": 1000,  # 最多1000只
            "po": 1,
            "np": 1,
            "ut": "bd1d9ddb04089700cf9c27f6f7426281",
            "fltt": 2,
            "invt": 2,
            "fid": "f3",
            "fs": "b:MK0021,b:MK0022,b:MK0023,b:MK0024",  # ETF基金分类
            "fields": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18",
            "_": int(datetime.now().timestamp() * 1000),
        }

        try:
            response = self.session.get(self.ETF_LIST_URL, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            if data.get("data") and data["data"].get("diff"):
                etfs = []
                diff_data = data["data"]["diff"]
                # diff可能是字典或列表
                if isinstance(diff_data, dict):
                    items = diff_data.values()
                else:
                    items = diff_data

                for item in items:
                    etf = {
                        "code": item.get("f12"),  # 基金代码
                        "name": item.get("f14"),  # 基金名称
                        "market": "SH" if item.get("f13") == 1 else "SZ",  # 市场
                        "ts_code": f"{item.get('f12')}.{'SH' if item.get('f13') == 1 else 'SZ'}",
                    }
                    etfs.append(etf)

                df = pd.DataFrame(etfs)
                logger.info(f"获取到 {len(df)} 只ETF")
                return df

        except Exception as e:
            logger.error(f"获取ETF列表失败: {e}")

        return pd.DataFrame()

    def get_etf_share_history(self, fund_code: str, market: str = "SH") -> pd.DataFrame:
        """
        获取ETF历史份额数据

        Args:
            fund_code: ETF代码（如 510300）
            market: 市场（SH 或 SZ）

        Returns:
            DataFrame包含日期、份额等字段
        """
        # 转换市场代码
        mkt = 1 if market == "SH" else 0

        params = {
            "type": "GPKH",  # 股票份额变化
            "sty": "MKT",
            "st": "2",  # 按日期排序
            "sr": "-1",  # 倒序
            "p": "1",
            "ps": "1000",  # 获取全部历史
            "js": "var data={pages:(pc),data:[(x)]}",
            "mkt": mkt,
            "code": fund_code,
            "rt": "53450604",
        }

        try:
            response = self.session.get(self.ETF_SHARE_URL, params=params, timeout=30)
            response.raise_for_status()

            # 解析返回的JavaScript数据
            text = response.text
            # 提取JSON数组
            match = re.search(r"data:\[(.*?)\]", text)
            if match:
                json_str = f"[{match.group(1)}]"
                data = json.loads(json_str)

                records = []
                for item in data:
                    # item格式: "日期,基金份额(万份),基金规模(亿元)"
                    parts = item.split(",")
                    if len(parts) >= 2:
                        records.append(
                            {
                                "trade_date": parts[0],
                                "fund_share": float(parts[1])
                                if parts[1]
                                else None,  # 万份
                                "fund_scale": float(parts[2])
                                if len(parts) > 2 and parts[2]
                                else None,  # 亿元
                            }
                        )

                df = pd.DataFrame(records)
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df = df.sort_values("trade_date")
                df["fund_code"] = fund_code

                logger.info(f"{fund_code}: 获取到 {len(df)} 条份额历史")
                return df

        except Exception as e:
            logger.error(f"获取ETF {fund_code} 份额历史失败: {e}")

        return pd.DataFrame()

    def get_etf_realtime_quote(self, codes: List[str]) -> pd.DataFrame:
        """
        获取ETF实时行情

        Args:
            codes: ETF代码列表（如 ["510300", "510500"]）

        Returns:
            DataFrame包含实时行情数据
        """
        # 构建代码字符串
        code_str = ",".join(
            [f"1.{code}" if code.startswith("5") else f"0.{code}" for code in codes]
        )

        params = {
            "ut": "bd1d9ddb04089700cf9c27f6f7426281",
            "fltt": 2,
            "invt": 2,
            "fields": "f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,f31,f32,f33,f34,f35,f36,f37,f38,f39,f40,f41,f42,f43,f44,f45,f46,f47,f48,f49,f50,f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65,f66,f67,f68,f69,f70,f71,f72,f73,f74,f75,f76,f77,f78,f79,f80,f81,f82,f83,f84,f85,f86,f87,f88,f89,f90,f91,f92,f93,f94,f95,f96,f97,f98,f99,f100",
            "secids": code_str,
            "_": int(datetime.now().timestamp() * 1000),
        }

        try:
            response = self.session.get(self.ETF_QUOTE_URL, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            if data.get("data") and data["data"].get("diff"):
                quotes = []
                diff_data = data["data"]["diff"]
                # diff可能是字典或列表
                if isinstance(diff_data, dict):
                    items = diff_data.values()
                else:
                    items = diff_data

                for item in items:
                    quote = {
                        "code": item.get("f12"),
                        "name": item.get("f14"),
                        "price": item.get("f2"),  # 最新价
                        "change_pct": item.get("f3"),  # 涨跌幅
                        "volume": item.get("f5"),  # 成交量
                        "amount": item.get("f6"),  # 成交额
                        "iopv": item.get("f80"),  # 基金份额参考净值(IOPV)
                        "premium_rate": item.get("f187"),  # 折溢价率
                    }
                    quotes.append(quote)

                df = pd.DataFrame(quotes)
                logger.info(f"获取到 {len(df)} 只ETF实时行情")
                return df

        except Exception as e:
            logger.error(f"获取ETF实时行情失败: {e}")

        return pd.DataFrame()

    def batch_download_etf_shares(
        self, etf_list: Optional[pd.DataFrame] = None, output_dir: Optional[Path] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        批量下载ETF份额数据

        Args:
            etf_list: ETF列表DataFrame，为None时自动获取
            output_dir: 输出目录，为None时不保存文件

        Returns:
            字典，key为ETF代码，value为份额历史DataFrame
        """
        if etf_list is None:
            etf_list = self.get_etf_list()

        results = {}
        total = len(etf_list)

        for idx, row in etf_list.iterrows():
            code = row["code"]
            market = row.get("market", "SH")

            logger.info(f"[{idx + 1}/{total}] 下载 {code} 份额数据...")

            df = self.get_etf_share_history(code, market)
            if not df.empty:
                results[code] = df

                # 保存到文件
                if output_dir:
                    output_dir = Path(output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)

                    file_path = output_dir / f"{code}_{market}_shares.parquet"
                    df.to_parquet(file_path, index=False)
                    logger.info(f"  已保存: {file_path}")

            # 添加延迟避免被封
            import time

            time.sleep(0.5)

        logger.info(f"批量下载完成: {len(results)}/{total}")
        return results


if __name__ == "__main__":
    # 测试爬虫
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    crawler = EastmoneyETFCrawler()

    # 测试1: 获取ETF列表
    print("\n=== 获取ETF列表 ===")
    etf_list = crawler.get_etf_list()
    print(f"共 {len(etf_list)} 只ETF")
    print(etf_list.head())

    # 测试2: 获取单只ETF份额历史（沪深300ETF）
    print("\n=== 获取510300份额历史 ===")
    share_df = crawler.get_etf_share_history("510300", "SH")
    print(f"共 {len(share_df)} 条记录")
    print(share_df.tail())

    # 测试3: 获取实时行情
    print("\n=== 获取实时行情 ===")
    test_codes = ["510300", "510500", "159915"]
    quote_df = crawler.get_etf_realtime_quote(test_codes)
    print(quote_df)
