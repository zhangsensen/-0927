"""
新浪财经ETF数据爬虫

数据来源：新浪财经
- ETF历史份额数据（替代东财失败的接口）
- ETF净值数据
- ETF分时数据
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class SinaETFCrawler:
    """新浪财经ETF数据爬虫"""

    # ETF历史数据API（包含份额）
    ETF_HIST_URL = (
        "https://stock.finance.sina.com.cn/fund/api/jsonp.php/CNMarktData.getKLineData"
    )
    # ETF列表API
    ETF_LIST_URL = "https://stock.finance.sina.com.cn/fund/api/jsonp.php/FundDataService.getEtfList"
    # ETF实时行情API
    ETF_QUOTE_URL = "https://hq.sinajs.cn/list"

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

        # 设置headers - 新浪需要特定的Referer
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Referer": "https://finance.sina.com.cn/fund/",
                "Accept": "*/*",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            }
        )

    def _convert_code_to_sina_format(self, code: str, market: str = "SH") -> str:
        """
        将ETF代码转换为新浪格式

        Args:
            code: ETF代码（如 510300）
            market: 市场（SH 或 SZ）

        Returns:
            新浪格式代码（如 sh510300）
        """
        market_lower = market.lower()
        return f"{market_lower}{code}"

    def get_etf_share_history(
        self,
        code: str,
        market: str = "SH",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        获取ETF历史份额数据

        Args:
            code: ETF代码（如 510300）
            market: 市场（SH 或 SZ）
            start_date: 开始日期（格式：YYYY-MM-DD），可选
            end_date: 结束日期（格式：YYYY-MM-DD），可选

        Returns:
            DataFrame包含日期、份额等字段
        """
        symbol = self._convert_code_to_sina_format(code, market)

        params = {
            "symbol": symbol,
        }

        try:
            logger.info(f"Fetching share history for {symbol}...")
            response = self.session.get(self.ETF_HIST_URL, params=params, timeout=30)
            response.raise_for_status()

            # 新浪返回的是JSONP格式：CNMarktData.getKLineData({...})
            text = response.text

            # 提取JSON部分
            match = re.search(r"CNMarktData\.getKLineData\((.*)\)", text, re.DOTALL)
            if not match:
                logger.error(f"Failed to parse JSONP response for {symbol}")
                return pd.DataFrame()

            json_str = match.group(1)
            data = json.loads(json_str)

            if not data or not isinstance(data, list):
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()

            # 解析数据
            records = []
            for item in data:
                # 新浪数据格式: [日期, 开盘价, 收盘价, 最低价, 最高价, 成交量, 份额, 净值]
                if len(item) >= 7:
                    record = {
                        "trade_date": item[0],
                        "open": float(item[1]) if item[1] else None,
                        "close": float(item[2]) if item[2] else None,
                        "low": float(item[3]) if item[3] else None,
                        "high": float(item[4]) if item[4] else None,
                        "volume": int(float(item[5])) if item[5] else None,
                        "fund_share": float(item[6])
                        if item[6]
                        else None,  # 关键字段：份额
                        "nav": float(item[7])
                        if len(item) > 7 and item[7]
                        else None,  # 净值
                    }
                    records.append(record)

            df = pd.DataFrame(records)
            if df.empty:
                return df

            # 转换日期格式
            df["trade_date"] = pd.to_datetime(df["trade_date"])

            # 添加代码信息
            df["code"] = code
            df["market"] = market
            df["symbol"] = symbol

            # 过滤日期范围
            if start_date:
                df = df[df["trade_date"] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df["trade_date"] <= pd.to_datetime(end_date)]

            # 计算份额变动
            if "fund_share" in df.columns and not df["fund_share"].isna().all():
                df["share_change"] = df["fund_share"].diff()
                df["share_change_pct"] = df["fund_share"].pct_change() * 100

            logger.info(
                f"{code}: 获取到 {len(df)} 条历史记录，"
                f"时间范围：{df['trade_date'].min().date()} 至 {df['trade_date'].max().date()}"
            )

            return df.sort_values("trade_date").reset_index(drop=True)

        except Exception as e:
            logger.error(f"获取ETF {code} 份额历史失败: {e}")
            return pd.DataFrame()

    def get_etf_realtime_quote(self, codes: List[Union[str, tuple]]) -> pd.DataFrame:
        """
        获取ETF实时行情

        Args:
            codes: ETF代码列表，可以是字符串（如 "510300.SH"）或元组（如 ("510300", "SH")）

        Returns:
            DataFrame包含实时行情数据
        """
        # 转换代码格式
        symbols = []
        for code in codes:
            if isinstance(code, str) and "." in code:
                # 格式: 510300.SH
                parts = code.split(".")
                symbol = self._convert_code_to_sina_format(parts[0], parts[1])
            elif isinstance(code, tuple) and len(code) == 2:
                # 格式: ("510300", "SH")
                symbol = self._convert_code_to_sina_format(code[0], code[1])
            else:
                # 默认上海
                symbol = self._convert_code_to_sina_format(code, "SH")
            symbols.append(symbol)

        # 构建URL（新浪支持批量查询）
        symbol_str = ",".join(symbols)
        url = f"{self.ETF_QUOTE_URL}={symbol_str}"

        try:
            response = self.session.get(url, timeout=30)
            response.encoding = "gb2312"  # 新浪使用GB2312编码
            response.raise_for_status()

            # 解析返回数据
            text = response.text
            quotes = []

            for symbol in symbols:
                # 查找对应的数据
                pattern = rf'var hq_str_{symbol}="(.*?)";'
                match = re.search(pattern, text)

                if match:
                    data_str = match.group(1)
                    if data_str:
                        parts = data_str.split(",")
                        if len(parts) >= 10:
                            quote = {
                                "symbol": symbol,
                                "code": symbol[2:],  # 去掉sh/sz前缀
                                "market": "SH" if symbol.startswith("sh") else "SZ",
                                "name": parts[0],  # 名称
                                "open": float(parts[1]) if parts[1] else None,
                                "close_yesterday": float(parts[2])
                                if parts[2]
                                else None,
                                "price": float(parts[3]) if parts[3] else None,
                                "high": float(parts[4]) if parts[4] else None,
                                "low": float(parts[5]) if parts[5] else None,
                                "volume": int(float(parts[8])) if parts[8] else None,
                                "amount": float(parts[9]) if parts[9] else None,
                            }
                            quotes.append(quote)

            df = pd.DataFrame(quotes)
            if not df.empty:
                # 计算涨跌幅
                df["change_pct"] = (
                    (df["price"] - df["close_yesterday"]) / df["close_yesterday"] * 100
                ).round(2)
                logger.info(f"获取到 {len(df)} 只ETF实时行情")

            return df

        except Exception as e:
            logger.error(f"获取ETF实时行情失败: {e}")
            return pd.DataFrame()

    def batch_download_shares(
        self,
        etf_list: Optional[pd.DataFrame] = None,
        output_dir: Optional[Path] = None,
        delay: float = 0.5,
    ) -> Dict[str, pd.DataFrame]:
        """
        批量下载ETF份额数据

        Args:
            etf_list: ETF列表DataFrame，需要包含code和market列
            output_dir: 输出目录，为None时不保存文件
            delay: 请求间隔（秒）

        Returns:
            字典，key为ETF代码，value为份额历史DataFrame
        """
        import time

        if etf_list is None:
            # 默认下载主要的宽基ETF
            etf_list = pd.DataFrame(
                {
                    "code": [
                        "510300",
                        "510500",
                        "510050",
                        "159915",
                        "159901",
                        "159949",
                        "512000",
                        "512880",
                        "515000",
                        "513100",
                    ],
                    "market": [
                        "SH",
                        "SH",
                        "SH",
                        "SZ",
                        "SZ",
                        "SZ",
                        "SH",
                        "SH",
                        "SH",
                        "SH",
                    ],
                }
            )

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

            # 添加延迟
            if idx < total - 1:  # 最后一只不需要延迟
                time.sleep(delay)

        logger.info(f"批量下载完成: {len(results)}/{total}")
        return results


if __name__ == "__main__":
    # 测试爬虫
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    crawler = SinaETFCrawler()

    # 测试1: 获取单只ETF份额历史
    print("\n=== 测试获取510300份额历史 ===")
    share_df = crawler.get_etf_share_history("510300", "SH")
    print(f"获取到 {len(share_df)} 条记录")
    if not share_df.empty:
        print("最近5条:")
        print(
            share_df[
                [
                    "trade_date",
                    "close",
                    "fund_share",
                    "share_change",
                    "share_change_pct",
                ]
            ].tail()
        )

    # 测试2: 获取实时行情
    print("\n=== 测试获取实时行情 ===")
    test_codes = [("510300", "SH"), ("159915", "SZ"), ("513100", "SH")]
    quote_df = crawler.get_etf_realtime_quote(test_codes)
    print(f"获取到 {len(quote_df)} 只ETF")
    if not quote_df.empty:
        print(quote_df[["code", "name", "price", "change_pct"]])
