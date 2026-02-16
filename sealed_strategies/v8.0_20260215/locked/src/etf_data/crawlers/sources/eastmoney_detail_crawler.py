"""
东方财富ETF详情页爬虫

数据来源：fund.eastmoney.com/pingzhongdata/{code}.js
- ETF历史净值
- ETF份额变动（Data_fundSharesPositions）
- ETF申购赎回数据（Data_buySedemption）
- ETF持有人结构（Data_holderStructure）
- ETF资产配置（Data_assetAllocation）
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


class EastmoneyDetailCrawler:
    """东方财富ETF详情页数据爬虫"""

    BASE_URL = "http://fund.eastmoney.com/pingzhongdata/{code}.js"

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
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Referer": "http://fund.eastmoney.com/",
                "Accept": "*/*",
            }
        )

    def _fetch_js_data(self, code: str) -> str:
        """获取JS数据文件"""
        url = self.BASE_URL.format(code=code)

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"获取 {code} 数据失败: {e}")
            return ""

    def _extract_json(self, text: str, var_name: str) -> Optional[Union[list, dict]]:
        """从JS文本中提取JSON变量"""
        # 尝试多种匹配模式
        patterns = [
            rf"var {re.escape(var_name)}\s*=\s*(\[.*?\]);",
            rf"var {re.escape(var_name)}\s*=\s*(\{{.*?\}});",
            rf"{re.escape(var_name)}\s*=\s*(\[.*?\]);",
            rf"{re.escape(var_name)}\s*=\s*(\{{.*?\}});",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue

        return None

    def get_networth_history(self, code: str) -> pd.DataFrame:
        """
        获取ETF历史净值数据

        Args:
            code: ETF代码（如 510300）

        Returns:
            DataFrame包含日期、单位净值、累计净值等
        """
        text = self._fetch_js_data(code)
        if not text:
            return pd.DataFrame()

        # 提取Data_netWorthTrend（单位净值趋势）
        data = self._extract_json(text, "Data_netWorthTrend")
        if not data:
            logger.warning(f"{code}: 未找到净值数据")
            return pd.DataFrame()

        records = []
        for item in data:
            record = {
                "trade_date": datetime.fromtimestamp(item["x"] / 1000),
                "nav": item.get("y"),  # 单位净值
                "equity_return": item.get("equityReturn"),  # 净值回报
                "unit_money": item.get("unitMoney", ""),  # 每份派送金额
            }
            records.append(record)

        df = pd.DataFrame(records)
        logger.info(f"{code}: 获取到 {len(df)} 条净值记录")
        return df

    def get_share_positions(self, code: str) -> pd.DataFrame:
        """
        获取ETF份额仓位数据

        Args:
            code: ETF代码（如 510300）

        Returns:
            DataFrame包含日期、份额占比等
        """
        text = self._fetch_js_data(code)
        if not text:
            return pd.DataFrame()

        # 提取Data_fundSharesPositions
        data = self._extract_json(text, "Data_fundSharesPositions")
        if not data:
            logger.warning(f"{code}: 未找到份额仓位数据")
            return pd.DataFrame()

        records = []
        for item in data:
            records.append(
                {
                    "trade_date": datetime.fromtimestamp(item[0] / 1000),
                    "share_position_pct": item[1],  # 份额仓位占比
                }
            )

        df = pd.DataFrame(records)
        logger.info(f"{code}: 获取到 {len(df)} 条份额仓位记录")
        return df

    def get_buy_sedemption(self, code: str) -> Optional[pd.DataFrame]:
        """
        获取ETF申购赎回数据（季报数据）

        Args:
            code: ETF代码（如 510300）

        Returns:
            DataFrame包含申购、赎回数据，或None
        """
        text = self._fetch_js_data(code)
        if not text:
            return None

        # 提取Data_buySedemption
        data = self._extract_json(text, "Data_buySedemption")
        if not data or "series" not in data:
            logger.warning(f"{code}: 未找到申赎数据")
            return None

        # 解析数据
        series_data = data["series"]
        categories = data.get("categories", [])

        records = []
        for i, cat in enumerate(categories):
            record = {"period": cat}
            for series in series_data:
                name = series["name"]
                values = series["data"]
                if i < len(values):
                    record[name] = values[i]
            records.append(record)

        df = pd.DataFrame(records)
        logger.info(f"{code}: 获取到 {len(df)} 期申赎数据")
        return df

    def get_holder_structure(self, code: str) -> Optional[pd.DataFrame]:
        """
        获取ETF持有人结构数据

        Args:
            code: ETF代码（如 510300）

        Returns:
            DataFrame包含机构、个人持有比例
        """
        text = self._fetch_js_data(code)
        if not text:
            return None

        # 提取Data_holderStructure
        data = self._extract_json(text, "Data_holderStructure")
        if not data or "series" not in data:
            logger.warning(f"{code}: 未找到持有人结构数据")
            return None

        # 解析数据
        series_data = data["series"]
        categories = data.get("categories", [])

        records = []
        for i, cat in enumerate(categories):
            record = {"date": cat}
            for series in series_data:
                name = series["name"]
                values = series["data"]
                if i < len(values):
                    record[name] = values[i]
            records.append(record)

        df = pd.DataFrame(records)
        logger.info(f"{code}: 获取到 {len(df)} 期持有人结构数据")
        return df

    def get_top10_holders(self, code: str) -> pd.DataFrame:
        """
        获取ETF前十大持有人数据（包含北向资金HKSCC）
        
        Args:
            code: ETF代码（如 510300）
            
        Returns:
            DataFrame包含日期、持有人名称、持有份额、占比等
        """
        # 尝试方法1: 接口 API
        # url = "http://fundf10.eastmoney.com/FundArchivesDatas.aspx"
        # params = {"type": "sdcyr", "code": code, "year": ""}
        
        # 尝试方法2: 直接抓取网页 (更稳定)
        url = f"http://fundf10.eastmoney.com/cyrjg_{code}.html"
        
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Referer": "http://fund.eastmoney.com/",
            }
            
            response = self.session.get(url, headers=headers, timeout=10)
            response.encoding = "utf-8"
            
            # logger.info(f"Page Status: {response.status_code}")
            
            if response.status_code != 200:
                return pd.DataFrame()
                
            # 解析页面中的表格
            # 通常第一个表是持有人结构(机构/个人)，第二个表可能是十大持有人(或者最新的)
            # 需要遍历找到包含 "持有人名称" 或 "份额" 的表格
            from io import StringIO
            dfs = pd.read_html(StringIO(response.text))
            
            target_df = pd.DataFrame()
            
            logger.info(f"Found {len(dfs)} tables")
            for i, df in enumerate(dfs):
                cols = [str(c) for c in df.columns]
                logger.info(f"Table {i} cols: {cols}")
                
                # 检查列名是否符合十大持有人特征
                # 东财表格列名通常是: 序号, 持有人名称, 持有份额, 占总份额比例, 增减, 占流通股比例, 截止日期

                # 东财表格列名通常是: 序号, 持有人名称, 持有份额, 占总份额比例, 增减, 占流通股比例, 截止日期
                # 或者: 2024-06-30 (作为表头) -> 这种情况下列名是日期
                
                # 策略: 寻找包含 "份额" 或 "比例" 的宽表
                if any("份额" in c for c in cols) or any("名称" in c for c in cols):
                    # 进一步检查内容
                    if len(df) > 0:
                        target_df = df
                        # logger.info(f"Found candidate table with cols: {cols}")
                        # 我们可以假设包含最多行的那个表是历史十大持有人列表? 
                        # 不，这通常是按季度显示的。
                        break
            
            # 如果没找到，尝试特定索引
            if target_df.empty and len(dfs) >= 2:
                 # 往往第二个表是十大持有人明细
                 target_df = dfs[1]

            if target_df.empty:
                logger.warning(f"{code}: 未找到十大持有人表格")
                return pd.DataFrame()
            
            # 增加代码标识
            target_df["code"] = code
            
            # 简单清洗: 如果列名包含日期，尝试转换
            # 东财网页版可能把日期放在表头第一行，或者是多个表
            # 这里先原样返回，后续在入库时清洗
            
            logger.info(f"{code}: 获取到持有人数据表格 {len(target_df)} 行")
            return target_df
            
        except Exception as e:
            logger.error(f"获取 {code} 十大持有人数据失败: {e}")
            return pd.DataFrame()

    def get_all_data(self, code: str) -> Dict[str, pd.DataFrame]:
        """
        获取所有可用数据

        Args:
            code: ETF代码（如 510300）

        Returns:
            字典包含所有数据类型
        """
        return {
            "networth": self.get_networth_history(code),
            "share_positions": self.get_share_positions(code),
            "buy_sedemption": self.get_buy_sedemption(code),
            "holder_structure": self.get_holder_structure(code),
            "top10_holders": self.get_top10_holders(code),
        }


if __name__ == "__main__":
    # 测试爬虫
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    crawler = EastmoneyDetailCrawler()

    # 测试获取510300的所有数据
    print("\n=== 测试获取510300数据 ===")
    data_dict = crawler.get_all_data("510300")

    for data_type, df in data_dict.items():
        if df is not None and not df.empty:
            print(f"\n{data_type}: {len(df)} 条记录")
            print(df.head(3))
        else:
            print(f"\n{data_type}: 无数据")
