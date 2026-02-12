#!/usr/bin/env python3
"""
ETF增量数据爬虫 - 资金流向(大单) + 份额变化

数据来源：东方财富
1. 资金流向：超大单/大单/中单/小单 净流入（日级）
2. 份额变化：ETF份额净增减（日级）

用途：为因子库提供非OHLCV维度的增量数据
"""

import json
import time
import logging
import sys
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ETF市场代码映射: 1=上海(51xxxx, 58xxxx), 0=深圳(15xxxx)
def get_market_code(etf_code: str) -> str:
    if etf_code.startswith("15"):
        return "0"  # 深圳
    return "1"  # 上海


class ETFFundFlowCrawler:
    """东财ETF资金流向爬虫"""

    FUND_FLOW_URL = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0",
        "Referer": "https://data.eastmoney.com/",
    }

    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir or ROOT / "raw" / "ETF" / "moneyflow")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503])
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def crawl_fund_flow(self, etf_code: str, limit: int = 2000) -> pd.DataFrame:
        """
        爬取单个ETF的日级资金流向数据

        返回列: date, main_net, xl_net, l_net, m_net, s_net
        - main_net: 主力净流入 (超大单+大单)
        - xl_net: 超大单净流入
        - l_net: 大单净流入
        - m_net: 中单净流入
        - s_net: 小单净流入
        """
        market = get_market_code(etf_code)
        params = {
            "secid": f"{market}.{etf_code}",
            "fields1": "f1,f2,f3,f7",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
            "klt": "101",  # 日线
            "lmt": str(limit),
        }

        try:
            resp = self.session.get(
                self.FUND_FLOW_URL, params=params, headers=self.HEADERS, timeout=15
            )
            data = resp.json()

            if not data.get("data") or not data["data"].get("klines"):
                logger.warning(f"  {etf_code}: 无资金流向数据")
                return pd.DataFrame()

            rows = []
            for line in data["data"]["klines"]:
                parts = line.split(",")
                if len(parts) >= 13:
                    rows.append({
                        "date": parts[0],
                        "main_net": float(parts[1]),     # 主力净流入
                        "main_net_pct": float(parts[6]),  # 主力净占比%
                        "xl_net": float(parts[5]),        # 超大单净流入
                        "xl_net_pct": float(parts[10]),   # 超大单净占比%
                        "l_net": float(parts[3]),         # 大单净流入
                        "l_net_pct": float(parts[8]),     # 大单净占比%
                        "m_net": float(parts[11]),        # 中单净流入
                        "m_net_pct": float(parts[12]),    # 中单净占比%
                        "s_net": float(parts[9]),         # 小单净流入
                        "s_net_pct": float(parts[14]) if len(parts) > 14 else 0,
                    })

            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            return df

        except Exception as e:
            logger.error(f"  {etf_code}: 请求失败 - {e}")
            return pd.DataFrame()

    def crawl_share_change(self, etf_code: str) -> pd.DataFrame:
        """
        从东财详情页爬取ETF份额变动数据
        """
        try:
            from etf_data.crawlers.sources.eastmoney_detail_crawler import EastmoneyDetailCrawler
            crawler = EastmoneyDetailCrawler()
            df = crawler.get_share_positions(etf_code)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            logger.warning(f"  {etf_code}: 份额数据爬取失败 - {e}")
        return pd.DataFrame()

    def batch_crawl(self, etf_codes: list, sleep_sec: float = 0.5):
        """批量爬取所有ETF"""
        total = len(etf_codes)
        success_flow = 0
        success_share = 0

        logger.info(f"🚀 开始爬取 {total} 个ETF的资金流向和份额数据")

        for i, code in enumerate(etf_codes, 1):
            logger.info(f"[{i}/{total}] {code}")

            # 1. 资金流向
            df_flow = self.crawl_fund_flow(code, limit=2000)
            if not df_flow.empty:
                out_path = self.output_dir / f"fund_flow_{code}.parquet"
                df_flow.to_parquet(out_path, index=False)
                logger.info(f"  ✅ 资金流向: {len(df_flow)}天 → {out_path.name}")
                success_flow += 1
            else:
                logger.info(f"  ❌ 资金流向: 无数据")

            # 2. 份额变化
            df_share = self.crawl_share_change(code)
            if not df_share.empty:
                share_dir = ROOT / "raw" / "ETF" / "shares"
                share_dir.mkdir(parents=True, exist_ok=True)
                out_path = share_dir / f"share_change_{code}.parquet"
                df_share.to_parquet(out_path, index=False)
                logger.info(f"  ✅ 份额数据: {len(df_share)}条 → {out_path.name}")
                success_share += 1
            else:
                logger.info(f"  ❌ 份额数据: 无数据")

            time.sleep(sleep_sec)

        logger.info(f"\n{'='*60}")
        logger.info(f"📋 爬取完成:")
        logger.info(f"  资金流向: {success_flow}/{total} 成功")
        logger.info(f"  份额变化: {success_share}/{total} 成功")


if __name__ == "__main__":
    import yaml

    # 从配置读取ETF列表
    config_path = ROOT / "configs" / "combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    etf_codes = config["data"]["symbols"]
    logger.info(f"ETF列表: {len(etf_codes)} 个")

    crawler = ETFFundFlowCrawler()

    # 先测试一个
    logger.info("=" * 60)
    logger.info("🧪 先测试 510300 (Limit=2000)...")
    df = crawler.crawl_fund_flow("510300", limit=2000)
    if not df.empty:
        logger.info(f"✅ 测试成功! 共 {len(df)} 天数据")
        logger.info(f"  列: {df.columns.tolist()}")
        logger.info(f"  日期范围: {df['date'].min()} → {df['date'].max()}")
        logger.info(f"  最近5天:")
        print(df.head(5).to_string(index=False)) # 打印前5天看看历史多长
        print("...")
        print(df.tail(5).to_string(index=False))
        print()

        # 全量爬取
        logger.info("=" * 60)
        crawler.batch_crawl(etf_codes, sleep_sec=0.3)
    else:
        logger.error("❌ 测试失败，检查网络")
