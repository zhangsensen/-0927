
"""
ETF份额数据爬虫 (Tushare源)

获取ETF的历史份额数据 (fund_share)
"""

import logging
import os
import time
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import tushare as ts

logger = logging.getLogger(__name__)

class TushareSharesCrawler:
    """Tushare数据源爬虫 - ETF份额"""
    
    def __init__(self, token: str = None):
        self.token = token or os.environ.get("TUSHARE_TOKEN")
        if not self.token:
            # 尝试读取本地配置
            try:
                # 假设通常配置在 ~/.tushare/tushare.csv, tushare库会自动读取
                # 这里只作为备选，如果tushare库找不到再说
                pass 
            except:
                pass
        
        if self.token:
            ts.set_token(self.token)
            self.pro = ts.pro_api()
        else:
            try:
                self.pro = ts.pro_api() # 尝试自动读取配置
            except Exception as e:
                logger.warning(f"Tushare初始化失败: {e}. 请设置 TUSHARE_TOKEN 环境变量")
                self.pro = None

        self.output_dir = Path(__file__).resolve().parents[4] / "raw" / "ETF" / "shares"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fetch_shares_history(self, etf_code: str, start_date: str = "20200101", end_date: str = None) -> pd.DataFrame:
        """
        获取单只ETF的历史份额
        etf_code: 510300 (需转换为 510300.SH)
        """
        if not self.pro:
            logger.error("Tushare未初始化")
            return pd.DataFrame()

        ts_code = self._convert_code(etf_code)
        if not end_date:
            end_date = datetime.now().strftime("%Y%m%d")

        try:
            # Tushare fund_share 接口
            # 限制：每分钟可能有限制，需要注意
            df = self.pro.fund_share(ts_code=ts_code, start_date=start_date, end_date=end_date)
            
            if df is None or df.empty:
                logger.warning(f"{etf_code}: 无份额数据")
                return pd.DataFrame()
                
            # 清洗数据
            # 字段: ts_code, trade_date, fd_share (份额: 万份)
            # 我们只需要 trade_date, fd_share
            # 注意: Tushare返回的是'万份'，通常我们存'份'还是'万份'？
            # 保持原始单位或统一单位。这里统一转换为 '份' (share * 10000) 方便计算? 
            # 还是保持万份? QMT通常是具体股数。为了精度，建议存具体股数。
            # fd_share 是 float
            
            df = df[['trade_date', 'fd_share']].copy()
            df['date'] = pd.to_datetime(df['trade_date'])
            df['shares'] = df['fd_share'] * 10000 # 转换为份
            
            df = df[['date', 'shares']].sort_values('date').reset_index(drop=True)
            return df
            
        except Exception as e:
            logger.error(f"{etf_code}: Tushare请求失败 - {e}")
            return pd.DataFrame()

    def _convert_code(self, code: str) -> str:
        """转换为Tushare格式 (510300 -> 510300.SH)"""
        if code.startswith('5') or code.startswith('1'): # 上海/深圳ETF
             if code.startswith('5'): return f"{code}.SH"
             if code.startswith('1'): return f"{code}.SZ"
        return f"{code}.SH" # 默认

    def update_shares(self, etf_code: str):
        """增量更新"""
        file_path = self.output_dir / f"{etf_code}.csv" # 保持csv格式方便查看? PROJECT用parquet
        # 修正: 项目统一用 parquet
        file_path = self.output_dir / f"shares_{etf_code}.parquet"
        
        old_df = pd.DataFrame()
        start_date = "20200101"
        
        if file_path.exists():
            try:
                old_df = pd.read_parquet(file_path)
                if not old_df.empty:
                    last_date = old_df['date'].max()
                    start_date = (last_date + timedelta(days=1)).strftime("%Y%m%d")
            except:
                pass
        
        # 只有当start_date < today才更新
        if start_date > datetime.now().strftime("%Y%m%d"):
            return

        new_df = self.fetch_shares_history(etf_code, start_date=start_date)
        
        if not new_df.empty:
            if not old_df.empty:
                combined = pd.concat([old_df, new_df])
                combined = combined.drop_duplicates(subset=['date'], keep='last')
                combined = combined.sort_values('date')
            else:
                combined = new_df
                
            combined.to_parquet(file_path, index=False)
            logger.info(f"✅ {etf_code} 份额更新: {len(new_df)} 条")
        else:
            logger.info(f"{etf_code} 份额无更新")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    crawler = TushareSharesCrawler()
    # Test
    crawler.update_shares("510300")
