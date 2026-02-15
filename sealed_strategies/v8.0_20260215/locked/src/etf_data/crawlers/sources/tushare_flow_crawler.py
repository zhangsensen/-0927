
"""
ETF资金流向爬虫 (Tushare源)

获取ETF的历史资金流向数据 (moneyflow)
用于补全超过120天的历史数据
"""

import logging
import os
import time
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import tushare as ts

logger = logging.getLogger(__name__)

class TushareFlowCrawler:
    """Tushare数据源爬虫 - ETF资金流向"""
    
    def __init__(self, token: str = None):
        self.token = token or os.environ.get("TUSHARE_TOKEN")
        
        self.pro = None
        if self.token:
            ts.set_token(self.token)
            self.pro = ts.pro_api()
        else:
            try:
                self.pro = ts.pro_api()
            except:
                pass

        self.output_dir = Path(__file__).resolve().parents[4] / "raw" / "ETF" / "moneyflow"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fetch_flow_history(self, etf_code: str, start_date: str = "20200101", end_date: str = None) -> pd.DataFrame:
        """
        获取单只ETF的历史资金流向 (映射到东财格式)
        etf_code: 510300 (需转换为 510300.SH)
        """
        if not self.pro:
            return pd.DataFrame()

        ts_code = self._convert_code(etf_code)
        if not end_date:
            end_date = datetime.now().strftime("%Y%m%d")

        try:
            # Tushare moneyflow 接口 (个股/ETF资金流向)
            # 字段: trade_date, buy_sm_amount, sell_sm_amount, ... (万元)
            df = self.pro.moneyflow(ts_code=ts_code, start_date=start_date, end_date=end_date)
            
            if df is None or df.empty:
                return pd.DataFrame()
                
            # 计算净流入 (万元)
            # Tushare: amount is in 万元 (Ten Thousand CNY)
            # Eastmoney crawler structure:
            # date, main_net, main_net_pct, xl_net, xl_net_pct, l_net, l_net_pct, m_net, m_net_pct, s_net, s_net_pct
            
            df = df.fillna(0)
            
            # Net Flows
            df['s_net'] = df['buy_sm_amount'] - df['sell_sm_amount']
            df['m_net'] = df['buy_md_amount'] - df['sell_md_amount']
            df['l_net'] = df['buy_lg_amount'] - df['sell_lg_amount']
            df['xl_net'] = df['buy_elg_amount'] - df['sell_elg_amount']
            df['main_net'] = df['l_net'] + df['xl_net']
            
            # Pct needs total turnover
            # Tushare moneyflow doesn't explicit give turnover, but we can sum buys and sells?
            # Or use amount directly?
            # Eastmoney's pct is "net_inflow / total_turnover"? Or "net_inflow / market_cap"?
            # Usually it's net_inflow / turnover.
            # Total turnover = buy_sm + buy_md + ... + sell_sm + ... ? No, turnover is usually (buy_val + sell_val) / 2
            
            # Let's approximate turnover from the flows provided
            buy_total = df['buy_sm_amount'] + df['buy_md_amount'] + df['buy_lg_amount'] + df['buy_elg_amount']
            sell_total = df['sell_sm_amount'] + df['sell_md_amount'] + df['sell_lg_amount'] + df['sell_elg_amount']
            total_turnover = (buy_total + sell_total) / 2
            
            # Avoid division by zero
            total_turnover = total_turnover.replace(0, 1)
            
            df['s_net_pct'] = (df['s_net'] / total_turnover) * 100
            df['m_net_pct'] = (df['m_net'] / total_turnover) * 100
            df['l_net_pct'] = (df['l_net'] / total_turnover) * 100
            df['xl_net_pct'] = (df['xl_net'] / total_turnover) * 100
            df['main_net_pct'] = (df['main_net'] / total_turnover) * 100
            
            # Rename and Select
            df['date'] = pd.to_datetime(df['trade_date'])
            
            result = df[[
                'date', 
                'main_net', 'main_net_pct',
                'xl_net', 'xl_net_pct',
                'l_net', 'l_net_pct',
                'm_net', 'm_net_pct',
                's_net', 's_net_pct'
            ]].sort_values('date').reset_index(drop=True)
            
            return result
            
        except Exception as e:
            logger.error(f"{etf_code}: Tushare Flow请求失败 - {e}")
            return pd.DataFrame()

    def _convert_code(self, code: str) -> str:
        if code.startswith('5') or code.startswith('1'):
             if code.startswith('5'): return f"{code}.SH"
             if code.startswith('1'): return f"{code}.SZ"
        return f"{code}.SH"

    def update_flow_history(self, etf_code: str):
        """增量补全"""
        # Eastmoney file name: fund_flow_{code}.parquet
        file_path = self.output_dir / f"fund_flow_{etf_code}.parquet"
        
        old_df = pd.DataFrame()
        earliest_date = datetime.now()
        
        if file_path.exists():
            try:
                old_df = pd.read_parquet(file_path)
                if not old_df.empty:
                    earliest_date = old_df['date'].min()
            except:
                pass
        
        # If we have data back to 2020-01-01, skip
        target_start = datetime(2020, 1, 1)
        if earliest_date <= target_start + timedelta(days=30):
            # logger.info(f"{etf_code} 历史数据已存在 (最早: {earliest_date.date()})")
            return 
            
        # Fetch history before earliest_date
        # Tushare limits: verify if we can fetch all at once or need batches
        # moneyflow limit is usually 5000 rows, enough for 5 years
        
        end_str = (earliest_date - timedelta(days=1)).strftime("%Y%m%d")
        new_df = self.fetch_flow_history(etf_code, start_date="20200101", end_date=end_str)
        
        if not new_df.empty:
            if not old_df.empty:
                # Merge: new_df (old history) + old_df (recent data)
                combined = pd.concat([new_df, old_df])
                combined = combined.drop_duplicates(subset=['date'], keep='last')
                combined = combined.sort_values('date').reset_index(drop=True)
            else:
                combined = new_df
                
            combined.to_parquet(file_path, index=False)
            logger.info(f"✅ {etf_code} 资金流补全: {len(new_df)} 条 (最早: {combined['date'].min().date()})")
        else:
             logger.info(f"{etf_code} 无更早历史数据")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    crawler = TushareFlowCrawler()
    # Test
    crawler.update_flow_history("510300")
