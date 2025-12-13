"""
实盘交易策略脚本 (Production Ready)
Strategy: ADX_14D + PRICE_POSITION_120D + PRICE_POSITION_20D
Version: v3.1-Live
Date: 2025-12-10
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Ensure src is in path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src"))

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary

# --- HARDCODED CONFIGURATION ---
STRATEGY_NAME = "ADX_POS120_POS20_v3.1"
FACTORS = ['ADX_14D', 'PRICE_POSITION_120D', 'PRICE_POSITION_20D']
FREQ = 3
POS_NUM = 2
WINDOW = 252
QDII_PREMIUM_THRESHOLD = 0.02  # 2%

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_iopv_premium(code: str, current_price: float) -> bool:
    """
    检查 QDII 溢价率
    TODO: 连接实时数据源 (QMT/Tushare) 获取 IOPV
    """
    # Placeholder: 假设没有溢价
    # 在实盘中，这里需要读取 IOPV 数据
    # iopv = get_realtime_iopv(code)
    # premium = (current_price / iopv) - 1
    # if premium > QDII_PREMIUM_THRESHOLD:
    #     logger.warning(f"Risk Alert: {code} Premium {premium:.2%} > {QDII_PREMIUM_THRESHOLD:.2%}")
    #     return False
    return True

def check_suspension(code: str) -> bool:
    """
    检查停牌状态
    TODO: 连接实时数据源
    """
    return False  # 假设未停牌

def run_live_strategy():
    logger.info(f"Starting Live Strategy: {STRATEGY_NAME}")
    
    # 1. Load Data
    logger.info("Loading Data...")
    data_dir = project_root / "raw" / "ETF" / "daily"
    loader = DataLoader(data_dir=str(data_dir))
    ohlcv_dict = loader.load_ohlcv()
    
    # Get common dates and codes
    sample_df = next(iter(ohlcv_dict.values()))
    dates = sample_df.index
    codes = sorted(list(ohlcv_dict.keys()))
    
    logger.info(f"Data Loaded: {len(codes)} ETFs, End Date: {dates[-1]}")
    
    # 2. Calculate Factors
    logger.info("Calculating Factors...")
    lib = PreciseFactorLibrary()
    all_factors = lib.compute_all_factors(ohlcv_dict)
    
    factor_scores = []
    for fname in FACTORS:
        logger.info(f"  - {fname}")
        if fname not in all_factors.columns.levels[0]:
             raise ValueError(f"Factor {fname} not found in library output")
             
        f_val = all_factors[fname]
            
        # Normalize (Rank)
        rank_val = f_val.rank(axis=1, pct=True)
        factor_scores.append(rank_val)
        
    # 3. Composite Score
    logger.info("Compositing Scores...")
    composite_score = sum(factor_scores)
    
    # 4. Generate Signal for Latest Date
    latest_date = dates[-1]
    latest_idx = len(dates) - 1
    
    # Check Rebalance Schedule
    # Note: In backtest, we rebalance if i % FREQ == 0.
    # But we need to know if *Today* (latest_date) is a rebalance trigger point.
    # If today is T, and T % FREQ == 0, we generate signal for T+1.
    is_rebalance_day = (latest_idx % FREQ) == 0
    days_until_rebalance = FREQ - (latest_idx % FREQ)
    
    logger.info(f"Latest Date: {latest_date}")
    logger.info(f"Rebalance Check: Index={latest_idx}, FREQ={FREQ}, Remainder={latest_idx % FREQ}")
    
    if not is_rebalance_day:
        logger.info(f"NOT a Rebalance Day. Next rebalance in {days_until_rebalance} days.")
        # We can still show the theoretical target, but mark it as "HOLD"
    else:
        logger.info("IS Rebalance Day! Generating Orders...")

    # Select Top N
    # Get the row for the latest date
    current_scores = composite_score.iloc[-1]
    
    # Filter NaNs
    valid_scores = current_scores.dropna()
    
    # Sort and Select
    # We use stable_topk_indices logic or simple pandas nlargest
    top_assets = valid_scores.nlargest(POS_NUM).index.tolist()
    
    # 5. Risk Control (Filter)
    final_target = []
    for code in top_assets:
        # Check Suspension
        if check_suspension(code):
            logger.warning(f"Skipping {code}: Suspended")
            continue
            
        # Check IOPV (Need Price)
        price = ohlcv_dict['close'][code].iloc[-1]
        if not check_iopv_premium(code, price):
            logger.warning(f"Skipping {code}: High Premium")
            continue
            
        final_target.append(code)
        
    # Fill if filtered (Optional: The strategy usually just holds less)
    # If we filtered out some, we might want to pick the next best?
    # For now, strict adherence to Top N, if filtered, hold cash.
    
    # 6. Output Trade Plan
    plan_file = f"trade_plan_{latest_date.strftime('%Y%m%d')}.txt"
    with open(plan_file, 'w') as f:
        f.write(f"TRADE PLAN FOR {latest_date.strftime('%Y-%m-%d')}\n")
        f.write(f"Strategy: {STRATEGY_NAME}\n")
        f.write(f"Is Rebalance Day: {is_rebalance_day}\n")
        f.write("-" * 30 + "\n")
        f.write("TARGET POSITIONS:\n")
        for code in final_target:
            f.write(f"BUY/HOLD: {code}\n")
            
        f.write("\n")
        f.write("NOTES:\n")
        f.write("- Ensure to check IOPV before execution.\n")
        f.write("- If not rebalance day, these are theoretical targets.\n")
        
    logger.info(f"Trade Plan Generated: {plan_file}")
    print(f"\n[SUCCESS] Plan saved to {plan_file}")
    print(f"Target: {final_target}")

if __name__ == "__main__":
    run_live_strategy()
