#!/usr/bin/env python3
"""
每日交易信号生成脚本 (Daily Signal Generator)
================================================================================
功能：
1. 读取最新的 ETF 日线数据 (raw/ETF/daily/*.parquet)
2. 计算 "Golden Strategy" 的 5 个核心因子
3. 执行横截面标准化 (Winsorize + Z-Score)
4. 统一因子方向 (MAX_DD_60D 取反)
5. 等权合成总分，输出 Top 2 标的

策略配置 (Golden Strategy v3.1):
- 因子: ADX_14D, MAX_DD_60D, PRICE_POSITION_120D, PRICE_POSITION_20D, SHARPE_RATIO_20D
- 权重: 等权 (Equal Weight)
- 方向: MAX_DD_60D 为反向因子 (Low is Good)，其余为正向
- 持仓: Top 2

用法：
    uv run python scripts/run_daily_signal.py

注意：
    请确保在运行前已更新 raw/ETF/daily/ 下的数据。
    脚本会自动使用数据中最新的日期作为"信号日期"。
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from tabulate import tabulate

# 添加项目根目录到 sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from src.etf_strategy.core.cross_section_processor import CrossSectionProcessor
from src.etf_strategy.core.data_loader import DataLoader

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# 策略定义
STRATEGY_FACTORS = [
    "ADX_14D",
    "MAX_DD_60D",
    "PRICE_POSITION_120D",
    "PRICE_POSITION_20D",
    "SHARPE_RATIO_20D",
]

# 因子方向 (1: High is Good, -1: Low is Good)
FACTOR_DIRECTION = {
    "ADX_14D": 1,
    "MAX_DD_60D": -1,  # 回撤越小越好
    "PRICE_POSITION_120D": 1,
    "PRICE_POSITION_20D": 1,
    "SHARPE_RATIO_20D": 1,
}

# ETF 池定义 (v3.1)
ETF_POOL = {
    "513100": "纳指100",
    "513500": "标普500",
    "159920": "恒生ETF",
    "513050": "中概互联",
    "513130": "恒生科技",
    "510300": "沪深300",
    "510500": "中证500",
    "510050": "上证50",
    "159915": "创业板",
    "512100": "中证1000",
    "512880": "证券ETF",
    "515790": "光伏ETF",
    "512010": "医药ETF",
    "512480": "半导体",
    "512690": "酒ETF",
    "518880": "黄金ETF",
    "511260": "十年国债",  # 避险资产
    "511010": "国债ETF",    # 避险资产
}

# 排除列表 (如果有)
EXCLUDE_LIST = []

def main():
    logger.info("🚀 开始生成每日交易信号 (Golden Strategy v3.1)")
    
    # 1. 加载数据
    data_dir = project_root / "raw/ETF/daily"
    logger.info(f"📂 加载数据: {data_dir}")
    
    try:
        loader = DataLoader(data_dir=str(data_dir))
        # 加载所有 ETF 数据 (DataLoader 会自动处理 vol -> volume 和 adj_ 前缀)
        prices = loader.load_ohlcv(
            etf_codes=list(ETF_POOL.keys()),
            use_cache=False  # 强制不使用缓存，确保读取最新文件
        )
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        return

    latest_date = prices["close"].index[-1]
    logger.info(f"📅 最新数据日期: {latest_date.strftime('%Y-%m-%d')}")
    
    # 2. 计算因子
    logger.info("🔧 计算核心因子...")
    lib = PreciseFactorLibrary()
    
    # 只计算需要的因子 (虽然 compute_all_factors 计算所有，但我们只取需要的)
    # 注意：compute_all_factors 很快，因为它是向量化的
    all_factors = lib.compute_all_factors(prices)
    
    # 提取策略因子
    strategy_factors = {}
    for name in STRATEGY_FACTORS:
        if name not in all_factors:
            logger.error(f"❌ 缺失因子: {name}")
            return
        strategy_factors[name] = all_factors[name]

    # 3. 标准化处理
    logger.info("📐 执行横截面标准化 (Winsorize + Z-Score)...")
    processor = CrossSectionProcessor(verbose=False)
    processed_factors = processor.process_all_factors(strategy_factors)

    # 4. 提取最新一天的因子值并打分
    logger.info("📊 生成最终评分...")
    
    # 获取最新日期的数据
    latest_scores = pd.DataFrame(index=prices["close"].columns)
    
    # 存储原始值用于展示
    raw_values = pd.DataFrame(index=prices["close"].columns)

    for name in STRATEGY_FACTORS:
        # 获取最新一天的标准化值
        factor_series = processed_factors[name].loc[latest_date]
        
        # 获取最新一天的原始值
        raw_series = strategy_factors[name].loc[latest_date]
        
        # 应用方向调整
        direction = FACTOR_DIRECTION.get(name, 1)
        adjusted_score = factor_series * direction
        
        latest_scores[name] = adjusted_score
        raw_values[name] = raw_series

    # 计算总分 (等权求和)
    latest_scores["TOTAL_SCORE"] = latest_scores.sum(axis=1)
    
    # 排序
    ranked = latest_scores.sort_values("TOTAL_SCORE", ascending=False)
    
    # 5. 输出结果
    print("\n" + "=" * 80)
    print(f"🏆 交易信号 | 日期: {latest_date.strftime('%Y-%m-%d')} | 策略: Golden Strategy v3.1")
    print("=" * 80)
    
    # 准备表格数据
    table_data = []
    for code in ranked.index:
        name = ETF_POOL.get(code, code)
        score = ranked.loc[code, "TOTAL_SCORE"]
        
        row = [code, name, f"{score:.4f}"]
        
        # 添加各因子原始值
        for factor in STRATEGY_FACTORS:
            val = raw_values.loc[code, factor]
            # 格式化：百分比或小数
            if "RATIO" in factor or "POSITION" in factor:
                row.append(f"{val:.2f}")
            elif "DD" in factor:
                row.append(f"{val:.2f}%")
            else:
                row.append(f"{val:.2f}")
                
        table_data.append(row)

    headers = ["代码", "名称", "总分"] + STRATEGY_FACTORS
    
    # 打印 Top 10
    print(f"\n📌 Top 10 排名 (建议持仓 Top 2):")
    print(tabulate(table_data[:10], headers=headers, tablefmt="simple_grid"))
    
    # 打印持仓建议
    top2 = ranked.index[:2].tolist()
    print("\n💡 交易建议 (14:50 执行):")
    print(f"   买入/持有: {top2[0]} ({ETF_POOL.get(top2[0])}), {top2[1]} ({ETF_POOL.get(top2[1])})")
    
    # 检查是否有 QDII
    qdii_codes = ["513100", "513500", "159920", "513050", "513130"]
    has_qdii = any(c in qdii_codes for c in top2)
    
    if has_qdii:
        print("\n⚠️  注意: 包含 QDII ETF。")
        print("   - 请忽略 IOPV 溢价 (结构性成本)")
        print("   - 建议使用限价单 (Limit Order) 在卖一/买一价之间成交")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
