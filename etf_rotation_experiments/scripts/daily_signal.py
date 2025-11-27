#!/usr/bin/env python3
"""
实盘信号生成器 (Daily Signal Generator)
功能: 基于 ML 排序选出的 Top-1 组合，生成明日调仓信号
"""
import sys
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.data_loader import DataLoader
from core.precise_factor_library_v2 import PreciseFactorLibrary
from core.cross_section_processor import CrossSectionProcessor
from core.simple_trader import SimpleTrader
from strategies.backtest.production_backtest import compute_spearman_ic_numba

# ================= 配置区域 =================
# ML 排序 Top-1 组合 (Platinum Candidate - Rank 10813)
# Annual Ret: 20.09%, Sharpe: 0.97, Max DD: -17.75%
TARGET_COMBO = [
    "OBV_SLOPE_10D",
    "PRICE_POSITION_20D",
    "RSI_14",
    "SLOPE_20D",
    "VORTEX_14D"
]
LOOKBACK_WINDOW = 252  # IC 权重计算窗口
TOP_N = 5              # 持仓数量
# ===========================================

def main():
    parser = argparse.ArgumentParser(description="实盘信号生成与交易执行")
    parser.add_argument("--execute", action="store_true", help="执行交易并记录日志")
    parser.add_argument("--capital", type=float, default=100000.0, help="初始资金 (仅首次运行时有效)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"🚀 实盘信号生成 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.execute:
        print("⚠️  注意: 交易执行模式已开启 (将写入交易日志)")
    print("=" * 60)
    print(f"策略组合: {' + '.join(TARGET_COMBO)}")
    print(f"持仓数量: Top {TOP_N}")
    print("-" * 60)

    # 1. 加载数据
    print("1. 加载数据...")
    # 数据目录在项目根目录的上级目录 (repo root)
    REPO_ROOT = PROJECT_ROOT.parent
    
    loader = DataLoader(
        data_dir=REPO_ROOT / "raw" / "ETF" / "daily",
        cache_dir=REPO_ROOT / "raw" / "cache"
    )
    # 加载所有 ETF 数据
    # 注意: 这里需要确保 data_loader 能自动发现所有 ETF
    # 我们使用默认配置中的 ETF 列表 (如果能获取到)
    # 这里简化处理，直接加载目录下所有 CSV
    # 或者使用 configs/etf_pools.yaml
    
    # 尝试读取 etf_pools.yaml
    import yaml
    pool_config = REPO_ROOT / "configs" / "etf_pools.yaml"
    if pool_config.exists():
        with open(pool_config) as f:
            conf = yaml.safe_load(f)
            etf_codes = conf.get("etf_pool", [])
    else:
        # Fallback: 扫描目录
        etf_codes = [f.stem for f in (REPO_ROOT / "raw" / "ETF" / "daily").glob("*.csv")]
    
    ohlcv = loader.load_ohlcv(etf_codes=etf_codes)
    print(f"   覆盖 {len(ohlcv['close'].columns)} 只 ETF, {len(ohlcv['close'])} 个交易日")

    # 2. 计算因子
    print("2. 计算因子...")
    lib = PreciseFactorLibrary()
    factors_df = lib.compute_all_factors(ohlcv)
    
    # 3. 横截面标准化
    print("3. 横截面标准化...")
    processor = CrossSectionProcessor(
        lower_percentile=2.5,
        upper_percentile=97.5
    )
    # 转换为 dict 格式供 processor 使用
    factors_dict = {name: factors_df[name] for name in lib.list_factors()}
    standardized = processor.process_all_factors(factors_dict)
    
    # 4. 提取目标因子数据
    print("4. 计算组合信号...")
    # 准备数据: (T, N, F)
    T, N = ohlcv['close'].shape
    F = len(TARGET_COMBO)
    
    factors_data = np.zeros((T, N, F))
    for i, fname in enumerate(TARGET_COMBO):
        if fname in standardized:
            factors_data[:, :, i] = standardized[fname].values
        else:
            print(f"❌ 错误: 因子 {fname} 未找到!")
            return

    returns = ohlcv['close'].pct_change().values
    
    # 5. 计算 IC 权重 (使用最近 LOOKBACK_WINDOW 天)
    # 取倒数第2天到倒数第1天的数据来计算IC (因为最后一天没有收益)
    # 实际上我们需要截至 T-1 的数据来决定 T 的持仓
    # 今天的信号基于截至昨天的收盘数据 (如果今天是交易日结束)
    # 或者基于今天收盘数据 (如果为了明天交易)
    
    # 假设当前是 T 日收盘后，我们要生成 T+1 日的信号
    # 我们使用 T 日及之前的因子，和 T 日及之前的收益来计算权重?
    # 不，权重是基于历史表现。
    # 我们使用 T-LOOKBACK 到 T 的数据计算 IC
    
    valid_start = T - LOOKBACK_WINDOW - 1
    if valid_start < 0:
        print("⚠️ 数据不足计算完整 IC 窗口，使用可用数据")
        valid_start = 0
        
    hist_factors = factors_data[valid_start:-1] # T-1 之前的因子
    hist_returns = returns[valid_start:-1]      # T-1 之前的收益
    
    # 计算权重
    ics = np.zeros(F)
    for f in range(F):
        ics[f] = compute_spearman_ic_numba(hist_factors[:, :, f], hist_returns)
    
    abs_ics = np.abs(ics)
    if abs_ics.sum() > 0:
        weights = abs_ics / abs_ics.sum()
    else:
        weights = np.ones(F) / F
        
    print(f"   因子权重 (基于过去 {LOOKBACK_WINDOW} 天 IC):")
    for i, fname in enumerate(TARGET_COMBO):
        print(f"   - {fname:<30}: {weights[i]:.4f} (IC: {ics[i]:.4f})")

    # 6. 计算最终信号 (使用最新一天的因子值)
    latest_factors = factors_data[-1] # (N, F)
    
    # 加权求和
    # 处理 NaN: 如果某个因子是 NaN，则该 ETF 信号为 NaN
    # 或者忽略该因子? production_backtest 是忽略 NaN 因子并归一化权重
    
    final_scores = np.zeros(N)
    valid_mask = np.ones(N, dtype=bool)
    
    for n in range(N):
        score = 0.0
        w_sum = 0.0
        for f in range(F):
            val = latest_factors[n, f]
            if not np.isnan(val):
                score += val * weights[f]
                w_sum += weights[f]
            else:
                # 如果有因子缺失，该 ETF 降级或忽略?
                # 简单起见，如果有因子缺失，权重不加
                pass
        
        if w_sum > 0:
            final_scores[n] = score / w_sum
        else:
            final_scores[n] = -999 # 无效
            valid_mask[n] = False

    # 7. 排序并输出
    print("-" * 60)
    print(f"📅 信号日期: {ohlcv['close'].index[-1].strftime('%Y-%m-%d')}")
    print("🏆 推荐持仓 (Top 5):")
    
    # 创建 DataFrame 展示
    df_res = pd.DataFrame({
        'code': ohlcv['close'].columns,
        'score': final_scores,
        'price': ohlcv['close'].iloc[-1].values
    })
    
    df_res = df_res[valid_mask].sort_values('score', ascending=False).head(TOP_N)
    
    target_weights = {}
    for i, row in df_res.iterrows():
        print(f"   {i+1}. {row['code']}  |  得分: {row['score']:.4f}  |  现价: {row['price']:.3f}")
        target_weights[row['code']] = 1.0 / TOP_N # 等权重
        
    print("-" * 60)
    
    # ================= 交易执行逻辑 =================
    trader = SimpleTrader(
        data_dir=PROJECT_ROOT / "_trading_data",
        initial_capital=args.capital
    )
    
    # 获取当前价格字典
    current_prices = dict(zip(ohlcv['close'].columns, ohlcv['close'].iloc[-1].values))
    
    print("\n💼 当前账户状态:")
    print(f"   现金: {trader.get_cash():.2f}")
    print(f"   持仓: {trader.get_holdings()}")
    nav = trader.calculate_nav(current_prices)
    print(f"   总资产 (NAV): {nav:.2f}")
    
    print("\n📋 交易计划:")
    orders = trader.generate_rebalance_orders(target_weights, current_prices)
    
    if not orders:
        print("   无需调仓 (No trades needed)")
    else:
        for order in orders:
            print(f"   {order['action']:<4} {order['ticker']} x {order['quantity']} @ {order['price']:.3f}")
            
    if args.execute:
        if orders:
            print("\n⚡ 执行交易中...")
            for order in orders:
                trader.execute_order(order)
        
        # 无论是否有交易，都记录 NAV
        trader.log_daily_nav(current_prices)
        print("\n✅ 交易执行完成，日志已更新。")
    else:
        if orders:
            print("\n💡 提示: 使用 --execute 参数执行上述交易")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
