"""
深度审计脚本：逐行验证 Rank 8983 策略
SHARPE_RATIO_20D + VORTEX_14D
"""
import os
import sys
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime

sys.path.append(os.getcwd())

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor

COMBO = "SHARPE_RATIO_20D + VORTEX_14D"
FREQ = 8
POS_SIZE = 3
INITIAL_CAPITAL = 1_000_000.0
COMMISSION_RATE = 0.0002

def deep_audit():
    print("="*80)
    print("🔬 深度审计：逐行检查策略 8983")
    print("="*80)
    
    # 1. Load Data
    print("\n[1] 加载数据...")
    with open("configs/combo_wfo_config.yaml") as f:
        config = yaml.safe_load(f)
    
    loader = DataLoader(
        data_dir=config["data"]["data_dir"],
        cache_dir=config["data"]["cache_dir"]
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
        use_cache=True
    )
    
    close = ohlcv["close"]
    high = ohlcv["high"]
    low = ohlcv["low"]
    
    print(f"   数据范围: {close.index[0]} ~ {close.index[-1]}")
    print(f"   ETF数量: {close.shape[1]}")
    print(f"   交易日数: {close.shape[0]}")
    
    # 2. 手动计算因子 (不使用库，从头计算验证)
    print("\n[2] 手动计算因子 (验证因子库是否正确)...")
    
    # SHARPE_RATIO_20D: mean(ret) / std(ret) * sqrt(252)
    returns = close.pct_change(fill_method=None)
    mean_ret = returns.rolling(20, min_periods=20).mean()
    std_ret = returns.rolling(20, min_periods=20).std()
    sharpe_manual = (mean_ret / (std_ret + 1e-10)) * np.sqrt(252)
    sharpe_manual = sharpe_manual.where(std_ret >= 1e-10, np.nan)
    
    # VORTEX_14D: VI+ - VI-
    # 逐列计算
    vortex_manual = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for col in close.columns:
        vm_p = (high[col] - low[col].shift(1)).abs()
        vm_m = (low[col] - high[col].shift(1)).abs()
        pc = close[col].shift(1)
        tr1 = high[col] - low[col]
        tr2 = (high[col] - pc).abs()
        tr3 = (low[col] - pc).abs()
        tr_col = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        vm_p_sum = vm_p.rolling(14, min_periods=14).sum()
        vm_m_sum = vm_m.rolling(14, min_periods=14).sum()
        tr_sum = tr_col.rolling(14, min_periods=14).sum()
        
        vi_plus = vm_p_sum / (tr_sum + 1e-10)
        vi_minus = vm_m_sum / (tr_sum + 1e-10)
        vortex_manual[col] = vi_plus - vi_minus
    
    # 与因子库对比
    lib = PreciseFactorLibrary()
    factors_df = lib.compute_all_factors(ohlcv)
    
    sharpe_lib = factors_df["SHARPE_RATIO_20D"]
    vortex_lib = factors_df["VORTEX_14D"]
    
    # 检查差异
    sharpe_diff = (sharpe_manual - sharpe_lib).abs()
    vortex_diff = (vortex_manual - vortex_lib).abs()
    
    print(f"   SHARPE_RATIO_20D 手动 vs 库 最大差异: {sharpe_diff.max().max():.10f}")
    print(f"   VORTEX_14D 手动 vs 库 最大差异: {vortex_diff.max().max():.10f}")
    
    if sharpe_diff.max().max() > 1e-6 or vortex_diff.max().max() > 1e-6:
        print("   ⚠️  警告：因子计算存在差异！")
    else:
        print("   ✅ 因子计算验证通过")
    
    # 3. Cross Section 处理
    print("\n[3] 检查截面标准化...")
    proc = CrossSectionProcessor(
        lower_percentile=config["cross_section"]["winsorize_lower"] * 100,
        upper_percentile=config["cross_section"]["winsorize_upper"] * 100,
        verbose=False
    )
    
    factors_dict = {"SHARPE_RATIO_20D": sharpe_lib, "VORTEX_14D": vortex_lib}
    std_factors = proc.process_all_factors(factors_dict)
    
    sharpe_std = std_factors["SHARPE_RATIO_20D"]
    vortex_std = std_factors["VORTEX_14D"]
    
    # 检查标准化后的分布
    print(f"   SHARPE_RATIO_20D 标准化后: mean={sharpe_std.mean().mean():.4f}, std={sharpe_std.std().mean():.4f}")
    print(f"   VORTEX_14D 标准化后: mean={vortex_std.mean().mean():.4f}, std={vortex_std.std().mean():.4f}")
    
    # 4. 检查信号生成逻辑
    print("\n[4] 逐日检查信号生成 (抽样5个调仓日)...")
    
    combined_score = sharpe_std + vortex_std
    dates = close.index
    lookback = 252
    
    rebalance_dates = [d for i, d in enumerate(dates) if i >= lookback and i % FREQ == 0]
    sample_dates = rebalance_dates[::len(rebalance_dates)//5][:5]  # 均匀抽样5个
    
    for rd in sample_dates:
        t = dates.get_loc(rd)
        signal_date = dates[t-1]  # T-1日信号
        exec_date = dates[t]       # T日执行
        
        # T-1日的因子值
        sharpe_t1 = sharpe_std.loc[signal_date]
        vortex_t1 = vortex_std.loc[signal_date]
        score_t1 = sharpe_t1 + vortex_t1
        
        # 检查是否使用了T日数据
        sharpe_t = sharpe_std.loc[exec_date]
        vortex_t = vortex_std.loc[exec_date]
        
        # 排名选股
        valid_scores = score_t1.dropna()
        if len(valid_scores) >= POS_SIZE:
            top3 = valid_scores.nlargest(POS_SIZE).index.tolist()
        else:
            top3 = []
        
        print(f"\n   调仓日: {exec_date.strftime('%Y-%m-%d')}")
        print(f"   信号日: {signal_date.strftime('%Y-%m-%d')} (T-1)")
        print(f"   Top 3 ETF: {top3}")
        print(f"   Top 3 得分: {[f'{score_t1[e]:.4f}' for e in top3]}")
        
        # 检查是否存在"未来数据"泄露
        # 如果 T 日得分和 T-1 日得分完全相同，说明可能有问题
        score_match = all(abs(score_t1[e] - (sharpe_t[e] + vortex_t[e])) < 0.01 for e in top3 if e in sharpe_t.index)
        if score_match:
            print(f"   ⚠️  T日与T-1日得分接近（可能正常，因为因子有延续性）")
    
    # 5. 完整回测 (带详细日志)
    print("\n[5] 完整回测 (逐笔记录)...")
    
    close_arr = close.values
    etf_codes = close.columns.tolist()
    T, N = close_arr.shape
    
    cash = INITIAL_CAPITAL
    holdings = {}
    trade_log = []
    equity_curve = []
    
    for t in range(lookback, T):
        current_date = dates[t]
        
        # Mark to Market
        current_value = cash
        for idx, info in holdings.items():
            current_value += info['shares'] * close_arr[t, idx]
        equity_curve.append({'date': current_date, 'equity': current_value})
        
        if t % FREQ == 0:
            # Signal: T-1
            score_t1 = combined_score.iloc[t-1]
            valid_mask = ~score_t1.isna()
            
            if valid_mask.sum() >= POS_SIZE:
                top_k = score_t1[valid_mask].nlargest(POS_SIZE).index.tolist()
                target_indices = [etf_codes.index(e) for e in top_k]
            else:
                target_indices = []
            
            # Sell
            current_indices = list(holdings.keys())
            for idx in current_indices:
                if idx not in target_indices:
                    info = holdings[idx]
                    price = close_arr[t, idx]
                    proceeds = info['shares'] * price * (1 - COMMISSION_RATE)
                    cash += proceeds
                    
                    pnl = (price - info['entry_price']) / info['entry_price']
                    trade_log.append({
                        'entry_date': info['entry_date'],
                        'exit_date': current_date,
                        'ticker': etf_codes[idx],
                        'entry_price': info['entry_price'],
                        'exit_price': price,
                        'pnl_pct': pnl
                    })
                    del holdings[idx]
            
            # Buy
            target_pos_value = current_value / POS_SIZE
            for idx in target_indices:
                price = close_arr[t, idx]
                if np.isnan(price): continue
                
                if idx not in holdings:
                    shares = target_pos_value / price
                    cost = shares * price * (1 + COMMISSION_RATE)
                    if cash >= cost:
                        cash -= cost
                        holdings[idx] = {
                            'shares': shares,
                            'entry_price': price,
                            'entry_date': current_date
                        }
    
    # Close all
    final_date = dates[-1]
    for idx, info in holdings.items():
        price = close_arr[-1, idx]
        proceeds = info['shares'] * price * (1 - COMMISSION_RATE)
        cash += proceeds
        pnl = (price - info['entry_price']) / info['entry_price']
        trade_log.append({
            'entry_date': info['entry_date'],
            'exit_date': final_date,
            'ticker': etf_codes[idx],
            'entry_price': info['entry_price'],
            'exit_price': price,
            'pnl_pct': pnl
        })
    
    # 6. 统计
    df_trades = pd.DataFrame(trade_log)
    df_equity = pd.DataFrame(equity_curve)
    
    total_trades = len(df_trades)
    wins = df_trades[df_trades['pnl_pct'] > 0]
    losses = df_trades[df_trades['pnl_pct'] <= 0]
    win_rate = len(wins) / total_trades if total_trades > 0 else 0
    
    win_amount = wins['pnl_pct'].sum() if len(wins) > 0 else 0
    loss_amount = abs(losses['pnl_pct'].sum()) if len(losses) > 0 else 0.0001
    profit_factor = win_amount / loss_amount
    
    total_ret = (cash - INITIAL_CAPITAL) / INITIAL_CAPITAL
    
    # Max Drawdown
    df_equity['peak'] = df_equity['equity'].cummax()
    df_equity['dd'] = (df_equity['equity'] - df_equity['peak']) / df_equity['peak']
    max_dd = df_equity['dd'].min()
    
    print(f"\n📊 深度审计结果:")
    print(f"   总交易次数: {total_trades}")
    print(f"   胜率: {win_rate:.2%}")
    print(f"   盈亏比: {profit_factor:.2f}")
    print(f"   总收益率: {total_ret:.2%}")
    print(f"   最大回撤: {max_dd:.2%}")
    print(f"   最终资金: {cash:,.0f}")
    
    # 7. 交易分布检查
    print(f"\n[6] 交易分布检查...")
    df_trades['year'] = pd.to_datetime(df_trades['exit_date']).dt.year
    yearly_stats = df_trades.groupby('year').agg({
        'pnl_pct': ['count', 'mean', lambda x: (x > 0).sum() / len(x)]
    })
    yearly_stats.columns = ['trades', 'avg_pnl', 'win_rate']
    print(yearly_stats.to_string())
    
    # 8. 检查是否有异常大盈利
    print(f"\n[7] 异常检查...")
    top5_trades = df_trades.nlargest(5, 'pnl_pct')
    print("   Top 5 盈利交易:")
    for _, row in top5_trades.iterrows():
        print(f"   {row['ticker']} | {row['entry_date'].strftime('%Y-%m-%d')} -> {row['exit_date'].strftime('%Y-%m-%d')} | +{row['pnl_pct']:.2%}")
    
    worst5_trades = df_trades.nsmallest(5, 'pnl_pct')
    print("   Top 5 亏损交易:")
    for _, row in worst5_trades.iterrows():
        print(f"   {row['ticker']} | {row['entry_date'].strftime('%Y-%m-%d')} -> {row['exit_date'].strftime('%Y-%m-%d')} | {row['pnl_pct']:.2%}")
    
    # 保存
    df_trades.to_csv("results/deep_audit_trades.csv", index=False)
    df_equity.to_csv("results/deep_audit_equity.csv", index=False)
    print(f"\n💾 审计结果已保存: results/deep_audit_*.csv")

if __name__ == "__main__":
    deep_audit()
