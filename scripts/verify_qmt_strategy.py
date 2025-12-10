#!/usr/bin/env python3
"""
QMT 数据策略验证脚本

功能：
1. 使用 QMT 修复后的数据运行最佳策略回测
2. 对比原始数据的结果
3. 验证策略可复现性

使用方法:
    uv run python scripts/verify_qmt_strategy.py
"""

import sys
from pathlib import Path
from datetime import datetime
import yaml
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor


def load_config(config_path: Path) -> dict:
    """加载配置文件"""
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_simple_backtest(ohlcv: dict, factor_names: list, config: dict) -> dict:
    """运行简化版回测"""
    
    # 计算因子
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv)
    
    factor_names_list = raw_factors_df.columns.get_level_values(0).unique().tolist()
    raw_factors = {fname: raw_factors_df[fname] for fname in factor_names_list}
    
    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors(raw_factors)
    
    # 获取指定因子
    available_factors = [f for f in factor_names if f in std_factors]
    if len(available_factors) != len(factor_names):
        missing = set(factor_names) - set(available_factors)
        print(f"⚠️ 缺少因子: {missing}")
    
    # 合并因子得分
    first_factor = std_factors[available_factors[0]]
    dates = first_factor.index
    etf_codes = first_factor.columns.tolist()
    
    combined_score = sum(std_factors[f] for f in available_factors) / len(available_factors)
    
    # 获取价格数据
    close_prices = ohlcv["close"][etf_codes].ffill().bfill()
    
    # 回测参数
    backtest_cfg = config.get("backtest", {})
    freq = backtest_cfg.get("freq", 3)
    pos_size = backtest_cfg.get("pos_size", 2)
    initial_capital = float(backtest_cfg.get("initial_capital", 1_000_000))
    commission_rate = float(backtest_cfg.get("commission_rate", 0.0002))
    lookback = backtest_cfg.get("lookback", 252)
    
    print(f"  回测参数: FREQ={freq}, POS={pos_size}, 初始资金={initial_capital:,.0f}")
    
    # 简单回测逻辑
    portfolio_values = [initial_capital]
    
    # 生成调仓日
    valid_dates = dates[dates >= dates[0] + pd.Timedelta(days=lookback)]
    rebalance_dates = valid_dates[::freq]
    
    current_holdings = {}  # {etf: shares}
    cash = initial_capital
    
    for i, date in enumerate(valid_dates):
        # 获取当日收盘价
        day_prices = close_prices.loc[date]
        
        # 如果是调仓日
        if date in rebalance_dates.values:
            # 清仓
            for etf, shares in current_holdings.items():
                if shares > 0:
                    sell_value = shares * day_prices[etf] * (1 - commission_rate)
                    cash += sell_value
            current_holdings = {}
            
            # 选股（得分最高的 pos_size 只）
            scores = combined_score.loc[date].dropna()
            if len(scores) > 0:
                top_etfs = scores.nlargest(pos_size).index.tolist()
                
                # 等权买入
                buy_value_per_etf = cash / pos_size
                for etf in top_etfs:
                    price = day_prices[etf]
                    if pd.notna(price) and price > 0:
                        shares = int(buy_value_per_etf / price / 100) * 100
                        if shares > 0:
                            cost = shares * price * (1 + commission_rate)
                            if cost <= cash:
                                current_holdings[etf] = shares
                                cash -= cost
        
        # 计算当日组合价值
        portfolio_value = cash
        for etf, shares in current_holdings.items():
            price = day_prices.get(etf, 0)
            if pd.notna(price):
                portfolio_value += shares * price
        
        portfolio_values.append(portfolio_value)
    
    # 计算指标
    pv_series = pd.Series(portfolio_values[1:], index=valid_dates)
    daily_returns = pv_series.pct_change().dropna()
    
    total_return = (portfolio_values[-1] / initial_capital - 1) * 100
    
    # 年化收益
    years = (valid_dates[-1] - valid_dates[0]).days / 365
    annual_return = ((portfolio_values[-1] / initial_capital) ** (1/years) - 1) * 100
    
    # Sharpe
    annual_vol = daily_returns.std() * np.sqrt(252) * 100
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    
    # 最大回撤
    cummax = pv_series.cummax()
    drawdown = (pv_series - cummax) / cummax
    max_drawdown = drawdown.min() * 100
    
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "final_value": portfolio_values[-1],
        "days": len(valid_dates),
        "start_date": str(valid_dates[0].date()),
        "end_date": str(valid_dates[-1].date()),
    }


def main():
    print("=" * 70)
    print("QMT 数据策略验证")
    print("=" * 70)
    
    # 最佳策略因子组合
    best_factors = [
        "ADX_14D",
        "MAX_DD_60D",
        "PRICE_POSITION_120D",
        "PRICE_POSITION_20D",
        "SHARPE_RATIO_20D",
    ]
    print(f"最佳策略因子: {' + '.join(best_factors)}")
    
    # 加载配置
    qmt_config_path = ROOT / "configs" / "combo_wfo_config_qmt.yaml"
    orig_config_path = ROOT / "configs" / "combo_wfo_config.yaml"
    
    if not qmt_config_path.exists():
        print(f"⚠️ QMT 配置文件不存在，使用原始配置")
        qmt_config_path = orig_config_path
    
    qmt_config = load_config(qmt_config_path)
    orig_config = load_config(orig_config_path)
    
    print(f"\n{'='*70}")
    print("1. 使用 QMT 修复后数据回测")
    print("=" * 70)
    
    qmt_loader = DataLoader(
        data_dir=qmt_config["data"]["data_dir"],
        cache_dir=qmt_config["data"]["cache_dir"],
    )
    qmt_ohlcv = qmt_loader.load_ohlcv(
        etf_codes=qmt_config["data"]["symbols"],
        start_date=qmt_config["data"]["start_date"],
        end_date=qmt_config["data"]["end_date"],
        use_cache=False,
    )
    print(f"  日期范围: {qmt_config['data']['start_date']} 至 {qmt_config['data']['end_date']}")
    print(f"  ETF 数量: {len(qmt_ohlcv['close'].columns)}")
    print(f"  数据天数: {len(qmt_ohlcv['close'])}")
    
    qmt_result = run_simple_backtest(qmt_ohlcv, best_factors, qmt_config)
    print(f"\n  QMT 结果:")
    print(f"    累计收益: {qmt_result['total_return']:.2f}%")
    print(f"    年化收益: {qmt_result['annual_return']:.2f}%")
    print(f"    Sharpe:   {qmt_result['sharpe']:.3f}")
    print(f"    最大回撤: {qmt_result['max_drawdown']:.2f}%")
    print(f"    期间: {qmt_result['start_date']} 至 {qmt_result['end_date']}")
    
    print(f"\n{'='*70}")
    print("2. 使用原始数据回测 (作为对照)")
    print("=" * 70)
    
    orig_loader = DataLoader(
        data_dir=orig_config["data"]["data_dir"],
        cache_dir=orig_config["data"]["cache_dir"],
    )
    orig_ohlcv = orig_loader.load_ohlcv(
        etf_codes=orig_config["data"]["symbols"],
        start_date=orig_config["data"]["start_date"],
        end_date=orig_config["data"]["end_date"],
        use_cache=False,
    )
    print(f"  日期范围: {orig_config['data']['start_date']} 至 {orig_config['data']['end_date']}")
    print(f"  ETF 数量: {len(orig_ohlcv['close'].columns)}")
    print(f"  数据天数: {len(orig_ohlcv['close'])}")
    
    orig_result = run_simple_backtest(orig_ohlcv, best_factors, orig_config)
    print(f"\n  原始结果:")
    print(f"    累计收益: {orig_result['total_return']:.2f}%")
    print(f"    年化收益: {orig_result['annual_return']:.2f}%")
    print(f"    Sharpe:   {orig_result['sharpe']:.3f}")
    print(f"    最大回撤: {orig_result['max_drawdown']:.2f}%")
    print(f"    期间: {orig_result['start_date']} 至 {orig_result['end_date']}")
    
    print(f"\n{'='*70}")
    print("3. 对比分析")
    print("=" * 70)
    
    # 使用相同日期范围对比
    print("\n使用相同日期范围 (2020-01-01 至 2025-10-14) 对比:")
    
    # 重新加载 QMT 数据，使用原始配置的日期范围
    qmt_ohlcv_aligned = qmt_loader.load_ohlcv(
        etf_codes=qmt_config["data"]["symbols"],
        start_date=orig_config["data"]["start_date"],
        end_date=orig_config["data"]["end_date"],
        use_cache=False,
    )
    
    aligned_result = run_simple_backtest(qmt_ohlcv_aligned, best_factors, orig_config)
    print(f"\n  QMT (对齐日期):")
    print(f"    累计收益: {aligned_result['total_return']:.2f}%")
    print(f"    年化收益: {aligned_result['annual_return']:.2f}%")
    print(f"    Sharpe:   {aligned_result['sharpe']:.3f}")
    print(f"    最大回撤: {aligned_result['max_drawdown']:.2f}%")
    
    print(f"\n  差异分析:")
    ret_diff = aligned_result['total_return'] - orig_result['total_return']
    sharpe_diff = aligned_result['sharpe'] - orig_result['sharpe']
    dd_diff = aligned_result['max_drawdown'] - orig_result['max_drawdown']
    
    print(f"    收益差异: {ret_diff:+.2f}%")
    print(f"    Sharpe差异: {sharpe_diff:+.3f}")
    print(f"    回撤差异: {dd_diff:+.2f}%")
    
    # 判断结果
    print(f"\n{'='*70}")
    print("4. 结论")
    print("=" * 70)
    
    if abs(ret_diff) < 5:  # 5% 以内认为可接受
        print("✅ QMT 数据验证通过！")
        print(f"   收益差异在可接受范围内 ({ret_diff:+.2f}%)")
        print("   策略在 QMT 数据上可复现")
    else:
        print("⚠️ QMT 数据存在较大差异")
        print(f"   收益差异: {ret_diff:+.2f}%")
        print("   需要进一步调查原因")


if __name__ == "__main__":
    main()
