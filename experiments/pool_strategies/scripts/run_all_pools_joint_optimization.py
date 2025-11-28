#!/usr/bin/env python3
"""
全子池 WFO + VEC：因子 × 风控参数联合优化
================================================================================
把风控参数（止损/止盈/跟踪止损）也加入 WFO 搜索空间：

搜索空间:
- 因子组合: C(18,2) + C(18,3) + C(18,4) + C(18,5) = 12,597
- 止损: [-5%, -8%, -10%, -15%, 无]
- 止盈: [+15%, +20%, +30%, +50%, 无]
- 跟踪止损: [-5%, -8%, -10%, -15%, 无]

总策略数: 12,597 × 5 × 5 × 5 = 1,574,625 (每池)

用法:
    uv run python scripts/run_all_pools_joint_optimization.py
================================================================================
"""

import sys
from pathlib import Path
from datetime import datetime
from itertools import combinations, product
import logging
import json
import time

import numpy as np
import pandas as pd
import yaml
from numba import njit, prange

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "etf_rotation_optimized"))

from core.data_loader import DataLoader
from core.precise_factor_library_v2 import PreciseFactorLibrary
from core.cross_section_processor import CrossSectionProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# 参数配置
# =============================================================================

COMBO_SIZES = [2, 3, 4, 5]
LOOKBACK = 252
FREQ = 8
POS_SIZE = 3
INITIAL_CAPITAL = 1_000_000
COMMISSION_RATE = 0.0002

# 风控参数搜索空间
STOP_LOSS_OPTIONS = np.array([-0.05, -0.08, -0.10, -0.15, -1.0])      # -1.0 = 无止损
TAKE_PROFIT_OPTIONS = np.array([0.15, 0.20, 0.30, 0.50, 10.0])        # 10.0 = 无止盈
TRAILING_STOP_OPTIONS = np.array([-0.05, -0.08, -0.10, -0.15, -1.0])  # -1.0 = 无跟踪止损


# =============================================================================
# Numba 加速：VEC 回测 (带风控参数)
# =============================================================================

@njit(cache=True)
def _vec_backtest_with_risk(
    close_prices: np.ndarray,
    factors_3d: np.ndarray,
    factor_indices: np.ndarray,
    n_factors: int,
    rebalance_days: np.ndarray,
    pos_size: int,
    initial_capital: float,
    commission_rate: float,
    stop_loss: float,
    take_profit: float,
    trailing_stop: float,
) -> tuple:
    """带风控的 VEC 回测"""
    T, N = close_prices.shape
    
    cash = initial_capital
    holdings = np.zeros(N)
    entry_prices = np.zeros(N)
    highest_prices = np.zeros(N)
    
    daily_values = np.zeros(T)
    daily_values[0] = initial_capital
    
    total_trades = 0
    wins = 0
    
    for t in range(1, T):
        # 更新最高价
        for n in range(N):
            if holdings[n] > 0 and close_prices[t, n] > highest_prices[n]:
                highest_prices[n] = close_prices[t, n]
        
        # 风控检查
        for n in range(N):
            if holdings[n] > 0:
                price = close_prices[t, n]
                pnl = (price - entry_prices[n]) / entry_prices[n]
                trailing_dd = (price - highest_prices[n]) / highest_prices[n] if highest_prices[n] > 0 else 0
                
                should_exit = (
                    pnl <= stop_loss or 
                    pnl >= take_profit or 
                    (pnl > 0 and trailing_dd <= trailing_stop)
                )
                
                if should_exit:
                    sell_value = holdings[n] * price
                    commission = sell_value * commission_rate
                    cash += sell_value - commission
                    
                    if price > entry_prices[n]:
                        wins += 1
                    total_trades += 1
                    
                    holdings[n] = 0
                    entry_prices[n] = 0
                    highest_prices[n] = 0
        
        # 计算净值
        pv = cash
        for n in range(N):
            if holdings[n] > 0:
                pv += holdings[n] * close_prices[t, n]
        daily_values[t] = pv
        
        # 调仓检查
        is_rebal = False
        for r in range(len(rebalance_days)):
            if rebalance_days[r] == t:
                is_rebal = True
                break
        
        if not is_rebal:
            continue
        
        # 卖出
        for n in range(N):
            if holdings[n] > 0:
                sell_value = holdings[n] * close_prices[t, n]
                commission = sell_value * commission_rate
                cash += sell_value - commission
                
                if close_prices[t, n] > entry_prices[n]:
                    wins += 1
                total_trades += 1
                
                holdings[n] = 0
                entry_prices[n] = 0
                highest_prices[n] = 0
        
        # 计算因子得分
        scores = np.zeros(N)
        for n in range(N):
            s = 0.0
            cnt = 0
            for i in range(n_factors):
                f_idx = factor_indices[i]
                v = factors_3d[t-1, n, f_idx]
                if not np.isnan(v):
                    s += v
                    cnt += 1
            scores[n] = s / cnt if cnt > 0 else -1e9
        
        # 选股
        top_indices = np.argsort(scores)[::-1][:pos_size]
        
        # 买入
        capital_per = cash / pos_size
        for idx in top_indices:
            if scores[idx] <= -1e8:
                continue
            price = close_prices[t, idx]
            if price <= 0:
                continue
            shares = int(capital_per / price / 100) * 100
            if shares <= 0:
                continue
            buy_value = shares * price
            commission = buy_value * commission_rate
            if buy_value + commission > cash:
                continue
            cash -= buy_value + commission
            holdings[idx] = shares
            entry_prices[idx] = price
            highest_prices[idx] = price
    
    # 计算指标
    final_value = daily_values[-1]
    total_return = (final_value - initial_capital) / initial_capital
    
    max_dd = 0.0
    peak = daily_values[0]
    for t in range(T):
        if daily_values[t] > peak:
            peak = daily_values[t]
        dd = (peak - daily_values[t]) / peak
        if dd > max_dd:
            max_dd = dd
    
    win_rate = wins / total_trades if total_trades > 0 else 0.0
    
    years = T / 252.0
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
    
    returns = np.zeros(T - 1)
    for t in range(1, T):
        if daily_values[t-1] > 0:
            returns[t-1] = (daily_values[t] - daily_values[t-1]) / daily_values[t-1]
    
    ret_std = np.std(returns)
    sharpe = np.mean(returns) / ret_std * np.sqrt(252) if ret_std > 0 else 0.0
    
    return total_return, annual_return, max_dd, sharpe, win_rate, total_trades


@njit(parallel=True, cache=True)
def _run_all_strategies(
    close_prices: np.ndarray,
    factors_3d: np.ndarray,
    all_combo_indices: np.ndarray,
    combo_sizes: np.ndarray,
    n_combos: int,
    risk_params: np.ndarray,  # [n_risk_combos, 3] - (sl, tp, ts)
    n_risk_combos: int,
    rebalance_days: np.ndarray,
    pos_size: int,
    initial_capital: float,
    commission_rate: float,
) -> np.ndarray:
    """并行运行所有策略（因子组合 × 风控参数）"""
    total_strategies = n_combos * n_risk_combos
    results = np.zeros((total_strategies, 6))
    
    for i in prange(total_strategies):
        combo_idx = i // n_risk_combos
        risk_idx = i % n_risk_combos
        
        size = combo_sizes[combo_idx]
        factor_indices = all_combo_indices[combo_idx, :size]
        
        sl = risk_params[risk_idx, 0]
        tp = risk_params[risk_idx, 1]
        ts = risk_params[risk_idx, 2]
        
        (tr, ar, mdd, sharpe, wr, nt) = _vec_backtest_with_risk(
            close_prices, factors_3d, factor_indices, size, rebalance_days,
            pos_size, initial_capital, commission_rate, sl, tp, ts
        )
        
        results[i, 0] = tr
        results[i, 1] = ar
        results[i, 2] = mdd
        results[i, 3] = sharpe
        results[i, 4] = wr
        results[i, 5] = nt
    
    return results


# =============================================================================
# 单池处理
# =============================================================================

def process_single_pool(
    pool_name: str,
    pool_symbols: list,
    ohlcv: dict,
    factor_names: list,
    output_dir: Path,
) -> dict:
    """处理单个池"""
    logger.info(f"")
    logger.info(f"{'='*80}")
    logger.info(f"📊 处理池: {pool_name} ({len(pool_symbols)} ETFs)")
    logger.info(f"{'='*80}")
    
    valid_symbols = [s for s in pool_symbols if s in ohlcv["close"].columns]
    if len(valid_symbols) < 3:
        logger.warning(f"   ⚠️ 有效 ETF 不足 3 个，跳过")
        return None
    
    logger.info(f"   有效 ETF: {len(valid_symbols)}")
    
    pool_ohlcv = {key: df[valid_symbols] for key, df in ohlcv.items()}
    
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(pool_ohlcv)
    raw_factors = {fname: raw_factors_df[fname] for fname in factor_names}
    
    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors(raw_factors)
    
    T = len(pool_ohlcv["close"])
    N = len(valid_symbols)
    F = len(factor_names)
    
    factors_3d = np.stack([std_factors[f].values for f in factor_names], axis=-1)
    close_prices = pool_ohlcv["close"].ffill().bfill().values.astype(np.float64)
    
    # 生成因子组合
    all_combos = []
    for size in COMBO_SIZES:
        combos = list(combinations(range(F), size))
        all_combos.extend([(c, size) for c in combos])
    
    n_combos = len(all_combos)
    
    max_combo_size = max(COMBO_SIZES)
    all_combo_indices = np.full((n_combos, max_combo_size), -1, dtype=np.int64)
    combo_sizes_arr = np.zeros(n_combos, dtype=np.int64)
    
    for i, (combo, size) in enumerate(all_combos):
        combo_sizes_arr[i] = size
        for j, idx in enumerate(combo):
            all_combo_indices[i, j] = idx
    
    # 生成风控参数组合
    risk_combos = list(product(STOP_LOSS_OPTIONS, TAKE_PROFIT_OPTIONS, TRAILING_STOP_OPTIONS))
    n_risk_combos = len(risk_combos)
    risk_params = np.array(risk_combos, dtype=np.float64)
    
    total_strategies = n_combos * n_risk_combos
    logger.info(f"   因子组合: {n_combos} | 风控组合: {n_risk_combos} | 总策略: {total_strategies:,}")
    
    # 调仓日
    rebalance_days = np.array([t for t in range(LOOKBACK, T, FREQ)], dtype=np.int64)
    pool_pos_size = min(POS_SIZE, N - 1) if N > 1 else 1
    
    # 运行所有策略
    t0 = time.time()
    results = _run_all_strategies(
        close_prices, factors_3d, all_combo_indices, combo_sizes_arr, n_combos,
        risk_params, n_risk_combos, rebalance_days, pool_pos_size,
        INITIAL_CAPITAL, COMMISSION_RATE
    )
    elapsed = time.time() - t0
    logger.info(f"   运行耗时: {elapsed:.1f}s ({total_strategies/elapsed:.0f} 策略/秒)")
    
    # 整合结果
    records = []
    for i in range(total_strategies):
        combo_idx = i // n_risk_combos
        risk_idx = i % n_risk_combos
        
        combo, size = all_combos[combo_idx]
        combo_names = [factor_names[idx] for idx in combo]
        combo_str = " + ".join(combo_names)
        
        sl, tp, ts = risk_combos[risk_idx]
        
        tr, ar, mdd, sharpe, wr, nt = results[i]
        
        records.append({
            "pool": pool_name,
            "combo": combo_str,
            "combo_size": size,
            "stop_loss": sl if sl > -0.99 else None,
            "take_profit": tp if tp < 9.9 else None,
            "trailing_stop": ts if ts > -0.99 else None,
            "total_return": tr,
            "annual_return": ar,
            "max_drawdown": mdd,
            "sharpe_ratio": sharpe,
            "win_rate": wr,
            "num_trades": int(nt),
        })
    
    df = pd.DataFrame(records)
    
    # 排序（夏普优先）
    df = df.sort_values("sharpe_ratio", ascending=False).reset_index(drop=True)
    
    # 保存
    pool_dir = output_dir / pool_name
    pool_dir.mkdir(parents=True, exist_ok=True)
    df.head(10000).to_parquet(pool_dir / "top10000.parquet", index=False)
    df.head(100).to_csv(pool_dir / "top100.csv", index=False)
    
    # 最优策略
    best = df.iloc[0]
    best_info = {
        "pool": pool_name,
        "n_symbols": len(valid_symbols),
        "symbols": valid_symbols,
        "total_strategies": total_strategies,
        "best_combo": best["combo"],
        "best_factors": best["combo"].split(" + "),
        "best_risk_params": {
            "stop_loss": best["stop_loss"],
            "take_profit": best["take_profit"],
            "trailing_stop": best["trailing_stop"],
        },
        "total_return": float(best["total_return"]),
        "annual_return": float(best["annual_return"]),
        "max_drawdown": float(best["max_drawdown"]),
        "sharpe_ratio": float(best["sharpe_ratio"]),
        "win_rate": float(best["win_rate"]),
    }
    
    with open(pool_dir / "best_strategy.json", "w") as f:
        json.dump(best_info, f, indent=2, ensure_ascii=False)
    
    # 格式化风控参数
    sl_str = f"{best['stop_loss']*100:.0f}%" if best['stop_loss'] else "无"
    tp_str = f"+{best['take_profit']*100:.0f}%" if best['take_profit'] else "无"
    ts_str = f"{best['trailing_stop']*100:.0f}%" if best['trailing_stop'] else "无"
    
    logger.info(f"   🏆 最优策略:")
    logger.info(f"      因子: {best['combo']}")
    logger.info(f"      风控: 止损{sl_str} | 止盈{tp_str} | 跟踪{ts_str}")
    logger.info(f"      收益: {best['total_return']*100:.1f}% | 夏普: {best['sharpe_ratio']:.2f} | 回撤: {best['max_drawdown']*100:.1f}%")
    
    return best_info


# =============================================================================
# 主函数
# =============================================================================

def main():
    start_time = datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    
    logger.info("=" * 80)
    logger.info("🚀 全子池 WFO + VEC：因子 × 风控参数联合优化")
    logger.info("=" * 80)
    logger.info(f"   止损选项: {[f'{x*100:.0f}%' if x > -0.99 else '无' for x in STOP_LOSS_OPTIONS]}")
    logger.info(f"   止盈选项: {[f'+{x*100:.0f}%' if x < 9.9 else '无' for x in TAKE_PROFIT_OPTIONS]}")
    logger.info(f"   跟踪选项: {[f'{x*100:.0f}%' if x > -0.99 else '无' for x in TRAILING_STOP_OPTIONS]}")
    
    config_pools_path = ROOT / "configs/etf_pools.yaml"
    config_wfo_path = ROOT / "configs/combo_wfo_config.yaml"
    
    with open(config_pools_path) as f:
        config_pools = yaml.safe_load(f)
    with open(config_wfo_path) as f:
        config_wfo = yaml.safe_load(f)
    
    pools = config_pools["pools"]
    logger.info(f"📊 共 {len(pools)} 个子池")
    
    all_symbols = set()
    for pool_name, pool_info in pools.items():
        all_symbols.update(pool_info["symbols"])
    all_symbols = sorted(all_symbols)
    
    logger.info("📊 加载数据...")
    loader = DataLoader(
        data_dir=config_wfo["data"].get("data_dir"),
        cache_dir=config_wfo["data"].get("cache_dir"),
    )
    
    ohlcv = loader.load_ohlcv(
        etf_codes=all_symbols,
        start_date=config_wfo["data"]["start_date"],
        end_date=config_wfo["data"]["end_date"],
    )
    
    factor_lib = PreciseFactorLibrary()
    sample_ohlcv = {k: v.iloc[:10] for k, v in ohlcv.items()}
    sample_factors = factor_lib.compute_all_factors(sample_ohlcv)
    factor_names = sorted(sample_factors.columns.get_level_values(0).unique().tolist())
    
    output_dir = ROOT / "results" / f"joint_optimization_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_best = {}
    for pool_name, pool_info in pools.items():
        best_info = process_single_pool(
            pool_name=pool_name,
            pool_symbols=pool_info["symbols"],
            ohlcv=ohlcv,
            factor_names=factor_names,
            output_dir=output_dir,
        )
        if best_info:
            all_best[pool_name] = best_info
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"✅ 全部完成 | 总耗时: {elapsed:.1f}秒")
    logger.info("=" * 80)
    
    logger.info("")
    logger.info("🏆 各池最优策略汇总（因子 + 风控联合优化）")
    logger.info("-" * 140)
    logger.info(f"{'池名':<18} {'收益':>8} {'夏普':>6} {'回撤':>6} {'止损':>6} {'止盈':>6} {'跟踪':>6} {'最优因子组合'}")
    logger.info("-" * 140)
    
    for pool_name, info in sorted(all_best.items(), key=lambda x: -x[1]["sharpe_ratio"]):
        rp = info["best_risk_params"]
        sl_str = f"{rp['stop_loss']*100:.0f}%" if rp['stop_loss'] else "无"
        tp_str = f"+{rp['take_profit']*100:.0f}%" if rp['take_profit'] else "无"
        ts_str = f"{rp['trailing_stop']*100:.0f}%" if rp['trailing_stop'] else "无"
        
        logger.info(
            f"{pool_name:<18} {info['total_return']*100:>7.1f}% {info['sharpe_ratio']:>6.2f} "
            f"{info['max_drawdown']*100:>5.1f}% {sl_str:>6} {tp_str:>6} {ts_str:>6} {info['best_combo']}"
        )
    
    # 保存汇总
    summary = {
        "timestamp": timestamp,
        "risk_search_space": {
            "stop_loss": STOP_LOSS_OPTIONS.tolist(),
            "take_profit": TAKE_PROFIT_OPTIONS.tolist(),
            "trailing_stop": TRAILING_STOP_OPTIONS.tolist(),
        },
        "elapsed_seconds": elapsed,
        "n_pools": len(all_best),
        "pools": all_best,
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 保存最优配置
    best_config = {
        "timestamp": timestamp,
        "pool_strategies": {
            pool: {
                "factors": info["best_factors"],
                "risk_params": info["best_risk_params"],
            }
            for pool, info in all_best.items()
        },
    }
    
    with open(output_dir / "best_config.json", "w") as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False)
    
    with open(ROOT / "results" / "joint_optimization_best_config_latest.json", "w") as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False)
    
    logger.info("")
    logger.info(f"📁 结果已保存至: {output_dir}")
    
    return summary


if __name__ == "__main__":
    main()
