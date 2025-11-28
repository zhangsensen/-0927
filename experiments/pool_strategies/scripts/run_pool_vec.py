#!/usr/bin/env python3
"""
分池 VEC 回测 | Pool-Specific VEC Backtest
================================================================================
使用 WFO 优化输出的因子配置执行向量化回测

输入：
- results/pool_wfo_best_config_latest.json (WFO 输出)

输出：
- 回测指标（收益率、夏普、回撤）
- 与基准对比

用法:
    uv run python scripts/run_pool_vec.py
    uv run python scripts/run_pool_vec.py --config results/pool_wfo_xxx/best_config.json
================================================================================
"""

import argparse
import logging
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from numba import njit

import sys
from pathlib import Path

# 添加项目根目录到路径
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "etf_rotation_optimized"))

from core.data_loader import DataLoader
from core.precise_factor_library_v2 import PreciseFactorLibrary
from core.cross_section_processor import CrossSectionProcessor
from core.utils.rebalance import generate_rebalance_schedule

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# 全局参数
LOOKBACK = 252
FREQ = 8
INITIAL_CAPITAL = 1_000_000
COMMISSION_RATE = 0.0002


# =============================================================================
# Numba 回测核心
# =============================================================================

@njit(cache=True)
def _pool_backtest_kernel(
    close_prices: np.ndarray,      # [T, N]
    pool_selections: np.ndarray,   # [num_rebal, num_pools, max_pos] 选中的 ETF 索引
    pool_weights: np.ndarray,      # [num_pools] 池权重
    pool_pos_sizes: np.ndarray,    # [num_pools] 每个池的持仓数
    rebalance_schedule: np.ndarray,
    initial_capital: float,
    commission_rate: float,
) -> Tuple[float, float, float, int, np.ndarray]:
    """
    分池回测核心
    """
    T, N = close_prices.shape
    num_rebal = len(rebalance_schedule)
    num_pools = len(pool_weights)
    
    cash = initial_capital
    holdings = np.zeros(N)  # 持仓股数
    entry_prices = np.zeros(N)
    
    daily_values = np.zeros(T)
    daily_values[0] = initial_capital
    
    total_trades = 0
    wins = 0
    
    for t in range(1, T):
        # 计算当前资产价值
        portfolio_value = cash
        for n in range(N):
            if holdings[n] > 0:
                portfolio_value += holdings[n] * close_prices[t, n]
        
        daily_values[t] = portfolio_value
        
        # 检查是否是调仓日
        is_rebal = False
        rebal_idx = -1
        for r in range(num_rebal):
            if rebalance_schedule[r] == t:
                is_rebal = True
                rebal_idx = r
                break
        
        if not is_rebal:
            continue
        
        # === 调仓逻辑 ===
        # 1. 清仓所有持仓
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
        
        # 2. 按池分配资金
        total_weight = 0.0
        for p in range(num_pools):
            total_weight += pool_weights[p]
        
        for p in range(num_pools):
            pool_capital = cash * (pool_weights[p] / total_weight)
            pos_size = pool_pos_sizes[p]
            
            if pos_size == 0:
                continue
            
            capital_per_etf = pool_capital / pos_size
            
            # 3. 池内选股买入
            for i in range(pos_size):
                etf_idx = pool_selections[rebal_idx, p, i]
                if etf_idx < 0:
                    continue
                
                price = close_prices[t, etf_idx]
                if price <= 0:
                    continue
                
                shares = int(capital_per_etf / price / 100) * 100  # 100 股整数
                if shares <= 0:
                    continue
                
                buy_value = shares * price
                commission = buy_value * commission_rate
                
                if buy_value + commission > cash:
                    continue
                
                cash -= buy_value + commission
                holdings[etf_idx] += shares
                entry_prices[etf_idx] = price
    
    # 计算最终指标
    final_value = daily_values[-1]
    total_return = (final_value - initial_capital) / initial_capital
    
    # 最大回撤
    max_dd = 0.0
    peak = daily_values[0]
    for t in range(T):
        if daily_values[t] > peak:
            peak = daily_values[t]
        dd = (peak - daily_values[t]) / peak
        if dd > max_dd:
            max_dd = dd
    
    # 胜率
    win_rate = wins / total_trades if total_trades > 0 else 0.0
    
    return total_return, max_dd, win_rate, total_trades, daily_values


# =============================================================================
# 回测引擎
# =============================================================================

class PoolVECBacktester:
    """分池向量化回测器"""
    
    def __init__(self, wfo_config_path: str):
        self.wfo_config_path = Path(wfo_config_path)
        
        # 加载配置
        with open(self.wfo_config_path) as f:
            self.wfo_config = json.load(f)
        
        config_wfo_path = ROOT / "configs/combo_wfo_config.yaml"
        config_pools_path = ROOT / "configs/etf_pools.yaml"
        
        with open(config_wfo_path) as f:
            self.config_wfo = yaml.safe_load(f)
        with open(config_pools_path) as f:
            self.config_pools = yaml.safe_load(f)
        
        self.loader = DataLoader(
            data_dir=self.config_wfo["data"].get("data_dir"),
            cache_dir=self.config_wfo["data"].get("cache_dir"),
        )
        self.factor_lib = PreciseFactorLibrary()
        self.processor = CrossSectionProcessor(verbose=False)
        
        self.ohlcv = None
        self.etf_codes = []
        self.factors_3d = None
        self.factor_names = []
        
    def load_data(self):
        """加载数据"""
        logger.info("📊 加载数据...")
        
        all_symbols = []
        for pool in self.config_pools["pools"].values():
            all_symbols.extend(pool["symbols"])
        all_symbols = sorted(list(set(all_symbols)))
        
        self.ohlcv = self.loader.load_ohlcv(
            etf_codes=all_symbols,
            start_date=self.config_wfo["data"]["start_date"],
            end_date=self.config_wfo["data"]["end_date"],
        )
        
        self.etf_codes = self.ohlcv["close"].columns.tolist()
        logger.info(f"✅ 数据加载完成: {len(self.etf_codes)} ETF")
        
    def compute_factors(self):
        """计算因子"""
        logger.info("🔧 计算因子...")
        
        raw_factors_df = self.factor_lib.compute_all_factors(self.ohlcv)
        factor_dict = {
            fname: raw_factors_df[fname] 
            for fname in raw_factors_df.columns.get_level_values(0).unique()
        }
        
        std_factors = self.processor.process_all_factors(factor_dict)
        
        self.factor_names = sorted(std_factors.keys())
        T = len(self.ohlcv["close"])
        N = len(self.etf_codes)
        F = len(self.factor_names)
        
        self.factors_3d = np.zeros((T, N, F))
        for f_idx, fname in enumerate(self.factor_names):
            self.factors_3d[:, :, f_idx] = std_factors[fname].values
        
        logger.info(f"✅ 因子计算完成: {F} 个因子")
        
    def _select_top_n_for_pool(
        self,
        t: int,
        pool_symbols: List[str],
        factor_names: List[str],
        n: int,
    ) -> List[int]:
        """为池内选择 Top N 个 ETF"""
        factor_indices = [
            self.factor_names.index(f) for f in factor_names 
            if f in self.factor_names
        ]
        
        if not factor_indices:
            # 回退到动量
            if "MOM_20D" in self.factor_names:
                factor_indices = [self.factor_names.index("MOM_20D")]
            else:
                return []
        
        # 计算池内每个 ETF 的得分
        scores = []
        for sym in pool_symbols:
            if sym not in self.etf_codes:
                continue
            
            etf_idx = self.etf_codes.index(sym)
            score = 0.0
            valid = 0
            
            for f_idx in factor_indices:
                val = self.factors_3d[t-1, etf_idx, f_idx]
                if not np.isnan(val):
                    score += val
                    valid += 1
            
            if valid > 0:
                scores.append((etf_idx, score / valid))
        
        # 排序选择
        scores.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in scores[:n]]
    
    def run_backtest(self) -> Dict:
        """执行回测"""
        if self.ohlcv is None:
            self.load_data()
        if self.factors_3d is None:
            self.compute_factors()
        
        logger.info("⚡ 执行回测...")
        
        T = len(self.ohlcv["close"])
        N = len(self.etf_codes)
        
        close_prices = self.ohlcv["close"][self.etf_codes].ffill().bfill().values
        
        # 生成调仓日程
        rebalance_schedule = generate_rebalance_schedule(
            total_periods=T,
            lookback_window=LOOKBACK,
            freq=FREQ,
        )
        num_rebal = len(rebalance_schedule)
        
        # 获取池配置
        pool_factors = self.wfo_config["pool_factors"]
        pool_weights_dict = self.wfo_config.get("pool_weights", {})
        
        target_pools = [p for p in self.config_pools["pools"] if p != "A_SHARE_LIVE"]
        num_pools = len(target_pools)
        
        # 确定每个池的持仓数
        pool_pos_sizes = []
        for pool_name in target_pools:
            pool_config = self.config_pools["pools"][pool_name]
            n_symbols = len(pool_config["symbols"])
            
            if pool_name in ["BOND", "COMMODITY"]:
                pos_size = min(2, n_symbols)
            else:
                pos_size = min(3, n_symbols)
            
            pool_pos_sizes.append(pos_size)
        
        pool_pos_sizes = np.array(pool_pos_sizes, dtype=np.int64)
        max_pos = max(pool_pos_sizes) if len(pool_pos_sizes) > 0 else 3
        
        # 预计算每个调仓日的选股
        pool_selections = np.full((num_rebal, num_pools, max_pos), -1, dtype=np.int64)
        
        for r, t in enumerate(rebalance_schedule):
            if t >= T:
                continue
            
            for p, pool_name in enumerate(target_pools):
                pool_config = self.config_pools["pools"][pool_name]
                pool_symbols = pool_config["symbols"]
                factors = pool_factors.get(pool_name, ["MOM_20D", "SHARPE_RATIO_20D"])
                pos_size = pool_pos_sizes[p]
                
                selected = self._select_top_n_for_pool(t, pool_symbols, factors, pos_size)
                
                for i, etf_idx in enumerate(selected):
                    pool_selections[r, p, i] = etf_idx
        
        # 池权重
        pool_weights = np.array([
            pool_weights_dict.get(pool_name, 0.1) 
            for pool_name in target_pools
        ])
        
        # 执行回测
        total_return, max_dd, win_rate, num_trades, daily_values = _pool_backtest_kernel(
            close_prices,
            pool_selections,
            pool_weights,
            pool_pos_sizes,
            rebalance_schedule,
            INITIAL_CAPITAL,
            COMMISSION_RATE,
        )
        
        # 计算夏普
        returns = pd.Series(daily_values).pct_change(fill_method=None).dropna()
        if len(returns) > 0 and returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe = 0.0
        
        result = {
            "total_return": total_return,
            "max_drawdown": max_dd,
            "sharpe_ratio": sharpe,
            "win_rate": win_rate,
            "num_trades": num_trades,
            "daily_values": daily_values,
        }
        
        return result
    
    def run_baseline(self) -> Dict:
        """运行基准回测（固定 Rank 3 因子）"""
        # 临时替换为基准因子
        original_factors = self.wfo_config["pool_factors"].copy()
        
        baseline_factors = ["ADX_14D", "PRICE_POSITION_20D", "SHARPE_RATIO_20D", "SLOPE_20D"]
        
        self.wfo_config["pool_factors"] = {
            pool_name: baseline_factors
            for pool_name in original_factors
        }
        
        result = self.run_backtest()
        
        # 恢复
        self.wfo_config["pool_factors"] = original_factors
        
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "results/pool_wfo_best_config_latest.json"),
        help="WFO 配置文件路径",
    )
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("🚀 分池 VEC 回测 | Pool-Specific VEC Backtest")
    logger.info("=" * 80)
    
    backtester = PoolVECBacktester(args.config)
    
    # 运行 WFO 优化配置
    logger.info("\n📈 运行 WFO 优化配置...")
    wfo_result = backtester.run_backtest()
    
    # 运行基准配置
    logger.info("\n📊 运行基准配置 (Rank 3)...")
    baseline_result = backtester.run_baseline()
    
    # 打印结果
    logger.info("\n" + "=" * 80)
    logger.info("📊 回测结果对比")
    logger.info("=" * 80)
    
    logger.info(f"\n{'指标':<20} {'WFO优化':<15} {'基准(Rank3)':<15} {'差异':<15}")
    logger.info("-" * 65)
    
    metrics = [
        ("总收益", "total_return", "{:.1%}"),
        ("最大回撤", "max_drawdown", "{:.1%}"),
        ("夏普比率", "sharpe_ratio", "{:.2f}"),
        ("胜率", "win_rate", "{:.1%}"),
        ("交易次数", "num_trades", "{:.0f}"),
    ]
    
    for name, key, fmt in metrics:
        wfo_val = wfo_result[key]
        base_val = baseline_result[key]
        
        if key in ["total_return", "sharpe_ratio", "win_rate"]:
            diff = wfo_val - base_val
            diff_str = f"+{fmt.format(diff)}" if diff > 0 else fmt.format(diff)
        elif key == "max_drawdown":
            diff = base_val - wfo_val  # 回撤越小越好
            diff_str = f"+{fmt.format(diff)}" if diff > 0 else fmt.format(diff)
        else:
            diff_str = "-"
        
        logger.info(f"{name:<20} {fmt.format(wfo_val):<15} {fmt.format(base_val):<15} {diff_str:<15}")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT / "results" / f"pool_vec_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存日线数据
    dates = backtester.ohlcv["close"].index
    daily_df = pd.DataFrame({
        "date": dates,
        "wfo_value": wfo_result["daily_values"],
        "baseline_value": baseline_result["daily_values"],
    })
    daily_df.to_csv(output_dir / "daily_values.csv", index=False)
    
    # 保存汇总
    summary = {
        "timestamp": timestamp,
        "wfo_config": str(args.config),
        "wfo_result": {k: float(v) if isinstance(v, (np.floating, float)) else v 
                       for k, v in wfo_result.items() if k != "daily_values"},
        "baseline_result": {k: float(v) if isinstance(v, (np.floating, float)) else v 
                           for k, v in baseline_result.items() if k != "daily_values"},
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n💾 结果已保存至: {output_dir}")


if __name__ == "__main__":
    main()
