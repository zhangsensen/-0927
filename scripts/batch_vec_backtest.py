#!/usr/bin/env python3
"""
批量 VEC 回测：遍历 WFO 输出的全部组合，逐个用向量化引擎回测并保存结果。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "etf_rotation_optimized"))

import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from numba import njit

from core.data_loader import DataLoader
from core.precise_factor_library_v2 import PreciseFactorLibrary
from core.cross_section_processor import CrossSectionProcessor
from core.market_timing import LightTimingModule
from core.utils.rebalance import shift_timing_signal, generate_rebalance_schedule, ensure_price_views

FREQ = 8
POS_SIZE = 3
INITIAL_CAPITAL = 1_000_000.0
COMMISSION_RATE = 0.0002
LOOKBACK = 252


@njit(cache=True)
def vec_backtest_kernel(
    factors_3d,
    close_prices,
    open_prices,  # ✅ 添加开盘价
    timing_arr,
    factor_indices,
    rebalance_schedule,  # ✅ 改用预生成的调仓日程数组
    pos_size,
    initial_capital,
    commission_rate,
):
    T, N, _ = factors_3d.shape

    cash = initial_capital
    holdings = np.full(N, -1.0)
    entry_prices = np.zeros(N)

    wins = 0
    losses = 0
    total_win_pnl = 0.0
    total_loss_pnl = 0.0

    combined_score = np.empty(N)
    target_set = np.zeros(N, dtype=np.bool_)
    buy_order = np.empty(pos_size, dtype=np.int64)
    new_targets = np.empty(pos_size, dtype=np.int64)

    # ✅ 使用预生成的调仓日程（与 BT 引擎完全一致）
    for i in range(len(rebalance_schedule)):
        t = rebalance_schedule[i]
        if t >= T:
            break

        valid = 0
        for n in range(N):
            score = 0.0
            has_value = False
            for idx in factor_indices:
                val = factors_3d[t - 1, n, idx]
                if not np.isnan(val):
                    score += val
                    has_value = True

            if has_value and score != 0.0:
                combined_score[n] = score
                valid += 1
            else:
                combined_score[n] = -np.inf

        for n in range(N):
            target_set[n] = False

        buy_count = 0
        if valid >= pos_size:
            sorted_idx = np.argsort(combined_score)
            for k in range(pos_size):
                idx = sorted_idx[N - 1 - k]
                if combined_score[idx] == -np.inf:
                    break
                target_set[idx] = True
                buy_order[buy_count] = idx
                buy_count += 1

        # ✅ 改用 t-1 日择时信号 (timing_arr 已在 main 中 shift(1)，故此处用 t 即为 t-1 日信号)
        timing_ratio = timing_arr[t]

        for n in range(N):
            if holdings[n] > 0.0 and not target_set[n]:
                # ✅ 使用收盘价（与 BT Cheat-On-Close 模式对齐）
                price = close_prices[t, n]
                proceeds = holdings[n] * price * (1.0 - commission_rate)
                cash += proceeds

                pnl = (price - entry_prices[n]) / entry_prices[n]
                if pnl > 0.0:
                    wins += 1
                    total_win_pnl += pnl
                else:
                    losses += 1
                    total_loss_pnl += abs(pnl)

                holdings[n] = -1.0
                entry_prices[n] = 0.0

        current_value = cash
        kept_value = 0.0
        for n in range(N):
            if holdings[n] > 0.0:
                # ✅ 使用收盘价
                val = holdings[n] * close_prices[t, n]
                current_value += val
                kept_value += val

        new_count = 0
        for k in range(buy_count):
            idx = buy_order[k]
            if holdings[idx] < 0.0:
                new_targets[new_count] = idx
                new_count += 1

        if new_count > 0:
            target_exposure = current_value * timing_ratio
            available_for_new = target_exposure - kept_value
            if available_for_new < 0.0:
                available_for_new = 0.0

            target_pos_value = available_for_new / new_count / (1.0 + commission_rate)

            if target_pos_value > 0.0:
                for k in range(new_count):
                    idx = new_targets[k]
                    # ✅ 使用收盘价（与 BT Cheat-On-Close 模式对齐）
                    price = close_prices[t, idx]
                    if np.isnan(price) or price <= 0.0:
                        continue

                    shares = target_pos_value / price
                    cost = shares * price * (1.0 + commission_rate)

                    if cash >= cost - 1e-5:
                        actual_cost = cost
                        if actual_cost > cash:
                            actual_cost = cash
                        cash -= actual_cost
                        holdings[idx] = shares
                        entry_prices[idx] = price

    final_value = cash
    for n in range(N):
        if holdings[n] > 0.0:
            # ✅ 改用收盘价平仓（回测结束）
            price = close_prices[T - 1, n]
            if np.isnan(price):
                price = entry_prices[n]

            final_value += holdings[n] * price * (1.0 - commission_rate)

            pnl = (price - entry_prices[n]) / entry_prices[n]
            if pnl > 0.0:
                wins += 1
                total_win_pnl += pnl
            else:
                losses += 1
                total_loss_pnl += abs(pnl)

    num_trades = wins + losses
    total_return = (final_value - initial_capital) / initial_capital
    win_rate = wins / num_trades if num_trades > 0 else 0.0

    if losses > 0:
        avg_win = total_win_pnl / max(wins, 1)
        avg_loss = total_loss_pnl / losses
        profit_factor = avg_win / max(avg_loss, 0.0001)
    else:
        profit_factor = 0.0

    return total_return, win_rate, profit_factor, num_trades


def run_vec_backtest(factors_3d, close_prices, open_prices, timing_arr, factor_indices):
    """运行单个策略的 VEC 回测
    
    Args:
        factors_3d: 因子数据 [T, N, F]
        close_prices: 收盘价 [T, N]
        open_prices: 开盘价 [T, N]，如果为 None 则回退到收盘价并发出警告
        timing_arr: 已 shift 的择时信号 [T]
        factor_indices: 因子索引列表
    
    Returns:
        (total_return, win_rate, profit_factor, num_trades)
    """
    factor_indices_arr = np.array(factor_indices, dtype=np.int64)
    T = factors_3d.shape[0]
    
    # ✅ 使用 ensure_price_views 验证和回退 open_prices
    _, open_arr, close_arr = ensure_price_views(
        close_prices,
        open_prices,
        copy_if_missing=True,
        warn_if_copied=True,
        validate=True,
        min_valid_index=LOOKBACK,  # 跳过预热期的验证
    )
    
    # ✅ 使用共享 helper 生成调仓日程（与 BT 引擎一致）
    rebalance_schedule = generate_rebalance_schedule(
        total_periods=T,
        lookback_window=LOOKBACK,
        freq=FREQ,
    )
    return vec_backtest_kernel(
        factors_3d,
        close_arr,
        open_arr,
        timing_arr,
        factor_indices_arr,
        rebalance_schedule,  # ✅ 传入调仓日程数组
        POS_SIZE,
        INITIAL_CAPITAL,
        COMMISSION_RATE,
    )


def main():
    print("=" * 80)
    print("批量 VEC 回测：遍历全部 WFO 组合")
    print("=" * 80)

    # 1. 加载 WFO 结果
    wfo_dirs = sorted((ROOT / "results").glob("unified_wfo_*"))
    if not wfo_dirs:
        print("❌ 未找到 WFO 结果目录")
        return
    latest_wfo = wfo_dirs[-1]
    combos_path = latest_wfo / "all_combos.parquet"
    if not combos_path.exists():
        print(f"❌ 未找到 {combos_path}")
        return

    df_combos = pd.read_parquet(combos_path)
    print(f"✅ 加载 WFO 结果：{len(df_combos)} 个组合")

    # 2. 加载数据
    config_path = ROOT / "configs/combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )

    # 3. 计算因子
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv)

    factor_names_list = raw_factors_df.columns.get_level_values(0).unique().tolist()
    raw_factors = {fname: raw_factors_df[fname] for fname in factor_names_list}

    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors(raw_factors)

    factor_names = sorted(std_factors.keys())
    first_factor = std_factors[factor_names[0]]
    dates = first_factor.index
    etf_codes = first_factor.columns.tolist()
    T, N = first_factor.shape

    factors_3d = np.stack([std_factors[f].values for f in factor_names], axis=-1)
    # 价格数据处理：先 ffill 再 bfill，确保无 NaN
    close_prices = ohlcv["close"][etf_codes].ffill().bfill().values
    open_prices = ohlcv["open"][etf_codes].ffill().bfill().values

    timing_module = LightTimingModule()
    timing_series = timing_module.compute_position_ratios(ohlcv["close"])
    timing_arr_raw = timing_series.reindex(dates).fillna(1.0).values
    # ✅ 使用共享 helper shift_timing_signal: t 日调仓用 t-1 日的择时信号
    timing_arr = shift_timing_signal(timing_arr_raw)

    print(f"✅ 数据加载完成：{T} 天 × {N} 只 ETF × {len(factor_names)} 个因子")

    # 4. 批量回测
    results = []
    factor_index_map = {name: idx for idx, name in enumerate(factor_names)}
    combo_strings = df_combos["combo"].tolist()
    combo_indices = [
        [factor_index_map[f.strip()] for f in combo.split(" + ")]
        for combo in combo_strings
    ]

    for combo_str, factor_indices in tqdm(
        zip(combo_strings, combo_indices),
        total=len(combo_strings),
        desc="VEC 回测",
    ):
        ret, wr, pf, trades = run_vec_backtest(
            factors_3d, close_prices, open_prices, timing_arr, factor_indices
        )

        results.append(
            {
                "combo": combo_str,
                "vec_return": ret,
                "vec_win_rate": wr,
                "vec_profit_factor": pf,
                "vec_trades": trades,
            }
        )

    # 5. 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT / "results" / f"vec_full_backtest_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    df_results = pd.DataFrame(results)
    df_results.to_parquet(output_dir / "vec_all_combos.parquet", index=False)
    df_results.to_csv(output_dir / "vec_all_combos.csv", index=False)

    print(f"\n✅ VEC 批量回测完成")
    print(f"   输出目录: {output_dir}")
    print(f"   组合数: {len(df_results)}")


if __name__ == "__main__":
    main()
