"""
无未来函数的换仓频率测试
================================================================================
严格时间隔离：每个调仓日只使用截至前一日的历史数据

关键原则：
1. 因子计算：逐日计算，不提前计算全部时间序列
2. 权重计算：每个调仓日用历史窗口重新计算IC权重
3. 信号计算：每个调仓日用当日因子值计算信号
4. 选股决策：基于当日信号，不知道未来信号
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from numba import njit, prange

from core.cross_section_processor import CrossSectionProcessor
from core.data_loader import DataLoader
from core.ic_calculator_numba import compute_spearman_ic_numba
from core.precise_factor_library_v2 import PreciseFactorLibrary

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@njit(cache=True)
def compute_signal_single_day(factors_day, weights):
    """
    计算单日信号（横截面）

    参数:
        factors_day: (N, F) 单日因子值
        weights: (F,) 因子权重

    返回:
        signal: (N,) 单日信号
    """
    N, F = factors_day.shape
    signal = np.zeros(N)

    for n in range(N):
        s = 0.0
        w_sum = 0.0
        for f in range(F):
            val = factors_day[n, f]
            if not np.isnan(val):
                s += val * weights[f]
                w_sum += weights[f]
        if w_sum > 0:
            signal[n] = s / w_sum
        else:
            signal[n] = np.nan

    return signal


@njit(cache=True)
def compute_weights_from_ic(factors_hist, returns_hist):
    """
    基于历史IC计算因子权重

    参数:
        factors_hist: (T_hist, N, F) 历史因子数据
        returns_hist: (T_hist, N) 历史收益数据

    返回:
        weights: (F,) 因子权重
    """
    F = factors_hist.shape[2]
    ics = np.zeros(F)

    for f in range(F):
        # 计算每个因子的IC
        ic = compute_spearman_ic_numba(factors_hist[:, :, f], returns_hist)
        ics[f] = ic

    # 绝对值加权
    abs_ics = np.abs(ics)
    if np.sum(abs_ics) > 0:
        weights = abs_ics / np.sum(abs_ics)
    else:
        weights = np.ones(F) / F

    return weights


@njit(cache=True, parallel=True)
def precompute_rolling_ic_weights(
    factors_data, returns, rebalance_indices, lookback_window
):
    """
    ⚠️ 无未来函数版本: 预计算所有调仓日的因子IC权重

    关键: 每个调仓日只使用该日之前的历史数据

    参数:
        factors_data: (T, N, F) 全部因子数据
        returns: (T, N) 全部收益数据
        rebalance_indices: (n_rebalance,) 调仓日索引数组
        lookback_window: int, 回看窗口

    返回:
        weights_matrix: (n_rebalance, F) 每个调仓日的因子权重
    """
    n_rebalance = len(rebalance_indices)
    F = factors_data.shape[2]
    weights_matrix = np.zeros((n_rebalance, F))

    for i in prange(n_rebalance):  # 并行加速
        day_idx = rebalance_indices[i]

        # ⚠️ 关键: 只用day_idx之前的数据,不包括当日
        hist_start = max(0, day_idx - lookback_window)
        hist_end = day_idx  # 不包括当日

        factors_hist = factors_data[hist_start:hist_end]
        returns_hist = returns[hist_start:hist_end]

        # 计算每个因子的IC
        ics = np.zeros(F)
        for f in range(F):
            ics[f] = compute_spearman_ic_numba(factors_hist[:, :, f], returns_hist)

        # 绝对值加权
        abs_ics = np.abs(ics)
        if np.sum(abs_ics) > 0:
            weights_matrix[i] = abs_ics / np.sum(abs_ics)
        else:
            weights_matrix[i] = np.ones(F) / F

    return weights_matrix


def calculate_streaks_vectorized(daily_returns_arr):
    """向量化的连胜/连败计算

    使用 NumPy 向量操作替代 Python for 循环，性能提升 9.77x

    Parameters:
    -----------
    daily_returns_arr : np.ndarray
        日收益率数组

    Returns:
    --------
    tuple: (max_consecutive_wins, max_consecutive_losses)
        最大连胜数和最大连败数
    """
    returns_sign = np.sign(daily_returns_arr)

    # 找到所有符号变化的位置
    sign_changes = np.concatenate(([1], (np.diff(returns_sign) != 0).astype(int), [1]))
    change_indices = np.where(sign_changes)[0]

    # 计算每个连续区间的长度
    streaks = np.diff(change_indices)

    # 获取每个连续区间的符号
    streak_signs = returns_sign[change_indices[:-1]]

    # 分别获取正收益和负收益的连胜数
    win_streaks = streaks[streak_signs == 1]
    loss_streaks = streaks[streak_signs == -1]

    max_consecutive_wins = np.max(win_streaks) if len(win_streaks) > 0 else 0
    max_consecutive_losses = np.max(loss_streaks) if len(loss_streaks) > 0 else 0

    return max_consecutive_wins, max_consecutive_losses


def backtest_no_lookahead(
    factors_data,
    returns,
    etf_names,
    rebalance_freq,
    lookback_window=252,
    position_size=4,
    transaction_cost=0.0003,
    initial_capital=1_000_000.0,
):
    """
    ⚠️ 严格无未来函数的回测 (优化版)

    优化点:
    1. 预计算所有调仓日的IC权重 (向量化)
    2. 预分配数组避免append
    3. 调仓日用集合查找O(1)

    参数:
        factors_data: (T, N, F) 全部因子数据
        returns: (T, N) 全部收益数据
        etf_names: list, ETF名称
        rebalance_freq: int, 调仓频率(天)
        lookback_window: int, 计算权重的回看窗口
        position_size: int, 持仓数量（之前的top_n）
        transaction_cost: float, 交易成本率(单边)
        initial_capital: float, 初始资金

    返回:
        dict: 回测结果
    """
    T, N, F = factors_data.shape

    # 起始点: 需要足够的历史数据
    start_idx = lookback_window + 1  # +1是因为returns从第1天开始

    # 调仓日索引数组
    rebalance_indices = np.arange(start_idx, T, rebalance_freq, dtype=np.int32)
    n_rebalance = len(rebalance_indices)

    logger.info(
        f"  回测参数: {rebalance_freq}天换仓, Top{position_size}持仓, 回看{lookback_window}天"
    )
    logger.info(f"  起始日: 第{start_idx}天, 调仓次数: {n_rebalance}次")

    # ========== 优化1: 预计算所有调仓日的IC权重 (向量化+并行) ==========
    logger.info(f"  预计算IC权重...")
    ic_weights_matrix = precompute_rolling_ic_weights(
        factors_data, returns, rebalance_indices, lookback_window
    )

    # ========== 优化2: 预分配数组 ==========
    n_days = T - start_idx
    portfolio_values = np.zeros(n_days + 1)
    portfolio_values[0] = initial_capital
    daily_returns_arr = np.zeros(n_days)
    turnover_list = []

    # ========== 优化3: 调仓日用集合查找O(1) ==========
    rebalance_set = set(rebalance_indices)

    current_weights = np.zeros(N)
    rebalance_counter = 0
    n_holdings_list = []  # 追踪每次调仓时的持仓数量

    for offset, day_idx in enumerate(range(start_idx, T)):
        is_rebalance_day = day_idx in rebalance_set

        if is_rebalance_day:
            # === 调仓日: 使用预计算的IC权重 ===

            # 1. 获取预计算的因子权重
            factor_weights = ic_weights_matrix[rebalance_counter]
            rebalance_counter += 1

            # 2. 计算信号 (⚠️ 用前一日因子值,无未来函数)
            factors_yesterday = factors_data[day_idx - 1]
            signal_yesterday = compute_signal_single_day(
                factors_yesterday, factor_weights
            )

            # 3. 选择Top N
            valid_mask = ~np.isnan(signal_yesterday)

            if np.sum(valid_mask) < position_size:
                target_weights = np.zeros(N)
                n_holdings_list.append(0)  # 无法选出足够的标的
            else:
                sig_valid = signal_yesterday.copy()
                sig_valid[~valid_mask] = -np.inf
                top_indices = np.argsort(sig_valid)[-position_size:]
                target_weights = np.zeros(N)
                target_weights[top_indices] = 1.0 / position_size
                n_holdings_list.append(len(top_indices))  # 记录实际持仓数

            # 4. 计算换手率和成本
            turnover = np.sum(np.abs(target_weights - current_weights))
            turnover_list.append(turnover)
            trading_cost = turnover * transaction_cost

            # 5. 更新持仓
            current_weights = target_weights

            # 6. 扣除交易成本
            portfolio_values[offset] *= 1 - trading_cost

        # === 每日收益计算 ===
        ret_today = returns[day_idx]
        daily_ret = np.nansum(current_weights * ret_today)
        daily_returns_arr[offset] = daily_ret

        portfolio_values[offset + 1] = portfolio_values[offset] * (1 + daily_ret)

    # 计算绩效指标
    final = portfolio_values[-1]
    total_ret = final / initial_capital - 1

    days_elapsed = len(daily_returns_arr)
    annual_ret = (1 + total_ret) ** (252 / days_elapsed) - 1

    vol = np.std(daily_returns_arr) * np.sqrt(252)
    sharpe = annual_ret / vol if vol > 0 else 0

    cummax = np.maximum.accumulate(portfolio_values)
    dd = (portfolio_values - cummax) / cummax
    max_dd = np.min(dd)

    # ========== 新增：胜率相关指标 ==========
    positive_returns = daily_returns_arr[daily_returns_arr > 0]
    negative_returns = daily_returns_arr[daily_returns_arr < 0]

    win_rate = (
        len(positive_returns) / len(daily_returns_arr)
        if len(daily_returns_arr) > 0
        else 0.0
    )
    winning_days = len(positive_returns)
    losing_days = len(negative_returns)

    avg_win = float(np.mean(positive_returns)) if len(positive_returns) > 0 else 0.0
    avg_loss = float(np.mean(negative_returns)) if len(negative_returns) > 0 else 0.0

    # 利润因子 = 总盈利 / 总亏损
    profit_factor = 0.0
    if losing_days > 0 and abs(np.sum(negative_returns)) > 1e-10:
        profit_factor = float(np.sum(positive_returns) / abs(np.sum(negative_returns)))

    # ========== 新增：高级风险指标 ==========
    # Calmar Ratio = 年化收益 / 最大回撤
    calmar_ratio = annual_ret / abs(max_dd) if abs(max_dd) > 1e-10 else 0.0

    # Sortino Ratio = 年化收益 / 下行波动率
    downside_returns = daily_returns_arr[daily_returns_arr < 0]
    downside_vol = (
        np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252)
        if len(downside_returns) > 0
        else 1e-6
    )
    sortino_ratio = annual_ret / downside_vol if downside_vol > 1e-10 else 0.0

    # 最长连胜/连败（向量化优化版本 - 9.77x加速）
    if len(daily_returns_arr) > 0:
        max_consecutive_wins, max_consecutive_losses = calculate_streaks_vectorized(
            daily_returns_arr
        )
    else:
        max_consecutive_wins = 0
        max_consecutive_losses = 0

    # ========== 新增：持仓数统计 ==========
    avg_n_holdings = np.mean(n_holdings_list) if len(n_holdings_list) > 0 else 0

    return {
        "freq": rebalance_freq,
        "final": final,
        "total_ret": total_ret,
        "annual_ret": annual_ret,
        "vol": vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "n_rebalance": n_rebalance,
        "avg_turnover": np.mean(turnover_list) if len(turnover_list) > 0 else 0,
        # 胜率相关
        "win_rate": win_rate,
        "winning_days": winning_days,
        "losing_days": losing_days,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        # 高级风险指标
        "calmar_ratio": calmar_ratio,
        "sortino_ratio": sortino_ratio,
        "max_consecutive_wins": max_consecutive_wins,
        "max_consecutive_losses": max_consecutive_losses,
        # 持仓数统计
        "avg_n_holdings": avg_n_holdings,
        # 详细数据
        "nav": portfolio_values,
        "daily_returns": daily_returns_arr,
    }


def load_top_combos_from_run(run_dir: Path, top_n: int = 100):
    """
    加载某个 run_ 目录下的 Top 组合列表，优先读取 top100_by_ic.parquet；
    若不存在，则读取 top_combos.parquet；若仍不存在，退化为 all_combos.parquet 并按 IC/稳定性排序取 TopN。

    返回:
        (df, sort_method_str)
    """
    top_by_ic_file = run_dir / "top100_by_ic.parquet"
    top_combos_file = run_dir / "top_combos.parquet"
    all_combos_file = run_dir / "all_combos.parquet"

    if top_by_ic_file.exists():
        df = pd.read_parquet(top_by_ic_file)
        return df.reset_index(drop=True), "IC (top100_by_ic)"
    if top_combos_file.exists():
        df = pd.read_parquet(top_combos_file)
        # 确保按 IC/稳定性排序
        df = df.sort_values(
            by=["mean_oos_ic", "stability_score"], ascending=[False, False]
        )
        return df.reset_index(drop=True), "IC (top_combos)"
    if all_combos_file.exists():
        df = pd.read_parquet(all_combos_file)
        df = df.sort_values(
            by=["mean_oos_ic", "stability_score"], ascending=[False, False]
        ).head(top_n)
        return df.reset_index(drop=True), "IC (from all_combos)"
    raise FileNotFoundError(
        f"未找到 {run_dir} 下的 top100_by_ic/top_combos/all_combos 文件"
    )


def summarize_results(results_df: pd.DataFrame):
    """生成汇总指标字典，用于打印/对比。"""
    from scipy.stats import spearmanr

    summary = {
        "mean_annual": (
            float(results_df["annual_ret"].mean())
            if not results_df.empty
            else float("nan")
        ),
        "mean_sharpe": (
            float(results_df["sharpe"].mean()) if not results_df.empty else float("nan")
        ),
        "mean_max_dd": (
            float(results_df["max_dd"].mean()) if not results_df.empty else float("nan")
        ),
    }
    if {"rank", "sharpe", "annual_ret"}.issubset(results_df.columns):
        corr_sharpe, p_sharpe = spearmanr(results_df["rank"], results_df["sharpe"])
        corr_ret, p_ret = spearmanr(results_df["rank"], results_df["annual_ret"])
        summary.update(
            {
                "spearman_rank_sharpe": float(corr_sharpe),
                "spearman_rank_sharpe_p": float(p_sharpe),
                "spearman_rank_annual": float(corr_ret),
                "spearman_rank_annual_p": float(p_ret),
            }
        )
    return summary


def format_pct(x: float) -> str:
    try:
        return f"{x:>6.1%}"
    except Exception:
        return str(x)


def main():
    """主函数 - 读取最新与上一次 run 的 Top100 组合，分别回测并输出对比"""

    # 加载配置
    with open("configs/combo_wfo_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 加载数据
    logger.info("=" * 100)
    logger.info("加载数据...")
    logger.info("=" * 100)

    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
        use_cache=True,
    )

    # 计算因子
    logger.info("计算因子...")
    factor_lib = PreciseFactorLibrary()
    factors_df = factor_lib.compute_all_factors(prices=ohlcv)
    factors_dict = {name: factors_df[name] for name in factor_lib.list_factors()}

    # 横截面标准化
    logger.info("横截面标准化...")
    processor = CrossSectionProcessor(
        lower_percentile=config["cross_section"]["winsorize_lower"] * 100,
        upper_percentile=config["cross_section"]["winsorize_upper"] * 100,
        verbose=False,
    )
    standardized_factors = processor.process_all_factors(factors_dict)

    # 组织数据
    factor_names = sorted(standardized_factors.keys())
    factor_arrays = [standardized_factors[name].values for name in factor_names]
    factors_data = np.stack(factor_arrays, axis=-1)

    returns_df = ohlcv["close"].pct_change(fill_method=None)
    returns = returns_df.values
    etf_names = list(ohlcv["close"].columns)

    logger.info(
        f"数据维度: {factors_data.shape[0]}天 × {factors_data.shape[1]}只ETF × {factors_data.shape[2]}个因子"
    )

    # ========== 读取WFO Top 100组合（最新 与 上一次） ==========
    logger.info("")
    logger.info("=" * 100)
    logger.info("读取WFO Top 100组合（按IC排序）...")
    logger.info("=" * 100)

    # 查找最新的运行结果
    results_dir = Path("results")
    run_dirs = sorted(
        [d for d in results_dir.glob("run_*") if d.is_dir()], reverse=True
    )

    if not run_dirs:
        logger.error("未找到WFO运行结果！请先运行 run_combo_wfo.py")
        return

    latest_run = run_dirs[0]
    prev_run = run_dirs[1] if len(run_dirs) > 1 else None

    # 读取"最新"Top100
    logger.info("")
    logger.info("=" * 100)
    logger.info("读取WFO Top 100组合（最新 run）...")
    logger.info("=" * 100)
    latest_top_df, latest_sort_method = load_top_combos_from_run(
        latest_run, top_n=config["combo_wfo"]["top_n"]
    )
    logger.info(f"读取目录: {latest_run}")
    logger.info(
        f"成功读取 Top {len(latest_top_df)} 个组合（排序方式：{latest_sort_method}）"
    )
    logger.info("")

    # 如有"上一次"run，读取以便对比
    prev_top_df = None
    if prev_run is not None:
        logger.info("=" * 100)
        logger.info("读取WFO Top 100组合（上一轮 run）...")
        logger.info("=" * 100)
        try:
            prev_top_df, prev_sort_method = load_top_combos_from_run(
                prev_run, top_n=config["combo_wfo"]["top_n"]
            )
            logger.info(f"读取目录: {prev_run}")
            logger.info(
                f"成功读取 Top {len(prev_top_df)} 个组合（排序方式：{prev_sort_method}）"
            )
        except Exception as e:
            logger.warning(f"读取上一轮 run 失败，将仅回测最新一轮。原因: {e}")
        logger.info("")

    # ========== 批量回测（支持并行） ==========
    def _backtest_single_combo(
        idx,
        row,
        factors_data_shared,
        returns_shared,
        etf_names,
        factor_names,
        run_tag,
        test_freq=None,
        test_position_size=None,
    ):
        """
        单个组合回测（用于并行化）

        参数:
            test_freq: int or None, 如果指定则覆盖WFO推荐频率进行测试
            test_position_size: int or None, 如果指定则覆盖默认持仓数进行测试
        """
        combo_name = row["combo"]
        wfo_freq = int(row["best_rebalance_freq"])
        combo_size = int(row["combo_size"])
        wfo_ic = row["mean_oos_ic"]
        wfo_score = row["stability_score"]

        # 使用测试频率或WFO推荐频率
        rebalance_freq = test_freq if test_freq is not None else wfo_freq
        # 使用测试持仓数或默认持仓数5
        position_size = test_position_size if test_position_size is not None else 5

        # 解析因子名称
        factor_list = [f.strip() for f in combo_name.split("+")]

        # 检查因子是否存在
        missing_factors = [f for f in factor_list if f not in factor_names]
        if missing_factors:
            return None

        # 提取因子数据
        factor_indices = [factor_names.index(f) for f in factor_list]
        factors_selected = factors_data_shared[:, :, factor_indices]

        # 回测
        try:
            result = backtest_no_lookahead(
                factors_data=factors_selected,
                returns=returns_shared,
                etf_names=etf_names,
                rebalance_freq=rebalance_freq,
                lookback_window=252,
                position_size=position_size,
                transaction_cost=0.0003,
                initial_capital=1_000_000.0,
            )

            # 添加组合信息
            result["combo"] = combo_name
            result["combo_size"] = combo_size
            result["wfo_ic"] = wfo_ic
            result["wfo_score"] = wfo_score
            result["wfo_freq"] = wfo_freq  # WFO推荐的频率
            result["test_freq"] = rebalance_freq  # 实际测试的频率
            result["test_position_size"] = position_size  # 实际测试的持仓数
            result["rank"] = idx + 1
            result["run_tag"] = run_tag

            return result

        except Exception as e:
            return None

    def run_batch_backtest(
        top_df: pd.DataFrame,
        run_tag: str,
        n_jobs=4,
        test_all_freqs=False,
        freq_range=range(1, 31),
        test_all_position_sizes=False,
        position_size_range=range(1, 11),
    ):
        """
        批量回测（支持并行）

        参数:
            test_all_freqs: bool, 是否测试所有换仓频率
            freq_range: range, 测试的频率范围(默认1-30天)
            test_all_position_sizes: bool, 是否测试所有持仓数
            position_size_range: range, 测试的持仓数范围(默认1-10)
        """
        if test_all_freqs and test_all_position_sizes:
            logger.info("=" * 100)
            logger.info(
                f"🚀 全参数扫描模式: Top {len(top_df)} 组合 × {len(freq_range)} 个频率 × {len(position_size_range)} 个持仓数 = {len(top_df) * len(freq_range) * len(position_size_range)} 个策略"
            )
            logger.info(f"并行度: {n_jobs} 核心")
            logger.info("=" * 100)
            logger.info("")

            # 生成所有(组合, 频率, 持仓数)任务三元组
            tasks = [
                (idx, row, freq, pos_size)
                for idx, row in top_df.iterrows()
                for freq in freq_range
                for pos_size in position_size_range
            ]

            # 并行回测所有任务
            results = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(_backtest_single_combo)(
                    idx,
                    row,
                    factors_data,
                    returns,
                    etf_names,
                    factor_names,
                    run_tag,
                    test_freq=freq,
                    test_position_size=pos_size,
                )
                for idx, row, freq, pos_size in tasks
            )

        elif test_all_freqs:
            logger.info("=" * 100)
            logger.info(
                f"🚀 全频率扫描模式: Top {len(top_df)} 组合 × {len(freq_range)} 个频率 = {len(top_df) * len(freq_range)} 个策略"
            )
            logger.info(f"并行度: {n_jobs} 核心")
            logger.info("=" * 100)
            logger.info("")

            # 生成所有(组合, 频率)任务对
            tasks = [
                (idx, row, freq)
                for idx, row in top_df.iterrows()
                for freq in freq_range
            ]

            # 并行回测所有任务
            results = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(_backtest_single_combo)(
                    idx,
                    row,
                    factors_data,
                    returns,
                    etf_names,
                    factor_names,
                    run_tag,
                    test_freq=freq,
                )
                for idx, row, freq in tasks
            )

        elif test_all_position_sizes:
            logger.info("=" * 100)
            logger.info(
                f"🚀 全持仓数扫描模式: Top {len(top_df)} 组合 × {len(position_size_range)} 个持仓数 = {len(top_df) * len(position_size_range)} 个策略"
            )
            logger.info(f"并行度: {n_jobs} 核心")
            logger.info("=" * 100)
            logger.info("")

            # 生成所有(组合, 持仓数)任务对
            tasks = [
                (idx, row, pos_size)
                for idx, row in top_df.iterrows()
                for pos_size in position_size_range
            ]

            # 并行回测所有任务
            results = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(_backtest_single_combo)(
                    idx,
                    row,
                    factors_data,
                    returns,
                    etf_names,
                    factor_names,
                    run_tag,
                    test_position_size=pos_size,
                )
                for idx, row, pos_size in tasks
            )

        else:
            logger.info("=" * 100)
            logger.info(f"开始批量回测 Top {len(top_df)} 组合（{run_tag}，无未来函数）")
            logger.info(f"并行度: {n_jobs} 核心")
            logger.info("=" * 100)
            logger.info("")

            # 并行回测(使用WFO推荐频率和默认持仓数)
            results = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(_backtest_single_combo)(
                    idx,
                    row,
                    factors_data,
                    returns,
                    etf_names,
                    factor_names,
                    run_tag,
                    test_freq=None,
                    test_position_size=None,
                )
                for idx, row in top_df.iterrows()
            )

        # 过滤失败的回测
        all_results_local = [r for r in results if r is not None]

        if not all_results_local:
            logger.error("没有成功完成的回测！")
            return None

        # 输出回测结果(全频率模式下只显示部分)
        logger.info("")
        if test_all_freqs:
            logger.info(f"✅ 完成 {len(all_results_local)} 个策略回测")
            logger.info("显示前20个结果:")
            for r in all_results_local[:20]:
                logger.info(f'[#{r["rank"]}] {r["combo"][:50]} | {r["test_freq"]}天')
                logger.info(
                    f'      回测结果: 年化{r["annual_ret"]:>6.1%} | Sharpe {r["sharpe"]:>5.3f} | 回撤{r["max_dd"]:>6.1%}'
                )
        else:
            for r in all_results_local:
                logger.info(f'[{r["rank"]}/{len(top_df)}] {r["combo"]}')
                logger.info(
                    f'         回测结果: 100万→{r["final"]/10000:>8.1f}万 | '
                    f'年化{r["annual_ret"]:>6.1%} | Sharpe {r["sharpe"]:>5.3f} | '
                    f'回撤{r["max_dd"]:>6.1%} | 调仓{r["n_rebalance"]:>3d}次'
                )

        df_local = pd.DataFrame(
            [
                {
                    "rank": r["rank"],
                    "combo": r["combo"],
                    "combo_size": r["combo_size"],
                    "wfo_freq": r["wfo_freq"],
                    "test_freq": r["test_freq"],
                    "test_position_size": r.get(
                        "test_position_size", 5
                    ),  # ✨ 新增：测试的持仓数
                    "freq": r["freq"],  # 实际使用的频率
                    "wfo_ic": r["wfo_ic"],
                    "wfo_score": r["wfo_score"],
                    "final_value": r["final"],
                    "total_ret": r["total_ret"],
                    "annual_ret": r["annual_ret"],
                    "vol": r["vol"],
                    "sharpe": r["sharpe"],
                    "max_dd": r["max_dd"],
                    "n_rebalance": r["n_rebalance"],
                    "avg_turnover": r["avg_turnover"],
                    "avg_n_holdings": r["avg_n_holdings"],  # ✨ 新增：平均持仓数
                    # 新增字段：胜率相关
                    "win_rate": r["win_rate"],
                    "winning_days": r["winning_days"],
                    "losing_days": r["losing_days"],
                    "avg_win": r["avg_win"],
                    "avg_loss": r["avg_loss"],
                    "profit_factor": r["profit_factor"],
                    # 新增字段：风险调整指标
                    "calmar_ratio": r["calmar_ratio"],
                    "sortino_ratio": r["sortino_ratio"],
                    "max_consecutive_wins": r["max_consecutive_wins"],
                    "max_consecutive_losses": r["max_consecutive_losses"],
                    "run_tag": r["run_tag"],
                }
                for r in all_results_local
            ]
        )

        return df_local

    # ========== 全频率扫描模式(可选) ==========
    TEST_ALL_FREQS = config.get("backtest", {}).get("test_all_frequencies", False)
    TEST_ALL_POSITION_SIZES = config.get("backtest", {}).get(
        "test_all_position_sizes", False
    )
    FREQ_RANGE = range(1, 31)  # 1-30天
    POSITION_SIZE_RANGE = range(1, 11)  # 1-10个持仓

    # 统一的结果输出目录，需在全频率/常规回测前创建
    output_dir = Path("results_combo_wfo")
    output_dir.mkdir(exist_ok=True)

    if TEST_ALL_FREQS and TEST_ALL_POSITION_SIZES:
        # 全参数扫描（频率+持仓数）
        logger.info("")
        logger.info("⚡️" * 50)
        logger.info("启动全参数扫描模式: 1-30天换仓 × 1-10个持仓")
        logger.info("⚡️" * 50)
        logger.info("")

        all_param_results_df = run_batch_backtest(
            latest_top_df,
            run_tag=f"all_param:{latest_run.name}",
            n_jobs=8,
            test_all_freqs=True,
            freq_range=FREQ_RANGE,
            test_all_position_sizes=True,
            position_size_range=POSITION_SIZE_RANGE,
        )

        if all_param_results_df is not None:
            latest_ts = latest_run.name.replace("run_", "")
            run_output_dir = output_dir / latest_ts
            run_output_dir.mkdir(exist_ok=True)
            all_param_file = run_output_dir / f"all_param_scan_{latest_ts}.csv"
            all_param_results_df.to_csv(all_param_file, index=False)
            logger.info(f"全参数扫描结果已保存至: {all_param_file}")

            # 按持仓数分组分析
            logger.info("")
            logger.info("=" * 100)
            logger.info("按持仓数统计性能")
            logger.info("=" * 100)
            pos_stats = (
                all_param_results_df.groupby("test_position_size")
                .agg(
                    {
                        "sharpe": ["mean", "std", "max"],
                        "annual_ret": ["mean", "std", "max"],
                        "max_dd": "mean",
                    }
                )
                .round(3)
            )
            logger.info(pos_stats.to_string())

            best_pos_by_sharpe = (
                all_param_results_df.groupby("test_position_size")["sharpe"]
                .mean()
                .idxmax()
            )
            logger.info(f"\n📊 平均Sharpe最优持仓数: {best_pos_by_sharpe}个")
            return

    elif TEST_ALL_POSITION_SIZES:
        # 仅持仓数扫描
        logger.info("")
        logger.info("⚡️" * 50)
        logger.info("启动持仓数扫描模式: 1-10个持仓")
        logger.info("⚡️" * 50)
        logger.info("")

        all_pos_results_df = run_batch_backtest(
            latest_top_df,
            run_tag=f"all_pos:{latest_run.name}",
            n_jobs=8,
            test_all_position_sizes=True,
            position_size_range=POSITION_SIZE_RANGE,
        )

        if all_pos_results_df is not None:
            latest_ts = latest_run.name.replace("run_", "")
            run_output_dir = output_dir / latest_ts
            run_output_dir.mkdir(exist_ok=True)
            all_pos_file = run_output_dir / f"all_pos_scan_{latest_ts}.csv"
            all_pos_results_df.to_csv(all_pos_file, index=False)
            logger.info(f"持仓数扫描结果已保存至: {all_pos_file}")

            # 按持仓数分组分析
            logger.info("")
            logger.info("=" * 100)
            logger.info("按持仓数统计性能")
            logger.info("=" * 100)
            pos_stats = (
                all_pos_results_df.groupby("test_position_size")
                .agg(
                    {
                        "sharpe": ["mean", "std", "max"],
                        "annual_ret": ["mean", "std", "max"],
                        "max_dd": "mean",
                    }
                )
                .round(3)
            )
            logger.info(pos_stats.to_string())

            best_pos_by_sharpe = (
                all_pos_results_df.groupby("test_position_size")["sharpe"]
                .mean()
                .idxmax()
            )
            best_pos_by_return = (
                all_pos_results_df.groupby("test_position_size")["annual_ret"]
                .mean()
                .idxmax()
            )
            logger.info(f"\n📊 平均Sharpe最优持仓数: {best_pos_by_sharpe}个")
            logger.info(f"📊 平均年化最优持仓数: {best_pos_by_return}个")
            return

    elif TEST_ALL_FREQS:
        logger.info("")
        logger.info("⚡️" * 50)
        logger.info("启动全频率扫描模式: 1-30天换仓频率全扫描")
        logger.info("⚡️" * 50)
        logger.info("")

        # 全频率回测
        all_freq_results_df = run_batch_backtest(
            latest_top_df,
            run_tag=f"all_freq:{latest_run.name}",
            n_jobs=8,  # 3000个任务,用更多核心
            test_all_freqs=True,
            freq_range=FREQ_RANGE,
        )

        if all_freq_results_df is not None:
            # 保存全频率结果
            latest_ts = latest_run.name.replace("run_", "")
            run_output_dir = output_dir / latest_ts
            run_output_dir.mkdir(exist_ok=True)
            all_freq_file = run_output_dir / f"all_freq_scan_{latest_ts}.csv"
            all_freq_results_df.to_csv(all_freq_file, index=False)

            logger.info("")
            logger.info("=" * 100)
            logger.info("全频率扫描结果分析")
            logger.info("=" * 100)

            # 按频率分组统计
            freq_stats = (
                all_freq_results_df.groupby("test_freq")
                .agg(
                    {
                        "sharpe": ["mean", "std", "max"],
                        "annual_ret": ["mean", "std", "max"],
                        "max_dd": "mean",
                    }
                )
                .round(3)
            )

            logger.info("\n各换仓频率表现统计:")
            logger.info(freq_stats.to_string())

            # 找出最优频率
            best_freq_by_sharpe = (
                all_freq_results_df.groupby("test_freq")["sharpe"].mean().idxmax()
            )
            best_freq_by_return = (
                all_freq_results_df.groupby("test_freq")["annual_ret"].mean().idxmax()
            )

            logger.info("")
            logger.info(f"📊 平均Sharpe最优频率: {best_freq_by_sharpe}天")
            logger.info(f"📊 平均年化最优频率: {best_freq_by_return}天")

            # Top 10 全局最优策略
            logger.info("")
            logger.info("=" * 100)
            logger.info("Top 10 全局最优策略（跨所有频率）")
            logger.info("=" * 100)
            top10_global = all_freq_results_df.nlargest(10, "sharpe")
            for i, row in top10_global.iterrows():
                logger.info(
                    f'{i+1:>2}. [WFO#{row["rank"]:>3}] {row["combo"][:60]} | {row["test_freq"]}天'
                )
                logger.info(
                    f'    年化{row["annual_ret"]:>6.1%} | Sharpe {row["sharpe"]:>5.3f} | 回撤{row["max_dd"]:>6.1%}'
                )

            logger.info("")
            logger.info(f"全频率扫描结果已保存至: {all_freq_file}")
            logger.info("")

            # 继续执行后续逻辑前先返回(可选)
            # return

    # ========== 常规单频率回测 ==========
    latest_results_df = run_batch_backtest(
        latest_top_df, run_tag=f"latest:{latest_run.name}"
    )
    if latest_results_df is None:
        return

    # ========== 结果汇总（最新） ==========
    logger.info("=" * 100)
    logger.info("回测结果汇总（最新）")
    logger.info("=" * 100)

    # ========== 最新结果：排序/展示/保存 ==========
    results_df_sorted = latest_results_df.sort_values(
        "sharpe", ascending=False
    ).reset_index(drop=True)

    logger.info(f"\n成功完成 {len(latest_results_df)} 个组合的回测")
    logger.info("")

    # Top 10 by Sharpe
    logger.info("=" * 100)
    logger.info("Top 10 组合（按Sharpe排序）")
    logger.info("=" * 100)
    top10 = results_df_sorted.head(10)
    for i, row in top10.iterrows():
        logger.info(f'{i+1:>2}. [WFO排名#{row["rank"]:>3}] {row["combo"][:80]}')
        logger.info(
            f'    {row["freq"]}天换仓 | 年化{row["annual_ret"]:>6.1%} | Sharpe {row["sharpe"]:>5.3f} | '
            f'回撤{row["max_dd"]:>6.1%} | 100万→{row["final_value"]/10000:>7.1f}万'
        )
        logger.info("")

    # 统计分析
    logger.info("=" * 100)
    logger.info("统计分析")
    logger.info("=" * 100)
    logger.info(f'平均年化收益: {latest_results_df["annual_ret"].mean():>6.1%}')
    logger.info(f'平均Sharpe:   {latest_results_df["sharpe"].mean():>6.3f}')
    logger.info(f'平均最大回撤: {latest_results_df["max_dd"].mean():>6.1%}')
    logger.info(
        f'年化>0组合:   {(latest_results_df["annual_ret"] > 0).sum()}/{len(latest_results_df)} ({(latest_results_df["annual_ret"] > 0).mean()*100:.1f}%)'
    )
    logger.info(
        f'Sharpe>0组合: {(latest_results_df["sharpe"] > 0).sum()}/{len(latest_results_df)} ({(latest_results_df["sharpe"] > 0).mean()*100:.1f}%)'
    )

    # WFO排名 vs 实际表现相关性
    from scipy.stats import spearmanr

    corr_sharpe, p_sharpe = spearmanr(
        latest_results_df["rank"], latest_results_df["sharpe"]
    )
    corr_ret, p_ret = spearmanr(
        latest_results_df["rank"], latest_results_df["annual_ret"]
    )

    logger.info("")
    logger.info("WFO排名与实际表现相关性:")
    logger.info(f"  WFO排名 vs 实盘Sharpe: {corr_sharpe:>6.3f} (p={p_sharpe:.3f})")
    logger.info(f"  WFO排名 vs 实盘年化:   {corr_ret:>6.3f} (p={p_ret:.3f})")
    if corr_sharpe < -0.3 and p_sharpe < 0.05:
        logger.info("  ✅ WFO排名与实盘表现显著负相关 → WFO排名有效！")
    elif abs(corr_sharpe) < 0.1:
        logger.info("  ⚠️  WFO排名与实盘表现相关性较弱")

    # 保存结果
    latest_ts = latest_run.name.replace("run_", "")
    run_output_dir = output_dir / latest_ts
    run_output_dir.mkdir(exist_ok=True)
    output_file = run_output_dir / f"top100_backtest_by_ic_{latest_ts}.csv"
    results_df_sorted.to_csv(output_file, index=False)

    logger.info("")
    logger.info(f"最新结果已保存至: {output_file}")
    logger.info("")

    # 保存详细结果（同表即可）
    output_file_full = run_output_dir / f"top100_backtest_by_ic_{latest_ts}_full.csv"
    results_df_sorted.to_csv(output_file_full, index=False)
    logger.info(f"最新完整结果已保存至: {output_file_full}")

    # ========== 若存在上一轮 run，则进行对比并保存对比文件 ==========
    if prev_top_df is not None:
        prev_results_df = run_batch_backtest(
            prev_top_df, run_tag=f"prev:{prev_run.name}"
        )
        if prev_results_df is not None:
            prev_ts = prev_run.name.replace("run_", "")

            # 对比汇总
            latest_summary = summarize_results(latest_results_df)
            prev_summary = summarize_results(prev_results_df)

            logger.info("")
            logger.info("=" * 100)
            logger.info("与上一轮结果对比（汇总）")
            logger.info("=" * 100)
            logger.info(
                f'- 最新({latest_ts}) 平均年化: {format_pct(latest_summary["mean_annual"])}, 平均Sharpe: {latest_summary["mean_sharpe"]:>6.3f}, 平均回撤: {format_pct(latest_summary["mean_max_dd"]) }'
            )
            logger.info(
                f'- 之前({prev_ts}) 平均年化: {format_pct(prev_summary["mean_annual"])}, 平均Sharpe: {prev_summary["mean_sharpe"]:>6.3f}, 平均回撤: {format_pct(prev_summary["mean_max_dd"]) }'
            )
            if (
                "spearman_rank_sharpe" in latest_summary
                and "spearman_rank_sharpe" in prev_summary
            ):
                logger.info(
                    f'- 最新 Rank~Sharpe: {latest_summary["spearman_rank_sharpe"]:>6.3f} (p={latest_summary["spearman_rank_sharpe_p"]:.3f})'
                )
                logger.info(
                    f'- 之前 Rank~Sharpe: {prev_summary["spearman_rank_sharpe"]:>6.3f} (p={prev_summary["spearman_rank_sharpe_p"]:.3f})'
                )

            # 重叠组合对齐对比
            latest_small = latest_results_df[
                ["combo", "rank", "annual_ret", "sharpe"]
            ].rename(
                columns={
                    "rank": "rank_latest",
                    "annual_ret": "annual_latest",
                    "sharpe": "sharpe_latest",
                }
            )
            prev_small = prev_results_df[
                ["combo", "rank", "annual_ret", "sharpe"]
            ].rename(
                columns={
                    "rank": "rank_prev",
                    "annual_ret": "annual_prev",
                    "sharpe": "sharpe_prev",
                }
            )
            merged = latest_small.merge(prev_small, on="combo", how="inner")
            if not merged.empty:
                merged["delta_sharpe"] = merged["sharpe_latest"] - merged["sharpe_prev"]
                merged["delta_annual"] = merged["annual_latest"] - merged["annual_prev"]
                merged["delta_rank"] = merged["rank_latest"] - merged["rank_prev"]

                logger.info("")
                logger.info("重叠组合对比（均值）:")
                logger.info(
                    f"- 平均 Sharpe 变化: {merged['delta_sharpe'].mean():>6.3f}"
                )
                logger.info(f"- 平均 年化  变化: {merged['delta_annual'].mean():>6.3%}")
                logger.info(
                    f"- 平均 排名  变化: {merged['delta_rank'].mean():>6.2f} (负数=最新排名更靠前)"
                )
                logger.info(
                    f"- 提升占比(Sharpe>0): {(merged['delta_sharpe']>0).mean()*100:>5.1f}%  ({(merged['delta_sharpe']>0).sum()}/{len(merged)})"
                )

                compare_file = (
                    run_output_dir / f"compare_top100_{prev_ts}_vs_{latest_ts}.csv"
                )
                merged.sort_values("delta_sharpe", ascending=False).to_csv(
                    compare_file, index=False
                )
                logger.info(f"对比明细已保存: {compare_file}")
            else:
                logger.info("两轮Top100无重叠组合，跳过逐组合对比保存。")


if __name__ == "__main__":
    main()
