"""
生产回测主脚本（无未来函数保障 + 可选性能/诊断开关）
================================================================================
严格时间隔离：每个调仓日只使用截至前一日的历史数据。

核心原则
---------
1) 因子计算：逐日计算，不提前计算全部时间序列。
2) 权重计算：每个调仓日用历史窗口计算 IC 权重（可走“日级IC预计算”路径）。
3) 信号计算：每个调仓日用当日因子值计算信号。
4) 选股决策：基于当日信号，不知道未来信号。

关键环境变量（只列最常用）
--------------------------
- RB_DAILY_IC_PRECOMP=1    启用“日级IC预计算 + 前缀和 O(1) 滑窗”。
- RB_DAILY_IC_MEMMAP=1     通过 np.memmap 在多进程间共享日级IC矩阵。
- RB_STABLE_RANK=1         Spearman 使用“平均 ties”的稳定排名（更鲁棒）。
- RB_PRELOAD_IC=1          预装常用 (freq×factor) 配对以提高缓存命中。
- RB_NUMBA_WARMUP=1        进程启动时对关键 numba 路径做一次预热。
- RB_ENFORCE_NO_LOOKAHEAD  开启抽样重算做自检（与稳定排名路径存在微小数值差异）。
- RB_NL_CHECK_TOL          自检权重差异容差（稳定排名建议 1e-2）。
- RB_OUTLIER_REPORT        打印组合级耗时 outlier 诊断（仅诊断期开）。
- RB_PROFILE_BACKTEST      输出分阶段耗时统计（mean/median/p95/p99）。

说明
----
稳定排名路径（RB_STABLE_RANK=1）在日级IC预计算与旧的“窗口内即时重算”之间可能存在轻微数值差异，
属于并列秩处理方式不同导致的可解释偏差。严格审计时请适当放宽 RB_NL_CHECK_TOL（如 1e-2）。
"""

import logging
import os
import hashlib
from multiprocessing import Manager
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from numba import njit, prange

# --- ensure package import works when launched from repo root or any cwd ---
_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent  # etf_rotation_optimized
for p in (_HERE, _PKG_ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.append(sp)

from core.cross_section_processor import CrossSectionProcessor
from core.data_loader import DataLoader
from core.ic_calculator_numba import compute_spearman_ic_numba
from core.precise_factor_library_v2 import PreciseFactorLibrary

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# 全局滚动IC权重缓存：避免跨组合重复计算
# key = (rebalance_key: bytes, lookback: int) -> {
#   "ic": np.ndarray (n_rebalance, F_total),
#   "filled": np.ndarray (F_total,) bool
# }
IC_CACHE = {}
# 预计算的“每日IC矩阵”(T, F_total) 及其索引缓存
# key 采用底层内存地址与shape构成，尽可能避免误用（进程内有效）
PRECOMP_DAILY_IC = {}
NUMBA_WARMED_UP = False


def _arr_mem_key(arr: np.ndarray) -> tuple:
    try:
        # 使用底层data指针地址+shape+dtype作为键，以减少误判
        return (int(arr.__array_interface__["data"][0]), arr.shape, str(arr.dtype))
    except Exception:
        # 回退：仅用id和shape
        return (id(arr), arr.shape, str(arr.dtype))


@njit(cache=True)
def _spearman_single_day_simple(x: np.ndarray, y: np.ndarray) -> float:
    """简单秩实现（当前默认），并列使用次序秩，不做平均。"""
    mask = ~(np.isnan(x) | np.isnan(y))
    n_valid = np.sum(mask)
    if n_valid <= 2:
        return np.nan
    xv = x[mask]
    yv = y[mask]
    xr = np.argsort(np.argsort(xv)).astype(np.float64)
    yr = np.argsort(np.argsort(yv)).astype(np.float64)
    xm = np.mean(xr)
    ym = np.mean(yr)
    num = np.sum((xr - xm) * (yr - ym))
    xs = np.sqrt(np.sum((xr - xm) ** 2))
    ys = np.sqrt(np.sum((yr - ym) ** 2))
    if xs > 0 and ys > 0:
        return num / (xs * ys)
    return np.nan


@njit(cache=True)
def _average_ranks(values: np.ndarray) -> np.ndarray:
    """平均并列秩（stable rank）。
    算法: 排序后顺序扫描，将相等值区间赋予 (start+end)/2 平均秩。
    返回: float64 ranks 数组。
    """
    n = values.shape[0]
    order = np.argsort(values)
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        v = values[order[i]]
        while j < n and values[order[j]] == v:
            j += 1
        # 平均秩: (i + j - 1)/2
        avg = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return ranks

def _average_ranks_py(values: np.ndarray) -> np.ndarray:
    """Python fallback 使用 scipy.stats.rankdata(method='average').
    仅在需要与外部库对齐或调试时使用，避免在 numba 下引入额外依赖。
    """
    try:
        from scipy.stats import rankdata
        return rankdata(values, method='average').astype(np.float64) - 1.0  # 转为0基
    except Exception:
        return _average_ranks(values)


@njit(cache=True)
def _spearman_single_day_stable(x: np.ndarray, y: np.ndarray) -> float:
    """稳定秩 Spearman：并列取平均秩。"""
    mask = ~(np.isnan(x) | np.isnan(y))
    n_valid = np.sum(mask)
    if n_valid <= 2:
        return np.nan
    xv = x[mask]
    yv = y[mask]
    xr = _average_ranks(xv)
    yr = _average_ranks(yv)
    xm = np.mean(xr)
    ym = np.mean(yr)
    num = np.sum((xr - xm) * (yr - ym))
    xs = np.sqrt(np.sum((xr - xm) ** 2))
    ys = np.sqrt(np.sum((yr - ym) ** 2))
    if xs > 0 and ys > 0:
        return num / (xs * ys)
    return np.nan


@njit(cache=True)
def _window_ic_for_factor_stable(factors_hist_2d: np.ndarray, returns_hist_2d: np.ndarray) -> float:
    """在历史窗口内，按日计算稳定秩 Spearman 再取均值。"""
    T_hist = factors_hist_2d.shape[0]
    s = 0.0
    c = 0
    for t in range(T_hist):
        ic = _spearman_single_day_stable(factors_hist_2d[t], returns_hist_2d[t])
        if not np.isnan(ic):
            s += ic
            c += 1
    return s / c if c > 0 else 0.0


@njit(cache=True)
def _compute_ic_for_all_factors_stable(factors_hist: np.ndarray, returns_hist: np.ndarray) -> np.ndarray:
    """计算窗口内所有因子的稳定秩IC（按日Spearman均值）。"""
    F_sel = factors_hist.shape[2]
    ics = np.zeros(F_sel, dtype=np.float64)
    for f in range(F_sel):
        ics[f] = _window_ic_for_factor_stable(factors_hist[:, :, f], returns_hist)
    return ics


@njit(parallel=True, cache=True)
def _compute_daily_ic_all_factors_simple(factors_data_full: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """每日IC矩阵（简单秩）。"""
    T, N, F_total = factors_data_full.shape
    out = np.empty((T, F_total), dtype=np.float64)
    for f in prange(F_total):
        for t in range(T):
            out[t, f] = _spearman_single_day_simple(factors_data_full[t, :, f], returns[t])
    return out


@njit(parallel=True, cache=True)
def _compute_daily_ic_all_factors_stable(factors_data_full: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """每日IC矩阵（稳定平均并列秩）。"""
    T, N, F_total = factors_data_full.shape
    out = np.empty((T, F_total), dtype=np.float64)
    for f in prange(F_total):
        for t in range(T):
            out[t, f] = _spearman_single_day_stable(factors_data_full[t, :, f], returns[t])
    return out


def _numba_warmup():
    """一次性小样本调用以触发numba编译，避免首批任务抖动。"""
    global NUMBA_WARMED_UP
    if NUMBA_WARMED_UP:
        return
    try:
        T, N, F = 16, 8, 4
        factors = np.random.rand(T, N, F).astype(np.float64)
        rets = np.random.randn(T, N).astype(np.float64) * 0.001
        _compute_daily_ic_all_factors_simple(factors, rets)
        _compute_daily_ic_all_factors_stable(factors, rets)
        # 构造一个 (T,N) 信号用于 compute_spearman_ic_numba
        from core.ic_calculator_numba import compute_spearman_ic_numba as _csi
        sig = factors[:, :, 0]
        _ = _csi(sig, rets)
        NUMBA_WARMED_UP = True
    except Exception:
        # 静默失败，不影响主流程
        NUMBA_WARMED_UP = True


def _compute_or_load_daily_ic_memmap(factors_data_full: np.ndarray, returns: np.ndarray, stable_rank: bool) -> np.ndarray:
    """
    基于 memmap 的跨进程共享每日 IC 矩阵。
    环境变量:
        RB_DAILY_IC_MEMMAP_DIR: 目录（默认 .cache）
        RB_DAILY_IC_MEMMAP_FP32: =1 时存 float32 (节省 IO/内存)
        RB_DAILY_IC_MEMMAP_KEY: 自定义文件 key（避免同形不同内容冲突）
    文件命名: daily_ic_{T}_{N}_{F}_{key}_{dtype}.mmap
    """
    memmap_dir = os.environ.get("RB_DAILY_IC_MEMMAP_DIR", ".cache").strip()
    os.makedirs(memmap_dir, exist_ok=True)
    T, N, F_total = factors_data_full.shape
    use_fp32 = os.environ.get("RB_DAILY_IC_MEMMAP_FP32", "0").strip().lower() in ("1", "true", "yes")
    custom_key = os.environ.get("RB_DAILY_IC_MEMMAP_KEY", "").strip()
    if not custom_key:
        # 模式敏感 key：加入排名模式 + 算法版本，避免 stable/simple 交叉污染
        algo_version = "v2stable" if stable_rank else "v1simple"
        try:
            sample = returns.ravel()[:256]
            h = hashlib.sha1(sample.tobytes()).hexdigest()[:12]
        except Exception:
            h = "nohash"
        custom_key = f"auto_{algo_version}_{T}_{N}_{F_total}_{h}"
    dtype_str = "fp32" if use_fp32 else "fp64"
    file_name = f"daily_ic_{custom_key}_{dtype_str}.mmap"
    path = os.path.join(memmap_dir, file_name)

    if os.path.exists(path):
        mm = np.memmap(path, dtype=np.float32 if use_fp32 else np.float64, mode="r", shape=(T, F_total))
        logger.info(f"[daily_ic_memmap] reuse {path} mode={'stable' if stable_rank else 'simple'}")
        return np.asarray(mm, dtype=np.float64)

    # 简单文件锁避免并发竞争
    lock_path = path + ".lock"
    got_lock = False
    for _ in range(50):  # 最多等待 ~5s
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            got_lock = True
            break
        except FileExistsError:
            time.sleep(0.1)
    if not got_lock and os.path.exists(path):
        mm = np.memmap(path, dtype=np.float32 if use_fp32 else np.float64, mode="r", shape=(T, F_total))
        logger.info(f"[daily_ic_memmap] locked – fallback reuse {path} mode={'stable' if stable_rank else 'simple'}")
        return np.asarray(mm, dtype=np.float64)
    if not got_lock:
        # 仍未获取锁：退化为内存计算（不写盘，保证隔离）
        logger.info("[daily_ic_memmap] lock wait exhausted, compute in-memory only")
        daily_ic_mat = _compute_daily_ic_all_factors_stable(factors_data_full, returns) if stable_rank else _compute_daily_ic_all_factors_simple(factors_data_full, returns)
        return daily_ic_mat

    try:
        daily_ic_mat = _compute_daily_ic_all_factors_stable(factors_data_full, returns) if stable_rank else _compute_daily_ic_all_factors_simple(factors_data_full, returns)
        arr_to_store = daily_ic_mat.astype(np.float32 if use_fp32 else np.float64)
        mm = np.memmap(path, dtype=arr_to_store.dtype, mode="w+", shape=(T, F_total))
        mm[:] = arr_to_store[:]
        del mm  # 关闭文件句柄
        logger.info(f"[daily_ic_memmap] built {path} mode={'stable' if stable_rank else 'simple'}")
    finally:
        try:
            os.remove(lock_path)
        except Exception:
            pass
    return daily_ic_mat


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


@njit(cache=True)
def _compute_ic_column_for_factor(factors_data_full, returns, rebalance_indices, lookback_window, factor_idx):
    """
    计算某个因子在所有调仓日的滚动IC序列（无未来泄露）。

    返回: (n_rebalance,) 对应每个调仓日的IC
    """
    n_rebalance = len(rebalance_indices)
    ics = np.empty(n_rebalance, dtype=np.float64)
    for i in range(n_rebalance):
        day_idx = rebalance_indices[i]
        hist_start = 0 if day_idx - lookback_window < 0 else day_idx - lookback_window
        hist_end = day_idx
        ics[i] = compute_spearman_ic_numba(
            factors_data_full[hist_start:hist_end, :, factor_idx],
            returns[hist_start:hist_end],
        )
    return ics

@njit(parallel=True, cache=True)
def _compute_ic_columns_for_factors(
    factors_data_full,
    returns,
    rebalance_indices,
    lookback_window,
    factor_indices,
):
    """
    批量计算多个因子的滚动IC列，返回形状 (n_rebalance, K) 的矩阵，其中 K=len(factor_indices)。
    外层对因子并行，内层对调仓日顺序循环。
    """
    n_rebalance = len(rebalance_indices)
    K = len(factor_indices)
    out = np.empty((n_rebalance, K), dtype=np.float64)

    for k in prange(K):  # 并行遍历因子
        fi = factor_indices[k]
        for i in range(n_rebalance):
            day_idx = rebalance_indices[i]
            hist_start = 0 if day_idx - lookback_window < 0 else day_idx - lookback_window
            hist_end = day_idx
            out[i, k] = compute_spearman_ic_numba(
                factors_data_full[hist_start:hist_end, :, fi],
                returns[hist_start:hist_end],
            )
    return out

def get_ic_weights_matrix_cached(
    factors_data_full: np.ndarray,
    returns: np.ndarray,
    rebalance_indices: np.ndarray,
    lookback_window: int,
    factor_indices: np.ndarray,
):
    """
    基于全局缓存返回某组合所需的IC权重矩阵（n_rebalance, F_selected）。
    仅在第一次请求某个(调仓日, 回看窗口, 因子)时计算并缓存对应列，后续复用。
    """
    F_total = factors_data_full.shape[2]
    n_reb = len(rebalance_indices)
    # 以rebalance_indices和lookback窗口构造缓存key
    rebalance_key = rebalance_indices.tobytes()
    key = (rebalance_key, int(lookback_window))

    cache_entry = IC_CACHE.get(key)
    if cache_entry is None:
        ic_mat = np.empty((n_reb, F_total), dtype=np.float64)
        ic_mat[:] = np.nan
        filled = np.zeros(F_total, dtype=bool)
        IC_CACHE[key] = {"ic": ic_mat, "filled": filled}
        cache_entry = IC_CACHE[key]

    ic_mat = cache_entry["ic"]
    filled = cache_entry["filled"]

    # ============ 新增：基于“每日IC矩阵”的快速滚动平均路径 ============
    enable_daily_ic = (
        os.environ.get("RB_DAILY_IC_PRECOMP", "0").strip().lower() in ("1", "true", "yes")
    )
    enable_memmap = (
        os.environ.get("RB_DAILY_IC_MEMMAP", "0").strip().lower() in ("1", "true", "yes")
    )
    enable_warmup = (
        os.environ.get("RB_NUMBA_WARMUP", "1").strip().lower() in ("1", "true", "yes")
    )
    daily_ic = None
    stable_rank_enabled = os.environ.get("RB_STABLE_RANK", "0").strip().lower() in ("1", "true", "yes")
    if enable_daily_ic:
        if enable_warmup:
            _numba_warmup()
        # 生成 / 复用全局 daily ic 矩阵
        if enable_memmap:
            daily_ic = _compute_or_load_daily_ic_memmap(factors_data_full, returns, stable_rank_enabled)
        else:
            factors_key = _arr_mem_key(factors_data_full)
            returns_key = _arr_mem_key(returns)
            main_key = (factors_key, returns_key)
            daily_entry = PRECOMP_DAILY_IC.get(main_key)
            if daily_entry is None:
                # 计算 (T, F_total) 跨截面 spearman (当日)
                daily_ic_mat = _compute_daily_ic_all_factors_stable(factors_data_full, returns) if stable_rank_enabled else _compute_daily_ic_all_factors_simple(factors_data_full, returns)
                PRECOMP_DAILY_IC[main_key] = {"daily_ic": daily_ic_mat}
                daily_entry = PRECOMP_DAILY_IC[main_key]
            daily_ic = daily_entry["daily_ic"]  # (T, F_total)

    # 需要补齐的列
    need_list = [int(fi) for fi in factor_indices if not filled[int(fi)]]
    if len(need_list) > 0:
        need_arr = np.asarray(need_list, dtype=np.int64)

        if enable_daily_ic and daily_ic is not None:
            # 使用每日 ic 的滚动平均：ic_window_mean = mean(daily_ic[hist_start:hist_end, fi])
            # 预先构造 cumulative sum 以 O(1) 获取窗口均值
            # 注意：rebalance_indices 中 day_idx 不包含当日 -> hist_end = day_idx
            sel_daily = daily_ic[:, need_arr]  # (T, K)
            # 将 NaN 视为缺失，使用计数矩阵
            valid_mask = ~np.isnan(sel_daily)
            daily_filled = np.where(valid_mask, sel_daily, 0.0)
            cumsum = np.vstack([np.zeros((1, daily_filled.shape[1])), np.cumsum(daily_filled, axis=0)])  # (T+1,K)
            cnt = np.vstack([np.zeros((1, daily_filled.shape[1])), np.cumsum(valid_mask.astype(np.int64), axis=0)])
            for i, day_idx in enumerate(rebalance_indices):
                hist_start = 0 if day_idx - lookback_window < 0 else day_idx - lookback_window
                hist_end = day_idx
                win_sum = cumsum[hist_end, :] - cumsum[hist_start, :]
                win_cnt = cnt[hist_end, :] - cnt[hist_start, :]
                win_mean = np.zeros_like(win_sum)
                nz = win_cnt > 0
                win_mean[nz] = win_sum[nz] / win_cnt[nz]
                ic_mat[i, need_arr] = win_mean
            for fi in need_arr:
                filled[int(fi)] = True
        else:
            # 回退：原有单列/批量 numba 计算路径
            if len(need_arr) == 1:
                fi = int(need_arr[0])
                ic_col = _compute_ic_column_for_factor(
                    factors_data_full, returns, rebalance_indices, lookback_window, fi
                )
                ic_mat[:, fi] = ic_col
                filled[fi] = True
            else:
                batch = _compute_ic_columns_for_factors(
                    factors_data_full, returns, rebalance_indices, lookback_window, need_arr
                )  # (n_rebalance, K)
                for idx_k in range(len(need_arr)):
                    fi = int(need_arr[idx_k])
                    ic_mat[:, fi] = batch[:, idx_k]
                    filled[fi] = True

    # 构造并返回所需因子权重矩阵 (n_rebalance, len(factor_indices))
    sel_ic = ic_mat[:, factor_indices]  # 可能含NaN
    abs_ic = np.abs(sel_ic)
    weights = np.zeros_like(sel_ic)
    row_sums = np.nansum(abs_ic, axis=1)
    F_sel = len(factor_indices)
    for i in range(len(row_sums)):
        s = row_sums[i]
        if s > 0:
            w = abs_ic[i] / s
        else:
            w = np.full(F_sel, 1.0 / F_sel)
        weights[i] = w
    return weights


def backtest_no_lookahead(
    factors_data,
    returns,
    etf_names,
    rebalance_freq,
    lookback_window=252,
    position_size=4,
    initial_capital=1_000_000.0,
    commission_rate=0.00005,
    commission_min=0.0,
    *,
    factors_data_full=None,
    factor_indices_for_cache=None,
):
    """
    ⚠️ 严格无未来函数的回测 (优化版)

    参数:
        factors_data: (T, N, F) 全部因子数据
        returns: (T, N) 全部收盘到收盘收益 (定义: close[t]/close[t-1]-1)
        etf_names: list, ETF名称
        rebalance_freq: int, 调仓频率(天)
        lookback_window: int, 计算权重的回看窗口
        position_size: int, 持仓数量（默认按Top N）
        initial_capital: float, 初始资金
    commission_rate: float, 佣金率（双边，买入和卖出都收取，ETF默认例0.5）
    commission_min: float, 佣金最低费用（绝对金额，默认0表示不启用）

    返回:
        dict: 回测结果
    """
    profile_enabled = os.environ.get("RB_PROFILE_BACKTEST", "0").strip().lower() in ("1", "true", "yes")
    profile_log = logger.info if profile_enabled else (lambda *args, **kwargs: None)
    profile_data = {} if profile_enabled else None

    enforce_nl = os.environ.get("RB_ENFORCE_NO_LOOKAHEAD", "0").strip().lower() in ("1", "true", "yes")
    nl_check_max = int(os.environ.get("RB_NL_CHECK_MAX", "5") or 5)
    try:
        nl_tol = float(os.environ.get("RB_NL_CHECK_TOL", "1e-9") or 1e-9)
    except Exception:
        nl_tol = 1e-9
    try:
        nl_rtol = float(os.environ.get("RB_NL_CHECK_RTOL", "0"))
    except Exception:
        nl_rtol = 0.0
    try:
        nl_atol = float(os.environ.get("RB_NL_CHECK_ATOL", str(nl_tol)))
    except Exception:
        nl_atol = nl_tol
    nl_checks_done = 0

    total_timer_start = time.perf_counter() if profile_enabled else None

    # 优先级2：确保内存布局连续，避免不必要的拷贝与 cache miss（不改变 dtype/数值）
    factors_data = np.ascontiguousarray(factors_data)
    returns = np.ascontiguousarray(returns)
    if factors_data_full is not None:
        factors_data_full = np.ascontiguousarray(factors_data_full)

    # 优先级3：默认启用日级IC预计算+memmap（仅当可用，且不覆盖用户显式设置）
    os.environ.setdefault("RB_DAILY_IC_PRECOMP", "1")
    os.environ.setdefault("RB_DAILY_IC_MEMMAP", "1")
    os.environ.setdefault("RB_NUMBA_WARMUP", "1")

    T, N, F = factors_data.shape

    start_idx = lookback_window + 1  # +1 因 returns 第1天不可用

    rebalance_indices = np.arange(start_idx, T, rebalance_freq, dtype=np.int32)
    n_rebalance = len(rebalance_indices)

    profile_log(
        f"  回测参数: {rebalance_freq}天换仓, Top{position_size}持仓, 回看{lookback_window}天"
    )
    profile_log(f"  起始日: 第{start_idx}天, 调仓次数: {n_rebalance}次")

    profile_log("  预计算IC权重...")
    ic_timer_start = time.perf_counter() if profile_enabled else None
    disable_cache = os.environ.get("RB_DISABLE_IC_CACHE", "0").strip().lower() in ("1", "true", "yes")

    # 若未提供全量数组/因子索引，退化为使用当前输入（不改变语义，仅使预计算路径可用）
    if factors_data_full is None:
        factors_data_full = factors_data
    if factor_indices_for_cache is None:
        factor_indices_for_cache = np.arange(F, dtype=np.int64)

    if not disable_cache:
        ic_weights_matrix = get_ic_weights_matrix_cached(
            factors_data_full=factors_data_full,
            returns=returns,
            rebalance_indices=rebalance_indices,
            lookback_window=lookback_window,
            factor_indices=np.asarray(factor_indices_for_cache, dtype=np.int64),
        )
        ic_path_type = (
            "daily_stable"
            if (
                os.environ.get("RB_DAILY_IC_PRECOMP", "0").strip().lower() in ("1", "true", "yes")
                and os.environ.get("RB_STABLE_RANK", "0").strip().lower() in ("1", "true", "yes")
            )
            else (
                "daily_simple"
                if (os.environ.get("RB_DAILY_IC_PRECOMP", "0").strip().lower() in ("1", "true", "yes"))
                else "cached_batch"
            )
        )
    else:
        F_sel = factors_data.shape[2]
        tmp_ic = np.zeros((n_rebalance, F_sel), dtype=np.float64)
        for i in range(n_rebalance):
            day_idx = rebalance_indices[i]
            hist_start = max(0, day_idx - lookback_window)
            hist_end = day_idx
            factors_hist = factors_data[hist_start:hist_end]
            returns_hist = returns[hist_start:hist_end]
            ics = np.zeros(F_sel)
            for f in range(F_sel):
                ics[f] = compute_spearman_ic_numba(factors_hist[:, :, f], returns_hist)
            abs_ics = np.abs(ics)
            tmp_ic[i] = abs_ics / abs_ics.sum() if abs_ics.sum() > 0 else np.ones(F_sel) / F_sel
        ic_weights_matrix = tmp_ic
        ic_path_type = "fallback_simple"

    if profile_enabled:
        profile_data["time_precompute_ic"] = time.perf_counter() - ic_timer_start if ic_timer_start is not None else 0.0
        profile_data["n_rebalance"] = int(n_rebalance)
        profile_data["n_days"] = int(T - start_idx)

    n_days = T - start_idx
    portfolio_values = np.zeros(n_days + 1)
    portfolio_values[0] = initial_capital
    daily_returns_arr = np.zeros(n_days)

    # 预分配：按“调仓事件”次数，而非按日
    turnover_arr = np.empty(n_rebalance, dtype=float) if n_rebalance > 0 else np.empty(0, dtype=float)
    cost_rate_arr = np.empty(n_rebalance, dtype=float) if n_rebalance > 0 else np.empty(0, dtype=float)
    cost_amount_arr = np.empty(n_rebalance, dtype=float) if n_rebalance > 0 else np.empty(0, dtype=float)
    n_holdings_arr = np.empty(n_rebalance, dtype=np.int32) if n_rebalance > 0 else np.empty(0, dtype=np.int32)

    current_weights = np.zeros(N)
    rebalance_counter = 0

    loop_timer_start = time.perf_counter() if profile_enabled else None

    for offset, day_idx in enumerate(range(start_idx, T)):
        is_rebalance_day = (
            rebalance_counter < n_rebalance and day_idx == rebalance_indices[rebalance_counter]
        )

        if is_rebalance_day:
            factor_weights = ic_weights_matrix[rebalance_counter]
            if enforce_nl and nl_checks_done < nl_check_max:
                try:
                    stride = max(1, n_rebalance // max(1, nl_check_max))
                    if (rebalance_counter % stride) == 0:
                        hist_start = max(0, day_idx - lookback_window)
                        hist_end = day_idx
                        F_sel = factors_data.shape[2]
                        ics = np.zeros(F_sel, dtype=np.float64)
                        for f in range(F_sel):
                            ics[f] = compute_spearman_ic_numba(
                                factors_data[hist_start:hist_end, :, f],
                                returns[hist_start:hist_end],
                            )
                        abs_ics = np.abs(ics)
                        w_chk = abs_ics / abs_ics.sum() if abs_ics.sum() > 0 else np.full(F_sel, 1.0 / F_sel)
                        stable_rank_enabled = os.environ.get("RB_STABLE_RANK", "0").strip().lower() in ("1", "true", "yes")
                        daily_precomp_enabled = os.environ.get("RB_DAILY_IC_PRECOMP", "0").strip().lower() in ("1", "true", "yes")
                        can_use_daily = (
                            stable_rank_enabled and daily_precomp_enabled and (factors_data_full is not None) and (factor_indices_for_cache is not None)
                        )
                        if can_use_daily:
                            try:
                                daily_ic_full = _compute_or_load_daily_ic_memmap(factors_data_full, returns, stable_rank=True)
                                cols = np.asarray(factor_indices_for_cache, dtype=np.int64)
                                di_slice = daily_ic_full[hist_start:hist_end][:, cols]
                                window_mean = np.nanmean(di_slice, axis=0)
                                abs_ics_local = np.abs(window_mean)
                                if np.isfinite(abs_ics_local).any() and np.nansum(abs_ics_local) > 0:
                                    w_chk = abs_ics_local / np.nansum(abs_ics_local)
                                else:
                                    w_chk = np.full(F_sel, 1.0 / F_sel)
                            except Exception:
                                ics = np.zeros(F_sel, dtype=np.float64)
                                for f in range(F_sel):
                                    ics[f] = compute_spearman_ic_numba(
                                        factors_data[hist_start:hist_end, :, f],
                                        returns[hist_start:hist_end],
                                    )
                                abs_ics = np.abs(ics)
                                w_chk = (abs_ics / abs_ics.sum()) if abs_ics.sum() > 0 else np.full(F_sel, 1.0 / F_sel)
                        else:
                            ics = np.zeros(F_sel, dtype=np.float64)
                            for f in range(F_sel):
                                ics[f] = compute_spearman_ic_numba(
                                    factors_data[hist_start:hist_end, :, f],
                                    returns[hist_start:hist_end],
                                )
                            abs_ics = np.abs(ics)
                            w_chk = (abs_ics / abs_ics.sum()) if abs_ics.sum() > 0 else np.full(F_sel, 1.0 / F_sel)
                        if not np.allclose(w_chk, factor_weights, rtol=nl_rtol, atol=nl_atol, equal_nan=True):
                            diff = np.nanmax(np.abs(w_chk - factor_weights))
                            raise RuntimeError(
                                f"NO_LOOKAHEAD_CHECK_FAILED: day_idx={day_idx}, max_weight_diff={diff:.3e} (rtol={nl_rtol}, atol={nl_atol})"
                            )
                        nl_checks_done += 1
                except Exception:
                    raise

            # 记录当前调仓事件索引（自增前）
            idx_rb = rebalance_counter
            rebalance_counter += 1

            prev_weights = current_weights.copy()

            factors_yesterday = factors_data[day_idx - 1]
            signal_yesterday = compute_signal_single_day(
                factors_yesterday, factor_weights
            )

            valid_mask = ~np.isnan(signal_yesterday)

            if np.sum(valid_mask) < position_size:
                target_weights = np.zeros(N)
                n_holdings_arr[idx_rb] = 0
            else:
                sig_valid = signal_yesterday.copy()
                sig_valid[~valid_mask] = -np.inf
                kth_val = np.partition(sig_valid, -position_size)[-position_size]
                candidates = np.where(sig_valid >= kth_val)[0]
                if len(candidates) > position_size:
                    order = np.lexsort((candidates, -sig_valid[candidates]))
                    chosen = candidates[order][:position_size]
                else:
                    chosen = candidates[:position_size]
                top_indices = chosen
                target_weights = np.zeros(N)
                target_weights[top_indices] = 1.0 / position_size
                n_holdings_arr[idx_rb] = len(top_indices)

            # 换手与成本（按调仓事件）
            delta_weights = target_weights - prev_weights
            buy_turnover = float(np.sum(delta_weights[delta_weights > 0]))
            sell_turnover = float(np.sum(-delta_weights[delta_weights < 0]))
            turnover = buy_turnover + sell_turnover
            turnover_arr[idx_rb] = turnover

            portfolio_before_cost = portfolio_values[offset]
            trade_notional = (buy_turnover + sell_turnover) * portfolio_before_cost
            commission_value = trade_notional * commission_rate
            if commission_min > 0 and turnover > 1e-12:
                commission_value = max(commission_value, commission_min)
            total_cost_amount = commission_value

            if portfolio_before_cost > 1e-12 and total_cost_amount > 0:
                total_cost_amount = min(total_cost_amount, portfolio_before_cost)
                cost_rate = total_cost_amount / portfolio_before_cost
                portfolio_values[offset] = portfolio_before_cost - total_cost_amount
            else:
                cost_rate = 0.0
                total_cost_amount = 0.0

            cost_rate_arr[idx_rb] = cost_rate
            cost_amount_arr[idx_rb] = total_cost_amount

            current_weights = target_weights

        # === 每日收益计算 ===
        # 显式使用收盘到收盘定义 (close[t]/close[t-1]-1)
        close_to_close_ret = returns[day_idx]  # 等价于 (close[t]/close[t-1]-1)
        daily_ret = np.nansum(current_weights * close_to_close_ret)
        daily_returns_arr[offset] = daily_ret

        portfolio_values[offset + 1] = portfolio_values[offset] * (1 + daily_ret)

    if profile_enabled and loop_timer_start is not None:
        profile_data["time_main_loop"] = time.perf_counter() - loop_timer_start

    final = portfolio_values[-1]
    total_ret = final / initial_capital - 1

    days_elapsed = len(daily_returns_arr)
    annual_ret = (1 + total_ret) ** (252 / days_elapsed) - 1

    vol = np.std(daily_returns_arr) * np.sqrt(252)
    sharpe = annual_ret / vol if vol > 0 else 0

    cummax = np.maximum.accumulate(portfolio_values)
    dd = (portfolio_values - cummax) / cummax
    max_dd = np.min(dd)

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

    profit_factor = 0.0
    if losing_days > 0 and abs(np.sum(negative_returns)) > 1e-10:
        profit_factor = float(np.sum(positive_returns) / abs(np.sum(negative_returns)))

    calmar_ratio = annual_ret / abs(max_dd) if abs(max_dd) > 1e-10 else 0.0

    downside_returns = daily_returns_arr[daily_returns_arr < 0]
    downside_vol = (
        np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252)
        if len(downside_returns) > 0
        else 1e-6
    )
    sortino_ratio = annual_ret / downside_vol if downside_vol > 1e-10 else 0.0

    if len(daily_returns_arr) > 0:
        max_consecutive_wins, max_consecutive_losses = calculate_streaks_vectorized(
            daily_returns_arr
        )
    else:
        max_consecutive_wins = 0
        max_consecutive_losses = 0

    avg_n_holdings = float(np.mean(n_holdings_arr)) if n_rebalance > 0 else 0.0

    result = {
        "freq": rebalance_freq,
        "final": final,
        "total_ret": total_ret,
        "annual_ret": annual_ret,
        "vol": vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "n_rebalance": n_rebalance,
        "avg_turnover": float(np.mean(turnover_arr)) if n_rebalance > 0 else 0.0,
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
        # 持仓数统计（按调仓事件）
        "avg_n_holdings": avg_n_holdings,
        # 详细数据
        "nav": portfolio_values,
        "daily_returns": daily_returns_arr,
        "turnover_series": turnover_arr,
        "cost_rate_series": cost_rate_arr,
        "cost_amount_series": cost_amount_arr,
    }

    if profile_enabled and total_timer_start is not None:
        profile_data["time_total"] = time.perf_counter() - total_timer_start
        profile_data["rebalance_executed"] = int(rebalance_counter)
        profile_data["loop_iterations"] = int(n_days)
        profile_data["avg_turnover"] = float(result["avg_turnover"])
        profile_data["ic_path"] = ic_path_type
        profile_data["stable_rank"] = os.environ.get("RB_STABLE_RANK", "0").strip().lower() in ("1", "true", "yes")
        result["profile"] = profile_data

    return result


def calculate_streaks_vectorized(daily_returns_arr: np.ndarray):
    """向量化计算最长连续盈利与亏损天数 (0 收益视为中断)。

    Args:
        daily_returns_arr: (T,) 日收益序列
    Returns:
        (max_consecutive_wins, max_consecutive_losses)
    """
    if daily_returns_arr.size == 0:
        return 0, 0
    signs = np.sign(daily_returns_arr).astype(np.int32)
    # 正数->1, 负数->-1, 0 保持0作为分隔
    signs[signs > 0] = 1
    signs[signs < 0] = -1
    max_win = 0
    max_loss = 0
    cur_len = 0
    cur_sign = 0
    for s in signs:
        if s == 0 or s != cur_sign:
            # 结算上一个序列
            if cur_sign == 1:
                if cur_len > max_win:
                    max_win = cur_len
            elif cur_sign == -1:
                if cur_len > max_loss:
                    max_loss = cur_len
            # 重置
            cur_sign = s
            cur_len = 1 if s != 0 else 0
        else:
            cur_len += 1
    # 结算最后一个序列
    if cur_sign == 1:
        if cur_len > max_win:
            max_win = cur_len
    elif cur_sign == -1:
        if cur_len > max_loss:
            max_loss = cur_len
    return max_win, max_loss


def load_top_combos_from_run(run_dir: Path, top_n: int = 100, load_all: bool = False):
    """
    加载某个 run_ 目录下的组合列表。
    
    Args:
        run_dir: WFO run目录
        top_n: 加载TopN组合（当load_all=False时生效）
        load_all: 是否加载全量组合（忽略top_n限制）
    
    优先级：
    1. 若load_all=True，直接加载all_combos.parquet全量数据
    2. 否则，优先读取 top100_by_ic.parquet；
    3. 若不存在，则读取 top_combos.parquet；
    4. 若仍不存在，退化为 all_combos.parquet 并按 IC/稳定性排序取 TopN。

    返回:
        (df, sort_method_str)
    """
    top_by_ic_file = run_dir / "top100_by_ic.parquet"
    top_combos_file = run_dir / "top_combos.parquet"
    all_combos_file = run_dir / "all_combos.parquet"

    # 🔥 新增：支持加载全量组合（用于完整样本训练）
    # 若存在校准分或预测列则优先使用校准排序；否则按IC/稳定性
    def _sort_df(df: pd.DataFrame) -> pd.DataFrame:
        if "calibrated_sharpe_pred" in df.columns:
            return df.sort_values(
                by=["calibrated_sharpe_pred", "stability_score"], ascending=[False, False]
            )
        if "calibrated_sharpe_full" in df.columns:
            return df.sort_values(
                by=["calibrated_sharpe_full", "stability_score"], ascending=[False, False]
            )
        return df.sort_values(
            by=["mean_oos_ic", "stability_score"], ascending=[False, False]
        )

    if load_all:
        if all_combos_file.exists():
            df = pd.read_parquet(all_combos_file)
            df = _sort_df(df)
            sort_label = "ALL calibrated" if ("calibrated_sharpe_pred" in df.columns or "calibrated_sharpe_full" in df.columns) else "ALL IC"
            return df.reset_index(drop=True), f"ALL ({len(df)} combos from all_combos, {sort_label})"
        else:
            raise FileNotFoundError(f"全量回测模式需要 all_combos.parquet，但未找到: {all_combos_file}")

    if top_by_ic_file.exists():
        df = pd.read_parquet(top_by_ic_file).reset_index(drop=True)
        df = _sort_df(df)
        if len(df) >= top_n:
            lbl = "calibrated (top100_by_ic)" if ("calibrated_sharpe_pred" in df.columns or "calibrated_sharpe_full" in df.columns) else "IC (top100_by_ic)"
            return df.head(top_n), lbl
        elif all_combos_file.exists():
            df2 = pd.read_parquet(all_combos_file)
            df2 = _sort_df(df2).head(top_n)
            lbl = "calibrated (from all_combos)" if ("calibrated_sharpe_pred" in df2.columns or "calibrated_sharpe_full" in df2.columns) else "IC (from all_combos)"
            return df2.reset_index(drop=True), lbl
        else:
            return df, "IC (top100_by_ic)"
    if top_combos_file.exists():
        df = pd.read_parquet(top_combos_file)
        df = _sort_df(df)
        lbl = "calibrated (top_combos)" if ("calibrated_sharpe_pred" in df.columns or "calibrated_sharpe_full" in df.columns) else "IC (top_combos)"
        return df.reset_index(drop=True), lbl
    if all_combos_file.exists():
        df = pd.read_parquet(all_combos_file)
        df = _sort_df(df).head(top_n)
        lbl = "calibrated (from all_combos)" if ("calibrated_sharpe_pred" in df.columns or "calibrated_sharpe_full" in df.columns) else "IC (from all_combos)"
    # 修复参数名错误: drop_more -> drop
    return df.reset_index(drop=True), lbl
    raise FileNotFoundError(
        f"未找到 {run_dir} 下的 top100_by_ic/top_combos/all_combos 文件"
    )


def summarize_results(results_df: pd.DataFrame):
    """生成汇总指标字典，用于打印/对比。"""
    from scipy.stats import spearmanr  # 如果缺失，会在调用处捕获

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

    # 加载配置（增强路径鲁棒性）
    cfg_candidates = []
    # 1) CWD 相对路径
    cfg_candidates.append(Path("configs/combo_wfo_config.yaml").resolve())
    # 2) 脚本所在工程路径上一级的 configs
    try:
        cfg_candidates.append((Path(__file__).resolve().parent.parent / "configs" / "combo_wfo_config.yaml").resolve())
    except Exception:
        pass
    # 3) 环境变量覆盖
    env_cfg = os.environ.get("RB_CONFIG_FILE")
    if env_cfg:
        cfg_candidates.insert(0, Path(env_cfg).expanduser().resolve())
    config_path = next((p for p in cfg_candidates if p.exists()), None)
    if not config_path:
        raise FileNotFoundError(f"未找到配置文件，已尝试: {cfg_candidates}. 可设置 RB_CONFIG_FILE 指定绝对路径")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"使用配置文件: {config_path}")
    commission_rate_cfg = config["backtest"].get("commission_rate", 0.00005)

    # 运行参数覆盖（通过环境变量，可不改配置快速切换批量规模与扫描模式）
    # RB_TOPK: 覆盖 combo_wfo.top_n，例如 1000
    # RB_BACKTEST_ALL: "1"/"true" 回测全量组合（忽略RB_TOPK限制）
    # RB_TEST_ALL_FREQS: "1"/"true" 开启全频扫描
    # RB_FREQ_SUBSET: 逗号分隔的频率子集，如 "6,7,8,9,10,11,12,13,21"；设置后仅扫描该子集
    # RB_SKIP_PREV: "1" 跳过上一轮 run 的对比以节省时间
    
    # 解析RB_BACKTEST_ALL
    backtest_all = os.environ.get("RB_BACKTEST_ALL", "0").strip().lower() in ("1", "true", "yes")
    if backtest_all:
        logger.info("⚙️  全量回测模式已启用 (RB_BACKTEST_ALL=1)，将回测WFO所有组合")
    
    env_topk = os.environ.get("RB_TOPK")
    if env_topk is not None:
        try:
            config["combo_wfo"]["top_n"] = int(env_topk)
        except Exception:
            logger.warning(f"RB_TOPK 无法解析为整数: {env_topk}")

    env_test_all_freqs = os.environ.get("RB_TEST_ALL_FREQS")
    if env_test_all_freqs is not None:
        val = env_test_all_freqs.strip().lower()
        config.setdefault("backtest", {})["test_all_frequencies"] = val in ("1", "true", "yes")
    # 强制锁定频率为8天（已验证最优），若关闭 test_all_frequencies 则使用 combo_wfo.rebalance_frequencies=[8]
    if not config.get("backtest", {}).get("test_all_frequencies", False):
        config.setdefault("combo_wfo", {})["rebalance_frequencies"] = [8]

    env_freq_subset = os.environ.get("RB_FREQ_SUBSET")
    freq_subset_list = None
    if env_freq_subset:
        try:
            freq_subset_list = [int(x) for x in env_freq_subset.split(",") if x.strip()]
        except Exception:
            logger.warning(f"RB_FREQ_SUBSET 解析失败: {env_freq_subset}")

    skip_prev = os.environ.get("RB_SKIP_PREV", "0").strip() in ("1", "true", "yes")

    # 🔥 白名单机制已移除：直接使用WFO TopK结果，无需额外约束
    # 移除原因：白名单依赖不稳定的WFO结果，且无法提升预测准确性
    # 替代方案：使用回归模型学习WFO→真实回测的映射关系

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

    # 查找最新的运行结果 (增强: 多路径/环境变量回退)
    candidate_roots = []
    env_root = os.environ.get("RB_WFO_ROOT")
    if env_root:
        p = Path(env_root).expanduser().resolve()
        if p.name.startswith("run_") and p.is_dir():
            candidate_roots.append(p.parent)
        else:
            candidate_roots.append(p)
    # 当前工作目录下 results
    candidate_roots.append(Path("results").resolve())
    # 脚本上级(etf_rotation_optimized)下 results
    try:
        script_results = (Path(__file__).resolve().parent.parent / "results").resolve()
        candidate_roots.append(script_results)
    except Exception:
        pass
    # 🔥 移除白名单路径推断逻辑（已无白名单机制）
    # 去重
    unique_roots = []
    seen = set()
    for r in candidate_roots:
        if r and r not in seen and r.exists():
            seen.add(r)
            unique_roots.append(r)
    run_dirs = []
    for root in unique_roots:
        run_dirs.extend([d for d in root.glob("run_*") if d.is_dir()])
    run_dirs = sorted({d.resolve() for d in run_dirs}, reverse=True)
    if not run_dirs:
        logger.error("未找到WFO运行结果！请先运行 run_combo_wfo.py 或设置 RB_WFO_ROOT 指向含 run_* 的目录")
        logger.debug(f"已尝试目录: {unique_roots}")
        return

    latest_run = run_dirs[0]
    prev_run = run_dirs[1] if (not skip_prev and len(run_dirs) > 1) else None

    # 读取"最新"组合（支持全量或TopN模式）
    logger.info("")
    logger.info("=" * 100)
    if backtest_all:
        logger.info("读取WFO全量组合（ALL模式）...")
    else:
        logger.info(f"读取WFO Top {config['combo_wfo']['top_n']} 组合（最新 run）...")
    logger.info("=" * 100)
    
    latest_top_df, latest_sort_method = load_top_combos_from_run(
        latest_run, 
        top_n=config["combo_wfo"]["top_n"],
        load_all=backtest_all
    )
    logger.info(f"读取目录: {latest_run}")
    logger.info(
        f"成功读取 Top {len(latest_top_df)} 个组合（排序方式：{latest_sort_method}）"
    )
    logger.info("")

    # 若指定白名单，则应用到最新TopN
    def _load_whitelist(path: str):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"白名单文件不存在: {p}")
        combos = []
        try:
            if p.suffix.lower() in [".csv", ".parquet", ".feather", ".pkl", ".pickle", ".tsv", ".txt"]:
                if p.suffix.lower() == ".csv":
                    df_wl = pd.read_csv(p)
                elif p.suffix.lower() == ".tsv":
                    df_wl = pd.read_csv(p, sep='\t')
                elif p.suffix.lower() == ".parquet":
                    df_wl = pd.read_parquet(p)
                elif p.suffix.lower() in [".feather"]:
                    df_wl = pd.read_feather(p)
                elif p.suffix.lower() in [".pkl", ".pickle"]:
                    import pickle
                    with open(p, "rb") as f:
                        obj = pickle.load(f)
                    if isinstance(obj, (list, tuple)):
                        return [str(x) for x in obj]
                    elif isinstance(obj, pd.DataFrame):
                        df_wl = obj
                    else:
                        return [str(obj)]
                else:
                    # .txt 尝试逐行读取
                    with open(p, "r", encoding="utf-8") as f:
                        return [line.strip() for line in f if line.strip()]

                # DataFrame 情况：优先取 'combo' 列；否则取第一列
                if isinstance(df_wl, pd.DataFrame):
                    if 'combo' in df_wl.columns:
                        combos = df_wl['combo'].astype(str).tolist()
                    else:
                        combos = df_wl.iloc[:,0].astype(str).tolist()
                else:
                    combos = []
            else:
                with open(p, "r", encoding="utf-8") as f:
                    combos = [line.strip() for line in f if line.strip()]
        except Exception as e:
            raise RuntimeError(f"白名单文件解析失败: {p}, 原因: {e}")
        return combos

    # 🔥 白名单机制已完全移除
    # 原代码：whitelist_combos 筛选逻辑
    # 新逻辑：直接使用 latest_top_df (WFO TopK)，无任何约束
    logger.info("⚠️  白名单机制已禁用，直接使用WFO排序结果")

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
            # 🔥 移除上一轮白名单筛选逻辑（已无白名单机制）
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
        progress_meta=None,
        progress_counter=None,
        total_tasks: int | None = None,
        progress_step: int = 10,
    ):
        """
        单个组合回测（用于并行化）

        参数:
            test_freq: int or None, 如果指定则覆盖WFO推荐频率进行测试
            test_position_size: int or None, 如果指定则覆盖默认持仓数进行测试
        """
        # 可选：进程级 numba 预热（仅首次）
        if os.environ.get("RB_NUMBA_WARMUP", "1").strip().lower() in ("1", "true", "yes"):
            _numba_warmup()

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

        progress_enabled = bool(progress_meta is not None and progress_counter is not None and total_tasks is not None)
        if progress_enabled:
            try:
                logger.info(
                    f"[START {idx+1}/{total_tasks}] combo={combo_name[:60]} freq={rebalance_freq} size={combo_size}"
                )
            except Exception:
                pass

        # 回测
        try:
            result = backtest_no_lookahead(
                factors_data=factors_selected,
                returns=returns_shared,
                etf_names=etf_names,
                rebalance_freq=rebalance_freq,
                lookback_window=252,
                position_size=position_size,
                commission_rate=commission_rate_cfg,
                initial_capital=1_000_000.0,
                factors_data_full=factors_data_shared,
                factor_indices_for_cache=factor_indices,
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
            # 进度更新与完成日志
            if progress_enabled:
                try:
                    # 更新计数
                    with progress_counter.get_lock():
                        progress_counter.value += 1
                        done = progress_counter.value
                    # 单组合耗时（若profile开启）
                    combo_time_ms = (
                        result.get("profile", {}).get("time_total", 0.0) * 1000.0
                    )
                    logger.info(
                        f"[DONE  {idx+1}/{total_tasks}] combo={combo_name[:50]} time={combo_time_ms:.1f}ms annual={result['annual_ret']:.2%} sharpe={result['sharpe']:.3f}"
                    )
                    if done % progress_step == 0 or done == total_tasks:
                        start_ts = progress_meta.get("start_ts", None)
                        if start_ts is not None:
                            elapsed = time.perf_counter() - start_ts
                            avg = elapsed / done
                            eta = avg * (total_tasks - done)
                            logger.info(
                                f"[PROGRESS] {done}/{total_tasks} ({done/total_tasks:.1%}) avg={avg:.3f}s ETA={eta/60:.1f}m elapsed={elapsed:.1f}s"
                            )
                except Exception:
                    pass
            return result

        except Exception as e:
            try:
                logger.warning(f"回测失败: combo={combo_name[:60]} freq={rebalance_freq} err={e}")
            except Exception:
                pass
            return None

    def run_batch_backtest(
        top_df: pd.DataFrame,
        run_tag: str,
        n_jobs=8,  # ✅ 提升默认并行度到8核（从4核）
        test_all_freqs=False,
        freq_range=range(1, 31),
        test_all_position_sizes=False,
        position_size_range=range(1, 11),
        force_freq: int | None = None,
    ):
        """
        批量回测（支持并行）

        参数:
            test_all_freqs: bool, 是否测试所有换仓频率
            freq_range: range, 测试的频率范围(默认1-30天)
            test_all_position_sizes: bool, 是否测试所有持仓数
            position_size_range: range, 测试的持仓数范围(默认1-10)
        """
        profile_enabled = os.environ.get("RB_PROFILE_BACKTEST", "0").strip().lower() in ("1", "true", "yes")
        # 进度控制环境变量
        progress_enabled = os.environ.get("RB_ENABLE_PROGRESS", "0").strip().lower() in ("1", "true", "yes")
        progress_step = int(os.environ.get("RB_PROGRESS_STEP", "10") or 10)
        manager = None
        progress_counter = None
        progress_meta = None
        if progress_enabled:
            try:
                manager = Manager()
                progress_counter = manager.Value('i', 0)
                progress_meta = manager.dict()
                progress_meta['start_ts'] = time.perf_counter()
            except Exception as e:
                logger.warning(f"进度管理器初始化失败: {e}")
                progress_enabled = False

        # ================= 预加载所有 TopK 因子列滚动IC（避免首列填充慢点） =================
        preload_ic = os.environ.get("RB_PRELOAD_IC", "0").strip().lower() in ("1", "true", "yes")
        lookback_window = 252
        if preload_ic and not (test_all_freqs or test_all_position_sizes):
            try:
                t_preload_start = time.perf_counter()
                # 收集所有因子索引
                all_factor_names = set()
                for _, r in top_df.iterrows():
                    for f in str(r["combo"]).split("+"):
                        fn = f.strip()
                        if fn:
                            all_factor_names.add(fn)
                factor_index_set = {factor_names.index(fn) for fn in all_factor_names if fn in factor_names}
                factor_index_arr = np.asarray(sorted(factor_index_set), dtype=np.int64)
                # 收集所有频率（或强制频率）
                freq_candidates = set()
                if force_freq is not None:
                    freq_candidates.add(int(force_freq))
                else:
                    for _, r in top_df.iterrows():
                        try:
                            freq_candidates.add(int(r["best_rebalance_freq"]))
                        except Exception:
                            pass
                # 逐频率填充缓存
                filled_pairs = 0
                for freq_val in sorted(freq_candidates):
                    rebalance_indices = np.arange(lookback_window + 1, factors_data.shape[0], freq_val, dtype=np.int32)
                    _ = get_ic_weights_matrix_cached(
                        factors_data_full=factors_data,
                        returns=returns,
                        rebalance_indices=rebalance_indices,
                        lookback_window=lookback_window,
                        factor_indices=factor_index_arr,
                    )
                    filled_pairs += 1
                t_preload_elapsed = time.perf_counter() - t_preload_start
                logger.info(
                    f"🔄 预加载IC缓存完成: 因子列={len(factor_index_arr)} 频率集合={sorted(freq_candidates)} 次数={filled_pairs} 用时={t_preload_elapsed:.2f}s"
                )
            except Exception as e:
                logger.warning(f"预加载IC缓存失败: {e}")

        # ================= 任务批内合并（减少调度开销） =================
        task_batch_size_env = os.environ.get("RB_TASK_BATCH_SIZE", "1")
        try:
            task_batch_size = max(1, int(task_batch_size_env))
        except Exception:
            task_batch_size = 1
        def _group_batches(task_list):
            if task_batch_size <= 1:
                return [[t] for t in task_list]
            return [task_list[i:i+task_batch_size] for i in range(0, len(task_list), task_batch_size)]
        def _run_batch_wrapper(batch, mode_tag, total_tasks):
            # batch: list of tuples representing original tasks arguments
            results_local = []
            for args in batch:
                results_local.append(args())
            return results_local

        if test_all_freqs and test_all_position_sizes:
            logger.info("=" * 100)
            logger.info(
                f"🚀 全参数扫描模式: Top {len(top_df)} 组合 × {len(freq_range)} 个频率 × {len(position_size_range)} 个持仓数 = {len(top_df) * len(freq_range) * len(position_size_range)} 个策略"
            )
            logger.info(f"并行度: {n_jobs} 核心")
            logger.info("=" * 100)
            logger.info("")

            # 生成所有(组合, 频率, 持仓数)任务三元组
            tasks_raw = []
            for idx, row in top_df.iterrows():
                for freq in freq_range:
                    for pos_size in position_size_range:
                        tasks_raw.append(lambda idx=idx, row=row, freq=freq, pos_size=pos_size: _backtest_single_combo(
                            idx,
                            row,
                            factors_data,
                            returns,
                            etf_names,
                            factor_names,
                            run_tag,
                            test_freq=freq,
                            test_position_size=pos_size,
                            progress_meta=progress_meta if progress_enabled else None,
                            progress_counter=progress_counter if progress_enabled else None,
                            total_tasks=None if not progress_enabled else len(top_df) * len(freq_range) * len(position_size_range),
                            progress_step=progress_step,
                        ))
            batches = _group_batches(tasks_raw)

            total_tasks = len(tasks_raw)
            results_nested = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(_run_batch_wrapper)(batch, "all_param", total_tasks) for batch in batches
            )
            results = [r for batch in results_nested for r in batch]

        elif test_all_freqs:
            logger.info("=" * 100)
            logger.info(
                f"🚀 全频率扫描模式: Top {len(top_df)} 组合 × {len(freq_range)} 个频率 = {len(top_df) * len(freq_range)} 个策略"
            )
            logger.info(f"并行度: {n_jobs} 核心")
            logger.info("=" * 100)
            logger.info("")

            # 生成所有(组合, 频率)任务对
            tasks_raw = []
            for idx, row in top_df.iterrows():
                for freq in freq_range:
                    tasks_raw.append(lambda idx=idx, row=row, freq=freq: _backtest_single_combo(
                        idx,
                        row,
                        factors_data,
                        returns,
                        etf_names,
                        factor_names,
                        run_tag,
                        test_freq=freq,
                        progress_meta=progress_meta if progress_enabled else None,
                        progress_counter=progress_counter if progress_enabled else None,
                        total_tasks=None if not progress_enabled else len(top_df) * len(freq_range),
                        progress_step=progress_step,
                    ))
            batches = _group_batches(tasks_raw)
            total_tasks = len(tasks_raw)
            results_nested = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(_run_batch_wrapper)(batch, "all_freq", total_tasks) for batch in batches
            )
            results = [r for batch in results_nested for r in batch]

        elif test_all_position_sizes:
            logger.info("=" * 100)
            logger.info(
                f"🚀 全持仓数扫描模式: Top {len(top_df)} 组合 × {len(position_size_range)} 个持仓数 = {len(top_df) * len(position_size_range)} 个策略"
            )
            logger.info(f"并行度: {n_jobs} 核心")
            logger.info("=" * 100)
            logger.info("")

            # 生成所有(组合, 持仓数)任务对
            tasks_raw = []
            for idx, row in top_df.iterrows():
                for pos_size in position_size_range:
                    tasks_raw.append(lambda idx=idx, row=row, pos_size=pos_size: _backtest_single_combo(
                        idx,
                        row,
                        factors_data,
                        returns,
                        etf_names,
                        factor_names,
                        run_tag,
                        test_position_size=pos_size,
                        progress_meta=progress_meta if progress_enabled else None,
                        progress_counter=progress_counter if progress_enabled else None,
                        total_tasks=None if not progress_enabled else len(top_df) * len(position_size_range),
                        progress_step=progress_step,
                    ))
            batches = _group_batches(tasks_raw)
            total_tasks = len(tasks_raw)
            results_nested = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(_run_batch_wrapper)(batch, "all_pos", total_tasks) for batch in batches
            )
            results = [r for batch in results_nested for r in batch]

        else:
            logger.info("=" * 100)
            logger.info(f"开始批量回测 Top {len(top_df)} 组合（{run_tag}，无未来函数）")
            logger.info(f"并行度: {n_jobs} 核心")
            logger.info("=" * 100)
            logger.info("")

            # 并行回测(使用WFO推荐频率和默认持仓数)
            tasks_raw = []
            for idx, row in top_df.iterrows():
                tasks_raw.append(lambda idx=idx, row=row: _backtest_single_combo(
                    idx,
                    row,
                    factors_data,
                    returns,
                    etf_names,
                    factor_names,
                    run_tag,
                    test_freq=force_freq if force_freq is not None else None,
                    test_position_size=None,
                    progress_meta=progress_meta if progress_enabled else None,
                    progress_counter=progress_counter if progress_enabled else None,
                    total_tasks=None if not progress_enabled else len(top_df),
                    progress_step=progress_step,
                ))
            batches = _group_batches(tasks_raw)
            total_tasks = len(tasks_raw)
            results_nested = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(_run_batch_wrapper)(batch, "default", total_tasks) for batch in batches
            )
            results = [r for batch in results_nested for r in batch]

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

        rows = []
        for r in all_results_local:
            row_dict = {
                "rank": r["rank"],
                "combo": r["combo"],
                "combo_size": r["combo_size"],
                "wfo_freq": r["wfo_freq"],
                "test_freq": r["test_freq"],
                "test_position_size": r.get("test_position_size", 5),  # ✨ 新增：测试的持仓数
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
            if profile_enabled:
                profile = r.get("profile") or {}
                row_dict["profile_time_total"] = profile.get("time_total")
                row_dict["profile_time_precompute_ic"] = profile.get("time_precompute_ic")
                row_dict["profile_time_main_loop"] = profile.get("time_main_loop")
                row_dict["profile_rebalance_executed"] = profile.get("rebalance_executed")
                row_dict["profile_loop_iterations"] = profile.get("loop_iterations")
                row_dict["profile_avg_turnover"] = profile.get("avg_turnover")
            rows.append(row_dict)

        df_local = pd.DataFrame(rows)

        if profile_enabled:
            profile_cols = [
                "profile_time_total",
                "profile_time_precompute_ic",
                "profile_time_main_loop",
            ]
            available_cols = [c for c in profile_cols if c in df_local.columns]
            if available_cols and not df_local[available_cols].dropna(how="all").empty:
                logger.info("")
                logger.info("🕒 Profiling摘要 (ms)")
                profile_stats = df_local[available_cols].dropna()
                for col in available_cols:
                    series_ms = profile_stats[col] * 1000
                    if series_ms.empty:
                        continue
                    logger.info(
                        f"  {col.replace('profile_', '')}: mean {series_ms.mean():7.1f} | median {series_ms.median():7.1f} | max {series_ms.max():7.1f}"
                    )

                if "profile_time_total" in profile_stats.columns:
                    worst_idx = profile_stats["profile_time_total"].idxmax()
                    if worst_idx is not None and not pd.isna(worst_idx):
                        worst_row = df_local.loc[worst_idx]
                        logger.info(
                            f"  最慢组合: rank {worst_row['rank']} | {worst_row['combo'][:60]} | time_total {profile_stats.loc[worst_idx, 'profile_time_total']*1000:7.1f} ms"
                        )
                # ====== Outlier诊断 (可选) ======
                if os.environ.get("RB_OUTLIER_REPORT","0").strip().lower() in ("1","true","yes"):
                    try:
                        total_ms = profile_stats["profile_time_total"] * 1000
                        p95 = float(np.percentile(total_ms.values, 95))
                        p99 = float(np.percentile(total_ms.values, 99))
                        outliers = profile_stats[total_ms > p95].index.tolist()
                        logger.info(f"  Outlier阈值: p95={p95:.1f}ms p99={p99:.1f}ms (count>{len(outliers)})")
                        for oid in outliers:
                            row = df_local.loc[oid]
                            path = row.get("profile_ic_path","?") if "profile_ic_path" in row else row.get("ic_path","?")
                            stable_flag = row.get("profile_stable_rank", row.get("stable_rank", False))
                            loop_ms = row.get("profile_time_main_loop", 0.0) * 1000
                            pre_ms = row.get("profile_time_precompute_ic", 0.0) * 1000
                            total_val = row.get("profile_time_total", 0.0) * 1000
                            loop_ratio = (loop_ms / total_val) if total_val > 0 else 0.0
                            logger.info(
                                f"    [OUTLIER] rank={row['rank']} combo={row['combo'][:50]} total={total_val:.1f}ms pre_ic={pre_ms:.1f}ms loop={loop_ms:.1f}ms loop_ratio={loop_ratio:.2f} ic_path={path} stable={stable_flag}"
                            )
                    except Exception as e:
                        logger.warning(f"  Outlier报告失败: {e}")

        return df_local

    # ========== 全频率扫描模式(可选) ==========
    # 原逻辑根据配置/环境变量决定是否触发 1-30 天全频率扫描；当前已验证 8 天频率最优，
    # 为避免误触导致 30x 扩容的巨量任务，这里强制关闭全频率扫描。
    # 如果后续确需重新开启，请将下面的 TEST_ALL_FREQS 改为原来的读取配置方式：
    # TEST_ALL_FREQS = config.get("backtest", {}).get("test_all_frequencies", False)
    TEST_ALL_FREQS = False  # 🔒 强制禁用全频率扫描
    # 同步回写（防止后续代码再次读取配置触发 True）
    if "backtest" in config:
        config["backtest"]["test_all_frequencies"] = False
    TEST_ALL_POSITION_SIZES = config.get("backtest", {}).get(
        "test_all_position_sizes", False
    )
    # 频率范围：支持通过环境变量传入子集
    FREQ_RANGE = range(1, 31) if not freq_subset_list else list(sorted(set(freq_subset_list)))
    POSITION_SIZE_RANGE = range(1, 11)  # 1-10个持仓

    # 统一的结果输出目录，需在全频率/常规回测前创建
    output_dir = Path("results_combo_wfo")
    output_dir.mkdir(exist_ok=True)
    # 每次脚本调用生成独立调用时间戳，避免覆盖同一 WFO run 输出
    invocation_ts = os.environ.get("RB_RESULT_TS") or datetime.now().strftime("%Y%m%d_%H%M%S")
    def _make_run_output_dir(latest_ts: str) -> Path:
        d = output_dir / f"{latest_ts}_{invocation_ts}"
        d.mkdir(parents=True, exist_ok=True)
        return d

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
            run_output_dir = _make_run_output_dir(latest_ts)
            all_param_file = run_output_dir / f"all_param_scan_{latest_ts}_{invocation_ts}.csv"
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
            run_output_dir = _make_run_output_dir(latest_ts)
            all_pos_file = run_output_dir / f"all_pos_scan_{latest_ts}_{invocation_ts}.csv"
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

    elif TEST_ALL_FREQS:  # 此分支现在不可达（TEST_ALL_FREQS 强制为 False）
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
            run_output_dir = _make_run_output_dir(latest_ts)
            all_freq_file = run_output_dir / f"all_freq_scan_{latest_ts}_{invocation_ts}.csv"
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
            # 频率相关性与最佳频率分布摘要输出
            try:
                latest_ts = latest_run.name.replace("run_", "")
                run_output_dir = _make_run_output_dir(latest_ts)
                base_freq = 8 if 8 in all_freq_results_df['test_freq'].unique() else None
                summary = {}
                if base_freq is not None:
                    base_df = all_freq_results_df[all_freq_results_df['test_freq']==base_freq][['combo','sharpe']].rename(columns={'sharpe':f'sharpe_{base_freq}'})
                    import math  # 使用全局已导入的 numpy as np，避免在函数作用域重新绑定
                    try:
                        from scipy.stats import spearmanr  # type: ignore
                        _spearman = lambda a,b: spearmanr(a,b).correlation
                    except Exception:
                        def _spearman(a,b):
                            a = np.asarray(a); b = np.asarray(b)
                            ra = np.argsort(np.argsort(a)).astype(float)
                            rb = np.argsort(np.argsort(b)).astype(float)
                            ra -= ra.mean(); rb -= rb.mean()
                            num = (ra*rb).sum(); den = np.sqrt((ra**2).sum()*(rb**2).sum())
                            return float(num/den) if den!=0 else 0.0
                    corrs = {}
                    for f in sorted(all_freq_results_df['test_freq'].unique()):
                        if f == base_freq: continue
                        cur = all_freq_results_df[all_freq_results_df['test_freq']==f][['combo','sharpe']].rename(columns={'sharpe':f'sharpe_{f}'})
                        merged = base_df.merge(cur, on='combo')
                        if len(merged) >= 30:
                            val = _spearman(merged[f'sharpe_{base_freq}'], merged[f'sharpe_{f}'])
                            if val is not None and not math.isnan(val):
                                corrs[int(f)] = float(val)
                    if corrs:
                        summary['base_freq'] = base_freq
                        summary['spearman_vs_base'] = corrs
                        summary['median_spearman'] = float(np.median(list(corrs.values())))
                best_freq_series = all_freq_results_df.sort_values(['combo','sharpe'], ascending=[True,False]).groupby('combo').first()['test_freq']
                summary['best_freq_counts'] = {int(k): int(v) for k,v in best_freq_series.value_counts().to_dict().items()}
                summary['n_combos'] = int(all_freq_results_df['combo'].nunique())
                summary['n_rows'] = int(len(all_freq_results_df))
                summary_path = run_output_dir / f"freq_correlation_summary_{latest_ts}_{invocation_ts}.json"
                import json
                with open(summary_path,'w') as fp:
                    json.dump(summary, fp, ensure_ascii=False, indent=2)
                logger.info(f"频率相关性摘要已保存: {summary_path}")
            except Exception as e:
                logger.warning(f"频率相关性摘要生成失败: {e}")

    # ========== 常规单频率回测 ==========
    # 支持通过环境变量 RB_FORCE_FREQ 强制覆盖所有组合的回测频率（例如统一用8天验证排序一致性）
    force_freq_env = os.environ.get("RB_FORCE_FREQ")
    force_freq = None
    if force_freq_env:
        try:
            force_freq = int(force_freq_env)
            logger.info(f"⚙️ 启用强制频率: 所有组合统一使用 {force_freq} 天换仓")
        except Exception:
            logger.warning(f"RB_FORCE_FREQ 解析失败: {force_freq_env}")
    latest_results_df = run_batch_backtest(
        latest_top_df, run_tag=f"latest:{latest_run.name}", force_freq=force_freq
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
    run_output_dir = _make_run_output_dir(latest_ts)
    topN = len(latest_top_df)
    output_file = run_output_dir / f"top{topN}_backtest_by_ic_{latest_ts}_{invocation_ts}.csv"
    results_df_sorted.to_csv(output_file, index=False)

    logger.info("")
    logger.info(f"最新结果已保存至: {output_file}")
    logger.info("")

    # 保存详细结果（同表即可）
    output_file_full = run_output_dir / f"top{topN}_backtest_by_ic_{latest_ts}_{invocation_ts}_full.csv"
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
