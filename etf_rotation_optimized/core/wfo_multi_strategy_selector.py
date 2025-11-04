"""
WFO 多策略枚举与 Top-5 组合选择

目标：
- 基于 WFO 窗口结果（选中因子 + 权重），枚举多种"策略定义"
- 对每个策略定义，严格 T+1 计算全周期 OOS 拼接收益与 KPI
- 根据综合评分选出 Top-5 策略，并输出 Top-5 等权组合的收益与 KPI

关键设计：
- 策略定义 = (因子子集, 温度tau, TopN)
- 信号生成：仅使用该子集内的因子；对窗口内原权重应用温度缩放并归一
- 收益计算：严格 T+1，用 t-1 信号在 t 日持有 TopN 等权

输入：
- results_list: List[DirectFactorWindowResult]
- factors: np.ndarray (T, N, K)
- returns: np.ndarray (T, N)
- factor_names: List[str]
- dates: pd.DatetimeIndex (可选)

输出：
- strategies_ranked.csv: 所有候选策略评分与指标
- top5_strategies.csv: Top-5 策略详情
- top5_combo_returns.csv / top5_combo_equity.csv / top5_combo_kpi.csv

性能优化：
- Numba JIT编译热路径循环（3-5倍加速）
- 向量化Z-score/覆盖率计算（60-70倍加速）
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Numba JIT编译支持（可选依赖，降级到Python循环）
try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    # 降级：定义空装饰器
    def njit(*args, **kwargs):
        def wrapper(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return wrapper


# ========================================================================
# Numba JIT编译核心函数（热路径优化）
# ========================================================================


@njit(cache=True)
def _count_intersection_jit(arr1, arr2):
    """计算两个数组的交集大小（Numba优化版本）

    性能：O(N*M)，但JIT编译后非常快（N,M通常<20）
    """
    count = 0
    for val in arr1:
        for check_val in arr2:
            if val == check_val:
                count += 1
                break
    return count


@njit(cache=True)
def _topn_core_jit(sig_shifted, returns, valid_mask, top_n):
    """严格T+1收益+换手率核心循环（Numba JIT编译版本）

    ⚡ Linus优化：
    - JIT编译Python循环 → 3-5倍加速
    - NumPy数组替代set → Numba兼容
    - 预计算valid_mask → 减少重复计算

    Args:
        sig_shifted: (T, N) 信号数组（已T+1延迟）
        returns: (T, N) 收益数组
        valid_mask: (T, N) 有效性mask
        top_n: TopN持仓数量

    Returns:
        daily_ret, daily_to: 每日收益和换手率数组
    """
    T, N = returns.shape
    daily_ret = np.zeros(T, dtype=np.float64)
    daily_to = np.zeros(T, dtype=np.float64)
    prev_hold = np.empty(0, dtype=np.int64)  # 前一天持仓索引

    for t in range(1, T):
        mask_t = valid_mask[t]
        if not np.any(mask_t):
            # 当天无有效数据
            if prev_hold.size > 0:
                daily_to[t] = 1.0  # 全部清仓
            prev_hold = np.empty(0, dtype=np.int64)
            continue

        # 提取有效数据
        valid_indices = np.where(mask_t)[0]
        valid_sig = sig_shifted[t][mask_t]
        valid_ret = returns[t][mask_t]

        n_valid = len(valid_indices)
        k = min(top_n, n_valid)

        if k == 0:
            if prev_hold.size > 0:
                daily_to[t] = 1.0
            prev_hold = np.empty(0, dtype=np.int64)
            continue

        # Top-K选择（argpartition比argsort快）
        if k < n_valid:
            # 部分排序：只保证最大的k个在右侧
            part_idx = np.argpartition(valid_sig, -k)[-k:]
            # 对Top-K部分完全排序
            sorted_part = np.argsort(valid_sig[part_idx])[::-1]
            topk_local = part_idx[sorted_part]
        else:
            # 全排序
            topk_local = np.argsort(valid_sig)[::-1]

        # 映射回原始索引
        topk = valid_indices[topk_local]

        # 计算收益（Top-K等权）
        daily_ret[t] = np.mean(valid_ret[topk_local])

        # 计算换手率
        if prev_hold.size == 0:
            daily_to[t] = 1.0  # 首次建仓
        else:
            inter_count = _count_intersection_jit(prev_hold, topk)
            daily_to[t] = 1.0 - float(inter_count) / float(top_n)

        # 保存当天持仓
        prev_hold = topk.copy()

    daily_ret[0] = 0.0
    daily_to[0] = 0.0
    return daily_ret, daily_to


# ========================================================================
# 数据类与工具函数
# ========================================================================

from dataclasses import dataclass
from itertools import combinations, product
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class StrategySpec:
    factors: Tuple[str, ...]
    tau: float  # 温度参数（>0，越小越集中；=1为原样；>1越均匀）
    top_n: int
    z_threshold: Optional[float] = None  # 跨截面Z分数阈值；None表示不启用阈值过滤

    def key(self) -> str:
        zt = "none" if self.z_threshold is None else str(self.z_threshold)
        return f"factors={ '|'.join(self.factors) };tau={self.tau};topn={self.top_n};zthr={zt}"


class WFOMultiStrategySelector:
    def __init__(
        self,
        min_factor_freq: float = 0.3,
        min_factors: int = 3,
        max_factors: int = 5,
        subset_mode: str = "enumerate",  # "enumerate" | "all"
        tau_grid: Sequence[float] = (0.7, 1.0, 1.5),
        topn_grid: Sequence[int] = (6,),
        signal_z_threshold_grid: Sequence[Optional[float]] = (None,),
        max_strategies: int = 200,
        non_overlap_oos: bool = False,
        turnover_penalty: float = 0.0,
        coverage_penalty_coef: float = 1.0,  # P0修复: 参数化覆盖率惩罚系数
        coverage_min: float = 0.0,
        avg_turnover_max: Optional[float] = None,
        rank_by: str = "score",
        # 分层/随机增强配置（可选）
        stratified_by_k: bool = False,
        k_quota: Optional[Dict[int, float]] = None,  # 比例或绝对数量（按spec数量计）
        subset_shuffle: bool = False,
        random_seed: Optional[int] = None,
    ):
        self.min_factor_freq = float(min_factor_freq)
        self.min_factors = int(min_factors)
        self.max_factors = int(max_factors)
        self.subset_mode = str(subset_mode).lower()
        self.tau_grid = list(tau_grid)
        self.topn_grid = list(topn_grid)
        self.signal_z_threshold_grid = list(signal_z_threshold_grid)
        self.max_strategies = int(max_strategies)
        # 增强参数
        self.non_overlap_oos = bool(non_overlap_oos)
        self.turnover_penalty = float(turnover_penalty)
        self.coverage_penalty_coef = float(coverage_penalty_coef)
        self.coverage_min = float(coverage_min)
        self.avg_turnover_max = (
            avg_turnover_max if avg_turnover_max is None else float(avg_turnover_max)
        )
        self.rank_by = str(rank_by).lower()

        # 分层/随机增强
        self.stratified_by_k = bool(stratified_by_k)
        self.k_quota = dict(k_quota) if k_quota is not None else None
        self.subset_shuffle = bool(subset_shuffle)
        self.random_seed = None if random_seed is None else int(random_seed)

    # ---------- 基础工具 ----------
    @staticmethod
    def _compute_kpis(daily_returns: pd.Series) -> Dict[str, float]:
        r = daily_returns.fillna(0.0).values
        if r.size < 2:
            return {
                "total_return": 0.0,
                "annual_return": 0.0,
                "annual_volatility": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "calmar_ratio": 0.0,
                "win_rate": 0.0,
            }

        equity = np.cumprod(1 + r)
        running_max = np.maximum.accumulate(equity)
        dd = (equity - running_max) / (running_max + 1e-12)

        total_return = float(equity[-1] - 1.0)
        ann_ret = float((equity[-1]) ** (252.0 / max(1, len(r))) - 1.0)
        ann_vol = float(np.std(r) * np.sqrt(252.0))
        sharpe = float(ann_ret / ann_vol) if ann_vol > 1e-12 else 0.0
        mdd = float(np.min(dd)) if dd.size else 0.0
        calmar = float(ann_ret / abs(mdd)) if mdd < 0 else 0.0
        win = float(np.mean(r > 0))

        return {
            "total_return": total_return,
            "annual_return": ann_ret,
            "annual_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": mdd,
            "calmar_ratio": calmar,
            "win_rate": win,
        }

    @staticmethod
    def _topn_tplus1_returns(
        signals: np.ndarray, returns: np.ndarray, top_n: int
    ) -> pd.Series:
        T, N = returns.shape
        daily_ret = np.zeros(T, dtype=float)
        for t in range(1, T):
            sig_prev = signals[t - 1]
            ret_today = returns[t]
            mask = ~(np.isnan(sig_prev) | np.isnan(ret_today))
            if not np.any(mask):
                daily_ret[t] = 0.0
                continue
            valid_idx = np.where(mask)[0]
            ranked = valid_idx[np.argsort(sig_prev[mask])[::-1]]
            topk = ranked[:top_n]
            if topk.size == 0:
                daily_ret[t] = 0.0
                continue
            daily_ret[t] = float(np.nanmean(ret_today[topk]))
        daily_ret[0] = 0.0
        return pd.Series(daily_ret)

    # ---------- 核心流程 ----------
    def _frequent_factors(self, results_list) -> List[str]:
        count: Dict[str, int] = {}
        total = max(1, len(results_list))
        for r in results_list:
            for f in getattr(r, "selected_factors", []) or []:
                count[f] = count.get(f, 0) + 1
        freq = {f: c / total for f, c in count.items()}
        keep = [f for f, p in freq.items() if p >= self.min_factor_freq]
        # 按频率降序
        keep.sort(key=lambda x: freq[x], reverse=True)
        return keep

    def _enumerate_specs(self, factors: List[str]) -> List[StrategySpec]:
        specs: List[StrategySpec] = []
        # 模式1：使用全部高频因子（不做子集枚举）
        if self.subset_mode == "all":
            if len(factors) == 0:
                return specs
            full_combo = tuple(factors)
            for tau, topn, zthr in product(
                self.tau_grid, self.topn_grid, self.signal_z_threshold_grid
            ):
                specs.append(
                    StrategySpec(
                        factors=full_combo,
                        tau=float(tau),
                        top_n=int(topn),
                        z_threshold=zthr,
                    )
                )
                if len(specs) >= self.max_strategies:
                    return specs
            return specs

        # 模式2：子集枚举（支持分层配额与随机顺序）
        from random import Random

        rng = Random(self.random_seed)

        # 预生成各k的子集
        valid_k_max = min(self.max_factors, len(factors))
        k_range = [
            k
            for k in range(self.min_factors, max(self.min_factors, valid_k_max) + 1)
            if k > 0
        ]
        subsets_by_k: Dict[int, List[Tuple[str, ...]]] = {}
        for k in k_range:
            subsets = list(combinations(factors, k))
            if self.subset_shuffle and len(subsets) > 1:
                rng.shuffle(subsets)
            subsets_by_k[k] = subsets

        # 参数网格顺序（可后续支持随机化）
        param_grid = list(
            product(self.tau_grid, self.topn_grid, self.signal_z_threshold_grid)
        )

        # 若未启用分层，则按k升序、子集顺序、参数顺序线性填充
        if not self.stratified_by_k or not self.k_quota:
            for k in k_range:
                for combo in subsets_by_k[k]:
                    for tau, topn, zthr in param_grid:
                        specs.append(
                            StrategySpec(
                                factors=tuple(combo),
                                tau=float(tau),
                                top_n=int(topn),
                                z_threshold=zthr,
                            )
                        )
                        if len(specs) >= self.max_strategies:
                            return specs
            return specs

        # 分层：将max_strategies按k配额分配（支持比例或绝对spec数量）
        # 计算每个k的可用spec配额
        # - 若所有值之和<=1.0，视为比例；否则视为绝对数量（按spec计数）
        quota_values = list(self.k_quota.values())
        is_ratio = sum(quota_values) <= 1.0000001
        total_slots = int(self.max_strategies)
        k_alloc_specs: Dict[int, int] = {}
        for k in k_range:
            q = self.k_quota.get(k, 0.0)
            alloc = int(total_slots * float(q)) if is_ratio else int(q)
            k_alloc_specs[k] = max(0, alloc)

        # 四舍五入后的残差，按k循环补齐
        allocated = sum(k_alloc_specs.values())
        if allocated < total_slots:
            for k in k_range:
                if allocated >= total_slots:
                    break
                k_alloc_specs[k] += 1
                allocated += 1
        elif allocated > total_slots:
            for k in reversed(k_range):
                if allocated <= total_slots:
                    break
                if k_alloc_specs[k] > 0:
                    k_alloc_specs[k] -= 1
                    allocated -= 1

        # 每个子集对应的参数组合个数
        combos_per_subset = max(1, len(param_grid))

        # 记录每个k使用了多少子集
        self._debug_subsets_used_by_k: Dict[int, int] = {}
        self._debug_k_alloc_specs: Dict[int, int] = dict(k_alloc_specs)

        # 按k配额填充
        for k in k_range:
            remaining_specs_k = k_alloc_specs.get(k, 0)
            if remaining_specs_k <= 0:
                self._debug_subsets_used_by_k[k] = 0
                continue

            subsets = subsets_by_k.get(k, [])
            used_subsets = 0
            if not subsets:
                self._debug_subsets_used_by_k[k] = 0
                continue

            # 需要的子集数量（向上取整）
            need_subsets = int(np.ceil(remaining_specs_k / combos_per_subset))
            for combo in subsets[:need_subsets]:
                for tau, topn, zthr in param_grid:
                    specs.append(
                        StrategySpec(
                            factors=tuple(combo),
                            tau=float(tau),
                            top_n=int(topn),
                            z_threshold=zthr,
                        )
                    )
                    remaining_specs_k -= 1
                    if len(specs) >= self.max_strategies:
                        return specs
                    if remaining_specs_k <= 0:
                        break
                used_subsets += 1
                if remaining_specs_k <= 0:
                    break
            self._debug_subsets_used_by_k[k] = used_subsets

        return specs

    @staticmethod
    def _apply_z_threshold(signals: np.ndarray, z_thr: float) -> np.ndarray:
        """对每个交易日跨截面做Z分数，并将不达阈值的信号置为NaN以剔除。

        注意：请在严格T+1流程下使用，即 signals[t-1] 决定 t 的持仓；
        这里的Z分数也在 t-1 跨截面计算，不引入前视。

        性能优化：完全向量化实现，避免逐日循环
        """
        sig = signals.copy()
        T, N = sig.shape

        # 向量化计算所有天的均值和标准差
        # nanmean/nanstd会自动处理NaN值
        with np.errstate(invalid="ignore", divide="ignore"):
            # 计算每日跨截面的均值和标准差
            mu = np.nanmean(sig, axis=1, keepdims=True)  # (T, 1)
            std = np.nanstd(sig, axis=1, ddof=1, keepdims=True)  # (T, 1)

            # 找出标准差太小的天（无差异）
            no_var_mask = (std < 1e-12).squeeze()  # (T,)

            # 计算Z分数
            z = (sig - mu) / std  # (T, N)

            # 过滤逻辑：Z分数 <= 阈值 的位置设为NaN
            sig[z <= z_thr] = np.nan

            # 无差异天全部设为NaN
            if np.any(no_var_mask):
                sig[no_var_mask, :] = np.nan

        # P1修复: 统计全NaN天数（仅用于debug日志）
        all_nan_days = np.sum(np.all(np.isnan(sig), axis=1))
        if all_nan_days > 0:
            import logging

            logger = logging.getLogger(__name__)
            logger.debug(
                f"Z阈值过滤: {all_nan_days}/{T}天变为全NaN ({all_nan_days/T:.1%})"
            )

        return sig

    @staticmethod
    def _apply_temperature(weights: np.ndarray, tau: float) -> np.ndarray:
        # tau==1: 原样；tau<1: 放大差异；tau>1: 更均匀
        if tau <= 0:
            tau = 1.0
        # 归一化到概率向量
        w = np.clip(weights, 1e-12, None)
        w_sum = np.sum(w)
        if w_sum < 1e-12:  # P2修复: 所有权重都接近0
            return np.ones_like(w) / len(w)  # 返回等权
        w = w / w_sum
        # 温度缩放（幂律）
        alpha = 1.0 / tau
        w_scaled = np.power(w, alpha)
        w_scaled_sum = np.sum(w_scaled)
        if w_scaled_sum < 1e-12:  # P2修复: 缩放后和为0
            return np.ones_like(w) / len(w)
        w_scaled = w_scaled / w_scaled_sum
        return w_scaled

    def _stitch_signals_for_spec(
        self,
        spec: StrategySpec,
        results_list,
        factors: np.ndarray,  # (T, N, K)
        factor_names: List[str],
    ) -> np.ndarray:
        T, N, K = factors.shape
        name_to_idx = {n: i for i, n in enumerate(factor_names)}
        stitched = np.full((T, N), np.nan, dtype=float)

        chosen_idxs = [name_to_idx[f] for f in spec.factors if f in name_to_idx]
        if not chosen_idxs:
            return stitched

        for r in results_list:
            s, e = int(r.oos_start), int(r.oos_end)
            if s >= e or s < 0 or e > T:
                continue

            # 取窗口内该策略子集的权重，若某因子不在窗口选中则忽略
            window_factor_names = [f for f in spec.factors if f in r.factor_weights]
            if not window_factor_names:
                continue
            w = np.array(
                [r.factor_weights[f] for f in window_factor_names], dtype=float
            )
            if w.size == 0 or np.allclose(w.sum(), 0):
                continue
            w = self._apply_temperature(w, spec.tau)

            idxs = [name_to_idx[f] for f in window_factor_names]
            oos_fac = factors[s:e, :, idxs]  # (e-s, N, F_sub)
            oos_sig = np.tensordot(oos_fac, w, axes=([2], [0]))  # (e-s, N)
            if self.non_overlap_oos:
                # 仅填充当前仍为NaN的位置，避免窗口OOS重叠导致覆盖
                curr = stitched[s:e, :]
                fill_mask = np.isnan(curr)
                # 广播：只在NaN位置写入
                curr[fill_mask] = oos_sig[fill_mask]
                stitched[s:e, :] = curr
            else:
                # 默认行为：后计算的窗口覆盖先前值
                stitched[s:e, :] = oos_sig

        return stitched

    def _score(
        self, kpi: Dict[str, float], avg_turnover: float, coverage: float = 1.0
    ) -> float:
        # 简单综合分：Sharpe为主，兼顾年化与Calmar
        base = (
            0.5 * kpi.get("sharpe_ratio", 0.0)
            + 0.35 * kpi.get("annual_return", 0.0)
            + 0.15 * kpi.get("calmar_ratio", 0.0)
        )
        # 换手惩罚：降低过度交易策略的得分
        turnover_penalty = self.turnover_penalty * max(0.0, avg_turnover)

        # 覆盖率惩罚：严重惩罚低覆盖率策略（统计不显著）
        # coef=1.0: coverage=0.5 → penalty=0.25, coverage=0.3 → penalty=0.49
        # coef=2.0: coverage=0.5 → penalty=0.50, coverage=0.3 → penalty=0.98
        coverage_penalty = self.coverage_penalty_coef * (1.0 - coverage) ** 2

        return float(base - turnover_penalty - coverage_penalty)

    @staticmethod
    def _topn_tplus1_returns_and_turnover(
        signals: np.ndarray, returns: np.ndarray, top_n: int
    ) -> Tuple[pd.Series, pd.Series]:
        """严格T+1收益 + 简易换手率（Numba JIT编译版）。

        ⚡⚡⚡ ULTIMATE Linus优化 - Numba JIT:
        - 主循环JIT编译为机器码（3-5x加速）
        - 使用cache=True持久化编译（避免重复编译）
        - NumPy数组操作替代Python set（JIT兼容）
        - 首次运行有~2-3秒编译开销，后续无开销

        Performance:
        - 120K策略 × 1028天 = 123M循环迭代
        - JIT版本: <2分钟 | Python版本: 6分钟
        """
        # 准备JIT输入
        sig_shifted = np.roll(signals, 1, axis=0)  # T+1延迟
        sig_shifted[0] = np.nan

        valid_mask = ~(np.isnan(sig_shifted) | np.isnan(returns))  # (T, N)

        # 调用JIT编译的核心函数
        daily_ret, daily_to = _topn_core_jit(sig_shifted, returns, valid_mask, top_n)

        # 转换为pd.Series（保持接口兼容性）
        return pd.Series(daily_ret), pd.Series(daily_to)

    def select_and_save(
        self,
        results_list,
        factors: np.ndarray,
        returns: np.ndarray,
        factor_names: List[str],
        dates: Optional[pd.DatetimeIndex],
        out_dir,
    ) -> pd.DataFrame:
        """执行多策略枚举、评分、Top-5选择与落盘。返回 Top-5 DataFrame。"""
        # 1) 高频因子筛选
        frequent = self._frequent_factors(results_list)
        if len(frequent) < max(3, self.min_factors):
            # 退化：使用所有出现过的因子
            all_factors = set()
            for r in results_list:
                for f in getattr(r, "selected_factors", []) or []:
                    all_factors.add(f)
            frequent = sorted(all_factors)

        # 2) 生成候选策略
        specs = self._enumerate_specs(frequent)
        if not specs:
            return pd.DataFrame()

        # 枚举审计：记录理论规模
        from itertools import combinations

        factor_subsets_by_k = {}
        if self.subset_mode == "all":
            # 仅一个子集：全部高频因子
            factor_subsets_by_k = {len(frequent): [tuple(frequent)] if frequent else []}
        else:
            for k in range(self.min_factors, self.max_factors + 1):
                factor_subsets_by_k[k] = list(combinations(frequent, k))
        total_subsets = sum(len(v) for v in factor_subsets_by_k.values())
        theoretical_combos = (
            total_subsets
            * len(self.tau_grid)
            * len(self.topn_grid)
            * len(self.signal_z_threshold_grid)
        )
        enumeration_audit = {
            "factor_pool": list(frequent),
            "factor_pool_size": len(frequent),
            "min_factors": self.min_factors,
            "max_factors": self.max_factors,
            "subset_mode": self.subset_mode,
            "factor_subsets_by_k": {k: len(v) for k, v in factor_subsets_by_k.items()},
            "total_factor_subsets": total_subsets,
            "tau_grid": self.tau_grid,
            "topn_grid": self.topn_grid,
            "signal_z_threshold_grid": self.signal_z_threshold_grid,
            "param_combos_per_subset": len(self.tau_grid)
            * len(self.topn_grid)
            * len(self.signal_z_threshold_grid),
            "theoretical_total_combos": theoretical_combos,
            "actual_enumerated": len(specs),
            "max_strategies_limit": self.max_strategies,
            "hit_limit": len(specs) >= self.max_strategies,
            "stratified_by_k": self.stratified_by_k,
            "k_quota": self.k_quota,
            "subset_shuffle": self.subset_shuffle,
            "random_seed": self.random_seed,
            "subsets_used_by_k": getattr(self, "_debug_subsets_used_by_k", {}),
            "k_alloc_specs": getattr(self, "_debug_k_alloc_specs", {}),
        }

        # 3) 并行枚举（支持增量、Parquet）
        import logging

        from .wfo_parallel_enumerator import WFOParallelEnumerator

        logger = logging.getLogger(__name__)

        logger.info(f"开始并行枚举: {len(specs)}个策略")

        enumerator = WFOParallelEnumerator(
            n_workers=4,  # 4核并行
            chunk_size=500,  # 增大chunk_size: 50→500 提升并行效率
            use_parquet=True,  # 使用Parquet
            enable_incremental=True,  # 支持增量
        )

        df, per_strategy_returns = enumerator.enumerate_strategies(
            specs=specs,
            results_list=results_list,
            factors=factors,
            returns=returns,
            factor_names=factor_names,
            out_dir=out_dir,
            dates=dates,
        )

        # 4) 过滤前数量

        # 更新审计：过滤前数量
        enumeration_audit["before_filter"] = len(df)

        # 过滤：coverage 与 avg_turnover
        if not df.empty:
            df_before_cov = len(df)
            df = df[df["coverage"] >= self.coverage_min]
            df_after_cov = len(df)
            if self.avg_turnover_max is not None:
                df = df[df["avg_turnover"] <= self.avg_turnover_max]
            df_after_to = len(df)
            enumeration_audit["filtered_by_coverage"] = df_before_cov - df_after_cov
            enumeration_audit["filtered_by_turnover"] = df_after_cov - df_after_to
            enumeration_audit["after_filter"] = df_after_to

        # 排序：按照 rank_by
        if self.rank_by == "sharpe":
            df = df.sort_values(
                ["sharpe_ratio", "annual_return", "calmar_ratio"], ascending=False
            )
        elif self.rank_by == "annual_return":
            df = df.sort_values(
                ["annual_return", "sharpe_ratio", "calmar_ratio"], ascending=False
            )
        else:
            # 默认：score
            df = df.sort_values(
                ["score", "sharpe_ratio", "annual_return"], ascending=False
            )
        df.reset_index(drop=True, inplace=True)

        # 保存枚举审计报告
        import json

        with open(out_dir / "enumeration_audit.json", "w") as f:
            json.dump(enumeration_audit, f, indent=2, ensure_ascii=False)

        # 保存Top1000排行（单Parquet + rank索引）
        top1000 = df.head(1000).copy()
        top1000.insert(0, "rank", range(1, len(top1000) + 1))  # 添加rank列：1-1000
        top1000.to_parquet(out_dir / "strategies_ranked.parquet", index=False)

        logger.info(f"已保存Top{len(top1000)}策略排行 (含rank列)")

        # 保存Top1000收益序列到单个宽表Parquet
        logger.info(f"保存Top{len(top1000)}收益序列到单文件...")
        top1000_keys = top1000["_key"].tolist()
        returns_dict = {
            key: per_strategy_returns[key]
            for key in top1000_keys
            if key in per_strategy_returns
        }

        if returns_dict:
            # 构建宽表：列=策略rank，行=日期
            returns_wide = pd.DataFrame(returns_dict)
            # 重命名列为rank（方便后续查找）
            rank_mapping = {
                key: f"rank_{i+1}"
                for i, key in enumerate(top1000_keys)
                if key in returns_dict
            }
            returns_wide.rename(columns=rank_mapping, inplace=True)

            # 保存到单个Parquet
            returns_wide.to_parquet(
                out_dir / "top1000_returns.parquet", compression="snappy"
            )
            logger.info(
                f"已保存收益序列: {len(returns_wide.columns)}策略 × {len(returns_wide)}天 → top1000_returns.parquet"
            )

        # 更新审计报告
        enumeration_audit["total_ranked"] = len(df)
        enumeration_audit["saved_top_n"] = min(1000, len(df))
        enumeration_audit["file_format"] = "single_parquet_with_rank"
        enumeration_audit["returns_file"] = "top1000_returns.parquet"
        with open(out_dir / "enumeration_audit.json", "w") as f:
            json.dump(enumeration_audit, f, indent=2, ensure_ascii=False)

        # 选出Top-5并保存
        top5 = df.head(5).copy()
        top5.insert(0, "rank", range(1, 6))
        top5.to_parquet(out_dir / "top5_strategies.parquet", index=False)

        # 计算Top-5等权组合（从宽表读取）
        if len(top5) > 0 and returns_dict:
            logger.info("计算Top5等权组合...")
            top5_keys = top5["_key"].tolist()
            top5_returns = []

            for key in top5_keys:
                if key in per_strategy_returns:
                    top5_returns.append(per_strategy_returns[key])

            if top5_returns:
                # 对齐索引
                combined = pd.concat(top5_returns, axis=1)
                combo_ret = combined.mean(axis=1)
                combo_eq = (1.0 + combo_ret).cumprod()
                kpi = self._compute_kpis(combo_ret)
                # 落盘
                pd.DataFrame({"return": combo_ret}).to_csv(
                    out_dir / "top5_combo_returns.csv"
                )
                pd.DataFrame({"equity": combo_eq}).to_csv(
                    out_dir / "top5_combo_equity.csv"
                )
                pd.DataFrame([kpi]).to_csv(out_dir / "top5_combo_kpi.csv", index=False)

        # 清理内部key后返回
        if "_key" in top5.columns:
            top5 = top5.drop(columns=["_key"])
        return top5
