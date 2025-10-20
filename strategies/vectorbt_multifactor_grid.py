#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多因子轮动暴力枚举回测脚手架 (增强版)

特性：
1. 读取 ETF 横截面因子面板，支持自动加载因子排序结果
2. 支持每日截面标准化，构建多因子加权得分
3. 暴力遍历预设权重组合 + Top-N 选股，调用 vectorbt 做净值回测
4. 分批/抽样执行：支持几十万组合规模的稳定运行
5. 并发优化：向量化处理、缓存、线程池优化
6. 完善输出：时间戳目录、断点续跑、详细统计

用法示例：
    # 基础运行
    python strategies/vectorbt_multifactor_grid.py \\
        --top-factors-json production_factor_results/top_factors_20251017_124205.json \\
        --top-k 10 --output results_multifactor.csv

    # 大规模暴力搜索
    python strategies/vectorbt_multifactor_grid.py \\
        --top-factors-json production_factor_results/top_factors_*.json \\
        --top-k 20 --batch-size 10000 --max-total-combos 500000 \\
        --weight-grid 0.0 0.5 1.0 --top-n-list 3 5 8 \\
        --output factor_discovery_results/vbt_bruteforce/

    # Sanity Run (快速验证)
    python strategies/vectorbt_multifactor_grid.py \\
        --factors RETURN_20 RETURN_60 PRICE_POSITION_60 \\
        --weight-grid 0.0 1.0 --max-total-combos 1000 \\
        --sanity-run --debug

依赖：
    pip install vectorbt==0.24.3 pandas numpy numba tqdm
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# === 设置BLAS线程数，避免多进程冲突 ===
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

_DEFAULT_NUMBA_CACHE = Path(".numba_cache")
_DEFAULT_NUMBA_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(_DEFAULT_NUMBA_CACHE.resolve()))
os.environ.setdefault("NUMBA_DISABLE_JIT", "0")

try:
    import vectorbt as vbt  # type: ignore

    _HAS_VECTORBT = True
except ImportError:
    vbt = None  # type: ignore
    _HAS_VECTORBT = False


# --------------------------------------------------------------------------- #
# 数据加载与预处理
# --------------------------------------------------------------------------- #


def load_top_factors_from_json(json_path: str, top_k: int = 10) -> List[str]:
    """从production_factor_results的JSON文件中加载Top K因子

    Args:
        json_path: 因子分析结果JSON文件路径，支持通配符
        top_k: 选择的因子数量

    Returns:
        Top K因子名称列表
    """
    # 黑名单：严禁使用的未来函数和有问题的因子
    BLACKLISTED_FACTORS = [
        "RETURN_",  # ❌ 未来函数 - 严格禁止
        "FUTURE_",  # ❌ 未来函数 - 严格禁止
        "TARGET_",  # ❌ 未来函数 - 严格禁止
        # 其他潜在问题因子暂保留，待验证
    ]

    # 支持通配符，自动找到最新的文件
    if "*" in json_path:
        matching_files = glob.glob(json_path)
        if not matching_files:
            raise FileNotFoundError(f"未找到匹配的因子文件: {json_path}")
        json_path = max(matching_files, key=os.path.getctime)
        print(f"📂 自动选择最新因子文件: {json_path}")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"因子文件不存在: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 提取因子信息 - 适配新的JSON格式
    # 优先使用 all_factors，特别是当 top_k > 10 时
    if "all_factors" in data and top_k > 10:
        # 如果all_factors存在且需要超过10个因子，使用IC值排序取前top_k个
        all_factors = data["all_factors"]
        # 按ic_mean或ic_ir排序，优先使用ic_ir
        sorted_factors = sorted(
            all_factors, key=lambda x: x.get("ic_ir", x.get("ic_mean", 0)), reverse=True
        )
        top_factors = sorted_factors[:top_k]
        print(f"📊 使用 all_factors，按IC_IR排序取前 {top_k} 个因子")
    elif "top_factors" in data:
        top_factors = data["top_factors"][:top_k]
        print(f"📊 使用 top_factors，取前 {top_k} 个因子")
    elif "factor_analysis" in data and "top_factors" in data["factor_analysis"]:
        top_factors = data["factor_analysis"]["top_factors"][:top_k]
        print(f"📊 使用 factor_analysis.top_factors，取前 {top_k} 个因子")
    elif "all_factors" in data:
        # 如果all_factors存在，使用IC值排序取前top_k个
        all_factors = data["all_factors"]
        # 按ic_mean或ic_ir排序，优先使用ic_ir
        sorted_factors = sorted(
            all_factors, key=lambda x: x.get("ic_ir", x.get("ic_mean", 0)), reverse=True
        )
        top_factors = sorted_factors[:top_k]
        print(f"📊 使用 all_factors，按IC_IR排序取前 {top_k} 个因子")
    else:
        raise ValueError(f"无法从JSON文件中解析因子数据: {json_path}")

    # 从新的JSON格式中提取panel_column作为因子名称
    factor_names = []
    for item in top_factors:
        if isinstance(item, dict):
            # 新格式：使用panel_column字段
            factor_name = item.get(
                "panel_column", item.get("factor", item.get("display_name", ""))
            )
            if factor_name:
                factor_names.append(factor_name)
        else:
            # 兼容旧格式
            factor_names.append(item)

    # 严格过滤黑名单因子
    original_count = len(factor_names)
    factor_names = [
        f
        for f in factor_names
        if not any(f.startswith(blacklisted) for blacklisted in BLACKLISTED_FACTORS)
    ]

    if len(factor_names) < original_count:
        filtered_count = original_count - len(factor_names)
        print(f"🚨 黑名单过滤: 移除了 {filtered_count} 个黑名单因子")
        print(f"   黑名单模式: {BLACKLISTED_FACTORS}")

    if not factor_names:
        raise ValueError("过滤后没有可用的因子，请检查因子源")

    # 显示加载的因子信息
    print(f"🏆 已加载Top {len(factor_names)}因子:")
    for i, factor in enumerate(factor_names[:10], 1):  # 只显示前10个
        if isinstance(top_factors[i - 1], dict):
            ic_ir = top_factors[i - 1].get("ic_ir", 0)
            category = top_factors[i - 1].get("category", "unknown")
            print(f"   {i:2d}. {factor:<20} (IC_IR: {ic_ir:.4f}, 类别: {category})")
    if len(factor_names) > 10:
        print(f"   ... 还有 {len(factor_names) - 10} 个因子")

    return factor_names


def validate_factors_safety(factors: List[str]) -> List[str]:
    """验证因子安全性，过滤黑名单因子

    Args:
        factors: 候选因子列表

    Returns:
        过滤后的安全因子列表
    """
    # 黑名单：严禁使用的未来函数和有问题的因子
    BLACKLISTED_FACTORS = [
        "RETURN_",  # ❌ 未来函数 - 严格禁止
    ]

    original_count = len(factors)
    safe_factors = [
        f
        for f in factors
        if not any(f.startswith(blacklisted) for blacklisted in BLACKLISTED_FACTORS)
    ]

    if len(safe_factors) < original_count:
        filtered_count = original_count - len(safe_factors)
        print(f"🚨 安全过滤: 移除了 {filtered_count} 个黑名单因子")
        removed_factors = [
            f
            for f in factors
            if any(f.startswith(blacklisted) for blacklisted in BLACKLISTED_FACTORS)
        ]
        print(f"   移除的因子: {', '.join(removed_factors)}")
        print(f"   黑名单模式: {BLACKLISTED_FACTORS}")

    if not safe_factors:
        raise ValueError("❌ 所有因子都在黑名单中！无法继续执行")

    return safe_factors


# Schema缓存（避免重复读取）
_SCHEMA_CACHE: Dict[Path, Set[str]] = {}


def _read_parquet_schema(panel_path: Path) -> Set[str]:
    """读取Parquet文件schema（只读元数据，带缓存）

    Args:
        panel_path: Parquet文件路径

    Returns:
        列名集合
    """
    # 检查缓存
    if panel_path in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[panel_path]

    panel_columns = None

    # 方法1: pyarrow.ParquetFile (最快)
    try:
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(panel_path)
        panel_columns = set(parquet_file.schema_arrow.names)
    except ImportError:
        # pyarrow不可用，给出一次性警告
        import warnings

        warnings.warn("pyarrow不可用，将使用pandas读取schema（较慢）", UserWarning)
    except Exception:
        pass

    # 方法2: 兜底方案（全量读取）
    if panel_columns is None:
        try:
            df = pd.read_parquet(panel_path)
            panel_columns = set(df.columns)
        except Exception as e:
            raise ValueError(f"无法读取面板文件schema: {panel_path}") from e

    # 缓存结果
    _SCHEMA_CACHE[panel_path] = panel_columns
    return panel_columns


def map_factor_names_to_panel(factors: List[str], panel_path: Path) -> List[str]:
    """验证因子名称是否与面板列一致，并返回实际可用的列名。

    约束：
        - 因子筛选阶段必须给出真实的 `panel_column`
        - 若存在别名差异，直接在筛选脚本中修正，而不是在这里猜测
    """
    panel_columns = _read_parquet_schema(panel_path)
    if not panel_columns:
        raise ValueError(f"无法读取面板列信息: {panel_path}")

    # 构造大小写无关的索引，避免手工大小写差异
    panel_upper = {col.upper(): col for col in panel_columns}

    mapped: List[str] = []
    missing: List[str] = []

    for factor in factors:
        if factor in panel_columns:
            mapped.append(factor)
            continue

        # 大小写差异自动对齐
        upper_match = panel_upper.get(factor.upper())
        if upper_match:
            mapped.append(upper_match)
            continue

        missing.append(factor)

    if missing:
        raise ValueError(
            "以下因子在面板中找不到，请在因子筛选阶段修正命名：" + ", ".join(missing)
        )

    print(f"🔗 因子映射完成: {len(factors)} 个因子全部与面板列一致")
    return mapped


def load_factor_panel(panel_path: Path, factors: Sequence[str]) -> pd.DataFrame:
    """加载因子面板并裁剪到指定因子列。

    返回 MultiIndex(symbol, date) -> 因子值 的 DataFrame。
    """
    panel = pd.read_parquet(panel_path).sort_index()

    missing = [f for f in factors if f not in panel.columns]
    if missing:
        raise ValueError(f"因子列缺失：{missing}")

    return panel[list(factors)].copy()


def load_price_pivot(data_dir: Path) -> pd.DataFrame:
    """合并 raw/ETF/daily 下的报价文件，返回 (date x symbol) 的收盘价矩阵。"""
    frames: List[pd.DataFrame] = []
    for fp in data_dir.glob("*.parquet"):
        try:
            df = pd.read_parquet(fp, columns=["trade_date", "close"])
        except ValueError:
            df = pd.read_parquet(fp)
            df = df[["trade_date", "close"]]
        df["symbol"] = fp.stem.split("_")[0]
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"目录 {data_dir} 下未找到 parquet 数据文件")

    prices = pd.concat(frames, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["trade_date"])

    pivot = (
        prices.pivot(index="date", columns="symbol", values="close")
        .sort_index()
        .sort_index(axis=1)
    )
    pivot = pivot.ffill().dropna(how="all")
    return pivot


def normalize_factors(panel: pd.DataFrame, method: str = "zscore") -> pd.DataFrame:
    """对因子进行截面标准化。

    Args:
        panel: MultiIndex(symbol, date) 的因子面板
        method: 'zscore' or 'rank'
    """
    grouped = panel.groupby(level="date")

    if method == "rank":
        normalized = grouped.rank(pct=True)
        return normalized - 0.5  # 居中

    if method == "zscore":

        def _zscore(df: pd.DataFrame) -> pd.DataFrame:
            return (df - df.mean()) / df.std(ddof=0)

        normalized = grouped.transform(_zscore)
        return normalized.fillna(0.0)

    raise ValueError(f"不支持的标准化方法: {method}")


# --------------------------------------------------------------------------- #
# Numpy 向量化预处理
# --------------------------------------------------------------------------- #


class VectorizedBacktestEngine:
    """向量化回测引擎：预对齐数据，批量numpy计算

    核心优化：
    1. 预先将 normalized_panel 转为 (n_dates, n_factors, n_etfs) tensor
    2. 价格/收益率矩阵预对齐为 (n_dates, n_etfs)
    3. 权重向量直接做矩阵乘法得到综合得分
    4. np.argpartition 做 Top-N 选股
    5. 纯 numpy 公式计算回测收益
    """

    def __init__(
        self,
        normalized_panel: pd.DataFrame,
        price_pivot: pd.DataFrame,
        factors: List[str],
        fees: float = 0.001,
        init_cash: float = 1_000_000.0,
        freq: str = "1D",
    ):
        """初始化向量化引擎

        Args:
            normalized_panel: MultiIndex(symbol, date) 标准化因子面板
            price_pivot: (date x symbol) 价格矩阵
            factors: 因子列表
            fees: 交易费用
            init_cash: 初始资金
            freq: 时间频率
        """
        self.factors = factors
        self.fees = fees
        self.init_cash = init_cash
        self.freq = freq

        # === 1. 数据对齐与转换 ===
        # 将 MultiIndex 面板转为 (date x symbol x factor) 格式
        panel_unstacked = normalized_panel.unstack(level="symbol")

        # 对齐日期索引
        common_dates = panel_unstacked.index.intersection(price_pivot.index)
        panel_unstacked = panel_unstacked.loc[common_dates]
        price_pivot = price_pivot.loc[common_dates]

        # 对齐标的列
        common_symbols = list(
            set(panel_unstacked.columns.get_level_values(1)) & set(price_pivot.columns)
        )
        common_symbols.sort()

        # 提取对齐后的数据
        self.dates = common_dates
        self.symbols = common_symbols
        self.n_dates = len(common_dates)
        self.n_etfs = len(common_symbols)
        self.n_factors = len(factors)

        # === 2. 转为 numpy 数组 ===
        # factor_tensor: (n_dates, n_factors, n_etfs)
        self.factor_tensor = np.zeros(
            (self.n_dates, self.n_factors, self.n_etfs), dtype=np.float32
        )
        for i, factor in enumerate(factors):
            factor_data = panel_unstacked[factor][common_symbols].values
            self.factor_tensor[:, i, :] = np.nan_to_num(factor_data, nan=0.0)

        # price_tensor: (n_dates, n_etfs) - 保留NaN用于收益率计算
        self.price_tensor = price_pivot[common_symbols].values.astype(np.float32)

        # returns_tensor: (n_dates, n_etfs) - 先计算收益率，再处理NaN和异常值
        self.returns_tensor = np.zeros_like(self.price_tensor)
        prev_prices = self.price_tensor[:-1]
        curr_prices = self.price_tensor[1:]

        # 只在前一日价格有效时计算收益率，避免除0
        valid_mask = (
            (prev_prices > 1e-6) & np.isfinite(prev_prices) & np.isfinite(curr_prices)
        )
        returns_raw = np.zeros_like(prev_prices)
        returns_raw[valid_mask] = (
            curr_prices[valid_mask] - prev_prices[valid_mask]
        ) / prev_prices[valid_mask]

        # 限制收益率在合理范围内（日收益率 ±100%），防止数据异常
        returns_raw = np.clip(returns_raw, -1.0, 1.0)
        self.returns_tensor[1:] = returns_raw

        # 最后统一填充NaN为0（此时已没有爆炸收益）
        self.price_tensor = np.nan_to_num(self.price_tensor, nan=0.0)
        self.returns_tensor = np.nan_to_num(self.returns_tensor, nan=0.0)

        # === 数据质量验证：检测异常收益率 ===
        self._validate_returns_quality()

        print(
            f"🚀 向量化引擎初始化完成: {self.n_dates}天 × {self.n_factors}因子 × {self.n_etfs}标的"
        )

    def _validate_returns_quality(self) -> None:
        """验证收益率数据质量，检测异常值

        Raises:
            ValueError: 如果发现极端异常的收益率
        """
        # 统计收益率分布
        abs_returns = np.abs(self.returns_tensor)
        max_return = np.max(abs_returns)
        mean_return = np.mean(abs_returns)
        std_return = np.std(abs_returns)

        # 检查是否有超出合理范围的收益率
        # 日收益率绝对值超过100%视为异常（已被clip限制，这里是保险检查）
        extreme_mask = abs_returns > 1.0
        n_extreme = np.sum(extreme_mask)

        if n_extreme > 0:
            raise ValueError(
                f"❌ 发现 {n_extreme} 个极端收益率（|r| > 100%），数据可能有问题。"
                f"最大绝对收益率: {max_return:.4f}"
            )

        # 检查是否有异常高的收益率（超过5个标准差）
        if max_return > mean_return + 10 * std_return and max_return > 0.5:
            import warnings

            warnings.warn(
                f"⚠️ 发现异常高的收益率: {max_return:.4f} "
                f"(均值={mean_return:.4f}, 标准差={std_return:.4f})，"
                f"可能存在数据质量问题",
                UserWarning,
            )

        # 统计信息（用于调试）
        non_zero_returns = abs_returns[abs_returns > 1e-6]
        if len(non_zero_returns) > 0:
            print(
                f"   收益率统计: 最大={max_return:.4f}, 均值={mean_return:.4f}, "
                f"标准差={std_return:.4f}, 非零率={len(non_zero_returns)/abs_returns.size:.2%}"
            )

    def compute_scores_batch(
        self, weight_matrix: np.ndarray, chunk_size: int = 2000
    ) -> np.ndarray:
        """批量计算多个权重组合的得分矩阵（分块优化，支持大规模计算）

        Args:
            weight_matrix: (n_combos, n_factors) 权重矩阵
            chunk_size: 分块大小，避免内存爆炸（自动调整）

        Returns:
            scores: (n_combos, n_dates, n_etfs) 得分张量
        """
        n_combos = weight_matrix.shape[0]

        # 自适应分块大小：大规模测试使用更小的块
        if n_combos > 20000:
            chunk_size = 1000  # 超大规模使用小分块
        elif n_combos > 10000:
            chunk_size = 2000  # 大规模使用中等分块
        else:
            chunk_size = min(chunk_size, n_combos)  # 小规模直接计算或使用默认值

        # 如果组合数较少，直接计算
        if n_combos <= chunk_size:
            scores = np.einsum("dfe,cf->cde", self.factor_tensor, weight_matrix)
            return scores

        # 分块计算以节省内存
        scores = np.empty((n_combos, self.n_dates, self.n_etfs), dtype=np.float32)

        print(f"🔧 分块计算得分: {n_combos} 组合，块大小 {chunk_size}")

        for i in range(0, n_combos, chunk_size):
            end = min(i + chunk_size, n_combos)
            chunk_weights = weight_matrix[i:end]

            # 分块 einsum: (n_dates, n_factors, n_etfs) @ (chunk_size, n_factors) -> (chunk_size, n_dates, n_etfs)
            chunk_scores = np.einsum("dfe,cf->cde", self.factor_tensor, chunk_weights)
            scores[i:end] = chunk_scores

            # 进度输出（每10个分块或每25%进度）
            if i % (chunk_size * 10) == 0 or end / n_combos >= 0.25 * (
                1 + i // (n_combos // 4)
            ):
                print(f"  已处理: {end}/{n_combos} ({end/n_combos:.1%})")

        print(f"✅ 分块计算完成")
        return scores

    def build_weights_batch(
        self, scores: np.ndarray, top_n: int, min_score: float = None
    ) -> np.ndarray:
        """批量构建目标权重（完全向量化 Top-N 选股）

        Args:
            scores: (n_combos, n_dates, n_etfs) 得分张量
            top_n: 每日持有数量
            min_score: 得分阈值

        Returns:
            weights: (n_combos, n_dates, n_etfs) 权重张量
        """
        n_combos, n_dates, n_etfs = scores.shape
        weights = np.zeros_like(scores, dtype=np.float32)

        if top_n >= n_etfs:
            # 全选，直接等权
            if min_score is None:
                weights[:, :, :] = 1.0 / n_etfs
            else:
                mask = scores >= min_score
                counts = np.sum(mask, axis=2, keepdims=True)
                weights = np.where(mask, 1.0 / np.maximum(counts, 1), 0.0)
        else:
            # 向量化 Top-N 选股：使用 argpartition + 广播
            # 将 scores reshape 为 (n_combos * n_dates, n_etfs) 以便批量处理
            scores_flat = scores.reshape(-1, n_etfs)  # (n_combos*n_dates, n_etfs)
            weights_flat = np.zeros_like(scores_flat)

            # 批量 argpartition：对每一行找 Top-N
            # argpartition 将最大的 top_n 个元素放到数组后部
            top_indices = np.argpartition(-scores_flat, top_n, axis=1)[:, :top_n]

            # 应用得分阈值（向量化）
            if min_score is not None:
                # 获取 top_n 位置的得分
                row_indices = np.arange(scores_flat.shape[0])[:, None]
                top_scores = scores_flat[row_indices, top_indices]
                valid_mask = top_scores >= min_score

                # 计算每行有效数量
                valid_counts = np.sum(valid_mask, axis=1, keepdims=True)

                # 设置权重：只对有效位置赋值
                valid_weights = np.where(
                    valid_mask, 1.0 / np.maximum(valid_counts, 1), 0.0
                )
                np.put_along_axis(weights_flat, top_indices, valid_weights, axis=1)
            else:
                # 无阈值，直接等权
                equal_weight = 1.0 / top_n
                np.put_along_axis(weights_flat, top_indices, equal_weight, axis=1)

            # reshape 回原始形状
            weights = weights_flat.reshape(n_combos, n_dates, n_etfs)

        return weights

    def run_backtest_batch(self, weights: np.ndarray) -> List[Dict[str, float]]:
        """批量运行回测（完全向量化，无Python循环）

        Args:
            weights: (n_combos, n_dates, n_etfs) 权重张量

        Returns:
            metrics: 每个组合的指标字典列表
        """
        n_combos, n_dates, n_etfs = weights.shape

        # === 1. 滞后权重（避免前视偏差）===
        # (C, D, E) -> (C, D, E)
        prev_weights = np.zeros_like(weights)
        prev_weights[:, 1:, :] = weights[:, :-1, :]

        # === 2. 组合收益 ===
        # (C, D, E) * (D, E) -> (C, D)
        gross_returns = np.sum(prev_weights * self.returns_tensor[None, :, :], axis=2)

        # === 3. 换手率 ===
        # (C, D-1, E) -> (C, D-1)
        weight_diff = np.sum(np.abs(np.diff(weights, axis=1)), axis=2)
        turnover = 0.5 * weight_diff
        # 第一天换手为0: (C, D-1) -> (C, D)
        turnover = np.pad(
            turnover, ((0, 0), (1, 0)), mode="constant", constant_values=0.0
        )

        # === 4. 净收益 ===
        net_returns = gross_returns - self.fees * turnover  # (C, D)

        # === 5. 权益曲线 ===
        equity_curve = np.cumprod(1.0 + net_returns, axis=1) * self.init_cash  # (C, D)

        # === 6. 批量计算所有指标 ===
        metrics_batch = self._compute_metrics_batch(equity_curve, net_returns, turnover)

        # === 7. 向量化字典转换 ===
        # 使用numpy数组和列表推导式完全向量化字典创建
        results = [
            {
                "annual_return": float(metrics_batch["annual_return"][c]),
                "max_drawdown": float(metrics_batch["max_drawdown"][c]),
                "sharpe": float(metrics_batch["sharpe"][c]),
                "calmar": float(metrics_batch["calmar"][c]),
                "win_rate": float(metrics_batch["win_rate"][c]),
                "turnover": float(metrics_batch["turnover"][c]),
            }
            for c in range(n_combos)
        ]

        return results

    def _compute_metrics_batch(
        self, equity_curve: np.ndarray, net_returns: np.ndarray, turnover: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """批量计算回测指标（完全向量化）

        Args:
            equity_curve: (n_combos, n_dates) 权益曲线
            net_returns: (n_combos, n_dates) 净收益率
            turnover: (n_combos, n_dates) 换手率

        Returns:
            指标字典，每个值为 (n_combos,) 数组
        """
        n_combos = equity_curve.shape[0]
        n_years = self.n_dates / 252.0

        # === 1. 年化收益率 ===
        total_return = equity_curve[:, -1] / self.init_cash - 1.0  # (C,)
        annual_return = np.where(
            n_years > 0, np.power(1.0 + total_return, 1.0 / n_years) - 1.0, 0.0
        )

        # === 2. 最大回撤 ===
        cummax = np.maximum.accumulate(equity_curve, axis=1)  # (C, D)
        drawdowns = (equity_curve - cummax) / (cummax + 1e-12)  # (C, D)
        max_drawdown = -np.min(drawdowns, axis=1)  # (C,)

        # === 3. 夏普比率 ===
        returns_mean = np.mean(net_returns, axis=1)  # (C,)
        returns_std = np.std(net_returns, axis=1, ddof=0)  # (C,)
        sharpe = np.where(
            returns_std > 1e-12, returns_mean / returns_std * np.sqrt(252), 0.0
        )

        # === 4. Calmar 比率 ===
        calmar = np.where(max_drawdown > 1e-6, annual_return / max_drawdown, 0.0)

        # === 5. 胜率 ===
        win_rate = np.mean(net_returns > 0, axis=1)  # (C,)

        # === 6. 平均换手率（年化）===
        avg_turnover = np.mean(turnover, axis=1) * 252  # (C,)

        return {
            "annual_return": annual_return,
            "max_drawdown": max_drawdown,
            "sharpe": sharpe,
            "calmar": calmar,
            "win_rate": win_rate,
            "turnover": avg_turnover,
        }


# --------------------------------------------------------------------------- #
# 权重网格生成与打分
# --------------------------------------------------------------------------- #


def generate_weight_grid_stream(
    num_factors: int,
    weight_grid: Sequence[float],
    normalize: bool = True,
    max_active_factors: Optional[int] = None,
    random_seed: Optional[int] = None,
    max_total_combos: Optional[int] = None,
    debug: bool = False,
) -> List[Tuple[float, ...]]:
    """生成权重组合（稳定排序，跨批次可复现）

    Args:
        num_factors: 组合中因子数量
        weight_grid: 每个因子的候选权重（例如 [0.0, 0.5, 1.0]）
        normalize: 是否将权重归一化到和为1
        max_active_factors: 最大非零因子数量，None表示不限制
        random_seed: 随机种子，用于可重现的抽样
        max_total_combos: 最大组合总数，None表示不限制
        debug: 是否输出调试信息

    Returns:
        权重组合列表（稳定排序，跨运行可复现）
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    max_active = max_active_factors or num_factors
    non_zero_weights = [w for w in weight_grid if w != 0]

    if not non_zero_weights:
        raise ValueError("weight_grid必须包含至少一个非零权重")

    # 纯公式计算组合数（不展开任何列表）
    def count_combos_for_active(active_count: int) -> int:
        """计算指定活跃因子数的组合数"""
        n_positions = math.comb(num_factors, active_count)
        # 活跃位置的非零权重组合数
        n_weight_combos = len(non_zero_weights) ** active_count
        return n_positions * n_weight_combos

    # 计算每个活跃度的组合数和累积概率（用于加权采样）
    active_counts = list(range(1, max_active + 1))
    combo_counts = [count_combos_for_active(ac) for ac in active_counts]
    total_valid_combos = sum(combo_counts)

    if total_valid_combos == 0:
        return []

    # 如果组合数很小，直接枚举
    if max_total_combos is None and total_valid_combos <= 50000:
        if debug:
            print(f"📊 组合数较小 ({total_valid_combos})，使用完全枚举")
        combos_set_small: Set[Tuple[float, ...]] = set()

        for active_count in active_counts:
            for active_indices in itertools.combinations(
                range(num_factors), active_count
            ):
                for weight_combo in itertools.product(
                    non_zero_weights, repeat=active_count
                ):
                    weights = [0.0] * num_factors
                    for pos, w in zip(active_indices, weight_combo):
                        weights[pos] = w

                    weights_arr = np.array(weights, dtype=float)
                    weight_sum = weights_arr.sum()
                    if normalize:
                        if abs(weight_sum) < 1e-12:
                            continue  # 跳过无法归一化的权重
                        weights_arr = weights_arr / weight_sum

                    combos_set_small.add(tuple(weights_arr))

        # 🔧 稳定排序：保证跨运行可复现
        combos_sorted = sorted(list(combos_set_small))
        if debug:
            print(f"📊 权重网格生成完成: {len(combos_sorted)} 组合（已排序）")
        return combos_sorted

    # 大规模组合：加权随机采样（无偏）
    DEFAULT_SAMPLE = 10000
    if max_total_combos is None:
        target_count = min(DEFAULT_SAMPLE, total_valid_combos)
        if total_valid_combos > DEFAULT_SAMPLE:
            print(
                f"⚠️  未指定 max_total_combos，仅随机采样 {target_count}/{total_valid_combos:,} 个组合。如需全量或更大规模，请显式设置 --max-total-combos。"
            )
    else:
        if max_total_combos <= 0:
            target_count = total_valid_combos
        else:
            target_count = min(max_total_combos, total_valid_combos)

    if debug:
        print(
            f"🎯 无偏随机生成 {target_count} 个组合 (理论总数: {total_valid_combos:,})"
        )

    combos_set: Set[Tuple[float, ...]] = set()
    attempts = 0
    max_attempts = max(target_count * 200, target_count * len(non_zero_weights))

    # 预计算累积权重用于加权采样
    cumulative_weights = []
    cumsum = 0
    for count in combo_counts:
        cumsum += count
        cumulative_weights.append(cumsum)

    last_report = 0
    report_interval = max(5000, target_count // 5) if debug else float("inf")

    while len(combos_set) < target_count and attempts < max_attempts:
        attempts += 1

        # 按组合数量加权选择活跃因子数（无偏采样的关键）
        rand_val = random.random() * total_valid_combos
        active_count = active_counts[0]
        for i, cum_weight in enumerate(cumulative_weights):
            if rand_val < cum_weight:
                active_count = active_counts[i]
                break

        # 随机选择活跃因子位置
        active_indices = random.sample(range(num_factors), active_count)

        # 为活跃位置随机分配非零权重
        active_weights = [random.choice(non_zero_weights) for _ in range(active_count)]

        # 构建完整权重向量
        weights = [0.0] * num_factors
        for i, idx in enumerate(active_indices):
            weights[idx] = active_weights[i]

        # 归一化
        weights_arr = np.array(weights, dtype=float)
        weight_sum = weights_arr.sum()
        if normalize:
            if abs(weight_sum) < 1e-12:
                continue  # 跳过无法归一化的样本
            weights_arr = weights_arr / weight_sum

        combos_set.add(tuple(weights_arr))

        # 受控的进度报告
        if debug and len(combos_set) - last_report >= report_interval:
            print(f"  ⏳ 已生成 {len(combos_set):,} 个唯一组合 (尝试 {attempts:,} 次)")
            last_report = len(combos_set)

    if len(combos_set) < target_count:
        if debug:
            print(
                f"⚠️ 警告: 生成 {len(combos_set)} 个唯一组合 (目标 {target_count}，尝试 {attempts:,} 次)"
            )
        if attempts >= max_attempts:
            print(f"⚠️ 达到最大尝试次数 {max_attempts:,}，可能存在重复率过高的问题")

    # 🔧 稳定排序：保证跨运行、跨批次可复现
    combos_sorted = sorted(list(combos_set))

    if debug:
        efficiency = len(combos_set) / attempts if attempts > 0 else 0
        print(
            f"📊 随机生成完成: {len(combos_sorted)} 组合 (效率: {efficiency:.1%}，已排序)"
        )

    return combos_sorted


def generate_batch_combos(
    all_combos: List[Tuple[float, ...]], batch_size: int, batch_idx: int
) -> List[Tuple[float, ...]]:
    """分批获取权重组合

    Args:
        all_combos: 所有权重组合
        batch_size: 批次大小
        batch_idx: 批次索引（从0开始）

    Returns:
        当前批次的权重组合
    """
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, len(all_combos))
    return all_combos[start_idx:end_idx]


def build_score_matrix(
    normalized_panel: pd.DataFrame,
    factors: Sequence[str],
    weights: Sequence[float],
) -> pd.DataFrame:
    """生成多因子综合得分矩阵 (date x symbol)。"""
    assert len(factors) == len(weights)

    weighted = pd.Series(0.0, index=normalized_panel.index)
    for factor, weight in zip(factors, weights):
        if weight == 0:
            continue
        weighted = weighted.add(normalized_panel[factor] * weight, fill_value=0.0)

    scores = weighted.unstack(level="symbol")
    scores = scores.sort_index()
    return scores


def build_target_weights(
    scores: pd.DataFrame,
    top_n: int,
    min_score: float | None = None,
) -> pd.DataFrame:
    """根据得分构建目标权重，等权持有 Top-N。

    Args:
        scores: (date x symbol) 综合得分矩阵
        top_n: 每日持有的 ETF 数量
        min_score: 过滤门槛（None 表示不设门槛）
    """
    ranks = scores.rank(axis=1, ascending=False, method="first")
    selection = ranks <= top_n

    if min_score is not None:
        selection &= scores >= min_score

    weights = selection.astype(float)
    weights = weights.div(weights.sum(axis=1), axis=0).fillna(0.0)
    return weights


def build_target_weights_multi(
    scores: pd.DataFrame,
    top_n_list: List[int],
    min_score_list: List[float | None],
    debug: bool = False,
) -> List[Tuple[int, float | None, pd.DataFrame]]:
    """批量构建多个Top-N和min_score组合的目标权重

    Args:
        scores: (date x symbol) 综合得分矩阵
        top_n_list: Top-N候选值列表
        min_score_list: min_score候选值列表
        debug: 是否输出调试信息

    Returns:
        (top_n, min_score, weights) 元组列表
    """
    results: List[Tuple[int, float | None, pd.DataFrame]] = []

    for top_n in top_n_list:
        for min_score in min_score_list:
            weights = build_target_weights(scores, top_n, min_score)
            results.append((top_n, min_score, weights))

    if debug:
        print(f"🎯 生成 {len(results)} 个Top-N组合: {top_n_list} x {min_score_list}")
    return results


# --------------------------------------------------------------------------- #
# 回测执行
# --------------------------------------------------------------------------- #


def run_backtest_safe(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    freq: str = "1D",
    init_cash: float = 1_000_000.0,
    fees: float = 0.001,
    check_data_quality: bool = True,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """基于目标权重运行安全回测（增强错误处理）

    Args:
        prices: 价格矩阵 (date x symbol)
        weights: 权重矩阵 (date x symbol)
        freq: 时间频率
        init_cash: 初始资金
        fees: 交易费用
        check_data_quality: 是否检查数据质量

    Returns:
        (equity_curve, net_returns, turnover)
    """
    if check_data_quality:
        # 数据质量检查
        if weights.isna().all().all():
            raise ValueError("权重矩阵全为NaN")
        if (weights.sum(axis=1) == 0).all():
            raise ValueError("所有日期权重和为0")
        if prices.isna().all().all():
            raise ValueError("价格矩阵全为NaN")

    # 数据对齐
    aligned_prices = prices.sort_index().reindex(weights.index).ffill()
    aligned_weights = weights.reindex(aligned_prices.index).fillna(0.0)

    # 检查是否有足够的有效数据
    if aligned_prices.dropna().shape[0] < 10:
        raise ValueError("有效数据点少于10个，无法进行回测")

    asset_returns = aligned_prices.pct_change().fillna(0.0)
    prev_weights = aligned_weights.shift().fillna(0.0)

    gross_returns = (prev_weights * asset_returns).sum(axis=1)

    # 近似交易成本：每日权重变化的一半作为换手比例
    weight_diff = aligned_weights.diff().abs().sum(axis=1).fillna(0.0)
    turnover = 0.5 * weight_diff
    net_returns = gross_returns - fees * turnover

    # 检查收益序列是否有效
    if net_returns.abs().sum() == 0:
        raise ValueError("收益序列全为0")

    equity_curve = (1.0 + net_returns).cumprod()
    equity_curve = equity_curve / equity_curve.iloc[0] * init_cash

    return equity_curve, net_returns, turnover


def run_batch_backtest(
    prices: pd.DataFrame,
    weights_list: List[pd.DataFrame],
    freq: str = "1D",
    init_cash: float = 1_000_000.0,
    fees: float = 0.001,
    parallel: bool = False,
    max_workers: int = 4,
) -> List[Tuple[pd.Series, pd.Series, pd.Series]]:
    """批量运行回测（支持并行）

    Args:
        prices: 价格矩阵
        weights_list: 权重矩阵列表
        freq: 时间频率
        init_cash: 初始资金
        fees: 交易费用
        parallel: 是否并行执行
        max_workers: 最大工作线程数

    Returns:
        (equity_curve, net_returns, turnover) 元组列表
    """

    def _single_backtest(
        weights: pd.DataFrame,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        return run_backtest_safe(prices, weights, freq, init_cash, fees)

    if not parallel:
        # 串行执行
        results = []
        for weights in tqdm(weights_list, desc="回测进度"):
            try:
                result = _single_backtest(weights)
                results.append(result)
            except Exception as e:
                print(f"⚠️ 回测失败: {e}")
                results.append((pd.Series(), pd.Series(), pd.Series()))
        return results

    # 并行执行
    results = [None] * len(weights_list)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_single_backtest, weights): idx
            for idx, weights in enumerate(weights_list)
        }

        for future in tqdm(
            as_completed(future_to_idx), total=len(weights_list), desc="并行回测"
        ):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results[idx] = result
            except Exception as e:
                print(f"⚠️ 回测失败 (idx={idx}): {e}")
                results[idx] = (pd.Series(), pd.Series(), pd.Series())

    return results


def evaluate_portfolio(
    equity_curve: pd.Series,
    net_returns: pd.Series,
    turnover: pd.Series,
    freq: str,
) -> Dict[str, float]:
    """抽取关键绩效指标。"""
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0
    periods = len(net_returns)
    periods_per_year = 252
    annual_return = (1.0 + total_return) ** (periods_per_year / periods) - 1.0
    sharpe = (
        net_returns.mean() / net_returns.std() * np.sqrt(periods_per_year)
        if net_returns.std() > 0
        else np.nan
    )
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    max_drawdown = drawdown.min()
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else np.nan
    turnover_ratio = float(turnover.sum())
    if _HAS_VECTORBT:
        pf = vbt.Portfolio.from_holding(
            close=equity_curve / equity_curve.iloc[0],
            init_cash=1.0,
            freq=freq,
        )
        stats = pf.stats()
        sharpe_vbt = stats.get("Sharpe Ratio", np.nan)
        if pd.notna(sharpe_vbt):
            sharpe = float(sharpe_vbt)
        calmar_vbt = stats.get("Calmar Ratio", np.nan)
        if pd.notna(calmar_vbt):
            calmar = float(calmar_vbt)
        total_return_vbt = stats.get("Total Return [%]", np.nan)
        if pd.notna(total_return_vbt):
            total_return = float(total_return_vbt) / 100.0
        annual_return_vbt = stats.get("CAGR [%]", np.nan)
        if pd.notna(annual_return_vbt):
            annual_return = float(annual_return_vbt) / 100.0
        max_drawdown_vbt = stats.get("Max Drawdown [%]", np.nan)
        if pd.notna(max_drawdown_vbt):
            max_drawdown = float(max_drawdown_vbt) / 100.0

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "calmar": calmar,
        "win_rate": float((net_returns > 0).mean()),
        "turnover": turnover_ratio,
    }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="多因子轮动暴力枚举回测脚手架 (增强版)"
    )

    # === 配置文件参数 ===
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML配置文件路径（优先级高于CLI参数）",
    )

    # === 数据加载参数 ===
    parser.add_argument(
        "--factor-panel",
        default="factor_output/etf_rotation/panel_optimized_v2_20200102_20251014.parquet",
        help="因子面板路径（MultiIndex parquet）",
    )
    parser.add_argument(
        "--data-dir",
        default="raw/ETF/daily",
        help="原始ETF行情目录（包含 *.parquet）",
    )
    parser.add_argument(
        "--top-factors-json",
        type=str,
        default=None,
        help="因子排序JSON文件路径，支持通配符，如：production_factor_results/top_factors_*.json",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="从因子排序文件中选择的Top K因子数量",
    )
    parser.add_argument(
        "--factors",
        nargs="+",
        default=None,
        help="手动指定参与组合的因子列表（优先于--top-factors-json）",
    )

    # === 权重网格参数 ===
    parser.add_argument(
        "--weight-grid",
        nargs="+",
        type=float,
        default=[0.0, 0.5, 1.0],
        help="每个因子候选权重（将自动归一化）",
    )
    parser.add_argument(
        "--max-active-factors",
        type=int,
        default=None,
        help="最大非零因子数量限制",
    )
    parser.add_argument(
        "--max-total-combos",
        type=int,
        default=None,
        help="最大组合总数限制，用于抽样",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="随机种子，用于可重现的抽样",
    )

    # === Top-N和筛选参数 ===
    parser.add_argument(
        "--top-n-list",
        nargs="+",
        type=int,
        default=[5],
        help="Top-N候选值列表，支持多个值如：3 5 8",
    )
    parser.add_argument(
        "--min-score-list",
        nargs="+",
        type=float,
        default=[None],
        help="得分阈值列表，支持多个值如：0.0 0.1",
    )

    # === 分批执行参数 ===
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="批次大小，用于大规模组合的分批处理",
    )
    parser.add_argument(
        "--batch-idx",
        type=int,
        default=0,
        help="批次索引，从0开始，用于断点续跑",
    )
    parser.add_argument(
        "--sanity-run",
        action="store_true",
        help="Sanity Run模式，仅运行少量组合验证设置",
    )

    # === 并发优化参数 ===
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="启用并行回测",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="最大工作线程数",
    )
    parser.add_argument(
        "--use-vectorized",
        action="store_true",
        default=True,
        help="使用向量化回测引擎（默认开启，100x速度提升）",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="多进程并行数（推荐M4 Pro使用4，单进程设1）",
    )

    # === 回测参数 ===
    parser.add_argument(
        "--fees",
        nargs="+",
        type=float,
        default=[0.001],
        help="单边交易费用列表（支持成本敏感性分析，如：0.001 0.002 0.003）",
    )
    parser.add_argument(
        "--init-cash",
        type=float,
        default=1_000_000.0,
        help="初始资金",
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="1D",
        help="时间频率（用于年化计算，默认日频）",
    )
    parser.add_argument(
        "--norm-method",
        choices=["zscore", "rank"],
        default="zscore",
        help="截面标准化方式",
    )

    # === 输出参数 ===
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出路径，支持CSV文件或目录（自动生成时间戳子目录）",
    )
    parser.add_argument(
        "--top-k-results",
        type=int,
        default=None,
        help="仅保留夏普最高的前K个结果写入输出和检查点（默认保留全部）",
    )
    parser.add_argument(
        "--keep-metrics-json",
        action="store_true",
        help="保存详细指标JSON",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="调试模式，输出详细日志",
    )

    return parser.parse_args()


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """从 YAML 文件加载配置

    Args:
        config_path: YAML 配置文件路径

    Returns:
        配置字典
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config.get("parameters", {})


def merge_config_with_args(args: argparse.Namespace) -> argparse.Namespace:
    """合并 YAML 配置和 CLI 参数（YAML 优先级更高）

    Args:
        args: CLI 参数

    Returns:
        合并后的参数
    """
    if args.config is None:
        return args

    config = load_config_from_yaml(args.config)

    # YAML 键名映射到 argparse 参数名（下划线转连字符）
    for key, value in config.items():
        arg_name = key.replace("-", "_")
        if hasattr(args, arg_name):
            setattr(args, arg_name, value)

    return args


def create_output_directory(base_path: str, timestamp: str) -> Tuple[Path, Path]:
    """创建带时间戳的输出目录"""
    if base_path is None:
        # 默认输出到results目录 (相对于脚本所在目录)
        base_path = "results/vbt_multifactor"

    # 获取脚本所在目录作为基准路径
    script_dir = Path(__file__).parent

    # 如果是绝对路径，直接使用；否则相对于脚本目录
    base_path = Path(base_path)
    if not base_path.is_absolute():
        base_path = script_dir / base_path

    # 如果以.csv结尾，则视为文件路径，返回父目录
    if base_path.suffix == ".csv":
        output_dir = base_path.parent
        csv_file = base_path
    else:
        # 创建带时间戳的子目录
        output_dir = base_path / f"run_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_file = output_dir / "results.csv"

    return output_dir, csv_file


def save_checkpoint(results: List[Dict], checkpoint_path: Path) -> None:
    """保存检查点文件"""
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def run_backtest_worker(
    combos_chunk: List[Tuple[float, ...]],
    global_start_idx: int,
    factors: List[str],
    normalized_panel: pd.DataFrame,
    price_pivot: pd.DataFrame,
    top_n_list: List[int],
    min_score_list: List[float],
    fees: float,
    init_cash: float,
    freq: str,
) -> List[Dict[str, float]]:
    """多进程worker函数：处理一个权重组合块

    Args:
        combos_chunk: 权重组合块
        global_start_idx: 该块在全局组合列表中的起始索引
        factors: 因子列表
        normalized_panel: 标准化因子面板
        price_pivot: 价格矩阵
        top_n_list: Top-N候选值列表
        min_score_list: min_score候选值列表
        fees: 交易费用
        init_cash: 初始资金
        freq: 时间频率

    Returns:
        结果列表
    """
    # 初始化引擎（每个worker独立初始化）
    engine = VectorizedBacktestEngine(
        normalized_panel=normalized_panel,
        price_pivot=price_pivot,
        factors=factors,
        fees=fees,
        init_cash=init_cash,
        freq=freq,
    )

    # 转为numpy数组
    weight_matrix = np.array(combos_chunk, dtype=np.float32)

    # 批量计算得分
    all_scores = engine.compute_scores_batch(weight_matrix)

    # 对每个Top-N和min_score组合进行回测
    results = []
    for top_n in top_n_list:
        for min_score in min_score_list:
            # 批量构建目标权重
            target_weights = engine.build_weights_batch(
                all_scores, top_n=top_n, min_score=min_score
            )

            # 批量运行回测
            batch_metrics = engine.run_backtest_batch(target_weights)

            # 构建结果条目（使用全局索引）
            for local_idx, (weights, metrics) in enumerate(
                zip(combos_chunk, batch_metrics)
            ):
                result_entry = {
                    "combo_idx": global_start_idx + local_idx,  # 🔧 全局索引
                    "weights": tuple(f"{w:.3f}" for w in weights),
                    "top_n": top_n,
                    "min_score": min_score,
                    **metrics,
                }

                # 过滤无效结果
                if not (np.isnan(metrics["sharpe"]) or metrics["sharpe"] < -10):
                    results.append(result_entry)

    return results


def main() -> None:
    start_time = time.time()
    args = parse_args()

    # 合并 YAML 配置（如果提供）
    args = merge_config_with_args(args)

    # Sanity Run模式调整
    if args.sanity_run:
        args.max_total_combos = min(args.max_total_combos or 1000, 1000)
        args.batch_size = min(args.batch_size, 100)
        print("🚀 Sanity Run模式已启用，限制组合数量")

    # 创建时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # === 1. 因子选择 ===
    if args.factors is None:
        if args.top_factors_json is None:
            raise ValueError("必须指定 --factors 或 --top-factors-json")
        factors = load_top_factors_from_json(args.top_factors_json, args.top_k)
    else:
        factors = args.factors
        print(f"📋 使用手动指定因子: {', '.join(factors)}")

    # 安全验证：过滤黑名单因子
    factors = validate_factors_safety(list(factors))

    # 因子名称映射：将筛选结果映射到实际面板列名
    panel_path = Path(args.factor_panel)
    factors = map_factor_names_to_panel(factors, panel_path)

    # === 2. 数据加载 ===
    print("📊 开始加载数据...")
    data_dir = Path(args.data_dir)

    factor_panel = load_factor_panel(panel_path, factors)
    normalized_panel = normalize_factors(factor_panel, method=args.norm_method)
    price_pivot = load_price_pivot(data_dir)

    print(f"✅ 数据加载完成: {len(factors)}个因子, {len(price_pivot)}个交易日")

    # === 3. 权重网格生成 ===
    if args.debug:
        print("🎯 生成权重网格...")
    all_weight_combos = generate_weight_grid_stream(
        len(factors),
        args.weight_grid,
        normalize=True,
        max_active_factors=args.max_active_factors,
        random_seed=args.random_seed,
        max_total_combos=args.max_total_combos,
        debug=args.debug,
    )

    if not all_weight_combos:
        raise SystemExit("未生成任何权重组合，请检查参数设置。")

    # === 4. 分批处理 ===
    total_combos = len(all_weight_combos)
    batch_combos = generate_batch_combos(
        all_weight_combos, args.batch_size, args.batch_idx
    )

    print(f"📦 批次 {args.batch_idx}: 处理 {len(batch_combos)}/{total_combos} 组合")

    # === 5. 创建输出目录 ===
    output_dir, csv_file = create_output_directory(args.output, timestamp)

    # === 6. 批量向量化回测（支持费率敏感性分析）===
    all_results: List[Dict[str, float]] = []

    # 🔧 计算当前batch在全局组合中的起始索引（考虑batch_idx）
    global_combo_offset = args.batch_idx * args.batch_size

    total_tasks = (
        len(batch_combos)
        * len(args.top_n_list)
        * len(args.min_score_list)
        * len(args.fees)
    )
    print(f"🎯 总任务数: {total_tasks} (权重组合 × Top-N × min_score × fees)")
    print(
        f"🔢 全局combo_idx范围: [{global_combo_offset}, {global_combo_offset + len(batch_combos) - 1}]"
    )
    print(f"💰 费率列表: {args.fees}")

    # 外层循环：遍历费率列表
    for fee_idx, current_fee in enumerate(args.fees):
        print(f"\n{'='*60}")
        print(f"💰 费率 {fee_idx+1}/{len(args.fees)}: {current_fee:.4f}")
        print(f"{'='*60}")

        results: List[Dict[str, float]] = []
        failed_count = 0

        if args.num_workers > 1:
            # === 多进程并行模式 ===
            print(f"🚀 启用多进程并行: {args.num_workers} workers")

            # 将组合分块，记录每块的全局起始索引（加上batch偏移）
            chunk_size = max(1, len(batch_combos) // args.num_workers)
            chunks_with_idx = []
            for i in range(0, len(batch_combos), chunk_size):
                chunk = batch_combos[i : i + chunk_size]
                global_start = (
                    global_combo_offset + i
                )  # 🔧 全局索引 = batch偏移 + 块内偏移
                chunks_with_idx.append((chunk, global_start))

            print(f"📦 分为 {len(chunks_with_idx)} 个块，每块约 {chunk_size} 个组合")

            # 准备worker参数（不包含chunk和global_start_idx）
            worker_fn = partial(
                run_backtest_worker,
                factors=factors,
                normalized_panel=normalized_panel,
                price_pivot=price_pivot,
                top_n_list=args.top_n_list,
                min_score_list=args.min_score_list,
                fees=current_fee,
                init_cash=args.init_cash,
                freq=args.freq,
            )

            # 多进程执行
            with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                # 提交任务时传入chunk和global_start_idx
                futures = [
                    executor.submit(worker_fn, chunk, global_start)
                    for chunk, global_start in chunks_with_idx
                ]

                with tqdm(
                    total=len(futures),
                    desc=f"批次{args.batch_idx}-费率{current_fee:.4f}",
                ) as pbar:
                    for future in as_completed(futures):
                        try:
                            chunk_results = future.result()
                            results.extend(chunk_results)
                            pbar.update(1)
                        except Exception as e:
                            if args.debug:
                                print(f"\n⚠️ Worker失败: {e}")
                            failed_count += (
                                chunk_size
                                * len(args.top_n_list)
                                * len(args.min_score_list)
                            )
                            pbar.update(1)

            # 补充batch_idx、factors和fee字段
            for result in results:
                result["batch_idx"] = args.batch_idx
                result["factors"] = factors
                result["timestamp"] = timestamp
                result["fee"] = current_fee

        else:
            # === 单进程模式 ===
            print("⚡ 初始化向量化回测引擎...")
            engine = VectorizedBacktestEngine(
                normalized_panel=normalized_panel,
                price_pivot=price_pivot,
                factors=factors,
                fees=current_fee,
                init_cash=args.init_cash,
                freq=args.freq,
            )

            print(f"🚀 开始向量化回测 {len(batch_combos)} 个组合...")

            # 将权重组合转为numpy数组
            weight_matrix = np.array(batch_combos, dtype=np.float32)

            # 批量计算所有组合的得分矩阵
            print("📊 批量计算因子得分...")
            all_scores = engine.compute_scores_batch(weight_matrix)

            task_count = (
                len(batch_combos) * len(args.top_n_list) * len(args.min_score_list)
            )
            with tqdm(
                total=task_count, desc=f"批次{args.batch_idx}-费率{current_fee:.4f}"
            ) as pbar:
                for top_n in args.top_n_list:
                    for min_score in args.min_score_list:
                        try:
                            # 批量构建目标权重
                            target_weights = engine.build_weights_batch(
                                all_scores, top_n=top_n, min_score=min_score
                            )

                            # 批量运行回测
                            batch_metrics = engine.run_backtest_batch(target_weights)

                            # 构建结果条目（使用全局索引）
                            for local_idx, (weights, metrics) in enumerate(
                                zip(batch_combos, batch_metrics)
                            ):
                                result_entry = {
                                    "batch_idx": args.batch_idx,
                                    "combo_idx": global_combo_offset
                                    + local_idx,  # 🔧 全局索引
                                    "weights": tuple(f"{w:.3f}" for w in weights),
                                    "factors": factors,
                                    "top_n": top_n,
                                    "min_score": min_score,
                                    "timestamp": timestamp,
                                    "fee": current_fee,
                                    **metrics,
                                }

                                # 过滤无效结果
                                if not (
                                    np.isnan(metrics["sharpe"])
                                    or metrics["sharpe"] < -10
                                ):
                                    results.append(result_entry)
                                else:
                                    failed_count += 1

                                pbar.update(1)

                        except Exception as e:
                            if args.debug:
                                print(
                                    f"\n⚠️ Top-N={top_n}, min_score={min_score} 批次失败: {e}"
                                )
                            failed_count += len(batch_combos)
                            pbar.update(len(batch_combos))
                            continue

        # 合并当前费率的结果到总结果
        all_results.extend(results)
        print(f"✅ 费率 {current_fee:.4f} 完成: {len(results)} 个有效结果")

    # === 8. 结果汇总 ===
    print(f"\n{'='*60}")
    print(f"📈 所有费率回测完成: {len(all_results)}个有效结果")
    print(f"{'='*60}")

    if not all_results:
        print("❌ 没有有效的回测结果")
        return

    result_df = pd.DataFrame(all_results)
    result_df = result_df.sort_values("sharpe", ascending=False)

    # === 9. 输出结果 ===
    execution_time = time.time() - start_time

    print("=" * 100)
    print(f"🏆 暴力枚举回测结果 - 批次 {args.batch_idx} (按夏普降序)")
    print("=" * 100)
    print(f"执行时间: {execution_time:.2f}秒 | 有效组合: {len(all_results)}")
    print(
        f"参数范围: {len(args.top_n_list)}个Top-N x {len(args.min_score_list)}个min-score x {len(args.fees)}个费率"
    )
    print("-" * 100)

    display_cols = [
        "weights",
        "top_n",
        "fee",
        "annual_return",
        "max_drawdown",
        "sharpe",
        "calmar",
        "win_rate",
        "turnover",
    ]

    top_results = result_df.head(20)[display_cols]
    print(top_results.to_string(index=False, float_format=lambda x: f"{x: .4f}"))

    # 根据参数裁剪输出规模
    filtered_df = result_df
    if args.top_k_results is not None:
        if args.top_k_results <= 0:
            print("⚠️ top_k_results <= 0，忽略该参数，保留全部结果。")
        else:
            keep_count = min(args.top_k_results, len(result_df))
            filtered_df = result_df.head(keep_count).copy()
            if keep_count < len(result_df):
                print(
                    f"📉 仅保留夏普最高的前 {keep_count} 个结果用于输出（原始 {len(result_df)} 个）。"
                )

    # 保存CSV结果 (默认保存)
    filtered_df.to_csv(csv_file, index=False)
    if args.output:
        print(f"\n📁 结果已保存到: {csv_file}")
    else:
        print(f"\n📁 结果已保存到默认路径: {csv_file}")

    # 保存详细JSON（如果需要）
    if args.keep_metrics_json:
        json_file = output_dir / f"detailed_results_{timestamp}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "execution_info": {
                        "timestamp": timestamp,
                        "execution_time": execution_time,
                        "batch_idx": args.batch_idx,
                        "batch_size": args.batch_size,
                        "total_combos": total_combos,
                        "valid_results": len(results),
                        "failed_count": failed_count,
                    },
                    "parameters": {
                        "factors": factors,
                        "weight_grid": args.weight_grid,
                        "top_n_list": args.top_n_list,
                        "min_score_list": args.min_score_list,
                        "fees": args.fees,
                        "norm_method": args.norm_method,
                    },
                    "results": filtered_df.to_dict("records"),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"📊 详细结果已保存到: {json_file}")

    # 保存检查点 (默认保存)
    checkpoint_file = output_dir / f"checkpoint_batch_{args.batch_idx}.json"
    save_checkpoint(filtered_df.to_dict("records"), checkpoint_file)
    if args.output:
        print(f"💾 检查点已保存到: {checkpoint_file}")
    else:
        print(f"💾 检查点已保存到默认路径: {checkpoint_file}")

    print(f"\n✅ 批次 {args.batch_idx} 完成!")
    if args.batch_idx * args.batch_size + len(batch_combos) < total_combos:
        next_batch = args.batch_idx + 1
        print(f"🔄 继续下一批次: python {__file__} --batch-idx {next_batch}")
    else:
        print("🎉 所有批次已完成!")


def run_regression_tests() -> None:
    """回归测试：验证关键修复是否生效（真实引擎测试）

    测试项：
    1. 收益率计算：真实引擎验证无爆炸收益
    2. 批量回测：实际跑完整流程
    3. 跨batch编号：验证 batch_idx > 0 时 combo_idx 正确
    """
    print("🧪 运行回归测试（真实引擎）...")

    # === 测试1：收益率计算（真实引擎）===
    print("\n[测试1] 收益率计算安全性（含异常数据）")

    # 构造有缺失值的价格数据
    test_dates = pd.date_range("2020-01-01", periods=50)
    test_prices = pd.DataFrame(
        {
            "ETF_A": [np.nan] * 10
            + list(10.0 + np.random.randn(40) * 0.5),  # 前10日NaN（未上市）
            "ETF_B": 100.0 + np.random.randn(50) * 2.0,  # 正常波动
            "ETF_C": [50.0] * 20
            + [0.0]
            + list(52.0 + np.random.randn(29) * 1.0),  # 第21日价格为0（异常）
        },
        index=test_dates,
    )

    # 构造因子面板
    test_factors = ["FACTOR_A", "FACTOR_B"]
    symbols = ["ETF_A", "ETF_B", "ETF_C"]
    test_panel = pd.DataFrame(
        {
            "FACTOR_A": np.random.randn(len(symbols) * len(test_dates)),
            "FACTOR_B": np.random.randn(len(symbols) * len(test_dates)),
        },
        index=pd.MultiIndex.from_product(
            [symbols, test_dates], names=["symbol", "date"]
        ),
    )

    try:
        engine = VectorizedBacktestEngine(
            normalized_panel=test_panel,
            price_pivot=test_prices,
            factors=test_factors,
            fees=0.001,
            init_cash=1_000_000.0,
            freq="1D",
        )

        # 检查收益率范围
        max_abs_return = np.max(np.abs(engine.returns_tensor))
        assert max_abs_return <= 1.0, f"❌ 爆炸收益率: {max_abs_return}"
        assert not np.any(np.isnan(engine.returns_tensor)), "❌ 收益率中存在NaN"
        assert not np.any(np.isinf(engine.returns_tensor)), "❌ 收益率中存在Inf"

        print(f"   ✅ 收益率范围正常: 最大={max_abs_return:.4f}")

    except Exception as e:
        print(f"   ❌ 收益率测试失败: {e}")
        raise

    # === 测试2：完整回测流程 ===
    print("\n[测试2] 完整回测流程（10组合 × 2Top-N）")

    # 生成测试权重组合
    test_combos = [
        (1.0, 0.0),
        (0.0, 1.0),
        (0.5, 0.5),
        (0.7, 0.3),
        (0.3, 0.7),
        (0.8, 0.2),
        (0.2, 0.8),
        (0.6, 0.4),
        (0.4, 0.6),
        (0.9, 0.1),
    ]

    weight_matrix = np.array(test_combos, dtype=np.float32)

    try:
        # 批量计算得分
        scores = engine.compute_scores_batch(weight_matrix)
        assert scores.shape == (len(test_combos), engine.n_dates, engine.n_etfs)

        # 批量构建权重并回测
        for top_n in [2, 3]:
            target_weights = engine.build_weights_batch(scores, top_n=top_n)
            metrics = engine.run_backtest_batch(target_weights)

            assert len(metrics) == len(
                test_combos
            ), f"结果数量不匹配: {len(metrics)} vs {len(test_combos)}"

            # 检查指标合理性
            for i, m in enumerate(metrics):
                assert -10 < m["sharpe"] < 10, f"组合{i} Sharpe异常: {m['sharpe']}"
                assert (
                    -1 < m["annual_return"] < 5
                ), f"组合{i} 年化收益异常: {m['annual_return']}"
                assert (
                    0 <= m["max_drawdown"] <= 1
                ), f"组合{i} 最大回撤异常: {m['max_drawdown']}"

        print(f"   ✅ 完整流程正常: 10组合 × 2Top-N = 20结果")

    except Exception as e:
        print(f"   ❌ 回测流程失败: {e}")
        raise

    # === 测试3：真实多进程路径测试 ===
    print("\n[测试3] 真实多进程路径（2 workers × 5组合）")

    try:
        # 生成10个测试组合
        test_combos_mp = [
            (1.0, 0.0),
            (0.0, 1.0),
            (0.5, 0.5),
            (0.7, 0.3),
            (0.3, 0.7),
            (0.8, 0.2),
            (0.2, 0.8),
            (0.6, 0.4),
            (0.4, 0.6),
            (0.9, 0.1),
        ]

        # 模拟跨batch场景：batch_idx=1（全局起始索引100）
        global_offset_test = 100

        # 分成2个chunk
        chunk_size = 5
        chunks_with_idx = [
            (test_combos_mp[0:5], global_offset_test + 0),
            (test_combos_mp[5:10], global_offset_test + 5),
        ]

        # 使用真实worker函数
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from functools import partial

        worker_fn = partial(
            run_backtest_worker,
            factors=test_factors,
            normalized_panel=test_panel,
            price_pivot=test_prices,
            top_n_list=[2],
            min_score_list=[None],
            fees=0.001,
            init_cash=1_000_000.0,
            freq="1D",
        )

        all_results = []
        try:
            with ProcessPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(worker_fn, chunk, global_start)
                    for chunk, global_start in chunks_with_idx
                ]

                for future in as_completed(futures):
                    chunk_results = future.result()
                    all_results.extend(chunk_results)
        except PermissionError:
            print("   ⚠️ 系统限制无法启动多进程，回退到单进程验证。")
            for chunk, global_start in chunks_with_idx:
                all_results.extend(worker_fn(chunk, global_start))

        # 验证结果
        assert len(all_results) == 10, f"结果数量错误: {len(all_results)}"

        # 提取combo_idx
        combo_ids = [r["combo_idx"] for r in all_results]

        # 检查唯一性
        assert (
            len(set(combo_ids)) == 10
        ), f"❌ combo_idx不唯一: {len(set(combo_ids))} unique"

        # 检查范围正确（应该是[100, 109]）
        expected_range = set(range(100, 110))
        actual_range = set(combo_ids)
        assert (
            expected_range == actual_range
        ), f"❌ combo_idx范围错误: 期望{expected_range}, 实际{actual_range}"

        print(f"   ✅ 多进程路径正常: 2 workers × 5组合，combo_idx=[100, 109]全部唯一")

    except Exception as e:
        print(f"   ❌ 多进程测试失败: {e}")
        raise

    # === 测试4：跨batch可复现性 ===
    print("\n[测试4] 权重生成稳定性（跨运行可复现）")

    try:
        # 生成两次相同参数的权重组合
        combos_1 = generate_weight_grid_stream(
            num_factors=3,
            weight_grid=[0.0, 0.5, 1.0],
            random_seed=42,
            max_total_combos=100,
            debug=False,
        )

        combos_2 = generate_weight_grid_stream(
            num_factors=3,
            weight_grid=[0.0, 0.5, 1.0],
            random_seed=42,
            max_total_combos=100,
            debug=False,
        )

        # 检查完全一致（包括顺序）
        assert len(combos_1) == len(
            combos_2
        ), f"数量不一致: {len(combos_1)} vs {len(combos_2)}"

        for i, (c1, c2) in enumerate(zip(combos_1, combos_2)):
            assert c1 == c2, f"索引{i}组合不一致: {c1} vs {c2}"

        print(
            f"   ✅ 权重生成可复现: 2次运行产生{len(combos_1)}个完全一致的组合（含顺序）"
        )

    except Exception as e:
        print(f"   ❌ 可复现性测试失败: {e}")
        raise

    print("\n✅ 所有回归测试通过！真实引擎+多进程路径验证完成。")


if __name__ == "__main__":
    # 如果环境变量REGRESSION_TEST=1，运行回归测试
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_regression_tests()
    else:
        main()
