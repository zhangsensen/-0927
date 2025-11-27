#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一回测引擎
整合production、profit、experimental三种回测模式
"""

import os
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from strategies.backtest.production_backtest import backtest_no_lookahead


@dataclass
class BacktestResult:
    """统一回测结果容器"""

    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    volatility: float
    win_rate: Optional[float] = None
    calmar_ratio: Optional[float] = None
    mode: str = "production"
    daily_returns: Optional[pd.Series] = None
    nav_curve: Optional[pd.Series] = None
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestRequest:
    """回测输入参数"""

    factors_data: np.ndarray
    returns: np.ndarray
    etf_names: Sequence[str]
    factor_indices: Sequence[int]
    rebalance_freq: int
    lookback_window: int = 252
    position_size: int = 5
    initial_capital: float = 1_000_000.0
    commission_rate: float = 0.00005
    commission_min: float = 0.0
    etf_stop_loss: float = 0.0  # V10: 单一ETF止损阈值（0=禁用）


@contextmanager
def _temporary_env(values: Dict[str, Optional[str]]):
    """临时切换环境变量，结束后自动恢复"""

    originals: Dict[str, Optional[str]] = {k: os.environ.get(k) for k in values}
    try:
        for key, val in values.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val
        yield
    finally:
        for key, val in originals.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val


def run_backtest(
    request: BacktestRequest,
    mode: str = "production",
    slippage_bps: int = 2,
    enable_lookahead_check: bool = True,
    rebalance_days: Optional[int] = None,
    position_size: Optional[int] = None,
    env_overrides: Optional[Dict[str, Optional[str]]] = None,
    **kwargs: Any,
) -> BacktestResult:
    """统一入口，根据模式调用具体实现"""

    effective_req = replace(
        request,
        rebalance_freq=rebalance_days if rebalance_days is not None else request.rebalance_freq,
        position_size=position_size if position_size is not None else request.position_size,
    )

    env_values: Dict[str, Optional[str]] = {"RB_ENFORCE_NO_LOOKAHEAD": "1" if enable_lookahead_check else "0"}
    if env_overrides:
        env_values.update(env_overrides)

    with _temporary_env(env_values):
        if mode == "production":
            payload = _run_production_mode(effective_req)
            return _to_result(payload, mode)
        if mode == "profit":
            payload = _run_profit_mode(effective_req, slippage_bps=slippage_bps)
            return _to_result(
                payload,
                mode,
                key_map={
                    "annual_return": "annual_ret_net",
                    "total_return": "total_ret_net",
                    "sharpe_ratio": "sharpe_net",
                    "max_drawdown": "max_dd_net",
                    "volatility": "vol_net",
                },
                nav_key="nav_net",
                daily_key="daily_returns_net",
            )
        if mode == "experimental":
            payload = _run_experimental_mode(effective_req, **kwargs)
            return _to_result(payload, mode)
    raise ValueError(f"Unknown mode: {mode}. Must be 'production', 'profit', or 'experimental'")


def _run_production_mode(request: BacktestRequest) -> Dict[str, Any]:
    factor_idx = np.asarray(request.factor_indices, dtype=np.int64)
    factors_sel = request.factors_data[:, :, factor_idx]
    return backtest_no_lookahead(
        factors_data=factors_sel,
        returns=request.returns,
        etf_names=list(request.etf_names),
        rebalance_freq=request.rebalance_freq,
        lookback_window=request.lookback_window,
        position_size=request.position_size,
        initial_capital=request.initial_capital,
        commission_rate=request.commission_rate,
        commission_min=request.commission_min,
        factors_data_full=request.factors_data,
        factor_indices_for_cache=factor_idx,
        etf_stop_loss=request.etf_stop_loss,
    )


def _run_profit_mode(request: BacktestRequest, slippage_bps: int) -> Dict[str, Any]:
    base_payload = _run_production_mode(request)
    slippage_rate = max(0.0, float(slippage_bps) / 10_000.0)
    enriched = apply_slippage_to_nav(base_payload, slippage_rate, request.rebalance_freq)
    nav_net = enriched.get("nav_net")
    if nav_net is not None:
        nav_arr = np.asarray(nav_net, dtype=float)
        if nav_arr.size > 1:
            daily_net = nav_arr[1:] / nav_arr[:-1] - 1.0
        else:
            daily_net = np.zeros(0, dtype=float)
        enriched["daily_returns_net"] = daily_net
    return enriched


def _run_experimental_mode(request: BacktestRequest, **kwargs: Any) -> Dict[str, Any]:
    exp_req = replace(
        request,
        commission_rate=float(kwargs.get("commission_rate", 0.0)),
        commission_min=0.0,
    )
    env_settings = {
        "RB_DISABLE_IC_CACHE": "1",
        "RB_DAILY_IC_PRECOMP": "0",
        "RB_NUMBA_WARMUP": "0",
    }
    with _temporary_env(env_settings):
        return _run_production_mode(exp_req)


def _to_result(
    payload: Dict[str, Any],
    mode: str,
    key_map: Optional[Dict[str, str]] = None,
    nav_key: str = "nav",
    daily_key: str = "daily_returns",
) -> BacktestResult:
    key_map = key_map or {}
    annual_key = key_map.get("annual_return", "annual_ret")
    sharpe_key = key_map.get("sharpe_ratio", "sharpe")
    maxdd_key = key_map.get("max_drawdown", "max_dd")
    total_key = key_map.get("total_return", "total_ret")
    vol_key = key_map.get("volatility", "vol")

    nav_raw = payload.get(nav_key)
    nav_series = pd.Series(np.asarray(nav_raw, dtype=float)) if nav_raw is not None else None
    daily_raw = payload.get(daily_key)
    if daily_raw is None and nav_series is not None and len(nav_series) > 1:
        nav_vals = nav_series.to_numpy(dtype=float)
        daily_raw = nav_vals[1:] / nav_vals[:-1] - 1.0
    daily_series = pd.Series(np.asarray(daily_raw, dtype=float)) if daily_raw is not None else None

    result = BacktestResult(
        annual_return=float(payload.get(annual_key, float("nan"))),
        sharpe_ratio=float(payload.get(sharpe_key, float("nan"))),
        max_drawdown=float(payload.get(maxdd_key, float("nan"))),
        total_return=float(payload.get(total_key, float("nan"))),
        volatility=float(payload.get(vol_key, float("nan"))),
        win_rate=float(payload.get("win_rate")) if payload.get("win_rate") is not None else None,
        calmar_ratio=float(payload.get("calmar_ratio")) if payload.get("calmar_ratio") is not None else None,
        mode=mode,
        daily_returns=daily_series,
        nav_curve=nav_series,
        extras={k: v for k, v in payload.items() if k not in {
            annual_key,
            sharpe_key,
            maxdd_key,
            total_key,
            vol_key,
            nav_key,
            daily_key,
        }},
    )
    return result


def apply_slippage_to_nav(
    result: Dict[str, Any],
    slippage_rate: float,
    rebalance_freq: int,
) -> Dict[str, Any]:
    """复制盈利优先回测中的滑点后处理逻辑"""

    out = dict(result)
    if slippage_rate <= 0:
        out.update(
            {
                "final_net": out.get("final"),
                "total_ret_net": out.get("total_ret"),
                "annual_ret_net": out.get("annual_ret"),
                "sharpe_net": out.get("sharpe"),
                "max_dd_net": out.get("max_dd"),
            }
        )
        return out

    nav = np.asarray(out.get("nav", []), dtype=float)
    if nav.size < 2:
        out.update(
            {
                "final_net": out.get("final"),
                "total_ret_net": out.get("total_ret"),
                "annual_ret_net": out.get("annual_ret"),
                "sharpe_net": out.get("sharpe"),
                "max_dd_net": out.get("max_dd"),
            }
        )
        return out

    cost_amount = np.asarray(out.get("cost_amount_series", np.zeros(0)), dtype=float)
    turnover = np.asarray(out.get("turnover_series", np.zeros(0)), dtype=float)
    offsets = [i * rebalance_freq for i in range(len(turnover)) if i * rebalance_freq < (nav.size - 1)]
    nav_adj = nav.copy()
    for idx, off in enumerate(offsets):
        nav_after_commission = nav_adj[off]
        commission = cost_amount[idx] if idx < cost_amount.size else 0.0
        nav_before_cost = nav_after_commission + commission
        extra_cost = slippage_rate * turnover[idx] * nav_before_cost
        if nav_after_commission <= 0 or extra_cost <= 0:
            continue
        new_val = max(nav_after_commission - extra_cost, 0.0)
        ratio = new_val / nav_after_commission if nav_after_commission > 0 else 1.0
        nav_adj[off:] = nav_adj[off:] * ratio

    init_cap = nav_adj[0]
    daily_ret_adj = nav_adj[1:] / nav_adj[:-1] - 1.0 if nav_adj.size > 1 else np.zeros(0, dtype=float)
    total_ret_net = nav_adj[-1] / init_cap - 1 if init_cap else 0.0
    days = max(len(daily_ret_adj), 1)
    annual_ret_net = (1 + total_ret_net) ** (252 / days) - 1
    vol_net = float(np.std(daily_ret_adj)) * np.sqrt(252) if daily_ret_adj.size > 0 else 0.0
    sharpe_net = annual_ret_net / vol_net if vol_net > 0 else 0.0
    cummax = np.maximum.accumulate(nav_adj)
    dd = (nav_adj - cummax) / cummax
    max_dd_net = float(np.min(dd)) if dd.size > 0 else 0.0

    out.update(
        {
            "final_net": float(nav_adj[-1]),
            "total_ret_net": float(total_ret_net),
            "annual_ret_net": float(annual_ret_net),
            "sharpe_net": float(sharpe_net),
            "max_dd_net": float(max_dd_net),
            "vol_net": float(vol_net),
            "nav_net": nav_adj,
        }
    )
    return out
