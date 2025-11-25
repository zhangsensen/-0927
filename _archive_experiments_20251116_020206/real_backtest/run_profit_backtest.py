#!/usr/bin/env python3
"""
盈利优先回测（不改稳定仓）：在稳定回测基础上叠加“常数滑点”并可选用利润校准器排序。

功能要点
--------
1) 读取最新 WFO run_* 目录的 TopK 组合（可选 ALL）
2) 复用稳定仓数据加载/因子/横截面处理
3) 调用稳定回测 backtest_no_lookahead 获得“含佣金”的基线结果
4) 基于调仓换手序列与NAV，按常数滑点(基点)做“事后扣减校正”，得到净收益曲线与指标
5) 若存在利润校准器(results/calibrator_gbdt_profit.joblib)，则先按预测年化降序选 TopK

约束
----
- 不改稳定项目代码，仅在 experiments 中新增脚本
- 滑点为“调仓事件上的瞬时成本”，以调仓前余额为基数：extra_cost = slippage_rate * turnover * P_before_cost
- P_before_cost = NAV_at_offset + commission_value_at_offset（用 cost_amount_series 还原）
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Suppress verbose logging from run_production_backtest
logging.basicConfig(level=logging.WARNING)
logging.getLogger('real_backtest.run_production_backtest').setLevel(logging.WARNING)


def _ensure_stable_paths():
    """将稳定仓加入 sys.path，保证可导入其模块与脚本。"""
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    stable_root = repo_root / "etf_rotation_optimized"
    stable_rb = stable_root / "real_backtest"
    for p in (stable_root, stable_rb):
        sp = str(p.resolve())
        if sp not in sys.path:
            sys.path.insert(0, sp)


_ensure_stable_paths()

# 依赖稳定仓的模块
from core.cross_section_processor import CrossSectionProcessor  # type: ignore
from core.data_loader import DataLoader  # type: ignore
from core.precise_factor_library_v2 import PreciseFactorLibrary  # type: ignore
from real_backtest.run_production_backtest import (  # type: ignore
    backtest_no_lookahead,
)


def load_config_candidates() -> Tuple[Path, dict]:
    """寻找 combo_wfo_config.yaml（支持 RB_CONFIG_FILE 覆盖）。"""
    candidates: List[Path] = []
    env_cfg = os.environ.get("RB_CONFIG_FILE", "").strip()
    if env_cfg:
        candidates.append(Path(env_cfg).expanduser().resolve())
    here = Path(__file__).resolve()
    # experiments 下相对路径优先
    candidates.append((here.parent.parent / "configs" / "combo_wfo_config.yaml").resolve())
    # 稳定仓回退
    repo_root = here.parents[2]
    candidates.append((repo_root / "etf_rotation_optimized" / "configs" / "combo_wfo_config.yaml").resolve())
    cfg = next((p for p in candidates if p.exists()), None)
    if cfg is None:
        raise FileNotFoundError(f"未找到配置文件，尝试: {candidates}")
    import yaml

    with open(cfg, "r") as f:
        conf = yaml.safe_load(f)
    return cfg, conf


def find_latest_run_dir() -> Path:
    """在多个候选路径下查找最新的 results/run_* 目录。"""
    here = Path(__file__).resolve()
    roots: List[Path] = []
    # 环境变量优先
    env_root = os.environ.get("RB_WFO_ROOT", "").strip()
    if env_root:
        p = Path(env_root).expanduser().resolve()
        env_base = p.parent if p.name.startswith("run_") else p
        if env_base.exists():
            env_runs = sorted([d.resolve() for d in env_base.glob("run_*") if d.is_dir()], reverse=True)
            if env_runs:
                return env_runs[0]
        roots.append(env_base)
    # experiments 的 results
    roots.append((here.parent.parent / "results").resolve())
    roots.append((here.parent.parent.parent / "results").resolve())
    # 稳定仓的 results
    repo_root = here.parents[2]
    roots.append((repo_root / "etf_rotation_optimized" / "results").resolve())
    uniq = [r for r in roots if r.exists()]
    run_dirs: List[Path] = []
    for r in uniq:
        run_dirs.extend([d for d in r.glob("run_*") if d.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"未找到 run_* 目录。已检查: {uniq}")
    def _extract_ts(path: Path) -> str:
        name = path.name
        return name.replace("run_", "") if name.startswith("run_") else name

    run_dirs = sorted({d.resolve() for d in run_dirs}, key=_extract_ts, reverse=True)
    return run_dirs[0]


def _detect_scoring_strategy(run_dir: Path) -> str:
    summary_path = run_dir / "wfo_summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            strat = summary.get("scoring_strategy")
            if isinstance(strat, str) and strat:
                return strat
        except Exception:
            pass
    for candidate in ("oos_sharpe_true", "oos_sharpe_proxy", "ic"):
        if (run_dir / f"top100_by_{candidate}.parquet").exists():
            return candidate
    return "ic"


def load_top_combos_from_run(run_dir: Path, top_n: Optional[int] = None, load_all: bool = False) -> Tuple[pd.DataFrame, str]:
    """与稳定版保持兼容：优先策略对应文件 → top_combos → all_combos 并排序。"""
    strategy = _detect_scoring_strategy(run_dir)
    top_by_strategy = run_dir / f"top100_by_{strategy}.parquet"
    top_by_ic = run_dir / "top100_by_ic.parquet"
    top_combos = run_dir / "top_combos.parquet"
    all_combos = run_dir / "all_combos.parquet"

    def _sort_df(df: pd.DataFrame, strat: str) -> pd.DataFrame:
        if "calibrated_sharpe_pred" in df.columns:
            return df.sort_values(by=["calibrated_sharpe_pred", "stability_score"], ascending=[False, False])
        if "calibrated_sharpe_full" in df.columns:
            return df.sort_values(by=["calibrated_sharpe_full", "stability_score"], ascending=[False, False])
        if strat == "oos_sharpe_true" and "mean_oos_sharpe" in df.columns:
            cols = ["mean_oos_sharpe", "stability_score", "oos_sharpe_proxy", "mean_oos_ic"]
            existing = [c for c in cols if c in df.columns]
            return df.sort_values(by=existing, ascending=[False] * len(existing))
        if strat != "ic" and "oos_sharpe_proxy" in df.columns:
            cols = ["oos_sharpe_proxy", "stability_score", "mean_oos_ic"]
            existing = [c for c in cols if c in df.columns]
            if existing:
                return df.sort_values(by=existing, ascending=[False] * len(existing))
        return df.sort_values(by=["mean_oos_ic", "stability_score"], ascending=[False, False])

    if load_all or top_n is None:
        if not all_combos.exists():
            raise FileNotFoundError(f"全量模式需要 all_combos.parquet，但未找到: {all_combos}")
        df = pd.read_parquet(all_combos)
        return _sort_df(df, strategy).reset_index(drop=True), f"ALL ({len(df)})"

    candidate_paths: List[Tuple[Path, str]] = []
    if top_by_strategy.exists():
        candidate_paths.append((top_by_strategy, f"top100_by_{strategy}"))
    if strategy != "ic" and top_by_ic.exists():
        candidate_paths.append((top_by_ic, "top100_by_ic"))
    if top_combos.exists():
        candidate_paths.append((top_combos, "top_combos"))
    if all_combos.exists():
        candidate_paths.append((all_combos, "from_all_combos"))

    for path, label in candidate_paths:
        df = pd.read_parquet(path).reset_index(drop=True)
        df = _sort_df(df, strategy)
        if label.startswith("top100") and len(df) < top_n and all_combos.exists():
            all_df = pd.read_parquet(all_combos)
            return _sort_df(all_df, strategy).head(top_n).reset_index(drop=True), f"{label}(fallback_all)"
        if len(df) >= top_n or not label.startswith("top100"):
            return df.head(top_n).reset_index(drop=True), label

    raise FileNotFoundError(f"未找到 {run_dir} 下的 top100/top_combos/all_combos 文件")


def maybe_apply_profit_calibrator(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """若存在利润校准器，返回按预测年化降序的 DataFrame。"""
    model_path = Path(__file__).parent.parent / "results" / "calibrator_gbdt_profit.joblib"
    if not model_path.exists():
        return df, "IC_or_calibrated_default"
    try:
        import joblib

        obj = joblib.load(model_path)
        model = obj["model"]
        scaler = obj["scaler"]
        feat_names = obj["feature_names"]
        # 缺失列补 NaN/常数后再填充中位数
        data = df.copy()
        for c in feat_names:
            if c not in data.columns:
                data[c] = np.nan
        X = data[feat_names].copy()
        for c in X.columns:
            if X[c].isna().any():
                X[c].fillna(X[c].median(), inplace=True)
        Xs = scaler.transform(X.values)
        y_pred = model.predict(Xs)
        data["calibrated_annual_pred"] = y_pred
        data = data.sort_values(by=["calibrated_annual_pred", "stability_score"], ascending=[False, False]).reset_index(drop=True)
        return data, "profit_calibrated"
    except Exception:
        return df, "IC_or_calibrated_default"


def apply_slippage_to_nav(result: Dict, slippage_rate: float, freq: int) -> Dict:
    """
    基于“调仓换手序列 + NAV + 佣金金额”，按常数滑点做净值校正，返回包含 *_net 指标的拷贝。
    说明：在 offset 处的额外成本 = slippage_rate * turnover * P_before_cost；
          P_before_cost = nav[offset] + commission_value_at_offset（用 cost_amount_series 恢复）。
    """
    out = dict(result)  # 浅拷贝
    if slippage_rate <= 0:
        # 透传并补全 *_net 字段
        out.update(
            {
                "final_net": out["final"],
                "total_ret_net": out["total_ret"],
                "annual_ret_net": out["annual_ret"],
                "sharpe_net": out["sharpe"],
                "max_dd_net": out["max_dd"],
            }
        )
        return out

    nav = np.asarray(out["nav"], dtype=float).copy()
    if nav.size < 2:
        # 边界：数据太短
        out.update(
            {
                "final_net": out["final"],
                "total_ret_net": out["total_ret"],
                "annual_ret_net": out["annual_ret"],
                "sharpe_net": out["sharpe"],
                "max_dd_net": out["max_dd"],
            }
        )
        return out

    cost_amount = np.asarray(out.get("cost_amount_series", np.zeros(0)), dtype=float)
    turnover = np.asarray(out.get("turnover_series", np.zeros(0)), dtype=float)
    n_rb = len(turnover)
    # 映射每次调仓到 daily offset：0, freq, 2*freq, ...
    offsets = [i * freq for i in range(n_rb) if i * freq < (nav.size - 1)]
    # 逐次对 NAV 做“点状扣减 + 之后全段按比例缩放”
    nav2 = nav.copy()
    for k, off in enumerate(offsets):
        P_after_commission = nav2[off]
        commission_k = float(cost_amount[k]) if k < len(cost_amount) else 0.0
        P_before = P_after_commission + commission_k
        extra_cost = float(slippage_rate) * float(turnover[k]) * P_before
        if P_after_commission <= 0 or extra_cost <= 0:
            continue
        # 将当前点的 NAV 直接扣减 extra_cost，并将后续 NAV 按比例缩放
        new_at_off = max(P_after_commission - extra_cost, 0.0)
        if P_after_commission > 0:
            ratio = new_at_off / P_after_commission
            nav2[off:] = nav2[off:] * ratio

    # 基于 nav2 重新计算日收益与指标
    init_cap = float(nav2[0]) if nav2.size > 0 else 1.0
    daily_ret2 = nav2[1:] / nav2[:-1] - 1.0
    final_net = float(nav2[-1])
    total_ret_net = final_net / init_cap - 1.0
    days = max(len(daily_ret2), 1)
    annual_ret_net = (1 + total_ret_net) ** (252 / days) - 1
    vol = float(np.std(daily_ret2)) * np.sqrt(252)
    sharpe_net = (annual_ret_net / vol) if vol > 0 else 0.0
    cummax = np.maximum.accumulate(nav2)
    dd = (nav2 - cummax) / cummax
    max_dd_net = float(np.min(dd)) if dd.size > 0 else 0.0

    out.update(
        {
            "final_net": final_net,
            "total_ret_net": total_ret_net,
            "annual_ret_net": annual_ret_net,
            "sharpe_net": sharpe_net,
            "max_dd_net": max_dd_net,
            "nav_net": nav2,
        }
    )
    return out


def main():
    parser = argparse.ArgumentParser(description="盈利优先回测（叠加常数滑点 + 可选利润校准排序）")
    # Support None for "run all combos" - only use default 100 if RB_TOPK is explicitly set
    env_topk = os.environ.get("RB_TOPK", "").strip()
    default_topk = int(env_topk) if env_topk else None
    parser.add_argument("--topk", type=int, default=default_topk, help="回测TopK（RB_TOPK）,不指定则跑全部")
    parser.add_argument("--all", action="store_true", help="回测全量组合（RB_BACKTEST_ALL=1 同效）")
    parser.add_argument("--slippage-bps", type=float, default=float(os.environ.get("RB_SLIPPAGE_BPS", "0") or 0), help="滑点基点(双边等效)，如5表示0.05%%")
    parser.add_argument("--force-freq", type=int, default=int(os.environ.get("RB_FORCE_FREQ", "0") or 0), help="强制频率覆盖（0=不用）")
    parser.add_argument("--n-jobs", type=int, default=int(os.environ.get("RB_N_JOBS", "8") or 8), help="并行核数（当前用于WFO外的部分，回测仍逐个调用）")
    parser.add_argument("--ranking-file", type=str, default=os.environ.get("RB_RANKING_FILE", ""), help="可选：指定排序结果文件（parquet），优先于默认排序")
    args = parser.parse_args()

    print("=" * 100)
    print("盈利优先回测 (含滑点 + 利润校准排序)")
    print("=" * 100)

    # 先加载配置,获取统一的排序设置
    cfg_path, cfg = load_config_candidates()
    ranking_config = cfg.get("ranking", {})
    ranking_method = ranking_config.get("method", "wfo")  # 默认 wfo 保持向后兼容
    config_top_n = ranking_config.get("top_n", None)
    
    # 统一 TopK 逻辑: 1) args.topk 优先, 2) 配置文件 ranking.top_n, 3) None (全部)
    final_topk = args.topk if args.topk else config_top_n
    topk_source = "参数" if args.topk else ("配置文件" if config_top_n else "默认(全部)")
    topk_display = final_topk if final_topk else "全部"
    
    print(f"参数: TopK={topk_display} (来源: {topk_source}), 滑点={args.slippage_bps}bps, 强制频率={args.force_freq or '无'}")
    print()

    print(f"✓ 配置文件: {cfg_path}")
    slippage_rate = max(0.0, float(args.slippage_bps) / 10000.0)
    print(f"✓ 滑点率: {slippage_rate:.4%}")
    print()

    # 1) 数据/因子/横截面（复用稳定仓）
    print("加载数据...")
    loader = DataLoader(data_dir=cfg["data"].get("data_dir"), cache_dir=cfg["data"].get("cache_dir"))
    ohlcv = loader.load_ohlcv(
        etf_codes=cfg["data"]["symbols"],
        start_date=cfg["data"]["start_date"],
        end_date=cfg["data"]["end_date"],
        use_cache=True,
    )
    print(f"✓ 数据: {len(ohlcv['close'])}天 × {len(ohlcv['close'].columns)}只ETF")
    
    print("计算因子...")
    factor_lib = PreciseFactorLibrary()
    factors_df = factor_lib.compute_all_factors(prices=ohlcv)
    factors_dict = {name: factors_df[name] for name in factor_lib.list_factors()}
    print(f"✓ 因子: {len(factors_dict)}个")
    
    print("横截面标准化...")
    processor = CrossSectionProcessor(
        lower_percentile=cfg["cross_section"]["winsorize_lower"] * 100,
        upper_percentile=cfg["cross_section"]["winsorize_upper"] * 100,
        verbose=False,
    )
    standardized = processor.process_all_factors(factors_dict)
    factor_names = sorted(standardized.keys())
    factors_data = np.stack([standardized[n].values for n in factor_names], axis=-1)
    returns = ohlcv["close"].pct_change(fill_method=None).values
    etf_names = list(ohlcv["close"].columns)
    print("✓ 标准化完成")
    print()

    # 2) 读取最新 WFO 组合，并根据统一排序配置选择数据源
    print("读取 WFO 组合...")
    latest_run: Optional[Path] = None
    override_ts = os.environ.get("RB_RUN_TS", "").strip()
    if override_ts:
        override_candidates: List[Path] = []
        override_path = Path(override_ts)
        if override_path.is_absolute():
            override_candidates.append(override_path)
        else:
            if override_ts.startswith("run_"):
                override_candidates.append((Path(__file__).parent.parent / "results" / override_ts).resolve())
            else:
                override_candidates.append((Path(__file__).parent.parent / "results" / f"run_{override_ts}").resolve())
            override_candidates.append((Path.cwd() / override_ts).resolve())
        latest_run = next((p for p in override_candidates if p.exists()), None)
        if latest_run is None:
            print(f"[WARN] RB_RUN_TS={override_ts} 未找到匹配目录，回退为最新 run")
    if latest_run is None:
        latest_run = find_latest_run_dir()
    latest_run = latest_run.resolve()
    print(f"✓ 最新 run: {latest_run}")

    ranking_arg = (args.ranking_file or "").strip()
    ranking_path: Optional[Path] = None
    src_label: str
    order_label: str

    # 优先级: 1) 显式 --ranking-file, 2) 根据 ranking.method 自动选择
    if ranking_arg:
        # 场景1: 用户显式指定排序文件,优先使用
        candidate_paths: List[Path] = []
        rp = Path(ranking_arg)
        if rp.is_absolute():
            candidate_paths.append(rp)
        else:
            candidate_paths.append((Path.cwd() / rp).resolve())
            candidate_paths.append((latest_run / rp).resolve())
            candidate_paths.append((latest_run / "ranking_blends" / rp.name).resolve())
        ranking_path = next((p for p in candidate_paths if p.exists()), None)
        if ranking_path is None:
            raise FileNotFoundError(
                f"指定的 ranking 文件不存在: {ranking_arg}. 尝试路径: {candidate_paths}"
            )
        ranking_df = pd.read_parquet(ranking_path).reset_index(drop=True)
        top_df_cal = ranking_df
        src_label = f"ranking_file:{ranking_path.name}"
        order_label = src_label
        print(f"✓ 使用排序文件: {ranking_path} (样本={len(top_df_cal)})")
        print(f"  来源: --ranking-file 参数 (显式指定)")
    else:
        # 场景2: 根据配置文件的 ranking.method 自动选择排序方式
        print(f"  排序模式: {ranking_method.upper()} (来源: 配置文件 ranking.method)")
        
        if ranking_method == "ml":
            # ML 排序: 查找 ML 排名文件
            ml_ranking_candidates = [
                latest_run / f"ranking_ml_top{final_topk}.parquet" if final_topk else None,
                latest_run / "ranking_ml_top200.parquet",  # 默认 top200
                latest_run / f"ranked_top{final_topk}.parquet" if final_topk else None,
                latest_run / "ranked_combos.parquet",  # 全量 ML 排序
            ]
            ml_ranking_candidates = [p for p in ml_ranking_candidates if p is not None]
            
            ml_ranking_file = next((p for p in ml_ranking_candidates if p.exists()), None)
            
            if ml_ranking_file:
                print(f"✓ 找到 ML 排序文件: {ml_ranking_file.name}")
                ranking_df = pd.read_parquet(ml_ranking_file).reset_index(drop=True)
                if final_topk and final_topk > 0:
                    ranking_df = ranking_df.head(final_topk).copy()
                top_df_cal = ranking_df
                src_label = f"ML排序:{ml_ranking_file.name}"
                order_label = "ML (LTR 模型)"
                print(f"✓ 排序方式: {order_label} ✅ 生产推荐")
                print(f"  样本数: {len(top_df_cal)}")
            else:
                # ML 排名文件不存在,回退到 WFO
                print(f"⚠️  警告: 未找到 ML 排序文件,尝试路径:")
                for p in ml_ranking_candidates:
                    print(f"    - {p}")
                print(f"⚠️  自动回退到 WFO 排序逻辑")
                ranking_method = "wfo"  # 强制回退
        
        if ranking_method == "wfo":
            # WFO 排序: 使用内部排序逻辑
            backtest_all = bool(args.all or (os.environ.get("RB_BACKTEST_ALL", "0").strip().lower() in ("1", "true", "yes")))
            top_df, src_label = load_top_combos_from_run(latest_run, top_n=final_topk, load_all=backtest_all)
            print(f"✓ 组合数: {len(top_df)} (来源: {src_label})")

            blend_dir = latest_run / "ranking_blends"
            unlimited_path = blend_dir / "ranking_two_stage_unlimited.parquet"
            if unlimited_path.exists():
                print(f"发现 Unlimited 排名: {unlimited_path}")
                ranking_df = pd.read_parquet(unlimited_path).reset_index(drop=True)
                if final_topk and final_topk > 0:
                    ranking_df = ranking_df.head(final_topk).copy()
                top_df_cal = ranking_df
                order_label = "two_stage_unlimited"
                src_label = f"{src_label}|two_stage_unlimited"
                print(f"✓ 排序方式: {order_label} (样本={len(top_df_cal)})")
            else:
                print("应用利润校准器...")
                top_df_cal, order_label = maybe_apply_profit_calibrator(top_df)
                print(f"✓ 排序方式: WFO 内部排序 ⚠️ 备用模式")
                print(f"  排序指标: {order_label}")
    print()

    # 3) 调用稳定回测，逐组合叠加常数滑点
    commission_rate_cfg = cfg.get("backtest", {}).get("commission_rate", 0.00005)
    lookback_window = cfg.get("backtest", {}).get("lookback_window", 252)
    force_freq = int(args.force_freq) if int(args.force_freq) > 0 else None
    results_rows: List[dict] = []

    invocation_ts = os.environ.get("RB_RESULT_TS") or datetime.now().strftime("%Y%m%d_%H%M%S")
    latest_ts = latest_run.name.replace("run_", "")
    out_dir = Path(__file__).parent.parent / "results_combo_wfo" / f"{latest_ts}_{invocation_ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"开始回测 {len(top_df_cal)} 个组合...")
    print(f"输出目录: {out_dir}")
    print()
    
    for idx, row in tqdm(top_df_cal.iterrows(), total=len(top_df_cal), desc="回测进度"):
        combo = str(row["combo"])
        wfo_freq = int(row["best_rebalance_freq"])
        freq = force_freq if force_freq is not None else wfo_freq
        # 提取因子
        factor_list = [s.strip() for s in combo.split("+")]
        if any((f not in factor_names) for f in factor_list):
            continue
        fi = [factor_names.index(f) for f in factor_list]
        factors_sel = factors_data[:, :, fi]
        try:
            base = backtest_no_lookahead(
                factors_data=factors_sel,
                returns=returns,
                etf_names=etf_names,
                rebalance_freq=freq,
                lookback_window=lookback_window,
                position_size=5,  # 与生产一致的默认TopN
                commission_rate=commission_rate_cfg,
                initial_capital=1_000_000.0,
                factors_data_full=factors_data,
                factor_indices_for_cache=np.asarray(fi, dtype=np.int64),
            )
            enriched = apply_slippage_to_nav(base, slippage_rate=slippage_rate, freq=freq)
        except Exception as e:
            # 跳过失败组合
            continue

        rec = {
            "rank": idx + 1,
            "combo": combo,
            "combo_size": int(row.get("combo_size", len(fi))),
            "wfo_freq": wfo_freq,
            "test_freq": freq,
            "test_position_size": 5,
            "freq": freq,
            "wfo_ic": float(row.get("mean_oos_ic", np.nan)),
            "wfo_score": float(row.get("stability_score", np.nan)),
            # 基线（含佣金）
            "final_value": float(enriched["final"]),
            "total_ret": float(enriched["total_ret"]),
            "annual_ret": float(enriched["annual_ret"]),
            "vol": float(enriched["vol"]),
            "sharpe": float(enriched["sharpe"]),
            "max_dd": float(enriched["max_dd"]),
            "n_rebalance": int(enriched["n_rebalance"]),
            "avg_turnover": float(enriched["avg_turnover"]),
            "avg_n_holdings": float(enriched.get("avg_n_holdings", np.nan)),
            "win_rate": float(enriched.get("win_rate", np.nan)),
            "winning_days": int(enriched.get("winning_days", 0)),
            "losing_days": int(enriched.get("losing_days", 0)),
            "avg_win": float(enriched.get("avg_win", np.nan)),
            "avg_loss": float(enriched.get("avg_loss", np.nan)),
            "profit_factor": float(enriched.get("profit_factor", np.nan)),
            "calmar_ratio": float(enriched.get("calmar_ratio", np.nan)),
            "sortino_ratio": float(enriched.get("sortino_ratio", np.nan)),
            "max_consecutive_wins": int(enriched.get("max_consecutive_wins", 0)),
            "max_consecutive_losses": int(enriched.get("max_consecutive_losses", 0)),
            # 净值（佣金+滑点）
            "final_value_net": float(enriched["final_net"]),
            "total_ret_net": float(enriched["total_ret_net"]),
            "annual_ret_net": float(enriched["annual_ret_net"]),
            "sharpe_net": float(enriched["sharpe_net"]),
            "max_dd_net": float(enriched["max_dd_net"]),
            "run_tag": f"{order_label}:{latest_run.name}",
        }
        # 若存在利润校准预测，附带
        if "calibrated_annual_pred" in top_df_cal.columns:
            rec["calibrated_annual_pred"] = float(row["calibrated_annual_pred"])
        results_rows.append(rec)

    if not results_rows:
        print("❌ 无可用回测结果")
        return

    df = pd.DataFrame(results_rows).sort_values("sharpe_net", ascending=False).reset_index(drop=True)
    # 保存
    tag = f"profit_backtest_slip{int(args.slippage_bps)}bps_{latest_ts}_{invocation_ts}"
    out_file = out_dir / f"top{len(df)}_{tag}.csv"
    df.to_csv(out_file, index=False)

    # 汇总
    summary = {
        "latest_run": str(latest_run),
        "config_file": str(cfg_path),
        "top_source": src_label,
        "order_label": order_label,
        "slippage_bps": float(args.slippage_bps),
        "count": int(len(df)),
        "mean_annual_net": float(df["annual_ret_net"].mean()),
        "median_annual_net": float(df["annual_ret_net"].median()),
        "mean_sharpe_net": float(df["sharpe_net"].mean()),
        "median_sharpe_net": float(df["sharpe_net"].median()),
    }
    with open(out_dir / f"SUMMARY_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 100)
    print(f"✅ 盈利优先回测完成 | 排序: {order_label} | 滑点: {args.slippage_bps} bps")
    print(f"输出目录: {out_dir}")
    print(f"保存文件: {out_file.name}")
    print(f"Top1年化(净): {df.loc[0,'annual_ret_net']:.2%} | Sharpe(净): {df.loc[0,'sharpe_net']:.3f}")
    print("=" * 100)


if __name__ == "__main__":
    main()


