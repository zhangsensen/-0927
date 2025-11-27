#!/usr/bin/env python3
"""
组合级WFO优化启动脚本

功能：
1. 加载OHLCV数据
2. 计算精确因子库
3. 横截面标准化处理
4. 执行组合级Walk-Forward优化
5. 保存Top组合到 results/run_XXXXXX/

输出：
- results/run_XXXXXX/top_combos.parquet
- results/run_XXXXXX/ranking_ic_top<top_n>.parquet
- results/run_XXXXXX/top100_by_ic.parquet（兼容旧流程）
- results/run_XXXXXX/all_combos.parquet
- results/run_XXXXXX/wfo_summary.json

用法：
    python applications/run_combo_wfo.py
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
import argparse
import subprocess

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strategies.wfo.combo_wfo_optimizer import ComboWFOOptimizer
from core.cross_section_processor import CrossSectionProcessor
from core.data_loader import DataLoader
from core.precise_factor_library_v2 import PreciseFactorLibrary

# 设置日志（输出到控制台和文件）
log_dir = PROJECT_ROOT / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"wfo_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"日志文件: {log_file}")

# ML排序模块 (在 logger 定义之后导入)
try:
    from applications.apply_ranker import apply_ltr_ranking

    ML_RANKER_AVAILABLE = True
except ImportError:
    ML_RANKER_AVAILABLE = False
    logger.warning("ML排序模块不可用,仅支持 WFO 排序模式")


def _discover_ranking_files(run_dir: Path):
    """发现可用于回测的排序/组合文件列表"""
    patterns = [
        "ranking_*_top*.parquet",
        "top100_by_*.parquet",
        "top_combos.parquet",
    ]
    found = []
    for pat in patterns:
        for f in run_dir.glob(pat):
            if f.is_file():
                found.append(f.resolve())
    uniq = sorted({p for p in found})
    return uniq


def _run_backtests(run_dir: Path, ranking_files, topk: int = None, slippage_bps: int = 2):
    """对发现的 ranking 文件逐个调用真实回测脚本"""
    backtest_script = PROJECT_ROOT / "real_backtest" / "run_profit_backtest.py"
    if not backtest_script.exists():
        logger.warning("回测脚本不存在，跳过自动回测: %s", backtest_script)
        return []
    results = []
    for rf in ranking_files:
        cmd = [
            sys.executable,
            str(backtest_script),
            "--ranking-file", str(rf),
            "--slippage-bps", str(slippage_bps),
        ]
        # 只有明确指定topk时才添加该参数（None表示跑全部）
        if topk is not None:
            cmd.extend(["--topk", str(topk)])
        logger.info("[AUTO-BACKTEST] %s", " ".join(cmd))
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        meta = {
            "ranking_file": rf.name,
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout.splitlines()[-10:],
            "stderr_head": proc.stderr.splitlines()[:10],
        }
        if proc.returncode != 0:
            logger.error("回测失败: %s", rf.name)
            logger.error("stderr片段: %s", "\n".join(meta["stderr_head"]))
        else:
            logger.info("回测完成: %s", rf.name)
        results.append(meta)
    return results


def main():
    """主函数"""

    # ========== 1. 加载配置 ==========
    logger.info("=" * 100)
    logger.info("🚀 组合级WFO优化启动")
    logger.info("=" * 100)
    logger.info("")

    # 解析命令行参数（支持外部指定配置路径）
    parser = argparse.ArgumentParser(description="Run combo-level WFO optimization (optional auto backtest)")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/combo_wfo_config.yaml",
        help="Path to YAML config file (default: configs/combo_wfo_config.yaml)",
    )
    parser.add_argument(
        "--auto-backtest",
        action="store_true",
        help="在WFO结束后自动发现 ranking 文件并执行真实回测",
    )
    parser.add_argument(
        "--backtest-topk",
        type=int,
        default=None,
        help="自动回测 topk (默认None=全部组合，可指定如100)",
    )
    parser.add_argument(
        "--backtest-slippage-bps",
        type=int,
        default=2,
        help="自动回测滑点bps (default 2)",
    )
    args = parser.parse_args()

    # 允许环境变量覆盖（优先级高于默认，低于 CLI）
    config_env = os.environ.get("WFO_CONFIG_PATH")
    config_path = Path(config_env) if config_env else Path(args.config)
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info("✅ 配置加载成功")
    logger.info(f'  - ETF数量: {len(config["data"]["symbols"])}')
    logger.info(
        f'  - 日期范围: {config["data"]["start_date"]} → {config["data"]["end_date"]}'
    )
    logger.info(f'  - 组合规模: {config["combo_wfo"]["combo_sizes"]}')
    logger.info(f'  - IS窗口: {config["combo_wfo"]["is_period"]}天')
    logger.info(f'  - OOS窗口: {config["combo_wfo"]["oos_period"]}天')
    logger.info("")

    # 环境变量覆盖：可通过 RB_FREQ_SUBSET 指定逗号分隔的换仓频率列表（例如 "8" 或 "8,16,24"）
    # 可通过 RB_RESULT_TS 指定输出目录的时间戳，以便与启动脚本的日志时间戳一致。
    env_freq = os.environ.get("RB_FREQ_SUBSET")
    if env_freq:
        try:
            override = [int(x.strip()) for x in env_freq.split(",") if x.strip()]
            if override:
                config["combo_wfo"]["rebalance_frequencies"] = override
                logger.info(f"🔧 通过环境变量覆盖频率: RB_FREQ_SUBSET={override}")
        except Exception as e:
            logger.warning(f"忽略非法的 RB_FREQ_SUBSET 值: {env_freq} ({e})")
    env_ts = os.environ.get("RB_RESULT_TS", "").strip()
    if env_ts:
        logger.info(f"🔧 使用外部指定时间戳 RB_RESULT_TS={env_ts}")

    # ========== 2. 加载数据 ==========
    logger.info("=" * 100)
    logger.info("📊 加载OHLCV数据")
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

    logger.info(f"✅ 数据加载完成")
    logger.info(f'  - 交易日数: {len(ohlcv["close"])}')
    logger.info(f'  - ETF数量: {len(ohlcv["close"].columns)}')
    logger.info("")

    # ========== 3. 计算因子 ==========
    logger.info("=" * 100)
    logger.info("🔧 计算精确因子库")
    logger.info("=" * 100)

    factor_lib = PreciseFactorLibrary()
    factors_df = factor_lib.compute_all_factors(prices=ohlcv)
    factors_dict = {name: factors_df[name] for name in factor_lib.list_factors()}

    logger.info(f"✅ 因子计算完成")
    logger.info(f"  - 因子数量: {len(factors_dict)}")
    logger.info(f'  - 因子列表: {", ".join(sorted(factors_dict.keys())[:10])}...')
    logger.info("")

    # ========== 4. 横截面标准化 ==========
    logger.info("=" * 100)
    logger.info("📐 横截面标准化处理")
    logger.info("=" * 100)

    processor = CrossSectionProcessor(
        lower_percentile=config["cross_section"]["winsorize_lower"] * 100,
        upper_percentile=config["cross_section"]["winsorize_upper"] * 100,
        verbose=False,
    )

    standardized_factors = processor.process_all_factors(factors_dict)

    logger.info(f"✅ 标准化完成")
    logger.info(
        f'  - Winsorize范围: [{config["cross_section"]["winsorize_lower"]}, {config["cross_section"]["winsorize_upper"]}]'
    )
    logger.info("")

    # ========== 5. 准备数据 ==========
    logger.info("=" * 100)
    logger.info("🔄 准备WFO输入数据")
    logger.info("=" * 100)

    # 组织因子数据
    factor_names = sorted(standardized_factors.keys())
    factor_arrays = [standardized_factors[name].values for name in factor_names]
    factors_data = np.stack(factor_arrays, axis=-1)

    # 准备收益率
    returns_df = ohlcv["close"].pct_change(fill_method=None)
    returns = returns_df.values

    logger.info(f"✅ 数据准备完成")
    logger.info(
        f"  - 数据维度: {factors_data.shape[0]}天 × {factors_data.shape[1]}只ETF × {factors_data.shape[2]}个因子"
    )
    logger.info(f"  - 因子名称: {factor_names}")
    logger.info("")

    # ========== 6. 执行WFO优化 ==========
    logger.info("=" * 100)
    logger.info("⚡ 执行组合级WFO优化")
    logger.info("=" * 100)
    logger.info("")

    scoring_cfg = config["combo_wfo"].get("scoring", {})

    optimizer = ComboWFOOptimizer(
        combo_sizes=config["combo_wfo"]["combo_sizes"],
        is_period=config["combo_wfo"]["is_period"],
        oos_period=config["combo_wfo"]["oos_period"],
        step_size=config["combo_wfo"]["step_size"],
        n_jobs=config["combo_wfo"]["n_jobs"],
        verbose=1 if config["combo_wfo"]["verbose"] else 0,
        enable_fdr=config["combo_wfo"]["enable_fdr"],
        fdr_alpha=config["combo_wfo"]["fdr_alpha"],
        complexity_penalty_lambda=scoring_cfg.get(
            "complexity_penalty_lambda", 0.01
        ),
        rebalance_frequencies=config["combo_wfo"]["rebalance_frequencies"],
        scoring_strategy=config["combo_wfo"].get("scoring_strategy", "ic"),
        scoring_position_size=scoring_cfg.get("position_size", 5),
    )

    top_combos_list, all_combos_df = optimizer.run_combo_search(
        factors_data=factors_data,
        returns=returns,
        factor_names=factor_names,
        top_n=config["combo_wfo"].get("top_n", 5000),
    )

    logger.info("")
    logger.info("✅ WFO优化完成")
    logger.info("")

    # ========== 7. 保存结果 ==========
    logger.info("=" * 100)
    logger.info("💾 保存结果")
    logger.info("=" * 100)

    # 创建输出目录（原子写入）：pending_run_<ts> -> run_<ts>
    timestamp = env_ts if env_ts else datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = PROJECT_ROOT / "results"
    final_dir = results_root / f"run_{timestamp}"
    pending_dir = results_root / f"pending_run_{timestamp}"
    if pending_dir.exists():
        import shutil
        logger.warning(f"清理残留的临时目录: {pending_dir}")
        shutil.rmtree(pending_dir, ignore_errors=True)
    pending_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置 (匹配现有格式)
    run_config = {
        "timestamp": timestamp,
        "config_file": str(config_path),
        "quick_mode": False,
        "parameters": {
            "run_id": config.get("run_id", "COMBO_WFO_DEEP_MINING"),
            "data": config["data"],
            "cross_section": config["cross_section"],
            "combo_wfo": config["combo_wfo"],
            "output_root": config.get("output_root", "results_combo_wfo"),
        },
    }

    with open(pending_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ 配置已保存: {pending_dir}/run_config.json")

    # 保存WFO结果
    all_combos_df.to_parquet(pending_dir / "all_combos.parquet", index=False)
    logger.info(
        f"✅ 全部组合已保存: {pending_dir}/all_combos.parquet ({len(all_combos_df)} 个组合)"
    )

    strategy_tag = optimizer.config.scoring_strategy
    primary_metric_map = {
        "ic": "mean_oos_ic",
        "oos_sharpe_proxy": "oos_sharpe_proxy",
        "oos_sharpe_true": "mean_oos_sharpe",
        "oos_sharpe_compound": "oos_compound_sharpe",
    }
    primary_metric = primary_metric_map.get(strategy_tag, "mean_oos_ic")

    # ========== 排序模式选择 ==========
    ranking_config = config.get("ranking", {})
    ranking_method = ranking_config.get("method", "ml")  # 默认改为 ml
    ranking_top_n = ranking_config.get("top_n", config["combo_wfo"].get("top_n", 5000))
    
    logger.info("")
    logger.info("=" * 100)
    logger.info("🔀 排序模式选择")
    logger.info("=" * 100)
    
    if ranking_method == "ml":
        logger.info("  📊 排序方式: ML (LTR 模型) ✅ 生产推荐")
    else:
        logger.info("  📊 排序方式: WFO (mean_oos_ic) ⚠️ 备用模式")
    
    logger.info(f"  TopN: {ranking_top_n}")
    
    if ranking_method == "ml":
        # ML排序模式
        if not ML_RANKER_AVAILABLE:
            logger.error("❌ ML排序模块不可用,请检查 applications/apply_ranker.py 是否存在")
            logger.error("   ⚠️ 自动回退到 WFO 排序模式")
            ranking_method = "wfo"
        else:
            ml_model_path = ranking_config.get("ml_model_path", "ml_ranker/models/ltr_ranker")
            model_full_path = PROJECT_ROOT / ml_model_path
            
            # 检查模型是否存在
            model_dir = model_full_path if model_full_path.is_dir() else model_full_path.parent
            if not model_dir.exists():
                logger.error(f"❌ ML模型不存在: {model_dir}")
                logger.error("   💡 提示: 请先运行 python run_ranking_pipeline.py 训练模型")
                logger.error("   ⚠️ 自动回退到 WFO 排序模式")
                ranking_method = "wfo"
            else:
                logger.info(f"  模型路径: {ml_model_path}")
                logger.info("")
                logger.info("⚡ 执行ML排序...")
                
                try:
                    # 调用ML排序
                    ranked_df = apply_ltr_ranking(
                        model_path=str(model_full_path),
                        wfo_dir=str(pending_dir),
                        output_path=None,  # 不在这里保存,后面统一处理
                        top_k=None,
                        verbose=False  # 避免过多日志
                    )
                    
                    logger.info(f"✅ ML排序完成: {len(ranked_df)} 个组合")
                    
                    # 使用ML排序结果作为后续的基准
                    all_combos_df = ranked_df
                    strategy_tag = "ml"  # 标记为ML排序
                    primary_metric = "ltr_score"
                    
                    logger.info(f"  Top-1 LTR分数: {ranked_df.iloc[0]['ltr_score']:.4f}")
                    logger.info(f"  Top-1 组合: {ranked_df.iloc[0]['combo']}")
                    
                except Exception as e:
                    logger.error(f"❌ ML排序失败: {e}")
                    logger.error("   ⚠️ 自动回退到 WFO 排序模式")
                    import traceback
                    traceback.print_exc()
                    ranking_method = "wfo"
    
    if ranking_method == "wfo":
        # WFO排序模式 (原有逻辑)
        logger.info("  使用 WFO 原始排序 (mean_oos_ic + stability_score)")
        logger.info("")
    
    # 保存Top组合 (匹配现有文件名: top_combos.parquet)
    top_n = ranking_top_n  # 使用 ranking 配置的 top_n
    top_combos = all_combos_df.head(top_n)  # 已经排序过了
    top_combos.to_parquet(pending_dir / "top_combos.parquet", index=False)
    logger.info(f"✅ Top{top_n}组合已保存: {pending_dir}/top_combos.parquet")

    ranking_filename = f"ranking_{strategy_tag}_top{top_n}.parquet"
    ranking_path = pending_dir / ranking_filename
    top_combos.to_parquet(ranking_path, index=False)
    logger.info(f"✅ 排名文件已保存: {ranking_path}")

    if strategy_tag == "ic":
        legacy_ranking = pending_dir / f"ranking_ic_top{top_n}.parquet"
        if legacy_ranking != ranking_path:
            top_combos.to_parquet(legacy_ranking, index=False)
            logger.info(f"✅ 兼容排名文件已保存: {legacy_ranking}")

    # 保存Top100（按策略命名）
    top_compat = top_combos.head(100)
    top100_filename = f"top100_by_{strategy_tag}.parquet"
    top100_path = pending_dir / top100_filename
    top_compat.to_parquet(top100_path, index=False)
    logger.info(f"✅ Top100组合已保存: {top100_path}")
    if strategy_tag == "ic" and top100_path.name != "top100_by_ic.parquet":
        compat_top100 = pending_dir / "top100_by_ic.parquet"
        top_compat.to_parquet(compat_top100, index=False)
        logger.info(f"✅ Top100兼容文件已保存: {compat_top100}")

    # 保存因子数据到 factors/ 目录
    factors_dir = pending_dir / "factors"
    factors_dir.mkdir(exist_ok=True)
    for factor_name in factor_names:
        factor_df = standardized_factors[factor_name]
        factor_df.to_parquet(factors_dir / f"{factor_name}.parquet")
    logger.info(f"✅ {len(factor_names)}个因子已保存: {factors_dir}/")

    # 保存因子筛选汇总 (匹配现有格式)
    factor_selection_summary = {
        "timestamp": timestamp,
        "n_factors": len(factor_names),
        "factor_names": factor_names,
        "data_shape": {
            "n_days": factors_data.shape[0],
            "n_etfs": factors_data.shape[1],
            "n_factors": factors_data.shape[2],
        },
        "winsorize": {
            "lower": config["cross_section"]["winsorize_lower"],
            "upper": config["cross_section"]["winsorize_upper"],
        },
    }

    with open(pending_dir / "factor_selection_summary.json", "w", encoding="utf-8") as f:
        json.dump(factor_selection_summary, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ 因子汇总已保存: {pending_dir}/factor_selection_summary.json")

    # 保存WFO汇总信息 (匹配现有格式)
    significant_combos = all_combos_df[all_combos_df.get("is_significant", True)]
    primary_metric_mean = float(all_combos_df[primary_metric].mean()) if primary_metric in all_combos_df.columns else float("nan")
    top_primary_value = float(top_combos.iloc[0].get(primary_metric, float("nan")))

    summary = {
        "timestamp": timestamp,
        "total_combos": len(all_combos_df),
        "significant_combos": len(significant_combos),
        "scoring_strategy": strategy_tag,
        "primary_metric": primary_metric,
        "primary_metric_mean": primary_metric_mean,
        "mean_ic": float(all_combos_df["mean_oos_ic"].mean()),
        "best_combo": {
            "combo": top_combos.iloc[0]["combo"],
            "metric_value": top_primary_value,
            "metric_name": primary_metric,
            "ic": float(top_combos.iloc[0].get("mean_oos_ic", float("nan"))),
            "score": float(top_combos.iloc[0]["stability_score"]),
            "freq": int(top_combos.iloc[0]["best_rebalance_freq"]),
        },
        "config": {
            "is_period": config["combo_wfo"]["is_period"],
            "oos_period": config["combo_wfo"]["oos_period"],
            "step_size": config["combo_wfo"]["step_size"],
            "combo_sizes": config["combo_wfo"]["combo_sizes"],
        },
        "runtime_minutes": 0.0,  # 运行时间将在后续更新
    }
    
    # 策略特定的元数据增强
    if strategy_tag == "oos_sharpe_true" and "oos_sharpe_std" in all_combos_df.columns:
        summary["oos_sharpe_std_mean"] = float(all_combos_df["oos_sharpe_std"].mean())
        if "mean_oos_sample_count" in all_combos_df.columns:
            summary["mean_oos_sample_count_global"] = float(all_combos_df["mean_oos_sample_count"].mean())
    elif strategy_tag == "oos_sharpe_compound":
        if "oos_compound_std" in all_combos_df.columns:
            summary["oos_compound_std_mean"] = float(all_combos_df["oos_compound_std"].mean())
        if "oos_compound_sample_count" in all_combos_df.columns:
            summary["oos_compound_sample_count_global"] = float(
                all_combos_df["oos_compound_sample_count"].mean()
            )

    with open(pending_dir / "wfo_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 策略特定的元数据增强
    if strategy_tag == "oos_sharpe_true" and "oos_sharpe_std" in all_combos_df.columns:
        summary["oos_sharpe_std_mean"] = float(all_combos_df["oos_sharpe_std"].mean())
        if "mean_oos_sample_count" in all_combos_df.columns:
            summary["mean_oos_sample_count_global"] = float(all_combos_df["mean_oos_sample_count"].mean())
    elif strategy_tag == "oos_sharpe_compound":
        if "oos_compound_std" in all_combos_df.columns:
            summary["oos_compound_std_mean"] = float(all_combos_df["oos_compound_std"].mean())
        if "oos_compound_sample_count" in all_combos_df.columns:
            summary["oos_compound_sample_count_global"] = float(
                all_combos_df["oos_compound_sample_count"].mean()
            )

    with open(pending_dir / "wfo_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"✅ WFO汇总已保存: {pending_dir}/wfo_summary.json")
    # 原子切换到最终目录
    try:
        if final_dir.exists():
            import shutil
            logger.warning(f"目标目录已存在，将被替换: {final_dir}")
            shutil.rmtree(final_dir, ignore_errors=True)
        pending_dir.rename(final_dir)
        # 写入就绪标记
        (final_dir / "READY").write_text("ok", encoding="utf-8")
        # 维护指针
        latest_ptr = results_root / ".latest_run"
        latest_ptr.write_text(final_dir.name, encoding="utf-8")
        latest_link = results_root / "run_latest"
        try:
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
            latest_link.symlink_to(final_dir)
        except Exception as e:
            logger.warning(f"创建最新运行符号链接失败: {e}")
    except Exception as e:
        logger.error(f"切换到最终目录失败: {e}")
        raise
    logger.info("")

    # ========== 8. 结果汇总 ==========
    logger.info("=" * 100)
    logger.info("📊 结果汇总")
    logger.info("=" * 100)
    logger.info("")
    logger.info(f"输出目录: {final_dir}")
    logger.info(f'总组合数: {summary["total_combos"]}')
    logger.info("")
    top_metric_name = summary["best_combo"].get("metric_name")
    top_metric_value = summary["best_combo"].get("metric_value")
    logger.info("🏆 Top 1 组合:")
    logger.info(f'  - 名称: {summary["best_combo"]["combo"]}')
    if top_metric_name and top_metric_value is not None and not np.isnan(top_metric_value):
        logger.info(f'  - 主排序指标({top_metric_name}): {top_metric_value:.4f}')
    if summary["best_combo"].get("ic") is not None and not np.isnan(summary["best_combo"].get("ic")):
        logger.info(f'  - OOS IC: {summary["best_combo"].get("ic"):.4f}')
    logger.info(f'  - 稳定性得分: {summary["best_combo"]["score"]:.2f}')
    logger.info(f'  - 最优换仓频率: {summary["best_combo"]["freq"]}天')
    logger.info("")
    logger.info("📈 整体统计:")
    logger.info(f'  - 平均OOS IC: {summary["mean_ic"]:.4f}')
    if summary.get("primary_metric_mean") is not None and not np.isnan(summary["primary_metric_mean"]):
        logger.info(
            "  - 主排序指标均值(%s): %.4f",
            summary.get("primary_metric"),
            summary["primary_metric_mean"],
        )
    logger.info(
        f'  - 显著组合数: {summary["significant_combos"]}/{summary["total_combos"]}'
    )
    logger.info("")
    logger.info("=" * 100)
    logger.info("✅ WFO优化完成！")
    logger.info("=" * 100)
    logger.info("")
    # 自动回测逻辑
    if args.auto_backtest or os.environ.get("AUTO_BACKTEST", "").lower() in ("1", "true", "yes"):
        logger.info("🚀 自动回测阶段启动 (--auto-backtest)")
        ranking_files = _discover_ranking_files(final_dir)
        if not ranking_files:
            logger.warning("未发现 ranking 文件，跳过回测。")
        else:
            logger.info("发现 %d 个 ranking 文件:", len(ranking_files))
            for rf in ranking_files:
                logger.info("  - %s", rf.name)
            backtest_meta = _run_backtests(
                final_dir,
                ranking_files,
                topk=args.backtest_topk,
                slippage_bps=args.backtest_slippage_bps,
            )
            auto_bt_summary = {
                "timestamp": timestamp,
                "run_dir": str(final_dir),
                "backtest_topk": args.backtest_topk,
                "backtest_slippage_bps": args.backtest_slippage_bps,
                "ranking_files": [f.name for f in ranking_files],
                "backtests": backtest_meta,
            }
            (final_dir / "auto_backtest_summary.json").write_text(
                json.dumps(auto_bt_summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("✅ 自动回测摘要写入: %s/auto_backtest_summary.json", final_dir)
    else:
        logger.info("ℹ️ 若需自动回测可使用 --auto-backtest 或设置 AUTO_BACKTEST=1")
        logger.info("💡 手动示例: python real_backtest/run_profit_backtest.py --topk 100 --ranking-file results/run_latest/ranking_ic_top5000.parquet")
    logger.info("")


if __name__ == "__main__":
    main()
