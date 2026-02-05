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
- results/run_XXXXXX/top100_by_ic.parquet
- results/run_XXXXXX/all_combos.parquet
- results/run_XXXXXX/wfo_summary.json

用法：
    python run_combo_wfo.py
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from core.combo_wfo_optimizer import ComboWFOOptimizer
from core.cross_section_processor import CrossSectionProcessor
from core.data_loader import DataLoader
from core.precise_factor_library_v2 import PreciseFactorLibrary

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    """主函数"""

    # ========== 1. 加载配置 ==========
    logger.info("=" * 100)
    logger.info("🚀 组合级WFO优化启动")
    logger.info("=" * 100)
    logger.info("")

    config_path = Path(
        os.environ.get("WFO_CONFIG_PATH", "configs/combo_wfo_config.yaml")
    )
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

    # 使用 training_end_date 如果设置了（Holdout验证模式）
    data_end_date = (
        config["data"].get("training_end_date") or config["data"]["end_date"]
    )

    if config["data"].get("training_end_date"):
        logger.info("=" * 100)
        logger.info("🔬 HOLDOUT验证模式")
        logger.info("=" * 100)
        logger.info(f"训练集截止日期: {data_end_date}")
        logger.info(f"完整数据截止日期: {config['data']['end_date']}")
        logger.info(f"Holdout期: {data_end_date} 至 {config['data']['end_date']}")
        logger.info("⚠️  注意: 当前仅使用训练集数据，Holdout期数据将用于最终验证")
        logger.info("")

    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=data_end_date,  # 使用训练集截止日期
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

    optimizer = ComboWFOOptimizer(
        combo_sizes=config["combo_wfo"]["combo_sizes"],
        is_period=config["combo_wfo"]["is_period"],
        oos_period=config["combo_wfo"]["oos_period"],
        step_size=config["combo_wfo"]["step_size"],
        n_jobs=config["combo_wfo"]["n_jobs"],
        verbose=1 if config["combo_wfo"]["verbose"] else 0,
        enable_fdr=config["combo_wfo"]["enable_fdr"],
        fdr_alpha=config["combo_wfo"]["fdr_alpha"],
        complexity_penalty_lambda=config["combo_wfo"]["scoring"][
            "complexity_penalty_lambda"
        ],
        rebalance_frequencies=config["combo_wfo"]["rebalance_frequencies"],
    )

    top_combos_list, all_combos_df = optimizer.run_combo_search(
        factors_data=factors_data,
        returns=returns,
        factor_names=factor_names,
        top_n=config["combo_wfo"].get("top_n", 100),
        pos_size=config["backtest"].get("pos_size", 2),
        commission_rate=config["backtest"].get("commission_rate", 0.0002),
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
    results_root = Path("results").resolve()
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
        "config_file": "configs/combo_wfo_config.yaml",
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

    # 保存Top组合 (匹配现有文件名: top_combos.parquet)
    top_n = config["combo_wfo"].get("top_n", 100)
    top_combos = all_combos_df.head(top_n)  # 已经排序过了
    top_combos.to_parquet(pending_dir / "top_combos.parquet", index=False)
    logger.info(f"✅ Top{top_n}组合已保存: {pending_dir}/top_combos.parquet")

    # 同时保存为 top100_by_ic.parquet (为了兼容回测脚本)
    top_combos.to_parquet(pending_dir / "top100_by_ic.parquet", index=False)
    logger.info(f"✅ Top{top_n}组合已保存(兼容): {pending_dir}/top100_by_ic.parquet")

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

    with open(
        pending_dir / "factor_selection_summary.json", "w", encoding="utf-8"
    ) as f:
        json.dump(factor_selection_summary, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ 因子汇总已保存: {pending_dir}/factor_selection_summary.json")

    # 保存WFO汇总信息 (匹配现有格式)
    significant_combos = all_combos_df[all_combos_df.get("is_significant", True)]

    summary = {
        "timestamp": timestamp,
        "total_combos": len(all_combos_df),
        "significant_combos": len(significant_combos),
        "mean_ic": float(all_combos_df["mean_oos_ic"].mean()),
        "mean_oos_return": (
            float(all_combos_df["mean_oos_return"].mean())
            if "mean_oos_return" in all_combos_df.columns
            else 0.0
        ),
        "best_combo": {
            "combo": top_combos.iloc[0]["combo"],
            "ic": float(top_combos.iloc[0]["mean_oos_ic"]),
            "score": float(top_combos.iloc[0]["stability_score"]),
            "freq": int(top_combos.iloc[0]["best_rebalance_freq"]),
            "mean_oos_return": (
                float(top_combos.iloc[0]["mean_oos_return"])
                if "mean_oos_return" in top_combos.columns
                else 0.0
            ),
            "cum_oos_return": (
                float(top_combos.iloc[0]["cum_oos_return"])
                if "cum_oos_return" in top_combos.columns
                else 0.0
            ),
        },
        "config": {
            "is_period": config["combo_wfo"]["is_period"],
            "oos_period": config["combo_wfo"]["oos_period"],
            "step_size": config["combo_wfo"]["step_size"],
            "combo_sizes": config["combo_wfo"]["combo_sizes"],
            "pos_size": config["backtest"].get("pos_size", 2),
        },
        "runtime_minutes": 0.0,  # 运行时间将在后续更新
    }

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
    logger.info("🏆 Top 1 组合:")
    logger.info(f'  - 名称: {summary["best_combo"]["combo"]}')
    logger.info(f'  - OOS Sharpe: {summary["best_combo"]["ic"]:.4f} (原IC字段)')
    logger.info(f'  - 稳定性得分: {summary["best_combo"]["score"]:.2f}')
    logger.info(f'  - 最优换仓频率: {summary["best_combo"]["freq"]}天')
    if "best_trailing_stop" in top_combos.iloc[0]:
        logger.info(
            f'  - 最优动态止损: {top_combos.iloc[0]["best_trailing_stop"]*100:.1f}%'
        )
    logger.info("")
    logger.info("📈 整体统计:")
    logger.info(f'  - 平均OOS Sharpe: {summary["mean_ic"]:.4f}')
    logger.info(
        f'  - 显著组合数: {summary["significant_combos"]}/{summary["total_combos"]}'
    )
    logger.info("")
    logger.info("=" * 100)
    logger.info("✅ WFO优化完成！")
    logger.info("=" * 100)
    logger.info("")
    logger.info("💡 下一步:")
    logger.info("   运行真实回测: python test_freq_no_lookahead.py")
    logger.info("")


if __name__ == "__main__":
    main()
