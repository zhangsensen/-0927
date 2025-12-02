#!/usr/bin/env python3
"""
Step 3: WFO优化执行 - 独立执行脚本

功能：
1. 读取 Step 2 的标准化因子数据
2. 执行Walk-Forward优化（55个窗口）
3. 保存WFO结果到 wfo/
4. 详细的窗口进度和IC统计日志

输入：
- factor_selection/{date}/{timestamp}/standardized/

输出：
- wfo/{timestamp}/wfo_results.pkl
- wfo/{timestamp}/wfo_report.txt
- wfo/{timestamp}/metadata.json
"""

import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent

from etf_strategy.core.constrained_walk_forward_optimizer import ConstrainedWalkForwardOptimizer


def setup_logging(output_dir: Path):
    """设置详细日志"""
    log_file = output_dir / "step3_wfo.log"

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def find_latest_selection(results_dir: Path):
    """查找最新的因子筛选数据目录"""
    selection_root = results_dir / "factor_selection"

    if not selection_root.exists():
        return None

    # 查找所有时间戳目录
    all_runs = []
    for date_dir in selection_root.iterdir():
        if not date_dir.is_dir():
            continue
        for timestamp_dir in date_dir.iterdir():
            if not timestamp_dir.is_dir():
                continue
            # 验证是否包含必要文件
            if (timestamp_dir / "metadata.json").exists():
                all_runs.append(timestamp_dir)

    if not all_runs:
        return None

    # 按时间戳排序，返回最新
    all_runs.sort(key=lambda p: p.name, reverse=True)
    return all_runs[0]


def load_standardized_factors(selection_dir: Path, logger):
    """加载标准化因子数据"""
    logger.info("-" * 80)
    logger.info("阶段 1/3: 加载标准化因子")
    logger.info("-" * 80)
    logger.info(f"输入目录: {selection_dir}")
    logger.info("")

    # 加载元数据
    metadata_path = selection_dir / "metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    logger.info("📋 因子筛选元数据:")
    logger.info(f"  时间戳: {metadata['timestamp']}")
    logger.info(f"  ETF数量: {metadata['etf_count']}")
    logger.info(
        f"  日期范围: {metadata['date_range'][0]} -> {metadata['date_range'][1]}"
    )
    logger.info(f"  标准化因子数: {metadata['standardized_factor_count']}")
    logger.info("")

    # 加载标准化因子
    standardized_dir = selection_dir / "standardized"
    factors_dict = {}

    for factor_name in metadata["standardized_factor_names"]:
        parquet_path = standardized_dir / f"{factor_name}.parquet"
        factor_df = pd.read_parquet(parquet_path)
        # 因子文件是DataFrame（日期×标的），直接使用
        factors_dict[factor_name] = factor_df

        nan_ratio = factor_df.isna().sum().sum() / factor_df.size
        logger.info(f"  ✅ {factor_name:25s} NaN率: {nan_ratio*100:.2f}%")

    logger.info("")

    return factors_dict, metadata


def run_wfo_optimization(factors_dict, metadata, ohlcv_data, output_dir, logger):
    """执行WFO优化"""
    logger.info("-" * 80)
    logger.info("阶段 2/3: WFO优化（Walk-Forward Optimization）")
    logger.info("-" * 80)

    # WFO参数
    in_sample_days = 252
    out_of_sample_days = 60
    step_days = 20
    target_factor_count = 5
    ic_threshold = 0.05

    logger.info("WFO参数配置:")
    logger.info(f"  样本内窗口: {in_sample_days} 天")
    logger.info(f"  样本外窗口: {out_of_sample_days} 天")
    logger.info(f"  滑动步长: {step_days} 天")
    logger.info(f"  目标因子数: {target_factor_count}")
    logger.info(f"  IC阈值: {ic_threshold}")
    logger.info("")

    # 准备数据：转换为3D numpy数组
    factor_names = list(factors_dict.keys())
    close_df = ohlcv_data["close"]
    returns_df = close_df.pct_change()

    # 🔧 修复：pct_change()第一行是NaN，需要对齐因子和收益率的时间索引
    # 跳过第一行，确保因子和收益率时间对齐
    returns_df = returns_df.iloc[1:]
    aligned_factors_dict = {k: v.iloc[1:] for k, v in factors_dict.items()}

    n_dates = len(returns_df)  # 使用对齐后的长度
    n_symbols = len(close_df.columns)
    n_factors = len(factor_names)

    import numpy as np

    factors_3d = np.full((n_dates, n_symbols, n_factors), np.nan)
    for idx, fname in enumerate(factor_names):
        factors_3d[:, :, idx] = aligned_factors_dict[fname].values

    logger.info("数据形状:")
    logger.info(f"  时间步: {n_dates}")
    logger.info(f"  标的数: {n_symbols}")
    logger.info(f"  因子数: {n_factors}")
    logger.info("")

    # 创建优化器
    optimizer = ConstrainedWalkForwardOptimizer()

    logger.info("🔄 开始WFO优化...")
    logger.info("")

    import time

    start_time = time.time()

    # 执行优化
    wfo_df, constraint_reports = optimizer.run_constrained_wfo(
        factors_data=factors_3d,
        returns=returns_df.values,
        factor_names=factor_names,
        is_period=in_sample_days,
        oos_period=out_of_sample_days,
        step_size=step_days,
        target_factor_count=target_factor_count,
    )

    elapsed = time.time() - start_time
    logger.info("")
    logger.info(f"✅ WFO优化完成（耗时 {elapsed:.1f}秒）")
    logger.info("")

    # 转换结果格式
    results = {
        "results_df": wfo_df,
        "constraint_reports": constraint_reports,
        "total_windows": len(wfo_df),
        "valid_windows": len(wfo_df),
    }

    if len(wfo_df) > 0:
        results["avg_oos_ic"] = wfo_df["oos_ic_mean"].mean()
        results["std_oos_ic"] = wfo_df["oos_ic_mean"].std()
        results["avg_ic_decay"] = wfo_df["ic_drop"].mean()
        results["std_ic_decay"] = wfo_df["ic_drop"].std()

        # 计算因子选择频率
        selected_lists = [
            [f.strip() for f in factors_str.split(",") if f.strip()]
            for factors_str in wfo_df["selected_factors"]
        ]
        selected_flat = [f for factors in selected_lists for f in factors]
        if selected_flat:
            factor_counts = (
                pd.Series(selected_flat).value_counts().sort_values(ascending=False)
            )
            selection_freq = [
                (fname, count / len(wfo_df)) for fname, count in factor_counts.items()
            ]
            results["factor_selection_freq"] = selection_freq
            results["top_oos_factors"] = [
                (fname, results["avg_oos_ic"]) for fname, _ in selection_freq[:5]
            ]
        else:
            results["factor_selection_freq"] = []
            results["top_oos_factors"] = []

        # 窗口详细结果
        results["window_results"] = []
        for idx, row in wfo_df.iterrows():
            results["window_results"].append(
                {
                    "window_id": idx + 1,
                    "is_start": row.get("is_start", ""),
                    "is_end": row.get("is_end", ""),
                    "oos_start": row.get("oos_start", ""),
                    "oos_end": row.get("oos_end", ""),
                    "selected_factors": [
                        f.strip() for f in row["selected_factors"].split(",")
                    ],
                    "oos_ic": row["oos_ic_mean"],
                    "ic_decay": row["ic_drop"],
                }
            )
    else:
        results["avg_oos_ic"] = 0.0
        results["std_oos_ic"] = 0.0
        results["avg_ic_decay"] = 0.0
        results["std_ic_decay"] = 0.0
        results["factor_selection_freq"] = []
        results["top_oos_factors"] = []
        results["window_results"] = []

    # 统计汇总
    logger.info("📊 WFO结果统计:")
    logger.info(f"  窗口总数: {results['total_windows']}")
    logger.info(f"  有效窗口: {results['valid_windows']}")
    logger.info("")

    logger.info("IC统计（样本外）:")
    logger.info(f"  平均OOS IC: {results['avg_oos_ic']:.4f}")
    logger.info(f"  OOS IC 标准差: {results['std_oos_ic']:.4f}")
    logger.info(f"  平均IC衰减: {results['avg_ic_decay']:.4f}")
    logger.info(f"  IC衰减标准差: {results['std_ic_decay']:.4f}")
    logger.info("")

    logger.info("TOP 5 样本外IC因子:")
    for idx, (fname, ic_val) in enumerate(results["top_oos_factors"][:5], start=1):
        logger.info(f"  {idx}. {fname:25s} IC={ic_val:.4f}")
    logger.info("")

    logger.info("因子选择频率（TOP 10）:")
    for idx, (fname, freq) in enumerate(results["factor_selection_freq"][:10], start=1):
        logger.info(f"  {idx:02d}. {fname:25s} {freq*100:6.2f}%")
    logger.info("")

    # 保存详细报告
    report_path = output_dir / "wfo_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("WFO优化详细报告\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"窗口总数: {results['total_windows']}\n")
        f.write(f"有效窗口: {results['valid_windows']}\n\n")

        f.write("IC统计:\n")
        f.write(f"  平均OOS IC: {results['avg_oos_ic']:.4f}\n")
        f.write(f"  OOS IC 标准差: {results['std_oos_ic']:.4f}\n")
        f.write(f"  平均IC衰减: {results['avg_ic_decay']:.4f}\n")
        f.write(f"  IC衰减标准差: {results['std_ic_decay']:.4f}\n\n")

        f.write("因子选择频率:\n")
        for fname, freq in results["factor_selection_freq"]:
            f.write(f"  {fname:25s} {freq*100:6.2f}%\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("各窗口详细结果\n")
        f.write("=" * 80 + "\n\n")

        for window_result in results["window_results"]:
            f.write(f"窗口 {window_result['window_id']}:\n")
            f.write(f"  IS: {window_result['is_start']} -> {window_result['is_end']}\n")
            f.write(
                f"  OOS: {window_result['oos_start']} -> {window_result['oos_end']}\n"
            )
            f.write(f"  选中因子: {', '.join(window_result['selected_factors'])}\n")
            f.write(f"  OOS IC: {window_result['oos_ic']:.4f}\n")
            f.write(f"  IC衰减: {window_result['ic_decay']:.4f}\n\n")

    logger.info(f"✅ 详细报告已保存: {report_path}")
    logger.info("")

    # 保存结果对象
    results_path = output_dir / "wfo_results.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(results, f)

    logger.info(f"💾 WFO结果对象已保存: {results_path}")
    logger.info("")

    return results


def save_wfo_metadata(results, metadata, output_dir, logger):
    """保存WFO元数据"""
    logger.info("-" * 80)
    logger.info("阶段 3/3: 保存WFO元数据")
    logger.info("-" * 80)

    wfo_metadata = {
        **metadata,
        "step": "wfo",
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "total_windows": results["total_windows"],
        "valid_windows": results["valid_windows"],
        "avg_oos_ic": results["avg_oos_ic"],
        "std_oos_ic": results["std_oos_ic"],
        "avg_ic_decay": results["avg_ic_decay"],
        "std_ic_decay": results["std_ic_decay"],
        "top_oos_factors": results["top_oos_factors"][:5],
        "factor_selection_freq": results["factor_selection_freq"],
        "output_dir": str(output_dir),
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(wfo_metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ 元数据已保存: {metadata_path}")
    logger.info("")

    return wfo_metadata


def main(selection_dir: Path = None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 输出目录
    output_root = PROJECT_ROOT / "results"
    wfo_dir = output_root / "wfo" / timestamp
    wfo_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(wfo_dir)

    logger.info("=" * 80)
    logger.info("Step 3: WFO优化执行（Walk-Forward Optimization）")
    logger.info("=" * 80)
    logger.info(f"输出目录: {wfo_dir}")
    logger.info(f"时间戳: {timestamp}")
    logger.info("")

    # 查找输入数据
    if selection_dir is None:
        logger.info("🔍 自动查找最新的因子筛选数据...")
        selection_dir = find_latest_selection(output_root)

        if selection_dir is None:
            logger.error("❌ 未找到因子筛选数据！请先运行 step2_factor_selection.py")
            sys.exit(1)

        logger.info(f"✅ 找到最新数据: {selection_dir}")
        logger.info("")

    # 1. 加载数据
    factors_dict, metadata = load_standardized_factors(selection_dir, logger)

    # 加载OHLCV数据（需要用于计算returns）
    # 使用find_latest_cross_section函数查找横截面数据
    cross_section_root = output_root / "cross_section"
    cross_section_dir = None

    if cross_section_root.exists():
        # 查找所有时间戳目录
        all_runs = []
        for date_dir in cross_section_root.iterdir():
            if not date_dir.is_dir():
                continue
            for timestamp_dir in date_dir.iterdir():
                if not timestamp_dir.is_dir():
                    continue
                # 验证是否包含必要文件
                if (timestamp_dir / "metadata.json").exists():
                    all_runs.append(timestamp_dir)

        if all_runs:
            # 按时间戳排序，返回最新
            all_runs.sort(key=lambda p: p.name, reverse=True)
            cross_section_dir = all_runs[0]

    if cross_section_dir is None:
        logger.error("❌ 无法找到横截面数据！请先运行 step1_cross_section.py")
        sys.exit(1)

    ohlcv_dir = cross_section_dir / "ohlcv"
    ohlcv_data = {}
    for col_name in ["open", "high", "low", "close", "volume"]:
        parquet_path = ohlcv_dir / f"{col_name}.parquet"
        ohlcv_data[col_name] = pd.read_parquet(parquet_path)

    logger.info(f"✅ 加载OHLCV数据: {ohlcv_data['close'].shape}")
    logger.info("")

    # 2. 运行WFO
    results = run_wfo_optimization(factors_dict, metadata, ohlcv_data, wfo_dir, logger)

    # 3. 保存元数据
    wfo_metadata = save_wfo_metadata(results, metadata, wfo_dir, logger)

    # 完成
    logger.info("=" * 80)
    logger.info("✅ Step 3 完成！WFO优化已执行")
    logger.info("=" * 80)
    logger.info(f"输出目录: {wfo_dir}")
    logger.info(f"  - wfo_results.pkl: WFO结果对象")
    logger.info(f"  - wfo_report.txt: 详细报告")
    logger.info(f"  - metadata.json: 元数据")
    logger.info(f"  - step3_wfo.log: 执行日志")
    logger.info("")
    logger.info("🎉 完整的3步流程执行完成！")
    logger.info("")
    logger.info("📊 关键结果:")
    logger.info(f"  - 窗口数: {results['total_windows']}")
    logger.info(f"  - 平均OOS IC: {results['avg_oos_ic']:.4f}")
    logger.info(f"  - IC衰减: {results['avg_ic_decay']:.4f}")
    logger.info(
        f"  - TOP因子: {', '.join([f[0] for f in results['top_oos_factors'][:3]])}"
    )
    logger.info("")

    return wfo_dir


if __name__ == "__main__":
    main()
