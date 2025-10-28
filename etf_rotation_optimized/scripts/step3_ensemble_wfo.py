"""
Step 3: Ensemble Walk-Forward Optimization

集成前向回测优化 - 1000组合采样 + Top10集成

输入: step2的标准化因子数据
输出: ensemble_wfo结果 (CSV + JSON)

运行:
    python scripts/step3_ensemble_wfo.py
    python scripts/step3_ensemble_wfo.py --factor-selection-dir results/factor_selection/20250128/20250128_120000
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.ensemble_wfo_optimizer import EnsembleWFOOptimizer


def setup_logging(output_dir: Path) -> logging.Logger:
    """配置日志系统"""
    log_file = output_dir / "step3_ensemble_wfo.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    return logging.getLogger(__name__)


def find_latest_factor_selection(output_root: Path) -> Path:
    """查找最新的因子选择结果目录"""
    factor_selection_root = output_root / "factor_selection"

    if not factor_selection_root.exists():
        return None

    # 查找最新的日期目录
    date_dirs = sorted(
        [d for d in factor_selection_root.iterdir() if d.is_dir() and d.name.isdigit()],
        reverse=True,
    )

    if not date_dirs:
        return None

    # 查找该日期下最新的时间戳目录
    latest_date_dir = date_dirs[0]
    timestamp_dirs = sorted(
        [d for d in latest_date_dir.iterdir() if d.is_dir()], reverse=True
    )

    if not timestamp_dirs:
        return None

    return timestamp_dirs[0]


def load_factor_selection_data(
    factor_selection_dir: Path, logger: logging.Logger
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict]:
    """
    加载step2的因子选择数据

    Returns:
        (ohlcv_data, factors_dict, metadata)
    """
    logger.info("-" * 80)
    logger.info("阶段 1/4: 加载因子选择数据")
    logger.info("-" * 80)
    logger.info(f"数据目录: {factor_selection_dir}")

    # 1. 加载OHLCV
    ohlcv_path = factor_selection_dir / "standardized" / "OHLCV.parquet"
    if not ohlcv_path.exists():
        raise FileNotFoundError(f"OHLCV数据不存在: {ohlcv_path}")

    ohlcv_data = pd.read_parquet(ohlcv_path)
    logger.info(f"✅ OHLCV数据: {ohlcv_data.shape}")

    # 2. 加载所有因子
    factors_dir = factor_selection_dir / "standardized"
    factors_dict = {}

    for factor_file in sorted(factors_dir.glob("*.parquet")):
        if factor_file.stem == "OHLCV":
            continue

        factor_name = factor_file.stem
        df = pd.read_parquet(factor_file)
        factors_dict[factor_name] = df

    logger.info(f"✅ 加载 {len(factors_dict)} 个因子")

    # 3. 加载元数据
    metadata_path = factor_selection_dir / "metadata.json"
    if metadata_path.exists():
        import json

        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    logger.info(f"✅ 元数据加载完成")
    logger.info("")

    return ohlcv_data, factors_dict, metadata


def prepare_wfo_data(
    ohlcv_data: pd.DataFrame, factors_dict: Dict[str, pd.DataFrame], logger: logging.Logger
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    准备WFO数据格式

    Returns:
        (factors_array, returns_array, factor_names)
        - factors_array: (T, N, K) ndarray
        - returns_array: (T, N) ndarray
        - factor_names: List[str]
    """
    logger.info("-" * 80)
    logger.info("阶段 2/4: 准备WFO数据格式")
    logger.info("-" * 80)

    # 1. 提取收益率
    if "RET_1D" in ohlcv_data.columns:
        returns_df = ohlcv_data["RET_1D"]
    else:
        logger.warning("⚠️  OHLCV中无RET_1D列，使用Close计算收益率")
        returns_df = ohlcv_data["Close"].pct_change()

    returns_array = returns_df.values  # (T, N)
    logger.info(f"✅ 收益率数组: {returns_array.shape}")

    # 2. 堆叠因子为3D数组
    factor_names = sorted(factors_dict.keys())
    factor_arrays = []

    for factor_name in factor_names:
        factor_df = factors_dict[factor_name]
        factor_arrays.append(factor_df.values)  # (T, N)

    # 堆叠: (K, T, N) → (T, N, K)
    factors_array = np.stack(factor_arrays, axis=0)  # (K, T, N)
    factors_array = np.transpose(factors_array, (1, 2, 0))  # (T, N, K)

    logger.info(f"✅ 因子数组: {factors_array.shape}")
    logger.info(f"   - 时间步: {factors_array.shape[0]}")
    logger.info(f"   - 资产数: {factors_array.shape[1]}")
    logger.info(f"   - 因子数: {factors_array.shape[2]}")
    logger.info("")

    return factors_array, returns_array, factor_names


def load_constraints_config(logger: logging.Logger) -> Dict:
    """加载因子约束配置"""
    constraints_path = PROJECT_ROOT / "configs" / "FACTOR_SELECTION_CONSTRAINTS.yaml"

    if not constraints_path.exists():
        logger.warning(f"⚠️  约束配置不存在: {constraints_path}")
        logger.warning("⚠️  将使用空约束配置")
        return {"family_quotas": {}, "mutually_exclusive_pairs": []}

    with open(constraints_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 格式转换: family_quota → family_quotas (兼容EnsembleSampler)
    if "family_quota" in config:
        family_quota_data = config["family_quota"]
        
        # 转换为EnsembleSampler期望的格式
        family_quotas = {}
        for family_name, family_config in family_quota_data.items():
            if isinstance(family_config, dict) and family_config.get("enabled", True):
                family_quotas[family_name] = {
                    "max_count": family_config.get("max_count", 2),
                    "candidates": family_config.get("factors", [])
                }
        
        config["family_quotas"] = family_quotas
    else:
        config["family_quotas"] = {}
    
    # 确保mutually_exclusive_pairs存在
    if "mutually_exclusive_pairs" not in config:
        config["mutually_exclusive_pairs"] = []

    logger.info(f"✅ 加载约束配置: {constraints_path}")
    logger.info(f"   - 家族配额: {len(config.get('family_quotas', {}))} 个")
    logger.info(f"   - 互斥对: {len(config.get('mutually_exclusive_pairs', []))} 对")

    return config


def run_ensemble_wfo(
    factors_array: np.ndarray,
    returns_array: np.ndarray,
    factor_names: list,
    constraints_config: Dict,
    output_dir: Path,
    logger: logging.Logger,
    n_samples: int = 1000,
    combo_size: int = 5,
    top_k: int = 10,
    weighting_scheme: str = "gradient_decay",
    is_period: int = 100,
    oos_period: int = 20,
    step_size: int = 20,
) -> pd.DataFrame:
    """
    运行Ensemble WFO优化

    Args:
        n_samples: 每窗口采样组合数 (默认1000)
        combo_size: 每组合因子数 (默认5)
        top_k: 集成的最优组合数 (默认10)
        weighting_scheme: 加权方案 (默认gradient_decay)
        is_period: IS窗口长度 (默认100)
        oos_period: OOS窗口长度 (默认20)
        step_size: 滑动步长 (默认20)

    Returns:
        汇总DataFrame
    """
    logger.info("-" * 80)
    logger.info("阶段 3/4: 运行Ensemble WFO优化")
    logger.info("-" * 80)
    logger.info(f"采样配置: {n_samples}个组合 × {combo_size}因子")
    logger.info(f"集成配置: Top{top_k}, 权重={weighting_scheme}")
    logger.info(f"窗口配置: IS={is_period}, OOS={oos_period}, step={step_size}")
    logger.info("")

    # 创建优化器
    optimizer = EnsembleWFOOptimizer(
        constraints_config=constraints_config,
        n_samples=n_samples,
        combo_size=combo_size,
        top_k=top_k,
        weighting_scheme=weighting_scheme,
        random_seed=42,
        verbose=True,
    )

    # 运行WFO
    summary_df = optimizer.run_ensemble_wfo(
        factors_data=factors_array,
        returns=returns_array,
        factor_names=factor_names,
        is_period=is_period,
        oos_period=oos_period,
        step_size=step_size,
    )

    # 保存结果
    optimizer.save_results(output_dir)

    logger.info("")
    logger.info(f"✅ Ensemble WFO优化完成")
    logger.info(f"   - 总窗口数: {len(summary_df)}")
    logger.info(f"   - 平均OOS IC: {summary_df['oos_ensemble_ic'].mean():.4f}")
    logger.info(
        f"   - 平均OOS Sharpe: {summary_df['oos_ensemble_sharpe'].mean():.2f}"
    )
    logger.info("")

    return summary_df


def generate_summary_report(
    summary_df: pd.DataFrame, output_dir: Path, logger: logging.Logger
):
    """生成汇总报告"""
    logger.info("-" * 80)
    logger.info("阶段 4/4: 生成汇总报告")
    logger.info("-" * 80)

    # 1. 性能统计
    stats = {
        "total_windows": len(summary_df),
        "mean_oos_ic": summary_df["oos_ensemble_ic"].mean(),
        "std_oos_ic": summary_df["oos_ensemble_ic"].std(),
        "mean_oos_sharpe": summary_df["oos_ensemble_sharpe"].mean(),
        "std_oos_sharpe": summary_df["oos_ensemble_sharpe"].std(),
        "positive_ic_ratio": (summary_df["oos_ensemble_ic"] > 0).mean(),
    }

    logger.info("📊 性能统计:")
    logger.info(f"   - 总窗口数: {stats['total_windows']}")
    logger.info(
        f"   - OOS IC: {stats['mean_oos_ic']:.4f} ± {stats['std_oos_ic']:.4f}"
    )
    logger.info(
        f"   - OOS Sharpe: {stats['mean_oos_sharpe']:.2f} ± {stats['std_oos_sharpe']:.2f}"
    )
    logger.info(f"   - 正IC比率: {stats['positive_ic_ratio']:.1%}")
    logger.info("")

    # 2. 保存统计
    stats_path = output_dir / "performance_stats.json"
    import json

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ 性能统计已保存: {stats_path}")

    # 3. 绘制性能曲线 (可选,如果有matplotlib)
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # IC曲线
        axes[0].plot(summary_df.index, summary_df["oos_ensemble_ic"], marker="o")
        axes[0].axhline(0, color="red", linestyle="--", alpha=0.5)
        axes[0].set_title("OOS Ensemble IC")
        axes[0].set_ylabel("IC")
        axes[0].grid(True, alpha=0.3)

        # Sharpe曲线
        axes[1].plot(
            summary_df.index, summary_df["oos_ensemble_sharpe"], marker="s", color="green"
        )
        axes[1].axhline(0, color="red", linestyle="--", alpha=0.5)
        axes[1].set_title("OOS Ensemble Sharpe")
        axes[1].set_xlabel("Window Index")
        axes[1].set_ylabel("Sharpe")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / "performance_curves.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"✅ 性能曲线已保存: {plot_path}")

    except ImportError:
        logger.warning("⚠️  matplotlib未安装,跳过性能曲线绘制")

    logger.info("")


def main(factor_selection_dir: Path = None):
    """主入口"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_date = timestamp[:8]

    # 输出目录
    output_root = PROJECT_ROOT / "results"
    ensemble_wfo_dir = output_root / "ensemble_wfo" / run_date / timestamp
    ensemble_wfo_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(ensemble_wfo_dir)

    logger.info("=" * 80)
    logger.info("Step 3: Ensemble Walk-Forward Optimization")
    logger.info("=" * 80)
    logger.info(f"输出目录: {ensemble_wfo_dir}")
    logger.info(f"时间戳: {timestamp}")
    logger.info("")

    # 查找输入数据
    if factor_selection_dir is None:
        logger.info("🔍 自动查找最新的因子选择数据...")
        factor_selection_dir = find_latest_factor_selection(output_root)

        if factor_selection_dir is None:
            logger.error("❌ 未找到因子选择数据！请先运行 step2_factor_selection.py")
            sys.exit(1)

        logger.info(f"✅ 找到最新数据: {factor_selection_dir}")
        logger.info("")

    # 1. 加载数据
    ohlcv_data, factors_dict, metadata = load_factor_selection_data(
        factor_selection_dir, logger
    )

    # 2. 准备WFO数据
    factors_array, returns_array, factor_names = prepare_wfo_data(
        ohlcv_data, factors_dict, logger
    )

    # 3. 加载约束配置
    constraints_config = load_constraints_config(logger)
    logger.info("")

    # 4. 运行Ensemble WFO
    summary_df = run_ensemble_wfo(
        factors_array=factors_array,
        returns_array=returns_array,
        factor_names=factor_names,
        constraints_config=constraints_config,
        output_dir=ensemble_wfo_dir,
        logger=logger,
    )

    # 5. 生成汇总报告
    generate_summary_report(summary_df, ensemble_wfo_dir, logger)

    # 完成
    logger.info("=" * 80)
    logger.info("✅ Step 3 完成")
    logger.info("=" * 80)
    logger.info(f"结果保存至: {ensemble_wfo_dir}")
    logger.info(f"   - ensemble_wfo_summary.csv")
    logger.info(f"   - ensemble_wfo_detailed.json")
    logger.info(f"   - performance_stats.json")
    logger.info("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Step 3: Ensemble WFO")
    parser.add_argument(
        "--factor-selection-dir",
        type=Path,
        default=None,
        help="因子选择结果目录 (默认自动查找最新)",
    )

    args = parser.parse_args()

    main(factor_selection_dir=args.factor_selection_dir)
