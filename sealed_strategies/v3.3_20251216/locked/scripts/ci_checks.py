#!/usr/bin/env python3
"""CI检查脚本 - 生产保险丝

检查项：
1. 静态扫描：强制shift(1)
2. 覆盖率骤降：≥10%
3. 有效因子数：<8
4. 目标波动缩放系数：<0.6
5. 月收益：>30%
"""

import logging
import sys
from pathlib import Path

import pandas as pd
from path_utils import get_ci_thresholds, get_paths
from vec_bt_alignment_check import compare_vec_bt_results

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

BANNED_SIMPLIFIED_KEYWORDS = (
    "run_unified",
    "direct_factor",
    "single-pass",
    "single_pass",
    "simple_wfo",
    "simplified_wfo",
    "_simple",
    "_simplified",
)

ALLOWED_ARCHIVE_PREFIXES = (
    ("scripts", "archive"),
    ("docs", "archive"),
    ("results",),
    ("raw",),
    ("etf_strategy", "docs", "archive"),
    ("etf_strategy", "scripts", "archive"),
    (".venv",),
)


def _has_allowed_prefix(parts, prefix):
    lowered_parts = tuple(part.lower() for part in parts)
    lowered_prefix = tuple(part.lower() for part in prefix)
    plen = len(lowered_prefix)
    for idx in range(len(lowered_parts) - plen + 1):
        if lowered_parts[idx : idx + plen] == lowered_prefix:
            return True
    return False


def _is_under_allowed_directory(parts):
    if not parts:
        return False
    if parts[0].lower().startswith("backup_"):
        return True
    for prefix in ALLOWED_ARCHIVE_PREFIXES:
        if _has_allowed_prefix(parts, prefix):
            return True
    return False


def detect_simplified_artifacts(project_root: Path):
    """Return a list of relative paths that violate the no-simplified policy."""

    hits = []
    for path in project_root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(project_root)
        if _is_under_allowed_directory(rel.parts):
            continue
        lowered = rel.as_posix().lower()
        if any(keyword in lowered for keyword in BANNED_SIMPLIFIED_KEYWORDS):
            hits.append(rel)
    return hits


def ci_checks(
    output_dir="factor_output/etf_rotation_production",
    min_annual_return=0.08,
    max_drawdown=-0.30,
    min_sharpe=0.50,
    min_winrate=0.45,
    *,
    results_root="results",
    vec_results_pattern="vec_full_backtest_*",
    bt_results_pattern="bt_backtest_full_*",
    vec_bt_threshold_pp=0.05,
    skip_vec_bt_check=False,
    skip_simplified_scan=False,
):
    """CI检查

    Args:
        output_dir: 输出目录（默认 factor_output/etf_rotation_production）
    """

    logger.info("=" * 80)
    logger.info("CI检查 - 生产保险丝")
    logger.info(f"检查目录: {output_dir}")
    logger.info("=" * 80)

    output_dir = Path(output_dir)
    all_passed = True

    if not skip_simplified_scan:
        logger.info("\n0. 简化版/备份脚本扫描")
        project_root = Path(__file__).resolve().parents[1]
        hits = detect_simplified_artifacts(project_root)
        if hits:
            logger.error("   ❌ 发现违规命名的脚本/文件：")
            for rel in hits:
                logger.error(f"      - {rel}")
            all_passed = False
        else:
            logger.info("   ✅ 未检测到违规命名")

    # 1. 静态扫描：强制shift(1)
    logger.info(f"\n1. 静态扫描：强制shift(1)")
    adapter_file = Path(
        "factor_system/factor_engine/adapters/vbt_adapter_production.py"
    )

    if adapter_file.exists():
        with open(adapter_file) as f:
            content = f.read()

        if "_apply_t1_shift" in content and (
            "result[min_history:]" in content or "np.roll" in content
        ):
            logger.info("   ✅ 适配器包含T+1 shift逻辑")
        else:
            logger.error("   ❌ 适配器缺少T+1 shift逻辑")
            all_passed = False
    else:
        logger.error("   ❌ 适配器文件不存在")
        all_passed = False

    # 2. 覆盖率骤降检查
    logger.info(f"\n2. 覆盖率骤降检查（≥10%）")
    summary_files = list(output_dir.glob("factor_summary_*.csv"))
    summary_file = summary_files[0] if summary_files else None

    if summary_file and summary_file.exists():
        summary = pd.read_csv(summary_file)
        avg_coverage = summary["coverage"].mean()

        logger.info(f"   平均覆盖率: {avg_coverage:.2%}")

        if avg_coverage >= 0.80:
            logger.info("   ✅ 覆盖率正常（≥80%）")
        elif avg_coverage >= 0.70:
            logger.warning("   ⚠️  覆盖率偏低（70-80%）")
        else:
            logger.error("   ❌ 覆盖率过低（<70%）")
            all_passed = False
    else:
        logger.error("   ❌ 因子概要文件不存在")
        all_passed = False

    # 3. 有效因子数检查
    logger.info(f"\n3. 有效因子数检查（<8）")
    production_file = output_dir / "production_factors.txt"

    if production_file.exists():
        with open(production_file) as f:
            production_factors = [line.strip() for line in f if line.strip()]

        logger.info(f"   生产因子数: {len(production_factors)}")

        if len(production_factors) >= 8:
            logger.info("   ✅ 因子数充足（≥8）")
        else:
            logger.error("   ❌ 因子数不足（<8）")
            all_passed = False
    else:
        logger.error("   ❌ 生产因子列表不存在")
        all_passed = False

    # 4/5. 基于回测的真实指标检查（从 backtest_metrics.json 读取）
    logger.info(f"\n4. 回测指标阈值检查（真实数据）")
    metrics_file = output_dir / "backtest_metrics.json"
    if metrics_file.exists():
        import json

        try:
            meta = json.loads(metrics_file.read_text())
            metrics = meta.get("metrics", {})

            # 解析工具
            def parse_pct(s):
                if isinstance(s, (int, float)):
                    return float(s)
                if isinstance(s, str) and s.endswith("%"):
                    return float(s.strip("%")) / 100.0
                return float(s)

            def parse_float(s):
                return float(s)

            # 读取指标
            ann_return = parse_pct(metrics.get("年化收益", "0%"))
            max_dd = parse_pct(metrics.get("最大回撤", "0%"))
            sharpe = parse_float(metrics.get("夏普比率", "0"))
            win_rate = parse_pct(metrics.get("月胜率", "0%"))
            # 阈值（可参数化，当前使用默认门槛）
            thresholds = {
                "min_年化收益": float(min_annual_return),
                "min_月胜率": float(min_winrate),
                "min_夏普比率": float(min_sharpe),
                "max_最大回撤": float(max_drawdown),  # 注意：回撤为负值，要求不小于该值
            }
            logger.info(
                f"   年化收益: {ann_return:.2%} (阈值≥{thresholds['min_年化收益']:.0%})"
            )
            logger.info(
                f"   最大回撤: {max_dd:.2%} (阈值≥{thresholds['max_最大回撤']:.0%})"
            )
            logger.info(
                f"   夏普比率: {sharpe:.2f} (阈值≥{thresholds['min_夏普比率']:.2f})"
            )
            logger.info(
                f"   月胜率: {win_rate:.2%} (阈值≥{thresholds['min_月胜率']:.0%})"
            )
            # 校验
            if ann_return < thresholds["min_年化收益"]:
                logger.error("   ❌ 年化收益未达标")
                all_passed = False
            if max_dd < thresholds["max_最大回撤"]:
                logger.error("   ❌ 最大回撤超标（跌幅过大）")
                all_passed = False
            if sharpe < thresholds["min_夏普比率"]:
                logger.error("   ❌ 夏普比率未达标")
                all_passed = False
            if win_rate < thresholds["min_月胜率"]:
                logger.error("   ❌ 月胜率未达标")
                all_passed = False
        except Exception as e:
            logger.error(f"   ❌ 回测指标解析失败: {e}")
            all_passed = False
    else:
        logger.warning("   ⚠️ 未找到回测指标文件，跳过回测指标检查")

    # 6. 索引规范检查
    logger.info(f"\n6. 索引规范检查")
    panel_files = list(output_dir.glob("panel_*.parquet"))
    panel_file = panel_files[0] if panel_files else None

    if panel_file and panel_file.exists():
        panel = pd.read_parquet(panel_file)

        if panel.index.names == ["symbol", "date"]:
            logger.info("   ✅ 索引名称正确: (symbol, date)")
        else:
            logger.error(f"   ❌ 索引名称错误: {panel.index.names}")
            all_passed = False

        if panel.index.is_unique:
            logger.info("   ✅ 索引唯一")
        else:
            logger.error("   ❌ 索引存在重复")
            all_passed = False
    else:
        logger.error("   ❌ 面板文件不存在")
        all_passed = False

    # 7. 零方差检查
    logger.info(f"\n7. 零方差检查")
    if summary_file and summary_file.exists():
        zero_var_count = summary["zero_variance"].sum()

        logger.info(f"   零方差因子数: {zero_var_count}/{len(summary)}")

        if zero_var_count == 0:
            logger.info("   ✅ 无零方差因子")
        elif zero_var_count < 10:
            logger.warning(f"   ⚠️  存在{zero_var_count}个零方差因子")
        else:
            logger.error(f"   ❌ 零方差因子过多（{zero_var_count}个）")
            all_passed = False

    # 8. 元数据完整性
    logger.info(f"\n8. 元数据完整性")
    meta_file = output_dir / "panel_meta.json"
    required_meta_fields = [
        "engine_version",
        "price_field",
        "price_field_priority",
        "generated_at",
        "data_range",
        "run_params",
        "panel_columns_hash",
    ]
    if meta_file.exists():
        import json

        meta = json.loads(meta_file.read_text())
        missing = [
            k
            for k in required_meta_fields
            if k not in meta or meta[k] in (None, "", {})
        ]
        if missing:
            logger.error(f"   ❌ 缺少关键meta字段: {missing}")
            all_passed = False
        else:
            logger.info("   ✅ 元数据字段完整")
    else:
        logger.error("   ❌ 元数据文件不存在")
        all_passed = False

    # 9. VEC/BT 对齐检查
    if not skip_vec_bt_check:
        logger.info("\n9. VEC/BT 对齐检查（阈值 %.3f pp）", vec_bt_threshold_pp)
        try:
            summary = compare_vec_bt_results(
                results_root=Path(results_root),
                vec_pattern=vec_results_pattern,
                bt_pattern=bt_results_pattern,
                threshold_pp=vec_bt_threshold_pp,
            )
            logger.info(
                "   ✅ 对齐通过：%s vs %s", summary.vec_dir.name, summary.bt_dir.name
            )
            logger.info(
                "   统计：avg=%.4fpp median=%.4fpp max=%.4fpp combos=%d",
                summary.avg_abs_diff_pp,
                summary.median_abs_diff_pp,
                summary.max_abs_diff_pp,
                summary.total_pairs,
            )
        except (FileNotFoundError, ValueError) as exc:
            logger.error("   ❌ %s", exc)
            all_passed = False

    logger.info(f"\n{'=' * 80}")
    if all_passed:
        logger.info("✅ CI检查通过")
    else:
        logger.error("❌ CI检查失败")
    logger.info("=" * 80)

    return all_passed


if __name__ == "__main__":
    import argparse

    # 从配置文件读取默认值
    paths = get_paths()
    thresholds = get_ci_thresholds()

    parser = argparse.ArgumentParser(description="CI检查 - 生产保险丝")
    parser.add_argument(
        "--output-dir",
        default=str(paths["output_root"]),
        help="输出目录（默认从配置文件读取）",
    )
    parser.add_argument(
        "--min-annual-return",
        type=float,
        default=thresholds["min_annual_return"],
        help="年化收益最小阈值",
    )
    parser.add_argument(
        "--max-drawdown",
        type=float,
        default=thresholds["max_drawdown"],
        help="最大回撤下限（负数）",
    )
    parser.add_argument(
        "--min-sharpe",
        type=float,
        default=thresholds["min_sharpe"],
        help="夏普最小阈值",
    )
    parser.add_argument(
        "--min-winrate",
        type=float,
        default=thresholds["min_winrate"],
        help="月胜率最小阈值",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="results",
        help="VEC/BT 结果根目录",
    )
    parser.add_argument(
        "--vec-pattern",
        type=str,
        default="vec_full_backtest_*",
        help="VEC 结果目录匹配模式",
    )
    parser.add_argument(
        "--bt-pattern",
        type=str,
        default="bt_backtest_full_*",
        help="BT 结果目录匹配模式",
    )
    parser.add_argument(
        "--vec-bt-threshold-pp",
        type=float,
        default=0.05,
        help="VEC/BT 差异阈值（百分点）",
    )
    parser.add_argument(
        "--skip-vec-bt-check",
        action="store_true",
        help="跳过 VEC/BT 对齐检查",
    )
    parser.add_argument(
        "--skip-simplified-scan",
        action="store_true",
        help="跳过简化版/备份命名扫描",
    )
    args = parser.parse_args()

    success = ci_checks(
        output_dir=args.output_dir,
        min_annual_return=args.min_annual_return,
        max_drawdown=args.max_drawdown,
        min_sharpe=args.min_sharpe,
        min_winrate=args.min_winrate,
        results_root=args.results_root,
        vec_results_pattern=args.vec_pattern,
        bt_results_pattern=args.bt_pattern,
        vec_bt_threshold_pp=args.vec_bt_threshold_pp,
        skip_vec_bt_check=args.skip_vec_bt_check,
        skip_simplified_scan=args.skip_simplified_scan,
    )
    sys.exit(0 if success else 1)
