"""
Combo WFO 入口脚本 - 全空间搜索
================================================================================
设计理念：策略开发初期应该追求**广度**而非精度

工作流程：
1. WFO阶段：全空间搜索（18因子 × [2,3,4,5,6,7]阶 ≈ 62,985组合）
   - 利用WFO快速验证能力（~2分钟完成）
   - 输出Top-N候选组合（默认Top-100）

2. VEC阶段：精细化验证（scripts/run_full_space_vec_backtest.py）
   - 对WFO输出的Top-N进行完整回测
   - 计算详细指标（收益、Sharpe、MaxDD等）

3. 筛选阶段：多维度过滤（scripts/select_strategy_v2.py）
   - IC门槛过滤
   - 综合得分排序
   - 复杂度约束
   - Holdout验证

⚠️ 不要在WFO阶段就做过多限制，让数据说话！
预期组合数：C(18,2) + C(18,3) + ... + C(18,7) ≈ 62,985
"""

import sys
import os
from pathlib import Path
import yaml
import logging
from datetime import datetime
import numpy as np
import pandas as pd

# ROOT应该指向项目根目录
ROOT = Path(__file__).parent.parent.parent

# 添加 src/ 到路径（确保 etf_strategy 包可导入）
sys.path.insert(0, str(ROOT / "src"))

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.combo_wfo_optimizer import ComboWFOOptimizer, ComboWFOConfig
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.regime_gate import compute_regime_gate_arr, gate_stats

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# 配置说明
# ============================================================================
# ⚠️ 不在此处硬编码因子和组合阶数，全部从配置文件读取
# 原因：策略开发初期应该追求广度，利用WFO/VEC快速验证能力
# 过滤和筛选应该在后续阶段进行（VEC验证 + 综合筛选）


def main():
    """主流程"""
    print("=" * 80)
    print("� Combo WFO - 全空间搜索（广度优先）")
    print("=" * 80)

    # 1. 加载配置
    config_path = Path(
        os.environ.get("WFO_CONFIG_PATH", str(ROOT / "configs/combo_wfo_config.yaml"))
    )
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"\n✅ 配置加载完成")
    print(f"  数据路径: {config['data']['data_dir']}")
    print(
        f"  训练期: {config['data']['start_date']} ~ {config['data'].get('training_end_date', config['data']['end_date'])}"
    )

    # 2. 加载数据
    data_loader = DataLoader(
        data_dir=config["data"]["data_dir"],
        cache_dir=config["data"]["cache_dir"],
    )

    training_end = config["data"].get("training_end_date", config["data"]["end_date"])

    ohlcv_data = data_loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=training_end,
    )

    print(f"\n✅ 数据加载完成")
    print(f"  日期数: {len(ohlcv_data['close'])}")
    print(f"  ETF数: {len(config['data']['symbols'])}")

    # 3. 计算因子（使用全部18个因子）
    print(f"\n🔧 计算因子（全空间）...")
    factor_lib = PreciseFactorLibrary()
    factors_raw = factor_lib.compute_all_factors(ohlcv_data)

    # 4. 提取因子名并转换为Dict格式
    all_factors = factors_raw.columns.get_level_values(0).unique().tolist()
    print(f"✅ 因子计算完成: {len(all_factors)} 个")

    # 转换为Dict格式以适配CrossSectionProcessor
    print(f"\n🔄 准备因子数据...")
    factors_dict = {}
    for factor_name in all_factors:
        factors_dict[factor_name] = factors_raw[factor_name]

    # 5. 横截面处理
    print(f"\n📐 横截面标准化...")
    processor = CrossSectionProcessor(
        lower_percentile=config["cross_section"]["winsorize_lower"],
        upper_percentile=config["cross_section"]["winsorize_upper"],
    )
    processed_factors = processor.process_all_factors(factors_dict)

    # 6. 择时信号
    print(f"\n⏰ 生成择时信号...")
    timing_config = config["backtest"]["timing"]
    timing_module = LightTimingModule(
        extreme_threshold=timing_config["extreme_threshold"],
        extreme_position=timing_config["extreme_position"],
    )

    timing_signals = timing_module.compute_position_ratios(ohlcv_data["close"])

    # 7. WFO配置（使用配置文件设置）
    print(f"\n⚙️ WFO配置:")
    wfo_cfg = config["combo_wfo"]
    combo_sizes = wfo_cfg.get("combo_sizes", [2, 3, 4, 5, 6, 7])
    print(f"  组合阶数: {combo_sizes}")
    print(f"  IS窗口: {wfo_cfg['is_period']} 天")
    print(f"  OOS窗口: {wfo_cfg['oos_period']} 天")
    print(f"  滚动步长: {wfo_cfg.get('step_size', 60)} 天")

    expected_combos = sum(
        [
            len(list(pd.Series(range(len(all_factors))).apply(lambda x: x).index))
            for size in combo_sizes
        ]
    )
    print(
        f"  预期组合数: ~{len(all_factors)}C{min(combo_sizes)}...{len(all_factors)}C{max(combo_sizes)}"
    )

    wfo_config = ComboWFOConfig(
        combo_sizes=combo_sizes,
        is_period=wfo_cfg["is_period"],
        oos_period=wfo_cfg["oos_period"],
        step_size=wfo_cfg.get("step_size", 60),
        n_jobs=wfo_cfg.get("n_jobs", -1),
        verbose=wfo_cfg.get("verbose", 1),
        enable_fdr=wfo_cfg.get("enable_fdr", True),
        fdr_alpha=wfo_cfg.get("fdr_alpha", 0.05),
        complexity_penalty_lambda=wfo_cfg["scoring"].get(
            "complexity_penalty_lambda", 0.01
        ),
    )

    # 7.5 正交因子集过滤
    active_factors_cfg = config.get("active_factors")
    if active_factors_cfg:
        active_set = set(active_factors_cfg)
        all_factor_set = set(processed_factors.keys())
        missing = active_set - all_factor_set
        if missing:
            raise ValueError(f"active_factors 中指定了不存在的因子: {sorted(missing)}")
        excluded = sorted(all_factor_set - active_set)
        processed_factors = {
            k: v for k, v in processed_factors.items() if k in active_set
        }
        all_factors = sorted(processed_factors.keys())
        print(f"✅ 正交因子集: {len(all_factors)}/{len(all_factor_set)} 个因子已激活")
        print(f"  已排除: {excluded}")

    # 8. 转换为 (T, N, F) 数组
    print(f"\n🔄 转换因子为3D数组...")
    factor_list = list(processed_factors.values())
    factors_array = np.stack([df.values for df in factor_list], axis=2)  # (T, N, F)
    factor_names = list(processed_factors.keys())
    print(f"  Shape: {factors_array.shape}")

    # 9. 计算收益率
    returns_df = ohlcv_data["close"].pct_change()
    returns = returns_df.values

    # Regime gate（作为交易规则的一部分进入 WFO：用于 OOS 收益模拟）
    backtest_cfg = config.get("backtest", {})
    gate_arr = compute_regime_gate_arr(
        ohlcv_data["close"],
        returns_df.index,
        backtest_config=backtest_cfg,
    )
    if bool((backtest_cfg.get("regime_gate") or {}).get("enabled", False)):
        stats = gate_stats(gate_arr)
        print(
            f"🧯 Regime gate enabled (WFO): mean={stats['mean']:.3f} min={stats['min']:.3f} max={stats['max']:.3f}"
        )

    # 10. 运行WFO
    print(f"\n🚀 开始WFO优化（全空间搜索）...")

    optimizer = ComboWFOOptimizer(
        combo_sizes=combo_sizes,
        is_period=wfo_cfg["is_period"],
        oos_period=wfo_cfg["oos_period"],
        step_size=wfo_cfg.get("step_size", 60),
        n_jobs=wfo_cfg.get("n_jobs", -1),
        verbose=wfo_cfg.get("verbose", 1),
        enable_fdr=wfo_cfg.get("enable_fdr", True),
        fdr_alpha=wfo_cfg.get("fdr_alpha", 0.05),
        complexity_penalty_lambda=wfo_cfg["scoring"].get(
            "complexity_penalty_lambda", 0.01
        ),
    )

    top_combos, results_df = optimizer.run_combo_search(
        factors_data=factors_array,
        returns=returns,
        factor_names=factor_names,
        top_n=wfo_cfg.get("top_n", 100),
        pos_size=config["backtest"]["pos_size"],
        commission_rate=config["backtest"]["commission_rate"],
        exposures=gate_arr,
    )

    # 11. 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT / f"results/run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存完整结果
    full_output_file = output_dir / "full_combo_results.csv"
    results_df.to_csv(full_output_file, index=False)

    # 保存 Top 组合
    top_output_file = output_dir / "top_combos.csv"
    top_df = pd.DataFrame(top_combos)
    top_df.to_csv(top_output_file, index=False)

    print(f"\n💾 结果保存至:")
    print(f"  完整结果: {full_output_file}")
    print(f"  Top组合: {top_output_file}")
    print(f"  组合总数: {len(results_df)}, Top-N: {len(top_combos)}")

    # 12. 输出Top20
    print(f"\n🏆 Top20 组合 (按IC排序)")
    print("-" * 80)

    # 使用WFO返回的列名
    results_sorted = results_df.sort_values("mean_oos_ic", ascending=False)

    for idx, row in results_sorted.head(20).iterrows():
        combo_display = (
            row["combo"][:65] + "..." if len(row["combo"]) > 68 else row["combo"]
        )
        print(f"{idx+1:3d}. {combo_display:68s}")
        print(
            f"     IC={row['mean_oos_ic']:+.4f} | IR={row.get('oos_ic_ir', 0):.2f} | "
            f"正率={row.get('positive_rate', 0):.1%} | 阶数={row.get('combo_size', 0)}"
        )

    print(f"\n✅ WFO完成（全空间搜索）")
    print("=" * 80)

    print(f"\n📋 下一步工作流程:")
    print(f"  1. VEC精算: uv run python scripts/run_full_space_vec_backtest.py")
    print(f"     - 对Top-{len(top_combos)}组合进行完整回测")
    print(f"     - 计算收益/Sharpe/MaxDD等详细指标")
    print(f"  ")
    print(f"  2. 策略筛选: uv run python scripts/select_strategy_v2.py")
    print(f"     - IC门槛过滤")
    print(f"     - 综合得分排序")
    print(f"     - 可选：复杂度约束、因子黑名单等")
    print(f"  ")
    print(f"  3. Holdout验证: 验证最终筛选结果的样本外表现")


if __name__ == "__main__":
    main()
