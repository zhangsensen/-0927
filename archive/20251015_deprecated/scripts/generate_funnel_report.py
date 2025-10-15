#!/usr/bin/env python3
"""生成漏斗报告 - 候选→过滤→最终

漏斗阶段：
1. 全量因子（209个）
2. 覆盖率筛选（≥80%）
3. 零方差筛选（=0）
4. 去重筛选（保留代表）
5. 生产因子（8-15个稳健价量因子）
"""

import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def generate_funnel_report():
    """生成漏斗报告"""

    logger.info("=" * 80)
    logger.info("因子筛选漏斗报告")
    logger.info("=" * 80)

    # 加载因子概要
    summary_file = Path(
        "factor_output/etf_rotation_production/factor_summary_20200102_20251014.csv"
    )
    summary = pd.read_csv(summary_file)

    logger.info(f"\n阶段1: 全量因子")
    logger.info(f"  因子数: {len(summary)}")

    # 阶段2: 覆盖率筛选
    coverage_threshold = 0.80
    stage2 = summary[summary["coverage"] >= coverage_threshold]
    logger.info(f"\n阶段2: 覆盖率筛选（≥{coverage_threshold:.0%}）")
    logger.info(f"  因子数: {len(stage2)}")
    logger.info(f"  淘汰: {len(summary) - len(stage2)}")

    # 阶段3: 零方差筛选
    stage3 = stage2[stage2["zero_variance"] == False]
    logger.info(f"\n阶段3: 零方差筛选（=0）")
    logger.info(f"  因子数: {len(stage3)}")
    logger.info(f"  淘汰: {len(stage2) - len(stage3)}")

    # 阶段4: 去重筛选（保留每组第一个）
    high_corr_file = Path(
        "factor_output/etf_rotation_production/high_correlation_pairs_3m.csv"
    )
    if high_corr_file.exists():
        high_corr = pd.read_csv(high_corr_file)

        # 构建去重集合
        to_remove = set()
        for _, row in high_corr.iterrows():
            if row["correlation"] > 0.99:  # 几乎完全相同
                # 保留factor1，移除factor2
                to_remove.add(row["factor2"])

        stage4 = stage3[~stage3["factor_id"].isin(to_remove)]
        logger.info(f"\n阶段4: 去重筛选（ρ>0.99）")
        logger.info(f"  因子数: {len(stage4)}")
        logger.info(f"  淘汰: {len(stage3) - len(stage4)}")
    else:
        stage4 = stage3
        logger.info(f"\n阶段4: 去重筛选（跳过，无相关性数据）")
        logger.info(f"  因子数: {len(stage4)}")

    # 阶段5: 生产因子筛选（8-15个稳健价量因子）
    logger.info(f"\n阶段5: 生产因子筛选（8-15个稳健价量因子）")

    # 定义生产因子规则：
    # - 动量：RSI, MACD
    # - 趋势：MA, EMA
    # - 波动：ATR, BBANDS
    # - 成交量：OBV, VOLUME_RATIO
    # - 收益：RETURN
    # - 价格位置：PRICE_POSITION

    production_factors = []

    # 动量类（2-3个）
    momentum_candidates = stage4[
        stage4["factor_id"].str.contains("RSI|MACD_SIGNAL", regex=True)
    ]
    if len(momentum_candidates) > 0:
        # 选择RSI_14
        rsi_14 = momentum_candidates[momentum_candidates["factor_id"] == "TA_RSI_14"]
        if len(rsi_14) > 0:
            production_factors.append(rsi_14.iloc[0]["factor_id"])

        # 选择MACD_SIGNAL_12_26_9
        macd = momentum_candidates[
            momentum_candidates["factor_id"] == "VBT_MACD_SIGNAL_12_26_9"
        ]
        if len(macd) > 0:
            production_factors.append(macd.iloc[0]["factor_id"])

    # 趋势类（2-3个）
    trend_candidates = stage4[
        stage4["factor_id"].str.contains("^VBT_MA_|^TA_SMA_", regex=True)
    ]
    for period in [20, 60]:
        ma = trend_candidates[trend_candidates["factor_id"] == f"VBT_MA_{period}"]
        if len(ma) > 0:
            production_factors.append(ma.iloc[0]["factor_id"])

    # 波动类（2个）
    volatility_candidates = stage4[
        stage4["factor_id"].str.contains("ATR|VOLATILITY", regex=True)
    ]
    atr_14 = volatility_candidates[volatility_candidates["factor_id"] == "TA_ATR_14"]
    if len(atr_14) > 0:
        production_factors.append(atr_14.iloc[0]["factor_id"])

    vol_20 = volatility_candidates[
        volatility_candidates["factor_id"] == "VOLATILITY_20"
    ]
    if len(vol_20) > 0:
        production_factors.append(vol_20.iloc[0]["factor_id"])

    # 成交量类（1-2个）
    volume_candidates = stage4[
        stage4["factor_id"].str.contains("OBV|VOLUME_RATIO", regex=True)
    ]
    vol_ratio_10 = volume_candidates[
        volume_candidates["factor_id"] == "VOLUME_RATIO_10"
    ]
    if len(vol_ratio_10) > 0:
        production_factors.append(vol_ratio_10.iloc[0]["factor_id"])

    # 收益类（2个）
    return_candidates = stage4[stage4["factor_id"].str.contains("^RETURN_", regex=True)]
    for period in [5, 20]:
        ret = return_candidates[return_candidates["factor_id"] == f"RETURN_{period}"]
        if len(ret) > 0:
            production_factors.append(ret.iloc[0]["factor_id"])

    # 价格位置类（1个）
    position_candidates = stage4[
        stage4["factor_id"].str.contains("PRICE_POSITION", regex=True)
    ]
    pos_20 = position_candidates[
        position_candidates["factor_id"] == "PRICE_POSITION_20"
    ]
    if len(pos_20) > 0:
        production_factors.append(pos_20.iloc[0]["factor_id"])

    # 动量类（补充）
    momentum_10 = stage4[stage4["factor_id"] == "MOMENTUM_10"]
    if len(momentum_10) > 0:
        production_factors.append(momentum_10.iloc[0]["factor_id"])

    logger.info(f"  因子数: {len(production_factors)}")
    logger.info(f"  淘汰: {len(stage4) - len(production_factors)}")

    logger.info(f"\n生产因子列表:")
    for i, factor_id in enumerate(production_factors, 1):
        factor_info = stage4[stage4["factor_id"] == factor_id].iloc[0]
        logger.info(f"  {i}. {factor_id}")
        logger.info(f"     覆盖率: {factor_info['coverage']:.2%}")
        logger.info(f"     零方差: {factor_info['zero_variance']}")

    # 保存生产因子列表
    output_dir = Path("factor_output/etf_rotation_production")
    production_file = output_dir / "production_factors.txt"
    with open(production_file, "w") as f:
        for factor_id in production_factors:
            f.write(f"{factor_id}\n")

    logger.info(f"\n✅ 生产因子列表已保存: {production_file}")

    # 生成漏斗统计
    funnel_stats = pd.DataFrame(
        {
            "阶段": ["全量因子", "覆盖率筛选", "零方差筛选", "去重筛选", "生产因子"],
            "因子数": [
                len(summary),
                len(stage2),
                len(stage3),
                len(stage4),
                len(production_factors),
            ],
            "淘汰数": [
                0,
                len(summary) - len(stage2),
                len(stage2) - len(stage3),
                len(stage3) - len(stage4),
                len(stage4) - len(production_factors),
            ],
        }
    )

    funnel_file = output_dir / "funnel_report.csv"
    funnel_stats.to_csv(funnel_file, index=False)
    logger.info(f"✅ 漏斗报告已保存: {funnel_file}")

    logger.info(f"\n{'=' * 80}")
    logger.info("✅ 漏斗报告生成完成")
    logger.info("=" * 80)

    return production_factors


if __name__ == "__main__":
    factors = generate_funnel_report()
    sys.exit(0 if len(factors) >= 8 else 1)
