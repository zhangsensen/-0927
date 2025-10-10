#!/usr/bin/env python3
"""
公平评分器 - 解决时间框架因子评分不公平问题
作者：量化首席工程师
版本：3.1.0
日期：2025-10-07
状态：生产就绪

核心功能：
1. 时间框架权重调整 - 解决样本量偏差
2. 经济意义奖励 - 补偿长周期因子价值
3. 稳定性奖励 - 鼓励稳健因子
4. 不确定性惩罚 - 防止过拟合
"""

import logging
import math
from typing import Dict, Any, Optional
from dataclasses import dataclass
import yaml
from pathlib import Path


@dataclass
class TimeframeAdjustment:
    """时间框架调整参数"""
    stability_bonus: float = 0.0          # 稳定性奖励
    uncertainty_penalty: float = 1.0      # 不确定性惩罚因子
    economic_significance_discount: float = 1.0  # 经济意义折扣
    sample_size_normalization: float = 1.0  # 样本量归一化权重
    predictive_power_boost: float = 0.0    # 预测能力奖励


class FairScorer:
    """公平评分器 - 平衡不同时间框架的因子评分"""

    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.timeframe_adjustments: Dict[str, TimeframeAdjustment] = {}
        self.enabled = False

        # 默认配置（如果配置文件不存在）
        self.default_adjustments = {
            "1min": TimeframeAdjustment(
                stability_bonus=0.0,
                uncertainty_penalty=1.0,
                economic_significance_discount=0.95,  # 轻微惩罚：缺乏经济意义
                sample_size_normalization=0.3,         # 大样本折扣
                predictive_power_boost=0.0
            ),
            "5min": TimeframeAdjustment(
                stability_bonus=0.0,
                uncertainty_penalty=1.0,
                economic_significance_discount=0.9,
                sample_size_normalization=0.5,
                predictive_power_boost=0.0
            ),
            "15min": TimeframeAdjustment(
                stability_bonus=0.02,
                uncertainty_penalty=0.95,
                economic_significance_discount=0.95,
                sample_size_normalization=0.7,
                predictive_power_boost=0.0
            ),
            "30min": TimeframeAdjustment(
                stability_bonus=0.03,
                uncertainty_penalty=0.9,
                economic_significance_discount=1.0,
                sample_size_normalization=0.8,
                predictive_power_boost=0.01
            ),
            "60min": TimeframeAdjustment(
                stability_bonus=0.05,
                uncertainty_penalty=0.85,
                economic_significance_discount=1.0,
                sample_size_normalization=0.9,
                predictive_power_boost=0.02
            ),
            "2h": TimeframeAdjustment(
                stability_bonus=0.08,
                uncertainty_penalty=0.8,
                economic_significance_discount=1.1,  # 奖励：经济意义强
                sample_size_normalization=1.0,       # 小样本不惩罚
                predictive_power_boost=0.05
            ),
            "4h": TimeframeAdjustment(
                stability_bonus=0.10,
                uncertainty_penalty=0.75,
                economic_significance_discount=1.15,
                sample_size_normalization=1.0,
                predictive_power_boost=0.08
            ),
            "1day": TimeframeAdjustment(
                stability_bonus=0.25,      # 高稳定性奖励
                uncertainty_penalty=0.85,   # 减少惩罚
                economic_significance_discount=1.3,  # 高经济意义奖励
                sample_size_normalization=1.0,
                predictive_power_boost=0.15  # 高预测能力奖励
            ),
        }

        if config_path:
            self.load_config(config_path)
        else:
            self._use_default_config()

    def load_config(self, config_path: str):
        """从配置文件加载公平评分参数"""
        try:
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)

                fair_config = config.get('fair_scorer', {})
                self.enabled = fair_config.get('enabled', False)

                if self.enabled:
                    tf_adjustments = fair_config.get('timeframe_adjustments', {})
                    for tf, params in tf_adjustments.items():
                        self.timeframe_adjustments[tf] = TimeframeAdjustment(**params)

                self.logger.info(f"✅ 公平评分配置加载成功: {config_path}")
                self.logger.info(f"   启用状态: {self.enabled}")
                self.logger.info(f"   时间框架数量: {len(self.timeframe_adjustments)}")
            else:
                self.logger.warning(f"⚠️ 配置文件不存在: {config_path}，使用默认配置")
                self._use_default_config()

        except Exception as e:
            self.logger.error(f"❌ 加载公平评分配置失败: {e}")
            self._use_default_config()

    def _use_default_config(self):
        """使用默认配置"""
        self.enabled = True
        self.timeframe_adjustments = self.default_adjustments
        self.logger.info("✅ 使用默认公平评分配置")

    def apply_fair_scoring(self,
                          original_score: float,
                          timeframe: str,
                          sample_size: int,
                          ic_mean: float,
                          stability_score: float,
                          predictive_score: float) -> float:
        """
        应用公平评分调整

        Args:
            original_score: 原始综合评分
            timeframe: 时间框架
            sample_size: 样本量
            ic_mean: 平均IC值
            stability_score: 稳定性评分
            predictive_score: 预测能力评分

        Returns:
            调整后的公平评分
        """
        if not self.enabled:
            return original_score

        # 获取时间框架调整参数
        adjustment = self.timeframe_adjustments.get(timeframe)
        if not adjustment:
            # 如果没有找到配置，使用最接近的配置
            if timeframe in ["2min", "3min"]:
                adjustment = self.timeframe_adjustments.get("1min")
            elif timeframe == "30min":
                adjustment = self.timeframe_adjustments.get("15min")
            else:
                adjustment = TimeframeAdjustment()  # 默认无调整

        if not adjustment:
            return original_score

        # 1. 样本量归一化 (解决大样本偏差)
        sample_weight = adjustment.sample_size_normalization
        if sample_size > 0:
            # 参考样本量：1000
            reference_size = 1000
            size_ratio = min(sample_size / reference_size, 10.0)  # 限制最大比例

            # 大样本折扣：样本越大，权重越低
            if size_ratio > 1.0:
                sample_weight = adjustment.sample_size_normalization / math.sqrt(size_ratio)

        # 2. 维度权重调整
        adjusted_score = (
            original_score * 0.4 +  # 保留40%原始分数
            predictive_score * 0.3 * sample_weight +  # 预测能力权重：样本量敏感
            stability_score * 0.3  # 稳定性权重：样本量不敏感
        )

        # 3. 时间框架特定调整
        final_score = adjusted_score

        # 稳定性奖励
        if adjustment.stability_bonus > 0:
            final_score += stability_score * adjustment.stability_bonus

        # 预测能力奖励（针对长周期）
        if adjustment.predictive_power_boost > 0:
            final_score += predictive_score * adjustment.predictive_power_boost

        # 不确定性惩罚
        if adjustment.uncertainty_penalty != 1.0:
            final_score *= adjustment.uncertainty_penalty

        # 经济意义折扣/奖励
        if adjustment.economic_significance_discount != 1.0:
            # 长周期因子通常经济意义更强
            economic_factor = adjustment.economic_significance_discount
            final_score *= economic_factor

        # 确保分数在合理范围内
        final_score = max(0.0, min(1.0, final_score))

        return final_score

    def get_adjustment_summary(self, timeframe: str) -> Dict[str, Any]:
        """获取时间框架调整参数摘要"""
        adjustment = self.timeframe_adjustments.get(timeframe)
        if not adjustment:
            return {}

        return {
            "stability_bonus": adjustment.stability_bonus,
            "uncertainty_penalty": adjustment.uncertainty_penalty,
            "economic_significance_discount": adjustment.economic_significance_discount,
            "sample_size_normalization": adjustment.sample_size_normalization,
            "predictive_power_boost": adjustment.predictive_power_boost,
        }

    def compare_scores(self,
                      original_score: float,
                      adjusted_score: float,
                      timeframe: str) -> Dict[str, Any]:
        """比较原始评分和公平评分的差异"""
        return {
            "timeframe": timeframe,
            "original_score": original_score,
            "adjusted_score": adjusted_score,
            "score_change": adjusted_score - original_score,
            "percent_change": (adjusted_score - original_score) / original_score * 100 if original_score > 0 else 0,
            "adjustment_applied": self.enabled,
            "adjustment_params": self.get_adjustment_summary(timeframe)
        }


def create_fair_scorer_from_config(config_manager) -> FairScorer:
    """从配置管理器创建公平评分器"""
    # 尝试从配置管理器获取公平评分配置路径
    fair_config_path = getattr(config_manager, 'fair_scoring_config_path', None)
    if not fair_config_path:
        # 使用默认路径
        fair_config_path = "./configs/fair_scoring_config.yaml"

    return FairScorer(fair_config_path)


if __name__ == "__main__":
    # 测试公平评分器
    scorer = FairScorer("./configs/fair_scoring_config.yaml")

    # 测试不同时间框架的评分调整
    test_cases = [
        ("1min", 40000, 0.05, 0.7, 0.8),   # 大样本，短周期
        ("60min", 800, 0.04, 0.8, 0.6),    # 中等样本，中周期
        ("1day", 120, 0.06, 0.9, 0.7),     # 小样本，长周期
    ]

    print("公平评分测试结果:")
    print("=" * 80)
    for timeframe, sample_size, ic_mean, stability, predictive in test_cases:
        original_score = 0.75
        adjusted_score = scorer.apply_fair_scoring(
            original_score, timeframe, sample_size, ic_mean, stability, predictive
        )

        comparison = scorer.compare_scores(original_score, adjusted_score, timeframe)
        print(f"时间框架: {timeframe}")
        print(f"  原始评分: {original_score:.3f}")
        print(f"  调整评分: {adjusted_score:.3f}")
        print(f"  变化: {comparison['percent_change']:.1f}%")
        print(f"  调整参数: {comparison['adjustment_params']}")
        print()