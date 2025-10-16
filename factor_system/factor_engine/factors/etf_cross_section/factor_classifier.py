#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF横截面因子分类系统
对筛选后的因子进行类别标注：动量、均值回归、成交量、波动率等
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import re
from collections import defaultdict

from .factor_screener import FactorScreeningResult
from .candidate_factor_generator import FactorVariant
from factor_system.utils import safe_operation, FactorSystemError

logger = logging.getLogger(__name__)


@dataclass
class FactorCategory:
    """因子类别定义"""
    name: str
    description: str
    keywords: List[str]
    subcategories: Dict[str, List[str]]
    characteristics: Dict[str, Any]


@dataclass
class ClassifiedFactor:
    """分类后的因子"""
    variant_id: str
    base_factor_id: str
    category: str
    subcategory: str
    style_tags: List[str]
    economic_meaning: str
    usage_scenarios: List[str]
    risk_profile: str
    expected_frequency: str


class FactorClassifier:
    """因子分类器"""

    def __init__(self):
        """初始化因子分类器"""
        self.categories = self._define_categories()
        self.classification_rules = self._define_classification_rules()
        logger.info("因子分类器初始化完成")

    def _define_categories(self) -> Dict[str, FactorCategory]:
        """定义因子类别"""
        categories = {
            "momentum": FactorCategory(
                name="动量类",
                description="基于价格动量和趋势的因子",
                keywords=["rsi", "macd", "sto", "mom", "roc", "adx", "aroon", "willr", "cci", "apo", "ppo"],
                subcategories={
                    "趋势跟踪": ["rsi", "macd", "adx", "aroon"],
                    "振荡指标": ["sto", "willr", "cci"],
                    "动量变化": ["mom", "roc", "apo", "ppo"],
                    "复合动量": ["ultosc", "stochrsi"]
                },
                characteristics={
                    "signal_duration": "medium",
                    "market_sensitivity": "high",
                    "reversal_frequency": "low"
                }
            ),

            "mean_reversion": FactorCategory(
                name="均值回归类",
                description="基于价格偏离均值回归特性的因子",
                keywords=["bb_", "bollinger", "keltner", "donchian", "stdev", "stdev2"],
                subcategories={
                    "布林带类": ["bb_", "bollinger"],
                    "通道突破": ["keltner", "donchian"],
                    "统计套利": ["stdev", "stdev2"]
                },
                characteristics={
                    "signal_duration": "short",
                    "market_sensitivity": "medium",
                    "reversal_frequency": "high"
                }
            ),

            "volume": FactorCategory(
                name="成交量类",
                description="基于成交量分析的因子",
                keywords=["vol", "obv", "vwap", "ad", "volume", "vpt", "pvi", "nvi", "eom", "cmf"],
                subcategories={
                    "成交量指标": ["obv", "ad", "vpt"],
                    "量价分析": ["vwap", "eom", "cmf"],
                    "成交量动量": ["volume_ratio", "volume_momentum"],
                    "能量指标": ["pvi", "nvi", "mfi"]
                },
                characteristics={
                    "signal_duration": "medium",
                    "market_sensitivity": "medium",
                    "reversal_frequency": "medium"
                }
            ),

            "volatility": FactorCategory(
                name="波动率类",
                description="基于价格波动率的因子",
                keywords=["atr", "natr", "stddev", "trange", "vhf", "var", "kurtosis", "skew"],
                subcategories={
                    "已实现波动率": ["atr", "natr", "trange"],
                    "统计波动率": ["stddev", "var"],
                    "波动率特征": ["vhf", "kurtosis", "skew"]
                },
                characteristics={
                    "signal_duration": "long",
                    "market_sensitivity": "low",
                    "reversal_frequency": "low"
                }
            ),

            "trend": FactorCategory(
                name="趋势类",
                description="基于趋势识别的因子",
                keywords=["ma", "ema", "sma", "wma", "kama", "trima", "t3", "sma", "dema", "tema"],
                subcategories={
                    "简单移动平均": ["sma", "ma"],
                    "指数移动平均": ["ema", "dema", "tema"],
                    "特殊移动平均": ["wma", "kama", "trima", "t3"],
                    "趋势强度": ["ht_trendline", "adx"]
                },
                characteristics={
                    "signal_duration": "long",
                    "market_sensitivity": "low",
                    "reversal_frequency": "low"
                }
            ),

            "overlap": FactorCategory(
                name="重叠类",
                description="需要价格重叠分析的因子",
                keywords=["sma", "ema", "overlap", "sar", "ht_"],
                subcategories={
                    "移动平均交叉": ["ma", "ema", "sma"],
                    "抛物线指标": ["sar"],
                    "希尔伯特变换": ["ht_"]
                },
                characteristics={
                    "signal_duration": "medium",
                    "market_sensitivity": "medium",
                    "reversal_frequency": "medium"
                }
            ),

            "candlestick": FactorCategory(
                name="K线形态类",
                description="基于K线形态识别的因子",
                keywords=["cdl", "candle", "doji", "hammer", "engulfing"],
                subcategories={
                    "反转形态": ["cdldoji", "cdlhammer", "cdlengulfing"],
                    "持续形态": ["cdlmarubozu", "cdlthree"],
                    "形态强度": ["cdl3outside", "cdl2crows"]
                },
                characteristics={
                    "signal_duration": "very_short",
                    "market_sensitivity": "high",
                    "reversal_frequency": "high"
                }
            )
        }

        return categories

    def _define_classification_rules(self) -> Dict[str, callable]:
        """定义分类规则"""
        rules = {
            "keyword_based": self._classify_by_keywords,
            "pattern_based": self._classify_by_pattern,
            "parameter_based": self._classify_by_parameters,
            "behavior_based": self._classify_by_behavior
        }
        return rules

    def _classify_by_keywords(self, variant_id: str, base_factor_id: str) -> Optional[str]:
        """基于关键词分类"""
        factor_id_lower = variant_id.lower()

        # 检查每个类别的关键词
        for category_name, category in self.categories.items():
            for keyword in category.keywords:
                if keyword in factor_id_lower:
                    return category_name

        return None

    def _classify_by_pattern(self, variant_id: str, base_factor_id: str) -> Optional[str]:
        """基于模式分类"""
        factor_id_lower = variant_id.lower()

        # 特殊模式匹配
        patterns = {
            r".*ma\d+.*": "trend",  # 移动平均
            r".*ema\d+.*": "trend",  # 指数移动平均
            r".*bb_\d+.*": "mean_reversion",  # 布林带
            r".*atr\d*.*": "volatility",  # ATR
            r".*obv.*": "volume",  # OBV
            r".*vwap.*": "volume",  # VWAP
            r".*cdl.*": "candlestick",  # K线形态
        }

        for pattern, category in patterns.items():
            if re.match(pattern, factor_id_lower):
                return category

        return None

    def _classify_by_parameters(self, variant_id: str, base_factor_id: str) -> Optional[str]:
        """基于参数分类"""
        factor_id_lower = variant_id.lower()

        # 基于参数长度和类型的分类
        if "timeperiod" in variant_id.lower():
            timeperiod_match = re.search(r"timeperiod(\d+)", variant_id.lower())
            if timeperiod_match:
                period = int(timeperiod_match.group(1))
                if period <= 14:
                    return "momentum"  # 短期动量
                elif period <= 50:
                    return "trend"  # 中期趋势
                else:
                    return "trend"  # 长期趋势

        return None

    def _classify_by_behavior(self, variant_id: str, base_factor_id: str) -> Optional[str]:
        """基于行为特征分类"""
        factor_id_lower = variant_id.lower()

        # 基于因子行为特征的分类
        behavior_patterns = {
            "oscillator": "momentum",      # 振荡器通常是动量类
            "momentum": "momentum",        # 明确的动量指标
            "trend": "trend",              # 趋势指标
            "volume": "volume",            # 成交量指标
            "volatility": "volatility",    # 波动率指标
            "reversal": "mean_reversion",  # 反转指标
        }

        for behavior, category in behavior_patterns.items():
            if behavior in factor_id_lower:
                return category

        return None

    def _determine_subcategory(self, category: str, variant_id: str, base_factor_id: str) -> str:
        """确定子类别"""
        if category not in self.categories:
            return "未分类"

        category_info = self.categories[category]
        factor_id_lower = variant_id.lower()

        # 检查子类别关键词
        for subcategory, keywords in category_info.subcategories.items():
            for keyword in keywords:
                if keyword in factor_id_lower:
                    return subcategory

        return "通用"

    def _generate_style_tags(self, category: str, variant_id: str, base_factor_id: str) -> List[str]:
        """生成风格标签"""
        tags = []

        # 基于类别的标签
        category_tags = {
            "momentum": ["趋势跟踪", "动量", "中短期"],
            "mean_reversion": ["均值回归", "反转", "短期"],
            "volume": ["成交量", "资金流", "量价配合"],
            "volatility": ["波动率", "风险", "防御性"],
            "trend": ["趋势", "长期", "稳定性"],
            "overlap": ["技术指标", "综合分析"],
            "candlestick": ["形态学", "短期反转", "技术分析"]
        }

        tags.extend(category_tags.get(category, []))

        # 基于参数的标签
        if "long" in variant_id.lower() or any(x in variant_id for x in ["120", "252", "200"]):
            tags.append("长期")
        elif "short" in variant_id.lower() or any(x in variant_id for x in ["5", "7", "10"]):
            tags.append("短期")
        else:
            tags.append("中期")

        # 基于指标复杂度的标签
        if "smooth" in variant_id.lower() or "ema" in variant_id.lower():
            tags.append("平滑化")
        if "rate" in variant_id.lower() or "change" in variant_id.lower():
            tags.append("变化率")

        return list(set(tags))  # 去重

    def _generate_usage_scenarios(self, category: str, subcategory: str) -> List[str]:
        """生成适用场景"""
        scenarios = []

        # 基于类别的场景
        category_scenarios = {
            "momentum": ["趋势市场", "突破策略", "动量轮动"],
            "mean_reversion": ["震荡市场", "均值回归策略", "反转交易"],
            "volume": ["量价分析", "资金流向", "市场情绪"],
            "volatility": ["风险管理", "波动率交易", "资产配置"],
            "trend": ["长期投资", "指数跟踪", "资产配置"],
            "overlap": ["综合分析", "多指标确认", "信号过滤"],
            "candlestick": ["短期交易", "形态识别", "反转信号"]
        }

        scenarios.extend(category_scenarios.get(category, ["通用场景"]))

        # 基于子类别的场景
        subcategory_scenarios = {
            "趋势跟踪": ["强势股跟踪", "动量策略"],
            "布林带类": ["通道交易", "波动率策略"],
            "成交量指标": ["量价配合", "资金流向分析"],
            "统计波动率": ["风险管理", "期权策略"],
            "简单移动平均": ["趋势确认", "支撑阻力"],
            "反转形态": ["短期反转", "形态交易"]
        }

        scenarios.extend(subcategory_scenarios.get(subcategory, []))

        return list(set(scenarios))  # 去重

    def classify_factor(self, variant_id: str, base_factor_id: str,
                       screening_result: Optional[FactorScreeningResult] = None) -> ClassifiedFactor:
        """
        分类单个因子

        Args:
            variant_id: 因子变体ID
            base_factor_id: 基础因子ID
            screening_result: 筛选结果（可选）

        Returns:
            分类结果
        """
        # 应用分类规则
        category = None

        for rule_name, rule_func in self.classification_rules.items():
            category = rule_func(variant_id, base_factor_id)
            if category:
                logger.debug(f"因子 {variant_id} 通过 {rule_name} 规则分类为 {category}")
                break

        if not category:
            category = "overlap"  # 默认分类
            logger.warning(f"因子 {variant_id} 无法分类，使用默认分类: {category}")

        # 确定子类别
        subcategory = self._determine_subcategory(category, variant_id, base_factor_id)

        # 生成风格标签
        style_tags = self._generate_style_tags(category, variant_id, base_factor_id)

        # 生成经济含义
        economic_meaning = self._generate_economic_meaning(category, subcategory)

        # 生成适用场景
        usage_scenarios = self._generate_usage_scenarios(category, subcategory)

        # 确定风险特征
        risk_profile = self._determine_risk_profile(category, subcategory)

        # 确定预期频率
        expected_frequency = self._determine_expected_frequency(category, subcategory)

        return ClassifiedFactor(
            variant_id=variant_id,
            base_factor_id=base_factor_id,
            category=category,
            subcategory=subcategory,
            style_tags=style_tags,
            economic_meaning=economic_meaning,
            usage_scenarios=usage_scenarios,
            risk_profile=risk_profile,
            expected_frequency=expected_frequency
        )

    def _generate_economic_meaning(self, category: str, subcategory: str) -> str:
        """生成经济含义"""
        meanings = {
            "momentum": "捕捉价格趋势的持续性，基于投资者行为偏差和市场情绪传导",
            "mean_reversion": "识别价格偏离均衡价值的机会，基于统计套利原理",
            "volume": "反映资金流向和市场参与度，体现买卖力量对比",
            "volatility": "衡量价格波动程度和不确定性，用于风险评估和资产配置",
            "trend": "识别长期价格方向，用于趋势跟踪和资产配置",
            "overlap": "综合多重技术信号，提供更可靠的市场研判",
            "candlestick": "识别短期价格反转模式，基于市场心理变化"
        }

        base_meaning = meanings.get(category, "技术分析指标")
        subcategory_modifier = f"，特别适用于{subcategory}场景"

        return base_meaning + subcategory_modifier

    def _determine_risk_profile(self, category: str, subcategory: str) -> str:
        """确定风险特征"""
        risk_profiles = {
            "momentum": "中高风险，趋势市场表现优异，震荡市可能出现较大回撤",
            "mean_reversion": "中等风险，震荡市表现稳定，强趋势市可能失效",
            "volume": "低风险，作为辅助指标能提供额外信息，单独使用效果有限",
            "volatility": "低风险，主要用于风险管理和资产配置",
            "trend": "低风险，信号稳定但响应较慢，适合长期投资",
            "overlap": "中等风险，综合多个指标能提高可靠性",
            "candlestick": "高风险，短期信号敏感但容易产生假信号"
        }

        return risk_profiles.get(category, "中等风险")

    def _determine_expected_frequency(self, category: str, subcategory: str) -> str:
        """确定预期频率"""
        frequencies = {
            "momentum": "周度至月度调仓",
            "mean_reversion": "日度至周度调仓",
            "volume": "日度调仓",
            "volatility": "月度调仓",
            "trend": "月度至季度调仓",
            "overlap": "周度调仓",
            "candlestick": "日度调仓"
        }

        return frequencies.get(category, "周度调仓")

    def batch_classify_factors(self, screening_results: Dict[str, FactorScreeningResult]) -> Dict[str, ClassifiedFactor]:
        """
        批量分类因子

        Args:
            screening_results: 筛选结果字典

        Returns:
            分类结果字典
        """
        logger.info(f"开始批量分类 {len(screening_results)} 个因子")

        classified_factors = {}

        for variant_id, screening_result in screening_results.items():
            if screening_result.screening_reason != "通过筛选":
                continue  # 只分类通过的因子

            # 解析基础因子ID
            base_factor_id = variant_id.split('_')[0] if '_' in variant_id else variant_id

            # 分类因子
            classified_factor = self.classify_factor(
                variant_id, base_factor_id, screening_result
            )

            classified_factors[variant_id] = classified_factor

        logger.info(f"批量分类完成: {len(classified_factors)} 个因子已分类")

        return classified_factors

    def save_classification_results(self, results: Dict[str, ClassifiedFactor],
                                   output_path: str):
        """
        保存分类结果

        Args:
            results: 分类结果
            output_path: 输出路径
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # 转换为DataFrame
        data = []
        for variant_id, result in results.items():
            data.append({
                "variant_id": result.variant_id,
                "base_factor_id": result.base_factor_id,
                "category": result.category,
                "subcategory": result.subcategory,
                "style_tags": ";".join(result.style_tags),
                "economic_meaning": result.economic_meaning,
                "usage_scenarios": ";".join(result.usage_scenarios),
                "risk_profile": result.risk_profile,
                "expected_frequency": result.expected_frequency
            })

        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)

        logger.info(f"分类结果已保存到: {output_file}")

        # 保存分类统计
        category_stats = df.groupby('category').size().to_dict()
        stats_file = output_file.parent / f"{output_file.stem}_stats.txt"
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("因子分类统计\n")
            f.write("=" * 50 + "\n")
            for category, count in category_stats.items():
                f.write(f"{category}: {count} 个因子\n")
            f.write(f"\n总计: {len(results)} 个因子\n")

        logger.info(f"分类统计已保存到: {stats_file}")


@safe_operation
def classify_etf_factors(screening_results: Dict[str, FactorScreeningResult],
                        output_dir: str = None) -> Dict[str, ClassifiedFactor]:
    """
    便捷函数：分类ETF因子

    Args:
        screening_results: 筛选结果字典
        output_dir: 输出目录

    Returns:
        分类结果字典
    """
    classifier = FactorClassifier()

    results = classifier.batch_classify_factors(screening_results)

    # 保存结果
    if output_dir is None:
        output_dir = "factor_system/factor_output/etf_cross_section/classification_results"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/factor_classification_{timestamp}.csv"
    classifier.save_classification_results(results, output_file)

    return results


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logger.info("因子分类系统测试完成")