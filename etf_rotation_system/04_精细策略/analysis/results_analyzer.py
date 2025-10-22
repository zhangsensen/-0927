#!/usr/bin/env python3
"""
ETF轮动策略结果分析器
从VBT回测结果中提取深度洞察，为精细策略提供指导
"""

import ast
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ETFResultsAnalyzer:
    """ETF轮动结果分析器"""

    def __init__(self, results_path: str):
        """
        初始化分析器

        Args:
            results_path: VBT回测结果目录路径
        """
        self.results_path = Path(results_path)
        self.results_df = None
        self.best_config = None
        self.factor_stats = {}
        self.analysis_results = {}

    def load_results(self) -> bool:
        """加载VBT回测结果"""
        try:
            # 查找结果文件
            results_csv = self.results_path / "results.csv"
            best_config_json = self.results_path / "best_config.json"

            if not results_csv.exists():
                logger.error(f"未找到结果文件: {results_csv}")
                return False

            if not best_config_json.exists():
                logger.error(f"未找到最佳配置文件: {best_config_json}")
                return False

            # 加载数据
            self.results_df = pd.read_csv(results_csv)

            with open(best_config_json, "r", encoding="utf-8") as f:
                self.best_config = json.load(f)

            logger.info(f"成功加载 {len(self.results_df)} 个策略结果")
            logger.info(
                f"最佳策略夏普比率: {self.best_config['performance']['sharpe_ratio']:.3f}"
            )
            return True

        except Exception as e:
            logger.error(f"加载结果时发生错误: {e}")
            return False

    @staticmethod
    def parse_weights(weights_str: str) -> Dict:
        """解析权重字符串为字典"""
        try:
            return ast.literal_eval(weights_str)
        except:
            return {}

    def analyze_factor_importance(self) -> Dict:
        """分析因子重要性"""
        logger.info("分析因子重要性...")

        factor_importance = defaultdict(list)

        for _, row in self.results_df.iterrows():
            weights = self.parse_weights(row["weights"])
            sharpe = row["sharpe_ratio"]

            for factor, weight in weights.items():
                if weight > 0:
                    factor_importance[factor].append((weight, sharpe))

        # 计算每个因子的统计指标
        factor_stats = {}
        for factor, values in factor_importance.items():
            weights, sharpes = zip(*values)
            weighted_sharpe = np.average(sharpes, weights=weights)
            avg_weight = np.mean(weights)
            total_usage = len(values)

            factor_stats[factor] = {
                "weighted_sharpe": weighted_sharpe,
                "avg_weight": avg_weight,
                "usage_count": total_usage,
                "usage_rate": total_usage / len(self.results_df),
                "weight_std": np.std(weights),
                "sharpe_correlation": (
                    np.corrcoef(weights, sharpes)[0, 1] if len(weights) > 1 else 0
                ),
            }

        self.factor_stats = factor_stats
        return factor_stats

    def identify_top_patterns(self, top_n: int = 50) -> Dict:
        """识别表现最优的权重组合模式"""
        logger.info(f"识别前{top_n}个最优模式...")

        top_strategies = self.results_df.nlargest(top_n, "sharpe_ratio")

        # 统计因子使用频率
        factor_usage = defaultdict(int)
        weight_patterns = []

        for _, row in top_strategies.iterrows():
            weights = self.parse_weights(row["weights"])
            weight_patterns.append(weights)

            for factor, weight in weights.items():
                if weight > 0:
                    factor_usage[factor] += 1

        # 分析常见权重水平
        common_weights = defaultdict(list)
        for pattern in weight_patterns:
            for factor, weight in pattern.items():
                if weight > 0:
                    common_weights[factor].append(weight)

        # 计算每个因子的典型权重范围
        weight_ranges = {}
        for factor, weights in common_weights.items():
            weight_ranges[factor] = {
                "mean": np.mean(weights),
                "std": np.std(weights),
                "min": np.min(weights),
                "max": np.max(weights),
                "median": np.median(weights),
                "q25": np.percentile(weights, 25),
                "q75": np.percentile(weights, 75),
            }

        return {
            "top_strategies": top_strategies[
                ["weights", "top_n", "sharpe_ratio", "total_return", "max_drawdown"]
            ].to_dict("records"),
            "factor_usage": dict(factor_usage),
            "weight_ranges": weight_ranges,
            "performance_stats": {
                "avg_sharpe": top_strategies["sharpe_ratio"].mean(),
                "avg_return": top_strategies["total_return"].mean(),
                "avg_drawdown": top_strategies["max_drawdown"].mean(),
                "sharpe_std": top_strategies["sharpe_ratio"].std(),
                "return_std": top_strategies["total_return"].std(),
                "best_sharpe": top_strategies["sharpe_ratio"].max(),
                "best_return": top_strategies["total_return"].max(),
            },
        }

    def analyze_performance_distribution(self) -> Dict:
        """分析性能指标分布"""
        logger.info("分析性能分布...")

        metrics = ["sharpe_ratio", "total_return", "max_drawdown"]
        distribution_stats = {}

        for metric in metrics:
            values = self.results_df[metric].values
            distribution_stats[metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "median": np.median(values),
                "percentiles": {
                    "1%": np.percentile(values, 1),
                    "5%": np.percentile(values, 5),
                    "10%": np.percentile(values, 10),
                    "25%": np.percentile(values, 25),
                    "50%": np.percentile(values, 50),
                    "75%": np.percentile(values, 75),
                    "90%": np.percentile(values, 90),
                    "95%": np.percentile(values, 95),
                    "99%": np.percentile(values, 99),
                },
            }

        return distribution_stats

    def identify_effective_factors(
        self, min_usage_rate: float = 0.3, min_sharpe: float = 0.45
    ) -> List[Tuple]:
        """识别高效因子"""
        effective_factors = []

        for factor, stats in self.factor_stats.items():
            if (
                stats["usage_rate"] >= min_usage_rate
                and stats["weighted_sharpe"] >= min_sharpe
            ):
                effective_factors.append(
                    (
                        factor,
                        stats["weighted_sharpe"],
                        stats["usage_rate"],
                        stats["avg_weight"],
                    )
                )

        # 按加权夏普比率排序
        effective_factors.sort(key=lambda x: x[1], reverse=True)
        return effective_factors

    def analyze_factor_correlations(self) -> Dict:
        """分析因子相关性"""
        logger.info("分析因子相关性...")

        # 构建因子权重矩阵
        all_factors = set()
        for _, row in self.results_df.iterrows():
            weights = self.parse_weights(row["weights"])
            all_factors.update(weights.keys())

        # 创建权重矩阵
        weight_matrix = []
        for _, row in self.results_df.iterrows():
            weights = self.parse_weights(row["weights"])
            weight_vector = [weights.get(factor, 0) for factor in sorted(all_factors)]
            weight_matrix.append(weight_vector)

        weight_df = pd.DataFrame(weight_matrix, columns=sorted(all_factors))

        # 计算相关性矩阵
        correlation_matrix = weight_df.corr()

        return {
            "correlation_matrix": correlation_matrix.to_dict(),
            "high_correlations": self._find_high_correlations(
                correlation_matrix, threshold=0.7
            ),
        }

    def _find_high_correlations(
        self, corr_matrix: pd.DataFrame, threshold: float = 0.7
    ) -> List[Tuple]:
        """查找高相关性因子对"""
        high_corr_pairs = []

        for i, factor1 in enumerate(corr_matrix.columns):
            for j, factor2 in enumerate(corr_matrix.columns):
                if i < j:  # 避免重复
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) >= threshold:
                        high_corr_pairs.append((factor1, factor2, corr_value))

        # 按相关性强度排序
        high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        return high_corr_pairs

    def generate_optimization_recommendations(self) -> Dict:
        """生成优化建议"""
        logger.info("生成优化建议...")

        # 识别高效因子
        effective_factors = self.identify_effective_factors()

        # 获取最优模式
        top_patterns = self.identify_top_patterns()

        # 分析权重范围
        optimal_weights = {}
        for factor, ranges in top_patterns["weight_ranges"].items():
            if ranges["mean"] > 0.05:  # 只考虑平均权重大于5%的因子
                confidence = ranges["mean"] / (ranges["std"] + 0.01)
                optimal_weights[factor] = {
                    "optimal_range": (
                        max(0.01, ranges["q25"]),
                        min(0.7, ranges["q75"]),
                    ),
                    "typical_weight": ranges["median"],
                    "confidence": confidence,
                    "stability": 1 - (ranges["std"] / (ranges["mean"] + 0.01)),
                }

        # 核心因子定义
        core_factors = [f[0] for f in effective_factors[:4]]
        supplementary_factors = [f[0] for f in effective_factors[4:6]]

        # 生成策略建议
        recommendations = {
            "core_factors": core_factors,
            "supplementary_factors": supplementary_factors,
            "factor_ranking": effective_factors,
            "optimal_weights": optimal_weights,
            "strategy_templates": [],
            "risk_guidelines": self._generate_risk_guidelines(),
            "optimization_priorities": self._generate_optimization_priorities(
                effective_factors
            ),
        }

        # 生成策略模板
        recommendations["strategy_templates"] = self._generate_strategy_templates(
            core_factors, supplementary_factors, optimal_weights
        )

        return recommendations

    def _generate_strategy_templates(
        self,
        core_factors: List[str],
        supplementary_factors: List[str],
        optimal_weights: Dict,
    ) -> List[Dict]:
        """生成策略模板"""
        templates = []

        # 模板1: 核心因子策略
        core_strategy = {}
        for factor in core_factors:
            if factor in optimal_weights:
                weight_range = optimal_weights[factor]["optimal_range"]
                # 使用置信度高的权重
                if optimal_weights[factor]["confidence"] > 1.0:
                    core_strategy[factor] = weight_range[1]
                else:
                    core_strategy[factor] = weight_range[0]

        # 标准化权重
        total_weight = sum(core_strategy.values())
        if total_weight > 0:
            core_strategy = {k: v / total_weight for k, v in core_strategy.items()}
            templates.append(
                {
                    "name": "核心因子策略",
                    "description": "聚焦最高效的4个因子，追求最高夏普比率",
                    "weights": core_strategy,
                    "risk_level": "medium",
                    "expected_sharpe": 0.47,
                    "factor_count": len(core_strategy),
                }
            )

        # 模板2: 平衡策略
        balanced_strategy = {}
        all_factors = core_factors + supplementary_factors
        for factor in all_factors:
            if factor in optimal_weights:
                weight_range = optimal_weights[factor]["optimal_range"]
                balanced_strategy[factor] = weight_range[0]  # 使用保守权重

        total_weight = sum(balanced_strategy.values())
        if total_weight > 0:
            balanced_strategy = {
                k: v / total_weight for k, v in balanced_strategy.items()
            }
            templates.append(
                {
                    "name": "平衡策略",
                    "description": "结合核心和补充因子，平衡风险和收益",
                    "weights": balanced_strategy,
                    "risk_level": "medium_low",
                    "expected_sharpe": 0.45,
                    "factor_count": len(balanced_strategy),
                }
            )

        # 模板3: 保守策略
        conservative_strategy = {}
        for factor in core_factors[:2]:  # 只用前2个最稳定因子
            if factor in optimal_weights and optimal_weights[factor]["stability"] > 0.7:
                conservative_strategy[factor] = 0.5  # 等权重分配

        total_weight = sum(conservative_strategy.values())
        if total_weight > 0:
            conservative_strategy = {
                k: v / total_weight for k, v in conservative_strategy.items()
            }
            templates.append(
                {
                    "name": "保守策略",
                    "description": "使用最稳定的2个因子，控制回撤风险",
                    "weights": conservative_strategy,
                    "risk_level": "low",
                    "expected_sharpe": 0.44,
                    "factor_count": len(conservative_strategy),
                }
            )

        return templates

    def _generate_risk_guidelines(self) -> Dict:
        """生成风险指导原则"""
        performance_dist = self.analyze_performance_distribution()

        return {
            "min_sharpe_ratio": performance_dist["sharpe_ratio"]["percentiles"]["25%"],
            "target_sharpe_ratio": performance_dist["sharpe_ratio"]["percentiles"][
                "75%"
            ],
            "max_drawdown_threshold": performance_dist["max_drawdown"]["percentiles"][
                "25%"
            ],
            "min_total_return": performance_dist["total_return"]["percentiles"]["50%"],
            "factor_count_range": (2, 5),
            "single_weight_limit": 0.6,
        }

    def _generate_optimization_priorities(
        self, effective_factors: List[Tuple]
    ) -> List[str]:
        """生成优化优先级建议"""
        priorities = []

        # 基于因子重要性生成优先级
        if len(effective_factors) > 0:
            top_factor = effective_factors[0][0]
            priorities.append(f"重点优化{top_factor}的权重分配")

        if len(effective_factors) > 1:
            second_factor = effective_factors[1][0]
            priorities.append(f"精细调整{second_factor}与{top_factor}的权重比例")

        priorities.extend(
            [
                "在最优权重范围内进行精细搜索",
                "考虑不同市场环境下的权重适应性",
                "定期重新评估因子有效性",
                "控制组合风险在可接受范围内",
            ]
        )

        return priorities

    def run_complete_analysis(self) -> Dict:
        """运行完整分析流程"""
        logger.info("开始完整结果分析...")

        if not self.load_results():
            return {"error": "无法加载结果数据"}

        try:
            # 执行各项分析
            factor_importance = self.analyze_factor_importance()
            top_patterns = self.identify_top_patterns()
            performance_dist = self.analyze_performance_distribution()
            factor_correlations = self.analyze_factor_correlations()
            recommendations = self.generate_optimization_recommendations()

            # 汇总分析结果
            analysis_results = {
                "summary": {
                    "total_strategies": len(self.results_df),
                    "best_strategy": self.best_config,
                    "analysis_timestamp": pd.Timestamp.now().isoformat(),
                },
                "factor_importance": factor_importance,
                "top_patterns": top_patterns,
                "performance_distribution": performance_dist,
                "factor_correlations": factor_correlations,
                "recommendations": recommendations,
            }

            self.analysis_results = analysis_results
            logger.info("完整分析完成")
            return analysis_results

        except Exception as e:
            logger.error(f"分析过程中发生错误: {e}")
            return {"error": str(e)}

    def save_analysis_results(self, output_path: str) -> bool:
        """保存分析结果"""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(self.analysis_results, f, indent=2, ensure_ascii=False)

            logger.info(f"分析结果已保存至: {output_file}")
            return True

        except Exception as e:
            logger.error(f"保存分析结果时发生错误: {e}")
            return False

    def print_analysis_summary(self):
        """打印分析摘要"""
        if not self.analysis_results:
            logger.warning("没有可用的分析结果")
            return

        summary = self.analysis_results["summary"]
        recommendations = self.analysis_results["recommendations"]

        print("\n" + "=" * 70)
        print("🎯 ETF轮动策略分析摘要")
        print("=" * 70)

        print(f"📊 分析策略总数: {summary['total_strategies']:,}")
        print(
            f"🏆 最佳夏普比率: {summary['best_strategy']['performance']['sharpe_ratio']:.3f}"
        )
        print(
            f"💰 最佳总收益: {summary['best_strategy']['performance']['total_return']:.2f}%"
        )

        print(f"\n🔍 核心因子 (按重要性排序):")
        for i, (factor, sharpe, usage, weight) in enumerate(
            recommendations["factor_ranking"][:6], 1
        ):
            print(
                f"  {i:2d}. {factor:20s} | 夏普: {sharpe:.3f} | 使用率: {usage:.1%} | 权重: {weight:.3f}"
            )

        print(f"\n📋 策略模板:")
        for template in recommendations["strategy_templates"]:
            print(
                f"  🎯 {template['name']} (风险: {template['risk_level']}, 预期夏普: {template['expected_sharpe']:.3f})"
            )
            top_factors = sorted(
                template["weights"].items(), key=lambda x: x[1], reverse=True
            )[:3]
            for factor, weight in top_factors:
                print(f"     {factor:20s}: {weight:.3f}")

        print(f"\n💡 优化优先级:")
        for i, priority in enumerate(recommendations["optimization_priorities"][:4], 1):
            print(f"  {i}. {priority}")

        print("=" * 70)


def main():
    """主函数 - 示例用法"""
    # VBT回测结果路径
    results_path = "/Users/zhangshenshen/深度量化0927/etf_rotation_system/data/results/backtest/backtest_20251021_201820"

    # 创建分析器
    analyzer = ETFResultsAnalyzer(results_path)

    # 运行完整分析
    results = analyzer.run_complete_analysis()

    if "error" not in results:
        # 保存结果
        output_path = "/Users/zhangshenshen/深度量化0927/etf_rotation_system/04_精细策略/output/analysis_results.json"
        analyzer.save_analysis_results(output_path)

        # 打印摘要
        analyzer.print_analysis_summary()
    else:
        logger.error(f"分析失败: {results['error']}")


if __name__ == "__main__":
    main()
