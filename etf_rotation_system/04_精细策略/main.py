#!/usr/bin/env python3
"""
ETF轮动精细策略主入口
完整的策略分析、筛选和优化流程
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from analysis.results_analyzer import ETFResultsAnalyzer
from optimization.strategy_optimizer import OptimizationConfig, StrategyOptimizer
from screening.strategy_screener import ScreeningConfig, StrategyScreener
from utils.config_loader import get_config_loader

logger = logging.getLogger(__name__)


class FineStrategyPipeline:
    """精细策略流水线"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化流水线

        Args:
            config_path: 配置文件路径
        """
        self.config_loader = get_config_loader(config_path)
        self.config = self.config_loader.config

        # 验证配置
        if not self.config_loader.validate_config():
            raise ValueError("配置验证失败")

        # 初始化组件
        self.analyzer = None
        self.screener = None
        self.optimizer = None

        # 结果存储
        self.results = {
            "analysis": None,
            "screening": None,
            "optimization": None,
            "pipeline_summary": None,
        }

    def setup_components(self):
        """设置各个组件"""
        logger.info("设置流水线组件...")

        # 设置分析器
        self.analyzer = ETFResultsAnalyzer("")  # 初始化时不设置路径

        # 设置筛选器
        screening_config = ScreeningConfig(
            min_sharpe_ratio=self.config_loader.get(
                "screening_config.min_sharpe_ratio"
            ),
            max_drawdown_threshold=self.config_loader.get(
                "screening_config.max_drawdown_threshold"
            ),
            min_total_return=self.config_loader.get(
                "screening_config.min_total_return"
            ),
            max_single_weight=self.config_loader.get(
                "screening_config.max_single_weight"
            ),
            min_effective_factors=self.config_loader.get(
                "screening_config.min_effective_factors"
            ),
            max_effective_factors=self.config_loader.get(
                "screening_config.max_effective_factors"
            ),
            n_workers=self.config_loader.get("screening_config.n_workers"),
            chunk_size=self.config_loader.get("screening_config.chunk_size"),
            enable_cache=self.config_loader.get("screening_config.enable_cache"),
        )
        self.screener = StrategyScreener(screening_config)

        # 设置优化器
        optimization_config = OptimizationConfig(
            max_iterations=self.config_loader.get(
                "optimization_config.genetic_algorithm.max_iterations"
            ),
            population_size=self.config_loader.get(
                "optimization_config.genetic_algorithm.population_size"
            ),
            mutation_rate=self.config_loader.get(
                "optimization_config.genetic_algorithm.mutation_rate"
            ),
            crossover_rate=self.config_loader.get(
                "optimization_config.genetic_algorithm.crossover_rate"
            ),
            n_workers=self.config_loader.get("optimization_config.n_workers"),
            chunk_size=self.config_loader.get("optimization_config.chunk_size"),
            sharpe_weight=self.config_loader.get(
                "optimization_config.objective_weights.sharpe_ratio"
            ),
            return_weight=self.config_loader.get(
                "optimization_config.objective_weights.total_return"
            ),
            drawdown_weight=self.config_loader.get(
                "optimization_config.objective_weights.max_drawdown"
            ),
            enable_local_search=self.config_loader.get(
                "optimization_config.local_search.enable"
            ),
            local_search_radius=self.config_loader.get(
                "optimization_config.local_search.radius"
            ),
        )
        self.optimizer = StrategyOptimizer(optimization_config)

        logger.info("组件设置完成")

    def run_analysis(self, vbt_results_path: str) -> bool:
        """运行分析阶段"""
        logger.info("=" * 60)
        logger.info("开始分析阶段")
        logger.info("=" * 60)

        try:
            # 设置分析器路径
            self.analyzer.results_path = Path(vbt_results_path)

            # 运行分析
            analysis_results = self.analyzer.run_complete_analysis()

            if "error" in analysis_results:
                logger.error(f"分析失败: {analysis_results['error']}")
                return False

            self.results["analysis"] = analysis_results

            # 保存分析结果
            output_path = (
                Path(self.config_loader.get("data_paths.analysis_output"))
                / "analysis_results.json"
            )
            self.analyzer.save_analysis_results(output_path)

            # 打印分析摘要
            self.analyzer.print_analysis_summary()

            logger.info("分析阶段完成")
            return True

        except Exception as e:
            logger.error(f"分析阶段发生错误: {e}")
            return False

    def run_screening(self, n_candidates: int = None) -> bool:
        """运行筛选阶段"""
        logger.info("=" * 60)
        logger.info("开始筛选阶段")
        logger.info("=" * 60)

        try:
            if n_candidates is None:
                n_candidates = self.config_loader.get(
                    "screening_config.candidate_generation.total_candidates"
                )

            # 加载分析结果
            analysis_path = (
                Path(self.config_loader.get("data_paths.analysis_output"))
                / "analysis_results.json"
            )

            if not analysis_path.exists():
                logger.error(f"分析结果文件不存在: {analysis_path}")
                return False

            # 运行筛选
            screening_results = self.screener.run_complete_screening(
                str(analysis_path), n_candidates
            )

            if not screening_results.get("success"):
                logger.error(f"筛选失败: {screening_results.get('error')}")
                return False

            self.results["screening"] = screening_results

            # 保存筛选结果
            output_path = (
                Path(self.config_loader.get("data_paths.analysis_output"))
                / "screening_results.json"
            )
            self.screener.save_screening_results(output_path)

            # 打印筛选摘要
            self.screener.print_screening_summary()

            logger.info("筛选阶段完成")
            return True

        except Exception as e:
            logger.error(f"筛选阶段发生错误: {e}")
            return False

    def run_optimization(self, optimization_targets: list = None) -> bool:
        """运行优化阶段"""
        logger.info("=" * 60)
        logger.info("开始优化阶段")
        logger.info("=" * 60)

        try:
            if optimization_targets is None:
                optimization_targets = [3, 5, 8]  # 默认Top-N值

            # 加载筛选结果
            screening_path = (
                Path(self.config_loader.get("data_paths.analysis_output"))
                / "screening_results.json"
            )

            if not screening_path.exists():
                logger.error(f"筛选结果文件不存在: {screening_path}")
                return False

            # 加载筛选结果到优化器
            if not self.optimizer.load_screening_results(str(screening_path)):
                logger.error("加载筛选结果到优化器失败")
                return False

            # 运行优化
            optimization_results = self.optimizer.run_optimization(optimization_targets)

            if not optimization_results.get("success"):
                logger.error(f"优化失败: {optimization_results.get('error')}")
                return False

            self.results["optimization"] = optimization_results

            # 保存优化结果
            output_path = (
                Path(self.config_loader.get("data_paths.analysis_output"))
                / "optimization_results.json"
            )
            self.optimizer.save_optimization_results(output_path)

            # 打印优化摘要
            self.optimizer.print_optimization_summary(optimization_results)

            logger.info("优化阶段完成")
            return True

        except Exception as e:
            logger.error(f"优化阶段发生错误: {e}")
            return False

    def generate_final_report(self) -> Dict:
        """生成最终报告"""
        logger.info("生成最终报告...")

        try:
            report = {
                "pipeline_info": {
                    "timestamp": datetime.now().isoformat(),
                    "config_file": str(self.config_loader.config_path),
                    "version": self.config_loader.get("project.version", "1.0.0"),
                },
                "execution_summary": {
                    "analysis_completed": self.results["analysis"] is not None,
                    "screening_completed": self.results["screening"] is not None,
                    "optimization_completed": self.results["optimization"] is not None,
                },
            }

            # 添加分析结果摘要
            if self.results["analysis"]:
                analysis = self.results["analysis"]
                report["analysis_summary"] = {
                    "total_strategies_analyzed": analysis["summary"][
                        "total_strategies"
                    ],
                    "best_sharpe_ratio": analysis["summary"]["best_strategy"][
                        "performance"
                    ]["sharpe_ratio"],
                    "best_total_return": analysis["summary"]["best_strategy"][
                        "performance"
                    ]["total_return"],
                    "core_factors": analysis["recommendations"]["core_factors"][:4],
                }

            # 添加筛选结果摘要
            if self.results["screening"]:
                screening = self.results["screening"]
                report["screening_summary"] = {
                    "candidates_generated": screening["candidates_generated"],
                    "strategies_screened": screening["strategies_screened"],
                    "best_sharpe_ratio": screening["screening_analysis"]["summary"][
                        "best_sharpe"
                    ],
                }

            # 添加优化结果摘要
            if self.results["optimization"]:
                optimization = self.results["optimization"]
                report["optimization_summary"] = {
                    "best_target": optimization["best_target"],
                    "best_score": optimization["best_score"],
                    "factor_universe": optimization["factor_universe"],
                }

            # 生成策略建议
            report["strategy_recommendations"] = (
                self._generate_strategy_recommendations()
            )

            self.results["pipeline_summary"] = report

            # 保存最终报告
            output_path = (
                Path(self.config_loader.get("data_paths.analysis_output"))
                / "final_strategy_report.json"
            )
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            logger.info(f"最终报告已保存至: {output_path}")
            return report

        except Exception as e:
            logger.error(f"生成最终报告失败: {e}")
            return {}

    def _generate_strategy_recommendations(self) -> Dict:
        """生成策略建议"""
        recommendations = {
            "recommended_strategies": [],
            "risk_considerations": [],
            "implementation_notes": [],
        }

        # 基于优化结果推荐策略
        if self.results["optimization"]:
            optimization = self.results["optimization"]
            opt_results = optimization["optimization_results"]

            for target, result in opt_results.items():
                strategy = {
                    "name": f"优化策略_{target}",
                    "target": target,
                    "weights": result["weights"],
                    "expected_performance": result["performance"],
                    "fitness_score": result["fitness_score"],
                }
                recommendations["recommended_strategies"].append(strategy)

            # 添加风险考虑
            recommendations["risk_considerations"] = [
                "基于历史数据优化，实际表现可能有所不同",
                "建议定期重新评估和调整策略参数",
                "关注市场环境变化对因子有效性的影响",
                "实施适当的风险控制和仓位管理",
            ]

            # 添加实施说明
            recommendations["implementation_notes"] = [
                f"推荐使用{optimization['best_target']}策略作为主要配置",
                "建议在模拟环境中先测试策略表现",
                "可根据个人风险承受能力调整权重配置",
                "定期监控因子表现并更新策略",
            ]

        return recommendations

    def run_complete_pipeline(
        self,
        vbt_results_path: str,
        n_candidates: int = None,
        optimization_targets: list = None,
    ) -> bool:
        """运行完整流水线"""
        logger.info("开始执行完整精细策略流水线")
        logger.info(f"VBT结果路径: {vbt_results_path}")

        start_time = datetime.now()

        try:
            # 设置组件
            self.setup_components()

            # 执行各个阶段
            if not self.run_analysis(vbt_results_path):
                return False

            if not self.run_screening(n_candidates):
                return False

            if not self.run_optimization(optimization_targets):
                return False

            # 生成最终报告
            final_report = self.generate_final_report()

            # 打印流水线摘要
            self._print_pipeline_summary(start_time, final_report)

            logger.info("完整流水线执行成功")
            return True

        except Exception as e:
            logger.error(f"流水线执行失败: {e}")
            return False

    def _print_pipeline_summary(self, start_time: datetime, final_report: Dict):
        """打印流水线摘要"""
        end_time = datetime.now()
        duration = end_time - start_time

        print("\n" + "=" * 80)
        print("🎯 ETF轮动精细策略流水线执行完成")
        print("=" * 80)

        print(f"⏱️  执行时间: {duration}")
        print(f"📅 完成时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

        if final_report.get("analysis_summary"):
            analysis = final_report["analysis_summary"]
            print(f"\n📊 分析阶段摘要:")
            print(f"  分析策略数: {analysis['total_strategies_analyzed']:,}")
            print(f"  最佳夏普比率: {analysis['best_sharpe_ratio']:.3f}")
            print(f"  最佳总收益: {analysis['best_total_return']:.2f}%")

        if final_report.get("screening_summary"):
            screening = final_report["screening_summary"]
            print(f"\n🔍 筛选阶段摘要:")
            print(f"  生成候选数: {screening['candidates_generated']:,}")
            print(f"  通过筛选数: {screening['strategies_screened']:,}")
            print(
                f"  筛选成功率: {screening['strategies_screened']/screening['candidates_generated']:.1%}"
            )

        if final_report.get("optimization_summary"):
            optimization = final_report["optimization_summary"]
            print(f"\n🚀 优化阶段摘要:")
            print(f"  最佳目标: {optimization['best_target']}")
            print(f"  最佳评分: {optimization['best_score']:.4f}")
            print(f"  因子数量: {len(optimization['factor_universe'])}")

        if final_report.get("strategy_recommendations"):
            recommendations = final_report["strategy_recommendations"]
            print(f"\n💡 策略推荐:")
            for i, strategy in enumerate(
                recommendations["recommended_strategies"][:3], 1
            ):
                print(
                    f"  {i}. {strategy['name']} (评分: {strategy['fitness_score']:.4f})"
                )

        print(
            f"\n📁 结果文件保存在: {self.config_loader.get('data_paths.analysis_output')}"
        )
        print("=" * 80)


def main():
    """主函数"""
    # 统一配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/fine_strategy.log", encoding="utf-8"),
        ],
    )

    parser = argparse.ArgumentParser(description="ETF轮动精细策略流水线")
    parser.add_argument(
        "--vbt-results", type=str, required=True, help="VBT回测结果路径"
    )
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--candidates", type=int, help="筛选候选数量")
    parser.add_argument(
        "--targets", nargs="+", type=int, default=[3, 5, 8], help="优化目标Top-N值"
    )
    parser.add_argument(
        "--stage",
        choices=["analysis", "screening", "optimization", "all"],
        default="all",
        help="执行阶段",
    )

    args = parser.parse_args()

    # 确保日志目录存在
    Path("logs").mkdir(exist_ok=True)

    # 创建流水线
    try:
        pipeline = FineStrategyPipeline(args.config)

        if args.stage == "all":
            success = pipeline.run_complete_pipeline(
                args.vbt_results, args.candidates, args.targets
            )
        elif args.stage == "analysis":
            pipeline.setup_components()
            success = pipeline.run_analysis(args.vbt_results)
        elif args.stage == "screening":
            pipeline.setup_components()
            success = pipeline.run_analysis(
                args.vbt_results
            ) and pipeline.run_screening(args.candidates)
        elif args.stage == "optimization":
            pipeline.setup_components()
            success = (
                pipeline.run_analysis(args.vbt_results)
                and pipeline.run_screening(args.candidates)
                and pipeline.run_optimization(args.targets)
            )

        if success:
            logger.info("流水线执行成功")
            return 0
        else:
            logger.error("流水线执行失败")
            return 1

    except Exception as e:
        logger.error(f"流水线执行异常: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
