#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF横截面因子挖掘完整流水线
整合候选因子生成、批量计算、IC分析、稳定性测试、多维筛选和分类标注
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from factor_system.factor_engine.providers.etf_cross_section_provider import ETFCrossSectionDataManager
from factor_system.factor_engine.factors.etf_cross_section.candidate_factor_generator import ETFCandidateFactorGenerator
from factor_system.factor_engine.factors.etf_cross_section.batch_factor_calculator import BatchFactorCalculator, calculate_all_etf_factors
from factor_system.factor_engine.factors.etf_cross_section.ic_analyzer import ICAnalyzer
from factor_system.factor_engine.factors.etf_cross_section.stability_analyzer import StabilityAnalyzer
from factor_system.factor_engine.factors.etf_cross_section.factor_screener import FactorScreener, ScreeningCriteria, screen_etf_factors
from factor_system.factor_engine.factors.etf_cross_section.factor_classifier import classify_etf_factors

from factor_system.utils import safe_operation, FactorSystemError

logger = logging.getLogger(__name__)


class ETFFactorMiningPipeline:
    """ETF因子挖掘流水线"""

    def __init__(self, output_base_dir: str = None):
        """
        初始化挖掘流水线

        Args:
            output_base_dir: 输出基础目录
        """
        self.output_base_dir = Path(output_base_dir) if output_base_dir else Path("factor_system/factor_output/etf_cross_section/mining_results")
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

        # 创建子目录
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_base_dir / f"run_{self.timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # 初始化组件
        self.data_manager = ETFCrossSectionDataManager()
        self.factor_generator = ETFCandidateFactorGenerator()
        self.calculator = BatchFactorCalculator()
        self.ic_analyzer = ICAnalyzer()
        self.stability_analyzer = StabilityAnalyzer(self.ic_analyzer)
        self.screener = FactorScreener()

        logger.info(f"ETF因子挖掘流水线初始化完成")
        logger.info(f"运行目录: {self.run_dir}")

    def load_price_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        加载价格数据

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            价格数据
        """
        logger.info(f"加载价格数据: {start_date} ~ {end_date}")

        try:
            # 获取ETF列表
            etf_list = self.data_manager.get_etf_list()
            logger.info(f"找到 {len(etf_list)} 只ETF")

            # 加载价格数据
            price_data = self.data_manager.load_price_data(etf_list, start_date, end_date)

            if price_data is None or price_data.empty:
                raise ValueError("无法加载价格数据")

            logger.info(f"成功加载价格数据: {len(price_data)} 条记录")
            logger.info(f"日期范围: {price_data['date'].min()} ~ {price_data['date'].max()}")
            logger.info(f"ETF数量: {price_data['symbol'].nunique()}")

            # 保存价格数据
            price_file = self.run_dir / "price_data.parquet"
            price_data.to_parquet(price_file, index=False)
            logger.info(f"价格数据已保存到: {price_file}")

            return price_data

        except Exception as e:
            logger.error(f"加载价格数据失败: {str(e)}")
            raise

    def generate_candidate_factors(self) -> List:
        """
        生成候选因子

        Returns:
            候选因子列表
        """
        logger.info("生成候选因子...")

        try:
            variants = self.factor_generator.generate_all_variants()

            logger.info(f"成功生成 {len(variants)} 个候选因子")

            # 保存候选因子列表
            variants_file = self.run_dir / "candidate_factors.csv"
            self.factor_generator.save_variants_to_file(variants, str(variants_file))

            return variants

        except Exception as e:
            logger.error(f"生成候选因子失败: {str(e)}")
            raise

    def calculate_factors(self, variants: List, price_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        计算因子数据

        Args:
            variants: 候选因子列表
            price_data: 价格数据

        Returns:
            因子数据字典
        """
        logger.info(f"开始计算 {len(variants)} 个因子...")

        try:
            # 准备参数
            symbols = price_data['symbol'].unique().tolist()
            start_date = price_data['date'].min().strftime('%Y-%m-%d')
            end_date = price_data['date'].max().strftime('%Y-%m-%d')

            # 创建因子计算目录
            factor_dir = self.run_dir / "calculated_factors"
            factor_dir.mkdir(exist_ok=True)

            # 批量计算因子
            factors_data = self.calculator.calculate_factors_batch(
                variants=variants,
                symbols=symbols,
                timeframe="daily",
                start_date=pd.to_datetime(start_date),
                end_date=pd.to_datetime(end_date),
                output_dir=str(factor_dir)
            )

            logger.info(f"因子计算完成: {len(factors_data)}/{len(variants)} 个因子成功")

            return factors_data

        except Exception as e:
            logger.error(f"因子计算失败: {str(e)}")
            raise

    def analyze_factors(self, factors_data: Dict[str, pd.DataFrame],
                       price_data: pd.DataFrame) -> tuple:
        """
        分析因子（IC分析和稳定性分析）

        Args:
            factors_data: 因子数据
            price_data: 价格数据

        Returns:
            (IC分析结果, 稳定性分析结果)
        """
        logger.info("开始因子分析...")

        try:
            # IC分析
            logger.info("执行IC分析...")
            ic_results = self.ic_analyzer.batch_analyze_factors(factors_data, price_data)

            # 保存IC分析结果
            ic_file = self.run_dir / "ic_analysis.csv"
            self.ic_analyzer.save_ic_analysis_results(ic_results, str(ic_file))

            logger.info(f"IC分析完成: {len(ic_results)} 个因子")

            # 稳定性分析
            logger.info("执行稳定性分析...")
            stability_results = self.stability_analyzer.batch_analyze_stability(factors_data, price_data)

            # 保存稳定性分析结果
            stability_file = self.run_dir / "stability_analysis.csv"
            self.stability_analyzer.save_stability_results(stability_results, str(stability_file))

            logger.info(f"稳定性分析完成: {len(stability_results)} 个因子")

            return ic_results, stability_results

        except Exception as e:
            logger.error(f"因子分析失败: {str(e)}")
            raise

    def screen_factors(self, factors_data: Dict[str, pd.DataFrame],
                      price_data: pd.DataFrame,
                      criteria: Optional[ScreeningCriteria] = None) -> Dict:
        """
        筛选因子

        Args:
            factors_data: 因子数据
            price_data: 价格数据
            criteria: 筛选标准

        Returns:
            筛选结果
        """
        logger.info("开始因子筛选...")

        try:
            screening_results = screen_etf_factors(
                factors_data=factors_data,
                price_data=price_data,
                criteria=criteria,
                output_dir=str(self.run_dir)
            )

            passed_count = sum(1 for r in screening_results.values() if r.screening_reason == "通过筛选")
            logger.info(f"因子筛选完成: {passed_count}/{len(factors_data)} 个因子通过筛选")

            return screening_results

        except Exception as e:
            logger.error(f"因子筛选失败: {str(e)}")
            raise

    def classify_factors(self, screening_results: Dict) -> Dict:
        """
        分类因子

        Args:
            screening_results: 筛选结果

        Returns:
            分类结果
        """
        logger.info("开始因子分类...")

        try:
            classification_results = classify_etf_factors(
                screening_results=screening_results,
                output_dir=str(self.run_dir)
            )

            logger.info(f"因子分类完成: {len(classification_results)} 个因子已分类")

            return classification_results

        except Exception as e:
            logger.error(f"因子分类失败: {str(e)}")
            raise

    def generate_final_report(self, screening_results: Dict,
                            classification_results: Dict) -> str:
        """
        生成最终报告

        Args:
            screening_results: 筛选结果
            classification_results: 分类结果

        Returns:
            报告文件路径
        """
        logger.info("生成最终报告...")

        try:
            report_file = self.run_dir / "final_report.md"

            # 统计信息
            total_factors = len(screening_results)
            passed_factors = [r for r in screening_results.values() if r.screening_reason == "通过筛选"]
            classified_factors = classification_results

            # 生成报告内容
            report_content = f"""# ETF横截面因子挖掘报告

## 基本信息
- 运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 运行ID: {self.timestamp}

## 挖掘结果统计
- 总候选因子数: {total_factors}
- 通过筛选因子数: {len(passed_factors)}
- 最终分类因子数: {len(classified_factors)}
- 通过率: {len(passed_factors)/total_factors:.1%}

## 筛选标准
- 最小IC均值: {self.screener.criteria.min_ic_mean}
- 最大IC p值: {self.screener.criteria.max_ic_pvalue}
- 最小IC胜率: {self.screener.criteria.min_ic_win_rate}
- 最小稳定性评分: {self.screener.criteria.min_stability_score}
- 最大相关性阈值: {self.screener.criteria.max_correlation}

## 分类统计
"""

            # 分类统计
            category_stats = {}
            for factor in classification_results.values():
                category = factor.category
                category_stats[category] = category_stats.get(category, 0) + 1

            for category, count in sorted(category_stats.items()):
                report_content += f"- {category}: {count} 个因子\n"

            # 顶级因子列表
            report_content += f"""
## 顶级因子（前20名）

| 排名 | 因子ID | 类别 | IC均值 | 稳定性评分 | 综合评分 |
|------|--------|------|--------|------------|----------|
"""

            # 排序并显示前20名
            passed_factors_sorted = sorted(
                [r for r in screening_results.values() if r.screening_reason == "通过筛选"],
                key=lambda x: x.overall_score,
                reverse=True
            )

            for i, result in enumerate(passed_factors_sorted[:20]):
                category = classification_results.get(result.variant_id)
                category_name = category.category if category else "未分类"

                report_content += f"| {i+1} | {result.variant_id} | {category_name} | {result.ic_mean:.4f} | {result.stability_score:.3f} | {result.overall_score:.2f} |\n"

            # 详细分析
            report_content += f"""
## 详细分析

### IC分析结果
- 平均IC均值: {np.mean([r.ic_mean for r in passed_factors]):.4f}
- 平均IC胜率: {np.mean([r.ic_win_rate for r in passed_factors]):.2%}
- 最大IC均值: {max([r.ic_mean for r in passed_factors]):.4f}
- 最小IC均值: {min([r.ic_mean for r in passed_factors]):.4f}

### 稳定性分析结果
- 平均稳定性评分: {np.mean([r.stability_score for r in passed_factors]):.3f}
- 最高稳定性评分: {max([r.stability_score for r in passed_factors]):.3f}
- 最低稳定性评分: {min([r.stability_score for r in passed_factors]):.3f}

### 类别分布
"""

            for category, count in sorted(category_stats.items()):
                percentage = count / len(classified_factors) * 100
                report_content += f"- {category}: {count} 个 ({percentage:.1f}%)\n"

            report_content += f"""
## 文件输出
所有结果文件已保存到: {self.run_dir}

主要文件:
- price_data.parquet: 原始价格数据
- candidate_factors.csv: 候选因子列表
- calculated_factors/: 计算后的因子数据
- ic_analysis.csv: IC分析结果
- stability_analysis.csv: 稳定性分析结果
- factor_screening_*.csv: 筛选结果
- factor_classification_*.csv: 分类结果
- final_report.md: 本报告

## 使用建议
1. 优先使用综合评分高的因子
2. 根据投资风格选择不同类别的因子
3. 注意因子的适用场景和风险特征
4. 建议结合多个不同类别的因子使用

---
报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

            # 保存报告
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)

            logger.info(f"最终报告已保存到: {report_file}")
            return str(report_file)

        except Exception as e:
            logger.error(f"生成报告失败: {str(e)}")
            raise

    def run_pipeline(self, start_date: str, end_date: str,
                    criteria: Optional[ScreeningCriteria] = None) -> Dict:
        """
        运行完整挖掘流水线

        Args:
            start_date: 开始日期
            end_date: 结束日期
            criteria: 筛选标准

        Returns:
            流水线结果字典
        """
        logger.info("开始运行ETF因子挖掘流水线...")
        logger.info(f"时间范围: {start_date} ~ {end_date}")

        pipeline_results = {}

        try:
            # 步骤1: 加载价格数据
            price_data = self.load_price_data(start_date, end_date)
            pipeline_results['price_data'] = price_data

            # 步骤2: 生成候选因子
            variants = self.generate_candidate_factors()
            pipeline_results['variants'] = variants

            # 步骤3: 计算因子
            factors_data = self.calculate_factors(variants, price_data)
            pipeline_results['factors_data'] = factors_data

            # 步骤4: 分析因子
            ic_results, stability_results = self.analyze_factors(factors_data, price_data)
            pipeline_results['ic_results'] = ic_results
            pipeline_results['stability_results'] = stability_results

            # 步骤5: 筛选因子
            screening_results = self.screen_factors(factors_data, price_data, criteria)
            pipeline_results['screening_results'] = screening_results

            # 步骤6: 分类因子
            classification_results = self.classify_factors(screening_results)
            pipeline_results['classification_results'] = classification_results

            # 步骤7: 生成报告
            report_file = self.generate_final_report(screening_results, classification_results)
            pipeline_results['report_file'] = report_file

            logger.info("ETF因子挖掘流水线运行完成！")
            logger.info(f"结果保存在: {self.run_dir}")
            logger.info(f"最终报告: {report_file}")

            return pipeline_results

        except Exception as e:
            logger.error(f"流水线运行失败: {str(e)}")
            raise


@safe_operation
def main():
    """主函数 - 运行ETF因子挖掘流水线"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('etf_factor_mining.log'),
            logging.StreamHandler()
        ]
    )

    # 运行参数
    start_date = "2024-01-01"
    end_date = "2025-10-14"

    # 自定义筛选标准（可选）
    criteria = ScreeningCriteria(
        min_ic_mean=0.015,          # 稍微降低IC要求
        max_ic_pvalue=0.1,          # 稍微放宽显著性要求
        min_ic_win_rate=0.45,       # 降低胜率要求
        min_stability_score=0.65,   # 降低稳定性要求
        max_correlation=0.9,        # 放宽相关性要求
        min_monotonicity_r2=0.7,    # 降低单调性要求
        min_sample_size=20          # 降低样本数要求
    )

    try:
        # 初始化并运行流水线
        pipeline = ETFFactorMiningPipeline()
        results = pipeline.run_pipeline(start_date, end_date, criteria)

        # 打印摘要
        screening_results = results['screening_results']
        classification_results = results['classification_results']

        passed_count = sum(1 for r in screening_results.values() if r.screening_reason == "通过筛选")
        total_count = len(screening_results)

        print(f"\n{'='*60}")
        print("🎉 ETF因子挖掘流水线运行完成！")
        print(f"{'='*60}")
        print(f"📊 挖掘统计:")
        print(f"   总候选因子: {total_count}")
        print(f"   通过筛选: {passed_count}")
        print(f"   最终保留: {len(classification_results)}")
        print(f"   通过率: {passed_count/total_count:.1%}")
        print(f"\n📁 结果目录: {pipeline.run_dir}")
        print(f"📄 最终报告: {results['report_file']}")
        print(f"{'='*60}")

    except Exception as e:
        logger.error(f"流水线运行失败: {str(e)}")
        raise


if __name__ == "__main__":
    main()