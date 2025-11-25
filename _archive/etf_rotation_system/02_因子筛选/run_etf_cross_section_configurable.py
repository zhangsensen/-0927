#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ETF横截面因子筛选 - 可配置版本 - Linus工程风格"""

import glob
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from etf_cross_section_config import ETF_STANDARD_CONFIG, ETFCrossSectionConfig
from scipy import stats


class ETFCrossSectionScreener:
    """ETF横截面因子筛选器 - 配置驱动实现"""

    def __init__(self, config: ETFCrossSectionConfig):
        self.config = config
        self._validate_config()

    def _validate_config(self):
        """验证配置有效性"""
        if not self.config.data_source.price_dir.exists():
            raise FileNotFoundError(
                f"价格数据目录不存在: {self.config.data_source.price_dir}"
            )

        if not self.config.data_source.panel_file.exists():
            raise FileNotFoundError(
                f"因子面板文件不存在: {self.config.data_source.panel_file}"
            )

    def _load_price_data(self) -> pd.DataFrame:
        """加载价格数据 - 配置驱动"""
        if self.config.progress_reporting:
            print(f"\n📈 加载价格数据: {self.config.data_source.price_dir}")

        price_files = sorted(
            glob.glob(
                str(
                    self.config.data_source.price_dir
                    / self.config.data_source.file_pattern
                )
            )
        )

        prices = []
        for f in price_files:
            df = pd.read_parquet(f, columns=self.config.data_source.price_columns)

            # 配置驱动的symbol提取
            if self.config.data_source.symbol_extract_method == "stem_split":
                symbol = Path(f).stem.split("_")[0]
            else:
                # 预留其他提取方法的扩展空间
                symbol = Path(f).stem.split("_")[0]

            df["symbol"] = symbol
            df["date"] = pd.to_datetime(df[self.config.data_source.price_columns[0]])
            prices.append(df)

        price_df = pd.concat(prices, ignore_index=True)
        price_df = price_df.set_index(["symbol", "date"]).sort_index()

        if self.config.progress_reporting:
            print(f"  ✅ 加载完成: {len(price_files)} 个ETF, {len(price_df)} 条记录")

        return price_df

    def calculate_multi_period_ic(
        self, panel: pd.DataFrame, price_df: pd.DataFrame
    ) -> pd.DataFrame:
        """多周期IC分析 - 完全配置驱动"""
        if self.config.progress_reporting:
            print(f"\n🔬 多周期IC分析: {self.config.analysis.ic_periods}")

        # 预计算所有周期的未来收益（向量化）
        # 使用shift(-period)计算真实的未来收益率（向前看）
        fwd_rets = {}
        for period in self.config.analysis.ic_periods:
            future_price = price_df.groupby(level="symbol")["close"].shift(-period)
            current_price = price_df["close"]
            fwd_rets[period] = future_price / current_price - 1

        results = []

        # 对每个因子向量化计算
        for factor_name in panel.columns:
            factor_data = panel[factor_name].dropna()
            period_ics = {}
            all_date_ics = []

            # 对每个周期向量化计算IC
            for period in self.config.analysis.ic_periods:
                fwd_ret = fwd_rets[period]

                # 对齐数据
                common_idx = factor_data.index.intersection(fwd_ret.index)
                f = factor_data.loc[common_idx]
                r = fwd_ret.loc[common_idx].dropna()

                final_idx = f.index.intersection(r.index)
                if len(final_idx) < self.config.analysis.min_observations:
                    continue

                # 真向量化：NumPy矩阵运算
                factor_pivot = f.loc[final_idx].unstack(level="symbol")
                return_pivot = r.loc[final_idx].unstack(level="symbol")

                # 对齐日期
                common_dates = factor_pivot.index.intersection(return_pivot.index)
                factor_mat = factor_pivot.loc[common_dates].values
                return_mat = return_pivot.loc[common_dates].values

                # 批量排名（向量化）
                def rank_row(row):
                    mask = ~np.isnan(row)
                    if mask.sum() < self.config.analysis.min_ranking_samples:
                        return np.full_like(row, np.nan)
                    ranked = np.full_like(row, np.nan)
                    ranked[mask] = stats.rankdata(row[mask])
                    return ranked

                factor_ranked = np.apply_along_axis(rank_row, 1, factor_mat)
                return_ranked = np.apply_along_axis(rank_row, 1, return_mat)

                # 批量计算IC
                date_ics = []
                for i in range(len(common_dates)):
                    f_rank = factor_ranked[i]
                    r_rank = return_ranked[i]
                    mask = ~(np.isnan(f_rank) | np.isnan(r_rank))
                    if mask.sum() >= self.config.analysis.min_ranking_samples:
                        f_valid = f_rank[mask]
                        r_valid = r_rank[mask]
                        if f_valid.std() > 0 and r_valid.std() > 0:
                            ic = np.corrcoef(f_valid, r_valid)[0, 1]
                            if not np.isnan(ic):
                                date_ics.append(ic)

                date_ics = np.array(date_ics)

                if len(date_ics) >= self.config.analysis.min_ic_observations:
                    period_ics[f"ic_{period}d"] = np.mean(date_ics)
                    period_ics[f"ir_{period}d"] = np.mean(date_ics) / (
                        float(np.std(date_ics))
                        + float(self.config.analysis.epsilon_small)
                    )
                    all_date_ics.extend(date_ics)

            if (
                not period_ics
                or len(all_date_ics) < self.config.analysis.min_ic_observations
            ):
                continue

            # Linus优化: 使用5D IC作为主指标,不混合多周期
            # 原逻辑: ic_mean = np.mean(all_date_ics) 混合了1D/5D/10D/20D
            # 新逻辑: 只用5D IC排序,避免长周期因子虚高
            ic_5d = period_ics.get("ic_5d", np.nan)
            ir_5d = period_ics.get("ir_5d", np.nan)

            # 保留综合IC供参考,但筛选用ic_5d
            ic_mean = np.mean(all_date_ics)
            ic_std = np.std(all_date_ics)
            ic_ir = ic_mean / ic_std if ic_std > 0 else 0

            # 稳定性：IC时间序列自相关
            half = int(len(all_date_ics) * self.config.analysis.stability_split_ratio)
            stability = (
                np.corrcoef(all_date_ics[:half], all_date_ics[half : 2 * half])[0, 1]
                if half > 10
                else 0
            )

            # t检验
            t_stat, p_value = stats.ttest_1samp(all_date_ics, 0)

            result = {
                "factor": factor_name,
                "ic_mean": ic_mean,  # 综合IC(供参考)
                "ic_std": ic_std,
                "ic_ir": ic_ir,  # 综合IR(供参考)
                "ic_5d": ic_5d,  # 🎯 5D IC(筛选用)
                "ir_5d": ir_5d,  # 🎯 5D IR(筛选用)
                "ic_positive_rate": np.mean(np.array(all_date_ics) > 0),
                "stability": stability,
                "t_stat": t_stat,
                "p_value": p_value,
                "sample_size": len(all_date_ics),
                "coverage": len(factor_data) / len(panel),
            }
            result.update(period_ics)
            results.append(result)

            if self.config.progress_reporting:
                # 显示5D IC,不显示混合IC
                print(
                    f"  {factor_name:30s} IC_5D={ic_5d:+.4f} IR_5D={ir_5d:+.4f} Stab={stability:+.3f}"
                )

        return pd.DataFrame(results).sort_values("ir_5d", ascending=False, key=abs)

    def apply_fdr_correction(self, ic_df: pd.DataFrame) -> pd.DataFrame:
        """FDR校正 - 配置驱动"""
        if not self.config.screening.use_fdr:
            return ic_df.copy()

        p_values = ic_df["p_value"].values
        n = len(p_values)

        # 排序
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]

        # BH临界值
        critical = np.arange(1, n + 1) * self.config.screening.fdr_alpha / n

        # 找到最大的满足条件的索引
        rejected = sorted_p <= critical
        if rejected.any():
            max_idx = np.where(rejected)[0].max()
            passed_idx = sorted_idx[: max_idx + 1]
            return ic_df.iloc[passed_idx].copy()

        return pd.DataFrame()

    def remove_correlated_factors(
        self, ic_df: pd.DataFrame, panel: pd.DataFrame
    ) -> pd.DataFrame:
        """去除高相关因子 - 配置驱动"""
        if len(ic_df) <= 1:
            return ic_df

        factors = ic_df["factor"].tolist()
        factor_data = panel[factors]

        # 计算相关矩阵
        corr_matrix = factor_data.corr(
            method=self.config.analysis.correlation_method,
            min_periods=self.config.analysis.correlation_min_periods,
        ).abs()

        # 贪心去重：保留IC_IR更高的
        to_remove = set()
        for i, f1 in enumerate(factors):
            if f1 in to_remove:
                continue
            for f2 in factors[i + 1 :]:
                if f2 in to_remove:
                    continue
                if corr_matrix.loc[f1, f2] > self.config.screening.max_correlation:
                    ir1 = ic_df[ic_df["factor"] == f1]["ic_ir"].values[0]
                    ir2 = ic_df[ic_df["factor"] == f2]["ic_ir"].values[0]
                    to_remove.add(f2 if abs(ir1) > abs(ir2) else f1)

        return ic_df[~ic_df["factor"].isin(to_remove)].copy()

    def _get_factor_tier(self, ic_mean: float, ic_ir: float) -> str:
        """获取因子评级 - 配置驱动"""
        thresholds = self.config.screening.tier_thresholds
        labels = self.config.screening.tier_labels

        if (
            abs(ic_mean) >= thresholds["core"]["ic"]
            and abs(ic_ir) >= thresholds["core"]["ir"]
        ):
            return labels["core"]
        elif (
            abs(ic_mean) >= thresholds["supplement"]["ic"]
            and abs(ic_ir) >= thresholds["supplement"]["ir"]
        ):
            return labels["supplement"]
        else:
            return labels["research"]

    def screen_factors(self, ic_df: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
        """因子筛选 - Linus优化: 用5D IC筛选,不混合多周期"""
        config = self.config.screening

        if self.config.progress_reporting:
            print(f"\n🎯 筛选标准 (Linus优化):")
            print(f"  IC_5D >= {config.min_ic} ({config.min_ic:.1%}) ⚡ 只用5D IC")
            print(f"  IR_5D >= {config.min_ir}")
            print(f"  p-value <= {config.max_pvalue}")
            print(f"  覆盖率 >= {config.min_coverage}")
            print(f"  最大相关性 = {config.max_correlation} (从0.65放宽)")
            print(f"  FDR校正 = {'启用' if config.use_fdr else '禁用'}")
            if config.force_include_factors:
                print(f"  🔒 强制保留: {config.force_include_factors}")
            if config.max_factors < 50:
                print(f"  📊 最大因子数: {config.max_factors}")

        # 🔒 第0步：分离强制保留因子
        force_include = (
            config.force_include_factors if config.force_include_factors else []
        )
        mandatory_mask = ic_df["factor"].isin(force_include)
        mandatory_df = ic_df[mandatory_mask].copy()
        other_df = ic_df[~mandatory_mask].copy()

        if self.config.progress_reporting and len(mandatory_df) > 0:
            print(f"\n🔒 强制保留: {len(mandatory_df)} 因子")
            for _, row in mandatory_df.iterrows():
                print(
                    f"  ✓ {row['factor']:30s} IC_5D={row['ic_5d']:+.4f} IR_5D={row['ir_5d']:+.4f}"
                )

        # Linus修改: 用5D IC/IR筛选,不用混合IC
        mask = (
            (other_df["ic_5d"].abs() >= config.min_ic)  # 改用ic_5d
            & (other_df["ir_5d"].abs() >= config.min_ir)  # 改用ir_5d
            & (other_df["p_value"] <= config.max_pvalue)
            & (other_df["coverage"] >= config.min_coverage)
        )
        passed = other_df[mask].copy()

        if self.config.progress_reporting:
            print(f"\n✅ 基础筛选: {len(passed)}/{len(other_df)} 因子")

        # 第2步：FDR校正（仅对非强制因子）
        if config.use_fdr and len(passed) > 0:
            passed_fdr = self.apply_fdr_correction(passed)
            if self.config.progress_reporting:
                print(f"✅ FDR校正: {len(passed_fdr)}/{len(passed)} 因子")
            if len(passed_fdr) == 0:
                if self.config.progress_reporting:
                    print("⚠️ FDR校正后无因子通过，返回基础筛选结果")
                passed_fdr = passed
        else:
            passed_fdr = passed

        # 🔗 第3步：合并强制因子和筛选通过因子
        combined_df = pd.concat([mandatory_df, passed_fdr], ignore_index=True)

        if len(combined_df) == 0:
            return combined_df

        # 第4步：去重（强制因子参与，但优先保留）
        passed_final = self._remove_correlated_with_priority(
            combined_df, panel, force_include
        )
        if self.config.progress_reporting:
            print(f"✅ 去重后: {len(passed_final)}/{len(combined_df)} 因子")

        # 📊 第5步：限制最大因子数（基于priority_metric排序）
        if config.max_factors < len(passed_final):
            if self.config.progress_reporting:
                print(f"\n⚠️ 因子数超限: {len(passed_final)} > {config.max_factors}")

            # 分离强制和非强制
            final_mandatory = passed_final[passed_final["factor"].isin(force_include)]
            final_optional = passed_final[~passed_final["factor"].isin(force_include)]

            # 按优先级排序非强制因子
            if config.priority_metric == "ic_ir":
                final_optional = final_optional.sort_values(
                    "ic_ir", ascending=False, key=abs
                )
            elif config.priority_metric == "ic_mean":
                final_optional = final_optional.sort_values(
                    "ic_mean", ascending=False, key=abs
                )
            elif config.priority_metric == "combined":
                final_optional["combined_score"] = (
                    final_optional["ic_mean"].abs() * 0.5
                    + final_optional["ic_ir"].abs() * 0.5
                )
                final_optional = final_optional.sort_values(
                    "combined_score", ascending=False
                )

            # 保留top因子
            n_optional = max(0, config.max_factors - len(final_mandatory))
            final_optional = final_optional.head(n_optional)
            passed_final = pd.concat(
                [final_mandatory, final_optional], ignore_index=True
            )

            if self.config.progress_reporting:
                print(
                    f"✂️ 截断为: {len(passed_final)} 因子 (强制{len(final_mandatory)} + Top{len(final_optional)})"
                )

        # 第6步：分层评级
        if self.config.progress_reporting:
            print(f"\n📊 最终因子分层评级:")
            for _, row in passed_final.iterrows():
                tier = self._get_factor_tier(row["ic_mean"], row["ic_ir"])
                force_mark = "🔒" if row["factor"] in force_include else "  "
                print(
                    f"  {force_mark}{tier} {row['factor']:30s} IC={row['ic_mean']:+.4f} IR={row['ic_ir']:+.4f}"
                )

        return passed_final

    def _remove_correlated_with_priority(
        self, ic_df: pd.DataFrame, panel: pd.DataFrame, priority_factors: List[str]
    ) -> pd.DataFrame:
        """去重逻辑 - 优先保留强制因子"""
        config = self.config.screening
        factors = ic_df["factor"].tolist()

        if len(factors) <= 1:
            return ic_df

        # 构建相关性矩阵（直接从panel提取因子列）
        factor_data = panel[factors]
        corr_matrix = factor_data.corr(
            method=self.config.analysis.correlation_method,
            min_periods=self.config.analysis.correlation_min_periods,
        ).abs()

        # 贪心去重，优先保留强制因子
        to_remove = set()
        for i, f1 in enumerate(factors):
            if f1 in to_remove:
                continue
            for f2 in factors[i + 1 :]:
                if f2 in to_remove:
                    continue

                if corr_matrix.loc[f1, f2] > config.max_correlation:
                    # 优先保留强制因子
                    if f1 in priority_factors and f2 not in priority_factors:
                        to_remove.add(f2)
                    elif f2 in priority_factors and f1 not in priority_factors:
                        to_remove.add(f1)
                        break  # f1已被移除，跳出内层循环
                    else:
                        # 都是强制或都不是强制，按IC_IR决定
                        ir1 = ic_df[ic_df["factor"] == f1]["ic_ir"].values[0]
                        ir2 = ic_df[ic_df["factor"] == f2]["ic_ir"].values[0]
                        to_remove.add(f2 if abs(ir1) >= abs(ir2) else f1)
                        if f1 in to_remove:
                            break

        return ic_df[~ic_df["factor"].isin(to_remove)].copy()

    def _save_results(self, ic_df: pd.DataFrame, passed_factors: pd.DataFrame) -> Path:
        """保存结果 - 配置驱动"""
        output_dir = self.config.output.output_dir

        if self.config.output.use_timestamp_subdir:
            timestamp = datetime.now().strftime(self.config.output.timestamp_format)
            timestamp_dir = (
                output_dir / f"{self.config.output.subdir_prefix}{timestamp}"
            )
        else:
            timestamp_dir = output_dir

        timestamp_dir.mkdir(parents=True, exist_ok=True)

        # 保存完整IC分析
        ic_file = timestamp_dir / self.config.output.files["ic_analysis"]
        ic_df.to_csv(ic_file, index=False)
        if self.config.progress_reporting:
            print(f"\n💾 IC分析: {ic_file}")

        # 保存筛选结果和报告
        if len(passed_factors) > 0:
            passed_file = timestamp_dir / self.config.output.files["passed_factors"]
            passed_factors.to_csv(passed_file, index=False)
            if self.config.progress_reporting:
                print(f"💾 筛选结果: {passed_file}")

            self._generate_screening_report(timestamp_dir, ic_df, passed_factors)
        else:
            self._generate_empty_report(timestamp_dir)

        return timestamp_dir

    def _generate_screening_report(
        self, output_dir: Path, ic_df: pd.DataFrame, passed: pd.DataFrame
    ):
        """生成筛选报告 - 配置驱动"""
        report_file = output_dir / self.config.output.files["screening_report"]

        with open(report_file, "w", encoding=self.config.output.encoding) as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"ETF横截面因子筛选报告\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"筛选时间: {timestamp}\n")
            f.write(f"面板文件: {self.config.data_source.panel_file}\n")
            f.write(f"价格数据目录: {self.config.data_source.price_dir}\n\n")

            f.write(f"筛选标准:\n")
            f.write(
                f"  IC均值 >= {self.config.screening.min_ic} ({self.config.screening.min_ic:.1%})\n"
            )
            f.write(f"  IC_IR >= {self.config.screening.min_ir}\n")
            f.write(f"  p-value <= {self.config.screening.max_pvalue}\n")
            f.write(f"  覆盖率 >= {self.config.screening.min_coverage}\n")
            f.write(f"  最大相关性 = {self.config.screening.max_correlation}\n")
            f.write(
                f"  FDR校正 = {'启用' if self.config.screening.use_fdr else '禁用'}\n\n"
            )

            f.write(f"筛选结果:\n")
            f.write(f"  总因子数: {len(ic_df)}\n")
            f.write(f"  通过筛选: {len(passed)}\n")
            f.write(f"  通过率: {len(passed)/len(ic_df):.1%}\n\n")

            if self.config.output.include_factor_details:
                f.write(f"🏆 因子评级详情:\n")
                for _, row in passed.iterrows():
                    tier = self._get_factor_tier(row["ic_mean"], row["ic_ir"])
                    f.write(
                        f"  {tier} {row['factor']:30s} IC={row['ic_mean']:+.4f} IR={row['ic_ir']:+.4f} p={row['p_value']:.2e}\n"
                    )

        if self.config.progress_reporting:
            print(f"💾 筛选报告: {report_file}")

    def _generate_empty_report(self, output_dir: Path):
        """生成无因子通过的报告"""
        report_file = output_dir / self.config.output.files["screening_report"]

        with open(report_file, "w", encoding=self.config.output.encoding) as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"ETF横截面因子筛选报告\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"筛选时间: {timestamp}\n")
            f.write(f"⚠️ 无因子通过筛选\n")

        if self.config.progress_reporting:
            print(f"💾 筛选报告: {report_file}")

    def run(self) -> Dict[str, Any]:
        """运行完整筛选流程"""
        if self.config.progress_reporting:
            print("=" * 80)
            print("ETF横截面因子筛选 - 可配置版本")
            print("=" * 80)

        # 加载数据
        panel = pd.read_parquet(self.config.data_source.panel_file)
        price_df = self._load_price_data()

        if self.config.progress_reporting:
            print(f"\n📊 因子面板: {panel.shape[1]} 因子, {panel.shape[0]} 观测值")

        # IC分析
        ic_df = self.calculate_multi_period_ic(panel, price_df)

        # 筛选
        passed = self.screen_factors(ic_df, panel)

        # 保存结果
        output_dir = self._save_results(ic_df, passed)

        # 返回结果摘要
        result_summary = {
            "total_factors": len(ic_df),
            "passed_factors": len(passed),
            "pass_rate": len(passed) / len(ic_df) if len(ic_df) > 0 else 0,
            "output_dir": output_dir,
            "ic_analysis_file": output_dir / self.config.output.files["ic_analysis"],
            "passed_factors_file": (
                output_dir / self.config.output.files["passed_factors"]
                if len(passed) > 0
                else None
            ),
            "report_file": output_dir / self.config.output.files["screening_report"],
        }

        if self.config.progress_reporting:
            print(f"\n📁 所有结果保存在: {output_dir}")
            if len(passed) > 0:
                print(f"\n🏆 Top 10 因子:")
                display_cols = [
                    "factor",
                    "ic_mean",
                    "ic_ir",
                    "ic_positive_rate",
                    "p_value",
                ]
                print(passed.head(10)[display_cols].to_string(index=False))
            print("✅ 完成")

        return result_summary


def main():
    """主函数 - 支持命令行和配置文件"""
    import argparse

    parser = argparse.ArgumentParser(description="ETF横截面因子筛选 - 可配置版本")
    parser.add_argument("--config", help="YAML配置文件路径")
    parser.add_argument(
        "--create-config", help="创建默认配置文件到指定路径", action="store_true"
    )
    parser.add_argument("--panel", help="因子面板parquet文件 (覆盖配置文件)")
    parser.add_argument("--price-dir", help="价格数据目录 (覆盖配置文件)")
    parser.add_argument("--output-dir", help="输出目录 (覆盖配置文件)")
    parser.add_argument("--strict", action="store_true", help="使用严格筛选标准")
    parser.add_argument("--relaxed", action="store_true", help="使用宽松筛选标准")

    args = parser.parse_args()

    # 创建配置文件
    if args.create_config:
        config_path = Path("etf_cross_section_config.yaml")
        from etf_cross_section_config import create_default_config_file

        create_default_config_file(config_path)
        print(f"✅ 默认配置文件已创建: {config_path}")
        return

    # 加载配置
    if args.config:
        config = ETFCrossSectionConfig.from_yaml(Path(args.config))
    elif args.strict:
        from etf_cross_section_config import ETF_STRICT_CONFIG

        config = ETF_STRICT_CONFIG
    elif args.relaxed:
        from etf_cross_section_config import ETF_RELAXED_CONFIG

        config = ETF_RELAXED_CONFIG
    else:
        config = ETF_STANDARD_CONFIG

    # 命令行参数覆盖配置
    if args.panel:
        config.data_source.panel_file = Path(args.panel)
    if args.price_dir:
        config.data_source.price_dir = Path(args.price_dir)
    if args.output_dir:
        config.output.output_dir = Path(args.output_dir)

    # 运行筛选
    screener = ETFCrossSectionScreener(config)
    results = screener.run()

    # 输出结果摘要
    if not config.progress_reporting:
        print(
            f"✅ 完成: {results['passed_factors']}/{results['total_factors']} 因子通过筛选"
        )
        print(f"📁 结果保存在: {results['output_dir']}")


if __name__ == "__main__":
    main()
