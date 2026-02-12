"""
VEC 集成模块 | VEC Integration for Factor Mining
================================================================================
Layer 2.5: 让挖掘因子能在真实执行框架下验证

提供批量VEC回测能力，替代纯IC分析，输出生产级绩效指标。
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


@dataclass
class VECValidationReport:
    """VEC回测验证报告"""

    factor_name: str

    # 收益指标
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0

    # 风险指标
    max_drawdown: float = 0.0
    volatility: float = 0.0
    calmar_ratio: float = 0.0

    # 交易指标
    n_trades: int = 0
    turnover: float = 0.0

    # 对比基准
    vs_benchmark: float = 0.0  # 相对等权基准的超额

    # 综合
    vec_score: float = 0.0  # 复合评分
    passed: bool = False

    def to_dict(self) -> dict:
        return {
            "factor_name": self.factor_name,
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "volatility": self.volatility,
            "calmar_ratio": self.calmar_ratio,
            "n_trades": self.n_trades,
            "turnover": self.turnover,
            "vs_benchmark": self.vs_benchmark,
            "vec_score": self.vec_score,
            "passed": self.passed,
        }


class VECFactorValidator:
    """
    VEC回测因子验证器

    对挖掘出的候选因子，使用真实执行参数进行快速VEC回测。
    """

    def __init__(
        self,
        close: pd.DataFrame,
        open_price: Optional[pd.DataFrame] = None,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
    ):
        """
        参数:
            close: 收盘价
            open_price: 开盘价（T1_OPEN执行模型需要），为None时用close替代
            config: 配置字典，为None时尝试从config_path加载
            config_path: 配置文件路径
        """
        self.close = close
        self.open_price = open_price if open_price is not None else close

        # 加载配置
        if config is None:
            if config_path is None:
                # 默认配置路径
                project_root = (
                    Path(__file__).resolve().parent.parent.parent.parent.parent
                )
                config_path = project_root / "configs" / "combo_wfo_config.yaml"

            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config

        # 提取生产级参数
        backtest_cfg = self.config.get("backtest", {})
        self.freq = backtest_cfg.get("freq", 5)
        self.pos_size = backtest_cfg.get("pos_size", 2)
        self.commission = backtest_cfg.get("commission_rate", 0.0002)
        self.initial_capital = backtest_cfg.get("initial_capital", 1_000_000)

        # 迟滞参数
        hyst_cfg = backtest_cfg.get("hysteresis", {})
        self.delta_rank = hyst_cfg.get("delta_rank", 0.10)
        self.min_hold_days = hyst_cfg.get("min_hold_days", 9)

        logger.info(
            "VECFactorValidator initialized: freq=%d, pos_size=%d, "
            "delta_rank=%.2f, min_hold_days=%d",
            self.freq,
            self.pos_size,
            self.delta_rank,
            self.min_hold_days,
        )

    def validate_single(
        self,
        factor_name: str,
        factor_df: pd.DataFrame,
    ) -> VECValidationReport:
        """
        对单个因子进行VEC回测验证

        简化版VEC回测（仅验证因子有效性，非完整策略）
        """
        report = VECValidationReport(factor_name=factor_name)

        try:
            # 生成调仓日
            rebalance_dates = self._generate_rebalance_schedule(factor_df.index)

            if len(rebalance_dates) < 10:
                logger.warning("%s: 调仓日不足10个，跳过VEC验证", factor_name)
                return report

            # 简化回测：只在调仓日，根据因子排名选Top2
            portfolio_values = []
            current_value = self.initial_capital
            holdings = []  # 当前持仓
            hold_days = {}  # 持仓天数计数

            for i, date in enumerate(rebalance_dates):
                if date not in factor_df.index:
                    continue

                # 获取当日因子值
                factor_values = factor_df.loc[date]

                # 排名（降序，高分=好）
                rankings = factor_values.rank(ascending=False, na_option="bottom")

                # 选Top pos_size
                top_symbols = rankings.nsmallest(self.pos_size).index.tolist()

                # 简化的持仓逻辑（无迟滞，仅验证因子区分度）
                if i == 0:
                    # 初始化持仓
                    holdings = top_symbols
                    hold_days = {s: 0 for s in holdings}
                else:
                    # 简化：直接切换到新的Top2（实际生产有迟滞，但这里只看因子本身）
                    holdings = top_symbols
                    hold_days = {s: 0 for s in holdings}

                # 计算到下一个调仓日的收益
                if i < len(rebalance_dates) - 1:
                    next_date = rebalance_dates[i + 1]

                    # 使用T+1开盘价（简化：用close替代）
                    period_return = 0
                    for symbol in holdings:
                        if symbol in self.close.columns:
                            try:
                                buy_price = self.close.loc[date, symbol]
                                sell_price = self.close.loc[next_date, symbol]
                                if (
                                    pd.notna(buy_price)
                                    and pd.notna(sell_price)
                                    and buy_price > 0
                                ):
                                    symbol_return = (sell_price / buy_price - 1) * (
                                        1 - self.commission
                                    )
                                    period_return += symbol_return / len(holdings)
                            except KeyError:
                                pass

                    current_value *= 1 + period_return
                    portfolio_values.append(current_value)

                    # 更新持仓天数
                    hold_days = {s: d + self.freq for s, d in hold_days.items()}

            # 计算绩效指标
            if len(portfolio_values) > 1:
                values = np.array(portfolio_values)
                returns = np.diff(values) / values[:-1]

                report.total_return = (values[-1] / self.initial_capital - 1) * 100

                # 年化收益（假设252交易日/年）
                n_periods = len(returns)
                periods_per_year = 252 / self.freq
                report.annual_return = (
                    (
                        (values[-1] / self.initial_capital)
                        ** (periods_per_year / n_periods)
                        - 1
                    )
                    * 100
                    if n_periods > 0
                    else 0
                )

                # 波动率
                report.volatility = (
                    np.std(returns) * np.sqrt(periods_per_year) * 100
                    if len(returns) > 1
                    else 0
                )

                # 夏普（简化：假设无风险利率0）
                report.sharpe_ratio = (
                    (report.annual_return / report.volatility)
                    if report.volatility > 0
                    else 0
                )

                # 最大回撤
                peak = np.maximum.accumulate(values)
                drawdown = (peak - values) / peak
                report.max_drawdown = np.max(drawdown) * 100

                # 交易次数估算
                report.n_trades = len(rebalance_dates) * self.pos_size

                # 换手率估算
                report.turnover = report.n_trades / (
                    len(rebalance_dates) / periods_per_year
                )

                # Calmar
                report.calmar_ratio = (
                    report.annual_return / report.max_drawdown
                    if report.max_drawdown > 0
                    else 0
                )

                # 相对基准（等权43 ETF）
                benchmark_return = self._calculate_benchmark_return(rebalance_dates)
                report.vs_benchmark = report.total_return - benchmark_return

                # 综合评分（类似WFO评分）
                report.vec_score = (
                    report.annual_return * 0.4
                    + report.sharpe_ratio * 30 * 0.3  # 夏普通常<2，放大
                    + (100 - report.max_drawdown) * 0.3
                ) / 100  # 归一化到0-1

                # 通过标准（可调整）
                report.passed = (
                    report.total_return > 30  # 总收益>30%
                    and report.sharpe_ratio > 0.3  # 夏普>0.3
                    and report.max_drawdown < 50  # 回撤<50%
                )

        except Exception as e:
            logger.error("%s: VEC验证失败 - %s", factor_name, str(e))

        return report

    def batch_validate(
        self,
        entries: List[Any],  # List[FactorEntry]
        factors_dict: Dict[str, pd.DataFrame],
        top_n: int = 50,
    ) -> Tuple[List[Any], List[VECValidationReport]]:
        """
        批量验证因子

        参数:
            entries: FactorEntry列表
            factors_dict: {factor_name: DataFrame}
            top_n: 只验证VEC评分最高的前N个

        返回:
            (通过VEC验证的entries, 所有验证报告)
        """
        logger.info("批量VEC验证: %d 个候选因子", len(entries))

        reports = []
        for entry in entries:
            if entry.name in factors_dict:
                report = self.validate_single(entry.name, factors_dict[entry.name])
                reports.append(report)

                # 更新entry的元数据
                entry.metadata["vec_total_return"] = report.total_return
                entry.metadata["vec_sharpe"] = report.sharpe_ratio
                entry.metadata["vec_max_dd"] = report.max_drawdown
                entry.metadata["vec_score"] = report.vec_score
                entry.metadata["vec_passed"] = report.passed
            else:
                logger.warning("Factor %s not found in factors_dict", entry.name)

        # 按VEC评分排序
        reports_sorted = sorted(reports, key=lambda r: r.vec_score, reverse=True)

        # 只保留top_n
        top_reports = reports_sorted[:top_n]
        passed_names = {r.factor_name for r in top_reports if r.passed}

        passed_entries = [e for e in entries if e.name in passed_names]

        logger.info("VEC验证完成: %d 通过 / %d 候选", len(passed_entries), len(entries))

        return passed_entries, reports

    def _generate_rebalance_schedule(
        self, index: pd.DatetimeIndex
    ) -> List[pd.Timestamp]:
        """生成调仓日序列"""
        return index[:: self.freq].tolist()

    def _calculate_benchmark_return(self, rebalance_dates: List[pd.Timestamp]) -> float:
        """计算等权基准收益"""
        if len(rebalance_dates) < 2:
            return 0.0

        initial_value = self.initial_capital
        current_value = initial_value

        for i in range(len(rebalance_dates) - 1):
            date = rebalance_dates[i]
            next_date = rebalance_dates[i + 1]

            # 等权持有所有ETF
            period_return = 0
            valid_count = 0

            for symbol in self.close.columns:
                if symbol in self.close.columns:
                    try:
                        buy_price = self.close.loc[date, symbol]
                        sell_price = self.close.loc[next_date, symbol]
                        if (
                            pd.notna(buy_price)
                            and pd.notna(sell_price)
                            and buy_price > 0
                        ):
                            symbol_return = sell_price / buy_price - 1
                            period_return += symbol_return
                            valid_count += 1
                    except KeyError:
                        pass

            if valid_count > 0:
                current_value *= 1 + period_return / valid_count

        return (current_value / initial_value - 1) * 100
