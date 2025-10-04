"""Post-backtest review helpers."""

from __future__ import annotations

import pandas as pd
import vectorbt as vbt

from hk_midfreq.backtest_engine import BacktestArtifacts


def portfolio_statistics(portfolio: vbt.Portfolio) -> pd.Series:
    """Return a core statistics series from the vectorbt portfolio."""

    return portfolio.stats()


def trade_overview(portfolio: vbt.Portfolio, limit: int = 10) -> pd.DataFrame:
    """Return the top ``limit`` trades in a readable format."""

    return portfolio.trades.records_readable.head(limit)


def compile_review(artifacts: BacktestArtifacts, limit: int = 10) -> dict:
    """Collect statistics and trade summaries for downstream reporting."""

    stats = portfolio_statistics(artifacts.portfolio)
    trades = trade_overview(artifacts.portfolio, limit=limit)
    return {"stats": stats, "trades": trades}


def print_review(artifacts: BacktestArtifacts, limit: int = 10) -> None:
    """Pretty-print the review bundle to stdout."""

    review = compile_review(artifacts, limit=limit)
    print(review["stats"])  # noqa: T201 - human-facing summary output
    print("=" * 60)
    print(review["trades"])  # noqa: T201


__all__ = ["portfolio_statistics", "trade_overview", "compile_review", "print_review"]
