#!/usr/bin/env python3
"""
ETF轮动系统 - 统一入口

替代 scripts/step*.py 的手动流程，提供配置驱动的自动化执行

工作流:
  横截面加工 -> 因子筛选 -> WFO验证 -> VBT回测

命令示例:
  python main.py run --config configs/default.yaml
  python main.py run-steps --config configs/default.yaml --steps cross_section factor_selection
  python main.py run-steps --config configs/default.yaml --steps wfo

作者: Linus Refactor
日期: 2025-10-28
"""

import sys
from pathlib import Path

import click

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.pipeline import Pipeline


@click.group()
def cli():
    """ETF轮动系统 - 统一命令行入口"""
    pass


@cli.command()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="配置文件路径 (YAML)",
)
def run(config: Path):
    """
    运行完整流程

    执行所有步骤: 横截面 -> 因子筛选 -> WFO -> 回测
    """
    click.echo(f"🚀 启动完整流程: {config}")
    pipeline = Pipeline.from_config(config)
    pipeline.run()


@cli.command()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="配置文件路径 (YAML)",
)
@click.option(
    "--steps",
    multiple=True,
    type=click.Choice(["cross_section", "factor_selection", "wfo", "backtest"]),
    required=True,
    help="要执行的步骤（可多选）",
)
def run_steps(config: Path, steps: tuple):
    """
    运行指定步骤

    支持单独运行某个或多个步骤，用于调试或断点续跑

    示例:
      python main.py run-steps --config configs/default.yaml --steps cross_section
      python main.py run-steps --config configs/default.yaml --steps wfo --steps backtest
    """
    click.echo(f"🚀 启动指定步骤: {', '.join(steps)}")
    pipeline = Pipeline.from_config(config)

    for step in steps:
        click.echo(f"\n▶️  执行步骤: {step}")
        pipeline.run_step(step)


@cli.command()
def version():
    """显示版本信息"""
    click.echo("ETF轮动系统优化版 v2.0")
    click.echo("重构日期: 2025-10-28")
    click.echo("架构: 横截面 -> 因子筛选 -> WFO -> VBT回测")


if __name__ == "__main__":
    cli()
