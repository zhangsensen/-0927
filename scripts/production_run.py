#!/usr/bin/env python3
"""
生产环境因子计算 - 完整版
1. 读取配置文件
2. 加载价格+资金流数据
3. 计算150+技术指标 + 11个资金流因子
4. 完整日志和结果审查
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from factor_system.factor_engine.core.engine import FactorEngine
from factor_system.factor_engine.core.registry import get_global_registry
from factor_system.factor_engine.providers.parquet_provider import ParquetDataProvider
from factor_system.factor_engine.providers.combined_provider import CombinedMoneyFlowProvider


def setup_logging(log_dir: Path):
    """配置日志系统"""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"production_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """加载配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_data_paths(config: dict, project_root: Path, logger):
    """验证数据路径"""
    logger.info("=" * 70)
    logger.info("📂 数据路径验证")
    logger.info("=" * 70)

    # 价格数据
    price_dir = project_root / config["data_paths"]["price_data_dir"]
    logger.info(f"价格数据目录: {price_dir}")
    logger.info(f"  存在: {price_dir.exists()}")
    if price_dir.exists():
        files = list(price_dir.glob("*.parquet"))
        logger.info(f"  文件数: {len(files)}")
        logger.info(f"  样本: {[f.name for f in files[:3]]}")

    # 资金流数据
    money_flow_dir = project_root / config["data_paths"]["money_flow_dir"]
    logger.info(f"资金流数据目录: {money_flow_dir}")
    logger.info(f"  存在: {money_flow_dir.exists()}")
    if money_flow_dir.exists():
        files = list(money_flow_dir.glob("*.parquet"))
        logger.info(f"  文件数: {len(files)}")
        logger.info(f"  样本: {[f.name for f in files[:3]]}")


def get_available_symbols(price_dir: Path, money_flow_dir: Path, logger) -> list:
    """获取有价格和资金流数据的股票"""
    price_symbols = {f.stem for f in price_dir.glob("*.parquet")}
    mf_symbols = {
        f.stem.replace("_moneyflow", "").replace("_money_flow", "")
        for f in money_flow_dir.glob("*.parquet")
    }

    # 交集
    common_symbols = price_symbols & mf_symbols
    logger.info(f"价格数据: {len(price_symbols)} 个股票")
    logger.info(f"资金流数据: {len(mf_symbols)} 个股票")
    logger.info(f"交集: {len(common_symbols)} 个股票")

    return sorted(list(common_symbols))[:5]  # 先取5个测试


def main():
    # 1. 加载配置
    config_path = project_root / "factor_system/config/money_flow_config.yaml"
    config = load_config(config_path)

    # 2. 设置日志
    log_dir = project_root / config["data_paths"]["log_dir"]
    logger = setup_logging(log_dir)

    logger.info("=" * 70)
    logger.info("🚀 生产环境 - 因子计算启动")
    logger.info("=" * 70)
    logger.info(f"配置文件: {config_path}")

    # 3. 验证数据路径
    validate_data_paths(config, project_root, logger)

    # 4. 获取可用股票
    price_dir = project_root / config["data_paths"]["price_data_dir"]
    money_flow_dir = project_root / config["data_paths"]["money_flow_dir"]
    symbols = get_available_symbols(price_dir, money_flow_dir, logger)

    if not symbols:
        logger.error("❌ 没有找到同时有价格和资金流数据的股票")
        return None

    logger.info(f"✅ 选定股票: {symbols}")

    # 5. 初始化注册表（包含150+技术指标 + 11个资金流因子）
    logger.info("\n" + "=" * 70)
    logger.info("📋 初始化因子注册表")
    logger.info("=" * 70)
    registry = get_global_registry(include_money_flow=True)
    logger.info(f"✅ 已注册 {len(registry.metadata)} 个因子")

    # 列出所有因子
    logger.info("\n因子分类:")
    categories = {}
    for factor_id, meta in registry.metadata.items():
        cat = meta.get("category", "unknown")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(factor_id)

    for cat, factors in sorted(categories.items()):
        logger.info(f"  {cat}: {len(factors)} 个")

    # 6. 初始化数据提供者（使用CombinedMoneyFlowProvider）
    logger.info("\n" + "=" * 70)
    logger.info("📊 初始化数据提供者")
    logger.info("=" * 70)

    price_provider = ParquetDataProvider(raw_data_dir=project_root / "raw")
    logger.info(f"✅ 价格数据提供者: {price_dir}")

    # 使用组合提供者（自动合并价格+资金流）
    combined_provider = CombinedMoneyFlowProvider(
        price_provider=price_provider,
        money_flow_dir=money_flow_dir,
        enforce_t_plus_1=config["time_config"]["enforce_t_plus_1"],
    )
    logger.info(f"✅ 组合数据提供者: 价格 + 资金流")
    logger.info(f"   资金流目录: {money_flow_dir}")
    logger.info(f"   T+1时序安全: {config['time_config']['enforce_t_plus_1']}")

    # 7. 创建FactorEngine
    logger.info("\n" + "=" * 70)
    logger.info("🔧 创建FactorEngine")
    logger.info("=" * 70)

    engine = FactorEngine(
        data_provider=combined_provider,
        registry=registry,
    )
    logger.info("✅ FactorEngine已就绪")

    # 8. 选择因子集（从YAML配置加载）
    time_config = config["time_config"]
    start_date = datetime.strptime(time_config["start_date"], "%Y-%m-%d")
    end_date = datetime.strptime(time_config["end_date"], "%Y-%m-%d")
    # 统一timeframe，支持列表配置
    def normalize_timeframe(value: str) -> str:
        tf = str(value).strip().lower()
        return "daily" if tf in ("1day", "daily") else tf

    cfg_timeframes = time_config.get("timeframes")
    if cfg_timeframes:
        timeframes = [normalize_timeframe(tf) for tf in cfg_timeframes]
    else:
        timeframes = [normalize_timeframe(time_config.get("timeframe", "daily"))]

    # 从配置或命令行参数获取因子集名称
    import sys
    # 默认跑全量因子集
    factor_set_name = "all"  # 默认集合：all -> 注册表中的全部因子
    
    # 支持命令行参数 --set
    if "--set" in sys.argv:
        idx = sys.argv.index("--set")
        if idx + 1 < len(sys.argv):
            factor_set_name = sys.argv[idx + 1]
    
    logger.info(f"使用因子集: {factor_set_name}")
    
    # 从注册表解析因子集
    try:
        factor_ids = registry.get_factor_ids_by_set(factor_set_name)
        logger.info(f"✅ 因子集解析成功: {len(factor_ids)} 个因子")
        logger.info(f"前20个因子: {factor_ids[:20]}")
        if len(factor_ids) > 20:
            logger.info(f"... 还有 {len(factor_ids) - 20} 个因子")
    except Exception as e:
        # 兜底：直接使用注册表中全部可用因子（factors + metadata）
        logger.warning(f"⚠️ 因子集 '{factor_set_name}' 加载失败，改为全量因子: {e}")
        all_from_registry = sorted(set(list(registry.factors.keys()) + list(registry.metadata.keys())))
        factor_ids = all_from_registry
        logger.info(f"✅ 全量因子解析成功: {len(factor_ids)} 个因子")
        logger.info(f"前20个因子: {factor_ids[:20]}")

    logger.info("\n" + "=" * 70)
    logger.info("🎯 全局配置")
    logger.info("=" * 70)
    logger.info(f"股票: {symbols}")
    logger.info(f"时间: {start_date.date()} ~ {end_date.date()}")
    logger.info(f"时间框架: {timeframes}")
    logger.info(f"因子集: {factor_set_name} ({len(factor_ids)} 个)")

    # 9. 循环计算多个时间框架
    import time
    qc_cfg = config.get("quality_control", {})
    
    for timeframe in timeframes:
        logger.info("\n" + "=" * 70)
        logger.info(f"⚙️ 时间框架: {timeframe}")
        logger.info("=" * 70)

        try:
            start_time = time.time()

            # 非日线周期自动过滤资金流类因子
            current_factor_ids = list(factor_ids)
            if timeframe != "daily":
                categories = {fid: registry.metadata.get(fid, {}).get("category", "") for fid in factor_ids}
                current_factor_ids = [
                    fid for fid in factor_ids
                    if not categories.get(fid, "").startswith(("money_flow", "money_flow_enhanced"))
                    and not fid.lower().startswith(("moneyflow", "money_flow"))
                ]
                logger.info(f"⛳ 过滤资金流因子: {len(factor_ids)} → {len(current_factor_ids)}")

            logger.info(f"因子数: {len(current_factor_ids)}")

            # 计算（1min需要更大内存限制）
            max_ram_mb = 8192 if timeframe == "1min" else 2048
            result = engine.calculate_factors(
                factor_ids=current_factor_ids,
                symbols=symbols,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                use_cache=True,
                max_ram_mb=max_ram_mb,
            )

            elapsed_time = time.time() - start_time

            if result.empty:
                logger.warning(f"⚠️ {timeframe} 结果为空，跳过")
                continue

            logger.info(f"✅ 数据形状: {result.shape}")
            logger.info(f"⏱️ 耗时: {elapsed_time:.2f}秒")

            # 按股票独立保存
            output_dir = project_root / config["data_paths"]["output_root"] / "production" / timeframe
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 拆分每个股票独立存储
            for symbol in result.index.get_level_values('symbol').unique():
                symbol_data = result.xs(symbol, level='symbol')
                output_file = output_dir / f"{symbol}_{timeframe}_{timestamp}.parquet"
                symbol_data.to_parquet(output_file, compression="snappy", index=True)
                file_size = output_file.stat().st_size / 1024 / 1024
                logger.info(f"💾 {symbol}: {output_file.name} ({file_size:.2f} MB)")

            # 生成报告（含每日K线数校验）
            report_file = output_dir / f"report_{timestamp}.md"
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(f"# 因子计算报告 - {timeframe}\n\n")
                f.write(f"生成时间: {datetime.now()}\n\n")
                f.write(f"## 配置\n\n")
                f.write(f"- 股票: {symbols}\n")
                f.write(f"- 时间: {start_date.date()} ~ {end_date.date()}\n")
                f.write(f"- 周期: {timeframe}\n")
                f.write(f"- 因子数: {len(current_factor_ids)}\n\n")
                f.write(f"## 结果\n\n")
                f.write(f"- 数据形状: {result.shape}\n")
                
                # A股分钟级数据：校验每日K线数
                if timeframe != "daily" and any(s.endswith(('.SH', '.SZ')) for s in symbols):
                    expected_bars = {
                        '1min': 241, '5min': 48, '15min': 16, '30min': 8, '60min': 4,
                        '120min': 2, '240min': 1
                    }
                    expected = expected_bars.get(timeframe, None)
                    
                    if expected:
                        f.write(f"\n## 数据质量校验（A股会话感知）\n\n")
                        f.write(f"| 股票 | 总行数 | 每日K线数 | 期望值 | 状态 |\n")
                        f.write(f"|------|--------|----------|--------|------|\n")
                        
                        for symbol in symbols:
                            if not symbol.endswith(('.SH', '.SZ')):
                                continue
                            symbol_data = result.xs(symbol, level='symbol')
                            daily_counts = symbol_data.groupby(symbol_data.index.date).size()
                            avg_bars = daily_counts.mean()
                            status = "✅" if abs(avg_bars - expected) < 0.1 else "❌"
                            f.write(f"| {symbol} | {len(symbol_data):,} | {avg_bars:.1f} | {expected} | {status} |\n")
                        
                        f.write(f"\n**说明**: A股交易时间 9:30-11:30, 13:00-15:00，会话感知重采样确保无跨午休K线。\n")
                f.write(f"- 计算耗时: {elapsed_time:.2f}秒\n")
                f.write(f"- 数据文件: {output_file.name}\n\n")
                f.write(f"## 数据质量\n\n")
                f.write(result.describe().to_markdown())
            logger.info(f"📄 报告: {report_file.name}")

        except Exception as e:
            logger.error(f"❌ {timeframe} 计算失败: {e}", exc_info=True)
            continue

    logger.info("\n" + "=" * 70)
    logger.info("✅ 所有时间框架计算完成")
    logger.info("=" * 70)
    return True


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result is not None else 1)
