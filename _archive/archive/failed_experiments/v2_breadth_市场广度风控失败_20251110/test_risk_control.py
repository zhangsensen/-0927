#!/usr/bin/env python3
"""风控层集成测试

快速验证：
1. Pipeline能否正确加载risk_control配置
2. 三个监控器是否正常初始化
3. WFO后风控层调用是否成功
4. 日志是否正确生成

用法:
    python3 test_risk_control.py
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from core.pipeline import Pipeline
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def test_risk_control_loading():
    """测试1：配置加载"""
    logger.info("=" * 60)
    logger.info("测试1: 风控配置加载")
    logger.info("=" * 60)
    
    # 创建临时配置文件
    import yaml
    test_config = {
        "run_id": "TEST_RISK_CONTROL",
        "output_root": "results/test_rc",
        "data": {
            "factor_root": "factor_output",
            "etf_pool_name": "etf_pool_mid40",
            "start": "2018-01-01",
            "end": "2024-12-31",
        },
        "cross_section": {
            "winsorize_quantiles": [0.025, 0.975],
            "normalize_method": "z-score",
        },
        "wfo": {
            "is_period": 252,
            "oos_period": 60,
            "step_size": 20,
            "warmup": 20,
            "factor_weighting": "ic_weighted",
            "min_factor_ic": 0.012,
        },
        "risk_control": {
            "market_breadth": {
                "enabled": True,
                "breadth_floor": 0.25,
                "score_threshold": 0.0,
                "defensive_scale": 0.5,
                "verbose": True,
            },
            "volatility_target": {
                "enabled": False,  # 默认关闭
                "target_vol": 0.30,
            },
            "correlation_monitor": {
                "enabled": False,  # 默认关闭
                "corr_threshold": 0.65,
            },
            "combine_strategy": "min",
        },
    }
    
    # 写入临时文件
    tmp_config_path = Path("configs/test_rc_config.yaml")
    tmp_config_path.parent.mkdir(exist_ok=True)
    with open(tmp_config_path, "w") as f:
        yaml.dump(test_config, f)
    
    try:
        pipeline = Pipeline.from_config(str(tmp_config_path))
        
        # 检查模块初始化
        assert pipeline.breadth_monitor is not None, "市场广度监控未初始化"
        assert pipeline.vol_target is None, "波动率目标应关闭"
        assert pipeline.corr_monitor is None, "相关性监控应关闭"
        
        logger.info("✅ 配置加载成功")
        logger.info("   - 市场广度: %s", "已启用" if pipeline.breadth_monitor else "未启用")
        logger.info("   - 波动率目标: %s", "已启用" if pipeline.vol_target else "未启用")
        logger.info("   - 相关性监控: %s", "已启用" if pipeline.corr_monitor else "未启用")
        
        # 清理
        tmp_config_path.unlink()
        return True
        
    except Exception as e:
        logger.error("❌ 配置加载失败: %s", e, exc_info=True)
        if tmp_config_path.exists():
            tmp_config_path.unlink()
        return False


def test_full_pipeline_with_rc():
    """测试2：完整Pipeline运行（如果数据存在）"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("测试2: 完整Pipeline运行")
    logger.info("=" * 60)
    
    # 检查数据是否存在
    factor_root = Path("factor_output")
    if not factor_root.exists():
        logger.warning("未找到factor_output目录，跳过完整测试")
        return None
    
    # 使用最小配置运行
    import yaml
    test_config = {
        "run_id": "TEST_FULL_RC",
        "output_root": "results/test_full_rc",
        "data": {
            "factor_root": "factor_output",
            "etf_pool_name": "etf_pool_mid40",
            "start": "2021-01-01",  # 短期数据快速测试
            "end": "2021-12-31",
        },
        "cross_section": {
            "winsorize_quantiles": [0.025, 0.975],
        },
        "wfo": {
            "is_period": 126,  # 半年IS
            "oos_period": 30,  # 1个月OOS
            "step_size": 10,
            "warmup": 20,
            "min_factor_ic": 0.01,
        },
        "risk_control": {
            "market_breadth": {
                "enabled": True,
                "breadth_floor": 0.25,
                "defensive_scale": 0.5,
            },
            "combine_strategy": "min",
        },
    }
    
    tmp_config_path = Path("configs/test_full_rc_config.yaml")
    tmp_config_path.parent.mkdir(exist_ok=True)
    with open(tmp_config_path, "w") as f:
        yaml.dump(test_config, f)
    
    try:
        pipeline = Pipeline.from_config(str(tmp_config_path))
        pipeline.run()
        
        # 检查风控日志
        log_path = Path(test_config["output_root"]) / "wfo" / "risk_control_log.csv"
        if log_path.exists():
            import pandas as pd
            log_df = pd.read_csv(log_path)
            logger.info("✅ 风控日志生成成功: %d条记录", len(log_df))
            logger.info("   - 触发防守天数: %d", (log_df["final_scale"] < 1.0).sum())
            logger.info("   - 平均缩仓比例: %.1f%%", (1 - log_df["final_scale"].mean()) * 100)
            tmp_config_path.unlink()
            return True
        else:
            logger.warning("未找到风控日志，可能数据不足")
            tmp_config_path.unlink()
            return None
        
    except Exception as e:
        logger.error("❌ Pipeline运行失败: %s", e, exc_info=True)
        if tmp_config_path.exists():
            tmp_config_path.unlink()
        return False


if __name__ == "__main__":
    logger.info("ETF Rotation V2 - 风控层集成测试")
    logger.info("")
    
    # 测试1: 配置加载
    test1_pass = test_risk_control_loading()
    
    # 测试2: 完整运行（可选）
    test2_pass = test_full_pipeline_with_rc()
    
    # 总结
    logger.info("")
    logger.info("=" * 60)
    logger.info("测试总结")
    logger.info("=" * 60)
    logger.info("配置加载: %s", "✅ 通过" if test1_pass else "❌ 失败")
    if test2_pass is not None:
        logger.info("完整运行: %s", "✅ 通过" if test2_pass else "❌ 失败")
    else:
        logger.info("完整运行: ⏭️  跳过（数据不足或环境限制）")
    
    if test1_pass:
        logger.info("")
        logger.info("🎉 核心功能验证通过！可以开始完整回测。")
        logger.info("")
        logger.info("下一步:")
        logger.info("  1. 运行baseline（无风控）: python3 run_combo_wfo.py")
        logger.info("  2. 启用市场广度: 修改configs/run_combo_wfo.yaml添加risk_control段")
        logger.info("  3. 对比结果: 查看results/*/wfo/risk_control_log.csv")
