"""
E2E 集成测试（简化版）

验证核心流程：
1. 数据加载 → 特征构建 → 模型预测
2. 配置解析与路径处理
3. 输出文件完整性
"""
import json
from pathlib import Path
import pandas as pd
import pytest


class TestE2EWorkflow:
    """端到端工作流测试（轻量级）"""
    
    def test_wfo_data_pipeline(self):
        """测试 WFO 数据管道完整性"""
        # 检查最新运行结果
        results_dir = Path(__file__).parent.parent / 'results'
        if not results_dir.exists():
            pytest.skip("无 WFO 运行结果")
        
        runs = sorted(results_dir.glob('run_*'), key=lambda x: x.name, reverse=True)
        if not runs:
            pytest.skip("无可用运行结果")
        
        latest_run = runs[0]
        
        # 验证核心输出文件
        assert (latest_run / 'all_combos.parquet').exists(), "缺失 all_combos.parquet"
        
        # 加载验证数据结构
        df = pd.read_parquet(latest_run / 'all_combos.parquet')
        
        # 核心字段必须存在
        required_cols = ['combo', 'mean_oos_ic', 'oos_ic_std', 'best_rebalance_freq']
        for col in required_cols:
            assert col in df.columns, f"缺失字段: {col}"
        
        # 数据合理性检查
        assert len(df) > 0, "空结果"
        assert df['mean_oos_ic'].notna().all(), "IC 包含空值"
    
    def test_ml_ranking_outputs(self):
        """测试 ML 排序输出完整性"""
        results_dir = Path(__file__).parent.parent / 'results'
        if not results_dir.exists():
            pytest.skip("无 WFO 运行结果")
        
        runs = sorted(results_dir.glob('run_*'), key=lambda x: x.name, reverse=True)
        ml_runs = [r for r in runs if (r / 'ranked_combos.csv').exists()]
        
        if not ml_runs:
            pytest.skip("无 ML 排序结果")
        
        latest_ml = ml_runs[0]
        df_ranked = pd.read_csv(latest_ml / 'ranked_combos.csv')
        
        # 验证 ML 排序字段
        assert 'ltr_score' in df_ranked.columns, "缺失 ltr_score"
        assert 'ltr_rank' in df_ranked.columns, "缺失 ltr_rank"
        assert 'rank_change' in df_ranked.columns, "缺失 rank_change"
        
        # 排名合理性
        assert df_ranked['ltr_rank'].min() == 1, "排名未从1开始"
        assert df_ranked['ltr_rank'].is_monotonic_increasing, "排名不连续"
    
    def test_config_parsing(self):
        """测试配置文件解析"""
        config_path = Path(__file__).parent.parent / 'configs' / 'combo_wfo_config.yaml'
        
        if not config_path.exists():
            pytest.skip("配置文件不存在")
        
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        
        # 核心配置项
        assert 'data' in cfg, "缺失 data"
        assert 'symbols' in cfg['data'], "缺失 symbols"
        assert 'cache_dir' in cfg['data'], "缺失 cache_dir"
        
        # 路径合理性
        cache_dir = cfg['data']['cache_dir']
        assert not Path(cache_dir).is_absolute() or Path(cache_dir).exists(), \
            "cache_dir 配置不合理"

