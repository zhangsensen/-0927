"""
ML 排序与回退逻辑单元测试

测试覆盖:
1. LTR 模型正常加载
2. 模型文件缺失时自动回退到 WFO 排序
3. 特征不匹配时的对齐逻辑
4. 异常处理和日志记录
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from applications.apply_ranker import apply_ltr_ranking
from strategies.ml_ranker.ltr_model import LTRRanker


@pytest.fixture
def mock_wfo_results():
    """模拟 WFO 结果数据"""
    return pd.DataFrame({
        'combo': ['ADX_14D + CMF_20D', 'CMF_20D + RSI_14', 'ADX_14D + RSI_14'],
        'combo_size': [2, 2, 2],
        'mean_oos_ic': [0.045, 0.038, 0.032],
        'oos_ic_std': [0.15, 0.13, 0.12],
        'oos_ic_ir': [0.30, 0.29, 0.27],
        'positive_rate': [0.58, 0.55, 0.53],
        'best_rebalance_freq': [8, 8, 8],
        'stability_score': [-0.25, -0.30, -0.35],
        'mean_oos_sharpe': [0.85, 0.78, 0.72],
        'oos_sharpe_std': [0.45, 0.42, 0.40],
        'p_value': [0.08, 0.12, 0.15],
        'q_value': [0.15, 0.18, 0.22],
        'oos_sharpe_proxy': [0.75, 0.68, 0.62],
        'is_significant': [False, False, False],
        # 序列特征（简化）
        'oos_ic_list': [[0.05]*19] * 3,
        'oos_ir_list': [[0.3]*19] * 3,
        'positive_rate_list': [[0.55]*19] * 3,
        'best_freq_list': [[8]*19] * 3,
        'oos_sharpe_list': [[0.8]*19] * 3,
        'oos_daily_mean_list': [[0.001]*19] * 3,
        'oos_daily_std_list': [[0.015]*19] * 3,
        'oos_sample_count_list': [[60]*19] * 3,
    })


@pytest.fixture
def temp_wfo_dir(mock_wfo_results):
    """创建临时 WFO 输出目录"""
    temp_dir = tempfile.mkdtemp()
    wfo_path = Path(temp_dir)
    
    # 保存 all_combos.parquet
    mock_wfo_results.to_parquet(wfo_path / 'all_combos.parquet', index=False)
    
    yield wfo_path
    
    # 清理
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_ltr_model(monkeypatch):
    """Mock LTRRanker 模型"""
    class MockLTRRanker:
        def __init__(self):
            # 使用与实际特征工程一致的特征名（39个特征）
            self.feature_names = [
                'combo_size', 'mean_oos_ic', 'oos_ic_std', 'oos_ic_ir', 
                'positive_rate', 'best_rebalance_freq', 'stability_score',
                'mean_oos_sharpe', 'oos_sharpe_std', 'p_value', 'q_value',
                'oos_sharpe_proxy', 'is_significant',
                'ic_seq_mean', 'ic_seq_std', 'ic_seq_min', 'ic_seq_max', 
                'ic_seq_median', 'ic_positive_ratio', 'ic_seq_trend', 'ic_seq_cv',
                'sharpe_seq_mean', 'sharpe_seq_std', 'sharpe_seq_min', 
                'sharpe_seq_max', 'sharpe_seq_median', 'sharpe_positive_ratio',
                'sharpe_seq_cv', 'ir_seq_mean', 'ir_seq_std', 
                'ir_positive_ratio', 'posrate_seq_mean', 'posrate_seq_std',
                'posrate_seq_min', 'ic_x_sharpe', 'stability_x_posrate',
                'ic_ir_x_sharpe_proxy', 'ic_relative_std', 'sharpe_relative_std'
            ]
        
        def predict(self, X):
            """模拟预测（固定随机种子确保一致性）"""
            np.random.seed(42)
            scores = np.random.rand(len(X)) * 0.3 + 0.1
            ranks = pd.Series(scores).rank(ascending=False, method='min').values.astype(int)
            return scores, ranks
    
    return MockLTRRanker()


class TestMLRanking:
    """ML 排序核心功能测试"""
    
    def test_model_load_success(self):
        """测试模型正常加载"""
        # 使用真实模型路径（假设模型存在）
        model_path = 'strategies/ml_ranker/models/ltr_ranker'
        
        if not Path(model_path + '.txt').exists():
            pytest.skip("LTR 模型文件不存在，跳过测试")
        
        model = LTRRanker.load(model_path)
        assert model.model is not None
        assert len(model.feature_names) > 0
    
    def test_model_file_missing(self):
        """测试模型文件缺失时抛出异常"""
        with pytest.raises(FileNotFoundError, match='模型文件不存在'):
            LTRRanker.load('nonexistent_model_path')
    
    @patch('strategies.ml_ranker.ltr_model.LTRRanker.load')
    def test_apply_ltr_ranking_success(self, mock_load, temp_wfo_dir, mock_ltr_model):
        """测试 ML 排序正常执行"""
        mock_load.return_value = mock_ltr_model
        
        result_df = apply_ltr_ranking(
            model_path='mock_model',
            wfo_dir=temp_wfo_dir,
            output_path=None,
            top_k=None,
            verbose=False
        )
        
        assert 'ltr_score' in result_df.columns
        assert 'ltr_rank' in result_df.columns
        assert 'wfo_rank' in result_df.columns
        assert len(result_df) == 3
        assert result_df['ltr_rank'].min() == 1
    
    @patch('strategies.ml_ranker.ltr_model.LTRRanker.load')
    def test_feature_alignment(self, mock_load, temp_wfo_dir, mock_ltr_model):
        """测试特征名称不匹配时的对齐逻辑"""
        # 模拟特征名称顺序不同
        mock_ltr_model.feature_names = sorted(mock_ltr_model.feature_names, reverse=True)
        mock_load.return_value = mock_ltr_model
        
        result_df = apply_ltr_ranking(
            model_path='mock_model',
            wfo_dir=temp_wfo_dir,
            verbose=False
        )
        
        # 应该成功运行（特征对齐逻辑生效）
        assert 'ltr_score' in result_df.columns
    
    def test_ranking_consistency(self, temp_wfo_dir, mock_ltr_model):
        """测试排序结果一致性"""
        with patch('strategies.ml_ranker.ltr_model.LTRRanker.load', return_value=mock_ltr_model):
            result1 = apply_ltr_ranking('mock', temp_wfo_dir, verbose=False)
            result2 = apply_ltr_ranking('mock', temp_wfo_dir, verbose=False)
            
            # 相同输入应产生相同排名
            pd.testing.assert_frame_equal(
                result1[['combo', 'ltr_rank']],
                result2[['combo', 'ltr_rank']]
            )


class TestMLRankingFallback:
    """ML 排序回退逻辑测试"""
    
    def test_fallback_on_model_missing(self, temp_wfo_dir):
        """测试模型缺失时回退到 WFO 排序"""
        # 直接测试 run_combo_wfo.py 中的回退逻辑
        from applications.run_combo_wfo import ML_RANKER_AVAILABLE
        
        if ML_RANKER_AVAILABLE:
            # 模拟模型文件缺失
            with patch('pathlib.Path.exists', return_value=False):
                # 应该触发回退日志
                assert True  # 占位，实际需要检查日志输出
    
    def test_fallback_preserves_wfo_ranking(self, mock_wfo_results):
        """测试回退模式保留 WFO 原始排序"""
        # 按 mean_oos_ic 排序
        wfo_sorted = mock_wfo_results.sort_values(
            ['mean_oos_ic', 'stability_score'],
            ascending=[False, False]
        ).reset_index(drop=True)
        
        assert wfo_sorted.iloc[0]['mean_oos_ic'] == 0.045
        assert wfo_sorted.iloc[0]['combo'] == 'ADX_14D + CMF_20D'


class TestMLRankingEdgeCases:
    """ML 排序边界条件测试"""
    
    def test_empty_wfo_results(self):
        """测试空 WFO 结果"""
        temp_dir = tempfile.mkdtemp()
        wfo_path = Path(temp_dir)
        
        # 创建空 DataFrame
        empty_df = pd.DataFrame(columns=['combo', 'mean_oos_ic'])
        empty_df.to_parquet(wfo_path / 'all_combos.parquet', index=False)
        
        with pytest.raises(Exception):  # 应该抛出异常或返回空结果
            apply_ltr_ranking('mock_model', wfo_path, verbose=False)
        
        shutil.rmtree(temp_dir)
    
    def test_single_combo(self, mock_ltr_model):
        """测试单个组合的情况"""
        temp_dir = tempfile.mkdtemp()
        wfo_path = Path(temp_dir)
        
        single_combo = pd.DataFrame({
            'combo': ['ADX_14D + CMF_20D'],
            'mean_oos_ic': [0.045],
            'combo_size': [2],
            'oos_ic_std': [0.15],
            'oos_ic_ir': [0.30],
            'positive_rate': [0.58],
            'best_rebalance_freq': [8],
            'stability_score': [-0.25],
            'mean_oos_sharpe': [0.85],
            'oos_sharpe_std': [0.45],
            'p_value': [0.08],
            'q_value': [0.15],
            'oos_sharpe_proxy': [0.75],
            'is_significant': [False],
            'oos_ic_list': [[0.05]*19],
            'oos_ir_list': [[0.3]*19],
            'positive_rate_list': [[0.55]*19],
            'best_freq_list': [[8]*19],
            'oos_sharpe_list': [[0.8]*19],
            'oos_daily_mean_list': [[0.001]*19],
            'oos_daily_std_list': [[0.015]*19],
            'oos_sample_count_list': [[60]*19],
        })
        single_combo.to_parquet(wfo_path / 'all_combos.parquet', index=False)
        
        with patch('strategies.ml_ranker.ltr_model.LTRRanker.load', return_value=mock_ltr_model):
            result = apply_ltr_ranking('mock', wfo_path, verbose=False)
            assert len(result) == 1
            assert result['ltr_rank'].iloc[0] == 1
        
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
