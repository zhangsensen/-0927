"""
WFO → 真实回测 校准器

问题定义：
    WFO输出 mean_oos_ic 与真实回测 Sharpe 相关性仅 0.07，导致排序失效。
    
解决方案：
    用回归模型学习映射关系：f(WFO特征) → 真实Sharpe
    
特征工程：
    - mean_oos_ic: WFO核心指标（当前唯一排序依据）
    - oos_ic_std: OOS窗口IC标准差（稳定性）
    - positive_rate: OOS窗口IC>0比例（鲁棒性）
    - stability_score: 综合稳定性得分
    - combo_size: 因子数量（复杂度惩罚）
    - factor_diversity: 因子类型多样性（避免过度集中）
    
模型选择：
    - Ridge回归：线性解释性强，防止过拟合
    - XGBoost：捕捉非线性关系，处理因子交互效应
    - Stacking: Ridge + XGBoost 集成
    
验证策略：
    - 5-Fold CV: 内部交叉验证
    - Out-of-Sample Test: 用已有Top2000回测结果作为测试集
    - 增量学习: 每次WFO后更新模型，累积历史数据
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
import joblib

logger = logging.getLogger(__name__)


class WFORealBacktestCalibrator:
    """
    WFO → 真实回测校准器
    
    核心思路：
        既然WFO的mean_oos_ic与真实Sharpe相关性差，那就用回归模型
        学习它们之间的映射关系。每次WFO后，用历史数据训练模型，
        然后用校准后的预测Sharpe重新排序。
    """
    
    def __init__(
        self,
        model_type: str = "ridge",  # "ridge", "gbdt", "stacking"
        alpha: float = 1.0,  # Ridge正则化强度
        n_estimators: int = 100,  # GBDT树数量
        max_depth: int = 3,  # GBDT最大深度（防止过拟合）
        cv_folds: int = 5,
    ):
        self.model_type = model_type
        self.alpha = alpha
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.cv_folds = cv_folds
        
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = [
            'mean_oos_ic',
            'oos_ic_std',
            'positive_rate',
            'stability_score',
            'combo_size',
        ]
        
        # 训练历史记录
        self.train_history = []
        
    def extract_features(self, wfo_df: pd.DataFrame) -> pd.DataFrame:
        """
        从WFO结果中提取特征
        
        Args:
            wfo_df: WFO输出的all_combos.parquet
        
        Returns:
            特征矩阵 DataFrame
        """
        features = wfo_df[self.feature_names].copy()
        
        # 处理缺失值（用中位数填充）
        for col in features.columns:
            if features[col].isna().any():
                features[col].fillna(features[col].median(), inplace=True)
        
        return features
    
    def fit(
        self,
        wfo_df: pd.DataFrame,
        backtest_df: pd.DataFrame,
        target_metric: str = 'sharpe',
    ) -> Dict[str, float]:
        """
        训练校准模型
        
        Args:
            wfo_df: WFO结果（必须包含 combo, mean_oos_ic, oos_ic_std 等字段）
            backtest_df: 真实回测结果（必须包含 combo, sharpe, annual_ret 等字段）
            target_metric: 目标指标（'sharpe' 或 'annual_ret'）
        
        Returns:
            训练评估指标字典
        """
        # 合并WFO和回测数据
        merged = wfo_df.merge(backtest_df[['combo', target_metric]], on='combo', how='inner')
        logger.info(f"训练集: {len(merged)} 组合 (WFO: {len(wfo_df)}, 回测: {len(backtest_df)})")
        
        if len(merged) < 100:
            raise ValueError(f"训练样本不足100个 ({len(merged)})，无法可靠训练模型")
        
        # 提取特征和目标
        X = self.extract_features(merged)
        y = merged[target_metric].values
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练模型
        if self.model_type == "ridge":
            self.model = Ridge(alpha=self.alpha, random_state=42)
        elif self.model_type == "gbdt":
            self.model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            )
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
        
        self.model.fit(X_scaled, y)
        
        # 交叉验证评估
        cv_scores = cross_val_score(
            self.model, X_scaled, y,
            cv=self.cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
        )
        cv_rmse = np.sqrt(-cv_scores.mean())
        
        # 预测并计算全集指标
        y_pred = self.model.predict(X_scaled)
        train_rmse = np.sqrt(mean_squared_error(y, y_pred))
        train_mae = mean_absolute_error(y, y_pred)
        train_r2 = r2_score(y, y_pred)
        train_corr, train_p = spearmanr(y, y_pred)
        
        metrics = {
            'n_samples': len(merged),
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'train_r2': train_r2,
            'train_spearman': train_corr,
            'train_spearman_p': train_p,
            'cv_rmse': cv_rmse,
        }
        
        # 记录训练历史
        self.train_history.append({
            'timestamp': pd.Timestamp.now(),
            'n_samples': len(merged),
            'target_metric': target_metric,
            **metrics,
        })
        
        logger.info(f"✅ 模型训练完成 ({self.model_type})")
        logger.info(f"  训练集RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}, Spearman: {train_corr:+.4f}")
        logger.info(f"  交叉验证RMSE: {cv_rmse:.4f}")
        
        return metrics
    
    def predict(self, wfo_df: pd.DataFrame) -> np.ndarray:
        """
        预测真实回测表现（校准后的Sharpe）
        
        Args:
            wfo_df: WFO结果 DataFrame
        
        Returns:
            预测的Sharpe值数组
        """
        if self.model is None:
            raise RuntimeError("模型未训练，请先调用 fit() 方法")
        
        X = self.extract_features(wfo_df)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def calibrate_and_rerank(
        self,
        wfo_df: pd.DataFrame,
        top_k: int = 2000,
    ) -> pd.DataFrame:
        """
        校准WFO结果并重新排序
        
        Args:
            wfo_df: 原始WFO结果
            top_k: 返回TopK组合
        
        Returns:
            校准后的DataFrame，新增 'calibrated_sharpe' 列，按此排序
        """
        if self.model is None:
            logger.warning("模型未训练，返回原始WFO排序（按mean_oos_ic）")
            return wfo_df.nlargest(top_k, 'mean_oos_ic').reset_index(drop=True)
        
        # 预测校准后的Sharpe
        wfo_df = wfo_df.copy()
        wfo_df['calibrated_sharpe'] = self.predict(wfo_df)
        
        # 按校准Sharpe重新排序
        calibrated_top = wfo_df.nlargest(top_k, 'calibrated_sharpe').reset_index(drop=True)
        
        logger.info(f"✅ WFO结果已校准，TopK={top_k}")
        logger.info(f"  校准后Sharpe: min={calibrated_top['calibrated_sharpe'].min():.3f}, "
                   f"max={calibrated_top['calibrated_sharpe'].max():.3f}, "
                   f"median={calibrated_top['calibrated_sharpe'].median():.3f}")
        
        return calibrated_top
    
    def save(self, save_path: Path):
        """保存模型和标准化器"""
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'train_history': self.train_history,
        }, save_path)
        logger.info(f"✅ 校准器已保存: {save_path}")
    
    @classmethod
    def load(cls, load_path: Path) -> 'WFORealBacktestCalibrator':
        """加载已训练模型"""
        data = joblib.load(load_path)
        calibrator = cls(model_type=data['model_type'])
        calibrator.model = data['model']
        calibrator.scaler = data['scaler']
        calibrator.feature_names = data['feature_names']
        calibrator.train_history = data.get('train_history', [])
        logger.info(f"✅ 校准器已加载: {load_path}")
        return calibrator
    
    def analyze_feature_importance(self) -> pd.DataFrame:
        """
        分析特征重要性（仅支持线性模型和GBDT）
        """
        if self.model is None:
            raise RuntimeError("模型未训练")
        
        if self.model_type == "ridge":
            importance = np.abs(self.model.coef_)
        elif self.model_type == "gbdt":
            importance = self.model.feature_importances_
        else:
            raise ValueError(f"模型类型 {self.model_type} 不支持特征重要性分析")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance,
        }).sort_values('importance', ascending=False)
        
        return importance_df


def main():
    """
    示例：使用已有Top2000回测数据训练校准器
    """
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    # 加载数据
    wfo_path = Path("results/run_20251108_193712/all_combos.parquet")
    backtest_path = Path("results_combo_wfo/20251108_193712_20251108_195135/"
                         "top2000_backtest_by_ic_20251108_193712_20251108_195135_full.csv")
    
    if not wfo_path.exists() or not backtest_path.exists():
        print("❌ 数据文件不存在，请先运行WFO和回测")
        return
    
    wfo_df = pd.read_parquet(wfo_path)
    backtest_df = pd.read_csv(backtest_path)
    
    # 训练校准器
    calibrator = WFORealBacktestCalibrator(model_type="ridge", alpha=10.0)
    metrics = calibrator.fit(wfo_df, backtest_df, target_metric='sharpe')
    
    print(f"\n{'='*80}")
    print("训练评估结果")
    print(f"{'='*80}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:20s}: {v:.4f}")
        else:
            print(f"{k:20s}: {v}")
    
    # 特征重要性
    print(f"\n{'='*80}")
    print("特征重要性")
    print(f"{'='*80}")
    importance = calibrator.analyze_feature_importance()
    print(importance.to_string(index=False))
    
    # 校准并重新排序Top100
    calibrated_top100 = calibrator.calibrate_and_rerank(wfo_df, top_k=100)
    
    # 与原始排序对比
    original_top100 = wfo_df.nlargest(100, 'mean_oos_ic')['combo'].tolist()
    calibrated_combo_list = calibrated_top100['combo'].tolist()
    overlap = len(set(original_top100) & set(calibrated_combo_list))
    
    print(f"\n{'='*80}")
    print("排序变化分析")
    print(f"{'='*80}")
    print(f"原始WFO Top100 vs 校准后Top100 重叠: {overlap}/100 ({overlap}%)")
    
    # 保存模型
    save_path = Path("results/calibrator_ridge_alpha10.joblib")
    calibrator.save(save_path)
    
    print(f"\n✅ 校准器训练完成并保存至 {save_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()
