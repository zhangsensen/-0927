"""
LTR模型模块: LightGBM LambdaRank排序模型
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


class LTRRanker:
    """
    Learning-to-Rank排序模型
    
    使用LightGBM的lambdarank目标函数优化排序质量
    """
    
    def __init__(
        self,
        objective: str = "lambdarank",
        metric: str = "ndcg",
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
        min_data_in_leaf: int = 20,
        feature_fraction: float = 0.8,
        bagging_fraction: float = 0.8,
        bagging_freq: int = 5,
        lambda_l1: float = 0.1,
        lambda_l2: float = 0.1,
        random_state: int = 42,
        verbose: int = -1,
        **kwargs
    ):
        """
        初始化LTR模型参数
        
        Args:
            objective: 目标函数 (lambdarank, rank_xendcg等)
            metric: 评估指标 (ndcg, map等)
            其他参数: LightGBM标准参数
        """
        self.params = {
            "objective": objective,
            "metric": metric,
            "boosting_type": "gbdt",
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "min_data_in_leaf": min_data_in_leaf,
            "feature_fraction": feature_fraction,
            "bagging_fraction": bagging_fraction,
            "bagging_freq": bagging_freq,
            "lambda_l1": lambda_l1,
            "lambda_l2": lambda_l2,
            "random_state": random_state,
            "verbose": verbose,
            **kwargs
        }
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.cv_results = []
        
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5,
        early_stopping_rounds: int = 50,
        verbose_eval: int = 50
    ) -> Dict[str, Any]:
        """
        训练LTR模型 (带交叉验证)
        
        Args:
            X: 特征矩阵
            y: 目标变量 (用于排序的分数)
            cv_folds: 交叉验证折数
            early_stopping_rounds: 早停轮数
            verbose_eval: 打印频率
            
        Returns:
            训练结果dict包含CV分数、最佳迭代等
        """
        print(f"\n{'='*80}")
        print("开始训练LTR模型")
        print(f"{'='*80}")
        
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        self.feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else None
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X_array)
        
        # K折交叉验证
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self.params["random_state"])
        
        cv_scores = []
        fold_models = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_scaled), 1):
            print(f"\n--- Fold {fold_idx}/{cv_folds} ---")
            
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # 检查是否使用ranking目标 (需要query group)
            use_ranking = self.params["objective"] in ["lambdarank", "rank_xendcg"]
            
            # 创建LightGBM数据集
            if use_ranking:
                # LambdaRank需要group信息
                # 注意：每个fold作为单独query可能超过10000行限制
                # 如果超限，改用regression
                if len(train_idx) > 10000:
                    print(f"  ⚠️  训练集超过10000行，自动切换为regression模式")
                    self.params["objective"] = "regression"
                    self.params["metric"] = "rmse"
                    use_ranking = False
            
            if use_ranking:
                train_data = lgb.Dataset(
                    X_train, 
                    label=y_train.values,
                    group=[len(train_idx)],  # 整个fold作为一个query
                    feature_name=self.feature_names
                )
                
                val_data = lgb.Dataset(
                    X_val,
                    label=y_val.values,
                    group=[len(val_idx)],
                    reference=train_data,
                    feature_name=self.feature_names
                )
            else:
                # Regression模式：不需要query group
                train_data = lgb.Dataset(
                    X_train, 
                    label=y_train.values,
                    feature_name=self.feature_names
                )
                
                val_data = lgb.Dataset(
                    X_val,
                    label=y_val.values,
                    reference=train_data,
                    feature_name=self.feature_names
                )
            
            # 训练模型
            model = lgb.train(
                self.params,
                train_data,
                valid_sets=[train_data, val_data],
                valid_names=["train", "valid"],
                callbacks=[
                    lgb.early_stopping(early_stopping_rounds),
                    lgb.log_evaluation(verbose_eval)
                ]
            )
            
            fold_models.append(model)
            
            # 评估验证集
            val_pred = model.predict(X_val)
            
            # 计算Spearman相关性
            from scipy.stats import spearmanr
            spearman_corr, _ = spearmanr(y_val.values, val_pred)
            
            cv_scores.append(spearman_corr)
            
            self.cv_results.append({
                "fold": fold_idx,
                "best_iteration": model.best_iteration,
                "spearman_corr": spearman_corr,
                "n_train": len(train_idx),
                "n_val": len(val_idx)
            })
            
            print(f"  Fold {fold_idx} Spearman相关性: {spearman_corr:.4f}")
        
        # 选择最佳模型 (Spearman最高的)
        best_fold_idx = np.argmax(cv_scores)
        self.model = fold_models[best_fold_idx]
        
        # 训练结果汇总
        results = {
            "cv_mean_spearman": np.mean(cv_scores),
            "cv_std_spearman": np.std(cv_scores),
            "cv_scores": cv_scores,
            "best_fold": best_fold_idx + 1,
            "best_spearman": cv_scores[best_fold_idx],
            "feature_importance": self.get_feature_importance()
        }
        
        print(f"\n{'='*80}")
        print("训练完成")
        print(f"{'='*80}")
        print(f"CV平均Spearman相关性: {results['cv_mean_spearman']:.4f} ± {results['cv_std_spearman']:.4f}")
        print(f"最佳Fold: {results['best_fold']} (Spearman={results['best_spearman']:.4f})")
        
        return results
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测排序分数
        
        Args:
            X: 特征矩阵
            
        Returns:
            Tuple of (predicted_scores, ranks)
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用train()方法")
        
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        X_scaled = self.scaler.transform(X_array)
        
        # 预测分数
        scores = self.model.predict(X_scaled)
        
        # 计算排名 (分数越高排名越靠前)
        ranks = pd.Series(scores).rank(ascending=False, method="min").values.astype(int)
        
        return scores, ranks
    
    def get_feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """
        获取特征重要性
        
        Args:
            importance_type: 重要性类型 (gain, split)
            
        Returns:
            特征重要性DataFrame
        """
        if self.model is None:
            raise ValueError("模型未训练")
        
        importance = self.model.feature_importance(importance_type=importance_type)
        
        feature_names = self.feature_names or [f"f{i}" for i in range(len(importance))]
        
        df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        }).sort_values("importance", ascending=False)
        
        return df
    
    def save(self, save_dir: str | Path, model_name: str = "ltr_ranker"):
        """
        保存模型和元信息
        
        Args:
            save_dir: 保存目录
            model_name: 模型名称前缀
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存LightGBM模型
        model_path = save_dir / f"{model_name}.txt"
        self.model.save_model(str(model_path))
        
        # 保存scaler和特征信息
        meta_path = save_dir / f"{model_name}_meta.pkl"
        with open(meta_path, "wb") as f:
            pickle.dump({
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "params": self.params,
                "cv_results": self.cv_results
            }, f)
        
        # 保存特征列名 (JSON格式便于查看)
        if self.feature_names:
            feature_path = save_dir / f"{model_name}_features.json"
            with open(feature_path, "w") as f:
                json.dump(self.feature_names, f, indent=2)
        
        print(f"\n✓ 模型已保存到: {save_dir}")
        print(f"  - 模型文件: {model_path.name}")
        print(f"  - 元信息: {meta_path.name}")
        print(f"  - 特征列表: {model_name}_features.json")
    
    @classmethod
    def load(cls, save_dir: str | Path, model_name: str = "ltr_ranker") -> "LTRRanker":
        """
        加载已保存的模型
        
        Args:
            save_dir: 模型目录
            model_name: 模型名称前缀
            
        Returns:
            加载的LTRRanker实例
        """
        save_dir = Path(save_dir)
        
        model_path = save_dir / f"{model_name}.txt"
        meta_path = save_dir / f"{model_name}_meta.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"元信息文件不存在: {meta_path}")
        
        # 加载元信息
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        
        # 创建实例
        ranker = cls(**meta["params"])
        
        # 加载模型
        ranker.model = lgb.Booster(model_file=str(model_path))
        ranker.scaler = meta["scaler"]
        ranker.feature_names = meta["feature_names"]
        ranker.cv_results = meta.get("cv_results", [])
        
        print(f"✓ 模型已加载: {model_path}")
        
        return ranker
