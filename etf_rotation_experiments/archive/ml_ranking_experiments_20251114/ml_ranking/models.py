"""Model factories for the ETF supervised ranking project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover - optional dependency
    lgb = None


@dataclass(slots=True)
class ModelConfig:
    elasticnet_alpha: float = 0.1
    elasticnet_l1_ratio: float = 0.5
    tree_depth: int = 6
    tree_min_samples_leaf: int = 5
    lgbm_learning_rate: float = 0.05
    lgbm_num_leaves: int = 31
    lgbm_estimators: int = 300


def make_linear_model(config: ModelConfig) -> Pipeline:
    """Create a standardized ElasticNet regression model."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "regressor",
                ElasticNet(
                    alpha=config.elasticnet_alpha,
                    l1_ratio=config.elasticnet_l1_ratio,
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )


def make_tree_model(config: ModelConfig) -> DecisionTreeRegressor:
    """Create a simple decision tree regressor baseline."""

    return DecisionTreeRegressor(
        max_depth=config.tree_depth,
        min_samples_leaf=config.tree_min_samples_leaf,
        random_state=42,
    )


def make_lgbm_regressor(config: ModelConfig) -> "lgb.LGBMRegressor":
    """Create a LightGBM regressor for ranking scores."""

    if lgb is None:
        raise ModuleNotFoundError("lightgbm is not installed in the current environment")
    return lgb.LGBMRegressor(
        learning_rate=config.lgbm_learning_rate,
        num_leaves=config.lgbm_num_leaves,
        n_estimators=config.lgbm_estimators,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )


def make_lgbm_ranker(config: ModelConfig) -> "lgb.LGBMRanker":
    """Create a LightGBM LambdaMART style ranker."""

    if lgb is None:
        raise ModuleNotFoundError("lightgbm is not installed in the current environment")
    return lgb.LGBMRanker(
        learning_rate=config.lgbm_learning_rate,
        num_leaves=config.lgbm_num_leaves,
        n_estimators=config.lgbm_estimators,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="lambdarank",
        importance_type="gain",
    )


def baseline_model_registry(config: ModelConfig | None = None) -> Dict[str, Any]:
    """Return a mapping of baseline models to train."""

    cfg = config or ModelConfig()
    registry: Dict[str, Any] = {
        "elasticnet": make_linear_model(cfg),
        "decision_tree": make_tree_model(cfg),
    }
    try:
        registry["lgbm_regressor"] = make_lgbm_regressor(cfg)
    except ModuleNotFoundError:
        pass
    return registry
