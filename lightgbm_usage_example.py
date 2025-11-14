#!/usr/bin/env python3
"""
LightGBM使用示例 - 适用于Apple Silicon Mac
包含分类、回归和参数调优示例
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error
import time

# ================================
# 1. 分类任务示例
# ================================

def binary_classification_example():
    """二分类任务示例"""
    print("🎯 二分类任务示例")
    print("-" * 30)

    # 生成数据
    X, y = make_classification(
        n_samples=5000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Apple Silicon优化的参数
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'num_threads': 4,  # M芯片优化
        'seed': 42
    }

    # 训练
    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
    )

    # 预测
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred_binary)

    print(f"准确率: {accuracy:.4f}")
    print(f"训练轮数: {model.num_trees()}")

    return model

# ================================
# 2. 回归任务示例
# ================================

def regression_example():
    """回归任务示例"""
    print("\n📈 回归任务示例")
    print("-" * 30)

    # 生成数据
    X, y = make_regression(
        n_samples=3000,
        n_features=15,
        n_informative=10,
        noise=0.1,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 回归参数
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'verbose': -1,
        'num_threads': 4,
        'seed': 42
    }

    # 训练
    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
    )

    # 预测
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"RMSE: {rmse:.4f}")
    print(f"训练轮数: {model.num_trees()}")

    return model

# ================================
# 3. Scikit-learn接口示例
# ================================

def sklearn_interface_example():
    """使用Scikit-learn接口"""
    print("\n🔧 Scikit-learn接口示例")
    print("-" * 30)

    # 生成数据
    X, y = make_classification(
        n_samples=2000,
        n_features=10,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 使用LGBMClassifier
    clf = lgb.LGBMClassifier(
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=100,
        n_jobs=4,  # 并行处理
        random_state=42,
        verbose=-1
    )

    # 训练
    start_time = time.time()
    clf.fit(X_train, y_train)
    training_time = time.time() - start_time

    # 预测
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"准确率: {accuracy:.4f}")
    print(f"训练时间: {training_time:.3f}秒")

    return clf

# ================================
# 4. 参数调优示例
# ================================

def hyperparameter_tuning():
    """参数调优示例"""
    print("\n⚙️ 参数调优示例")
    print("-" * 30)

    # 生成小数据集用于调优
    X, y = make_classification(
        n_samples=1000,
        n_features=15,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 定义参数网格
    param_grid = {
        'num_leaves': [15, 31, 63],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [50, 100, 200],
        'min_child_samples': [10, 20, 30]
    }

    # 使用GridSearchCV
    lgb_clf = lgb.LGBMClassifier(
        n_jobs=4,
        random_state=42,
        verbose=-1
    )

    grid_search = GridSearchCV(
        lgb_clf,
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=1,
        verbose=0
    )

    print("正在进行参数调优...")
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    tuning_time = time.time() - start_time

    print(f"调优完成！用时: {tuning_time:.2f}秒")
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳分数: {grid_search.best_score_:.4f}")

    # 使用最佳模型
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    print(f"测试集准确率: {test_accuracy:.4f}")

# ================================
# 5. 处理Pandas DataFrame
# ================================

def pandas_dataframe_example():
    """处理Pandas DataFrame"""
    print("\n📊 Pandas DataFrame示例")
    print("-" * 30)

    # 创建示例DataFrame
    np.random.seed(42)
    data = {
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'feature3': np.random.choice(['A', 'B', 'C'], 1000),
        'feature4': np.random.rand(1000),
        'target': np.random.choice([0, 1], 1000)
    }

    df = pd.DataFrame(data)

    # 处理分类特征
    df_encoded = pd.get_dummies(df, columns=['feature3'])

    # 分离特征和目标
    X = df_encoded.drop('target', axis=1)
    y = df_encoded['target']

    # 使用feature_name指定特征名
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 训练模型
    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        feature_name=feature_names,
        categorical_feature=['feature3_A', 'feature3_B', 'feature3_C']
    )

    model = lgb.train(
        {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbose': -1,
            'num_threads': 4
        },
        train_data,
        num_boost_round=50
    )

    # 预测
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred_binary)

    print(f"准确率: {accuracy:.4f}")

    # 特征重要性
    feature_importance = dict(zip(feature_names, model.feature_importance()))
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]

    print("Top 3 重要特征:")
    for feature, importance in top_features:
        print(f"  {feature}: {importance:.2f}")

# ================================
# 主函数
# ================================

if __name__ == "__main__":
    print("🍎 LightGBM在Apple Silicon Mac上的使用示例")
    print("=" * 50)

    # 运行所有示例
    binary_classification_example()
    regression_example()
    sklearn_interface_example()
    hyperparameter_tuning()
    pandas_dataframe_example()

    print("\n💡 Apple Silicon Mac优化建议:")
    print("1. 使用 num_threads=4 (或你的M芯片核心数)")
    print("2. 优先使用CPU而非GPU")
    print("3. 设置 verbose=-1 减少输出开销")
    print("4. 使用 early_stopping 防止过拟合")
    print("5. 对于大数据集，使用 feature_fraction 和 bagging_fraction")

    print("\n🚀 开始你的LightGBM之旅吧！")