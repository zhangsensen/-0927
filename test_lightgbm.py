#!/usr/bin/env python3
"""
LightGBM安装验证和性能测试脚本
专为Apple Silicon Mac优化
"""

import lightgbm as lgb
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

def test_lightgbm_installation():
    """测试LightGBM安装和基本功能"""
    print("🍎 LightGBM在Apple Silicon Mac上的测试")
    print("=" * 50)

    # 显示版本信息
    print(f"LightGBM版本: {lgb.__version__}")
    print()

    # 创建测试数据
    print("📊 生成测试数据...")
    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=15,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    print()

    # Apple Silicon优化参数
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'device': 'cpu',  # Mac上CPU通常更稳定
        'num_threads': 4,  # M芯片优化，可根据核心数调整
        'seed': 42
    }

    # 训练模型
    print("🚀 开始训练LightGBM模型...")
    start_time = time.time()

    # 创建数据集
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # 训练
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
    )

    training_time = time.time() - start_time
    print(f"✅ 训练完成！用时: {training_time:.2f}秒")

    # 预测和评估
    print("📈 模型评估...")
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred_binary)

    print(f"准确率: {accuracy:.4f}")
    print(f"训练轮数: {model.num_trees()}")
    print()

    # 特征重要性
    feature_importance = model.feature_importance()
    top_features = np.argsort(feature_importance)[-5:][::-1]

    print("🎯 Top 5 重要特征:")
    for i, feature_idx in enumerate(top_features, 1):
        print(f"  {i}. 特征 {feature_idx}: {feature_importance[feature_idx]:.2f}")

    print()
    print("🎉 LightGBM测试完成！在你的Mac上运行完美。")

    return model

def performance_benchmark():
    """性能基准测试"""
    print("\n⚡ 性能基准测试")
    print("-" * 30)

    # 不同大小的数据集
    sizes = [1000, 5000, 10000]

    for size in sizes:
        print(f"\n数据集大小: {size:,}")

        X, y = make_classification(
            n_samples=size,
            n_features=20,
            random_state=42
        )

        # 测试训练时间
        start_time = time.time()

        train_data = lgb.Dataset(X, label=y)
        model = lgb.train(
            {
                'objective': 'binary',
                'verbose': -1,
                'num_threads': 4,
                'device': 'cpu'
            },
            train_data,
            num_boost_round=50
        )

        elapsed = time.time() - start_time
        print(f"  训练时间: {elapsed:.3f}秒")

if __name__ == "__main__":
    # 运行测试
    model = test_lightgbm_installation()
    performance_benchmark()

    print("\n💡 使用建议:")
    print("- 在你的M芯片Mac上，推荐使用CPU而非GPU")
    print("- 可以调整num_threads参数来优化性能")
    print("- 对于大数据集，LightGBM在Mac上表现优异")