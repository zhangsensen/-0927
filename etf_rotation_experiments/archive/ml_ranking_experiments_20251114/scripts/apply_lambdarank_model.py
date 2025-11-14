#!/usr/bin/env python3
"""
应用 LambdaMART 模型到最新 WFO run，生成新的排序结果
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from lightgbm import Booster


def main():
    parser = argparse.ArgumentParser(description='应用 LambdaMART 模型进行排序')
    parser.add_argument('--run-dir', type=str, required=True, help='WFO run 目录')
    parser.add_argument('--model-path', type=str, required=True, help='LambdaMART 模型路径')
    parser.add_argument('--output-suffix', type=str, default='lambdarank', help='输出文件后缀')
    
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    model_path = Path(args.model_path)
    
    print("=" * 80)
    print("🚀 应用 LambdaMART 模型")
    print("=" * 80)
    print(f"Run 目录: {run_dir}")
    print(f"模型路径: {model_path}")
    
    # 1. 加载模型
    print("\n加载模型...")
    model = Booster(model_file=str(model_path))
    
    # 加载特征列表
    model_dir = model_path.parent
    importance_file = model_dir / f"{model_path.stem}_importance.json"
    with open(importance_file) as f:
        importance_data = json.load(f)
    feature_names = list(importance_data.keys())
    print(f"✅ 模型加载成功，特征数: {len(feature_names)}")
    
    # 2. 加载 WFO 结果数据
    print("\n加载 WFO 结果...")
    baseline_file = run_dir / 'ranking_blends/ranking_baseline.parquet'
    if not baseline_file.exists():
        raise FileNotFoundError(f"未找到基准排序文件: {baseline_file}")
    
    df = pd.read_parquet(baseline_file)
    print(f"✅ 加载 {len(df)} 个策略组合")
    
    # 3. 准备特征
    print("\n准备特征...")
    # 检查特征可用性
    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features:
        print(f"⚠️  缺失 {len(missing_features)} 个特征，将用中位数填充")
        print(f"   示例: {missing_features[:5]}")
    
    # 提取特征矩阵
    X = df[feature_names].copy()
    
    # 处理数据类型
    for col in X.select_dtypes(include=['bool']).columns:
        X[col] = X[col].astype(int)
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # 填充缺失值
    if X.isna().any().any():
        X = X.fillna(X.median(numeric_only=True))
    
    print(f"✅ 特征矩阵准备完成: {X.shape}")
    
    # 4. 模型预测
    print("\n执行预测...")
    predictions = model.predict(X)
    
    # 5. 生成新排序
    print("\n生成排序结果...")
    result_df = df.copy()
    result_df[f'{args.output_suffix}_score'] = predictions
    result_df = result_df.sort_values(f'{args.output_suffix}_score', ascending=False)
    result_df['rank_score'] = result_df[f'{args.output_suffix}_score']  # 为回测脚本准备
    
    # 6. 保存结果
    output_dir = run_dir / 'ranking_blends'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'ranking_{args.output_suffix}.parquet'
    
    result_df.to_parquet(output_file, index=False)
    print(f"\n✅ 排序结果已保存: {output_file}")
    
    # 7. 统计信息
    print("\n" + "=" * 80)
    print("📊 排序统计")
    print("=" * 80)
    print(f"Top1 策略: {result_df.iloc[0]['combo']}")
    print(f"Top1 分数: {result_df.iloc[0][f'{args.output_suffix}_score']:.4f}")
    print(f"\n分数分布:")
    print(f"  最大值: {predictions.max():.4f}")
    print(f"  中位数: {float(pd.Series(predictions).median()):.4f}")
    print(f"  最小值: {predictions.min():.4f}")
    
    # 与 IC 排序的差异
    ic_top100 = set(df.nlargest(100, 'mean_oos_ic')['combo'].values)
    rank_top100 = set(result_df.head(100)['combo'].values)
    overlap = len(ic_top100 & rank_top100)
    print(f"\n与 IC Top100 重叠: {overlap}/100 ({overlap}%)")
    
    print("=" * 80)
    print("✅ 完成")
    print("=" * 80)


if __name__ == '__main__':
    main()
