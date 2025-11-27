#!/usr/bin/env python3
"""
应用已训练的LTR模型对新WFO结果排序
"""
import argparse
from pathlib import Path
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

from strategies.ml_ranker.data_loader import load_wfo_features
from strategies.ml_ranker.feature_engineer import build_feature_matrix
from strategies.ml_ranker.ltr_model import LTRRanker


def apply_ltr_ranking(model_path: str, wfo_dir: str | Path, output_path: str | Path = None, top_k: int = None, verbose: bool = True) -> pd.DataFrame:
    """
    应用LTR模型对WFO结果排序 (可复用函数)
    
    Args:
        model_path: 训练好的模型路径 (无扩展名, 如 ml_ranker/models/ltr_ranker)
        wfo_dir: WFO结果目录
        output_path: 输出CSV路径 (可选, 默认为 <wfo_dir>/ranked_combos.csv)
        top_k: 保存 Top-K 结果 (可选, 为 None 则不单独保存)
        verbose: 是否打印详细日志
        
    Returns:
        result_df: 包含 ltr_score, ltr_rank 的完整排序结果
        
    Example:
        >>> result_df = apply_ltr_ranking(
        ...     model_path="ml_ranker/models/ltr_ranker",
        ...     wfo_dir="results/run_20251114_155420",
        ...     top_k=200
        ... )
        >>> print(result_df.head())
    """
    wfo_dir = Path(wfo_dir)
    
    if verbose:
        print(f"\n{'='*80}")
        print("🎯 应用LTR排序模型")
        print(f"{'='*80}\n")
    
    # 1. 加载模型
    if verbose:
        print(f"📂 加载模型: {model_path}")
    model = LTRRanker.load(model_path)
    if verbose:
        print(f"  ✓ 模型加载成功")
        print(f"  特征数: {len(model.feature_names)}")
    
    # 2. 加载WFO数据
    if verbose:
        print(f"\n📂 加载WFO数据: {wfo_dir}")
    df_wfo = load_wfo_features(wfo_dir)
    if verbose:
        print(f"  ✓ 加载 {len(df_wfo)} 个策略组合")
    
    # 3. 构建特征
    if verbose:
        print(f"\n🛠️ 构建特征矩阵")
    X_df = build_feature_matrix(df_wfo)
    X = X_df.values
    feature_names = list(X_df.columns)
    if verbose:
        print(f"  ✓ 特征维度: {X.shape}")
    
    # 验证特征对齐
    if feature_names != model.feature_names:
        if verbose:
            print(f"  ⚠️  特征名称不匹配，尝试重新排列...")
        feature_df = pd.DataFrame(X, columns=feature_names)
        X = feature_df[model.feature_names].values
        feature_names = model.feature_names
        if verbose:
            print(f"  ✓ 特征已对齐")
    
    # 4. 预测排序
    if verbose:
        print(f"\n🎯 预测排序分数")
    scores, ranks = model.predict(X)
    if verbose:
        print(f"  ✓ 预测完成")
        print(f"  分数范围: [{scores.min():.4f}, {scores.max():.4f}]")
    
    # 5. 构建结果表
    if verbose:
        print(f"\n📊 生成排序结果")
    result_df = df_wfo.copy()
    result_df["ltr_score"] = scores
    result_df["ltr_rank"] = ranks
    
    # 添加原始WFO排名用于对比
    result_df["wfo_rank"] = result_df["mean_oos_ic"].rank(ascending=False, method="min").astype(int)
    result_df["rank_change"] = result_df["wfo_rank"] - result_df["ltr_rank"]
    
    # 按LTR排名排序
    result_df = result_df.sort_values("ltr_rank").reset_index(drop=True)
    
    # 6. 保存结果 (如果指定了输出路径)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        if verbose:
            print(f"\n💾 全量排序结果已保存: {output_path}")
        
        # 保存Top-K (如果指定)
        if top_k is not None and top_k > 0:
            top_k_df = result_df.head(top_k)
            topk_path = output_path.parent / f"ranked_top{top_k}.csv"
            top_k_df.to_csv(topk_path, index=False, encoding="utf-8-sig")
            if verbose:
                print(f"💾 Top-{top_k} 结果已保存: {topk_path}")
    
    # 7. 显示Top-10 (如果需要)
    if verbose and len(result_df) > 0:
        display_k = min(10, len(result_df))
        print(f"\n🏆 Top-{display_k} 策略 (LTR排序):")
        print(f"{'='*80}")
        for idx, row in result_df.head(display_k).iterrows():
            combo_str = row['combo'] if len(row['combo']) <= 60 else row['combo'][:57] + '...'
            print(f"  #{int(row['ltr_rank']):3d}  {combo_str:60s}  "
                  f"score={row['ltr_score']:7.4f}  "
                  f"WFO排名: #{int(row['wfo_rank']):4d}  "
                  f"变化: {int(row['rank_change']):+4d}")
    
    return result_df


def parse_args():
    parser = argparse.ArgumentParser(description="应用LTR模型对WFO结果排序")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="训练好的模型路径 (无扩展名, 如 ml_ranker/models/ltr_ranker)"
    )
    parser.add_argument(
        "--wfo-dir",
        type=str,
        required=True,
        help="WFO结果目录"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出CSV路径 (默认: <wfo-dir>/ranked_combos.csv)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="输出Top-K策略"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 调用核心函数
    result_df = apply_ltr_ranking(
        model_path=args.model,
        wfo_dir=args.wfo_dir,
        output_path=args.output if args.output else Path(args.wfo_dir) / "ranked_combos.csv",
        top_k=args.top_k,
        verbose=True
    )
    
    # 8. 统计摘要
    print(f"\n{'='*80}")
    print("📈 排序摘要")
    print(f"{'='*80}")
    
    print(f"  总策略数: {len(result_df)}")
    print(f"  LTR分数均值: {result_df['ltr_score'].mean():.4f}")
    print(f"  LTR分数标准差: {result_df['ltr_score'].std():.4f}")
    
    # Top-K的WFO指标平均值
    top_k_df = result_df.head(args.top_k)
    print(f"\n  Top-{args.top_k} 策略的WFO指标平均:")
    for col in ["mean_oos_ic", "oos_sharpe_proxy", "stability_score", "mean_oos_pvalue"]:
        if col in top_k_df.columns:
            print(f"    {col:20s}: {top_k_df[col].mean():8.4f}")
    
    # 与WFO Top-K对比
    wfo_topk = result_df.nsmallest(args.top_k, "wfo_rank")
    overlap = len(set(top_k_df["combo"]) & set(wfo_topk["combo"]))
    print(f"\n  与WFO Top-{args.top_k} 重叠数: {overlap}/{args.top_k} ({overlap/args.top_k*100:.1f}%)")
    
    print(f"\n{'='*80}")
    print("✅ 排序完成")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
