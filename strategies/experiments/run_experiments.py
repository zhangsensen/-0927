#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验管线运行器

功能：
1. 扫描 experiment_configs/ 目录下的 YAML 配置
2. 自动调用 vectorbt_multifactor_grid.py 执行实验
3. 控制单批组合数 ≤ max_total_combos
4. 记录运行日志（配置、时间、输出路径、最优指标）

用法：
    # 运行单个实验
    python strategies/experiments/run_experiments.py \\
        --config experiment_configs/p0_weight_grid_coarse.yaml
    
    # 运行所有 P0 实验
    python strategies/experiments/run_experiments.py \\
        --pattern "p0_*.yaml"
    
    # 运行所有实验
    python strategies/experiments/run_experiments.py --all
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml


def load_experiment_config(config_path: Path) -> Dict:
    """加载实验配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def run_single_experiment(
    config_path: Path,
    script_path: Path,
    dry_run: bool = False
) -> Dict:
    """运行单个实验
    
    Args:
        config_path: 实验配置文件路径
        script_path: vectorbt 脚本路径
        dry_run: 是否仅打印命令不执行
        
    Returns:
        实验运行结果字典
    """
    config = load_experiment_config(config_path)
    exp_info = config.get('experiment', {})
    
    print(f"\n{'='*80}")
    print(f"🧪 实验: {exp_info.get('name', 'Unknown')}")
    print(f"📝 描述: {exp_info.get('description', 'N/A')}")
    print(f"🏷️  阶段: {exp_info.get('phase', 'N/A')}")
    print(f"{'='*80}")
    
    # 构建命令
    cmd = [
        sys.executable,
        str(script_path),
        "--config", str(config_path)
    ]
    
    if dry_run:
        print(f"[DRY RUN] 命令: {' '.join(cmd)}")
        return {
            "config": str(config_path),
            "name": exp_info.get('name'),
            "status": "dry_run",
            "duration": 0.0
        }
    
    # 执行实验
    start_time = time.time()
    try:
        print(f"🚀 开始执行...")
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        duration = time.time() - start_time
        
        print(f"✅ 实验完成，耗时: {duration:.2f}秒")
        
        # 解析输出文件路径
        output_path = config['parameters'].get('output', 'N/A')
        
        return {
            "config": str(config_path),
            "name": exp_info.get('name'),
            "phase": exp_info.get('phase'),
            "status": "success",
            "duration": duration,
            "output_path": output_path,
            "timestamp": datetime.now().isoformat()
        }
        
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"❌ 实验失败，耗时: {duration:.2f}秒")
        print(f"错误输出:\n{e.stderr}")
        
        return {
            "config": str(config_path),
            "name": exp_info.get('name'),
            "phase": exp_info.get('phase'),
            "status": "failed",
            "duration": duration,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def find_experiment_configs(
    config_dir: Path,
    pattern: Optional[str] = None
) -> List[Path]:
    """查找实验配置文件
    
    Args:
        config_dir: 配置目录
        pattern: 文件名模式（支持通配符）
        
    Returns:
        配置文件路径列表
    """
    if pattern:
        configs = list(config_dir.glob(pattern))
    else:
        configs = list(config_dir.glob("*.yaml"))
    
    return sorted(configs)


def save_experiment_log(results: List[Dict], log_path: Path):
    """保存实验日志"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存 JSON 格式
    json_path = log_path.with_suffix('.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 保存 CSV 格式
    df = pd.DataFrame(results)
    csv_path = log_path.with_suffix('.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"\n📁 实验日志已保存:")
    print(f"  - JSON: {json_path}")
    print(f"  - CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="实验管线运行器"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="单个实验配置文件路径"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        help="实验配置文件名模式（如：p0_*.yaml）"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="运行所有实验配置"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印命令，不实际执行"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="strategies/experiments/experiment_configs",
        help="实验配置目录"
    )
    parser.add_argument(
        "--script",
        type=str,
        default="strategies/vectorbt_multifactor_grid.py",
        help="vectorbt 脚本路径"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="strategies/results/experiments",
        help="实验日志目录"
    )
    
    args = parser.parse_args()
    
    # 确定配置文件列表
    config_dir = Path(args.config_dir)
    script_path = Path(args.script)
    
    if not config_dir.exists():
        print(f"❌ 配置目录不存在: {config_dir}")
        return 1
    
    if not script_path.exists():
        print(f"❌ 脚本文件不存在: {script_path}")
        return 1
    
    if args.config:
        # 单个配置
        config_files = [Path(args.config)]
    elif args.pattern:
        # 模式匹配
        config_files = find_experiment_configs(config_dir, args.pattern)
    elif args.all:
        # 所有配置
        config_files = find_experiment_configs(config_dir)
    else:
        print("❌ 请指定 --config, --pattern 或 --all")
        return 1
    
    if not config_files:
        print(f"❌ 未找到匹配的配置文件")
        return 1
    
    print(f"📋 找到 {len(config_files)} 个实验配置:")
    for cf in config_files:
        print(f"  - {cf.name}")
    
    # 运行实验
    results = []
    for config_file in config_files:
        result = run_single_experiment(
            config_file,
            script_path,
            dry_run=args.dry_run
        )
        results.append(result)
    
    # 保存日志
    if not args.dry_run:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = Path(args.log_dir) / f"experiment_log_{timestamp}.json"
        save_experiment_log(results, log_path)
    
    # 汇总统计
    print(f"\n{'='*80}")
    print(f"📊 实验汇总")
    print(f"{'='*80}")
    
    if not args.dry_run:
        success_count = sum(1 for r in results if r['status'] == 'success')
        failed_count = sum(1 for r in results if r['status'] == 'failed')
        total_duration = sum(r['duration'] for r in results)
        
        print(f"✅ 成功: {success_count}")
        print(f"❌ 失败: {failed_count}")
        print(f"⏱️  总耗时: {total_duration:.2f}秒")
    else:
        print(f"[DRY RUN] 共 {len(results)} 个实验")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
