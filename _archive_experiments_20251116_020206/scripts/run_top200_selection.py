#!/usr/bin/env python3
"""
Top-200 组合筛选脚本

从约 2000 个回测组合中筛选出 Top-200，用于后续精开发。

使用方法:
    python scripts/run_top200_selection.py --input data.csv --output top200.csv
    
更多选项请运行:
    python scripts/run_top200_selection.py --help
"""

import sys
from pathlib import Path

# 添加项目根目录到 sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from selection.cli import main

if __name__ == '__main__':
    main()
