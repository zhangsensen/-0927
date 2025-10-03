# Utils package for factor screening system

import sys
import os
# 添加父目录到path以便导入utils.py
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from utils import (
        FactorFileAligner,
        find_aligned_factor_files,
        validate_factor_alignment
    )
except ImportError as e:
    # 如果无法导入，创建空的类和函数以避免程序崩溃
    print(f"Warning: Could not import from utils: {e}")
    class FactorFileAligner:
        def __init__(self, *args, **kwargs):
            pass

    def find_aligned_factor_files(*args, **kwargs):
        return []

    def validate_factor_alignment(*args, **kwargs):
        return True

__all__ = ['FactorFileAligner', 'find_aligned_factor_files', 'validate_factor_alignment']