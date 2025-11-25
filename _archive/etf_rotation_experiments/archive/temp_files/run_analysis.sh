#!/bin/bash
cd /Users/zhangshenshen/深度量化0927/etf_rotation_experiments
/usr/local/bin/python scripts/analyze_actual_results.py 2>&1 | tee ml_ranking/analysis_output.log
