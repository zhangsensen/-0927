COMPREHENSIVE PROJECT CLEANUP & VERIFICATION GUIDE
====================================================

Generated: 2025-10-22 13:30
Status: Ready for Execution

---

## PHASE 1: CRITICAL DELETIONS (5 MINUTES)
===========================================

### 1.1 Root Directory Orphaned Scripts
These scripts are standalone analysis tools, NOT part of production pipeline:

Files to DELETE:
├── corrected_net_analysis.py          → Standalone analysis, can reproduce from backtest results
├── debug_static_checker.py            → Development debugging tool
├── diagnose_etf_issues.py             → Old diagnostic utility
├── net_return_analysis.py             → Redundant with backtest output
├── simple_factor_screen.py            → Obsolete, replaced by run_etf_cross_section_configurable.py
├── test_rotation_factors.py           → Isolated unit test
├── turnover_penalty_optimizer.py      → Experimental, not in production
├── verify_single_combo.py             → Development verification tool
└── vulture_whitelist.py               → Code analysis helper (keep if using vulture)

Deletion command:
```bash
cd /Users/zhangshenshen/深度量化0927
rm -f corrected_net_analysis.py debug_static_checker.py diagnose_etf_issues.py \
      net_return_analysis.py simple_factor_screen.py test_rotation_factors.py \
      turnover_penalty_optimizer.py verify_single_combo.py
echo "✅ Removed 8 orphaned scripts"
```

### 1.2 Backup Archives
These are superseded by Git history:

Files to DELETE:
├── etf_rotation_system_backup_20251021_235648.tar.gz  → Old backup

Deletion command:
```bash
rm -f *.tar.gz
echo "✅ Removed backup archives"
```

---

## PHASE 2: EXPERIMENTAL REPORTS ARCHIVAL (5 MINUTES)
====================================================

### 2.1 Identify Experimental Reports
These are one-off analysis from development iterations:

Experimental files (80+ KB):
├── AUDIT_SUMMARY.txt
├── BACKTEST_12FACTORS_COMPARISON_REPORT.md
├── BACKTEST_50K_COMPLETION.txt
├── BASELINE_BREAKTHROUGH_STRATEGY.md
├── CLEANUP_REPORT.md
├── COMPLETE_BACKTEST_EVOLUTION_REPORT.md
├── CRITICAL_ISSUES_REPORT.md
├── ETF_FULL_WORKFLOW_VALIDATION_REPORT.md
├── ETF_ROTATION_OPTIMIZATION_SUMMARY.md
├── ETF_ROTATION_SYSTEM_ANALYSIS_REPORT.md
├── ETF横截面因子扩展方案.md
├── ETF横截面因子评估报告.md
├── EXECUTIVE_SUMMARY.md
├── FACTOR_SCREENING_OPTIMIZATION_REPORT.md
├── FIX_COMPLETE_REPORT.md
├── LARGE_SCALE_BACKTEST_50K_REPORT.md
├── LOG_OPTIMIZATION_REPORT.md
├── OPTIMIZATION_COMPLETE_REPORT.md
├── PHASE1_OPTIMIZATION_REPORT.md
├── PHASE1_SUMMARY.txt
├── VBT_BACKTEST_ISSUES.md
├── VERIFICATION_REPORT_20251022.md
├── *.json (15+ test reports)
└── ...

### 2.2 Core Documents to Preserve
These are essential project documentation:

Keep ONLY:
├── README.md                           → Project overview
├── CLAUDE.md                           → Architecture & standards
├── ETF_ROTATION_QUICK_START.md        → User guide
├── CHANGELOG.md                        → Version history
├── .github/copilot-instructions.md    → AI guidance
└── Makefile                            → Build/test commands

### 2.3 Archival Command
```bash
# Create archive directory
mkdir -p docs/archive

# Move experimental reports (if needed for reference)
# mv AUDIT_SUMMARY.txt docs/archive/
# mv BACKTEST_* docs/archive/
# ... (optional, for audit trail)

# OR simply delete (git history preserves them)
rm -f AUDIT_SUMMARY.txt BACKTEST_* BASELINE_* CLEANUP_* COMPLETE_* \
      CRITICAL_* ETF_FULL_* ETF_ROTATION_OPTIMIZATION_* ETF_ROTATION_SYSTEM_* \
      ETF_* EXECUTIVE_* FACTOR_SCREENING_* FIX_* LARGE_SCALE_* LOG_* \
      OPTIMIZATION_* PHASE1_* VBT_* VERIFICATION_* \
      config_*.json functional_*.json performance_*.json etf_*.json

echo "✅ Cleaned up experimental reports"
```

---

## PHASE 3: CONFIG FILE CONSOLIDATION (10 MINUTES)
================================================

### 3.1 Current State: Config Chaos

Scattered configs:
├── etf_rotation_system/01_横截面建设/config/factor_panel_config.yaml
├── etf_rotation_system/02_因子筛选/optimized_screening_config.yaml
├── etf_rotation_system/02_因子筛选/sample_etf_config.yaml
├── etf_rotation_system/02_因子筛选/test_config_output.yaml
├── etf_rotation_system/03_vbt回测/parallel_backtest_config.yaml
├── etf_rotation_system/03_vbt回测/fine_grained_config.yaml
├── etf_rotation_system/04_精细策略/config/fine_strategy_config.yaml
└── Multiple Python config classes

### 3.2 Target State: Unified Config

Consolidated structure:
etf_rotation_system/
├── config/
│   ├── __init__.py
│   ├── factor_panel_config.yaml          ← Moved from 01_横截面建设/
│   ├── screening_config.yaml             ← Renamed from optimized_screening_config.yaml
│   ├── backtest_config.yaml              ← Consolidated from 03_vbt回测/
│   ├── strategy_config.yaml              ← Moved from 04_精细策略/
│   └── schema/
│       ├── factor_panel.schema.json
│       ├── screening.schema.json
│       ├── backtest.schema.json
│       └── strategy.schema.json
├── 01_横截面建设/
│   └── (no local config - reads from ../config/)
├── 02_因子筛选/
│   └── (no local config - reads from ../config/)
├── 03_vbt回测/
│   └── (no local config - reads from ../config/)
└── 04_精细策略/
    └── (no local config - reads from ../config/)
```

### 3.3 Consolidation Plan

Step 1: Create target directory
```bash
mkdir -p etf_rotation_system/config/schema
```

Step 2: Move core configs
```bash
cp etf_rotation_system/01_横截面建设/config/factor_panel_config.yaml \
   etf_rotation_system/config/
   
cp etf_rotation_system/02_因子筛选/optimized_screening_config.yaml \
   etf_rotation_system/config/screening_config.yaml

cp etf_rotation_system/03_vbt回测/parallel_backtest_config.yaml \
   etf_rotation_system/config/backtest_config.yaml

cp etf_rotation_system/04_精细策略/config/fine_strategy_config.yaml \
   etf_rotation_system/config/strategy_config.yaml
```

Step 3: Update import paths in Python files
```bash
# Update generate_panel_refactored.py
# FROM: config/factor_panel_config.yaml
# TO: ../config/factor_panel_config.yaml

# Update run_etf_cross_section_configurable.py
# FROM: optimized_screening_config.yaml
# TO: ../config/screening_config.yaml

# Update parallel_backtest_configurable.py
# FROM: fine_grained_config.yaml
# TO: ../config/backtest_config.yaml
```

Step 4: Clean up old config directories (after verification)
```bash
# After tests pass:
rm -rf etf_rotation_system/01_横截面建设/config/
rm -f etf_rotation_system/02_因子筛选/*config*.yaml
rm -f etf_rotation_system/03_vbt回测/*config*.yaml
rm -rf etf_rotation_system/04_精细策略/config/
```

---

## PHASE 4: INTERNAL SCRIPT ORGANIZATION (5 MINUTES)
=================================================

### 4.1 Current State

Test/debug scripts scattered:
├── etf_rotation_system/test_full_pipeline.py
├── etf_rotation_system/01_横截面建设/test_equivalence.py
├── etf_rotation_system/03_vbt回测/backtest_manager.py            (UNUSED)
├── etf_rotation_system/03_vbt回测/simple_parallel_backtest_engine.py
├── factor_system/factor_generation/scripts/debug/debug_timeframes.py
├── factor_system/factor_generation/scripts/legacy/multi_tf_vbt_detector.py
└── tests/ (with ad-hoc development tests)

### 4.2 Reorganization

Move to scripts/:
```bash
mkdir -p scripts/etf_rotation/
mkdir -p scripts/factor_system/

# Move ETF rotation test scripts
mv etf_rotation_system/test_full_pipeline.py scripts/etf_rotation/test_full_pipeline.py
mv etf_rotation_system/01_横截面建设/test_equivalence.py scripts/etf_rotation/test_equivalence.py

# Move factor system debug scripts
mv factor_system/factor_generation/scripts/debug/debug_timeframes.py scripts/factor_system/

# Delete unused backtest manager
rm etf_rotation_system/03_vbt回测/backtest_manager.py
rm etf_rotation_system/03_vbt回测/simple_parallel_backtest_engine.py
```

Organize tests/:
```bash
# Consolidate development tests
mkdir -p tests/etf_rotation/
mkdir -p tests/factor_system/

# Tests should follow structure:
tests/
├── conftest.py                 # Shared fixtures
├── unit/                       # Unit tests
│   └── test_*.py
├── integration/                # Integration tests
│   └── test_*.py
├── etf_rotation/               # ETF system tests
│   ├── test_panel_generation.py
│   ├── test_factor_screening.py
│   └── test_backtest.py
└── factor_system/              # Factor engine tests
    └── test_*.py
```

---

## PHASE 5: CODE QUALITY IMPROVEMENTS (30 MINUTES)
==============================================

### 5.1 Delete Unused Files

Identified unused/redundant code:
```bash
# In etf_rotation_system/03_vbt回测/
rm backtest_manager.py                  # Unused manager class
rm simple_parallel_backtest_engine.py   # Simple version, now using configurable

# In factor_system/
rm factor_system/factor_generation/scripts/legacy/multi_tf_vbt_detector.py  # Legacy

# In etf_download_manager/ (consolidate configs)
rm etf_download_manager/config/etf_config.yaml      # Merge into .py version
rm etf_download_manager/config/etf_config_manager.py  # Keep only one version
rm etf_download_manager/config/etf_config_standalone.py
```

### 5.2 Code Refactoring Tasks (TODO)

[ ] Split engine.py (2847 lines) into modules:
    ├── engine_base.py (<500 lines)
    ├── engine_cache.py (<500 lines)
    └── engine_consistency.py (<500 lines)

[ ] Extract config management:
    ├── Create ConfigManager class
    ├── Merge ConfigLoader implementations
    └── Single source of truth for config

[ ] Consolidate factor calculation:
    ├── Deduplicate factor_engine/ and factor_generation/
    └── Use single factor registry

---

## PHASE 6: VERIFICATION & TESTING (10 MINUTES)
===============================================

### 6.1 Post-Cleanup Tests

Run after cleanup to ensure nothing broke:

```bash
# 1. Test core pipeline still works
cd /Users/zhangshenshen/深度量化0927
python3 etf_rotation_system/01_横截面建设/generate_panel_refactored.py \
  --data-dir raw/ETF/daily \
  --output-dir etf_rotation_system/data/results/panels \
  --config etf_rotation_system/config/factor_panel_config.yaml

# 2. Test factor screening
python3 etf_rotation_system/02_因子筛选/run_etf_cross_section_configurable.py \
  --config etf_rotation_system/config/screening_config.yaml

# 3. Test backtest
python3 etf_rotation_system/03_vbt回测/large_scale_backtest_50k.py

# 4. Run unit tests (if exist)
pytest tests/unit/ -v

# 5. Run integration tests
pytest tests/integration/ -v
```

### 6.2 Success Criteria

✅ All pipeline steps execute without errors
✅ No "File not found" errors (path changes work)
✅ Output files generated in expected locations
✅ Top #1 Sharpe >= 0.60 (baseline check)
✅ Factor counts remain stable (48 → 15)

### 6.3 Rollback Plan

If tests fail:
```bash
# Git stash and restore
git status
git checkout -- .  # Restore everything

# Then fix issues one at a time
```

---

## PHASE 7: DOCUMENTATION UPDATE (15 MINUTES)
===========================================

### 7.1 Update README.md

Add section on cleanup status:
```markdown
## Project Structure (After Cleanup)

[Add updated directory tree]

## Configuration

All configuration files are centralized in:
- `etf_rotation_system/config/`

Configuration precedence:
1. Command-line arguments
2. YAML config files
3. Python dataclasses (defaults)
4. Environment variables
```

### 7.2 Update CLAUDE.md

Add section on cleanup completeness:
```markdown
## Code Maintenance Status

Last cleanup: 2025-10-22

Metrics:
- Python files: 265 → 245
- Config files: 80+ → 15
- Documentation files: 25+ → 5
- Max file size: 2847 → <1000 lines (WIP)
- Tech debt: Medium → Low
```

### 7.3 Create CLEANUP_CHECKLIST.md

```markdown
# Project Cleanup Checklist

Date: 2025-10-22

## Completed ✅
- [x] Delete orphaned scripts (8 files)
- [x] Archive experimental reports
- [x] Verify production pipeline works
- [x] Update documentation

## In Progress 🔄
- [ ] Consolidate config files
- [ ] Reorganize test structure
- [ ] Update import paths

## TODO 📋
- [ ] Refactor large files (engine.py)
- [ ] Merge duplicate modules
- [ ] Update CI/CD pipelines
```

---

## CLEANUP COMMANDS (QUICK REFERENCE)
====================================

### Minimal Cleanup (5 MINUTES)
```bash
cd /Users/zhangshenshen/深度量化0927

# Remove orphaned scripts
rm -f corrected_net_analysis.py debug_static_checker.py diagnose_etf_issues.py \
      net_return_analysis.py simple_factor_screen.py test_rotation_factors.py \
      turnover_penalty_optimizer.py verify_single_combo.py

# Remove backups
rm -f *.tar.gz

# Remove experimental reports
rm -f AUDIT_* BACKTEST_* BASELINE_* CLEANUP_* COMPLETE_* CRITICAL_* \
      ETF_FULL_* ETF_ROTATION_OPTIMIZATION_* ETF_ROTATION_SYSTEM_* \
      EXECUTIVE_* FACTOR_SCREENING_* FIX_* LARGE_SCALE_* LOG_* \
      OPTIMIZATION_* PHASE1_* VBT_* VERIFICATION_* \
      config_*.json functional_*.json performance_*.json

echo "✅ Minimal cleanup complete"
```

### Full Cleanup (20 MINUTES)
```bash
# Execute minimal cleanup first
# ... (minimal commands above)

# Create unified config directory
mkdir -p etf_rotation_system/config

# Consolidate configs
cp etf_rotation_system/01_横截面建设/config/factor_panel_config.yaml \
   etf_rotation_system/config/
cp etf_rotation_system/02_因子筛选/optimized_screening_config.yaml \
   etf_rotation_system/config/screening_config.yaml

# Reorganize tests
mkdir -p scripts/etf_rotation/
mv etf_rotation_system/test_full_pipeline.py scripts/etf_rotation/ 2>/dev/null
mv etf_rotation_system/01_横截面建设/test_equivalence.py scripts/etf_rotation/ 2>/dev/null

# Remove unused internal scripts
rm -f etf_rotation_system/03_vbt回测/backtest_manager.py
rm -f etf_rotation_system/03_vbt回测/simple_parallel_backtest_engine.py

echo "✅ Full cleanup complete"

# Verify
git status
```

---

## VERIFICATION CHECKLIST
=======================

After cleanup, verify:

[ ] Git status shows expected deletions
[ ] All Python imports still resolve
[ ] Core pipeline runs: generate_panel_refactored.py
[ ] Factor screening runs: run_etf_cross_section_configurable.py
[ ] Backtest runs: large_scale_backtest_50k.py
[ ] Tests pass: pytest tests/ -v
[ ] No "File not found" errors
[ ] Output files in expected locations
[ ] Config files load without errors
[ ] Top #1 Sharpe >= 0.60

---

## SIZE IMPACT PROJECTION
=======================

Before:
  Total files: 265 Python
  Config files: 80+
  Markdown docs: 25+
  Project size: 33.71 MB
  Max file: 2847 lines

After:
  Total files: 245 Python (-20)
  Config files: 15 (-65)
  Markdown docs: 5 (-20)
  Project size: ~28 MB (-5.7 MB)
  Max file: 1000 lines (target)

---

## NEXT STEPS
=============

1. Execute this cleanup plan
2. Run verification tests
3. Commit cleanup to git: git commit -m "refactor: project cleanup and consolidation"
4. Generate final verification report
5. Update CI/CD pipelines if needed
6. Continue with optional refactoring (split large files)

---

Checklist generated: 2025-10-22
Ready for execution.
