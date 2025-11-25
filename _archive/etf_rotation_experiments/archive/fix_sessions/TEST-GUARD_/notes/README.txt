临时修复会话：TEST-GUARD_
创建时间：2025-11-10 23:01:45

约束：
1. 所有临时 shell 脚本放在 /Users/zhangshenshen/深度量化0927/etf_rotation_experiments/fix_sessions/TEST-GUARD_/scripts/
2. 所有一次性回归/校验测试放在 /Users/zhangshenshen/深度量化0927/etf_rotation_experiments/fix_sessions/TEST-GUARD_/tests/
3. 修复完成后务必运行：bash scripts/cleanup_fix.sh TEST-GUARD_
4. fix_sessions/ 在提交前必须清空，否则 pre-commit 会拦截
