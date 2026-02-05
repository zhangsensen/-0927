# 策略封板指南 (Sealing Guidelines)

**版本**: v1.0  
**日期**: 2025-12-16  
**适用范围**: 所有 v3.x+ 策略封板

---

## 📋 封板目的

策略封板是为了**冻结某一时刻的完整策略配置和代码**，确保：
1. **可复现性**: 任何人拿到封板包能完全复现结果
2. **可追溯性**: 清楚记录策略演进历史
3. **可交付性**: 生产环境可直接使用
4. **轻量化**: 封板包体积合理（~15-30MB）

---

## 🚫 封板禁止项（CRITICAL）

### ❌ 绝对禁止包含

| 项目 | 大小 | 原因 | 说明 |
|------|------|------|------|
| **`.venv/`** | ~1.2GB | 虚拟环境 | 用户本地生成 |
| **`venv/`** | ~1.2GB | 虚拟环境 | 用户本地生成 |
| **`__pycache__/`** | ~10MB | Python 缓存 | 自动生成 |
| **`*.pyc`** | ~1-5MB | 编译缓存 | 自动生成 |
| **`results/`** | ~100MB+ | 运行结果 | 只保留关键结果到 artifacts/ |
| **`.cache/`** | ~50MB+ | 数据缓存 | 自动生成 |
| **`.pytest_cache/`** | ~5MB | 测试缓存 | 自动生成 |

### ✅ 必须包含

| 项目 | 大小 | 说明 |
|------|------|------|
| **`pyproject.toml`** | ~5KB | 依赖定义 |
| **`uv.lock`** | ~50KB | 精确版本锁定 |
| **`configs/`** | ~100KB | 配置文件 |
| **`src/`** | ~5MB | 源代码 |
| **`scripts/`** | ~2MB | 运行脚本 |
| **`artifacts/`** | ~5MB | 关键结果文件 |
| **`CHECKSUMS.sha256`** | ~1KB | 完整性校验 |

---

## 📦 标准封板流程

### Step 1: 准备封板目录
```bash
cd /home/sensen/dev/projects/-0927/sealed_strategies
mkdir -p v3.X_YYYYMMDD/{locked,artifacts}
cd v3.X_YYYYMMDD
```

### Step 2: 复制核心文件
```bash
# 进入主项目
cd /home/sensen/dev/projects/-0927

# 复制到封板目录
cp -r src/ sealed_strategies/v3.X_YYYYMMDD/locked/
cp -r scripts/ sealed_strategies/v3.X_YYYYMMDD/locked/
cp -r configs/ sealed_strategies/v3.X_YYYYMMDD/locked/
cp -r docs/ sealed_strategies/v3.X_YYYYMMDD/locked/
cp pyproject.toml sealed_strategies/v3.X_YYYYMMDD/locked/
cp uv.lock sealed_strategies/v3.X_YYYYMMDD/locked/
cp Makefile sealed_strategies/v3.X_YYYYMMDD/locked/
```

### Step 3: 复制关键结果
```bash
# 只复制关键结果文件
cp results/production_pack_*/production_candidates.csv \
   sealed_strategies/v3.X_YYYYMMDD/artifacts/
```

### Step 4: 清理虚拟环境（CRITICAL）
```bash
cd sealed_strategies/v3.X_YYYYMMDD/locked

# 删除虚拟环境
rm -rf .venv venv env ENV

# 删除缓存
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null
rm -rf .cache

# 删除结果目录（如果不小心复制了）
rm -rf results/

# 验证清理
du -sh .  # 应该在 15-30MB 之间
```

### Step 5: 创建 .gitignore
```bash
cat > .gitignore << 'EOF'
# Virtual Environments (DO NOT COMMIT TO SEALED RELEASES)
.venv/
venv/
env/
ENV/

# Python Cache
__pycache__/
*.py[cod]
*$py.class
*.so

# Testing
.pytest_cache/
.coverage

# Project Specific
results/
.cache/
*.log
EOF
```

### Step 6: 生成文档
```bash
cd sealed_strategies/v3.X_YYYYMMDD

# 创建 README.md（项目说明）
# 创建 REPRODUCE.md（复现指南）
# 创建 RELEASE_NOTES.md（版本说明）
# 创建 VERIFICATION_REPORT.md（验证报告）
```

### Step 7: 生成校验文件
```bash
cd sealed_strategies/v3.X_YYYYMMDD

# 生成 SHA256 校验和
find . -type f ! -name "CHECKSUMS.sha256" -exec sha256sum {} \; | \
  sort > CHECKSUMS.sha256

# 生成清单
find . -type f ! -name "MANIFEST.json" | sort | \
  python3 -c "import sys, json; print(json.dumps([l.strip() for l in sys.stdin], indent=2))" \
  > MANIFEST.json
```

### Step 8: 最终验证
```bash
cd sealed_strategies/v3.X_YYYYMMDD

# 检查大小（应该 < 50MB）
du -sh .

# 检查是否有虚拟环境（应该为空）
find . -type d -name ".venv" -o -name "venv"

# 验证校验和
sha256sum -c CHECKSUMS.sha256

# 输出封板信息
echo "✅ 封板完成！"
echo "版本: v3.X_YYYYMMDD"
echo "大小: $(du -sh . | cut -f1)"
echo "文件数: $(find . -type f | wc -l)"
```

---

## 🔍 封板检查清单

### 必检项（Before Sealing）

- [ ] **删除了所有虚拟环境** (`.venv/`, `venv/`)
- [ ] **删除了所有缓存** (`__pycache__/`, `.pytest_cache/`, `.cache/`)
- [ ] **删除了结果目录** (`results/`，除非特别需要）
- [ ] **包含了 `pyproject.toml`** 和 **`uv.lock`**
- [ ] **包含了 `.gitignore`** 并正确配置
- [ ] **大小合理** (< 50MB，理想 15-30MB)
- [ ] **创建了完整文档** (README, REPRODUCE, RELEASE_NOTES)
- [ ] **生成了校验文件** (CHECKSUMS.sha256, MANIFEST.json)

### 验证项（After Sealing）

- [ ] **校验和通过**: `sha256sum -c CHECKSUMS.sha256`
- [ ] **环境可复现**: `uv sync --dev` 成功
- [ ] **代码可运行**: 至少跑一个简单测试
- [ ] **文档完整**: 所有说明清晰可读

---

## 📊 典型封板包大小参考

| 版本 | 大小 | 说明 |
|------|------|------|
| v3.1 | 13MB | ✅ 标准（无虚拟环境） |
| v3.2 | 14MB | ✅ 标准（无虚拟环境） |
| v3.3 | 23MB | ✅ 包含较多脚本 |
| v3.4 | 16MB | ✅ 清理后（原 1.2GB） |

> ⚠️ **如果封板包 > 100MB，必须检查是否误包含了虚拟环境或缓存！**

---

## 🛠️ 快速清理脚本

如果发现已封板的版本包含虚拟环境，使用此脚本快速清理：

```bash
#!/bin/bash
# 文件名: cleanup_sealed_version.sh
# 用法: ./cleanup_sealed_version.sh v3.X_YYYYMMDD

VERSION=$1
if [ -z "$VERSION" ]; then
  echo "Usage: $0 <version>"
  exit 1
fi

cd /home/sensen/dev/projects/-0927/sealed_strategies/$VERSION/locked

echo "🔍 清理前大小: $(du -sh .. | cut -f1)"

# 删除虚拟环境
rm -rf .venv venv env ENV

# 删除缓存
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null
rm -rf .cache

# 删除结果（如果有）
rm -rf results/

echo "✅ 清理后大小: $(du -sh .. | cut -f1)"

# 重新生成校验和
cd ..
find . -type f ! -name "CHECKSUMS.sha256" -exec sha256sum {} \; | \
  sort > CHECKSUMS.sha256
echo "✅ 校验和已更新"
```

---

## 📝 使用示例

### 用户拿到封板包后的操作

```bash
# 1. 解压（如果是压缩包）
tar -xzf v3.X_YYYYMMDD.tar.gz
cd v3.X_YYYYMMDD

# 2. 验证完整性
sha256sum -c CHECKSUMS.sha256

# 3. 进入工作目录
cd locked

# 4. 安装依赖（自动创建 .venv）
uv sync --dev

# 5. 验证环境
uv run python -c "import pandas, numpy, backtrader; print('✅ OK')"

# 6. 运行回测（示例）
uv run python scripts/batch_bt_backtest.py
```

**关键点**: 用户**不需要**手动创建虚拟环境，`uv sync` 会自动处理一切！

---

## ⚠️ 常见错误

### 错误 1: 封板包过大（>100MB）
**原因**: 包含了虚拟环境或缓存  
**解决**: 
```bash
cd locked
rm -rf .venv venv __pycache__ .pytest_cache .cache results/
```

### 错误 2: 用户无法复现环境
**原因**: 缺少 `pyproject.toml` 或 `uv.lock`  
**解决**: 确保这两个文件都在 `locked/` 目录

### 错误 3: 校验和失败
**原因**: 文件被修改或缺失  
**解决**: 重新生成校验和
```bash
find . -type f ! -name "CHECKSUMS.sha256" -exec sha256sum {} \; | \
  sort > CHECKSUMS.sha256
```

---

## 🎯 封板哲学

> "封板包应该像配方，而不是成品。"
> 
> - 不要打包虚拟环境（1.2GB）
> - 只打包配方（pyproject.toml, uv.lock）
> - 让用户本地"烹饪"（uv sync）
> - 结果完全一致，但包轻量（15MB）

---

## 📞 支持

如有封板问题，请联系项目维护者或查阅：
- [UV 官方文档](https://github.com/astral-sh/uv)
- [Python 打包指南](https://packaging.python.org/)

---

**最后更新**: 2025-12-16  
**维护者**: AI Quant Team
