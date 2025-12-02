<!-- ALLOW-MD -->
# Markdown 新建限制政策

本仓库启用了 Git pre-commit 钩子，防止 LLM 代理随意生成无意义的 Markdown 文档。

## 政策规则

✅ **允许**
- 在 `docs/` 或任何含 `docs/` 的目录下新建 `.md`（需包含允许标记）
- 编辑已存在的 `.md` 文件
- 编辑根目录 `README.md`

❌ **拦截**
- 在非 `docs/` 目录新建 `.md`
- 在 `docs/` 下新建 `.md` 但无允许标记

## 如何新增文档

在 `docs/` 目录新建 `.md` 时，必须在**文档前 20 行内**包含任一允许标记：

```markdown
<!-- ALLOW-MD -->
# 我的新文档

文档正文...
```

或

```yaml
[ALLOW-MD]

# 我的新文档
```

或

```yaml
ALLOW_MD: true

# 我的新文档
```

## 安装/卸载钩子

安装（仅需一次）：
```bash
bash scripts/install_git_hooks.sh
```

卸载：
```bash
rm .git/hooks/pre-commit
```

## 测试钩子

尝试在根目录新建 test.md：
```bash
echo "# Test" > test.md
git add test.md
git commit -m "test"
# 应该被拦截：❌ 拦截：禁止在非 docs/ 目录新增 Markdown
```

在 docs/ 下新建有标记的文档应该通过：
```bash
echo -e "<!-- ALLOW-MD -->\n# Test" > docs/test.md
git add docs/test.md
git commit -m "test"
# 应该成功 ✅
```
