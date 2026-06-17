# GitHub 上传指南

## 1. 本地检查

```bash
cd E:/Algorithm/MedAlign-RLBench
git status
```

确认仓库中不包含以下内容：

- `.env`
- Hugging Face token
- SSH 地址、密码或私钥
- `data/`
- `runs/`
- `outputs/`
- `*.safetensors`
- 完整预测 JSONL

可以用 `rg` 按需搜索 Hugging Face token、SSH 地址、密码、私钥等关键词，确认没有真实凭据进入仓库。

```bash
rg -n "<your-sensitive-keyword>" .
```

## 2. 初始化仓库

```bash
cd E:/Algorithm/MedAlign-RLBench
git init
git add .
git commit -m "Initial public release"
```

## 3. 创建 GitHub 仓库

在 GitHub 网页端新建仓库，例如：

```text
MedAlign-RLBench
```

建议不要勾选自动生成 README，因为本地已经有 README。

## 4. 推送

把 GitHub 页面给出的远程地址替换到下面命令：

```bash
git branch -M main
git remote add origin https://github.com/<your-name>/MedAlign-RLBench.git
git push -u origin main
```

## 5. 开启 GitHub Pages

如果要展示项目主页：

1. 打开仓库 Settings。
2. 进入 Pages。
3. Source 选择 `Deploy from a branch`。
4. Branch 选择 `main`，目录选择 `/root`。
5. 保存后访问 GitHub 生成的 Pages 地址。

本项目根目录已经包含 `index.html`，可以直接作为静态主页。
