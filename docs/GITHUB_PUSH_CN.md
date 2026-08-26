# GitHub 推送说明

本项目已经是独立 Git 仓库，并且已有初始 commit。

推荐仓库名：

```text
comp3710-lab2-pattern-recognition
```

## 方法一：GitHub 网页新建 repo

1. 打开 GitHub。
2. New repository。
3. Repository name 填：

```text
comp3710-lab2-pattern-recognition
```

4. 建议选择 Public，方便 demonstrator 查看。
5. 不要勾选创建 README、.gitignore、license，因为本地已经有。

创建后，在本项目目录运行：

```bash
cd "/Users/xct/Documents/New project/comp3710_lab2_demo"
git remote add origin https://github.com/Edward-xct/comp3710-lab2-pattern-recognition.git
git push -u origin main
```

如果你的 GitHub username 不是 `Edward-xct`，把 URL 里的 username 换成你的实际账号。

## 方法二：安装 gh CLI 后创建

```bash
cd "/Users/xct/Documents/New project/comp3710_lab2_demo"
gh auth login
gh repo create comp3710-lab2-pattern-recognition --public --source=. --remote=origin --push
```

## Demo 时展示

```bash
git status -sb
git log --oneline -5
git remote -v
```

确认点：

- `data/` 不在 GitHub 里。
- `outputs/` 不在 GitHub 里。
- `checkpoints/` 不在 GitHub 里。
- 代码、README、SLURM、demo 文档都在 GitHub 里。
