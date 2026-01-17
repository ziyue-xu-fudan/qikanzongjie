# 重置 Git 仓库并准备强制推送

为了解决 Git 报错（通常是冲突或配置错误），我们将执行“完全重置”策略。

## 1. 清理环境
*   删除 `.git` 目录：`rm -rf .git`。这会清除所有旧的提交记录和错误配置，让项目回到纯净状态。

## 2. 重新初始化
*   `git init`
*   `git add .`
*   `git commit -m "Reset and Force Deploy"`
*   `git branch -M main`

## 3. 重新关联远程
*   `git remote add origin https://github.com/ziyue-xu-fudan/qikanzongjie.git`

## 4. 用户操作指引
*   提供强制推送命令：`git push -f origin main`。这将解决绝大多数“rejected”或“conflict”错误。
