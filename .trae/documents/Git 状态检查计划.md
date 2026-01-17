# 检查 Git 状态

检查当前 Git 仓库的状态，确认是否所有更改都已提交并准备好推送。

## 1. 检查状态
*   运行 `git status`。
*   预期：应该显示 "nothing to commit, working tree clean" 且 "Your branch is ahead of 'origin/main' by X commits"。

## 2. 如果有未提交的
*   执行 `git add .` 和 `git commit`。

## 3. 再次提示推送
*   如果本地超前远程，再次提示用户执行 `git push`。
