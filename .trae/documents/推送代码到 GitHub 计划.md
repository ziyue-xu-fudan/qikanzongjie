# 推送代码到用户 GitHub 仓库

协助用户将本地代码推送到指定的 GitHub 仓库。

## 1. 关联远程仓库
*   执行 `git remote add origin https://github.com/ziyue-xu-fudan/qikanzongjie.git`。
*   如果提示 `remote origin already exists`，则执行 `git remote set-url origin ...`。

## 2. 推送代码
*   尝试执行 `git push -u origin main`。
*   **预期情况**: 很可能会因为需要身份验证而挂起或失败。
*   **应对措施**: 如果失败，我会输出明确的提示，请用户在终端中手动完成这一步（输入 Token）。

## 3. 后续指引
*   推送成功后，指导用户去 Streamlit Cloud 进行部署。
