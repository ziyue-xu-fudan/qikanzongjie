# 在 App V2 中展示结构化 AI 数据

为了充分利用新提取的深度数据，我们将全面升级 App V2 的展示层。

## 1. UI 升级
*   **Meta Tags**: 在标题下方增加一行彩色标签：
    *   🏷️ `Cancer Type` (例如：Lung Cancer)
    *   🔖 `Study Design` (例如：Cohort)
    *   ⚡ `Phase` (例如：Phase II)
*   **Highlights**: 在 Summary 下方增加一个 "✨ Key Highlights" 区域，列出 3 个 bullet points。

## 2. 交互升级
*   **侧边栏过滤器**: 增加 `Cancer Type` 多选框。这允许用户只看特定癌肿的文章（例如只看 Lung Cancer）。

## 3. 执行
*   修改 `app_v2.py`。
