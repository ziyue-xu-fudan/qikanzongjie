# 修正 III 期文章为汇总分析

为了纠正 AI 误判或数据不准确，我们将把所有被标记为 III 期的文章类型修正为 Pooled Analysis。

## 1. 修正脚本 (`fix_phase3_to_pooled.py`)
*   读取 `Literature_Screening_List.xlsx`。
*   筛选 `Phase` 列中包含 "III" 或 "3" 的行。
*   将这些行的：
    *   `Study_Design` 更新为 **"Pooled Analysis"**。
    *   `Phase` 更新为 **"N/A"** (或者保留为空，视您的偏好。这里设为 "N/A" 以去除紫色高亮)。
*   打印出被修改的文章标题以供确认。
*   保存文件。

## 2. 执行
*   运行脚本。
*   您刷新 App 即可看到变化（紫色标签消失，变为蓝色 Pooled Analysis 标签）。
