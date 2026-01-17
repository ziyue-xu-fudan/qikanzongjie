# 绘制 PRISMA 风格文献筛选漏斗图

为了在 PPT 中专业且美观地展示筛选流程，我们将使用 Python 绘制一张高清的 PRISMA 风格漏斗图。

## 1. 图表设计
*   **类型**: 垂直流程图 (PRISMA Flow Diagram 风格)，但通过 Python 绘制使其更具现代感。
*   **节点**:
    1.  **Identification**: Records identified from PubMed (Top 4 Journals) [n=Total]
    2.  **Screening**: Records screened by Title/Abstract & AI [n=637]
    3.  **Eligibility**: Full-text articles assessed for eligibility [n=Selected]
    4.  **Included**: Studies included in qualitative synthesis [n=Final]
*   **美学**:
    *   配色: 专业的医学蓝 (`#2E86C1`) 和 灰色 (`#D5D8DC`)。
    *   字体: 无衬线字体 (Arial/Helvetica)，清晰易读。
    *   布局: 经典的方框 + 箭头布局，而非简单的倒三角形漏斗。

## 2. 技术实现
*   使用 `matplotlib` 和 `patches` 库进行精确绘图。
*   动态读取 `Literature_Screening_List.xlsx` 获取当前的实时筛选数据（例如您已经 Include 了多少篇）。

## 3. 交付物
`PRISMA_Flowchart.png` (高清图片，直接插入 PPT)。
