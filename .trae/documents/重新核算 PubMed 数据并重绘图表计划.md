# 重新核算 PubMed 检索数据并重绘流程图

为了确保流程图数据绝对准确，我们将重新查询 PubMed API，获取“排除前”和“排除后”的真实文献数量。

## 1. 编写计数脚本 (`count_pubmed_results.py`)
*   定义两组 Query：
    *   **Query A (Raw)**: 仅包含期刊、时间、关键词/摘要结构限制。
    *   **Query B (Filtered)**: Query A + `NOT (RCT/Phase3...)`。
*   分别对 NEJM, Lancet, JAMA, BMJ 执行 `esearch` (仅计数，不下载)。
*   汇总得到 `Total Identified` (A) 和 `Final Screened` (B)。
*   计算 `Excluded RCTs` = A - B。

## 2. 更新绘图脚本 (`generate_prisma_chart.py`)
*   将 `count_pubmed_results.py` 的输出结果硬编码进绘图脚本（或自动调用）。
*   **布局调整**: 确保箭头指向 n=637 (或最新的真实值 B)。

## 3. 执行
*   运行计数脚本。
*   运行绘图脚本。
*   交付最终一致的 PNG 图片。
