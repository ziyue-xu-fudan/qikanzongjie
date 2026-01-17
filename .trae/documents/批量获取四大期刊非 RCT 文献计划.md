# 使用指定 API Key 批量获取四大期刊非 RCT 文献

我们将利用您提供的 PubMed API Key 和邮箱，高效抓取 NEJM, Lancet, JAMA, BMJ 在 2023-2026 年间的非 RCT 肿瘤研究文献。

## 1. 核心脚本配置 (`fetch_non_rct_cancer_papers.py`)

*   **API 配置**:
    *   `Entrez.email = "ziyuexu20@fudan.edu.cn"`
    *   `Entrez.api_key = "e3674f393e05e49020299c745b81574ea707"`
*   **检索式 (Hardcoded)**:
    1.  **NEJM**: `... NOT ("Randomized Controlled Trial"[Publication Type] OR ... OR "blind"[Title/Abstract])`
    2.  **Lancet**: `... NOT ("Randomized Controlled Trial"[Publication Type] OR ... OR "blind"[Title/Abstract])`
    3.  **JAMA**: `... NOT ("Randomized Controlled Trial"[Publication Type] OR ... OR "blind"[Title/Abstract])`
    4.  **BMJ**: `... NOT ("Randomized Controlled Trial"[Publication Type] OR ... OR "blind"[Title/Abstract])`

## 2. 执行流程

1.  **定义 Queries**: 在代码中直接写入这四个复杂的检索字符串。
2.  **API 请求**: 遍历 4 个 Query，调用 `esearch` 获取 ID 列表，再调用 `efetch` 获取详细元数据。
3.  **数据解析**: 提取 Title, Journal, PubDate, DOI, Abstract。
4.  **合并导出**: 将所有结果合并为一个 DataFrame，并导出为 `Top4_NonRCT_Cancer_Papers_2023_2026.xlsx`。

## 3. 结果交付
您将获得一个包含所有符合条件文献的 Excel 表格，可以直接用于后续的筛选和阅读。
