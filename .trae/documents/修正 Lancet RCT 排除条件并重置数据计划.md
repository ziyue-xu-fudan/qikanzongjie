# 修正检索式并重新发布到 Streamlit

我们将修正 Lancet 的检索式（加入 RCT 排除），重新抓取数据，并更新到 Streamlit 应用中。

## 1. 修改 `fetch_non_rct_cancer_papers.py`
*   **Lancet Query 更新**: 在排除条件中显式加入 `"Randomized Controlled Trial"[Publication Type]`。

## 2. 重新抓取数据
*   运行 `fetch_non_rct_cancer_papers.py`。
*   生成更新后的 `Top4_NonRCT_Cancer_Papers_2023_2026.xlsx`。

## 3. 更新筛选列表
*   运行 `create_screening_list.py`。
*   生成新的 `Literature_Screening_List.xlsx`。

## 4. 验证 Streamlit
*   由于 Streamlit 直接读取 Excel 文件，数据更新后，您只需在浏览器刷新页面即可看到新数据。
