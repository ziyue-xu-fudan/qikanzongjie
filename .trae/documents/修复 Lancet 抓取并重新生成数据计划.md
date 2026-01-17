# 修复 Lancet 抓取问题并重新运行

针对 Lancet 抓取结果为 0 的问题，我们将调整检索策略，放宽结构化摘要的匹配条件，同时保持对 RCT 的严格排除。

## 1. 修改脚本 `fetch_non_rct_cancer_papers.py`

*   **保留** NEJM, JAMA, BMJ 的成功结果（或者如果之前保存了，这次只跑 Lancet 也行，但为了完整性建议重新跑一次合并版）。
*   **修改 Lancet Query**:
    *   **旧策略**: `AND ("Methods" AND "Findings" AND "Interpretation")` (太严，可能漏掉索引不全的)
    *   **新策略**: `AND ("Lancet"[Journal]) AND ("Background"[Title/Abstract] OR "Methods"[Title/Abstract] OR "Findings"[Title/Abstract])` (只要出现任意一个标准标题词即可，大幅放宽)
    *   **排除条件保持不变**: 依然严格排除 `RCT`, `Phase 3`, `random`, `blind`。

## 2. 执行与验证
*   运行修改后的脚本。
*   检查 Lancet 的抓取数量是否正常（预期应有数十篇）。
*   覆盖保存 `Top4_NonRCT_Cancer_Papers_2023_2026.xlsx`。

## 3. 交付
更新后的 Excel 表格。
