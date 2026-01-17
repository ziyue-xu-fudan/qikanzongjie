# DeepSeek 深度相关性筛选

利用 AI 对现有文献库进行“肿瘤相关性”的严格二分类筛选，剔除伪肿瘤研究。

## 1. 编写脚本 `ai_relevance_check.py`
*   **输入**: 读取 `Literature_Screening_List.xlsx`。
*   **Prompt**: "判断该研究的核心主题是否为肿瘤/癌症？要求：必须以肿瘤为主要研究对象。若仅提及癌症但核心为心血管/糖尿病等其他领域，判为否。返回 JSON: `{is_relevant: bool, reason: str}`"。
*   **处理**: 遍历所有文章，调用 API。
*   **输出**: 更新 Excel，将非相关文章标记为 Excluded。

## 2. 执行筛选
*   运行脚本，处理 203 篇文献。
*   统计被剔除的文章数量。

## 3. 部署更新
*   提交更新后的 Excel 到 Git。
*   提示用户执行 Push。
