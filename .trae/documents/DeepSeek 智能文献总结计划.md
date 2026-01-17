# DeepSeek 智能文献总结与筛选升级计划

我们将利用 DeepSeek API 为 637 篇文献生成“一句话中文总结”，并将其集成到筛选应用中，实现极速初筛。

## 1. 批量总结脚本 (`batch_summarize_papers.py`)
*   **功能**:
    *   读取 `Literature_Screening_List.xlsx`。
    *   检查是否已存在 `AI_Summary` 列，若无则创建。
    *   **并发处理**: 使用 `ThreadPoolExecutor` (建议 10 线程) 加速 API 调用。
    *   **API 调用**: 使用 DeepSeek V3 (或 R1) 模型。
    *   **Prompt**: "请用中文一句话概括这篇医学论文的核心发现和研究设计（如队列、RWE）。50字以内。"
    *   **稳健性**: 遇到错误自动重试，每处理 20 篇自动保存一次 Excel。

## 2. 筛选应用升级 (`app_screening.py`)
*   **界面布局调整**:
    *   在 Title 下方、Abstract 上方，显著展示 **🤖 AI 一句话总结**。
    *   使用不同背景色（如淡蓝色）区分 AI 总结，使其一眼可见。
*   **搜索增强**: 搜索框将同时搜索 `AI_Summary` 内容，方便您搜 "队列"、"生物标志物" 等中文词。

## 3. 执行流程
1.  创建并运行 `batch_summarize_papers.py` (预计耗时 5-10 分钟)。
2.  更新 `app_screening.py` 代码。
3.  您刷新网页，即可看到带 AI 总结的增强版筛选界面。

## 4. API Key
将使用您之前提供的 DeepSeek API Key (SiliconFlow/DeepSeek 官方)。
