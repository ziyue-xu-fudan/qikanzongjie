# 彻底修复 Research Design 中的 "Error" 数据计划 (实时保存版)

根据您的指示，我们将使用**串行处理**模式，利用您提供的 API Key，彻底修复 `Research Design` 中剩余的 19 条 "Error" 数据，并确保**每处理完一条就保存一次**。

## 1. 目标锁定
*   **目标**: 修复 `Research Design` 为 "Error" 的所有记录（共 19 条）。
*   **核心要求**: 串行处理、使用指定 Key、**每次更新完数据立即保存**。

## 2. 实施方案
创建一个专用的 Python 脚本 `fix_design_errors.py`，执行以下逻辑：
1.  **精准筛选**: 读取 `multi_journal_analysis_report.xlsx`，提取所有 `Research Design == 'Error'` 的行。
2.  **串行循环**: 
    *   遍历这 19 条数据。
    *   轮换使用 4 个 API Key。
    *   调用 DeepSeek API 进行重新分类。
    *   **关键步骤**: 每成功获取一个结果，立即更新 DataFrame 并调用 `df.to_excel()` 保存文件。这意味着即使程序中途意外中断，已处理的数据也不会丢失。
3.  **完整分类体系**: 使用包含标准类型（RCT, Cohort 等）和新增类型（Modeling, Time Series 等）的完整 Prompt。

## 3. 验证
*   运行脚本后，再次打印 `Research Design` 的统计分布，确保 "Error" 归零。
*   重启可视化应用 `app_viz.py` 展示最终的完美数据。