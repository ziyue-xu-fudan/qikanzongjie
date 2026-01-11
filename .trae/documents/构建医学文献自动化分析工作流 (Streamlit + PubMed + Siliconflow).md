我将为您构建一个基于 **Streamlit** 的可视化文献分析工作流应用。该应用将集成 PubMed 检索、大模型分析（Siliconflow）和数据可视化功能。

### 🛠️ 技术方案

1.  **界面框架**: 使用 `Streamlit` 构建 Web 界面，提供实时进度条、日志显示和交互式图表。
2.  **PubMed 检索**: 使用 NCBI E-utilities API 根据 PMID 自动补全摘要。
3.  **AI 分析**: 调用 Siliconflow API (OpenAI 兼容接口) 对摘要进行深度分析。
4.  **数据处理**: 使用 Pandas 处理 Excel 数据流。

### 📅 实施步骤

#### 1. 环境准备
- 创建 `requirements.txt`，添加必要依赖：`streamlit`, `pandas`, `openpyxl`, `requests`, `plotly`, `openai`。

#### 2. 核心逻辑开发 (`paper_workflow.py`)
- **PubMed 模块**: 实现 `fetch_abstracts(pmids)` 函数，支持批量/并发获取摘要，处理 API 速率限制。
- **AI 分析模块**: 实现 `analyze_abstract(text)` 函数，设计特定的 Prompt 以提取：
    - 研究方式 (Research Design)
    - 聚焦疾病 (Focused Disease)
    - 目标人群 (Target Population)
    - 研究团队国家 (Research Team Country)
- **数据流控制器**: 负责读取 Excel，串联检索和分析步骤，并保存结果。

#### 3. 可视化界面开发 (`app.py`)
- **侧边栏**: 输入 Siliconflow API Key，选择要处理的文件（默认加载您指定的4个文件）。
- **主控制台**:
    - **状态面板**: 显示当前正在处理的文件、PMID 和 API 响应状态。
    - **进度条**: 实时显示总体进度和单文件进度。
    - **错误日志**: 捕获并展示 API 调用失败或解析错误，支持重试建议。
- **结果仪表盘**:
    - 处理完成后自动生成图表：
        - 🌍 研究团队国家分布地图/柱状图
        - 🦠 聚焦疾病词云或频次图
        - 👥 目标人群特征分布
    - 支持下载处理后的 Excel 文件（包含 AI 提取的字段）。

#### 4. 运行与验证
- 启动 Streamlit 应用。
- 测试整个流程，确保从读取文件到生成图表的链路畅通。

请确认是否开始执行此计划？（确认后我将首先创建环境配置和核心处理脚本）