# 重写 Streamlit 筛选应用 (App V2)

为了彻底解决数据显示不一致和交互复杂的问题，我们将开发一个全新的 `app_v2.py`。

## 1. 核心设计原则
*   **Zero Hidden Filters**: 启动时默认显示所有数据，没有任何预设的年份或期刊过滤器。
*   **List-First**: 默认采用列表视图，这是最高效的筛选方式。
*   **Direct IO**: 直接读写 `Literature_Screening_List.xlsx`，并在界面上实时显示行数统计。

## 2. 功能模块
*   **Header**: 显示 "Total Papers: 203 | Included: X | Excluded: Y | Pending: Z"。
*   **Main Table**: 使用 `st.data_editor` (如果支持) 或自定义列表。为了更好的阅读体验，我们将使用 **自定义卡片列表**：
    *   每行一个 Card。
    *   左侧：两个大按钮 `✅` `❌`。
    *   中间：Title (高亮搜索词) + AI Summary (高亮)。
    *   下方：折叠的 Abstract。
*   **Sidebar**: 仅保留最必要的 Search 和 Status Filter (All/Pending/Done)。

## 3. 执行
*   创建 `app_v2.py`。
*   停止旧应用，启动新应用。
