# 生成中英文双语 PRISMA 流程图

我们将根据您的具体数据要求（Total=619, Excluded=85），生成两张风格统一但语言不同的流程图。

## 1. 数据配置
*   **Records Identified**: 619
*   **Records Removed (RCTs)**: 85
*   **Records Screened**: 534 (= 619 - 85)
*   **Included**: *动态读取* (或者如果您想硬编码，我可以设为 TBD)

## 2. 脚本修改 (`generate_prisma_chart.py`)
*   **双语支持**: 封装绘图函数 `draw_prisma(lang='en')`。
*   **字体适配**:
    *   English: `Arial`
    *   Chinese: `Heiti TC` (macOS 默认黑体) 或 `SimHei` (如有)
*   **文本映射**: 建立中英文对照字典。
    *   "Records identified..." -> "PubMed 检索检出文献"
    *   "Records removed..." -> "排除文献 (RCT/三期临床)"
    *   "Records screened..." -> "题目/摘要及 AI 筛选"
    *   "Studies included..." -> "最终纳入定性分析"

## 3. 执行与交付
*   运行脚本。
*   生成 `PRISMA_EN.png` 和 `PRISMA_CN.png`。
