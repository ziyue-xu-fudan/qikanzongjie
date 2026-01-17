# 修正 PRISMA 流程图数据逻辑

根据您确认的现有文献数 (637)，我们将反推原始检索数，确保图表逻辑严密。

## 1. 数据修正
*   **Screened (现有 Excel)**: 637
*   **Removed (RCTs)**: 85 (用户指定)
*   **Identified (原始总数)**: 637 + 85 = 722

## 2. 脚本更新 (`generate_prisma_chart.py`)
*   修改 `main()` 函数中的数字配置。
*   保持中英文双语、字体适配逻辑不变。

## 3. 执行
*   重新生成 `PRISMA_EN.png` 和 `PRISMA_CN.png`。
