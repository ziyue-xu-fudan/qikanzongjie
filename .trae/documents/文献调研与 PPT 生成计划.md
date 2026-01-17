# 构建 Top 4 期刊“非典型”肿瘤研究检索式

我们将构建一个精确的 PubMed 检索式，旨在挖掘 NEJM, Lancet, JAMA, BMJ 上发表的、非三期临床试验的肿瘤学研究（侧重观察性、流行病学、真实世界证据）。

## 1. 检索式构建逻辑 (Components)

我们将检索式分为三个核心部分（Block），通过 `AND` 相连：

### A. 期刊限定 (Journal Filter)
锁定四大综合性医学顶刊：
*   *"The New England journal of medicine"[Journal]*
*   *"The Lancet"[Journal]*
*   *"JAMA"[Journal]*
*   *"BMJ (Clinical research ed.)"[Journal]*

### B. 主题限定 (Topic: Oncology)
覆盖肿瘤、癌症相关的所有主题：
*   *"Neoplasms"[MeSH Terms]* (医学主题词)
*   *"Cancer"[Title/Abstract]*
*   *"Oncology"[Title/Abstract]*
*   *"Tumor"[Title/Abstract]*

### C. 类型筛选 (Study Design Filter)
这是最关键的部分，采用 **"包含非干预"** 或 **"排除三期 RCT"** 的策略。为了确保查全率，建议采用 **"排除法" (Exclusion Strategy)**：

*   **排除 (NOT)**:
    *   *"Clinical Trial, Phase III"[Publication Type]*
    *   *"Randomized Controlled Trial"[Publication Type]* (如果您想彻底排除所有 RCT)
*   **或者 包含 (OR)** (如果您想通过白名单搜索):
    *   *"Observational Study"[Publication Type]*
    *   *"Cohort Studies"[MeSH Terms]*
    *   *"Retrospective Studies"[MeSH Terms]*
    *   *"Case-Control Studies"[MeSH Terms]*

## 2. 最终输出

我将为您提供**两个版本**的检索式：

1.  **宽泛版 (Broad Exclusion)**：仅排除三期临床，保留二期及其他所有类型（适合寻找早期探索性研究）。
2.  **精准版 (Non-Interventional)**：排除所有 RCT，专注于观察性、队列和流行病学研究（最符合"非干预"定义）。

## 3. 执行

确认计划后，我将直接生成这两个检索式供您复制使用。
