import pandas as pd
from Bio import Entrez
import time
import concurrent.futures
import json
import os
from openai import OpenAI

# -----------------------------------------------------------------------------
# 配置
# -----------------------------------------------------------------------------
Entrez.email = "your.email@example.com"  # 请替换为有效邮箱

API_KEYS = [
    "sk-37c1617db0da456d8491e1094e3f6ae3",
    "sk-82a00766192049fc91da7edbca74bfd2",
    "sk-c69f18b962d54e44b14298f079bc4c66",
    "sk-d98eb5841a0b4e6c9985b72b4106c74c"
]

FILE_PATH = "/Users/ziyuexu/Documents/trae_projects/paper1/multi_journal_analysis_report.xlsx"
MODEL_NAME = "deepseek-chat"

# 完整的分析 Prompt (合并了 Design, Timing, Disease)
ANALYSIS_PROMPT_TEMPLATE = """
请分析以下医学文献摘要，并提取以下五个关键信息。
请严格按照 JSON 格式返回，不要包含 Markdown 格式标记（如 ```json）。
如果无法提取某个字段，请填写 "Unknown"。

摘要内容:
{abstract}

需要提取的字段:
1. research_design
   - Options: [Randomized Controlled Trial, Cohort Study, Case-Control Study, Cross-sectional Study, Systematic Review, Meta-analysis, Case Report, Animal Study, In Vitro Study, Narrative Review, Clinical Observation, Diagnostic Accuracy Study, Time Series Analysis, Modeling Study, Economic Evaluation, Qualitative Study, Guideline/Consensus, Study Protocol]
   - If none match, use "Other".

2. study_timing
   - Options: [Retrospective, Prospective, Cross-sectional, Ambispective, Simulation/Model-based, Longitudinal]
   - If not applicable, use "Not Applicable".

3. focused_disease_system
   - Options: [Cardiovascular, Respiratory, Nervous, Digestive, Endocrine, Immune, Musculoskeletal, Urinary, Reproductive, Integumentary, Oncology, Infectious Disease, General Health/System, Other]

4. focused_disease
   - Specific disease name with ICD-10 if possible.
   - If general health/policy, use "Not Applicable".

5. research_team_country
   - Country name.

JSON 格式示例:
{{
    "research_design": "Cohort Study",
    "study_timing": "Prospective",
    "focused_disease_system": "Cardiovascular",
    "focused_disease": "Hypertension (I10)",
    "research_team_country": "USA"
}}
"""

# -----------------------------------------------------------------------------
# 辅助函数
# -----------------------------------------------------------------------------
def fetch_abstract_from_pubmed(pmid):
    """根据 PMID 从 PubMed 获取摘要"""
    if not pmid or str(pmid) == 'nan':
        return None
    try:
        handle = Entrez.efetch(db="pubmed", id=str(pmid), retmode="xml")
        records = Entrez.read(handle)
        handle.close()
        
        if not records or 'PubmedArticle' not in records:
            return None
            
        article = records['PubmedArticle'][0]['MedlineCitation']['Article']
        if 'Abstract' in article and 'AbstractText' in article['Abstract']:
            abstract_parts = article['Abstract']['AbstractText']
            # AbstractText 可能是一个列表（分段摘要）或字符串
            if isinstance(abstract_parts, list):
                return " ".join([str(part) for part in abstract_parts])
            return str(abstract_parts)
        else:
            return "No Abstract Available" # 确实没有摘要
    except Exception as e:
        print(f"Error fetching PMID {pmid}: {e}")
        return None

def get_client(index):
    api_key = API_KEYS[index % len(API_KEYS)]
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

def analyze_abstract(args):
    """分析单个摘要"""
    index, abstract, thread_idx = args
    if not abstract or len(abstract) < 20 or abstract == "No Abstract Available":
        return index, None

    client = get_client(thread_idx)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": ANALYSIS_PROMPT_TEMPLATE.format(abstract=abstract[:3000])}],
            temperature=0.1,
            response_format={'type': 'json_object'}
        )
        content = response.choices[0].message.content
        return index, json.loads(content)
    except Exception as e:
        print(f"AI Analysis failed for row {index}: {e}")
        return index, None

# -----------------------------------------------------------------------------
# 主逻辑
# -----------------------------------------------------------------------------
def main():
    print(f"📂 Reading {FILE_PATH}...")
    try:
        df = pd.read_excel(FILE_PATH, engine='openpyxl')
    except Exception as e:
        print(f"Fatal Error: {e}")
        return

    # 1. 找出需要重新抓取的行
    mask = (df['Abstract'].isnull()) | (df['Abstract'] == '') | (df['Abstract'] == 'Error') | (df['Abstract'].str.len() < 50)
    target_indices = df[mask].index.tolist()
    
    print(f"🔍 Found {len(target_indices)} rows with missing abstracts.")
    if len(target_indices) == 0:
        print("All abstracts look good.")
        return

    # 2. 批量抓取摘要 (串行或小并发，避免 PubMed 封禁)
    print("🌐 Fetching abstracts from PubMed...")
    fetched_count = 0
    rows_to_analyze = [] # (index, abstract)

    for i, idx in enumerate(target_indices):
        pmid = df.loc[idx, 'PMID']
        print(f"[{i+1}/{len(target_indices)}] Fetching PMID: {pmid}...")
        
        abstract = fetch_abstract_from_pubmed(pmid)
        if abstract:
            df.at[idx, 'Abstract'] = abstract
            if abstract != "No Abstract Available":
                fetched_count += 1
                rows_to_analyze.append((idx, abstract))
                print(f"  ✅ Got abstract ({len(abstract)} chars)")
                print(f"  📜 Preview: {abstract[:200]}...") # 打印预览
            else:
                print("  ⚠️ No abstract available on PubMed")
        else:
            print("  ❌ Fetch failed")
        
        time.sleep(0.5) # 礼貌延时

    print(f"📊 Fetched {fetched_count} new abstracts.")

    if not rows_to_analyze:
        print("No new abstracts to analyze.")
        # 即使没有新分析，也要保存抓取结果（比如 No Abstract Available）
        df.to_excel(FILE_PATH, index=False)
        return

    # 3. 对新抓取的摘要进行 AI 分析
    print(f"🧠 Analyzing {len(rows_to_analyze)} new abstracts with AI...")
    
    tasks = []
    for i, (idx, abstract) in enumerate(rows_to_analyze):
        tasks.append((idx, abstract, i))

    analyzed_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_idx = {executor.submit(analyze_abstract, task): task[0] for task in tasks}
        
        for future in concurrent.futures.as_completed(future_to_idx):
            idx, result = future.result()
            if result:
                df.at[idx, 'Research Design'] = result.get('research_design', 'Unknown')
                df.at[idx, 'Study Timing'] = result.get('study_timing', 'Unknown')
                df.at[idx, 'Focused Disease System'] = result.get('focused_disease_system', 'Unknown')
                df.at[idx, 'Focused Disease'] = result.get('focused_disease', 'Unknown')
                df.at[idx, 'Research Team Country'] = result.get('research_team_country', 'Unknown')
                analyzed_count += 1
                print(f"  ✅ Analyzed row {idx}")

    # 4. 保存
    backup_path = FILE_PATH.replace(".xlsx", "_backup_refetch.xlsx")
    df.to_excel(backup_path, index=False)
    df.to_excel(FILE_PATH, index=False)
    print(f"🚀 Updated file saved! ({analyzed_count} rows re-analyzed)")

if __name__ == "__main__":
    main()
