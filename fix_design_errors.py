import pandas as pd
import time
import json
import os
from openai import OpenAI

# -----------------------------------------------------------------------------
# 配置
# -----------------------------------------------------------------------------
API_KEYS = [
    "sk-37c1617db0da456d8491e1094e3f6ae3",
    "sk-82a00766192049fc91da7edbca74bfd2",
    "sk-c69f18b962d54e44b14298f079bc4c66",
    "sk-d98eb5841a0b4e6c9985b72b4106c74c"
]

FILE_PATH = "/Users/ziyuexu/Documents/trae_projects/paper1/multi_journal_analysis_report.xlsx"
MODEL_NAME = "deepseek-chat"

FIX_DESIGN_PROMPT_TEMPLATE = """
You are an expert medical research classifier.
The following abstract previously failed classification ("Error").
Please analyze it carefully and extract the Research Design.

Abstract:
{abstract}

---
ALLOWED RESEARCH DESIGNS (Select ONE):
1. Standard Types:
   [Randomized Controlled Trial, Cohort Study, Case-Control Study, Cross-sectional Study, Systematic Review, Meta-analysis, Case Report, Animal Study, In Vitro Study, Narrative Review, Clinical Observation]
2. Specific Types:
   [Diagnostic Accuracy Study, Time Series Analysis, Modeling Study, Economic Evaluation, Qualitative Study, Guideline/Consensus, Study Protocol]
3. If none match, use "Other".

---
Output strictly in JSON format:
{{
    "research_design": "..."
}}
"""

# -----------------------------------------------------------------------------
# 逻辑
# -----------------------------------------------------------------------------

def get_client(index):
    """轮询分配 Key"""
    api_key = API_KEYS[index % len(API_KEYS)]
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

def main():
    print(f"📂 Reading {FILE_PATH}...")
    try:
        df = pd.read_excel(FILE_PATH, engine='openpyxl')
    except Exception as e:
        print(f"Fatal Error: {e}")
        return

    # 筛选需要修复的行
    # 条件: Research Design 为 'Error'
    target_indices = df[df['Research Design'] == 'Error'].index.tolist()
    
    print(f"🔍 Found {len(target_indices)} 'Error' rows to fix.")
    if len(target_indices) == 0:
        print("Nothing to fix.")
        return

    # 串行处理
    print("🚀 Starting SERIAL processing...")
    
    for i, idx in enumerate(target_indices):
        row = df.loc[idx]
        abstract = row.get('Abstract', '')
        
        if not isinstance(abstract, str) or len(abstract) < 10:
            print(f"⚠️ Row {idx}: Invalid abstract, skipping.")
            continue

        client = get_client(i)
        prompt = FIX_DESIGN_PROMPT_TEMPLATE.format(abstract=abstract[:3000])

        print(f"[{i+1}/{len(target_indices)}] Processing Row {idx} (PMID: {row.get('PMID', 'N/A')})...")
        
        try:
            # 调用 API
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={'type': 'json_object'}
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            
            new_design = data.get('research_design', 'Error')
            
            # 更新 DataFrame
            df.at[idx, 'Research Design'] = new_design
            print(f"  ✅ Fixed: Error -> {new_design}")
            
            # 立即保存
            df.to_excel(FILE_PATH, index=False)
            print("  💾 Saved.")

        except Exception as e:
            print(f"  ❌ Failed: {e}")
        
        # 礼貌延时，避免速率限制
        time.sleep(1)

    print("\n--- Final Research Design Statistics ---")
    print(df['Research Design'].value_counts())

if __name__ == "__main__":
    main()
