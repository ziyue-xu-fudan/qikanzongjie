import pandas as pd
import concurrent.futures
import time
import json
import os
from openai import OpenAI
from threading import Lock

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

REFINE_SYSTEM_OTHER_PROMPT_TEMPLATE = """
You are an expert medical research classifier.
The following abstract was previously classified as "Other" for Focused Disease System.
Your task is to re-classify it into a more specific system.

Abstract:
{abstract}

Current Focused Disease: {focused_disease}

---
INSTRUCTIONS:
1. **Try to fit into Standard Systems**:
   [Cardiovascular, Respiratory, Nervous, Digestive, Endocrine, Immune, Musculoskeletal, Urinary, Reproductive, Integumentary, Oncology, Infectious Disease]

2. **If not standard, use these SPECIFIC NEW CATEGORIES**:
   - "Mental Health / Psychiatric" (e.g. depression, burnout, suicide)
   - "Public Health / Epidemiology" (e.g. tobacco, pollution, mortality trends)
   - "Genetics / Genomic Medicine" (e.g. rare diseases, sequencing)
   - "Trauma / Emergency Medicine" (e.g. injury, poisoning, falls)
   - "Occupational / Environmental Health"
   - "Substance Use / Addiction"
   - "Metabolic / Nutrition" (e.g. obesity, diet)
   - "Hematologic" (blood disorders)
   - "Opthalmology" (eye)
   - "General Health/System" (if truly broad/policy)

3. **Only use "Other" if absolutely nothing else fits.**

---
Output strictly in JSON format:
{{
    "focused_disease_system": "..."
}}
"""

# -----------------------------------------------------------------------------
# 逻辑
# -----------------------------------------------------------------------------

print_lock = Lock()

def get_client(index):
    api_key = API_KEYS[index % len(API_KEYS)]
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

def refine_row(args):
    index, row, thread_idx = args
    abstract = row.get('Abstract', '')
    
    if not isinstance(abstract, str) or len(abstract) < 10:
        return index, None

    client = get_client(thread_idx)
    prompt = REFINE_SYSTEM_OTHER_PROMPT_TEMPLATE.format(
        abstract=abstract[:3000],
        focused_disease=row.get('Focused Disease', 'Unknown')
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={'type': 'json_object'}
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        
        new_system = data.get('focused_disease_system', row['Focused Disease System'])
        
        if new_system != "Other":
            with print_lock:
                print(f"✅ Row {index}: Other -> {new_system} (Disease: {row.get('Focused Disease')})")
            
        return index, new_system

    except Exception as e:
        with print_lock:
            print(f"❌ Row {index} failed: {e}")
        return index, None

def main():
    print(f"📂 Reading {FILE_PATH}...")
    try:
        df = pd.read_excel(FILE_PATH, engine='openpyxl')
    except Exception as e:
        print(f"Fatal Error: {e}")
        return

    # 筛选 System == Other
    mask = (df['Focused Disease System'] == 'Other')
    target_indices = df[mask].index.tolist()
    
    print(f"🔍 Found {len(target_indices)} 'Other' System rows to refine.")
    if len(target_indices) == 0:
        print("Nothing to do.")
        return

    # 准备任务
    tasks = []
    for i, idx in enumerate(target_indices):
        tasks.append((idx, df.loc[idx], i))

    # 并发处理
    results = []
    print("🚀 Starting Refinement (4 threads)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_idx = {executor.submit(refine_row, task): task[0] for task in tasks}
        
        for future in concurrent.futures.as_completed(future_to_idx):
            idx, new_system = future.result()
            if new_system:
                results.append((idx, new_system))

    # 更新 DataFrame
    print("💾 Updating DataFrame...")
    update_count = 0
    for idx, sys in results:
        df.at[idx, 'Focused Disease System'] = sys
        update_count += 1

    print(f"✅ Processed {update_count} rows.")

    # 保存
    output_path = FILE_PATH
    df.to_excel(output_path, index=False)
    print(f"🚀 Overwritten original file: {output_path}")

    # 打印最终统计
    print("\n--- Final System Statistics ---")
    print(df['Focused Disease System'].value_counts())

if __name__ == "__main__":
    main()
