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

# 更激进的 Prompt，挖掘潜在的健康状况
REFINE_NA_PROMPT_TEMPLATE = """
You are an expert medical research classifier.
The following abstract was previously classified as "Not Applicable" for Focused Disease.
Your task is to TRY HARDER to identify a specific health condition, risk factor, or population focus.

Abstract:
{abstract}

---
INSTRUCTIONS:
1. **Identify Broad Health Conditions**:
   - Instead of just diseases, look for:
     - "Pregnancy / Maternal Health"
     - "Obesity / Overweight"
     - "Aging / Frailty"
     - "Child Development / Nutrition"
     - "Vaccination / Immunization" (Classify as 'Vaccine Preventable Diseases' or the specific virus)
     - "Smoking / Substance Use"
     - "Mental Well-being" (even if not a specific disorder)
     - "Antibiotic Resistance"
   - If found, use this as the "Focused Disease".

2. **Only use "Not Applicable" if**:
   - The paper is PURELY about hospital administration, editorial policies, general AI algorithms (without clinical context), or professional education.

3. **Focused Disease System**:
   - If you find a condition, classify it into a system (e.g. 'Reproductive', 'Metabolic', 'Immune').
   - If it's about vaccines, use 'Immune' or 'Infectious Disease'.
   - If strictly Not Applicable, use 'General Health/System'.

---
Output strictly in JSON format:
{{
    "focused_disease": "...",
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
        return index, None, None

    client = get_client(thread_idx)
    prompt = REFINE_NA_PROMPT_TEMPLATE.format(abstract=abstract[:3000])

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={'type': 'json_object'}
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        
        f_disease = data.get('focused_disease', row['Focused Disease'])
        f_system = data.get('focused_disease_system', row['Focused Disease System'])
        
        # 只在有变化时打印
        if f_disease != "Not Applicable":
            with print_lock:
                print(f"✅ Row {index} RECOVERED: Not Applicable -> {f_disease} | {f_system}")
        else:
            # 仍然是 N/A，不打印刷屏，除非是为了调试
             pass
            
        return index, f_disease, f_system

    except Exception as e:
        with print_lock:
            print(f"❌ Row {index} failed: {e}")
        return index, None, None

def main():
    print(f"📂 Reading {FILE_PATH}...")
    try:
        df = pd.read_excel(FILE_PATH, engine='openpyxl')
    except Exception as e:
        print(f"Fatal Error: {e}")
        return

    # 筛选 Not Applicable
    mask = (df['Focused Disease'] == 'Not Applicable')
    target_indices = df[mask].index.tolist()
    
    print(f"🔍 Found {len(target_indices)} 'Not Applicable' rows to deep clean.")
    if len(target_indices) == 0:
        print("Nothing to do.")
        return

    # 准备任务
    tasks = []
    for i, idx in enumerate(target_indices):
        tasks.append((idx, df.loc[idx], i))

    # 并发处理
    results = []
    print("🚀 Starting Deep Cleaning (4 threads)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_idx = {executor.submit(refine_row, task): task[0] for task in tasks}
        
        for future in concurrent.futures.as_completed(future_to_idx):
            idx, new_disease, new_system = future.result()
            if new_disease and new_system:
                results.append((idx, new_disease, new_system))

    # 更新 DataFrame
    print("💾 Updating DataFrame...")
    update_count = 0
    for idx, dis, sys in results:
        # 只有当提取出有效信息时才更新，或者确认是 N/A
        df.at[idx, 'Focused Disease'] = dis
        df.at[idx, 'Focused Disease System'] = sys
        update_count += 1

    print(f"✅ Processed {update_count} rows.")

    # 保存
    output_path = FILE_PATH
    df.to_excel(output_path, index=False)
    print(f"🚀 Overwritten original file: {output_path}")

    # 打印最终统计
    print("\n--- Final Disease Statistics (Top 20) ---")
    print(df['Focused Disease'].value_counts().head(20))

if __name__ == "__main__":
    main()
