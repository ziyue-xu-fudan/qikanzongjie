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

REFINE_DISEASE_PROMPT_TEMPLATE = """
You are an expert medical research classifier.
The following abstract had "Unknown" or "Error" for the "Focused Disease" field.
Your task is to RE-EXTRACT the disease or correctly label it as "Not Applicable".

Abstract:
{abstract}

---
INSTRUCTIONS:
1. If the paper focuses on a specific disease, condition, or symptom:
   - Extract the name and provide the ICD-10 code if possible.
   - Example: "Type 2 Diabetes Mellitus (E11)", "COVID-19 (U07.1)".
2. If the paper is about:
   - General health / Wellness (e.g. diet for healthy people)
   - Medical education / Training
   - Healthcare system / Policy / Economics (without a specific disease focus)
   - Artificial Intelligence technology itself
   - Basic science (e.g. method development)
   -> Label "Focused Disease" as "Not Applicable".
   -> Label "Focused Disease System" as "General Health/System".

3. "Focused Disease System" categories:
   [Cardiovascular, Respiratory, Nervous, Digestive, Endocrine, Immune, Musculoskeletal, Urinary, Reproductive, Integumentary, Oncology, Infectious Disease, General Health/System, Other]

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
    """根据线程索引分配 Key (简单的轮询策略)"""
    api_key = API_KEYS[index % len(API_KEYS)]
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

def refine_row(args):
    """处理单行数据的函数"""
    index, row, thread_idx = args
    abstract = row.get('Abstract', '')
    
    if not isinstance(abstract, str) or len(abstract) < 10:
        return index, None, None # 无效摘要，跳过

    client = get_client(thread_idx)
    prompt = REFINE_DISEASE_PROMPT_TEMPLATE.format(abstract=abstract[:3000])

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
        
        with print_lock:
            print(f"✅ Row {index}: {row['Focused Disease']} -> {f_disease} | {f_system}")
            
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

    # 筛选需要细化的行
    # 条件: Focused Disease 为 'Unknown', 'Error', 'None' 或者 NaN
    # 同时也可以检查 'Focused Disease System' 是否为 'Unknown'
    # 强制包含 'Error' 状态，即使它可能之前处理过
    target_values = ['Unknown', 'Error', 'None', 'nan']
    mask = df['Focused Disease'].astype(str).isin(target_values) | df['Focused Disease'].isnull()
    
    target_indices = df[mask].index.tolist()
    
    print(f"🔍 Found {len(target_indices)} rows to refine.")
    if len(target_indices) == 0:
        print("Nothing to do.")
        return

    # 准备任务
    tasks = []
    for i, idx in enumerate(target_indices):
        tasks.append((idx, df.loc[idx], i))

    # 并发处理
    results = []
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
        df.at[idx, 'Focused Disease'] = dis
        df.at[idx, 'Focused Disease System'] = sys
        update_count += 1

    print(f"✅ Successfully updated {update_count} rows.")

    # 保存
    output_path = FILE_PATH
    backup_path = FILE_PATH.replace(".xlsx", "_backup_disease.xlsx")
    df.to_excel(backup_path, index=False)
    print(f"📦 Backup saved to {backup_path}")
    
    df.to_excel(output_path, index=False)
    print(f"🚀 Overwritten original file: {output_path}")

    # 打印最终统计
    print("\n--- Final Disease Statistics ---")
    print(df['Focused Disease'].value_counts().head(10))

if __name__ == "__main__":
    main()
