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

REFINE_PROMPT_TEMPLATE = """
You are an expert medical research classifier.
The following abstract was previously classified as "Other" or had "Not Applicable" timing.
Your task is to RE-CLASSIFY it into a more specific category if possible, using the EXPANDED lists below.

Abstract:
{abstract}

---
ALLOWED RESEARCH DESIGNS (Select ONE):
1. Standard Types:
   [Randomized Controlled Trial, Cohort Study, Case-Control Study, Cross-sectional Study, Systematic Review, Meta-analysis, Case Report, Animal Study, In Vitro Study, Narrative Review, Clinical Observation]
2. NEW Specific Types (Prioritize these if applicable):
   [Diagnostic Accuracy Study, Time Series Analysis, Modeling Study, Economic Evaluation, Qualitative Study, Guideline/Consensus, Study Protocol]
3. If still none match, use "Other".

ALLOWED STUDY TIMINGS (Select ONE):
1. Standard Types:
   [Retrospective, Prospective, Cross-sectional, Ambispective]
2. NEW Specific Types:
   [Simulation/Model-based, Longitudinal]
3. If truly not applicable (e.g. reviews), use "Not Applicable".

---
Output strictly in JSON format:
{{
    "research_design": "...",
    "study_timing": "..."
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
    prompt = REFINE_PROMPT_TEMPLATE.format(abstract=abstract[:3000]) # 截断防止过长

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={'type': 'json_object'}
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        
        r_design = data.get('research_design', row['Research Design'])
        s_timing = data.get('study_timing', row['Study Timing'])
        
        with print_lock:
            print(f"✅ Row {index} refined: {row['Research Design']} -> {r_design} | {row['Study Timing']} -> {s_timing}")
            
        return index, r_design, s_timing

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
    # 条件: Research Design 为 'Other' 或者 Study Timing 为 'Not Applicable'
    mask = (df['Research Design'] == 'Other') | (df['Study Timing'] == 'Not Applicable')
    target_indices = df[mask].index.tolist()
    
    print(f"🔍 Found {len(target_indices)} rows to refine.")
    if len(target_indices) == 0:
        print("Nothing to do.")
        return

    # 准备任务
    tasks = []
    for i, idx in enumerate(target_indices):
        tasks.append((idx, df.loc[idx], i)) # i 用于分配 key

    # 并发处理
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # 提交所有任务
        future_to_idx = {executor.submit(refine_row, task): task[0] for task in tasks}
        
        for future in concurrent.futures.as_completed(future_to_idx):
            idx, new_design, new_timing = future.result()
            if new_design and new_timing:
                results.append((idx, new_design, new_timing))

    # 更新 DataFrame
    print("💾 Updating DataFrame...")
    update_count = 0
    for idx, des, tim in results:
        df.at[idx, 'Research Design'] = des
        df.at[idx, 'Study Timing'] = tim
        update_count += 1

    print(f"✅ Successfully updated {update_count} rows.")

    # 保存
    output_path = FILE_PATH # 覆盖原文件
    # 也可以先备份一下
    backup_path = FILE_PATH.replace(".xlsx", "_backup.xlsx")
    df.to_excel(backup_path, index=False)
    print(f"📦 Backup saved to {backup_path}")
    
    df.to_excel(output_path, index=False)
    print(f"🚀 Overwritten original file: {output_path}")

    # 打印最终统计
    print("\n--- Final Statistics ---")
    print(df['Research Design'].value_counts())
    print("\n")
    print(df['Study Timing'].value_counts())

if __name__ == "__main__":
    main()
