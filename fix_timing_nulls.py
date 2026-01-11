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

FIX_TIMING_PROMPT_TEMPLATE = """
You are an expert medical research classifier.
The following abstract has a missing "Study Timing" (NaN).
Please analyze it carefully and extract the Study Timing.

Abstract:
{abstract}

Research Design: {research_design}

---
ALLOWED STUDY TIMINGS (Select ONE):
1. **Retrospective**: Looking back at past data (e.g., chart review, historical cohort).
2. **Prospective**: Following subjects forward in time (e.g., RCT, prospective cohort).
3. **Cross-sectional**: Analyzing data at a single point in time (e.g., survey, prevalence study).
4. **Ambispective**: Both retrospective and prospective components.
5. **Simulation/Model-based**: Computer modeling, cost-effectiveness analysis, theoretical simulations.
6. **Longitudinal**: Repeated observations over time (when direction is unclear but time is involved).
7. **Not Applicable**: For narrative reviews, guidelines, editorials, or pure lab/animal studies without clinical timing context.

---
Output strictly in JSON format:
{{
    "study_timing": "..."
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
    prompt = FIX_TIMING_PROMPT_TEMPLATE.format(
        abstract=abstract[:3000],
        research_design=row.get('Research Design', 'Unknown')
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
        
        new_timing = data.get('study_timing', 'Unknown')
        
        if new_timing != "Unknown":
            with print_lock:
                print(f"✅ Row {index}: NaN -> {new_timing} (Design: {row.get('Research Design')})")
            
        return index, new_timing

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

    # 筛选 Timing 为 NaN 的行
    mask = df['Study Timing'].isnull()
    target_indices = df[mask].index.tolist()
    
    print(f"🔍 Found {len(target_indices)} rows with missing 'Study Timing' to fix.")
    if len(target_indices) == 0:
        print("Nothing to do.")
        return

    # 准备任务
    tasks = []
    for i, idx in enumerate(target_indices):
        tasks.append((idx, df.loc[idx], i))

    # 并发处理
    results = []
    print("🚀 Starting Timing Fix (4 threads)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_idx = {executor.submit(refine_row, task): task[0] for task in tasks}
        
        for future in concurrent.futures.as_completed(future_to_idx):
            idx, new_timing = future.result()
            if new_timing:
                results.append((idx, new_timing))

    # 更新 DataFrame
    print("💾 Updating DataFrame...")
    update_count = 0
    for idx, timing in results:
        df.at[idx, 'Study Timing'] = timing
        update_count += 1

    print(f"✅ Processed {update_count} rows.")

    # 保存
    output_path = FILE_PATH
    df.to_excel(output_path, index=False)
    print(f"🚀 Overwritten original file: {output_path}")

    # 打印最终统计
    print("\n--- Final Study Timing Statistics ---")
    print(df['Study Timing'].value_counts(dropna=False))

if __name__ == "__main__":
    main()
