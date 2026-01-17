import pandas as pd
import requests
import json
import time
import concurrent.futures
from tqdm import tqdm
import os
import random
import threading

# Configuration
API_KEYS = [
    "sk-035f4a026e724fb194a5083b3ed3c3b1",
    "sk-37c1617db0da456d8491e1094e3f6ae3",
    "sk-82a00766192049fc91da7edbca74bfd2",
    "sk-c69f18b962d54e44b14298f079bc4c66",
    "sk-d98eb5841a0b4e6c9985b72b4106c74c",
    "sk-0f2424b6b95b4ada8887fde557a22078"
]
BASE_URL = "https://api.deepseek.com/chat/completions"
FILE_PATH = "Literature_Screening_List.xlsx"
MODEL_NAME = "deepseek-chat"

# Thread-safe key management
key_lock = threading.Lock()
current_key_index = 0

def get_next_key():
    global current_key_index
    with key_lock:
        key = API_KEYS[current_key_index]
        current_key_index = (current_key_index + 1) % len(API_KEYS)
        return key

def get_ai_analysis(title, abstract, retries=3):
    if not abstract or len(str(abstract)) < 10:
        return {"Summary": "无摘要", "Cancer_Type": "", "Study_Design": "", "Phase": "", "Highlights": ""}
        
    prompt = f"""
    请阅读以下医学论文的标题和摘要：
    标题：{title}
    摘要：{abstract}
    
    任务：请提取关键信息并返回 JSON 格式。
    要求返回的 JSON 字段如下：
    1. "Cancer_Type": 具体的癌肿名称（中文，如“非小细胞肺癌”）。
    2. "Study_Design": 研究设计类型（中文，如“回顾性队列”、“随机对照试验”、“真实世界研究”）。
    3. "Phase": 临床试验分期（如“I期”、“II期”、“III期”；若非干预性研究，填“N/A”）。
    4. "Summary": 一句话核心总结（中文，包含研究类型、核心发现及临床价值，不超过60字）。
    5. "Highlights": 提炼3个核心创新点或发现（中文，用分号分隔）。

    请直接返回合法的 JSON 字符串，不要包含 Markdown 格式标记（如 ```json）。
    """
    
    for attempt in range(retries):
        api_key = get_next_key()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 500,
            "response_format": {"type": "json_object"}
        }
        
        try:
            response = requests.post(BASE_URL, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content'].strip()
                # Parse JSON
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    print(f"JSON Parse Error: {content[:50]}...")
                    continue
                    
            elif response.status_code in [401, 402]:
                print(f"Key {api_key[:8]}... failed ({response.status_code}), switching...")
                continue
            elif response.status_code == 429:
                time.sleep(2)
                continue
            else:
                print(f"Error {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"Request exception: {e}")
            time.sleep(1)
            
    return {"Summary": "Error: Failed", "Cancer_Type": "", "Study_Design": "", "Phase": "", "Highlights": ""}

def process_row(row):
    idx, data = row
    # Always re-process to get new fields
    return idx, get_ai_analysis(data['Title'], data['Abstract'])

def main():
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        return

    df = pd.read_excel(FILE_PATH)
    
    # Initialize new columns
    new_cols = ['Cancer_Type', 'Study_Design', 'Phase', 'Highlights']
    for col in new_cols:
        if col not in df.columns:
            df[col] = ""
            
    print(f"Total papers: {len(df)}")
    print(f"Using {len(API_KEYS)} API keys.")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_row, row): row[0] for row in df.iterrows()}
        
        count = 0
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(df), desc="Analyzing"):
            idx, result = future.result()
            
            # Update DataFrame
            df.at[idx, 'AI_Summary'] = result.get('Summary', '')
            df.at[idx, 'Cancer_Type'] = result.get('Cancer_Type', '')
            df.at[idx, 'Study_Design'] = result.get('Study_Design', '')
            df.at[idx, 'Phase'] = result.get('Phase', '')
            df.at[idx, 'Highlights'] = result.get('Highlights', '')
            
            # Print preview
            print(f"\n[Paper {idx}] {result.get('Cancer_Type', 'Unknown')} | {result.get('Phase', 'N/A')}")
            
            count += 1
            if count % 20 == 0:
                df.to_excel(FILE_PATH, index=False)
                
    df.to_excel(FILE_PATH, index=False)
    print("All structured analysis completed and saved!")

if __name__ == "__main__":
    main()
