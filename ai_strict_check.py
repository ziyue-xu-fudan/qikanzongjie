import pandas as pd
import requests
import json
import time
import os
import concurrent.futures
from tqdm import tqdm

# --- Config ---
FILE_PATH = "Literature_Screening_List.xlsx"
BASE_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL_NAME = "deepseek-chat"

# Key Pool
API_KEYS = [
    "sk-715eccd883fa4ab7b30ccdbfb6e04d41", 
    "sk-8c43916964174d6c975a6c3822262d1c", 
    "sk-855909191d3148f988e40409c37576d3",
    "sk-9a74c76b97624c96a75a9e33d0695071",
    "sk-f831936c34534062a74c7e63b655f058",
    "sk-a20c5678440d4304853039d567c87042",
    "sk-f6a39287c71a48c78c35a6396825c862",
    "sk-6d0d2979208044709425a80530514934"
]

key_index = 0

def get_next_key():
    global key_index
    k = API_KEYS[key_index]
    key_index = (key_index + 1) % len(API_KEYS)
    return k

def check_relevance_strict(title, abstract, retries=3):
    if not abstract or len(str(abstract)) < 10:
        return {"is_relevant": False, "reason": "No abstract available"}
        
    prompt = f"""
    STRICT RELEVANCE CHECK: Is the **PRIMARY** and **DOMINANT** focus of this study on CANCER/ONCOLOGY?
    
    Title: {title}
    Abstract: {abstract}
    
    CRITERIA FOR "YES" (KEEP):
    1. Study population is explicitly CANCER PATIENTS.
    2. OR Intervention is specifically for cancer treatment/screening/diagnosis.
    3. OR Outcome is specifically cancer incidence, mortality, or progression.
    
    CRITERIA FOR "NO" (EXCLUDE - BE STRICT):
    1. BROAD PUBLIC HEALTH: Studies on general population health, diet, or lifestyle where cancer is just one of many outcomes mentioned (e.g., "Association of BMI with 50 diseases").
    2. GENERAL MEDICINE: Studies on cardiovascular, diabetes, or other diseases where cancer is just a comorbidity or exclusion criteria.
    3. POLICY/ECONOMICS: General healthcare policy studies not specifically tailored to oncology departments.
    4. COVID-19: General COVID-19 studies (even if mentioning vulnerable cancer patients).
    
    If it's a "broad spectrum" study that includes cancer but doesn't focus on it, REJECT IT.
    
    Return JSON only:
    {{
        "is_relevant": true/false,
        "reason": "Brief explanation in Chinese (max 20 chars)"
    }}
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
            "temperature": 0.1,
            "max_tokens": 200,
            "response_format": {"type": "json_object"}
        }
        
        try:
            response = requests.post(BASE_URL, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content'].strip()
                try:
                    return json.loads(content)
                except:
                    if "true" in content.lower():
                        return {"is_relevant": True, "reason": "Parsed True"}
                    else:
                        return {"is_relevant": False, "reason": "Parsed False"}
                    
            elif response.status_code in [401, 402]:
                continue
            elif response.status_code == 429:
                time.sleep(2)
                continue
                
        except Exception as e:
            time.sleep(1)
            
    return {"is_relevant": True, "reason": "API Error (Default Keep)"}

def process_row(row):
    idx, data = row
    return idx, check_relevance_strict(data['Title'], data['Abstract'])

def main():
    if not os.path.exists(FILE_PATH):
        print("File not found.")
        return

    df = pd.read_excel(FILE_PATH)
    print(f"Checking strict relevance for {len(df)} papers...")
    
    excluded_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_row, row): row[0] for row in df.iterrows()}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(df), desc="Strict Screening"):
            idx, res = future.result()
            
            is_rel = "Y" if res['is_relevant'] else "N"
            reason = res['reason']
            
            # Only update if it's N (don't overwrite previous N with Y)
            if is_rel == "N":
                df.at[idx, 'Is_Cancer_Relevant'] = "N"
                df.at[idx, 'Relevance_Reason'] = reason
                df.at[idx, 'Select? (Y/N)'] = "N"
                excluded_count += 1
                print(f"\n[STRICT REJECT] {df.at[idx, 'Title'][:40]}... | {reason}")
                
    df.to_excel(FILE_PATH, index=False)
    print(f"\nStrict Screening Complete! Excluded {excluded_count} additional papers.")

if __name__ == "__main__":
    main()
