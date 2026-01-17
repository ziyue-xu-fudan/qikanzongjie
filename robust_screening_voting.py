import pandas as pd
import requests
import json
import time
import os
import concurrent.futures
from tqdm import tqdm
from collections import Counter

# --- Config ---
# Handle the specific filename provided by user
INPUT_FILE = "Literature_Screening_List（1）.xlsx"
OUTPUT_FILE = "Literature_Screening_List.xlsx" # We will overwrite the main file eventually

BASE_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL_NAME = "deepseek-chat"

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

def call_api_once(title, abstract):
    prompt = f"""
    STRICT RELEVANCE CHECK: Does this study have a **DIRECT** and **PRIMARY** connection to Cancer/Oncology?
    
    Title: {title}
    Abstract: {abstract}
    
    CRITERIA FOR "YES" (DIRECTLY RELEVANT):
    1. Subjects are CANCER PATIENTS.
    2. OR Intervention is CANCER TREATMENT/SCREENING.
    3. OR Outcome is CANCER SPECIFIC (e.g., survival in cancer, metastasis).
    
    CRITERIA FOR "NO" (NOT DIRECTLY RELEVANT):
    1. Study is about other diseases (Heart Failure, Diabetes, COVID-19) where cancer is just a comorbidity.
    2. Study is about general public health/policy (e.g., "Hospital admission rates") without specific oncology focus.
    3. Study explicitly excludes cancer patients or mentions them only as a subgroup in a non-cancer study.
    4. General genetic/biological studies not linked to specific cancer outcomes.
    
    Return JSON:
    {{
        "is_relevant": true/false,
        "reason": "Very brief reason (max 15 chars)"
    }}
    """
    
    api_key = get_next_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3, # Slight temp for variation in voting
        "max_tokens": 100,
        "response_format": {"type": "json_object"}
    }
    
    try:
        response = requests.post(BASE_URL, headers=headers, json=data, timeout=20)
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content'].strip()
            return json.loads(content)
        elif response.status_code == 402:
            print(f"Key {api_key[:8]} quota exceeded.")
            return None
    except:
        pass
    return None

def check_relevance_voting(title, abstract, n=3):
    if not abstract or len(str(abstract)) < 10:
        return {"final_decision": False, "vote_score": "0/3", "reason": "No abstract"}
    
    votes = []
    reasons = []
    
    for _ in range(n):
        res = call_api_once(title, abstract)
        if res:
            votes.append(res.get('is_relevant', True)) # Default True if key missing
            reasons.append(res.get('reason', ''))
        else:
            # If API fails, we don't count it. 
            pass
        time.sleep(0.5) # Avoid rapid fire
        
    if not votes:
        return {"final_decision": True, "vote_score": "Error", "reason": "API Failed"}
        
    # Majority Voting
    true_count = sum(votes)
    total_valid = len(votes)
    
    # If strictly more than half are True, then Keep.
    # e.g. 2/3 True -> Keep. 1/3 True -> Reject.
    is_relevant = true_count > (total_valid / 2)
    
    # Pick the most common reason
    final_reason = Counter(reasons).most_common(1)[0][0] if reasons else ""
    
    return {
        "final_decision": is_relevant,
        "vote_score": f"{true_count}/{total_valid}",
        "reason": final_reason
    }

def process_row(row):
    idx, data = row
    return idx, check_relevance_voting(data['Title'], data['Abstract'])

def main():
    # Check for file existence, handling potential unicode/spaces
    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found: {INPUT_FILE}")
        # Fallback to standard name if user renamed it
        if os.path.exists("Literature_Screening_List.xlsx"):
            print("Falling back to Literature_Screening_List.xlsx")
            df = pd.read_excel("Literature_Screening_List.xlsx")
        else:
            return
    else:
        df = pd.read_excel(INPUT_FILE)

    print(f"Starting Majority Voting (n=3) for {len(df)} papers...")
    
    results = {}
    
    # Use fewer workers to avoid rate limits since we do 3 calls per row
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_row, row): row[0] for row in df.iterrows()}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(df), desc="Voting"):
            idx, res = future.result()
            results[idx] = res
            
            if not res['final_decision']:
                print(f"\n[REJECT {res['vote_score']}] {df.at[idx, 'Title'][:40]}... | {res['reason']}")

    # Apply results
    excluded_count = 0
    df_clean = df.copy()
    
    for idx, res in results.items():
        if not res['final_decision']:
            df_clean.at[idx, 'Is_Cancer_Relevant'] = 'N'
            df_clean.at[idx, 'Relevance_Reason'] = f"Voting {res['vote_score']}: {res['reason']}"
            excluded_count += 1
        else:
            df_clean.at[idx, 'Is_Cancer_Relevant'] = 'Y'
            
    # Save CLEAN version (physically remove N)
    df_final = df_clean[df_clean['Is_Cancer_Relevant'] != 'N']
    
    print(f"\nVoting Complete!")
    print(f"Original: {len(df)}")
    print(f"Excluded: {excluded_count}")
    print(f"Remaining: {len(df_final)}")
    
    df_final.to_excel(OUTPUT_FILE, index=False)
    print(f"Saved cleaned database to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
