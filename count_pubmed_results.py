from Bio import Entrez
import time

# API Configuration
Entrez.email = "ziyuexu20@fudan.edu.cn"
Entrez.api_key = "e3674f393e05e49020299c745b81574ea707"

# Base Queries (Without RCT Exclusion)
# Using the same Cancer Keywords and Date Range as fetch script
CANCER_KEYWORDS = """
    AND ("Cancer"[Title/Abstract] OR "Tumor"[Title/Abstract] OR "Tumour"[Title/Abstract] OR "Oncology"[Title/Abstract] OR "Neoplasm"[Title/Abstract] OR "Carcinoma"[Title/Abstract])
"""
DATE_RANGE = '"2020/01/01"[Date - Publication] : "2026/12/31"[Date - Publication]'

QUERIES_RAW = {
    "NEJM": f"""
        "The New England journal of medicine"[Journal] 
        AND ({DATE_RANGE})
        {CANCER_KEYWORDS}
        AND ("Background"[Title/Abstract] AND "Methods"[Title/Abstract] AND "Results"[Title/Abstract])
    """,
    "Lancet": f"""
        "LANCET"[Journal] 
        AND ({DATE_RANGE})
        {CANCER_KEYWORDS}
        AND ("Methods"[Title/Abstract] AND "Findings"[Title/Abstract] AND "Interpretation"[Title/Abstract]) 
    """,
    "JAMA": f"""
        "JAMA"[Journal] 
        AND ({DATE_RANGE})
        {CANCER_KEYWORDS}
        AND ("Importance"[Title/Abstract] AND "Design"[Title/Abstract] AND "Results"[Title/Abstract])
    """,
    "BMJ": f"""
        "BMJ (Clinical research ed.)"[Journal] 
        AND ({DATE_RANGE})
        {CANCER_KEYWORDS}
        AND ("Objectives"[Title/Abstract] AND "Design"[Title/Abstract] AND "Results"[Title/Abstract])
    """
}

# Exclusion String
EXCLUSION = """
    NOT 
    ("Randomized Controlled Trial"[Publication Type] OR "Clinical Trial, Phase III"[Publication Type] OR "Phase 3"[Title/Abstract] OR "Phase III"[Title/Abstract] OR "Random"[Title/Abstract] OR "blind"[Title/Abstract])
"""

def get_count(query):
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=0)
        record = Entrez.read(handle)
        handle.close()
        return int(record["Count"])
    except Exception as e:
        print(f"Error: {e}")
        return 0

def main():
    print("Starting accurate count check...")
    
    total_raw = 0
    total_filtered = 0
    
    for journal, raw_query in QUERIES_RAW.items():
        # 1. Clean query
        clean_raw = " ".join(raw_query.split())
        
        # 2. Construct Filtered Query
        clean_filtered = clean_raw + " " + " ".join(EXCLUSION.split())
        
        # 3. Fetch Counts
        count_raw = get_count(clean_raw)
        time.sleep(0.5) # Rate limit
        count_filtered = get_count(clean_filtered)
        time.sleep(0.5)
        
        print(f"[{journal}] Raw: {count_raw} -> Filtered: {count_filtered} (Excluded: {count_raw - count_filtered})")
        
        total_raw += count_raw
        total_filtered += count_filtered
        
    print("-" * 30)
    print(f"TOTAL IDENTIFIED (Raw): {total_raw}")
    print(f"TOTAL REMOVED (RCTs): {total_raw - total_filtered}")
    print(f"TOTAL SCREENED (Filtered): {total_filtered}")

if __name__ == "__main__":
    main()
