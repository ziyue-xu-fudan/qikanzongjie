import pandas as pd
from Bio import Entrez
import time
from datetime import datetime

# 1. API Configuration
Entrez.email = "ziyuexu20@fudan.edu.cn"
Entrez.api_key = "e3674f393e05e49020299c745b81574ea707"

# 2. Define Queries (Hardcoded as requested)
CANCER_KEYWORDS = """
    AND ("Cancer"[Title/Abstract] OR "Tumor"[Title/Abstract] OR "Tumour"[Title/Abstract] OR "Oncology"[Title/Abstract] OR "Neoplasm"[Title/Abstract] OR "Carcinoma"[Title/Abstract])
"""
DATE_RANGE = '"2020/01/01"[Date - Publication] : "2026/12/31"[Date - Publication]'

QUERIES = {
    "NEJM": f"""
        "The New England journal of medicine"[Journal] 
        AND ({DATE_RANGE})
        {CANCER_KEYWORDS}
        AND ("Background"[Title/Abstract] AND "Methods"[Title/Abstract] AND "Results"[Title/Abstract])
        NOT 
        ("Randomized Controlled Trial"[Publication Type] OR "Clinical Trial, Phase III"[Publication Type] OR "Phase 3"[Title/Abstract] OR "Phase III"[Title/Abstract] OR "random"[Title/Abstract] OR "blind"[Title/Abstract])
    """,
    "Lancet": f"""
        "LANCET"[Journal] 
        AND ({DATE_RANGE})
        {CANCER_KEYWORDS}
        AND ("Methods"[Title/Abstract] AND "Findings"[Title/Abstract] AND "Interpretation"[Title/Abstract]) 
        NOT 
        ("Randomized Controlled Trial"[Publication Type] OR "Clinical Trial, Phase III"[Publication Type] OR "Phase 3"[Title/Abstract] OR "Phase III"[Title/Abstract] OR "Random"[Title/Abstract] OR "blind"[Title/Abstract])
    """,
    "JAMA": f"""
        "JAMA"[Journal] 
        AND ({DATE_RANGE})
        {CANCER_KEYWORDS}
        AND ("Importance"[Title/Abstract] AND "Design"[Title/Abstract] AND "Results"[Title/Abstract])
        NOT 
        ("Randomized Controlled Trial"[Publication Type] OR "Clinical Trial, Phase III"[Publication Type] OR "Phase 3"[Title/Abstract] OR "Phase III"[Title/Abstract] OR "random"[Title/Abstract] OR "blind"[Title/Abstract])
    """,
    "BMJ": f"""
        "BMJ (Clinical research ed.)"[Journal] 
        AND ({DATE_RANGE})
        {CANCER_KEYWORDS}
        AND ("Objectives"[Title/Abstract] AND "Design"[Title/Abstract] AND "Results"[Title/Abstract])
        NOT 
        ("Randomized Controlled Trial"[Publication Type] OR "Clinical Trial, Phase III"[Publication Type] OR "Phase 3"[Title/Abstract] OR "Phase III"[Title/Abstract] OR "random"[Title/Abstract] OR "blind"[Title/Abstract])
    """
}

def fetch_papers(journal_name, query):
    print(f"Fetching for {journal_name}...")
    try:
        # Search
        handle = Entrez.esearch(db="pubmed", term=query, retmax=1000, sort="date")
        record = Entrez.read(handle)
        handle.close()
        
        id_list = record["IdList"]
        print(f"  Found {len(id_list)} papers.")
        
        if not id_list:
            return []

        # Fetch details
        papers = []
        batch_size = 100
        
        for i in range(0, len(id_list), batch_size):
            batch_ids = id_list[i:i+batch_size]
            handle = Entrez.efetch(db="pubmed", id=batch_ids, rettype="medline", retmode="text")
            records = handle.read().split("\n\n") # Simple split, better to use Medline parser if complex
            
            # Using Bio.Medline parser for robustness
            from Bio import Medline
            handle = Entrez.efetch(db="pubmed", id=batch_ids, rettype="medline", retmode="text")
            parsed_records = Medline.parse(handle)
            
            for record in parsed_records:
                papers.append({
                    "Journal_Category": journal_name,
                    "Title": record.get("TI", ""),
                    "Journal": record.get("TA", ""),
                    "PubDate": record.get("DP", ""),
                    "PMID": record.get("PMID", ""),
                    "DOI": record.get("LID", "").replace(" [doi]", "") if "LID" in record else "",
                    "Abstract": record.get("AB", ""),
                    "PublicationType": "; ".join(record.get("PT", []))
                })
            handle.close()
            time.sleep(0.5) # Be gentle even with API key
            
        return papers
        
    except Exception as e:
        print(f"  Error fetching {journal_name}: {e}")
        return []

# 3. Main Execution
all_papers = []

for journal, query in QUERIES.items():
    # Clean up query string (remove newlines and extra spaces)
    clean_query = " ".join(query.split())
    results = fetch_papers(journal, clean_query)
    all_papers.extend(results)

# 4. Save to Excel
if all_papers:
    df = pd.DataFrame(all_papers)
    output_file = "Top4_NonRCT_Cancer_Papers_2023_2026.xlsx"
    df.to_excel(output_file, index=False)
    print(f"\nSuccessfully saved {len(df)} papers to {output_file}")
    
    # Print summary
    print("\nSummary by Journal:")
    print(df["Journal_Category"].value_counts())
else:
    print("\nNo papers found matching the criteria.")
