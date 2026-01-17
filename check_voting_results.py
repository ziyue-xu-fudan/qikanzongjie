import pandas as pd
import os

# Files
SOURCE_FILE = "Literature_Screening_List（1）.xlsx" # The file user provided (165 papers)
TARGET_FILE = "Literature_Screening_List.xlsx"    # The file after voting (153 papers)

def check_results():
    if not os.path.exists(SOURCE_FILE) or not os.path.exists(TARGET_FILE):
        print("Files not found.")
        # Fallback logic if filenames differ
        return

    df_source = pd.read_excel(SOURCE_FILE)
    df_target = pd.read_excel(TARGET_FILE)
    
    # Identify excluded IDs
    source_ids = set(df_source['PMID'].astype(str))
    target_ids = set(df_target['PMID'].astype(str))
    
    excluded_ids = source_ids - target_ids
    
    print(f"Checking exclusion from Majority Voting (n=3)...")
    print(f"Source Count: {len(df_source)}")
    print(f"Target Count: {len(df_target)}")
    print(f"Excluded: {len(excluded_ids)}\n")
    
    if len(excluded_ids) > 0:
        print("-" * 100)
        print(f"{'PMID':<10} | {'TITLE'}")
        print("-" * 100)
        
        excluded_df = df_source[df_source['PMID'].astype(str).isin(excluded_ids)]
        
        for idx, row in excluded_df.iterrows():
            title = row['Title'][:80] + "..."
            print(f"{row['PMID']:<10} | {title}")
            
        print("-" * 100)
        print("\nThese papers were rejected because at least 2 out of 3 AI votes were NO.")

if __name__ == "__main__":
    check_results()
