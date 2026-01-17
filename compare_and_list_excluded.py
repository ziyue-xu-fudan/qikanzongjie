import pandas as pd
import os

# Define files
# Note: Handling potential spaces or special chars in filename
OLD_FILE = "Literature_Screening_List(1).xlsx"
NEW_FILE = "Literature_Screening_List.xlsx"
OUTPUT_FILE = "Excluded_Papers_Review.xlsx"

def compare_files():
    if not os.path.exists(OLD_FILE):
        print(f"Error: Old file '{OLD_FILE}' not found.")
        return
    if not os.path.exists(NEW_FILE):
        print(f"Error: New file '{NEW_FILE}' not found.")
        return

    print("Reading files...")
    df_old = pd.read_excel(OLD_FILE)
    df_new = pd.read_excel(NEW_FILE)
    
    # Ensure PMID is string for comparison
    df_old['PMID'] = df_old['PMID'].astype(str).str.strip()
    df_new['PMID'] = df_new['PMID'].astype(str).str.strip()
    
    old_ids = set(df_old['PMID'])
    new_ids = set(df_new['PMID'])
    
    # Find excluded IDs
    excluded_ids = old_ids - new_ids
    
    print(f"Old Count: {len(df_old)}")
    print(f"New Count: {len(df_new)}")
    print(f"Excluded Count: {len(excluded_ids)}")
    
    if len(excluded_ids) > 0:
        # Extract full rows for excluded papers
        df_excluded = df_old[df_old['PMID'].isin(excluded_ids)].copy()
        
        # Save to file
        df_excluded.to_excel(OUTPUT_FILE, index=False)
        print(f"\nSaved {len(df_excluded)} excluded papers to '{OUTPUT_FILE}'")
        
        # Print list
        print("\n=== LIST OF EXCLUDED PAPERS ===")
        print(f"{'INDEX':<6} | {'TITLE'}")
        print("-" * 80)
        for idx, row in df_excluded.iterrows():
            print(f"{idx:<6} | {row['Title'][:80]}...")
            
    else:
        print("\nNo papers have been excluded (datasets match).")

if __name__ == "__main__":
    compare_files()
