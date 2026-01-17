import pandas as pd
import os
import shutil
import subprocess

FILE_PATH = "Literature_Screening_List.xlsx"
TEMP_PATH = "temp_current.xlsx"

def check_deleted():
    if not os.path.exists(FILE_PATH):
        print("Current file not found.")
        return

    # 1. Backup current clean version
    shutil.copy(FILE_PATH, TEMP_PATH)
    print("Backed up current database.")
    
    try:
        # 2. Retrieve old version (before cleaning)
        # We need to go back 2 commits (Commit "Add filtering scripts" -> Commit "Strictly removed..." -> Commit "Permanently removed...")
        # Actually, the file was full BEFORE "Permanently removed 21..." and "Strictly removed 17..."
        # So we try to checkout HEAD~2 or find the commit before cleaning.
        # Let's try HEAD~3 to be safe (before all cleaning)
        subprocess.run(["git", "checkout", "HEAD~3", FILE_PATH], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("Retrieved historical database (pre-cleaning).\n")
        
        df_old = pd.read_excel(FILE_PATH)
        
        # 3. Identify excluded papers
        # These are papers where Is_Cancer_Relevant is 'N' (from our scripts)
        # Note: In the old file, they might be marked 'N'.
        # Or, we can just compare IDs with the current file.
        
        df_current = pd.read_excel(TEMP_PATH)
        current_ids = set(df_current['PMID'].astype(str))
        
        # Find rows in old but not in current
        df_old['PMID'] = df_old['PMID'].astype(str)
        excluded_df = df_old[~df_old['PMID'].isin(current_ids)]
        
        print(f"Found {len(excluded_df)} excluded papers:\n")
        print("-" * 80)
        print(f"{'INDEX':<6} | {'REASON':<25} | {'TITLE'}")
        print("-" * 80)
        
        for idx, row in excluded_df.iterrows():
            reason = str(row.get('Relevance_Reason', 'N/A'))[:25]
            title = row['Title'][:60] + "..."
            print(f"{idx:<6} | {reason:<25} | {title}")
            
        print("-" * 80)
        
    except Exception as e:
        print(f"Error retrieving history: {e}")
        
    finally:
        # 4. Restore current version
        if os.path.exists(TEMP_PATH):
            shutil.move(TEMP_PATH, FILE_PATH)
            print("\nRestored current clean database.")
            # Reset git index for this file to avoid confusion
            subprocess.run(["git", "reset", "HEAD", FILE_PATH], stdout=subprocess.DEVNULL)
            subprocess.run(["git", "checkout", "--", FILE_PATH], stdout=subprocess.DEVNULL)

if __name__ == "__main__":
    check_deleted()
