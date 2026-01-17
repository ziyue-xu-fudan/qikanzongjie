import pandas as pd
import os

FILE_PATH = "Literature_Screening_List.xlsx"

def check_data():
    if not os.path.exists(FILE_PATH):
        print("File not found.")
        return

    df = pd.read_excel(FILE_PATH)
    total_rows = len(df)
    print(f"Total rows in Excel: {total_rows}")
    
    # Check duplicates based on Title (PMID sometimes missing or inconsistent)
    # Ideally use PMID, but Title is also good check
    duplicates_pmid = df[df.duplicated(subset=['PMID'], keep=False)]
    duplicates_title = df[df.duplicated(subset=['Title'], keep=False)]
    
    print(f"Duplicate PMIDs: {len(duplicates_pmid)}")
    print(f"Duplicate Titles: {len(duplicates_title)}")
    
    # Check if we should drop them
    if len(duplicates_title) > 0:
        print("Found duplicates! Cleaning...")
        # Drop duplicates, keeping the first occurrence
        # Prefer subset=['Title'] as PMID might be empty for some manual entries (though unlikely here)
        df_clean = df.drop_duplicates(subset=['Title'], keep='first')
        print(f"Rows after cleaning: {len(df_clean)}")
        
        # Save back
        df_clean.to_excel(FILE_PATH, index=False)
        print(f"Saved cleaned file to {FILE_PATH}")
        return len(df_clean)
    else:
        print("No duplicates found. The discrepancy might be in Streamlit filters.")
        return total_rows

if __name__ == "__main__":
    final_count = check_data()
    print(f"FINAL TRUE COUNT: {final_count}")
