import pandas as pd
import os

FILE_PATH = "Literature_Screening_List.xlsx"

def clean_db():
    if not os.path.exists(FILE_PATH):
        print("File not found.")
        return

    df = pd.read_excel(FILE_PATH)
    original_count = len(df)
    
    # Filter: Keep only rows where Is_Cancer_Relevant is NOT 'N'
    # We keep empty strings just in case (though we processed all), but mainly remove explicit 'N'
    df_clean = df[df['Is_Cancer_Relevant'] != 'N']
    
    new_count = len(df_clean)
    removed_count = original_count - new_count
    
    print(f"Original Count: {original_count}")
    print(f"New Count: {new_count}")
    print(f"Removed: {removed_count} non-cancer papers.")
    
    if removed_count > 0:
        df_clean.to_excel(FILE_PATH, index=False)
        print(f"Successfully saved cleaned database to {FILE_PATH}")
    else:
        print("No papers found to remove.")

if __name__ == "__main__":
    clean_db()
