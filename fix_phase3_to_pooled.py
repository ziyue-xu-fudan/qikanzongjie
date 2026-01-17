import pandas as pd
import os

FILE_PATH = "Literature_Screening_List.xlsx"

def fix_data():
    if not os.path.exists(FILE_PATH):
        print("File not found.")
        return

    df = pd.read_excel(FILE_PATH)
    
    # Identify rows with "III" or "3" in Phase column
    # Ensure Phase column is string
    df['Phase'] = df['Phase'].astype(str)
    
    mask = df['Phase'].str.contains("III|3", case=False, na=False)
    
    count = mask.sum()
    print(f"Found {count} papers marked as Phase III.")
    
    if count > 0:
        # Update them
        # Set Study Design to 'Pooled Analysis'
        df.loc[mask, 'Study_Design'] = "Pooled Analysis"
        # Set Phase to 'N/A' to remove purple warning
        df.loc[mask, 'Phase'] = "N/A"
        
        # Print titles for verification
        print("\nUpdated the following papers:")
        for idx, row in df[mask].iterrows():
            print(f"- [Row {idx}] {row['Title'][:50]}...")
            
        df.to_excel(FILE_PATH, index=False)
        print(f"\nSuccessfully updated {count} rows in {FILE_PATH}")
    else:
        print("No Phase III papers found to update.")

if __name__ == "__main__":
    fix_data()
