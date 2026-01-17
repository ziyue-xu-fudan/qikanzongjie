import pandas as pd
import os

# 1. Load Data
input_file = "Top4_NonRCT_Cancer_Papers_2023_2026.xlsx"
if not os.path.exists(input_file):
    print(f"Error: {input_file} not found. Please run fetch script first.")
    exit(1)

df = pd.read_excel(input_file)

# 2. Select and Rename Columns
# Keeping only essential columns for screening
columns_to_keep = [
    "Journal_Category", 
    "Title", 
    "PubDate", 
    "DOI", 
    "PMID",
    "Abstract"
]

# Filter columns that actually exist
existing_columns = [col for col in columns_to_keep if col in df.columns]
df_screen = df[existing_columns].copy()

# 3. Add Screening Columns (Empty)
df_screen.insert(0, "Select? (Y/N)", "")
df_screen.insert(1, "Reason / Notes", "")

# 4. Formatting
# Sort by Journal then Date (if possible, otherwise just Journal)
df_screen = df_screen.sort_values(by=["Journal_Category", "PubDate"], ascending=[True, False])

# 5. Save to new Excel
output_file = "Literature_Screening_List.xlsx"
df_screen.to_excel(output_file, index=False)

print(f"Successfully created screening list: {output_file}")
print(f"Total papers: {len(df_screen)}")
print("\nColumns included:")
for col in df_screen.columns:
    print(f"- {col}")
