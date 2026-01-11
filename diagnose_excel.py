import pandas as pd
import os

files = [
    "/Users/ziyuexu/Documents/trae_projects/paper1/BMJ.xlsx",
    "/Users/ziyuexu/Documents/trae_projects/paper1/JAMA.xlsx",
    "/Users/ziyuexu/Documents/trae_projects/paper1/Lancet.xlsx",
    "/Users/ziyuexu/Documents/trae_projects/paper1/NEJM.xlsx"
]

for f in files:
    print(f"--- Diagnosing {os.path.basename(f)} ---")
    if not os.path.exists(f):
        print("File does not exist!")
        continue
        
    try:
        # 尝试获取 Excel 文件信息
        xl = pd.ExcelFile(f, engine='openpyxl')
        print(f"Sheet names: {xl.sheet_names}")
        
        if len(xl.sheet_names) > 0:
            df = pd.read_excel(f, sheet_name=0, engine='openpyxl')
            print(f"Successfully read first sheet. Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
        else:
            print("ERROR: No sheets found!")
            
    except Exception as e:
        print(f"ERROR reading file: {e}")
    print("\n")
