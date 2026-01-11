import pandas as pd
import shutil
import os

def fix_bmj():
    print("🚑 Fixing BMJ.xlsx...")
    source = "bmj_articles_parsed.xlsx"
    target = "BMJ.xlsx"
    if os.path.exists(source):
        shutil.copy2(source, target)
        print(f"✅ Replaced {target} with valid file {source}")
    else:
        print(f"❌ Source file {source} not found!")

def fix_from_csv(csv_path, target_xlsx):
    print(f"🚑 Fixing {target_xlsx} from {csv_path}...")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            print(f"Read {len(df)} rows from CSV.")
            
            # 使用 openpyxl 引擎重新生成标准的 Excel
            with pd.ExcelWriter(target_xlsx, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Filtered_Literature', index=False)
            print(f"✅ Successfully regenerated {target_xlsx}")
        except Exception as e:
            print(f"❌ Failed to regenerate: {e}")
    else:
        print(f"❌ Source CSV {csv_path} not found!")

def main():
    # 1. Fix BMJ
    fix_bmj()
    
    # 2. Fix JAMA
    fix_from_csv("csv-JAMAJourna-set.csv", "JAMA.xlsx")
    
    # 3. Fix Lancet
    fix_from_csv("csv-LancetJour-set.csv", "Lancet.xlsx")

if __name__ == "__main__":
    main()
