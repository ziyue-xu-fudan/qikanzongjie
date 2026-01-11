import sys
import os
import pandas as pd

# 引入解析器
sys.path.append("/Users/ziyuexu/Documents/trae_projects/paper1")
from parse_bmj_abstracts import BMJAbstractParser

def regenerate_bmj():
    print("🔄 Regenerating BMJ.xlsx from source text...")
    
    input_file = "/Users/ziyuexu/Documents/trae_projects/paper1/abstract-BMJJournal-set (2).txt"
    output_file = "/Users/ziyuexu/Documents/trae_projects/paper1/BMJ.xlsx"
    
    if not os.path.exists(input_file):
        print(f"❌ Source file not found: {input_file}")
        return
        
    parser = BMJAbstractParser()
    articles = parser.parse_bmj_abstracts(input_file)
    
    if articles:
        df = parser.create_dataframe()
        if not df.empty:
            print(f"📊 Extracted {len(df)} articles.")
            
            # 关键步骤：使用最简单的 openpyxl 写入，不搞复杂的格式化，确保兼容性
            try:
                # 统一列名以匹配其他文件 (NEJM等)
                # NEJM columns: ['PMID', 'Title', 'Authors', 'Citation', 'First Author', 'Journal/Book', ...]
                # BMJ columns: ['文章编号', 'PMID', 'PMCID', 'DOI', '标题', '作者', ...]
                # 我们尽量保留所有信息，但把 '摘要' 列对应到 'Abstract' 
                # 等等，BMJ 解析出来的列里好像没有 'Abstract' ?
                # 看代码，它提取了 '研究目的', '研究设计', '结果', '结论' 等。
                # 我们需要把这些合并成 'Abstract' 列，以便 PaperWorkflow 处理。
                
                # 合并摘要部分
                def combine_abstract(row):
                    parts = []
                    if row.get('研究目的'): parts.append(f"OBJECTIVE: {row['研究目的']}")
                    if row.get('研究设计'): parts.append(f"DESIGN: {row['研究设计']}")
                    if row.get('研究设置'): parts.append(f"SETTING: {row['研究设置']}")
                    if row.get('参与者'): parts.append(f"PARTICIPANTS: {row['参与者']}")
                    if row.get('主要结果测量'): parts.append(f"MAIN OUTCOME MEASURES: {row['主要结果测量']}")
                    if row.get('结果'): parts.append(f"RESULTS: {row['结果']}")
                    if row.get('结论'): parts.append(f"CONCLUSIONS: {row['结论']}")
                    return " ".join(parts)
                
                df['Abstract'] = df.apply(combine_abstract, axis=1)
                
                # 重命名一些关键列
                df = df.rename(columns={
                    '标题': 'Title',
                    '作者': 'Authors',
                    '发表年份': 'Publication Year'
                })
                
                # 写入 Excel
                with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Filtered_Literature', index=False)
                
                print(f"✅ Successfully regenerated {output_file}")
                
                # 验证
                test_df = pd.read_excel(output_file, engine='openpyxl')
                print(f"✅ Verification successful! Read {len(test_df)} rows.")
                
            except Exception as e:
                print(f"❌ Failed to save Excel: {e}")
        else:
            print("❌ DataFrame is empty.")
    else:
        print("❌ No articles parsed.")

if __name__ == "__main__":
    regenerate_bmj()
