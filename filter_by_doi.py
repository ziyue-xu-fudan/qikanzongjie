#!/usr/bin/env python3
"""
NEJM文献DOI筛选工具
筛选DOI号包含"NEJMoa"的文献，其余都不要
"""

import pandas as pd
from pathlib import Path
import re

def filter_by_doi_pattern(input_file, doi_pattern="NEJMoa"):
    """
    根据DOI模式筛选NEJM文献
    
    Args:
        input_file: 输入Excel或CSV文件路径
        doi_pattern: DOI匹配模式，默认为"NEJMoa"
    
    Returns:
        筛选后的DataFrame
    """
    print(f"🔍 开始DOI筛选，模式: {doi_pattern}")
    print(f"📁 输入文件: {input_file}")
    
    try:
        # 根据文件扩展名选择读取方式
        input_path = Path(input_file)
        
        if input_path.suffix.lower() == '.csv':
            df = pd.read_csv(input_file)
            print("📄 读取CSV文件")
        elif input_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(input_file)
            print("📊 读取Excel文件")
        else:
            print("❌ 不支持的文件格式")
            return None
        
        print(f"📊 原始文献总数: {len(df)} 篇")
        print(f"📋 数据列: {', '.join(df.columns)}")
        
        # 找到DOI列
        doi_column = None
        possible_doi_columns = ['DOI', 'doi', 'Doi', 'DOI_Number', 'doi_number']
        
        for col in possible_doi_columns:
            if col in df.columns:
                doi_column = col
                break
        
        if doi_column is None:
            # 如果没有找到标准的DOI列，查找包含DOI的列
            doi_columns = [col for col in df.columns if 'doi' in col.lower()]
            if doi_columns:
                doi_column = doi_columns[0]
            else:
                print("⚠️  未找到DOI列，查看前几条数据:")
                print(df.head(2))
                return None
        
        print(f"✅ 使用DOI列: {doi_column}")
        
        # 显示DOI样本
        print("\n📄 DOI样本:")
        sample_dois = df[doi_column].dropna().head(5)
        for i, doi in enumerate(sample_dois, 1):
            print(f"  {i}. {doi}")
        
        # 筛选包含指定模式的DOI
        print(f"\n🎯 正在筛选包含'{doi_pattern}'的DOI...")
        
        # 使用正则表达式进行不区分大小写的匹配
        pattern = re.compile(doi_pattern, re.IGNORECASE)
        filtered_df = df[df[doi_column].notna() & df[doi_column].astype(str).str.contains(pattern, na=False)].copy()
        
        print(f"✅ 筛选完成！")
        print(f"📉 被过滤掉的文献: {len(df) - len(filtered_df)} 篇")
        print(f"📈 幸存的文献: {len(filtered_df)} 篇")
        print(f"💯 存活率: {(len(filtered_df) / len(df) * 100):.1f}%")
        
        # 显示筛选后的DOI样本
        if len(filtered_df) > 0:
            print(f"\n📄 筛选后的DOI样本:")
            filtered_dois = filtered_df[doi_column].head(5)
            for i, doi in enumerate(filtered_dois, 1):
                print(f"  {i}. {doi}")
        
        return filtered_df
        
    except FileNotFoundError:
        print(f"❌ 文件未找到: {input_file}")
        return None
    except Exception as e:
        print(f"❌ 处理文件时出错: {e}")
        return None

def save_filtered_results(filtered_df, input_file, doi_pattern="NEJMoa"):
    """保存筛选结果"""
    if filtered_df is None or filtered_df.empty:
        print("⚠️  没有数据需要保存")
        return
    
    # 生成输出文件名
    input_path = Path(input_file)
    base_name = input_path.stem
    
    # 如果文件名已经包含筛选标记，先移除
    if "_doi_" in base_name:
        base_name = base_name.split("_doi_")[0]
    
    output_name = f"{base_name}_doi_{doi_pattern}{input_path.suffix}"
    output_file = input_path.parent / output_name
    
    try:
        # 保存为相同格式
        if input_path.suffix.lower() == '.csv':
            filtered_df.to_csv(output_file, index=False)
            print(f"💾 CSV筛选结果已保存: {output_file}")
        else:
            # Excel文件
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                filtered_df.to_excel(writer, sheet_name='Filtered_Literature', index=False)
                
                # 获取工作表并添加一些格式化
                worksheet = writer.sheets['Filtered_Literature']
                
                # 设置列宽
                for col_num, col_name in enumerate(filtered_df.columns, 1):
                    col_letter = chr(64 + col_num)
                    if col_name in ['Title', 'Authors', 'Citation']:
                        worksheet.column_dimensions[col_letter].width = 50
                    elif col_name in ['DOI', 'doi']:
                        worksheet.column_dimensions[col_letter].width = 30
                    else:
                        worksheet.column_dimensions[col_letter].width = 15
            
            print(f"💾 Excel筛选结果已保存: {output_file}")
        
        # 生成统计报告
        stats_file = input_path.parent / f"{base_name}_doi_filter_stats.txt"
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write(f"NEJM文献DOI筛选统计报告\n")
            f.write(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"输入文件: {input_file}\n")
            f.write(f"筛选模式: {doi_pattern}\n\n")
            
            f.write(f"原始文献总数: {len(pd.read_csv(input_file)) if input_path.suffix.lower() == '.csv' else len(pd.read_excel(input_file))} 篇\n")
            f.write(f"筛选后文献数: {len(filtered_df)} 篇\n")
            f.write(f"被过滤文献数: {len(pd.read_csv(input_file)) if input_path.suffix.lower() == '.csv' else len(pd.read_excel(input_file))} - {len(filtered_df)} = {len(pd.read_csv(input_file)) - len(filtered_df) if input_path.suffix.lower() == '.csv' else len(pd.read_excel(input_file)) - len(filtered_df)} 篇\n")
            f.write(f"存活率: {(len(filtered_df) / (len(pd.read_csv(input_file)) if input_path.suffix.lower() == '.csv' else len(pd.read_excel(input_file))) * 100):.1f}%\n\n")
            
            # DOI统计
            doi_column = None
            for col in filtered_df.columns:
                if 'doi' in col.lower():
                    doi_column = col
                    break
            
            if doi_column:
                f.write("DOI模式匹配统计:\n")
                doi_counts = {}
                for doi in filtered_df[doi_column].dropna():
                    # 提取DOI前缀
                    if '/' in str(doi):
                        prefix = str(doi).split('/')[0]
                        doi_counts[prefix] = doi_counts.get(prefix, 0) + 1
                
                for prefix, count in sorted(doi_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                    f.write(f"  {prefix}: {count} 篇\n")
        
        print(f"📊 统计报告已保存: {stats_file}")
        
        # 显示前10篇幸存的文献作为样本
        if len(filtered_df) > 0:
            print(f"\n📖 前10篇幸存的文献样本:")
            print("-" * 80)
            
            # 找到关键列
            title_col = next((col for col in ['Title', 'title', 'ArticleTitle'] if col in filtered_df.columns), 'Unknown')
            authors_col = next((col for col in ['Authors', 'authors', 'Author'] if col in filtered_df.columns), 'Unknown')
            pmid_col = next((col for col in ['PMID', 'pmid'] if col in filtered_df.columns), 'Unknown')
            
            for i, (idx, row) in enumerate(filtered_df.head(10).iterrows(), 1):
                title = str(row.get(title_col, '无标题'))[:60] + "..." if len(str(row.get(title_col, ''))) > 60 else str(row.get(title_col, '无标题'))
                authors = str(row.get(authors_col, '未知作者'))
                pmid = str(row.get(pmid_col, '未知PMID'))
                doi = str(row.get(doi_column, '未知DOI'))
                
                print(f"{i:2d}. PMID: {pmid}")
                print(f"    标题: {title}")
                print(f"    作者: {authors[:80]}{'...' if len(authors) > 80 else ''}")
                print(f"    DOI: {doi}")
                print()
        
        return output_file
        
    except Exception as e:
        print(f"❌ 保存文件时出错: {e}")
        return None

def main():
    """主函数"""
    print("🔍 NEJM文献DOI筛选工具")
    print("=" * 60)
    print("专门筛选DOI包含特定模式的NEJM文献")
    print("=" * 60)
    
    # 输入文件路径
    input_file = "/Users/ziyuexu/Documents/trae_projects/paper1/csv-TheNewEngl-set (1)_authors_ge5.csv"
    
    # DOI匹配模式
    doi_pattern = "NEJMoa"
    
    print(f"📁 输入文件: {input_file}")
    print(f"🎯 筛选模式: {doi_pattern}")
    print()
    
    # 执行筛选
    filtered_df = filter_by_doi_pattern(input_file, doi_pattern)
    
    if filtered_df is not None:
        # 保存结果
        output_file = save_filtered_results(filtered_df, input_file, doi_pattern)
        
        if output_file:
            print(f"\n🎉 任务完成！")
            print(f"📄 筛选结果: {output_file}")
            print(f"💾 统计报告: {output_file.parent / f'{Path(input_file).stem}_doi_filter_stats.txt'}")
        else:
            print("\n❌ 结果保存失败")
    else:
        print("\n❌ 筛选过程失败")

if __name__ == "__main__":
    main()