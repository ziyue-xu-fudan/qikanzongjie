#!/usr/bin/env python3
"""
NEJM文献筛选工具
筛选作者数量≥5的文献，枪毙掉作者少于5个的文章
"""

import pandas as pd
import sys
from pathlib import Path

def count_authors(authors_str):
    """计算作者数量"""
    if pd.isna(authors_str) or not authors_str:
        return 0
    
    # 移除多余的空格和换行符
    authors_str = str(authors_str).strip()
    
    # 通过逗号分隔来计算作者数量
    # 有些作者名字可能包含逗号，所以需要更智能的处理
    authors = [author.strip() for author in authors_str.split(',') if author.strip()]
    
    return len(authors)

def filter_nejm_by_author_count(input_file, min_authors=5):
    """
    筛选NEJM文献，只保留作者数量≥指定数量的文章
    
    Args:
        input_file: 输入CSV文件路径
        min_authors: 最小作者数量（默认5个）
    
    Returns:
        筛选后的DataFrame
    """
    print(f"🔫 开始执行'枪毙'操作，目标：作者数量<{min_authors}的文献")
    print(f"📁 输入文件: {input_file}")
    
    try:
        # 读取CSV文件
        df = pd.read_csv(input_file)
        print(f"📊 原始文献总数: {len(df)} 篇")
        
        # 显示列名，帮助我们理解数据结构
        print(f"📋 数据列: {', '.join(df.columns)}")
        
        # 找到Authors列（可能有不同的列名）
        authors_column = None
        possible_author_columns = ['Authors', 'authors', 'Author', 'author', 'Authors_list']
        
        for col in possible_author_columns:
            if col in df.columns:
                authors_column = col
                break
        
        if authors_column is None:
            # 如果没有找到标准的作者列，查看所有列名
            print("⚠️  未找到标准的作者列，查看前几条数据:")
            print(df.head(2))
            return None
        
        print(f"✅ 使用作者列: {authors_column}")
        
        # 计算每篇文章的作者数量
        print("🧮 正在计算每篇文章的作者数量...")
        df['author_count'] = df[authors_column].apply(count_authors)
        
        # 显示作者数量分布
        author_dist = df['author_count'].value_counts().sort_index()
        print("📈 作者数量分布:")
        for count, freq in author_dist.head(10).items():
            print(f"  作者数量 {count}: {freq} 篇")
        
        # 筛选作者数量≥指定数量的文献
        print(f"🎯 正在筛选作者数量≥{min_authors}的文献...")
        filtered_df = df[df['author_count'] >= min_authors].copy()
        
        print(f"✅ 筛选完成！")
        print(f"📉 被'枪毙'的文献: {len(df) - len(filtered_df)} 篇")
        print(f"📈 幸存的文献: {len(filtered_df)} 篇")
        print(f"💯 存活率: {(len(filtered_df) / len(df) * 100):.1f}%")
        
        # 显示筛选后的作者数量分布
        filtered_dist = filtered_df['author_count'].value_counts().sort_index()
        print(f"\n📊 筛选后的作者数量分布:")
        for count, freq in filtered_dist.head(10).items():
            print(f"  作者数量 {count}: {freq} 篇")
        
        return filtered_df
        
    except FileNotFoundError:
        print(f"❌ 文件未找到: {input_file}")
        return None
    except Exception as e:
        print(f"❌ 处理文件时出错: {e}")
        return None

def save_filtered_results(filtered_df, input_file, min_authors=5):
    """保存筛选结果"""
    if filtered_df is None or filtered_df.empty:
        print("⚠️  没有数据需要保存")
        return
    
    # 生成输出文件名
    input_path = Path(input_file)
    output_name = f"{input_path.stem}_authors_ge{min_authors}{input_path.suffix}"
    output_file = input_path.parent / output_name
    
    try:
        # 保存为CSV
        filtered_df.to_csv(output_file, index=False)
        print(f"💾 筛选结果已保存: {output_file}")
        
        # 生成统计报告
        stats_file = input_path.parent / f"{input_path.stem}_filter_stats.txt"
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write(f"NEJM文献筛选统计报告\n")
            f.write(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"输入文件: {input_file}\n")
            f.write(f"筛选条件: 作者数量 ≥ {min_authors}\n\n")
            
            f.write(f"原始文献总数: {len(pd.read_csv(input_file))} 篇\n")
            f.write(f"筛选后文献数: {len(filtered_df)} 篇\n")
            f.write(f"被移除文献数: {len(pd.read_csv(input_file)) - len(filtered_df)} 篇\n")
            f.write(f"存活率: {(len(filtered_df) / len(pd.read_csv(input_file)) * 100):.1f}%\n\n")
            
            f.write("筛选后的作者数量分布:\n")
            filtered_dist = filtered_df['author_count'].value_counts().sort_index()
            for count, freq in filtered_dist.head(15).items():
                f.write(f"  作者数量 {count}: {freq} 篇\n")
            
            f.write(f"\n作者数量统计:\n")
            f.write(f"  最少作者数: {filtered_df['author_count'].min()}\n")
            f.write(f"  最多作者数: {filtered_df['author_count'].max()}\n")
            f.write(f"  平均作者数: {filtered_df['author_count'].mean():.1f}\n")
            f.write(f"  中位数作者数: {filtered_df['author_count'].median():.1f}\n")
        
        print(f"📊 统计报告已保存: {stats_file}")
        
        # 显示前10篇幸存的文献作为样本
        print(f"\n📖 前10篇幸存的文献样本:")
        print("-" * 80)
        for i, (idx, row) in enumerate(filtered_df.head(10).iterrows(), 1):
            title = str(row.get('Title', '无标题'))[:60] + "..." if len(str(row.get('Title', ''))) > 60 else str(row.get('Title', '无标题'))
            authors = str(row.get('authors', row.get('Authors', '未知作者')))
            author_count = row['author_count']
            pmid = str(row.get('PMID', '未知PMID'))
            
            print(f"{i:2d}. PMID: {pmid}")
            print(f"    标题: {title}")
            print(f"    作者: {authors[:80]}{'...' if len(authors) > 80 else ''}")
            print(f"    作者数量: {author_count}")
            print()
        
        return output_file
        
    except Exception as e:
        print(f"❌ 保存文件时出错: {e}")
        return None

def main():
    """主函数"""
    print("🔫 NEJM文献'枪毙'工具")
    print("=" * 60)
    print("专门筛选作者数量≥5的NEJM文献")
    print("=" * 60)
    
    # 输入文件路径
    input_file = "/Users/ziyuexu/Documents/trae_projects/paper1/csv-TheNewEngl-set (1).csv"
    
    # 最小作者数量
    min_authors = 5
    
    print(f"📁 输入文件: {input_file}")
    print(f"🎯 筛选条件: 作者数量 ≥ {min_authors}")
    print()
    
    # 执行筛选
    filtered_df = filter_nejm_by_author_count(input_file, min_authors)
    
    if filtered_df is not None:
        # 保存结果
        output_file = save_filtered_results(filtered_df, input_file, min_authors)
        
        if output_file:
            print(f"\n🎉 任务完成！")
            print(f"📄 筛选结果: {output_file}")
            print(f"💾 统计报告: {output_file.parent / f'{Path(input_file).stem}_filter_stats.txt'}")
        else:
            print("\n❌ 结果保存失败")
    else:
        print("\n❌ 筛选过程失败")

if __name__ == "__main__":
    main()