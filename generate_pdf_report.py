import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import datetime
import numpy as np

# -----------------------------------------------------------------------------
# 配置与数据加载
# -----------------------------------------------------------------------------
FILE_PATH = "/Users/ziyuexu/Documents/trae_projects/paper1/multi_journal_analysis_report.xlsx"
OUTPUT_PDF = "Medical_Journal_Analysis_Report.pdf"

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']

# 定义颜色
ACCENT_COLOR = "#2E86C1"
BAR_COLORS = plt.cm.viridis(np.linspace(0.2, 0.8, 20))

def load_data():
    try:
        df = pd.read_excel(FILE_PATH, engine='openpyxl')
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# -----------------------------------------------------------------------------
# 绘图辅助函数 (纯 Matplotlib 实现)
# -----------------------------------------------------------------------------
def create_title_page(pdf, total_papers, date_range):
    """创建封面页"""
    plt.figure(figsize=(11.69, 8.27))
    plt.axis('off')
    
    plt.text(0.5, 0.7, "Medical Journal Analysis Report", 
             ha='center', va='center', fontsize=36, weight='bold', color='#2C3E50')
    
    summary_text = (
        f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d')}\n\n"
        f"Total Papers Analyzed: {total_papers}\n"
        f"Data Source: {FILE_PATH.split('/')[-1]}\n"
        f"Date Range: {date_range}"
    )
    plt.text(0.5, 0.4, summary_text, 
             ha='center', va='center', fontsize=16, color='#566573', linespacing=1.8)
    
    plt.plot([0.1, 0.9], [0.1, 0.1], color=ACCENT_COLOR, linewidth=5)
    
    pdf.savefig()
    plt.close()

def plot_bar_chart(data, title, xlabel, ylabel, pdf, top_n=None, orientation='h'):
    """通用条形图"""
    plt.figure(figsize=(11, 7))
    
    if top_n:
        data = data.head(top_n)
    
    # 颜色
    colors = plt.cm.viridis(np.linspace(0.3, 0.8, len(data)))
    
    if orientation == 'h':
        bars = plt.barh(data.index, data.values, color=colors)
        plt.gca().invert_yaxis() # 让最高的在上面
        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            plt.text(width + (max(data.values)*0.01), bar.get_y() + bar.get_height()/2, 
                     f'{int(width)}', va='center', fontsize=10)
    else:
        bars = plt.bar(data.index, data.values, color=colors)
        plt.xticks(rotation=45, ha='right')
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + (max(data.values)*0.01), 
                     f'{int(height)}', ha='center', fontsize=10)

    plt.title(title, fontsize=18, pad=20, weight='bold', color='#2C3E50')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    plt.tight_layout()
    pdf.savefig()
    plt.close()

def plot_pie_chart(data, title, pdf):
    """通用饼图"""
    plt.figure(figsize=(10, 7))
    
    if len(data) > 8:
        top_8 = data.head(8)
        others_sum = data.iloc[8:].sum()
        others_series = pd.Series([others_sum], index=['Others'])
        data = pd.concat([top_8, others_series])
    
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(data)))
    
    plt.pie(data.values, labels=data.index, autopct='%1.1f%%', 
            startangle=140, colors=colors, 
            textprops={'fontsize': 11})
            
    plt.title(title, fontsize=18, pad=20, weight='bold', color='#2C3E50')
    plt.axis('equal')
    
    pdf.savefig()
    plt.close()

def plot_heatmap(df, pdf, journal_col='Journal'):
    """绘制期刊与研究设计的热力图"""
    plt.figure(figsize=(12, 8))
    
    ct = pd.crosstab(df[journal_col], df['Research Design'])
    # 归一化
    ct_norm = ct.div(ct.sum(axis=1), axis=0)
    
    # 绘制热力图 (手动)
    plt.imshow(ct_norm, cmap='YlGnBu', aspect='auto')
    
    # 设置坐标轴
    plt.xticks(range(len(ct_norm.columns)), ct_norm.columns, rotation=45, ha='right')
    plt.yticks(range(len(ct_norm.index)), ct_norm.index)
    
    # 添加数值
    for i in range(len(ct_norm.index)):
        for j in range(len(ct_norm.columns)):
            val = ct_norm.iloc[i, j]
            if val > 0.1: # 只显示比较大的值，避免太乱
                plt.text(j, i, f"{val:.1f}", ha="center", va="center", color="black" if val < 0.7 else "white", fontsize=8)

    plt.colorbar(label='Proportion')
    plt.title("Research Design Distribution by Journal", fontsize=18, pad=20, weight='bold', color='#2C3E50')
    plt.xlabel("Research Design", fontsize=12)
    plt.ylabel("Journal", fontsize=12)
    
    plt.tight_layout()
    pdf.savefig()
    plt.close()

# -----------------------------------------------------------------------------
# 主生成逻辑
# -----------------------------------------------------------------------------
def main():
    print("🚀 Starting PDF Report Generation (Matplotlib Only)...")
    df = load_data()
    if df is None:
        return

    date_range = "N/A"
    if 'Publication Date' in df.columns:
        try:
            df['Publication Date'] = pd.to_datetime(df['Publication Date'], errors='coerce')
            min_date = df['Publication Date'].min().strftime('%Y-%m')
            max_date = df['Publication Date'].max().strftime('%Y-%m')
            date_range = f"{min_date} to {max_date}"
        except:
            pass

    with PdfPages(OUTPUT_PDF) as pdf:
        # 1. 封面
        create_title_page(pdf, len(df), date_range)
        print("✅ Cover page created.")

        # 2. 期刊分布
        journal_col = 'Journal/Book' if 'Journal/Book' in df.columns else 'Journal'
        journal_counts = df[journal_col].value_counts()
        plot_bar_chart(journal_counts, "Top Journals by Publication Volume", 
                       "Number of Publications", "Journal", pdf, top_n=10)
        print("✅ Journal distribution chart created.")

        # 3. 研究设计分布
        design_counts = df['Research Design'].value_counts()
        plot_bar_chart(design_counts, "Distribution of Research Designs", 
                       "Count", "Research Design", pdf, top_n=15)
        print("✅ Research Design chart created.")

        # 4. 研究时序分布
        timing_counts = df['Study Timing'].value_counts()
        plot_pie_chart(timing_counts, "Study Timing Distribution", pdf)
        print("✅ Study Timing chart created.")

        # 5. 聚焦疾病系统
        system_counts = df['Focused Disease System'].value_counts()
        plot_bar_chart(system_counts, "Top Focused Disease Systems", 
                       "Count", "Disease System", pdf, top_n=15)
        print("✅ Disease System chart created.")

        # 6. 具体疾病 (Top 20)
        disease_counts = df['Focused Disease'].value_counts()
        disease_counts = disease_counts[disease_counts.index != "Not Applicable"]
        plot_bar_chart(disease_counts, "Top 20 Specific Diseases/Conditions", 
                       "Count", "Disease/Condition", pdf, top_n=20)
        print("✅ Specific Disease chart created.")

        # 7. 热力图
        plot_heatmap(df, pdf, journal_col)
        print("✅ Heatmap created.")

        # 8. 国家分布
        if 'Research Team Country' in df.columns:
            country_counts = df['Research Team Country'].value_counts()
            plot_bar_chart(country_counts, "Top Research Team Countries", 
                           "Count", "Country", pdf, top_n=15)
            print("✅ Country chart created.")

    print(f"🎉 PDF Report successfully generated: {OUTPUT_PDF}")

if __name__ == "__main__":
    main()
