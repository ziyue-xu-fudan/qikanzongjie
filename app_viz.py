import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# 设置页面配置
st.set_page_config(
    page_title="医学文献分析报告可视化",
    page_icon="📊",
    layout="wide"
)

st.title("📊 医学文献高级分析报告")
st.markdown("基于 `multi_journal_analysis_report.xlsx` 的可视化展示")

# 文件路径
FILE_PATH = "multi_journal_analysis_report.xlsx"

@st.cache_data
def load_data():
    if not os.path.exists(FILE_PATH):
        return None
    try:
        df = pd.read_excel(FILE_PATH, engine='openpyxl')
        return df
    except Exception as e:
        st.error(f"读取文件失败: {e}")
        return None

df = load_data()

if df is not None:
    st.success(f"✅ 成功加载数据，共 {len(df)} 条记录")
    
    # 过滤 Clinical Trial
    st.sidebar.header("🔍 数据筛选")
    exclude_trials = st.sidebar.checkbox("排除 Clinical Trials (RCT)", value=True)
    
    if exclude_trials:
        original_count = len(df)
        df = df[df['Research Design'] != 'Randomized Controlled Trial']
        filtered_count = len(df)
        st.info(f"已过滤掉 {original_count - filtered_count} 条 Clinical Trial 数据，剩余 {filtered_count} 条")

    # 数据预览
    with st.expander("查看原始数据"):
        st.dataframe(df)

    st.divider()
    
    # --- 图表组 1: 研究全景 ---
    st.header("1. 研究方法学全景 (Research Landscape)")
    tab1, tab2, tab3 = st.tabs(["研究设计分布", "时序类型", "设计 x 时序关联"])
    
    with tab1:
        if 'Research Design' in df.columns:
            # 统计各杂志的设计分布
            fig = px.histogram(df, x='Source File', color='Research Design', 
                               title="各杂志研究设计构成", barmode='group')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if 'Study Timing' in df.columns:
            fig = px.pie(df, names='Study Timing', title="总体时序类型分布", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if 'Research Design' in df.columns and 'Study Timing' in df.columns:
            try:
                heatmap_data = pd.crosstab(df['Research Design'], df['Study Timing'])
                fig = px.imshow(heatmap_data, text_auto=True, title="研究设计 vs 时序类型 热力图")
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.info("数据不足以生成热力图")

    # --- 图表组 2: 临床焦点 ---
    st.header("2. 临床疾病焦点 (Clinical Focus)")
    c1, c2 = st.columns(2)
    
    with c1:
        if 'Focused Disease System' in df.columns:
            # 简单的清洗，去除空值
            sys_df = df.dropna(subset=['Focused Disease System'])
            sys_counts = sys_df['Focused Disease System'].value_counts().reset_index()
            sys_counts.columns = ['System', 'Count']
            if not sys_counts.empty:
                fig = px.treemap(sys_counts, path=['System'], values='Count', title="疾病系统分布 (Treemap)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("无有效疾病系统数据")
            
    with c2:
        if 'Focused Disease' in df.columns:
            dis_df = df.dropna(subset=['Focused Disease'])
            dis_counts = dis_df['Focused Disease'].value_counts().head(10).reset_index()
            dis_counts.columns = ['Disease', 'Count']
            if not dis_counts.empty:
                fig = px.bar(dis_counts, x='Count', y='Disease', orientation='h', title="Top 10 聚焦疾病")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("无有效疾病名称数据")

    # --- 图表组 3: 全球视野 ---
    st.header("3. 全球研究视野 (Global View)")
    g1, g2 = st.columns(2)
    
    with g1:
        if 'Research Team Country' in df.columns:
            country_df = df.dropna(subset=['Research Team Country'])
            country_counts = country_df['Research Team Country'].value_counts().reset_index()
            country_counts.columns = ['Country', 'Count']
            if not country_counts.empty:
                fig = px.choropleth(country_counts, locations="Country", locationmode='country names',
                                    color="Count", hover_name="Country", title="全球发文量分布")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("无有效国家数据")

    with g2:
        if 'Research Team Country' in df.columns and 'Research Design' in df.columns:
            target_countries = ['China', 'USA', 'United States', 'China (Mainland)']
            mask = df['Research Team Country'].isin(target_countries)
            if mask.any():
                plot_df = df[mask].copy()
                # 统一国家名称
                plot_df['Country'] = plot_df['Research Team Country'].apply(
                    lambda x: 'USA' if isinstance(x, str) and ('United States' in x or 'USA' in x) else 'China'
                )
                
                design_by_country = pd.crosstab(plot_df['Research Design'], plot_df['Country'])
                
                # 确保有数据才画图
                if not design_by_country.empty:
                    categories = design_by_country.index.tolist()
                    fig = go.Figure()
                    
                    if 'China' in design_by_country.columns:
                        fig.add_trace(go.Scatterpolar(r=design_by_country['China'], theta=categories, fill='toself', name='China'))
                    if 'USA' in design_by_country.columns:
                        fig.add_trace(go.Scatterpolar(r=design_by_country['USA'], theta=categories, fill='toself', name='USA'))
                    
                    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), title="中美研究设计偏好对比")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("中美对比数据为空")
            else:
                st.info("暂无中美相关数据")

else:
    st.error(f"无法找到文件: {FILE_PATH}")
