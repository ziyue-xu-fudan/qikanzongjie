import streamlit as st
import pandas as pd
import os
from paper_workflow import PaperWorkflow
import plotly.express as px
from io import BytesIO

# 设置页面配置
st.set_page_config(
    page_title="医学文献智能分析工作流",
    page_icon="🧬",
    layout="wide"
)

# 标题
st.title("🧬 医学文献智能分析工作流")
st.markdown("集成 PubMed 检索与 AI 深度分析，自动提取研究特征并可视化。")

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 基础配置")
    
    # API Key 输入
    default_keys = """sk-37c1617db0da456d8491e1094e3f6ae3
sk-82a00766192049fc91da7edbca74bfd2
sk-c69f18b962d54e44b14298f079bc4c66
sk-d98eb5841a0b4e6c9985b72b4106c74c"""
    
    api_key_input = st.text_area("DeepSeek API Keys (每行一个)", height=100, value=default_keys, help="请输入您的 DeepSeek API Key，支持多个 Key 轮询使用 (https://platform.deepseek.com/api_keys)")
    api_keys = [k.strip() for k in api_key_input.split('\n') if k.strip()]
    
    st.divider()
    
    st.header("🔧 高级配置")
    with st.expander("模型与 Prompt 设置", expanded=False):
        # 模型选择
        model_options = [
            "deepseek-chat",
            "deepseek-reasoner",
        ]
        selected_model = st.selectbox("选择大模型", model_options, index=0)
        custom_model = st.text_input("或手动输入模型名称", help="如果在列表中未找到，可在此输入 DeepSeek 支持的其他模型名称")
        
        final_model = custom_model if custom_model else selected_model
        
        # Prompt 编辑
        default_prompt = """请分析以下医学文献摘要，并提取以下五个关键信息。
请严格按照 JSON 格式返回，不要包含 Markdown 格式标记（如 ```json）。
如果无法提取某个字段，请填写 "Unknown"。

摘要内容:
{abstract}

需要提取的字段:
1. research_design (研究方式 - 流行病学/临床分类)
   - 请从以下列表中选择最匹配的一项:
     [Randomized Controlled Trial, Cohort Study, Case-Control Study, Cross-sectional Study, Systematic Review, Meta-analysis, Case Report, Animal Study, In Vitro Study, Narrative Review, Clinical Observation]
   - 如果都不匹配，请填写 "Other".

2. study_timing (时序类型)
   - 请判断研究的时序性质:
     [Retrospective, Prospective, Cross-sectional, Ambispective, Not Applicable]
   - Not Applicable 适用于综述或实验室研究。

3. focused_disease_system (疾病系统)
   - 请归类疾病所属的系统，例如:
     [Cardiovascular, Respiratory, Nervous, Digestive, Endocrine, Immune, Musculoskeletal, Urinary, Reproductive, Integumentary, Oncology, Infectious Disease, Other]

4. focused_disease (聚焦疾病具体名称)
   - 请提取主要的疾病名称。
   - **重要**: 请尽可能提供该疾病对应的 ICD-10 编码，格式为 "Disease Name (ICD-10 Code)"。例如: "Type 2 Diabetes Mellitus (E11)".

5. research_team_country (研究团队主要国家)
   - 请提取通讯作者或第一作者所在的国家。
   - 请使用标准的英文国家名称。

JSON 格式示例:
{{
    "research_design": "Cohort Study",
    "study_timing": "Prospective",
    "focused_disease_system": "Cardiovascular",
    "focused_disease": "Hypertension (I10)",
    "research_team_country": "USA"
}}"""
        st.markdown("### 自定义 Prompt")
        st.info("💡 请保留 `{abstract}` 占位符，以便插入实际摘要内容。")
        user_prompt = st.text_area("编辑 Prompt 模板", value=default_prompt, height=400)

    st.divider()
    
    st.header("📂 文件选择")
    # 默认文件路径
    default_files = [
        "/Users/ziyuexu/Documents/trae_projects/paper1/NEJM.xlsx",
        "/Users/ziyuexu/Documents/trae_projects/paper1/BMJ.xlsx",
        "/Users/ziyuexu/Documents/trae_projects/paper1/JAMA.xlsx",
        "/Users/ziyuexu/Documents/trae_projects/paper1/Lancet.xlsx"
    ]
    
    # 文件选择模式
    input_mode = st.radio("选择文件来源", ["使用默认文件", "上传新文件"])
    
    selected_files = []
    if input_mode == "使用默认文件":
        st.info("将处理以下文件：")
        for f in default_files:
            if os.path.exists(f):
                st.success(f"✅ {os.path.basename(f)}")
                selected_files.append(f)
            else:
                st.error(f"❌ {os.path.basename(f)} (未找到)")
    else:
        uploaded_files = st.file_uploader("上传 Excel 文件", type=['xlsx', 'csv'], accept_multiple_files=True)
        if uploaded_files:
            selected_files = uploaded_files

    start_btn = st.button("🚀 开始分析", type="primary", disabled=not api_keys or not selected_files)

# 主界面逻辑
if start_btn:
    # 初始化 Workflow (使用多 Key 轮询)
    workflow = PaperWorkflow(api_keys=api_keys, model=final_model)
    
    st.success(f"已启动分析工作流，使用模型: **{final_model}**")
    
    # 创建总体进度条
    total_progress = st.progress(0, text="准备开始...")
    
    # 用于存储所有结果的列表
    all_results = []
    
    # 创建 tabs 分别显示每个文件的状态
    file_names = [os.path.basename(f) if isinstance(f, str) else f.name for f in selected_files]
    tabs = st.tabs(file_names)
    
    for i, file_obj in enumerate(selected_files):
        with tabs[i]:
            st.subheader(f"正在处理: {file_names[i]}")
            
            # 读取文件 (增加健壮性)
            try:
                if isinstance(file_obj, str):
                    if file_obj.endswith('.csv'):
                        df = pd.read_csv(file_obj)
                    else:
                        # 显式使用 openpyxl 引擎，避免默认引擎的不确定性
                        df = pd.read_excel(file_obj, engine='openpyxl')
                else:
                    if file_obj.name.endswith('.csv'):
                        df = pd.read_csv(file_obj)
                    else:
                        df = pd.read_excel(file_obj, engine='openpyxl')
                
                st.write(f"📊 读取到 {len(df)} 条记录")
                
                # 创建单文件进度条
                file_progress = st.progress(0, text="初始化...")
                status_text = st.empty()
                
                # 创建占位符用于实时显示数据
                result_placeholder = st.empty()
                
                # 定义回调函数更新进度
                def update_progress(current, total, message):
                    percent = int(current / total * 100)
                    file_progress.progress(percent, text=f"{percent}% - {message}")
                    
                    # 实时刷新数据预览 (每处理5条或最后一条刷新一次，避免过于频繁)
                    if current % 5 == 0 or current == total:
                        # 优化 DataFrame 显示列顺序
                        display_cols = ['PMID', 'Title', 'Abstract', 'Research Design', 'Study Timing', 'Focused Disease System', 'Focused Disease', 'Target Population', 'Research Team Country']
                        # 确保所有列都存在
                        display_cols = [col for col in display_cols if col in df.columns]
                        # 添加其他可能存在的列
                        other_cols = [col for col in df.columns if col not in display_cols]
                        final_display_df = df[display_cols + other_cols]
                        
                        result_placeholder.dataframe(final_display_df.head(min(current, 20)))

                # 处理数据
                with st.spinner("正在检索摘要并进行 AI 分析..."):
                    # 传递自定义 Prompt
                    processed_df = workflow.process_dataframe(
                        df, 
                        custom_prompt=user_prompt,
                        progress_callback=update_progress
                    )
                
                st.success("✅ 处理完成！")
                
                # 最终显示
                display_cols = ['PMID', 'Title', 'Abstract', 'Research Design', 'Study Timing', 'Focused Disease System', 'Focused Disease', 'Target Population', 'Research Team Country']
                display_cols = [col for col in display_cols if col in processed_df.columns]
                other_cols = [col for col in processed_df.columns if col not in display_cols]
                final_display_df = processed_df[display_cols + other_cols]
                result_placeholder.dataframe(final_display_df.head())

                # 详细视图：逐行展示
                with st.expander("👁️ 查看详细分析结果 (Title, Abstract & Analysis)"):
                    for idx, row in processed_df.iterrows():
                        st.markdown(f"### 📄 {row.get('Title', 'No Title')}")
                        st.markdown(f"**PMID**: {row.get('PMID', 'N/A')}")
                        
                        col_a, col_b = st.columns([1, 1])
                        with col_a:
                            st.markdown("#### 📝 摘要 (Abstract)")
                            st.info(row.get('Abstract', 'No Abstract Available'))
                        
                        with col_b:
                            st.markdown("#### 🤖 AI 分析结果")
                            st.write(f"**🔬 研究方式**: {row.get('Research Design', 'N/A')}")
                            st.write(f"**⏱️ 时序类型**: {row.get('Study Timing', 'N/A')}")
                            st.write(f"**🫁 疾病系统**: {row.get('Focused Disease System', 'N/A')}")
                            st.write(f"**🧬 聚焦疾病**: {row.get('Focused Disease', 'N/A')}")
                            st.write(f"**👥 目标人群**: {row.get('Target Population', 'N/A')}")
                            st.write(f"**🌍 研究国家**: {row.get('Research Team Country', 'N/A')}")
                        
                        st.divider()
                
                processed_df['Source File'] = file_names[i]
                all_results.append(processed_df)
                
                # 提供单文件下载
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    processed_df.to_excel(writer, index=False)
                
                st.download_button(
                    label=f"📥 下载 {file_names[i]} 结果",
                    data=output.getvalue(),
                    file_name=f"analyzed_{file_names[i]}",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
            except Exception as e:
                st.error(f"处理文件 {file_names[i]} 时出错: {str(e)}")
                # 即使出错也继续处理下一个文件
        
        # 更新总体进度
        total_progress.progress(int((i + 1) / len(selected_files) * 100), text=f"总体进度: {i + 1}/{len(selected_files)}")

    # 汇总分析
    if all_results:
        st.divider()
        st.header("📈 汇总分析报告")
        
        final_df = pd.concat(all_results, ignore_index=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌍 研究团队国家分布")
            if 'Research Team Country' in final_df.columns:
                country_counts = final_df['Research Team Country'].value_counts().reset_index()
                country_counts.columns = ['Country', 'Count']
                fig_country = px.bar(country_counts.head(10), x='Country', y='Count', title="Top 10 研究国家")
                st.plotly_chart(fig_country, use_container_width=True)
        
        with col2:
            st.subheader("🧬 聚焦疾病分布")
            if 'Focused Disease' in final_df.columns:
                disease_counts = final_df['Focused Disease'].value_counts().reset_index()
                disease_counts.columns = ['Disease', 'Count']
                fig_disease = px.pie(disease_counts.head(10), values='Count', names='Disease', title="Top 10 聚焦疾病")
                st.plotly_chart(fig_disease, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("🔬 研究方式分布")
            if 'Research Design' in final_df.columns:
                design_counts = final_df['Research Design'].value_counts().reset_index()
                design_counts.columns = ['Design', 'Count']
                fig_design = px.bar(design_counts.head(10), x='Design', y='Count', title="研究方式分布")
                st.plotly_chart(fig_design, use_container_width=True)
                
        with col4:
            st.subheader("📊 数据预览")
            st.dataframe(final_df)

        # 汇总下载
        output_all = BytesIO()
        with pd.ExcelWriter(output_all, engine='openpyxl') as writer:
            final_df.to_excel(writer, index=False)
            
        st.download_button(
            label="📥 下载所有汇总结果",
            data=output_all.getvalue(),
            file_name="all_analyzed_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )

else:
    st.info("👈 请在侧边栏输入 API Key 并点击开始分析")
