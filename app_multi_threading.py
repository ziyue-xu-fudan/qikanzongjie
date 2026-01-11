import streamlit as st
import pandas as pd
import os
from paper_workflow import PaperWorkflow
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import concurrent.futures
from Bio import Entrez
import time
import queue

# 设置页面配置
st.set_page_config(
    page_title="医学文献智能分析工作流 (Pro)",
    page_icon="🧬",
    layout="wide"
)

# 标题
st.title("🧬 医学文献智能分析工作流 (Pro)")
st.markdown("集成 PubMed 检索与 AI 深度分析，**一杂志一 Key** 极速并发。")

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 配置")
    
    st.markdown("### 🔧 模型与 Prompt")
    with st.expander("模型与 Prompt 设置", expanded=False):
        # 模型选择
        model_options = [
            "deepseek-chat",
            "deepseek-reasoner",
        ]
        selected_model = st.selectbox("选择大模型", model_options, index=0)
        
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
        user_prompt = st.text_area("编辑 Prompt 模板", value=default_prompt, height=200)

    st.divider()
    
    st.header("📂 文件与 Key 绑定")
    
    default_files = [
        "/Users/ziyuexu/Documents/trae_projects/paper1/NEJM.xlsx",
        "/Users/ziyuexu/Documents/trae_projects/paper1/BMJ.xlsx",
        "/Users/ziyuexu/Documents/trae_projects/paper1/JAMA.xlsx",
        "/Users/ziyuexu/Documents/trae_projects/paper1/Lancet.xlsx"
    ]
    
    default_keys = [
        "sk-37c1617db0da456d8491e1094e3f6ae3",
        "sk-82a00766192049fc91da7edbca74bfd2",
        "sk-c69f18b962d54e44b14298f079bc4c66",
        "sk-d98eb5841a0b4e6c9985b72b4106c74c"
    ]
    
    # 动态生成配置表单
    task_configs = []
    
    st.info("请为每个文件绑定一个 API Key：")
    
    for i, file_path in enumerate(default_files):
        if os.path.exists(file_path):
            file_name = os.path.basename(file_path)
            with st.expander(f"📄 {file_name}", expanded=True):
                # 默认填入对应的 Key
                default_key_val = default_keys[i] if i < len(default_keys) else ""
                key = st.text_input(f"API Key for {file_name}", value=default_key_val, type="password", key=f"key_{file_name}")
                if key:
                    task_configs.append({
                        'file_path': file_path,
                        'file_name': file_name,
                        'api_key': key
                    })
        else:
            st.error(f"❌ {os.path.basename(file_path)} (未找到)")

    start_btn = st.button("🚀 启动四路并发", type="primary", disabled=not task_configs)

# -----------------------------------------------------------------------------
# 核心处理逻辑 (纯函数，无 UI 副作用)
# -----------------------------------------------------------------------------

def pre_warm_entrez():
    """主线程预热 Entrez，避免并发写 DTD 冲突"""
    try:
        Entrez.email = "your.email@example.com"
        # 发起一个极小的请求来触发 DTD 下载
        handle = Entrez.efetch(db="pubmed", id="38446676", retmode="xml")
        handle.read()
        handle.close()
        return True
    except Exception as e:
        return str(e)

def process_single_task_with_queue(config, model, prompt, status_queue):
    """
    单个任务的处理函数，支持队列通信
    """
    try:
        df = config['dataframe'] # 预先读取好的 DF
        api_key = config['api_key']
        file_name = config['file_name']
        
        # 初始化 workflow (单 Key 模式)
        workflow = PaperWorkflow(api_keys=[api_key], model=model)
        
        # 定义回调函数，将进度推送到队列
        def progress_callback(current, total, message):
            # 将当前处理完的行发送回主线程
            # 注意：workflow 会原地修改 df，所以我们可以直接取当前行
            # 但为了线程安全，我们最好只发送必要的数据
            if current <= len(df):
                row_data = df.iloc[current-1].to_dict()
                status_queue.put({
                    'type': 'progress',
                    'file_name': file_name,
                    'current': current,
                    'total': total,
                    'message': message,
                    'row_data': row_data
                })

        # 执行处理
        processed_df = workflow.process_dataframe(
            df, 
            custom_prompt=prompt,
            progress_callback=progress_callback
        )
        processed_df['Source File'] = file_name
        
        # 发送完成消息
        status_queue.put({
            'type': 'complete',
            'file_name': file_name,
            'result': processed_df
        })
        return file_name, processed_df, None
    except Exception as e:
        # 发送错误消息
        status_queue.put({
            'type': 'error',
            'file_name': config['file_name'],
            'error': str(e)
        })
        return config['file_name'], None, str(e)

# -----------------------------------------------------------------------------
# 主界面逻辑
# -----------------------------------------------------------------------------

if start_btn:
    st.divider()
    status_container = st.container()
    
    # 1. 主线程预读取文件 & 预热 Entrez
    with status_container:
        with st.spinner("📦 正在预读取文件并预热 Entrez..."):
            # 预热 Entrez
            warm_result = pre_warm_entrez()
            if warm_result is not True:
                st.warning(f"Entrez 预热失败 (但这不一定致命): {warm_result}")
            
            # 预读取所有 Excel
            ready_tasks = []
            for conf in task_configs:
                try:
                    # 尝试读取文件
                    try:
                        df = pd.read_excel(conf['file_path'])
                    except Exception as e:
                        st.warning(f"⚠️ {conf['file_name']} 默认读取失败，尝试 openpyxl 引擎: {e}")
                        df = pd.read_excel(conf['file_path'], engine='openpyxl')
                    
                    if df.empty:
                         st.error(f"❌ 文件 {conf['file_name']} 是空的，已跳过。")
                         continue

                    # 创建一个新的配置对象，包含 DataFrame
                    ready_task = conf.copy()
                    ready_task['dataframe'] = df
                    ready_tasks.append(ready_task)
                    st.success(f"✅ 已读取 {conf['file_name']} ({len(df)} 行)")
                    
                except Exception as e:
                    st.warning(f"⚠️ {conf['file_name']} 无法读取，已跳过。")
                    with st.expander(f"查看 {conf['file_name']} 错误详情"):
                        st.error(f"错误信息: {str(e)}")
                        st.info("💡 建议：该文件可能损坏或格式不兼容。请在本地用 Excel 打开它，'另存为' .xlsx 格式后再试。")
            
            if not ready_tasks:
                st.error("没有可处理的文件，请检查文件是否正常。")
                st.stop()

    # 2. 初始化 UI 占位符
    st.subheader("🔄 并发处理进度")
    
    # 使用 tabs 来展示不同文件的详细进度，避免页面过长
    file_tabs = st.tabs([t['file_name'] for t in ready_tasks])
    ui_elements = {}
    
    for i, task in enumerate(ready_tasks):
        with file_tabs[i]:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"**进度监控**")
                progress = st.progress(0, text="等待启动...")
                status_text = st.empty()
            with col2:
                st.markdown("**最新处理结果预览**")
                table_placeholder = st.empty()
                
            ui_elements[task['file_name']] = {
                'progress': progress, 
                'status': status_text,
                'table': table_placeholder,
                'processed_rows': [] # 用于累积显示
            }

    # 3. 启动并发
    # 创建消息队列
    status_queue = queue.Queue()
    all_results = []
    active_tasks_count = len(ready_tasks)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(ready_tasks)) as executor:
        # 提交任务
        futures = [
            executor.submit(process_single_task_with_queue, task, selected_model, user_prompt, status_queue)
            for task in ready_tasks
        ]
        
        # 主线程循环监听队列
        while active_tasks_count > 0:
            try:
                # 非阻塞获取消息
                msg = status_queue.get(timeout=0.1)
                
                fname = msg.get('file_name')
                ui = ui_elements.get(fname)
                
                if not ui: continue
                
                if msg['type'] == 'progress':
                    # 更新进度条
                    percent = int(msg['current'] / msg['total'] * 100)
                    ui['progress'].progress(percent, text=f"正在处理 {msg['current']}/{msg['total']}")
                    ui['status'].info(f"🔄 {msg['message']}")
                    
                    # 更新表格
                    # 只保留最近 5 条或者累积所有？为了性能，累积所有但只显示最后几条
                    # 或者，为了用户体验，我们用 add_rows (Streamlit特性)？
                    # 简单起见，我们维护一个小的列表
                    row_data = msg['row_data']
                    # 筛选关键列
                    display_cols = ['PMID', 'Title', 'Abstract', 'Research Design', 'Focused Disease']
                    filtered_row = {k: row_data.get(k) for k in display_cols if k in row_data}
                    ui['processed_rows'].append(filtered_row)
                    
                    # 转换为 DF 显示，倒序（最新的在上面）
                    preview_df = pd.DataFrame(ui['processed_rows'])
                    ui['table'].dataframe(preview_df.iloc[::-1].head(10), use_container_width=True)
                    
                elif msg['type'] == 'complete':
                    ui['progress'].progress(100, text="✅ 完成")
                    ui['status'].success(f"完成！共 {len(msg['result'])} 条")
                    all_results.append(msg['result'])
                    active_tasks_count -= 1
                    
                elif msg['type'] == 'error':
                    ui['progress'].empty()
                    ui['status'].error(f"❌ 错误: {msg['error']}")
                    active_tasks_count -= 1
                    
            except queue.Empty:
                # 队列空，继续循环
                pass
            except Exception as e:
                st.error(f"UI 更新错误: {e}")
                break

    # 4. 结果展示与图表
    if all_results:
        st.divider()
        st.header("📊 高级分析报告")
        
        final_df = pd.concat(all_results, ignore_index=True)
        
        # --- 图表组 1: 研究全景 ---
        st.subheader("1. 研究方法学全景 (Research Landscape)")
        tab1, tab2, tab3 = st.tabs(["研究设计分布", "时序类型", "设计 x 时序关联"])
        
        with tab1:
            if 'Research Design' in final_df.columns:
                fig = px.histogram(final_df, x='Source File', color='Research Design', 
                                   title="各杂志研究设计构成", barmode='group')
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            if 'Study Timing' in final_df.columns:
                fig = px.pie(final_df, names='Study Timing', title="总体时序类型分布", hole=0.4)
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            if 'Research Design' in final_df.columns and 'Study Timing' in final_df.columns:
                try:
                    heatmap_data = pd.crosstab(final_df['Research Design'], final_df['Study Timing'])
                    fig = px.imshow(heatmap_data, text_auto=True, title="研究设计 vs 时序类型 热力图")
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.info("数据不足以生成热力图")

        # --- 图表组 2: 临床焦点 ---
        st.subheader("2. 临床疾病焦点 (Clinical Focus)")
        c1, c2 = st.columns(2)
        
        with c1:
            if 'Focused Disease System' in final_df.columns:
                sys_counts = final_df['Focused Disease System'].value_counts().reset_index()
                sys_counts.columns = ['System', 'Count']
                fig = px.treemap(sys_counts, path=['System'], values='Count', title="疾病系统分布 (Treemap)")
                st.plotly_chart(fig, use_container_width=True)
                
        with c2:
            if 'Focused Disease' in final_df.columns:
                dis_counts = final_df['Focused Disease'].value_counts().head(10).reset_index()
                dis_counts.columns = ['Disease', 'Count']
                fig = px.bar(dis_counts, x='Count', y='Disease', orientation='h', title="Top 10 聚焦疾病")
                st.plotly_chart(fig, use_container_width=True)

        # --- 图表组 3: 全球视野 ---
        st.subheader("3. 全球研究视野 (Global View)")
        g1, g2 = st.columns(2)
        
        with g1:
            if 'Research Team Country' in final_df.columns:
                country_counts = final_df['Research Team Country'].value_counts().reset_index()
                country_counts.columns = ['Country', 'Count']
                fig = px.choropleth(country_counts, locations="Country", locationmode='country names',
                                    color="Count", hover_name="Country", title="全球发文量分布")
                st.plotly_chart(fig, use_container_width=True)

        with g2:
            if 'Research Team Country' in final_df.columns and 'Research Design' in final_df.columns:
                target_countries = ['China', 'USA', 'United States', 'China (Mainland)']
                mask = final_df['Research Team Country'].isin(target_countries)
                if mask.any():
                    plot_df = final_df[mask].copy()
                    plot_df['Country'] = plot_df['Research Team Country'].apply(lambda x: 'USA' if 'United States' in x or 'USA' in x else 'China')
                    
                    design_by_country = pd.crosstab(plot_df['Research Design'], plot_df['Country'])
                    categories = design_by_country.index.tolist()
                    fig = go.Figure()
                    if 'China' in design_by_country.columns:
                        fig.add_trace(go.Scatterpolar(r=design_by_country['China'], theta=categories, fill='toself', name='China'))
                    if 'USA' in design_by_country.columns:
                        fig.add_trace(go.Scatterpolar(r=design_by_country['USA'], theta=categories, fill='toself', name='USA'))
                    
                    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), title="中美研究设计偏好对比")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("暂无中美相关数据")

        # 5. 数据下载
        st.divider()
        st.subheader("📥 数据下载")
        
        output_all = BytesIO()
        with pd.ExcelWriter(output_all, engine='openpyxl') as writer:
            final_df.to_excel(writer, index=False)
            
        st.download_button(
            label="📥 下载完整汇总报告 (Excel)",
            data=output_all.getvalue(),
            file_name="multi_journal_analysis_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )
