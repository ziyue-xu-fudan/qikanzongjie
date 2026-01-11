import pandas as pd
from paper_workflow import PaperWorkflow
import time

# 用户提供的 API Key
API_KEY = "sk-035f4a026e724fb194a5083b3ed3c3b1"

# 样本数据 (PMID 来自之前的 NEJM 样本)
# 10.1056/NEJMoa2309822 -> PMID 38446676
# 10.1056/NEJMoa2310168 -> PMID 38381674
sample_data = {
    'PMID': [38446676, 38381674],
    'Title': ['Microplastics and Nanoplastics in Atheromas and Cardiovascular Events', 'Biomarker Changes during 20 Years Preceding Alzheimer\'s Disease']
}

df = pd.DataFrame(sample_data)

print("🚀 开始调试...")
print(f"📊 样本数据:\n{df}")

workflow = PaperWorkflow(api_key=API_KEY, model="deepseek-chat")

# 1. 测试摘要获取
print("\n🔍 测试摘要获取 (fetch_abstract)...")
pmid = df.iloc[0]['PMID']
try:
    abstract = workflow.fetch_abstract(str(pmid))
    if abstract:
        print(f"✅ 成功获取摘要 (长度: {len(abstract)}):")
        print(f"   {abstract[:200]}...")
        df.at[0, 'Abstract'] = abstract
    else:
        print("❌ 未能获取摘要")
        # 手动设置一个假摘要以测试后续流程
        df.at[0, 'Abstract'] = "This is a test abstract about cardiovascular events and microplastics. The study was a cohort study involving 300 patients."
except Exception as e:
    print(f"❌ 获取摘要时出错: {e}")

# 2. 测试 AI 分析
print("\n🤖 测试 AI 分析 (analyze_abstract)...")
if pd.notna(df.iloc[0]['Abstract']):
    abstract_text = df.iloc[0]['Abstract']
    print(f"正在分析摘要: {abstract_text[:50]}...")
    
    # 使用 app.py 中的默认 Prompt
    default_prompt = """请分析以下医学文献摘要，并提取以下四个关键信息。
请严格按照 JSON 格式返回，不要包含 Markdown 格式标记（如 ```json）。
如果无法提取某个字段，请填写 "Unknown"。

摘要内容:
{abstract}

需要提取的字段:
1. research_design (研究方式)
   - 请从以下列表中选择最匹配的一项:
     [Randomized Controlled Trial, Cohort Study, Case-Control Study, Cross-sectional Study, Systematic Review, Meta-analysis, Case Report, Animal Study, In Vitro Study, Narrative Review]
   - 如果都不匹配，请填写 "Other".

2. focused_disease (聚焦疾病)
   - 请提取主要的疾病名称。
   - **重要**: 请尽可能提供该疾病对应的 ICD-10 编码，格式为 "Disease Name (ICD-10 Code)"。例如: "Type 2 Diabetes Mellitus (E11)", "Lung Cancer (C34)".
   - 如果无法确定 ICD-10 编码，仅填写疾病名称。

3. target_population (目标人群)
   - 请简要描述目标人群特征，包括年龄组、性别或特定状况。例如: "Adults aged 18-65 with hypertension".

4. research_team_country (研究团队主要国家)
   - 请提取通讯作者或第一作者所在的国家。
   - 请使用标准的英文国家名称。

JSON 格式示例:
{{
    "research_design": "Randomized Controlled Trial",
    "focused_disease": "Hypertension (I10)",
    "target_population": "Adults over 60",
    "research_team_country": "USA"
}}"""

    try:
        analysis = workflow.analyze_abstract(abstract_text, custom_prompt=default_prompt)
        print("✅ AI 分析结果:")
        print(analysis)
    except Exception as e:
        print(f"❌ AI 分析出错: {e}")
else:
    print("⚠️ 跳过 AI 分析，因为没有摘要")

print("\n🏁 调试结束")
