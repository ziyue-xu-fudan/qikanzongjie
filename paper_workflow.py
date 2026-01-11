import pandas as pd
import requests
import json
import time
from Bio import Entrez
import os
from openai import OpenAI
import re
from typing import Dict, List, Optional

# 设置 Entrez 邮箱，请在使用前设置
Entrez.email = "your.email@example.com"

class PaperWorkflow:
    def __init__(self, api_keys: List[str], base_url: str = "https://api.deepseek.com", model: str = "deepseek-chat"):
        """
        初始化工作流
        :param api_keys: DeepSeek API Keys 列表
        :param base_url: DeepSeek API Base URL
        :param model: 使用的模型名称
        """
        self.api_keys = [k.strip() for k in api_keys if k.strip()]
        self.base_url = base_url
        self.model = model
        self.current_key_index = 0
        
        if not self.api_keys:
            raise ValueError("至少需要提供一个 API Key")

    def _get_client(self):
        """获取当前轮询到的 OpenAI 客户端"""
        api_key = self.api_keys[self.current_key_index]
        # 轮询到下一个 key
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        return OpenAI(api_key=api_key, base_url=self.base_url)

    def fetch_abstract(self, pmid: str) -> Optional[str]:
        """
        根据 PMID 获取摘要
        """
        if not pmid or pd.isna(pmid):
            return None
            
        try:
            # 清理 PMID
            pmid = str(pmid).strip().replace('.0', '')
            
            # 使用 Entrez 获取摘要
            handle = Entrez.efetch(db="pubmed", id=pmid, retmode="xml")
            records = Entrez.read(handle)
            handle.close()
            
            if not records or 'PubmedArticle' not in records:
                return None
                
            article = records['PubmedArticle'][0]['MedlineCitation']['Article']
            
            if 'Abstract' in article and 'AbstractText' in article['Abstract']:
                abstract_parts = article['Abstract']['AbstractText']
                # 处理可能是列表或字符串的情况
                if isinstance(abstract_parts, list):
                    return " ".join([str(part) for part in abstract_parts])
                return str(abstract_parts)
                
            return None
            
        except Exception as e:
            print(f"Error fetching abstract for PMID {pmid}: {e}")
            return None

    def analyze_abstract(self, abstract: str, custom_prompt: Optional[str] = None) -> Dict:
        """
        使用大模型分析摘要
        """
        if not abstract:
            return {
                "research_design": "N/A",
                "focused_disease": "N/A",
                "target_population": "N/A",
                "research_team_country": "N/A"
            }

        # 使用默认 Prompt 或自定义 Prompt
        if custom_prompt:
            # 如果用户提供了自定义 Prompt，我们需要确保它包含摘要占位符，或者我们将摘要附加在最后
            # 简单的处理方式：假设用户知道要在 Prompt 中包含 {abstract} 占位符
            try:
                prompt = custom_prompt.format(abstract=abstract)
            except KeyError:
                # 如果用户没有包含 {abstract}，我们手动拼接
                prompt = f"{custom_prompt}\n\n摘要内容:\n{abstract}"
        else:
            prompt = f"""
            请分析以下医学文献摘要，并提取以下四个关键信息。
            请严格按照 JSON 格式返回，不要包含 Markdown 格式标记（如 ```json）。
            如果无法提取某个字段，请填写 "Unknown"。

            摘要内容:
            {abstract}

            需要提取的字段:
            1. research_design (研究方式，例如：RCT, Cohort Study, Case Report, Review 等)
            2. focused_disease (聚焦疾病，例如：Diabetes, Lung Cancer, COVID-19 等)
            3. target_population (目标人群，例如：Adults with Type 2 Diabetes, Elderly patients, Children 等)
            4. research_team_country (研究团队主要国家，通常在作者单位中，如果摘要未提及，请推断或填 Unknown)

            JSON 格式示例:
            {{
                "research_design": "Randomized Controlled Trial",
                "focused_disease": "Hypertension",
                "target_population": "Adults over 60",
                "research_team_country": "USA"
            }}
            """

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful medical research assistant. You specialize in extracting structured information from medical abstracts."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            content = response.choices[0].message.content.strip()
            
            # 尝试清理可能存在的 markdown 标记
            content = re.sub(r'^```json\s*', '', content)
            content = re.sub(r'\s*```$', '', content)
            
            return json.loads(content)
            
        except json.JSONDecodeError:
            print(f"JSON Parse Error. Raw content: {content}")
            return {
                "research_design": "Error",
                "focused_disease": "Error",
                "target_population": "Error",
                "research_team_country": "Error"
            }
        except Exception as e:
            print(f"API Call Error: {e}")
            return {
                "research_design": "Error",
                "focused_disease": "Error",
                "target_population": "Error",
                "research_team_country": "Error"
            }

    def process_dataframe(self, df: pd.DataFrame, pmid_col: str = 'PMID', custom_prompt: Optional[str] = None, progress_callback=None) -> pd.DataFrame:
        """
        处理 DataFrame
        """
        # 确保有 PMID 列
        if pmid_col not in df.columns:
            # 尝试寻找可能的 PMID 列名
            possible_cols = [c for c in df.columns if 'pmid' in c.lower()]
            if possible_cols:
                pmid_col = possible_cols[0]
            else:
                raise ValueError(f"Column '{pmid_col}' not found in DataFrame")

        # 初始化新列
        new_cols = ['Abstract', 'Research Design', 'Study Timing', 'Focused Disease System', 'Focused Disease', 'Target Population', 'Research Team Country']
        for col in new_cols:
            if col not in df.columns:
                df[col] = None

        total = len(df)
        
        for index, row in df.iterrows():
            pmid = row[pmid_col]
            
            # 1. 获取摘要 (如果还没有)
            abstract = row.get('Abstract')
            if pd.isna(abstract) or abstract == '':
                if pd.notna(pmid):
                    abstract = self.fetch_abstract(pmid)
                    df.at[index, 'Abstract'] = abstract
                    # 避免 API 速率限制
                    time.sleep(0.35) 
            
            # 2. AI 分析
            if abstract and (pd.isna(row.get('Research Design')) or row.get('Research Design') == ''):
                analysis = self.analyze_abstract(abstract, custom_prompt=custom_prompt)
                df.at[index, 'Research Design'] = analysis.get('research_design')
                df.at[index, 'Study Timing'] = analysis.get('study_timing')
                df.at[index, 'Focused Disease System'] = analysis.get('focused_disease_system')
                df.at[index, 'Focused Disease'] = analysis.get('focused_disease')
                df.at[index, 'Target Population'] = analysis.get('target_population')
                df.at[index, 'Research Team Country'] = analysis.get('research_team_country')
            
            # 回调进度
            if progress_callback:
                progress_callback(index + 1, total, f"Processing PMID: {pmid}")
                
        return df
