#!/usr/bin/env python3
"""
增强版NEJM文章爬取脚本
包含完整的XML解析功能
"""

import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import time
import os
from typing import List, Dict, Optional
from xml.etree import ElementTree as ET
from Bio import Entrez
import xmltodict

class EnhancedNEJMArticleScraper:
    def __init__(self, email: str = "your_email@example.com"):
        """
        初始化爬虫
        
        Args:
            email: 用于PubMed API的邮箱地址
        """
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.search_url = self.base_url + "esearch.fcgi"
        self.fetch_url = self.base_url + "efetch.fcgi"
        self.journal_name = "N Engl J Med"
        self.article_types = ["Original Article", "Correspondence"]
        
        # 设置Entrez邮箱（PubMed要求）
        Entrez.email = email
        
    def search_articles_advanced(self, start_date: str, end_date: str, retmax: int = 100) -> List[str]:
        """
        高级搜索NEJM文章
        
        Args:
            start_date: 开始日期 (YYYY/MM/DD)
            end_date: 结束日期 (YYYY/MM/DD)
            retmax: 每次返回的最大结果数
            
        Returns:
            PMID列表
        """
        # 使用Entrez进行搜索
        search_query = f'"{self.journal_name}"[Journal] AND ("{start_date}"[Date - Publication] : "{end_date}"[Date - Publication])'
        
        try:
            # 首先获取搜索计数
            search_handle = Entrez.esearch(
                db="pubmed", 
                term=search_query,
                retmax=retmax,
                usehistory="y"
            )
            search_results = Entrez.read(search_handle)
            search_handle.close()
            
            count = int(search_results["Count"])
            print(f"找到 {count} 篇NEJM文章")
            
            # 获取所有PMID
            pmids = search_results["IdList"]
            
            # 如果结果数量大于retmax，分批获取
            if count > retmax:
                for retstart in range(retmax, count, retmax):
                    search_handle = Entrez.esearch(
                        db="pubmed", 
                        term=search_query,
                        retmax=retmax,
                        retstart=retstart
                    )
                    batch_results = Entrez.read(search_handle)
                    search_handle.close()
                    pmids.extend(batch_results["IdList"])
                    time.sleep(0.3)  # 避免请求过快
            
            return pmids
            
        except Exception as e:
            print(f"搜索请求失败: {e}")
            return []
    
    def fetch_article_details_enhanced(self, pmids: List[str]) -> List[Dict]:
        """
        增强版获取文章详细信息
        
        Args:
            pmids: PMID列表
            
        Returns:
            文章详细信息列表
        """
        if not pmids:
            return []
            
        articles = []
        batch_size = 50  # 减少批次大小以提高稳定性
        
        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i+batch_size]
            
            try:
                # 使用Entrez获取文章详情
                fetch_handle = Entrez.efetch(
                    db="pubmed", 
                    id=",".join(batch_pmids),
                    rettype="xml", 
                    retmode="xml"
                )
                
                # 解析XML数据
                articles_batch = self.parse_pubmed_xml_enhanced(fetch_handle.read())
                articles.extend(articles_batch)
                fetch_handle.close()
                
                print(f"已处理 {i+len(batch_pmids)}/{len(pmids)} 篇文章")
                time.sleep(0.5)  # 增加延迟避免被封
                
            except Exception as e:
                print(f"获取文章详情失败 (批次 {i//batch_size + 1}): {e}")
                continue
                
        return articles
    
    def parse_pubmed_xml_enhanced(self, xml_content: str) -> List[Dict]:
        """
        增强版解析PubMed XML内容
        
        Args:
            xml_content: XML内容
            
        Returns:
            解析后的文章信息列表
        """
        articles = []
        
        try:
            # 使用xmltodict解析XML
            data = xmltodict.parse(xml_content)
            
            # 获取PubmedArticleSet
            article_set = data.get('PubmedArticleSet', {})
            pubmed_articles = article_set.get('PubmedArticle', [])
            
            # 确保pubmed_articles是列表
            if not isinstance(pubmed_articles, list):
                pubmed_articles = [pubmed_articles]
            
            for article_data in pubmed_articles:
                try:
                    article = self.extract_article_info(article_data)
                    if article:
                        articles.append(article)
                except Exception as e:
                    print(f"解析单篇文章失败: {e}")
                    continue
                    
        except Exception as e:
            print(f"XML解析失败: {e}")
            
        return articles
    
    def extract_article_info(self, article_data: Dict) -> Optional[Dict]:
        """
        从文章数据中提取关键信息
        
        Args:
            article_data: 单篇文章的原始数据
            
        Returns:
            提取后的文章信息
        """
        try:
            # 获取MedlineCitation
            medline_citation = article_data.get('MedlineCitation', {})
            
            # 获取文章基本信息
            article_info = medline_citation.get('Article', {})
            
            # PMID
            pmid = medline_citation.get('PMID', {}).get('#text', '')
            
            # 标题
            title = ""
            article_title = article_info.get('ArticleTitle', {})
            if isinstance(article_title, str):
                title = article_title
            elif isinstance(article_title, dict):
                title = article_title.get('#text', '')
            
            # 作者
            authors = []
            author_list = article_info.get('AuthorList', {})
            if author_list:
                authors_data = author_list.get('Author', [])
                if not isinstance(authors_data, list):
                    authors_data = [authors_data]
                
                for author in authors_data:
                    if isinstance(author, dict):
                        lastname = author.get('LastName', '')
                        forename = author.get('ForeName', '')
                        if lastname and forename:
                            authors.append(f"{lastname} {forename}")
                        elif lastname:
                            authors.append(lastname)
            
            # 期刊信息
            journal = article_info.get('Journal', {})
            journal_title = journal.get('Title', '')
            
            # 发表日期
            pub_date = ""
            journal_issue = journal.get('JournalIssue', {})
            pub_date_data = journal_issue.get('PubDate', {})
            
            if pub_date_data:
                year = pub_date_data.get('Year', '')
                month = pub_date_data.get('Month', '')
                day = pub_date_data.get('Day', '')
                
                if year:
                    pub_date = f"{year}"
                    if month:
                        pub_date += f"/{month}"
                        if day:
                            pub_date += f"/{day}"
            
            # DOI
            doi = ""
            elocation_ids = article_info.get('ELocationID', [])
            if not isinstance(elocation_ids, list):
                elocation_ids = [elocation_ids]
            
            for elocation_id in elocation_ids:
                if isinstance(elocation_ids, dict) and elocation_id.get('@EIdType') == 'doi':
                    doi = elocation_id.get('#text', '')
                    break
            
            # 摘要
            abstract = ""
            abstract_data = article_info.get('Abstract', {})
            if abstract_data:
                abstract_texts = abstract_data.get('AbstractText', [])
                if not isinstance(abstract_texts, list):
                    abstract_texts = [abstract_texts]
                
                for abstract_text in abstract_texts:
                    if isinstance(abstract_text, dict):
                        abstract += abstract_text.get('#text', '') + ' '
                    elif isinstance(abstract_text, str):
                        abstract += abstract_text + ' '
                
                abstract = abstract.strip()
            
            # 文章类型
            article_type = ""
            publication_types = article_info.get('PublicationTypeList', {})
            if publication_types:
                pub_types = publication_types.get('PublicationType', [])
                if not isinstance(pub_types, list):
                    pub_types = [pub_types]
                
                for pub_type in pub_types:
                    if isinstance(pub_type, dict):
                        type_name = pub_type.get('#text', '')
                        if type_name in self.article_types:
                            article_type = type_name
                            break
            
            # 关键词
            keywords = []
            keyword_list = medline_citation.get('KeywordList', {})
            if keyword_list:
                keywords_data = keyword_list.get('Keyword', [])
                if not isinstance(keywords_data, list):
                    keywords_data = [keywords_data]
                
                for keyword in keywords_data:
                    if isinstance(keyword, dict):
                        keyword_text = keyword.get('#text', '')
                        if keyword_text:
                            keywords.append(keyword_text)
                    elif isinstance(keyword, str):
                        keywords.append(keyword)
            
            # 构建文章信息字典
            article = {
                'pmid': pmid,
                'title': title,
                'authors': ', '.join(authors),
                'journal': journal_title,
                'pub_date': pub_date,
                'doi': doi,
                'abstract': abstract,
                'article_type': article_type,
                'keywords': ', '.join(keywords),
                'scraped_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return article
            
        except Exception as e:
            print(f"提取文章信息失败: {e}")
            return None
    
    def filter_by_article_type(self, articles: List[Dict]) -> List[Dict]:
        """
        根据文章类型筛选文章
        
        Args:
            articles: 文章列表
            
        Returns:
            筛选后的文章列表
        """
        filtered_articles = []
        
        for article in articles:
            article_type = article.get('article_type', '')
            
            # 检查文章类型是否符合要求
            if article_type in self.article_types:
                filtered_articles.append(article)
            else:
                # 如果没有明确的article_type，检查标题和内容
                title = article.get('title', '').lower()
                abstract = article.get('abstract', '').lower()
                
                # 简单的关键词匹配
                if any(keyword in title + abstract for keyword in ['original article', 'correspondence']):
                    filtered_articles.append(article)
        
        return filtered_articles
    
    def save_to_multiple_formats(self, articles: List[Dict], base_filename: str = None) -> Dict[str, str]:
        """
        保存文章数据到多种格式
        
        Args:
            articles: 文章数据列表
            base_filename: 基础文件名（不含扩展名）
            
        Returns:
            保存的文件路径字典
        """
        if not base_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"nejm_articles_{timestamp}"
        
        saved_files = {}
        
        # 保存为CSV
        csv_file = f"{base_filename}.csv"
        df = pd.DataFrame(articles)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        saved_files['csv'] = csv_file
        
        # 保存为JSON
        json_file = f"{base_filename}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        saved_files['json'] = json_file
        
        # 保存为Excel
        excel_file = f"{base_filename}.xlsx"
        df.to_excel(excel_file, index=False)
        saved_files['excel'] = excel_file
        
        # 保存为Markdown表格
        markdown_file = f"{base_filename}.md"
        df.to_markdown(markdown_file, index=False)
        saved_files['markdown'] = markdown_file
        
        return saved_files

def main():
    """
    主函数
    """
    # 设置邮箱（必需）
    email = "user@example.com"  # 使用默认邮箱自动运行
    print(f"使用邮箱: {email}")
    
    scraper = EnhancedNEJMArticleScraper(email=email)
    
    # 设置时间范围（近5年）
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    print(f"开始爬取NEJM文章...")
    print(f"时间范围: {start_date.strftime('%Y/%m/%d')} - {end_date.strftime('%Y/%m/%d')}")
    print(f"文章类型: {', '.join(scraper.article_types)}")
    
    # 搜索文章
    print("正在搜索文章...")
    pmids = scraper.search_articles_advanced(
        start_date=start_date.strftime('%Y/%m/%d'),
        end_date=end_date.strftime('%Y/%m/%d'),
        retmax=100
    )
    
    print(f"找到 {len(pmids)} 篇文章")
    
    if pmids:
        # 获取文章详情
        print("正在获取文章详情...")
        articles = scraper.fetch_article_details_enhanced(pmids)
        
        print(f"成功获取 {len(articles)} 篇文章的详细信息")
        
        # 根据文章类型筛选
        print("正在筛选文章类型...")
        filtered_articles = scraper.filter_by_article_type(articles)
        print(f"筛选后剩余 {len(filtered_articles)} 篇文章")
        
        if filtered_articles:
            # 保存数据到多种格式
            print("正在保存数据...")
            saved_files = scraper.save_to_multiple_formats(filtered_articles)
            
            print(f"数据已保存到:")
            for format_type, file_path in saved_files.items():
                print(f"  {format_type.upper()}: {file_path}")
            
            # 显示统计信息
            print(f"\n爬取统计:")
            print(f"  总文章数: {len(filtered_articles)}")
            print(f"  时间范围: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
            
            # 按年份统计
            year_counts = {}
            for article in filtered_articles:
                pub_date = article.get('pub_date', '')
                if pub_date and len(pub_date) >= 4:
                    year = pub_date[:4]
                    year_counts[year] = year_counts.get(year, 0) + 1
            
            if year_counts:
                print(f"  按年份分布:")
                for year in sorted(year_counts.keys()):
                    print(f"    {year}: {year_counts[year]} 篇")
        else:
            print("没有找到符合条件的文章")
    else:
        print("未找到任何文章")

if __name__ == "__main__":
    main()