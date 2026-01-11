#!/usr/bin/env python3
"""
NEJM文章爬取脚本
利用PubMed API爬取近5年NEJM的Original Article和Correspondence文章
"""

import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import time
import os
from typing import List, Dict, Optional

class NEJMArticleScraper:
    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.search_url = self.base_url + "esearch.fcgi"
        self.fetch_url = self.base_url + "efetch.fcgi"
        self.journal_name = "N Engl J Med"
        self.article_types = ["Original Article", "Correspondence"]
        
    def search_articles(self, start_date: str, end_date: str, retmax: int = 100) -> List[str]:
        """
        搜索NEJM文章
        
        Args:
            start_date: 开始日期 (YYYY/MM/DD)
            end_date: 结束日期 (YYYY/MM/DD)
            retmax: 每次返回的最大结果数
            
        Returns:
            PMID列表
        """
        # 构建搜索查询
        query = f'"{self.journal_name}"[Journal] AND ("{start_date}"[Date - Publication] : "{end_date}"[Date - Publication])'
        
        # 添加文章类型过滤
        type_query = " OR ".join([f'"{article_type}"[Publication Type]' for article_type in self.article_types])
        full_query = f"({query}) AND ({type_query})"
        
        params = {
            'db': 'pubmed',
            'term': full_query,
            'retmode': 'json',
            'retmax': retmax,
            'retstart': 0
        }
        
        pmids = []
        
        try:
            while True:
                response = requests.get(self.search_url, params=params)
                response.raise_for_status()
                
                data = response.json()
                esearchresult = data.get('esearchresult', {})
                
                batch_pmids = esearchresult.get('idlist', [])
                if not batch_pmids:
                    break
                    
                pmids.extend(batch_pmids)
                
                # 检查是否还有更多结果
                count = int(esearchresult.get('count', 0))
                retstart = params['retstart'] + retmax
                
                if retstart >= count:
                    break
                    
                params['retstart'] = retstart
                time.sleep(0.3)  # 避免请求过快
                
        except requests.exceptions.RequestException as e:
            print(f"搜索请求失败: {e}")
            
        return pmids
    
    def fetch_article_details(self, pmids: List[str]) -> List[Dict]:
        """
        获取文章详细信息
        
        Args:
            pmids: PMID列表
            
        Returns:
            文章详细信息列表
        """
        if not pmids:
            return []
            
        # 分批获取，避免请求过大
        batch_size = 100
        all_articles = []
        
        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i+batch_size]
            pmid_string = ",".join(batch_pmids)
            
            params = {
                'db': 'pubmed',
                'id': pmid_string,
                'retmode': 'xml'
            }
            
            try:
                response = requests.get(self.fetch_url, params=params)
                response.raise_for_status()
                
                # 这里简化处理，实际使用时需要解析XML
                # 可以使用xml.etree.ElementTree或BeautifulSoup
                articles = self.parse_pubmed_xml(response.text)
                all_articles.extend(articles)
                
                time.sleep(0.3)  # 避免请求过快
                
            except requests.exceptions.RequestException as e:
                print(f"获取文章详情失败: {e}")
                
        return all_articles
    
    def parse_pubmed_xml(self, xml_content: str) -> List[Dict]:
        """
        解析PubMed XML内容
        
        Args:
            xml_content: XML内容
            
        Returns:
            解析后的文章信息列表
        """
        articles = []
        
        # 这里简化处理，实际使用时需要完整的XML解析
        # 提取关键信息：标题、作者、发表日期、DOI、摘要等
        
        # 示例数据结构
        sample_article = {
            'pmid': '12345678',
            'title': 'Sample Article Title',
            'authors': ['Author 1', 'Author 2'],
            'journal': self.journal_name,
            'pub_date': '2024/01/01',
            'doi': '10.1056/NEJMoa123456',
            'abstract': 'Sample abstract text...',
            'article_type': 'Original Article'
        }
        
        articles.append(sample_article)
        return articles
    
    def save_to_csv(self, articles: List[Dict], filename: str = None) -> str:
        """
        保存文章数据到CSV文件
        
        Args:
            articles: 文章数据列表
            filename: 输出文件名
            
        Returns:
            保存的文件路径
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"nejm_articles_{timestamp}.csv"
            
        df = pd.DataFrame(articles)
        df.to_csv(filename, index=False, encoding='utf-8')
        
        return filename
    
    def save_to_json(self, articles: List[Dict], filename: str = None) -> str:
        """
        保存文章数据到JSON文件
        
        Args:
            articles: 文章数据列表
            filename: 输出文件名
            
        Returns:
            保存的文件路径
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"nejm_articles_{timestamp}.json"
            
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
            
        return filename

def main():
    """
    主函数
    """
    scraper = NEJMArticleScraper()
    
    # 设置时间范围（近5年）
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    print(f"开始爬取NEJM文章...")
    print(f"时间范围: {start_date.strftime('%Y/%m/%d')} - {end_date.strftime('%Y/%m/%d')}")
    
    # 搜索文章
    print("正在搜索文章...")
    pmids = scraper.search_articles(
        start_date=start_date.strftime('%Y/%m/%d'),
        end_date=end_date.strftime('%Y/%m/%d'),
        retmax=100
    )
    
    print(f"找到 {len(pmids)} 篇文章")
    
    if pmids:
        # 获取文章详情
        print("正在获取文章详情...")
        articles = scraper.fetch_article_details(pmids)
        
        # 保存数据
        csv_file = scraper.save_to_csv(articles)
        json_file = scraper.save_to_json(articles)
        
        print(f"数据已保存到:")
        print(f"CSV文件: {csv_file}")
        print(f"JSON文件: {json_file}")
    else:
        print("未找到符合条件的文章")

if __name__ == "__main__":
    main()