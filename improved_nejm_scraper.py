#!/usr/bin/env python3
"""
改进版NEJM文章爬取脚本
修正文章类型搜索问题
"""

import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import List, Dict, Optional

class ImprovedNEJMArticleScraper:
    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.search_url = self.base_url + "esearch.fcgi"
        self.fetch_url = self.base_url + "efetch.fcgi"
        self.journal_name = "N Engl J Med"
        
        # 修正的文章类型关键词
        self.article_type_keywords = {
            "original_article": ["original article", "research article", "clinical research", "original research"],
            "correspondence": ["correspondence", "letter", "reply", "response"]
        }
        
    def search_nejm_articles(self, start_date: str, end_date: str, retmax: int = 100) -> List[str]:
        """
        搜索NEJM文章（改进版）
        
        Args:
            start_date: 开始日期 (YYYY/MM/DD)
            end_date: 结束日期 (YYYY/MM/DD)
            retmax: 每次返回的最大结果数
            
        Returns:
            PMID列表
        """
        # 基础查询：NEJM期刊 + 时间范围
        base_query = f'"{self.journal_name}"[Journal] AND ("{start_date}"[Date - Publication] : "{end_date}"[Date - Publication])'
        
        pmids = []
        
        # 策略1：直接搜索所有NEJM文章，然后手动筛选
        print("策略1: 搜索所有NEJM文章...")
        params = {
            'db': 'pubmed',
            'term': base_query,
            'retmode': 'json',
            'retmax': retmax,
            'retstart': 0
        }
        
        try:
            total_count = 0
            while True:
                response = requests.get(self.search_url, params=params)
                response.raise_for_status()
                
                data = response.json()
                esearchresult = data.get('esearchresult', {})
                
                batch_pmids = esearchresult.get('idlist', [])
                if not batch_pmids:
                    break
                    
                pmids.extend(batch_pmids)
                total_count += len(batch_pmids)
                
                # 检查是否还有更多结果
                count = int(esearchresult.get('count', 0))
                retstart = params['retstart'] + retmax
                
                print(f"已获取 {total_count}/{count} 篇文章")
                
                if retstart >= count or total_count >= 1000:  # 限制数量避免过多
                    break
                    
                params['retstart'] = retstart
                time.sleep(0.3)
                
        except requests.exceptions.RequestException as e:
            print(f"搜索请求失败: {e}")
            
        print(f"总共找到 {len(pmids)} 篇NEJM文章")
        return pmids
    
    def fetch_article_details_batch(self, pmids: List[str]) -> List[Dict]:
        """
        批量获取文章详细信息
        """
        if not pmids:
            return []
            
        articles = []
        batch_size = 50
        
        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i+batch_size]
            print(f"获取文章详情: {i+1}-{min(i+batch_size, len(pmids))}/{len(pmids)}")
            
            params = {
                'db': 'pubmed',
                'id': ','.join(batch_pmids),
                'retmode': 'xml'
            }
            
            try:
                response = requests.get(self.fetch_url, params=params, timeout=60)
                response.raise_for_status()
                
                batch_articles = self.parse_xml_content(response.text)
                articles.extend(batch_articles)
                
                time.sleep(0.5)
                
            except Exception as e:
                print(f"获取文章详情失败: {e}")
                continue
                
        return articles
    
    def parse_xml_content(self, xml_content: str) -> List[Dict]:
        """
        简化的XML解析
        """
        articles = []
        
        # 简单的XML解析，提取关键信息
        import re
        
        # 找到所有文章块
        article_blocks = re.findall(r'<PubmedArticle>(.*?)</PubmedArticle>', xml_content, re.DOTALL)
        
        for block in article_blocks:
            try:
                article = {}
                
                # 提取PMID
                pmid_match = re.search(r'<PMID[^>]*>(\d+)</PMID>', block)
                article['pmid'] = pmid_match.group(1) if pmid_match else ''
                
                # 提取标题
                title_match = re.search(r'<ArticleTitle[^>]*>(.*?)</ArticleTitle>', block, re.DOTALL)
                article['title'] = self.clean_text(title_match.group(1)) if title_match else ''
                
                # 提取作者
                authors = []
                author_matches = re.findall(r'<Author[^>]*>(.*?)</Author>', block, re.DOTALL)
                for author_block in author_matches:
                    lastname_match = re.search(r'<LastName>(.*?)</LastName>', author_block)
                    forename_match = re.search(r'<ForeName>(.*?)</ForeName>', author_block)
                    
                    if lastname_match and forename_match:
                        authors.append(f"{lastname_match.group(1)} {forename_match.group(1)}")
                    elif lastname_match:
                        authors.append(lastname_match.group(1))
                
                article['authors'] = ', '.join(authors[:5])  # 限制前5个作者
                
                # 提取期刊
                journal_match = re.search(r'<Title>(.*?)</Title>', block)
                article['journal'] = journal_match.group(1) if journal_match else ''
                
                # 提取日期
                year_match = re.search(r'<Year>(\d{4})</Year>', block)
                month_match = re.search(r'<Month>(\w+)</Month>', block)
                
                if year_match:
                    date_str = year_match.group(1)
                    if month_match:
                        date_str += f" {month_match.group(1)}"
                    article['pub_date'] = date_str
                else:
                    article['pub_date'] = ''
                
                # 提取DOI
                doi_match = re.search(r'<ELocationID[^>]*EIdType="doi"[^>]*>(.*?)</ELocationID>', block)
                article['doi'] = doi_match.group(1) if doi_match else ''
                
                # 提取摘要
                abstract_match = re.search(r'<AbstractText[^>]*>(.*?)</AbstractText>', block, re.DOTALL)
                abstract = self.clean_text(abstract_match.group(1)) if abstract_match else ''
                article['abstract'] = abstract[:500] + '...' if len(abstract) > 500 else abstract
                
                # 判断文章类型
                article['article_type'] = self.classify_article_type(block, article['title'])
                
                if article['pmid'] and article['title']:  # 确保有关键信息
                    articles.append(article)
                    
            except Exception as e:
                print(f"解析文章失败: {e}")
                continue
                
        return articles
    
    def classify_article_type(self, xml_block: str, title: str) -> str:
        """
        基于内容和标题判断文章类型
        """
        content = (xml_block + " " + title).lower()
        
        # 检查Correspondence相关关键词
        correspondence_keywords = ['correspondence', 'letter', 'reply', 'response', 'to the editor']
        if any(keyword in content for keyword in correspondence_keywords):
            return 'Correspondence'
        
        # 检查Original Article相关特征
        # 通常Original Article有完整的结构：摘要、方法、结果、讨论等
        if ('abstract' in content and 
            ('method' in content or 'result' in content or 'conclusion' in content)):
            return 'Original Article'
        
        # 默认分类
        return 'Other'
    
    def clean_text(self, text: str) -> str:
        """
        清理文本内容
        """
        if not text:
            return ''
        
        # 移除XML标签
        import re
        text = re.sub(r'<[^>]+>', '', text)
        
        # 替换HTML实体
        text = text.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
        
        # 移除多余空白
        text = ' '.join(text.split())
        
        return text.strip()
    
    def filter_articles_by_type(self, articles: List[Dict]) -> List[Dict]:
        """
        筛选Original Article和Correspondence
        """
        filtered = []
        
        for article in articles:
            article_type = article.get('article_type', 'Other')
            if article_type in ['Original Article', 'Correspondence']:
                filtered.append(article)
        
        return filtered
    
    def save_results(self, articles: List[Dict], filename_prefix: str = "nejm_articles"):
        """
        保存结果到多个格式
        """
        if not articles:
            print("没有文章可保存")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存为CSV
        csv_file = f"{filename_prefix}_{timestamp}.csv"
        df = pd.DataFrame(articles)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"CSV文件已保存: {csv_file}")
        
        # 保存为JSON
        json_file = f"{filename_prefix}_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        print(f"JSON文件已保存: {json_file}")
        
        return csv_file, json_file

def main():
    """
    主函数
    """
    print("="*60)
    print("NEJM文章爬取脚本 - 改进版")
    print("="*60)
    
    scraper = ImprovedNEJMArticleScraper()
    
    # 设置时间范围（近5年）
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    print(f"时间范围: {start_date.strftime('%Y/%m/%d')} - {end_date.strftime('%Y/%m/%d')}")
    print(f"目标文章类型: Original Article, Correspondence")
    
    # 搜索文章
    print("\n第一步: 搜索NEJM文章...")
    pmids = scraper.search_nejm_articles(
        start_date=start_date.strftime('%Y/%m/%d'),
        end_date=end_date.strftime('%Y/%m/%d'),
        retmax=100
    )
    
    if not pmids:
        print("未找到任何文章，程序结束")
        return
    
    # 获取文章详情
    print("\n第二步: 获取文章详情...")
    articles = scraper.fetch_article_details_batch(pmids)
    
    print(f"成功获取 {len(articles)} 篇文章的详细信息")
    
    # 筛选文章类型
    print("\n第三步: 筛选文章类型...")
    filtered_articles = scraper.filter_articles_by_type(articles)
    
    print(f"筛选后剩余 {len(filtered_articles)} 篇Original Article或Correspondence")
    
    if filtered_articles:
        # 显示一些统计信息
        print("\n第四步: 统计信息")
        type_counts = {}
        for article in filtered_articles:
            article_type = article.get('article_type', 'Unknown')
            type_counts[article_type] = type_counts.get(article_type, 0) + 1
        
        print("文章类型分布:")
        for article_type, count in type_counts.items():
            print(f"  {article_type}: {count} 篇")
        
        # 保存结果
        print("\n第五步: 保存结果...")
        csv_file, json_file = scraper.save_results(filtered_articles)
        
        print("\n" + "="*60)
        print("✓ 爬取完成！")
        print(f"✓ 总共处理 {len(pmids)} 篇文章")
        print(f"✓ 获取到 {len(articles)} 篇详细信息")
        print(f"✓ 筛选出 {len(filtered_articles)} 篇目标文章")
        print(f"✓ 数据已保存为CSV和JSON格式")
        
    else:
        print("没有找到符合条件的文章")

if __name__ == "__main__":
    main()