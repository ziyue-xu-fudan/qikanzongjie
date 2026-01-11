#!/usr/bin/env python3
"""
专业版NEJM文献爬取脚本
专门针对新英格兰医学杂志的文献爬取，包含完整的文章类型识别和增量更新功能
"""

import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import re
from typing import List, Dict, Optional, Set
from pathlib import Path

class NEJMLiteratureScraper:
    def __init__(self, email: str = "nejm.scraper@example.com"):
        """
        初始化NEJM专业爬取器
        
        Args:
            email: 用于PubMed API的邮箱地址
        """
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.search_url = self.base_url + "esearch.fcgi"
        self.fetch_url = self.base_url + "efetch.fcgi"
        self.journal_name = "N Engl J Med"
        self.email = email
        
        # NEJM特定的文章类型识别模式
        self.nejm_patterns = {
            'original_article': [
                r'original article',
                r'original research',
                r'clinical research',
                r'research article',
                r'clinical trial',
                r'observational study',
                r'randomized.*trial',
                r'prospective.*study',
                r'retrospective.*study'
            ],
            'correspondence': [
                r'correspondence',
                r'letter.*to.*editor',
                r'reply.*to',
                r'response.*to',
                r'letter.*regarding',
                r're.*:',  # 回复类标题
                r'^[^.]*\.\s*reply\.',  # 以"reply."结尾的标题
                r'^[^.]*\.\s*response\.'  # 以"response."结尾的标题
            ],
            'review': [
                r'review.*article',
                r'systematic.*review',
                r'meta.*analysis',
                r'narrative.*review',
                r'clinical.*review'
            ],
            'case_report': [
                r'case.*report',
                r'case.*series',
                r'clinical.*case'
            ],
            'editorial': [
                r'editorial',
                r'editor\'s.*note',
                r'perspective',
                r'viewpoint'
            ]
        }
        
        # 设置请求头
        self.headers = {
            'User-Agent': f'NEJM-Literature-Scraper/1.0 ({email})',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        }
        
        # 缓存和状态管理
        self.cache_dir = Path("nejm_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.processed_pmids: Set[str] = set()
        self.stats = {
            'total_searched': 0,
            'total_fetched': 0,
            'by_type': {},
            'by_year': {},
            'errors': 0
        }

    def search_nejm_literature(self, start_date: str, end_date: str, max_results: int = 5000) -> List[str]:
        """
        搜索NEJM文献（专业版）
        
        Args:
            start_date: 开始日期 (YYYY/MM/DD)
            end_date: 结束日期 (YYYY/MM/DD)
            max_results: 最大返回结果数
            
        Returns:
            PMID列表
        """
        print(f"🔍 搜索NEJM文献: {start_date} - {end_date}")
        
        # 构建专业搜索查询
        base_query = f'"{self.journal_name}"[Journal] AND ("{start_date}"[Date - Publication] : "{end_date}"[Date - Publication])'
        
        all_pmids = []
        retmax = 200  # 每批次的数量
        retstart = 0
        
        while retstart < max_results:
            params = {
                'db': 'pubmed',
                'term': base_query,
                'retmode': 'json',
                'retmax': min(retmax, max_results - retstart),
                'retstart': retstart,
                'email': self.email
            }
            
            try:
                response = requests.get(self.search_url, params=params, headers=self.headers, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                esearchresult = data.get('esearchresult', {})
                
                batch_pmids = esearchresult.get('idlist', [])
                if not batch_pmids:
                    break
                    
                all_pmids.extend(batch_pmids)
                total_found = int(esearchresult.get('count', 0))
                
                print(f"📄 已获取: {len(all_pmids)}/{total_found} 篇文献")
                
                # 检查是否达到限制
                if len(all_pmids) >= max_results:
                    all_pmids = all_pmids[:max_results]
                    break
                
                retstart += len(batch_pmids)
                time.sleep(0.5)  # 更长的延迟，尊重服务器
                
            except requests.exceptions.RequestException as e:
                print(f"❌ 搜索请求失败: {e}")
                self.stats['errors'] += 1
                time.sleep(2)  # 错误后等待更长时间
                continue
                
        self.stats['total_searched'] = len(all_pmids)
        print(f"✅ 搜索完成，共找到 {len(all_pmids)} 篇NEJM文献")
        return all_pmids

    def fetch_literature_details(self, pmids: List[str]) -> List[Dict]:
        """
        获取文献详细信息（专业版）
        """
        if not pmids:
            return []
            
        print(f"📖 开始获取文献详情: {len(pmids)} 篇")
        articles = []
        batch_size = 50  # 较小的批次，提高稳定性
        
        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i+batch_size]
            print(f"📚 处理批次: {i+1}-{min(i+batch_size, len(pmids))}/{len(pmids)}")
            
            params = {
                'db': 'pubmed',
                'id': ','.join(batch_pmids),
                'retmode': 'xml',
                'email': self.email
            }
            
            try:
                response = requests.get(self.fetch_url, params=params, headers=self.headers, timeout=60)
                response.raise_for_status()
                
                batch_articles = self.parse_nejm_xml(response.text)
                articles.extend(batch_articles)
                
                print(f"✅ 批次完成: 获取到 {len(batch_articles)} 篇详细信息")
                time.sleep(1)  # 批次间较长延迟
                
            except Exception as e:
                print(f"❌ 获取批次详情失败: {e}")
                self.stats['errors'] += 1
                time.sleep(3)  # 错误后更长等待
                continue
        
        self.stats['total_fetched'] = len(articles)
        return articles

    def parse_nejm_xml(self, xml_content: str) -> List[Dict]:
        """
        专业版NEJM XML解析
        """
        articles = []
        
        # 使用正则表达式解析XML（更稳定）
        article_blocks = re.findall(r'<PubmedArticle>(.*?)</PubmedArticle>', xml_content, re.DOTALL)
        
        for block in article_blocks:
            try:
                article = self.extract_nejm_article_info(block)
                if article and article.get('title'):
                    articles.append(article)
            except Exception as e:
                print(f"⚠️  解析单篇文献失败: {e}")
                continue
                
        return articles

    def extract_nejm_article_info(self, xml_block: str) -> Optional[Dict]:
        """
        提取NEJM文章的专业信息
        """
        article = {}
        
        # PMID
        pmid_match = re.search(r'<PMID[^>]*>(\d+)</PMID>', xml_block)
        article['pmid'] = pmid_match.group(1) if pmid_match else ''
        
        if not article['pmid']:
            return None
        
        # 标题（更精确提取）
        title_match = re.search(r'<ArticleTitle[^>]*>(.*?)</ArticleTitle>', xml_block, re.DOTALL)
        if title_match:
            title = self.clean_xml_text(title_match.group(1))
            article['title'] = title
        else:
            article['title'] = ''
        
        # 作者（更完整提取）
        authors = []
        author_blocks = re.findall(r'<Author[^>]*>(.*?)</Author>', xml_block, re.DOTALL)
        
        for author_block in author_blocks:
            lastname = self.extract_xml_field(author_block, 'LastName')
            forename = self.extract_xml_field(author_block, 'ForeName')
            initials = self.extract_xml_field(author_block, 'Initials')
            
            if lastname:
                if forename:
                    authors.append(f"{lastname} {forename}")
                elif initials:
                    authors.append(f"{lastname} {initials}")
                else:
                    authors.append(lastname)
        
        article['authors'] = ', '.join(authors[:8])  # 限制前8个作者
        article['author_count'] = len(authors)
        
        # 期刊信息
        journal_title = self.extract_xml_field(xml_block, 'Title')
        article['journal'] = journal_title if journal_title else self.journal_name
        
        # 发表日期（更详细）
        year = self.extract_xml_field(xml_block, 'Year')
        month = self.extract_xml_field(xml_block, 'Month')
        day = self.extract_xml_field(xml_block, 'Day')
        
        article['pub_year'] = year
        article['pub_month'] = month
        article['pub_day'] = day
        article['pub_date'] = self.format_date(year, month, day)
        
        # DOI
        doi_match = re.search(r'<ELocationID[^>]*EIdType="doi"[^>]*>(.*?)</ELocationID>', xml_block)
        article['doi'] = doi_match.group(1) if doi_match else ''
        
        # 摘要（更完整提取）
        abstract_blocks = re.findall(r'<AbstractText[^>]*>(.*?)</AbstractText>', xml_block, re.DOTALL)
        abstract_parts = []
        
        for abstract_block in abstract_blocks:
            abstract_text = self.clean_xml_text(abstract_block)
            if abstract_text and len(abstract_text) > 10:  # 过滤过短的摘要
                abstract_parts.append(abstract_text)
        
        article['abstract'] = ' '.join(abstract_parts) if abstract_parts else ''
        article['abstract_length'] = len(article['abstract'])
        
        # 文章类型（专业识别）
        article['article_type'] = self.classify_nejm_article_type(xml_block, article['title'], article['abstract'])
        
        # 关键词
        keywords = []
        keyword_matches = re.findall(r'<Keyword[^>]*>(.*?)</Keyword>', xml_block)
        for keyword in keyword_matches:
            clean_keyword = self.clean_xml_text(keyword)
            if clean_keyword:
                keywords.append(clean_keyword)
        
        article['keywords'] = ', '.join(keywords[:10])  # 限制前10个关键词
        article['keyword_count'] = len(keywords)
        
        # 额外信息
        article['language'] = self.extract_xml_field(xml_block, 'Language')
        article['publication_types'] = self.extract_publication_types(xml_block)
        
        # 抓取时间戳
        article['scraped_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        article['scraped_timestamp'] = int(time.time())
        
        return article

    def extract_xml_field(self, xml_block: str, field_name: str) -> str:
        """提取XML字段"""
        pattern = rf'<{field_name}[^>]*>(.*?)</{field_name}>'
        match = re.search(pattern, xml_block, re.DOTALL)
        return self.clean_xml_text(match.group(1)) if match else ''

    def extract_publication_types(self, xml_block: str) -> str:
        """提取发表类型"""
        types = []
        type_matches = re.findall(r'<PublicationType[^>]*>(.*?)</PublicationType>', xml_block)
        
        for type_match in type_matches:
            clean_type = self.clean_xml_text(type_match)
            if clean_type:
                types.append(clean_type)
        
        return ', '.join(types[:5])  # 限制前5个类型

    def classify_nejm_article_type(self, xml_block: str, title: str, abstract: str) -> str:
        """
        专业NEJM文章类型分类
        """
        content = (xml_block + " " + title + " " + abstract).lower()
        
        # 检查Correspondence（最高优先级）
        for pattern in self.nejm_patterns['correspondence']:
            if re.search(pattern, content, re.IGNORECASE):
                return 'Correspondence'
        
        # 检查Original Article
        for pattern in self.nejm_patterns['original_article']:
            if re.search(pattern, content, re.IGNORECASE):
                return 'Original Article'
        
        # 检查Review
        for pattern in self.nejm_patterns['review']:
            if re.search(pattern, content, re.IGNORECASE):
                return 'Review'
        
        # 检查Case Report
        for pattern in self.nejm_patterns['case_report']:
            if re.search(pattern, content, re.IGNORECASE):
                return 'Case Report'
        
        # 检查Editorial
        for pattern in self.nejm_patterns['editorial']:
            if re.search(pattern, content, re.IGNORECASE):
                return 'Editorial'
        
        # 基于标题特征的额外检查
        title_lower = title.lower()
        
        # Correspondence的标题特征
        if any(word in title_lower for word in ['reply', 'response', 'letter', 'correspondence']):
            return 'Correspondence'
        
        # Original Article的标题特征
        if any(word in title_lower for word in ['trial', 'study', 'effect', 'efficacy', 'safety', 'outcome']):
            if len(abstract) > 500:  # Original Article通常有较长的摘要
                return 'Original Article'
        
        # 默认分类
        return 'Other'

    def clean_xml_text(self, text: str) -> str:
        """清理XML文本"""
        if not text:
            return ''
        
        # 移除XML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 替换HTML实体
        text = text.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
        text = text.replace('&quot;', '"').replace('&apos;', "'")
        
        # 移除多余空白和特殊字符
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text

    def format_date(self, year: str, month: str, day: str) -> str:
        """格式化日期"""
        if not year:
            return ''
        
        date_parts = [year]
        if month:
            date_parts.append(month)
        if day:
            date_parts.append(day)
        
        return ' '.join(date_parts)

    def filter_target_articles(self, articles: List[Dict]) -> List[Dict]:
        """
        筛选目标文章类型（Original Article和Correspondence）
        """
        target_types = ['Original Article', 'Correspondence']
        filtered = []
        
        for article in articles:
            article_type = article.get('article_type', 'Other')
            if article_type in target_types:
                filtered.append(article)
                
                # 更新统计
                self.stats['by_type'][article_type] = self.stats['by_type'].get(article_type, 0) + 1
                
                # 按年份统计
                year = article.get('pub_year', 'Unknown')
                if year and year.isdigit():
                    self.stats['by_year'][year] = self.stats['by_year'].get(year, 0) + 1
        
        return filtered

    def save_literature_data(self, articles: List[Dict], base_filename: str = None) -> Dict[str, str]:
        """
        保存文献数据到多种格式
        """
        if not base_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"nejm_literature_{timestamp}"
        
        saved_files = {}
        
        try:
            # CSV格式
            csv_file = f"{base_filename}.csv"
            df = pd.DataFrame(articles)
            
            # 确保列顺序合理
            columns = ['pmid', 'title', 'authors', 'author_count', 'journal', 'pub_year', 
                      'pub_month', 'pub_day', 'pub_date', 'doi', 'article_type', 
                      'abstract', 'abstract_length', 'keywords', 'keyword_count',
                      'language', 'publication_types', 'scraped_date']
            
            # 只保留存在的列
            existing_columns = [col for col in columns if col in df.columns]
            df = df[existing_columns]
            
            df.to_csv(csv_file, index=False, encoding='utf-8')
            saved_files['csv'] = csv_file
            
            # JSON格式（包含所有数据）
            json_file = f"{base_filename}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(articles, f, ensure_ascii=False, indent=2)
            saved_files['json'] = json_file
            
            # Excel格式
            excel_file = f"{base_filename}.xlsx"
            df.to_excel(excel_file, index=False)
            saved_files['excel'] = excel_file
            
            # Markdown格式（摘要版）
            markdown_file = f"{base_filename}_summary.md"
            self.create_markdown_summary(articles, markdown_file)
            saved_files['markdown'] = markdown_file
            
            # 统计报告
            stats_file = f"{base_filename}_stats.json"
            self.save_statistics(stats_file)
            saved_files['stats'] = stats_file
            
        except Exception as e:
            print(f"❌ 保存数据失败: {e}")
            
        return saved_files

    def create_markdown_summary(self, articles: List[Dict], filename: str):
        """创建Markdown格式的摘要报告"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# NEJM文献摘要报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"总文献数: {len(articles)}\n\n")
            
            # 按类型分组
            by_type = {}
            for article in articles:
                article_type = article.get('article_type', 'Unknown')
                if article_type not in by_type:
                    by_type[article_type] = []
                by_type[article_type].append(article)
            
            for article_type, type_articles in by_type.items():
                f.write(f"## {article_type} ({len(type_articles)}篇)\n\n")
                
                # 显示前10篇
                for i, article in enumerate(type_articles[:10], 1):
                    title = article.get('title', '无标题')
                    authors = article.get('authors', '未知作者')
                    pub_date = article.get('pub_date', '未知日期')
                    pmid = article.get('pmid', '')
                    
                    f.write(f"### {i}. {title}\n")
                    f.write(f"- **作者**: {authors}\n")
                    f.write(f"- **发表日期**: {pub_date}\n")
                    f.write(f"- **PMID**: {pmid}\n")
                    
                    abstract = article.get('abstract', '')
                    if abstract:
                        abstract_summary = abstract[:300] + '...' if len(abstract) > 300 else abstract
                        f.write(f"- **摘要**: {abstract_summary}\n")
                    
                    f.write("\n")
                
                if len(type_articles) > 10:
                    f.write(f"*... 还有 {len(type_articles) - 10} 篇文献*\n\n")

    def save_statistics(self, filename: str):
        """保存统计信息"""
        stats_data = {
            'summary': {
                'total_searched': self.stats['total_searched'],
                'total_fetched': self.stats['total_fetched'],
                'errors': self.stats['errors'],
                'success_rate': f"{(self.stats['total_fetched'] / max(self.stats['total_searched'], 1) * 100):.1f}%"
            },
            'by_type': self.stats['by_type'],
            'by_year': dict(sorted(self.stats['by_year'].items())),
            'generated_at': datetime.now().isoformat()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, ensure_ascii=False, indent=2)

    def display_statistics(self):
        """显示统计信息"""
        print("\n" + "="*60)
        print("📊 NEJM文献爬取统计报告")
        print("="*60)
        
        print(f"🔍 总搜索文献数: {self.stats['total_searched']}")
        print(f"📖 成功获取详情: {self.stats['total_fetched']}")
        print(f"✅ 成功率: {(self.stats['total_fetched'] / max(self.stats['total_searched'], 1) * 100):.1f}%")
        print(f"❌ 错误数: {self.stats['errors']}")
        
        if self.stats['by_type']:
            print(f"\n📚 按文章类型分布:")
            for article_type, count in sorted(self.stats['by_type'].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / sum(self.stats['by_type'].values())) * 100
                print(f"  {article_type}: {count}篇 ({percentage:.1f}%)")
        
        if self.stats['by_year']:
            print(f"\n📅 按发表年份分布:")
            for year, count in sorted(self.stats['by_year'].items(), reverse=True):
                print(f"  {year}: {count}篇")
        
        print("="*60)

def main():
    """
    主函数
    """
    print("🏥 NEJM专业文献爬取工具")
    print("="*60)
    print("专门用于爬取《新英格兰医学杂志》的高质量文献")
    print("="*60)
    
    # 创建爬取器
    scraper = NEJMLiteratureScraper(email="nejm.research@example.com")
    
    # 设置时间范围（近5年）
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    print(f"📅 时间范围: {start_date.strftime('%Y/%m/%d')} - {end_date.strftime('%Y/%m/%d')}")
    print(f"🎯 目标类型: Original Article, Correspondence")
    print(f"📧 使用邮箱: {scraper.email}")
    
    # 第一步：搜索文献
    print(f"\n🔍 第一步: 搜索NEJM文献...")
    pmids = scraper.search_nejm_literature(
        start_date=start_date.strftime('%Y/%m/%d'),
        end_date=end_date.strftime('%Y/%m/%d'),
        max_results=2000  # 限制数量避免过多
    )
    
    if not pmids:
        print("❌ 未找到任何文献，程序结束")
        return
    
    # 第二步：获取文献详情
    print(f"\n📖 第二步: 获取文献详情...")
    articles = scraper.fetch_literature_details(pmids)
    
    if not articles:
        print("❌ 未能获取文献详情，程序结束")
        return
    
    # 第三步：筛选目标类型
    print(f"\n🎯 第三步: 筛选目标文献类型...")
    target_articles = scraper.filter_target_articles(articles)
    
    print(f"✅ 筛选完成，共 {len(target_articles)} 篇目标文献")
    
    # 第四步：保存数据
    if target_articles:
        print(f"\n💾 第四步: 保存数据...")
        saved_files = scraper.save_literature_data(target_articles)
        
        print("✅ 数据保存完成:")
        for format_type, file_path in saved_files.items():
            print(f"  📄 {format_type.upper()}: {file_path}")
        
        # 显示统计信息
        scraper.display_statistics()
        
        # 显示部分结果预览
        print(f"\n📝 结果预览（前5篇）:")
        for i, article in enumerate(target_articles[:5], 1):
            print(f"  {i}. [{article.get('article_type', 'Unknown')}] {article.get('title', '无标题')}")
            print(f"     作者: {article.get('authors', '未知')} | PMID: {article.get('pmid', '')}")
            print(f"     发表: {article.get('pub_date', '未知')} | DOI: {article.get('doi', '')}")
            print()
        
        print(f"🎉 爬取完成！共获得 {len(target_articles)} 篇高质量NEJM文献")
        
    else:
        print("⚠️  没有找到符合条件的目标文献")

if __name__ == "__main__":
    main()