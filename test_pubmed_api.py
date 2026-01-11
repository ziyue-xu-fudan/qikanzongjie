#!/usr/bin/env python3
"""
简化版NEJM文章爬取脚本
用于快速测试PubMed API连接
"""

import requests
import json
from datetime import datetime, timedelta
import time

def test_pubmed_search():
    """测试PubMed搜索功能"""
    print("开始测试PubMed API...")
    
    # 设置时间范围（近5年）
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    # 基础搜索URL
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    
    # 简化搜索查询 - 只搜索NEJM期刊
    query = '"N Engl J Med"[Journal]'
    
    params = {
        'db': 'pubmed',
        'term': query,
        'retmode': 'json',
        'retmax': 10,  # 先测试少量结果
        'retstart': 0
    }
    
    try:
        print(f"搜索查询: {query}")
        print(f"时间范围: {start_date.strftime('%Y/%m/%d')} - {end_date.strftime('%Y/%m/%d')}")
        
        response = requests.get(search_url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        esearchresult = data.get('esearchresult', {})
        
        pmids = esearchresult.get('idlist', [])
        count = int(esearchresult.get('count', 0))
        
        print(f"找到 {count} 篇文章")
        print(f"获取到 {len(pmids)} 个PMID")
        
        if pmids:
            print("前10个PMID:")
            for pmid in pmids:
                print(f"  - {pmid}")
                
            # 测试获取第一篇文章的详情
            print("\n测试获取文章详情...")
            fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            
            fetch_params = {
                'db': 'pubmed',
                'id': pmids[0],
                'retmode': 'xml'
            }
            
            fetch_response = requests.get(fetch_url, params=fetch_params, timeout=30)
            fetch_response.raise_for_status()
            
            # 简单的XML内容检查
            xml_content = fetch_response.text
            if '<ArticleTitle>' in xml_content:
                print("✓ 成功获取文章XML数据")
                # 提取标题（简单方法）
                if '<ArticleTitle>' in xml_content:
                    start = xml_content.find('<ArticleTitle>') + len('<ArticleTitle>')
                    end = xml_content.find('</ArticleTitle>')
                    if start > len('<ArticleTitle>') and end > start:
                        title = xml_content[start:end]
                        print(f"文章标题: {title}")
            else:
                print("✗ 未能获取有效的文章数据")
                
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return False
    except Exception as e:
        print(f"发生错误: {e}")
        return False

def test_advanced_search():
    """测试高级搜索功能"""
    print("\n" + "="*50)
    print("测试高级搜索功能...")
    
    # 搜索Original Article
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    
    # 构建复杂查询
    query = '"N Engl J Med"[Journal] AND "Original Article"[Publication Type]'
    
    params = {
        'db': 'pubmed',
        'term': query,
        'retmode': 'json',
        'retmax': 5,
        'retstart': 0
    }
    
    try:
        print(f"搜索查询: {query}")
        
        response = requests.get(search_url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        esearchresult = data.get('esearchresult', {})
        
        pmids = esearchresult.get('idlist', [])
        count = int(esearchresult.get('count', 0))
        
        print(f"找到 {count} 篇Original Article")
        print(f"获取到 {len(pmids)} 个PMID")
        
        return count > 0
        
    except Exception as e:
        print(f"高级搜索失败: {e}")
        return False

if __name__ == "__main__":
    print("NEJM文章爬取脚本 - 测试版本")
    print("="*50)
    
    # 测试基础搜索
    success1 = test_pubmed_search()
    
    # 测试高级搜索
    success2 = test_advanced_search()
    
    print("\n" + "="*50)
    print("测试结果:")
    print(f"基础搜索: {'✓ 通过' if success1 else '✗ 失败'}")
    print(f"高级搜索: {'✓ 通过' if success2 else '✗ 失败'}")
    
    if success1 and success2:
        print("\n✓ PubMed API连接正常，可以开始完整爬取！")
        print("建议运行: python enhanced_nejm_scraper.py")
    else:
        print("\n✗ 存在连接问题，请检查网络设置或稍后重试")