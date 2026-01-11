# PubMed API 限制和最佳实践指南

## 📋 API限制概述

### 1. 请求频率限制
- **搜索接口**: 每秒最多3个请求
- **获取详情接口**: 每秒最多3个请求
- **批量获取**: 建议每批次之间间隔0.3-0.5秒

### 2. 批量大小限制
- **单次搜索**: 最多返回10,000个结果
- **单次获取详情**: 建议不超过100-200个PMID
- **URL长度限制**: GET请求URL不能超过2048字符

### 3. 使用要求
- **必须提供邮箱**: 用于身份识别和问题联系
- **用户代理**: 建议提供应用程序名称
- **合理使用时间**: 避免在高峰时段大量请求

### 4. 数据访问限制
- **每日总量**: 没有明确限制，但建议合理控制
- **并发连接**: 建议单线程或少量并发
- **重试机制**: 失败后建议等待1-3秒再重试

## 🚨 常见限制错误

### 429 Too Many Requests
```xml
<Error>
    <Code>429</Code>
    <Message>Too Many Requests</Message>
    <Details>Rate limit exceeded</Details>
</Error>
```

### 403 Forbidden
```xml
<Error>
    <Code>403</Code>
    <Message>Forbidden</Message>
    <Details>API key required or IP blocked</Details>
</Error>
```

### 500 Internal Error
```xml
<Error>
    <Code>500</Code>
    <Message>Internal Server Error</Message>
</Error>
```

## 💡 最佳实践建议

### 1. 请求间隔控制
```python
import time

# 搜索请求间隔
time.sleep(0.4)  # 推荐0.3-0.5秒

# 详情获取间隔  
time.sleep(0.5)  # 推荐0.5-1秒

# 错误重试间隔
time.sleep(3)    # 失败后等待3秒
```

### 2. 批量处理优化
```python
# 推荐批量大小
SEARCH_BATCH_SIZE = 100   # 搜索批次
FETCH_BATCH_SIZE = 50     # 获取详情批次
MAX_TOTAL_RESULTS = 10000  # 总结果限制
```

### 3. 错误处理和重试
```python
import requests
from time import sleep

def safe_request(url, params, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                return response
            elif response.status_code == 429:
                sleep_time = (attempt + 1) * 2  # 指数退避
                print(f"Rate limit hit, waiting {sleep_time} seconds...")
                sleep(sleep_time)
            else:
                print(f"Error {response.status_code}: {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            if attempt < max_retries - 1:
                sleep(2)
            else:
                return None
    return None
```

### 4. 请求头设置
```python
HEADERS = {
    'User-Agent': 'NEJM-Article-Scraper/1.0 (your_email@example.com)',
    'Accept': 'application/json',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive'
}
```

## 📊 性能优化策略

### 1. 渐进式爬取
```python
def progressive_crawl(target_count=1000):
    """渐进式爬取，避免一次性大量请求"""
    batch_size = 100
    total_fetched = 0
    
    while total_fetched < target_count:
        # 获取一批文章
        batch_pmids = fetch_batch(total_fetched, batch_size)
        
        if not batch_pmids:
            break
            
        # 处理这批文章
        process_batch(batch_pmids)
        
        total_fetched += len(batch_pmids)
        print(f"Progress: {total_fetched}/{target_count}")
        
        # 批次间休息
        if total_fetched < target_count:
            sleep(1)
```

### 2. 智能缓存机制
```python
import json
import os

class ArticleCache:
    def __init__(self, cache_dir="cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cached_article(self, pmid):
        """获取缓存的文章"""
        cache_file = os.path.join(self.cache_dir, f"{pmid}.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def cache_article(self, pmid, article_data):
        """缓存文章数据"""
        cache_file = os.path.join(self.cache_dir, f"{pmid}.json")
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(article_data, f, ensure_ascii=False, indent=2)
```

### 3. 断点续爬功能
```python
def crawl_with_checkpoint(start_pmids, checkpoint_file="crawl_checkpoint.json"):
    """支持断点续爬"""
    # 加载检查点
    processed_pmids = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            processed_pmids = set(checkpoint.get('processed_pmids', []))
            print(f"从检查点恢复，已处理 {len(processed_pmids)} 篇文章")
    
    # 过滤未处理的PMID
    remaining_pmids = [pmid for pmid in start_pmids if pmid not in processed_pmids]
    
    results = []
    for i, pmid in enumerate(remaining_pmids):
        try:
            # 获取文章详情
            article = fetch_article_detail(pmid)
            if article:
                results.append(article)
            
            # 更新检查点
            processed_pmids.add(pmid)
            
            # 定期保存检查点
            if (i + 1) % 50 == 0:
                save_checkpoint(checkpoint_file, processed_pmids, results)
                print(f"保存检查点: {i+1}/{len(remaining_pmids)}")
                
        except Exception as e:
            print(f"处理PMID {pmid} 失败: {e}")
            continue
    
    # 最终保存
    save_checkpoint(checkpoint_file, processed_pmids, results)
    return results
```

## 🎯 推荐的爬取策略

### 1. 分时段爬取
```python
def smart_crawl_schedule():
    """智能爬取时间安排"""
    import datetime
    
    now = datetime.datetime.now()
    hour = now.hour
    
    # 避开高峰时段 (9-17点)
    if 9 <= hour <= 17:
        print("高峰时段，延长等待时间")
        sleep_time = 2.0
    else:
        print("非高峰时段，正常速度")
        sleep_time = 0.5
    
    return sleep_time
```

### 2. 优先级队列
```python
from queue import PriorityQueue

def priority_based_crawl(pmids_with_priority):
    """基于优先级的爬取"""
    pq = PriorityQueue()
    
    # 添加PMID到优先级队列
    for priority, pmid in pmids_with_priority:
        pq.put((priority, pmid))
    
    results = []
    while not pq.empty():
        priority, pmid = pq.get()
        
        try:
            article = fetch_article_detail(pmid)
            if article:
                results.append((priority, article))
                print(f"高优先级文章获取成功: PMID {pmid}")
        except Exception as e:
            print(f"优先级 {priority} PMID {pmid} 失败: {e}")
        
        sleep(0.5)  # 控制频率
    
    return results
```

## 📈 监控和统计

### 1. 爬取统计
```python
class CrawlStats:
    def __init__(self):
        self.start_time = time.time()
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.rate_limit_hits = 0
    
    def log_request(self, success, rate_limited=False):
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        if rate_limited:
            self.rate_limit_hits += 1
    
    def get_stats(self):
        elapsed_time = time.time() - self.start_time
        success_rate = (self.successful_requests / max(self.total_requests, 1)) * 100
        
        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'rate_limit_hits': self.rate_limit_hits,
            'success_rate': f"{success_rate:.1f}%",
            'elapsed_time': f"{elapsed_time:.1f}秒",
            'requests_per_second': self.total_requests / max(elapsed_time, 1)
        }
```

### 2. 实时监控
```python
def monitor_crawl_progress(stats, check_interval=60):
    """实时监控爬取进度"""
    while True:
        current_stats = stats.get_stats()
        print(f"\n{'='*50}")
        print(f"📊 爬取统计 (每{check_interval}秒更新)")
        print(f"总请求数: {current_stats['total_requests']}")
        print(f"成功: {current_stats['successful_requests']}")
        print(f"失败: {current_stats['failed_requests']}")
        print(f"成功率: {current_stats['success_rate']}")
        print(f"速率限制: {current_stats['rate_limit_hits']}")
        print(f"请求速度: {current_stats['requests_per_second']:.2f}/秒")
        print(f"运行时间: {current_stats['elapsed_time']}")
        print(f"{'='*50}\n")
        
        time.sleep(check_interval)
```

## 🔧 故障排除

### 1. 连接超时
```python
# 增加超时时间
response = requests.get(url, params=params, timeout=60)

# 使用会话保持连接
session = requests.Session()
session.headers.update({'Connection': 'keep-alive'})
```

### 2. 内存优化
```python
# 流式处理大文件
def process_large_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 逐行处理，避免内存溢出
            process_line(line.strip())
```

### 3. 代理设置
```python
# 使用代理避免IP限制
proxies = {
    'http': 'http://proxy.example.com:8080',
    'https': 'https://proxy.example.com:8080'
}

response = requests.get(url, params=params, proxies=proxies)
```

## 📚 官方文档参考

- **PubMed E-utilities**: https://www.ncbi.nlm.nih.gov/books/NBK25501/
- **Rate Limiting Guidelines**: https://www.ncbi.nlm.nih.gov/home/about/policies/
- **Best Practices**: https://www.ncbi.nlm.nih.gov/pmc/tools/developers/

记住：**合理爬取，尊重服务器资源，避免影响其他用户！**