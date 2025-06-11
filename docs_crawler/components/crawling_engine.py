"""
Crawling Engine Module
Handles web crawling operations, URL processing, and content extraction
"""

import asyncio
import concurrent.futures
import re
import time
import json
import requests
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from urllib.parse import urlparse, urldefrag
from xml.etree import ElementTree
import streamlit as st

# Import crawling dependencies
try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
    from components.supabase_integration import get_supabase_integration
    CRAWL_DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    st.error(f"Crawling dependencies not available: {e}")
    CRAWL_DEPENDENCIES_AVAILABLE = False


class CrawlStatus(Enum):
    """Crawl job status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


@dataclass
class CrawlJob:
    """Configuration and status for a crawling job"""
    job_id: str
    project_id: str
    urls: List[str]
    status: CrawlStatus
    progress: float  # 0.0 to 1.0
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_message: Optional[str]
    results_summary: Dict[str, Any]
    
    # Configuration
    max_depth: int = 2
    max_concurrent: int = 10
    chunk_size: int = 4000
    include_patterns: List[str] = None
    exclude_patterns: List[str] = None
    rag_strategies: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        data['started_at'] = self.started_at.isoformat() if self.started_at else None
        data['completed_at'] = self.completed_at.isoformat() if self.completed_at else None
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CrawlJob':
        """Create from dictionary"""
        data['status'] = CrawlStatus(data['status'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data['started_at']:
            data['started_at'] = datetime.fromisoformat(data['started_at'])
        if data['completed_at']:
            data['completed_at'] = datetime.fromisoformat(data['completed_at'])
        return cls(**data)


@dataclass
class CrawlResult:
    """Result of crawling a single URL"""
    url: str
    title: str
    content: str
    chunks: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    status: str
    error_message: Optional[str]
    crawled_at: datetime


class CrawlingEngine:
    """Main crawling engine for processing URLs"""
    
    def __init__(self):
        self.active_jobs = {}
        self.job_counter = 0
        self.crawler = None
        self._load_jobs()
    
    def _load_jobs(self):
        """Load existing jobs from session state"""
        if 'crawl_jobs' in st.session_state:
            for job_id, job_data in st.session_state.crawl_jobs.items():
                self.active_jobs[job_id] = CrawlJob.from_dict(job_data)
    
    def _save_jobs(self):
        """Save jobs to session state"""
        if 'crawl_jobs' not in st.session_state:
            st.session_state.crawl_jobs = {}
        
        for job_id, job in self.active_jobs.items():
            st.session_state.crawl_jobs[job_id] = job.to_dict()
    
    async def initialize_crawler(self):
        """Initialize the Crawl4AI crawler"""
        if not CRAWL_DEPENDENCIES_AVAILABLE:
            raise Exception("Crawling dependencies not available")
        
        if self.crawler is None:
            browser_config = BrowserConfig(headless=True, verbose=False)
            self.crawler = AsyncWebCrawler(config=browser_config)
            await self.crawler.__aenter__()
        
        return self.crawler
    
    async def cleanup_crawler(self):
        """Clean up the crawler"""
        if self.crawler:
            await self.crawler.__aexit__(None, None, None)
            self.crawler = None
    
    def create_crawl_job(self, project_id: str, urls: List[str], config: Dict[str, Any]) -> CrawlJob:
        """Create a new crawling job"""
        
        self.job_counter += 1
        job_id = f"crawl_{self.job_counter:06d}"
        
        job = CrawlJob(
            job_id=job_id,
            project_id=project_id,
            urls=urls,
            status=CrawlStatus.PENDING,
            progress=0.0,
            created_at=datetime.now(),
            started_at=None,
            completed_at=None,
            error_message=None,
            results_summary={},
            max_depth=config.get('max_depth', 2),
            max_concurrent=config.get('max_concurrent', 10),
            chunk_size=config.get('chunk_size', 4000),
            include_patterns=config.get('include_patterns', []),
            exclude_patterns=config.get('exclude_patterns', []),
            rag_strategies=config.get('rag_strategies', ['vector_embeddings'])
        )
        
        self.active_jobs[job_id] = job
        self._save_jobs()
        return job
    
    async def start_crawl_job(self, job_id: str) -> bool:
        """Start executing a crawl job"""
        
        if job_id not in self.active_jobs:
            return False
        
        job = self.active_jobs[job_id]
        job.status = CrawlStatus.RUNNING
        job.started_at = datetime.now()
        
        try:
            # Initialize crawler
            await self.initialize_crawler()
            
            # Process URLs based on type detection
            results = await self._process_urls_smart(job)
            
            # Store results in Supabase
            await self._store_results_in_supabase(job.project_id, results)
            
            # Update job status
            job.status = CrawlStatus.COMPLETED
            job.completed_at = datetime.now()
            job.progress = 1.0
            job.results_summary = self._create_results_summary(results)
            
            self._save_jobs()
            return True
            
        except Exception as e:
            job.status = CrawlStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()
            self._save_jobs()
            return False
        finally:
            # Clean up crawler
            await self.cleanup_crawler()
    
    async def _process_urls_smart(self, job: CrawlJob) -> List[CrawlResult]:
        """Intelligently process URLs based on their type"""
        
        results = []
        
        for url in job.urls:
            url_type = URLProcessor.detect_url_type(url)
            
            if url_type == "sitemap":
                # Extract URLs from sitemap and crawl them
                sitemap_urls = self._parse_sitemap(url)
                batch_results = await self._crawl_batch(sitemap_urls, job)
                results.extend(batch_results)
                
            elif url_type == "text_file":
                # Crawl text file directly
                result = await self._crawl_text_file(url, job)
                if result:
                    results.append(result)
                    
            else:
                # Regular webpage - crawl recursively if depth > 1
                if job.max_depth > 1:
                    recursive_results = await self._crawl_recursive_internal_links([url], job)
                    results.extend(recursive_results)
                else:
                    result = await self._crawl_single_url(url, job)
                    if result:
                        results.append(result)
        
        return results
    
    def _parse_sitemap(self, sitemap_url: str) -> List[str]:
        """Parse a sitemap and extract URLs"""
        
        try:
            resp = requests.get(sitemap_url, timeout=30)
            urls = []
        
            if resp.status_code == 200:
                tree = ElementTree.fromstring(resp.content)
                urls = [loc.text for loc in tree.findall('.//{*}loc')]
        
            return urls
        except Exception as e:
            print(f"Error parsing sitemap {sitemap_url}: {e}")
            return []
    
    async def _crawl_text_file(self, url: str, job: CrawlJob) -> Optional[CrawlResult]:
        """Crawl a text file URL"""
        
        try:
            run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
            result = await self.crawler.arun(url=url, config=run_config)
            
            if result.success and result.markdown:
                chunks = self._smart_chunk_markdown(result.markdown, job.chunk_size)
                
                return CrawlResult(
                    url=url,
                    title=f"Text file from {url}",
                    content=result.markdown,
                    chunks=chunks,
                    metadata={
                        "content_type": "text/plain",
                        "word_count": len(result.markdown.split()),
                        "char_count": len(result.markdown)
                    },
                    status="success",
                    error_message=None,
                    crawled_at=datetime.now()
                )
            else:
                return CrawlResult(
                    url=url,
                    title="",
                    content="",
                    chunks=[],
                    metadata={},
                    status="failed",
                    error_message=result.error_message or "Unknown error",
                    crawled_at=datetime.now()
                )
                
        except Exception as e:
            return CrawlResult(
                url=url,
                title="",
                content="",
                chunks=[],
                metadata={},
                status="failed",
                error_message=str(e),
                crawled_at=datetime.now()
            )
    
    async def _crawl_batch(self, urls: List[str], job: CrawlJob) -> List[CrawlResult]:
        """Crawl multiple URLs in parallel"""
        
        # Apply include/exclude patterns
        if job.include_patterns or job.exclude_patterns:
            urls = URLProcessor.apply_include_exclude_patterns(
                urls, job.include_patterns or [], job.exclude_patterns or []
            )
        
        # Process in batches to avoid overwhelming the system
        results = []
        semaphore = asyncio.Semaphore(job.max_concurrent)
        
        async def crawl_with_semaphore(url: str) -> Optional[CrawlResult]:
            async with semaphore:
                return await self._crawl_single_url(url, job)
        
        total_urls = len(urls)
        processed = 0
        
        # Process URLs in chunks
        for i in range(0, len(urls), job.max_concurrent):
            batch_urls = urls[i:i + job.max_concurrent]
            
            tasks = [crawl_with_semaphore(url) for url in batch_urls]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter valid results
            for result in batch_results:
                if isinstance(result, CrawlResult):
                    results.append(result)
                processed += 1
                
                # Update progress
                job.progress = processed / total_urls
        
        return results
    
    async def _crawl_recursive_internal_links(self, start_urls: List[str], job: CrawlJob) -> List[CrawlResult]:
        """Crawl URLs recursively following internal links"""
        
        def normalize_url(url):
            """Normalize URL for comparison"""
            parsed = urlparse(url)
            return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        
        visited_urls = set()
        all_results = []
        current_depth = 0
        urls_to_process = start_urls.copy()
        
        base_domains = {urlparse(url).netloc for url in start_urls}
        
        while urls_to_process and current_depth < job.max_depth:
            current_batch = urls_to_process.copy()
            urls_to_process.clear()
            
            # Crawl current batch
            batch_results = await self._crawl_batch(current_batch, job)
            all_results.extend(batch_results)
            
            # Extract internal links for next depth level
            if current_depth < job.max_depth - 1:
                for result in batch_results:
                    if result.status == "success":
                        visited_urls.add(normalize_url(result.url))
                        
                        # Extract links from content (simplified)
                        links = self._extract_links_from_content(result.content, result.url)
                        
                        for link in links:
                            normalized_link = normalize_url(link)
                            link_domain = urlparse(link).netloc
                            
                            # Only follow internal links
                            if (link_domain in base_domains and 
                                normalized_link not in visited_urls and 
                                link not in urls_to_process):
                                urls_to_process.append(link)
            
            current_depth += 1
        
        return all_results
    
    def _extract_links_from_content(self, content: str, base_url: str) -> List[str]:
        """Extract links from markdown content"""
        
        # Simple regex to find markdown links
        link_pattern = r'\[([^\]]*)\]\(([^)]+)\)'
        matches = re.findall(link_pattern, content)
        
        links = []
        base_parsed = urlparse(base_url)
        
        for _, url in matches:
            # Handle relative URLs
            if url.startswith('/'):
                full_url = f"{base_parsed.scheme}://{base_parsed.netloc}{url}"
                links.append(full_url)
            elif url.startswith('http'):
                links.append(url)
        
        return links
    
    async def _crawl_single_url(self, url: str, job: CrawlJob) -> Optional[CrawlResult]:
        """Crawl a single URL and extract content"""
        
        try:
            run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
            result = await self.crawler.arun(url=url, config=run_config)
            
            if result.success and result.markdown:
                chunks = self._smart_chunk_markdown(result.markdown, job.chunk_size)
                
                return CrawlResult(
                    url=url,
                    title=getattr(result, 'title', f"Page from {url}"),
                    content=result.markdown,
                    chunks=chunks,
                    metadata={
                        "content_type": "text/html",
                        "word_count": len(result.markdown.split()),
                        "char_count": len(result.markdown),
                        "links": getattr(result, 'links', {})
                    },
                    status="success",
                    error_message=None,
                    crawled_at=datetime.now()
                )
            else:
                return CrawlResult(
                    url=url,
                    title="",
                    content="",
                    chunks=[],
                    metadata={},
                    status="failed",
                    error_message=result.error_message or "No content extracted",
                    crawled_at=datetime.now()
                )
                
        except Exception as e:
            return CrawlResult(
                url=url,
                title="",
                content="",
                chunks=[],
                metadata={},
                status="failed",
                error_message=str(e),
                crawled_at=datetime.now()
            )
    
    def _smart_chunk_markdown(self, text: str, chunk_size: int = 5000) -> List[Dict[str, Any]]:
        """Split text into chunks, respecting code blocks and paragraphs"""
        
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size

            if end >= text_length:
                chunk_text = text[start:].strip()
                if chunk_text:
                    chunks.append({
                        "text": chunk_text,
                        "chunk_index": len(chunks),
                        "char_count": len(chunk_text),
                        "word_count": len(chunk_text.split()),
                        "headers": self._extract_headers(chunk_text)
                    })
                break

            # Try to find a good break point
            chunk = text[start:end]
            
            # Look for code block boundary
            code_block = chunk.rfind('```')
            if code_block != -1 and code_block > chunk_size * 0.3:
                end = start + code_block
            # Look for paragraph break
            elif '\n\n' in chunk:
                last_break = chunk.rfind('\n\n')
                if last_break > chunk_size * 0.3:
                    end = start + last_break
            # Look for sentence break
            elif '. ' in chunk:
                last_period = chunk.rfind('. ')
                if last_period > chunk_size * 0.3:
                    end = start + last_period + 1

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "chunk_index": len(chunks),
                    "char_count": len(chunk_text),
                    "word_count": len(chunk_text.split()),
                    "headers": self._extract_headers(chunk_text)
                })

            start = end

        return chunks
    
    def _extract_headers(self, chunk: str) -> str:
        """Extract headers from a markdown chunk"""
        
        headers = re.findall(r'^(#+)\s+(.+)$', chunk, re.MULTILINE)
        return '; '.join([f'{h[0]} {h[1]}' for h in headers]) if headers else ''
    
    async def _store_results_in_supabase(self, project_id: str, results: List[CrawlResult]):
        """Store crawl results in Supabase"""
        
        try:
            supabase_integration = get_supabase_integration()
            
            # Prepare data for batch insertion
            urls = []
            chunk_numbers = []
            contents = []
            metadatas = []
            
            for result in results:
                if result.status == "success":
                    for chunk in result.chunks:
                        urls.append(result.url)
                        chunk_numbers.append(chunk['chunk_index'])
                        contents.append(chunk['text'])
                        
                        metadata = {
                            "title": result.title,
                            "chunk_size": chunk['char_count'],
                            "word_count": chunk['word_count'],
                            "headers": chunk['headers'],
                            "crawled_at": result.crawled_at.isoformat(),
                            **result.metadata
                        }
                        metadatas.append(metadata)
            
            # Store in Supabase
            if urls:
                success = supabase_integration.add_documents_to_supabase(
                    project_id=project_id,
                    urls=urls,
                    chunk_numbers=chunk_numbers,
                    contents=contents,
                    metadatas=metadatas
                )
                
                if not success:
                    raise Exception("Failed to store results in Supabase")
                    
        except Exception as e:
            print(f"Error storing results in Supabase: {e}")
            raise
    
    def _create_results_summary(self, results: List[CrawlResult]) -> Dict[str, Any]:
        """Create summary statistics for crawl results"""
        
        total_urls = len(results)
        successful = len([r for r in results if r.status == "success"])
        failed = total_urls - successful
        
        total_chunks = sum(len(r.chunks) for r in results)
        total_words = sum(r.metadata.get('word_count', 0) for r in results)
        
        return {
            "total_urls": total_urls,
            "successful": successful,
            "failed": failed,
            "total_chunks": total_chunks,
            "total_words": total_words,
            "avg_chunks_per_url": total_chunks / total_urls if total_urls > 0 else 0,
            "failed_urls": [r.url for r in results if r.status == "failed"]
        }
    
    def get_job_status(self, job_id: str) -> Optional[CrawlJob]:
        """Get the current status of a crawl job"""
        return self.active_jobs.get(job_id)
    
    def pause_job(self, job_id: str) -> bool:
        """Pause a running crawl job"""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            if job.status == CrawlStatus.RUNNING:
                job.status = CrawlStatus.PAUSED
                self._save_jobs()
                return True
        return False
    
    def resume_job(self, job_id: str) -> bool:
        """Resume a paused crawl job"""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            if job.status == CrawlStatus.PAUSED:
                job.status = CrawlStatus.RUNNING
                self._save_jobs()
                return True
        return False
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a crawl job"""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            job.status = CrawlStatus.CANCELLED
            job.completed_at = datetime.now()
            self._save_jobs()
            return True
        return False
    
    def list_jobs(self, project_id: str = None) -> List[CrawlJob]:
        """List all crawl jobs, optionally filtered by project"""
        
        jobs = list(self.active_jobs.values())
        
        if project_id:
            jobs = [job for job in jobs if job.project_id == project_id]
        
        return sorted(jobs, key=lambda x: x.created_at, reverse=True)
    
    def delete_job(self, job_id: str) -> bool:
        """Delete a completed or failed crawl job"""
        
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            if job.status in [CrawlStatus.COMPLETED, CrawlStatus.FAILED, CrawlStatus.CANCELLED]:
                del self.active_jobs[job_id]
                self._save_jobs()
                return True
        
        return False


class URLProcessor:
    """Utility class for URL processing and validation"""
    
    @staticmethod
    def detect_url_type(url: str) -> str:
        """Detect the type of URL (sitemap, text file, etc.)"""
        
        url_lower = url.lower()
        
        if 'sitemap' in url_lower or url_lower.endswith('sitemap.xml'):
            return "sitemap"
        elif url_lower.endswith('.txt'):
            return "text_file"
        elif url_lower.endswith('.xml'):
            return "xml_file"
        elif url_lower.endswith('.json'):
            return "json_file"
        else:
            return "webpage"
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate if URL is properly formatted"""
        
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        return url_pattern.match(url) is not None
    
    @staticmethod
    def apply_include_exclude_patterns(urls: List[str], include_patterns: List[str], 
                                     exclude_patterns: List[str]) -> List[str]:
        """Filter URLs based on include/exclude patterns"""
        
        filtered_urls = []
        
        for url in urls:
            # Check exclude patterns first
            excluded = False
            for pattern in exclude_patterns:
                if re.search(pattern, url):
                    excluded = True
                    break
            
            if excluded:
                continue
            
            # Check include patterns
            if not include_patterns:
                # No include patterns means include all (except excluded)
                filtered_urls.append(url)
            else:
                for pattern in include_patterns:
                    if re.search(pattern, url):
                        filtered_urls.append(url)
                        break
        
        return filtered_urls


# Global instance
crawling_engine = CrawlingEngine()


def get_crawling_engine() -> CrawlingEngine:
    """Get the global crawling engine instance"""
    return crawling_engine 