"""
Crawling Engine Module
Handles web crawling operations, URL processing, and content extraction
"""

import asyncio
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import streamlit as st


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
            rag_strategies=config.get('rag_strategies', ['contextual'])
        )
        
        self.active_jobs[job_id] = job
        return job
    
    async def start_crawl_job(self, job_id: str) -> bool:
        """Start executing a crawl job"""
        
        if job_id not in self.active_jobs:
            return False
        
        job = self.active_jobs[job_id]
        job.status = CrawlStatus.RUNNING
        job.started_at = datetime.now()
        
        try:
            # Process URLs
            results = await self._process_urls(job)
            
            # Update job status
            job.status = CrawlStatus.COMPLETED
            job.completed_at = datetime.now()
            job.progress = 1.0
            job.results_summary = self._create_results_summary(results)
            
            return True
            
        except Exception as e:
            job.status = CrawlStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()
            return False
    
    async def _process_urls(self, job: CrawlJob) -> List[CrawlResult]:
        """Process all URLs in a crawl job"""
        
        results = []
        discovered_urls = set(job.urls)
        processed_urls = set()
        
        # Smart URL detection
        if any('sitemap' in url.lower() for url in job.urls):
            sitemap_urls = await self._extract_sitemap_urls(job.urls)
            discovered_urls.update(sitemap_urls)
        
        total_urls = len(discovered_urls)
        
        # Process URLs with concurrency control
        semaphore = asyncio.Semaphore(job.max_concurrent)
        
        async def process_single_url(url: str) -> Optional[CrawlResult]:
            async with semaphore:
                if url in processed_urls:
                    return None
                
                processed_urls.add(url)
                result = await self._crawl_single_url(url, job)
                
                # Update progress
                job.progress = len(processed_urls) / total_urls
                
                return result
        
        # Execute crawling tasks
        tasks = [process_single_url(url) for url in discovered_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out None results and exceptions
        valid_results = [r for r in results if isinstance(r, CrawlResult)]
        
        return valid_results
    
    async def _crawl_single_url(self, url: str, job: CrawlJob) -> CrawlResult:
        """Crawl a single URL and extract content"""
        
        try:
            # TODO: Implement actual crawling logic
            # - Fetch URL content
            # - Extract text and metadata
            # - Apply include/exclude patterns
            # - Process with selected RAG strategies
            
            # Placeholder implementation
            content = f"Mock content from {url}"
            chunks = self._create_chunks(content, job.chunk_size)
            
            return CrawlResult(
                url=url,
                title=f"Title for {url}",
                content=content,
                chunks=chunks,
                metadata={"content_type": "text/html", "word_count": len(content.split())},
                status="success",
                error_message=None,
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
    
    def _create_chunks(self, content: str, chunk_size: int) -> List[Dict[str, Any]]:
        """Split content into chunks for processing"""
        
        words = content.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            
            chunks.append({
                "text": chunk_text,
                "start_index": i,
                "end_index": min(i + chunk_size, len(words)),
                "word_count": len(chunk_words)
            })
        
        return chunks
    
    async def _extract_sitemap_urls(self, urls: List[str]) -> Set[str]:
        """Extract URLs from sitemap files"""
        
        sitemap_urls = set()
        
        for url in urls:
            if 'sitemap' in url.lower():
                # TODO: Parse actual sitemap XML
                # For now, return mock URLs
                mock_urls = [
                    f"{url.replace('sitemap.xml', '')}page1",
                    f"{url.replace('sitemap.xml', '')}page2",
                    f"{url.replace('sitemap.xml', '')}page3"
                ]
                sitemap_urls.update(mock_urls)
        
        return sitemap_urls
    
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
            "avg_chunks_per_url": total_chunks / total_urls if total_urls > 0 else 0
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
                return True
        return False
    
    def resume_job(self, job_id: str) -> bool:
        """Resume a paused crawl job"""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            if job.status == CrawlStatus.PAUSED:
                job.status = CrawlStatus.RUNNING
                return True
        return False
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a crawl job"""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            job.status = CrawlStatus.CANCELLED
            job.completed_at = datetime.now()
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
                return True
        
        return False


class URLProcessor:
    """Utility class for URL processing and validation"""
    
    @staticmethod
    def detect_url_type(url: str) -> str:
        """Detect the type of URL (sitemap, text file, etc.)"""
        
        url_lower = url.lower()
        
        if 'sitemap' in url_lower:
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
        
        import re
        
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
        
        import re
        
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