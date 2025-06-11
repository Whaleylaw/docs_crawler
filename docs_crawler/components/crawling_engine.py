"""
Enhanced Crawling Engine Module
Handles web crawling operations, URL processing, and content extraction using Crawl4AI
"""

import asyncio
import aiohttp
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import streamlit as st
import re
from urllib.parse import urljoin, urlparse, urlencode
from crawl4ai import AsyncWebCrawler
from crawl4ai.extraction_strategy import LLMExtractionStrategy, JsonCssExtractionStrategy
from crawl4ai.chunking_strategy import RegexChunking, NlpSentenceChunking
import tiktoken


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
    include_patterns: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)
    rag_strategies: List[str] = field(default_factory=lambda: ['contextual'])
    
    # Progress tracking
    discovered_urls: Set[str] = field(default_factory=set)
    processed_urls: Set[str] = field(default_factory=set)
    failed_urls: Dict[str, str] = field(default_factory=dict)


@dataclass
class CrawlResult:
    """Result of crawling a single URL"""
    url: str
    title: str
    content: str
    clean_content: str
    markdown_content: str
    chunks: List[Dict[str, Any]]
    code_examples: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    status: str
    error_message: Optional[str]
    crawled_at: datetime


class SmartURLDetector:
    """Smart URL detection and processing"""
    
    @staticmethod
    async def detect_and_expand_urls(urls: List[str], max_depth: int = 2) -> Set[str]:
        """Detect URL types and expand them appropriately"""
        expanded_urls = set()
        
        for url in urls:
            url_type = SmartURLDetector.detect_url_type(url)
            
            if url_type == "sitemap":
                sitemap_urls = await SmartURLDetector.extract_sitemap_urls(url)
                expanded_urls.update(sitemap_urls)
            elif url_type == "text_file":
                text_urls = await SmartURLDetector.extract_text_file_urls(url)
                expanded_urls.update(text_urls)
            elif url_type == "webpage" and max_depth > 1:
                # For webpages, we might want to discover linked pages
                linked_urls = await SmartURLDetector.discover_linked_urls(url, max_depth)
                expanded_urls.update(linked_urls)
            else:
                expanded_urls.add(url)
        
        return expanded_urls
    
    @staticmethod
    def detect_url_type(url: str) -> str:
        """Detect the type of URL"""
        url_lower = url.lower()
        
        if 'sitemap' in url_lower and url_lower.endswith('.xml'):
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
    async def extract_sitemap_urls(sitemap_url: str) -> Set[str]:
        """Extract URLs from XML sitemap"""
        urls = set()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(sitemap_url) as response:
                    if response.status == 200:
                        content = await response.text()
                        
                        # Parse XML
                        root = ET.fromstring(content)
                        
                        # Handle different sitemap formats
                        for url_elem in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
                            loc_elem = url_elem.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                            if loc_elem is not None:
                                urls.add(loc_elem.text)
                        
                        # Handle sitemap index files
                        for sitemap_elem in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap'):
                            loc_elem = sitemap_elem.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                            if loc_elem is not None:
                                # Recursively process nested sitemaps
                                nested_urls = await SmartURLDetector.extract_sitemap_urls(loc_elem.text)
                                urls.update(nested_urls)
        
        except Exception as e:
            st.warning(f"Failed to parse sitemap {sitemap_url}: {str(e)}")
        
        return urls
    
    @staticmethod
    async def extract_text_file_urls(text_url: str) -> Set[str]:
        """Extract URLs from text file"""
        urls = set()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(text_url) as response:
                    if response.status == 200:
                        content = await response.text()
                        
                        # Extract URLs from text (one per line)
                        for line in content.split('\n'):
                            line = line.strip()
                            if line and URLProcessor.validate_url(line):
                                urls.add(line)
        
        except Exception as e:
            st.warning(f"Failed to parse text file {text_url}: {str(e)}")
        
        return urls
    
    @staticmethod
    async def discover_linked_urls(base_url: str, max_depth: int) -> Set[str]:
        """Discover linked URLs from a webpage (for future implementation)"""
        # For now, just return the base URL
        # In a full implementation, this would crawl the page and extract links
        return {base_url}


class ContentProcessor:
    """Process and chunk crawled content"""
    
    def __init__(self, chunk_size: int = 4000):
        self.chunk_size = chunk_size
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def process_content(self, result) -> tuple[str, str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Process crawled content and extract chunks and code examples"""
        
        # Extract clean text content
        clean_content = result.cleaned_html if hasattr(result, 'cleaned_html') else ""
        markdown_content = result.markdown if hasattr(result, 'markdown') else ""
        
        # Create content chunks
        chunks = self.create_content_chunks(clean_content)
        
        # Extract code examples
        code_examples = self.extract_code_examples(clean_content, markdown_content)
        
        return clean_content, markdown_content, chunks, code_examples
    
    def create_content_chunks(self, content: str) -> List[Dict[str, Any]]:
        """Split content into semantic chunks"""
        chunks = []
        
        if not content:
            return chunks
        
        # Use NLP sentence chunking for better semantic boundaries
        try:
            # Split content into sentences and group by token count
            sentences = re.split(r'[.!?]+', content)
            current_chunk = ""
            chunk_index = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # Check if adding this sentence would exceed chunk size
                potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
                token_count = len(self.tokenizer.encode(potential_chunk))
                
                if token_count > self.chunk_size and current_chunk:
                    # Save current chunk and start new one
                    chunks.append({
                        "chunk_index": chunk_index,
                        "text": current_chunk.strip(),
                        "token_count": len(self.tokenizer.encode(current_chunk)),
                        "start_char": content.find(current_chunk),
                        "metadata": {
                            "chunk_type": "semantic",
                            "sentence_count": len(re.split(r'[.!?]+', current_chunk))
                        }
                    })
                    
                    current_chunk = sentence
                    chunk_index += 1
                else:
                    current_chunk = potential_chunk
            
            # Add final chunk
            if current_chunk:
                chunks.append({
                    "chunk_index": chunk_index,
                    "text": current_chunk.strip(),
                    "token_count": len(self.tokenizer.encode(current_chunk)),
                    "start_char": content.find(current_chunk),
                    "metadata": {
                        "chunk_type": "semantic",
                        "sentence_count": len(re.split(r'[.!?]+', current_chunk))
                    }
                })
        
        except Exception as e:
            # Fallback to simple word-based chunking
            words = content.split()
            for i in range(0, len(words), self.chunk_size // 4):  # Rough estimate of 4 chars per word
                chunk_words = words[i:i + self.chunk_size // 4]
                chunk_text = " ".join(chunk_words)
                
                chunks.append({
                    "chunk_index": i // (self.chunk_size // 4),
                    "text": chunk_text,
                    "token_count": len(self.tokenizer.encode(chunk_text)),
                    "start_char": content.find(chunk_text),
                    "metadata": {
                        "chunk_type": "word_based",
                        "word_count": len(chunk_words)
                    }
                })
        
        return chunks
    
    def extract_code_examples(self, clean_content: str, markdown_content: str) -> List[Dict[str, Any]]:
        """Extract code examples from content"""
        code_examples = []
        
        # Extract from markdown code blocks
        if markdown_content:
            code_blocks = re.findall(r'```(\w+)?\n(.*?)\n```', markdown_content, re.DOTALL)
            for i, (language, code) in enumerate(code_blocks):
                code_examples.append({
                    "language": language or "unknown",
                    "code": code.strip(),
                    "source": "markdown_block",
                    "position": i,
                    "context": self._extract_code_context(markdown_content, code)
                })
        
        # Extract inline code
        inline_code = re.findall(r'`([^`]+)`', clean_content)
        for i, code in enumerate(inline_code):
            if len(code) > 10:  # Only meaningful code snippets
                code_examples.append({
                    "language": "inline",
                    "code": code,
                    "source": "inline_code",
                    "position": i,
                    "context": self._extract_code_context(clean_content, code)
                })
        
        return code_examples
    
    def _extract_code_context(self, content: str, code: str) -> str:
        """Extract surrounding context for code examples"""
        try:
            code_index = content.find(code)
            if code_index == -1:
                return ""
            
            # Extract 100 characters before and after
            start = max(0, code_index - 100)
            end = min(len(content), code_index + len(code) + 100)
            
            return content[start:end].strip()
        except:
            return ""


class CrawlingEngine:
    """Enhanced crawling engine using Crawl4AI"""
    
    def __init__(self):
        self.active_jobs = {}
        self.job_counter = 0
        self.url_processor = URLProcessor()
    
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
            # Discover URLs
            job.discovered_urls = await SmartURLDetector.detect_and_expand_urls(
                job.urls, job.max_depth
            )
            
            # Apply include/exclude patterns
            filtered_urls = URLProcessor.apply_include_exclude_patterns(
                list(job.discovered_urls), 
                job.include_patterns, 
                job.exclude_patterns
            )
            job.discovered_urls = set(filtered_urls)
            
            # Process URLs
            results = await self._process_urls_with_crawl4ai(job)
            
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
            st.error(f"Crawl job failed: {str(e)}")
            return False
    
    async def _process_urls_with_crawl4ai(self, job: CrawlJob) -> List[CrawlResult]:
        """Process URLs using Crawl4AI"""
        results = []
        content_processor = ContentProcessor(job.chunk_size)
        
        async with AsyncWebCrawler(
            verbose=True,
            headless=True,
            browser_type="chromium"
        ) as crawler:
            
            # Set up semaphore for concurrency control
            semaphore = asyncio.Semaphore(job.max_concurrent)
            total_urls = len(job.discovered_urls)
            
            async def crawl_single_url(url: str) -> Optional[CrawlResult]:
                async with semaphore:
                    if url in job.processed_urls:
                        return None
                    
                    try:
                        # Configure crawling strategy based on RAG strategy
                        extraction_strategy = None
                        if 'agentic' in job.rag_strategies:
                            extraction_strategy = LLMExtractionStrategy(
                                provider="openai",
                                api_token=st.secrets.get("OPENAI_API_KEY", ""),
                                schema={
                                    "name": "Documentation Extractor",
                                    "description": "Extract structured information from documentation",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "title": {"type": "string"},
                                            "main_content": {"type": "string"},
                                            "code_examples": {"type": "array"},
                                            "api_references": {"type": "array"}
                                        }
                                    }
                                }
                            )
                        
                        # Crawl the URL
                        result = await crawler.arun(
                            url=url,
                            extraction_strategy=extraction_strategy,
                            chunking_strategy=RegexChunking(),
                            bypass_cache=True,
                            include_raw_html=False,
                            remove_overlay_elements=True
                        )
                        
                        if result.success:
                            # Process content
                            clean_content, markdown_content, chunks, code_examples = content_processor.process_content(result)
                            
                            crawl_result = CrawlResult(
                                url=url,
                                title=result.metadata.get('title', ''),
                                content=result.cleaned_html or "",
                                clean_content=clean_content,
                                markdown_content=markdown_content,
                                chunks=chunks,
                                code_examples=code_examples,
                                metadata={
                                    "word_count": len(result.cleaned_html.split()) if result.cleaned_html else 0,
                                    "char_count": len(result.cleaned_html) if result.cleaned_html else 0,
                                    "links_found": len(result.links) if hasattr(result, 'links') else 0,
                                    "images_found": len(result.media) if hasattr(result, 'media') else 0,
                                    **result.metadata
                                },
                                status="success",
                                error_message=None,
                                crawled_at=datetime.now()
                            )
                            
                            job.processed_urls.add(url)
                            job.progress = len(job.processed_urls) / total_urls
                            
                            return crawl_result
                        
                        else:
                            job.failed_urls[url] = "Crawl4AI reported failure"
                            return CrawlResult(
                                url=url,
                                title="",
                                content="",
                                clean_content="",
                                markdown_content="",
                                chunks=[],
                                code_examples=[],
                                metadata={},
                                status="failed",
                                error_message="Crawl4AI reported failure",
                                crawled_at=datetime.now()
                            )
                    
                    except Exception as e:
                        job.failed_urls[url] = str(e)
                        return CrawlResult(
                            url=url,
                            title="",
                            content="",
                            clean_content="",
                            markdown_content="",
                            chunks=[],
                            code_examples=[],
                            metadata={},
                            status="failed",
                            error_message=str(e),
                            crawled_at=datetime.now()
                        )
            
            # Execute crawling tasks
            tasks = [crawl_single_url(url) for url in job.discovered_urls]
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out None results and exceptions
            results = [r for r in task_results if isinstance(r, CrawlResult)]
        
        return results
    
    def _create_results_summary(self, results: List[CrawlResult]) -> Dict[str, Any]:
        """Create summary statistics for crawl results"""
        
        if not results:
            return {
                "total_urls": 0,
                "successful": 0,
                "failed": 0,
                "total_chunks": 0,
                "total_words": 0,
                "total_code_examples": 0,
                "avg_chunks_per_url": 0
            }
        
        successful = len([r for r in results if r.status == "success"])
        failed = len(results) - successful
        
        total_chunks = sum(len(r.chunks) for r in results)
        total_words = sum(r.metadata.get('word_count', 0) for r in results)
        total_code_examples = sum(len(r.code_examples) for r in results)
        
        return {
            "total_urls": len(results),
            "successful": successful,
            "failed": failed,
            "total_chunks": total_chunks,
            "total_words": total_words,
            "total_code_examples": total_code_examples,
            "avg_chunks_per_url": total_chunks / len(results),
            "avg_words_per_url": total_words / len(results) if results else 0
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
    """Enhanced URL processing and validation"""
    
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
                try:
                    if re.search(pattern, url, re.IGNORECASE):
                        excluded = True
                        break
                except re.error:
                    # Invalid regex pattern, treat as literal string
                    if pattern.lower() in url.lower():
                        excluded = True
                        break
            
            if excluded:
                continue
            
            # Check include patterns
            if not include_patterns:
                # No include patterns means include all (except excluded)
                filtered_urls.append(url)
            else:
                included = False
                for pattern in include_patterns:
                    try:
                        if re.search(pattern, url, re.IGNORECASE):
                            included = True
                            break
                    except re.error:
                        # Invalid regex pattern, treat as literal string
                        if pattern.lower() in url.lower():
                            included = True
                            break
                
                if included:
                    filtered_urls.append(url)
        
        return filtered_urls


# Global instance
crawling_engine = CrawlingEngine()


def get_crawling_engine() -> CrawlingEngine:
    """Get the global crawling engine instance"""
    return crawling_engine 