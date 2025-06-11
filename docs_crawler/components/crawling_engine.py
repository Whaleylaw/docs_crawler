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
    from components.rag_strategies import get_rag_processor, RAGConfiguration, RAGStrategy
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
    rag_config: Optional[Dict[str, Any]] = None  # RAG configuration
    
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
    processed_content: Optional[str] = None  # RAG-processed content
    code_examples: List[Dict[str, Any]] = None  # Extracted code examples
    strategies_applied: List[str] = None  # Applied RAG strategies


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
            rag_strategies=config.get('rag_strategies', ['vector_embeddings']),
            rag_config=config.get('rag_config', {})  # Store RAG configuration
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
            
            # Initialize RAG processor with job configuration
            rag_config = self._create_rag_configuration(job)
            rag_processor = get_rag_processor(rag_config)
            
            # Process URLs based on type detection
            results = await self._process_urls_smart(job, rag_processor)
            
            # Store results in Supabase
            await self._store_results_in_supabase(job.project_id, results, rag_processor)
            
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
    
    def _create_rag_configuration(self, job: CrawlJob) -> RAGConfiguration:
        """Create RAG configuration from job settings"""
        
        # Convert string strategy names to RAGStrategy enums
        enabled_strategies = []
        for strategy_name in job.rag_strategies:
            try:
                strategy = RAGStrategy(strategy_name)
                enabled_strategies.append(strategy)
            except ValueError:
                continue  # Skip unknown strategies
        
        # Get configuration from job's rag_config
        rag_job_config = job.rag_config or {}
        
        return RAGConfiguration(
            enabled_strategies=enabled_strategies,
            embedding_model=rag_job_config.get('embedding_model', 'text-embedding-3-small'),
            use_contextual_embeddings=RAGStrategy.CONTEXTUAL_EMBEDDINGS in enabled_strategies,
            context_model=rag_job_config.get('context_model', 'gpt-3.5-turbo'),
            max_context_tokens=rag_job_config.get('max_context_tokens', 200),
            extract_code_examples=RAGStrategy.AGENTIC_RAG in enabled_strategies,
            min_code_length=rag_job_config.get('min_code_length', 1000),
            parallel_workers=rag_job_config.get('parallel_workers', 10),
            use_reranking=RAGStrategy.CROSS_ENCODER_RERANKING in enabled_strategies,
            use_hybrid_search=RAGStrategy.HYBRID_SEARCH in enabled_strategies,
            preview_mode=rag_job_config.get('preview_mode', False),
            preview_sample_size=rag_job_config.get('preview_sample_size', 3),
            batch_size=rag_job_config.get('batch_size', 20),
            max_retries=rag_job_config.get('max_retries', 3)
        )
    
    async def _process_urls_smart(self, job: CrawlJob, rag_processor) -> List[CrawlResult]:
        """Intelligently process URLs based on their type"""
        
        results = []
        
        for url in job.urls:
            url_type = URLProcessor.detect_url_type(url)
            
            if url_type == "sitemap":
                # Extract URLs from sitemap and crawl them
                sitemap_urls = self._parse_sitemap(url)
                batch_results = await self._crawl_batch(sitemap_urls, job, rag_processor)
                results.extend(batch_results)
                
            elif url_type == "text_file":
                # Crawl text file directly
                result = await self._crawl_text_file(url, job, rag_processor)
                if result:
                    results.append(result)
                    
            else:
                # Regular webpage - crawl recursively if depth > 1
                if job.max_depth > 1:
                    recursive_results = await self._crawl_recursive_internal_links([url], job, rag_processor)
                    results.extend(recursive_results)
                else:
                    result = await self._crawl_single_url(url, job, rag_processor)
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
    
    async def _crawl_text_file(self, url: str, job: CrawlJob, rag_processor) -> Optional[CrawlResult]:
        """Crawl a text file URL"""
        
        try:
            run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
            result = await self.crawler.arun(url=url, config=run_config)
            
            if result.success and result.markdown:
                chunks = self._smart_chunk_markdown(result.markdown, job.chunk_size)
                
                # Process content with RAG strategies
                processed_chunks = []
                for chunk in chunks:
                    processed = rag_processor.process_content(
                        content=chunk['text'],
                        url=url,
                        full_document=result.markdown
                    )
                    processed_chunks.append(processed)
                
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
                    crawled_at=datetime.now(),
                    processed_content=result.markdown,
                    code_examples=[ex for chunk in processed_chunks for ex in chunk.get('code_examples', [])],
                    strategies_applied=list(set(s for chunk in processed_chunks for s in chunk.get('strategies_applied', [])))
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
    
    async def _crawl_batch(self, urls: List[str], job: CrawlJob, rag_processor) -> List[CrawlResult]:
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
                return await self._crawl_single_url(url, job, rag_processor)
        
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
    
    async def _crawl_recursive_internal_links(self, start_urls: List[str], job: CrawlJob, rag_processor) -> List[CrawlResult]:
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
            batch_results = await self._crawl_batch(current_batch, job, rag_processor)
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
    
    async def _crawl_single_url(self, url: str, job: CrawlJob, rag_processor) -> Optional[CrawlResult]:
        """Crawl a single URL and extract content"""
        
        try:
            run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
            result = await self.crawler.arun(url=url, config=run_config)
            
            if result.success and result.markdown:
                chunks = self._smart_chunk_markdown(result.markdown, job.chunk_size)
                
                # Process content with RAG strategies
                processed_chunks = []
                all_code_examples = []
                all_strategies = set()
                
                for chunk in chunks:
                    processed = rag_processor.process_content(
                        content=chunk['text'],
                        url=url,
                        full_document=result.markdown
                    )
                    processed_chunks.append(processed)
                    all_code_examples.extend(processed.get('code_examples', []))
                    all_strategies.update(processed.get('strategies_applied', []))
                
                return CrawlResult(
                    url=url,
                    title=getattr(result, 'title', f"Page from {url}"),
                    content=result.markdown,
                    chunks=chunks,
                    metadata={
                        "content_type": "text/html",
                        "word_count": len(result.markdown.split()),
                        "char_count": len(result.markdown),
                        "links": getattr(result, 'links', {}),
                        "rag_strategies": list(all_strategies)
                    },
                    status="success",
                    error_message=None,
                    crawled_at=datetime.now(),
                    processed_content=result.markdown,
                    code_examples=all_code_examples,
                    strategies_applied=list(all_strategies)
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
    
    async def _store_results_in_supabase(self, project_id: str, results: List[CrawlResult], rag_processor):
        """Store crawl results in Supabase with RAG processing"""
        
        try:
            supabase_integration = get_supabase_integration()
            
            # Prepare data for batch insertion
            urls = []
            chunk_numbers = []
            contents = []
            metadatas = []
            
            # Prepare code examples data
            code_urls = []
            code_chunk_numbers = []
            code_examples = []
            code_summaries = []
            code_metadatas = []
            
            for result in results:
                if result.status == "success":
                    # Process regular content chunks
                    for chunk in result.chunks:
                        urls.append(result.url)
                        chunk_numbers.append(chunk['chunk_index'])
                        
                        # Use processed content if available
                        if result.processed_content and result.strategies_applied:
                            # Re-process this specific chunk with RAG
                            processed = rag_processor.process_content(
                                content=chunk['text'],
                                url=result.url,
                                full_document=result.content
                            )
                            content_to_store = processed['processed_content']
                        else:
                            content_to_store = chunk['text']
                        
                        contents.append(content_to_store)
                        
                        metadata = {
                            "title": result.title,
                            "chunk_size": chunk['char_count'],
                            "word_count": chunk['word_count'],
                            "headers": chunk['headers'],
                            "crawled_at": result.crawled_at.isoformat(),
                            "rag_strategies": result.strategies_applied or [],
                            **result.metadata
                        }
                        metadatas.append(metadata)
                    
                    # Process code examples if available
                    if result.code_examples:
                        for i, code_example in enumerate(result.code_examples):
                            code_urls.append(result.url)
                            code_chunk_numbers.append(i)
                            code_examples.append(code_example['code'])
                            code_summaries.append(code_example['summary'])
                            
                            code_metadata = {
                                "language": code_example['language'],
                                "char_count": len(code_example['code']),
                                "word_count": len(code_example['code'].split()),
                                "crawled_at": result.crawled_at.isoformat(),
                                "url": result.url
                            }
                            code_metadatas.append(code_metadata)
            
            # Store regular content in Supabase
            if urls:
                success = supabase_integration.add_documents_to_supabase(
                    project_id=project_id,
                    urls=urls,
                    chunk_numbers=chunk_numbers,
                    contents=contents,
                    metadatas=metadatas
                )
                
                if not success:
                    raise Exception("Failed to store documents in Supabase")
            
            # Store code examples if any
            if code_urls:
                try:
                    client = supabase_integration.get_supabase_client(project_id)
                    
                    # Use the add_code_examples_to_supabase function from reference
                    from components.rag_strategies import RAGProcessor
                    
                    # Create embeddings for code examples
                    embeddings = []
                    for code, summary in zip(code_examples, code_summaries):
                        combined_text = f"{code}\n\nSummary: {summary}"
                        embedding = rag_processor.create_embedding(combined_text)
                        embeddings.append(embedding)
                    
                    # Store code examples in batches
                    batch_size = 20
                    for i in range(0, len(code_examples), batch_size):
                        batch_end = min(i + batch_size, len(code_examples))
                        
                        batch_data = []
                        for j in range(i, batch_end):
                            # Extract source_id from URL
                            parsed_url = urlparse(code_urls[j])
                            source_id = parsed_url.netloc or parsed_url.path
                            
                            batch_data.append({
                                'url': code_urls[j],
                                'chunk_number': code_chunk_numbers[j],
                                'code_example': code_examples[j],
                                'summary': code_summaries[j],
                                'metadata': code_metadatas[j],
                                'source_id': source_id,
                                'embedding': embeddings[j]
                            })
                        
                        # Insert batch
                        client.table('code_examples').insert(batch_data).execute()
                    
                except Exception as e:
                    print(f"Warning: Failed to store code examples: {e}")
                    # Continue without code examples
                    
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
        total_code_examples = sum(len(r.code_examples or []) for r in results)
        
        # Collect applied strategies
        all_strategies = set()
        for result in results:
            if result.strategies_applied:
                all_strategies.update(result.strategies_applied)
        
        return {
            "total_urls": total_urls,
            "successful": successful,
            "failed": failed,
            "total_chunks": total_chunks,
            "total_words": total_words,
            "total_code_examples": total_code_examples,
            "avg_chunks_per_url": total_chunks / total_urls if total_urls > 0 else 0,
            "failed_urls": [r.url for r in results if r.status == "failed"],
            "rag_strategies_applied": list(all_strategies)
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