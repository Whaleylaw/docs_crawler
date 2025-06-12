"""
Unit tests for the crawling engine component.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime

from components.crawling_engine import (
    CrawlJob, CrawlResult, CrawlingEngine, URLValidationError
)
from tests import TestConfig


@pytest.mark.unit
class TestCrawlJob:
    """Test CrawlJob dataclass."""
    
    def test_creation(self):
        """Test CrawlJob creation."""
        job = CrawlJob(
            job_id="test-job-1",
            urls=["https://example.com"],
            project_id="proj-1"
        )
        
        assert job.job_id == "test-job-1"
        assert job.urls == ["https://example.com"]
        assert job.project_id == "proj-1"
        assert job.status == "pending"
        assert job.max_depth == 2
        assert job.max_concurrent == 10
    
    def test_default_values(self):
        """Test default values are set correctly."""
        job = CrawlJob(
            job_id="test",
            urls=["https://example.com"],
            project_id="proj"
        )
        
        assert job.max_depth == 2
        assert job.max_concurrent == 10
        assert job.include_patterns == []
        assert job.exclude_patterns == []
        assert job.rag_strategies == ["vector_embeddings"]
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        job = CrawlJob(
            job_id="test",
            urls=["https://example.com"],
            project_id="proj",
            max_depth=3,
            rag_strategies=["vector_embeddings", "contextual_embeddings"]
        )
        
        job_dict = job.to_dict()
        assert isinstance(job_dict, dict)
        assert job_dict["job_id"] == "test"
        assert job_dict["max_depth"] == 3
        assert len(job_dict["rag_strategies"]) == 2
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        job_data = {
            "job_id": "test",
            "urls": ["https://example.com"],
            "project_id": "proj",
            "max_depth": 5,
            "rag_strategies": ["agentic_rag"]
        }
        
        job = CrawlJob.from_dict(job_data)
        assert job.job_id == "test"
        assert job.max_depth == 5
        assert job.rag_strategies == ["agentic_rag"]


@pytest.mark.unit
class TestCrawlResult:
    """Test CrawlResult dataclass."""
    
    def test_creation(self):
        """Test CrawlResult creation."""
        result = CrawlResult(
            url="https://example.com",
            title="Example",
            content="Content",
            markdown="# Example"
        )
        
        assert result.url == "https://example.com"
        assert result.title == "Example"
        assert result.success is True
        assert result.error is None
    
    def test_error_result(self):
        """Test creating error result."""
        result = CrawlResult(
            url="https://example.com",
            title="",
            content="",
            markdown="",
            success=False,
            error="Connection timeout"
        )
        
        assert result.success is False
        assert result.error == "Connection timeout"
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = CrawlResult(
            url="https://example.com",
            title="Test",
            content="Content",
            markdown="# Test",
            metadata={"author": "Test Author"}
        )
        
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["url"] == "https://example.com"
        assert result_dict["metadata"]["author"] == "Test Author"


@pytest.mark.unit
class TestCrawlingEngine:
    """Test CrawlingEngine class."""
    
    def test_creation(self):
        """Test engine creation."""
        engine = CrawlingEngine()
        assert engine.active_jobs == {}
        assert engine.job_history == []
    
    def test_validate_url_valid(self):
        """Test URL validation with valid URLs."""
        engine = CrawlingEngine()
        
        valid_urls = [
            "https://example.com",
            "https://www.google.com/search?q=test",
            "http://localhost:8000/api"
        ]
        
        for url in valid_urls:
            # Should not raise
            engine._validate_url(url)
    
    def test_validate_url_invalid(self):
        """Test URL validation with invalid URLs."""
        engine = CrawlingEngine()
        
        invalid_urls = [
            "not-a-url",
            "ftp://example.com",
            "javascript:alert('xss')",
            ""
        ]
        
        for url in invalid_urls:
            with pytest.raises(URLValidationError):
                engine._validate_url(url)
    
    def test_extract_domain(self):
        """Test domain extraction from URLs."""
        engine = CrawlingEngine()
        
        test_cases = [
            ("https://example.com/path", "example.com"),
            ("https://www.google.com/search", "www.google.com"),
            ("http://localhost:8000", "localhost"),
            ("https://api.github.com/repos", "api.github.com")
        ]
        
        for url, expected_domain in test_cases:
            domain = engine._extract_domain(url)
            assert domain == expected_domain
    
    def test_should_crawl_url(self):
        """Test URL filtering logic."""
        engine = CrawlingEngine()
        
        # Test include patterns
        include_patterns = ["*.example.com", "/api/*"]
        assert engine._should_crawl_url("https://test.example.com", include_patterns, [])
        assert engine._should_crawl_url("https://site.com/api/test", include_patterns, [])
        assert not engine._should_crawl_url("https://other.com", include_patterns, [])
        
        # Test exclude patterns
        exclude_patterns = ["*/admin/*", "*.pdf"]
        assert not engine._should_crawl_url("https://example.com/admin/panel", [], exclude_patterns)
        assert not engine._should_crawl_url("https://example.com/doc.pdf", [], exclude_patterns)
        assert engine._should_crawl_url("https://example.com/page", [], exclude_patterns)
    
    @patch('components.crawling_engine.AsyncWebCrawler')
    async def test_crawl_single_url(self, mock_crawler_class):
        """Test crawling a single URL."""
        # Setup mock
        mock_crawler = AsyncMock()
        mock_crawler_class.return_value.__aenter__.return_value = mock_crawler
        
        mock_result = MagicMock()
        mock_result.markdown = TestConfig.SAMPLE_CONTENT
        mock_result.extracted_content = "Extracted content"
        mock_result.metadata = {"title": "Test Page"}
        mock_result.links = {"internal": ["https://example.com/page1"], "external": []}
        mock_result.media = {"images": []}
        
        mock_crawler.arun.return_value = mock_result
        
        # Test crawling
        engine = CrawlingEngine()
        result = await engine._crawl_single_url("https://example.com")
        
        assert result.success is True
        assert result.url == "https://example.com"
        assert result.markdown == TestConfig.SAMPLE_CONTENT
        assert result.title == "Test Page"
    
    @patch('components.crawling_engine.AsyncWebCrawler')
    async def test_crawl_single_url_error(self, mock_crawler_class):
        """Test handling crawling errors."""
        # Setup mock to raise exception
        mock_crawler = AsyncMock()
        mock_crawler_class.return_value.__aenter__.return_value = mock_crawler
        mock_crawler.arun.side_effect = Exception("Network error")
        
        # Test error handling
        engine = CrawlingEngine()
        result = await engine._crawl_single_url("https://example.com")
        
        assert result.success is False
        assert result.error == "Network error"
        assert result.url == "https://example.com"
    
    def test_create_job(self):
        """Test job creation."""
        engine = CrawlingEngine()
        
        job = engine.create_job(
            urls=["https://example.com"],
            project_id="test-project",
            max_depth=3,
            rag_strategies=["vector_embeddings"]
        )
        
        assert job.project_id == "test-project"
        assert job.max_depth == 3
        assert job.urls == ["https://example.com"]
        assert job.status == "pending"
        assert job.job_id in engine.active_jobs
    
    def test_get_job(self):
        """Test job retrieval."""
        engine = CrawlingEngine()
        
        job = engine.create_job(
            urls=["https://example.com"],
            project_id="test"
        )
        
        retrieved_job = engine.get_job(job.job_id)
        assert retrieved_job == job
        
        # Non-existent job should return None
        assert engine.get_job("non-existent") is None
    
    @patch('components.crawling_engine.asyncio.create_task')
    def test_start_crawl(self, mock_create_task):
        """Test starting a crawl job."""
        engine = CrawlingEngine()
        
        job = engine.create_job(
            urls=["https://example.com"],
            project_id="test"
        )
        
        # Mock task creation
        mock_task = MagicMock()
        mock_create_task.return_value = mock_task
        
        engine.start_crawl(job.job_id)
        
        # Verify task was created and job status updated
        mock_create_task.assert_called_once()
        assert job.status == "running"
        assert job.started_at is not None
    
    def test_stop_crawl(self):
        """Test stopping a crawl job."""
        engine = CrawlingEngine()
        
        job = engine.create_job(
            urls=["https://example.com"],
            project_id="test"
        )
        
        # Mock running task
        mock_task = MagicMock()
        job.status = "running"
        engine.active_jobs[job.job_id] = job
        engine.crawl_tasks[job.job_id] = mock_task
        
        engine.stop_crawl(job.job_id)
        
        # Verify task was cancelled and job status updated
        mock_task.cancel.assert_called_once()
        assert job.status == "cancelled"
    
    def test_get_job_status(self):
        """Test job status retrieval."""
        engine = CrawlingEngine()
        
        job = engine.create_job(
            urls=["https://example.com"],
            project_id="test"
        )
        
        status = engine.get_job_status(job.job_id)
        assert status["status"] == "pending"
        assert status["urls_total"] == 1
        assert status["urls_completed"] == 0
        assert status["urls_failed"] == 0
    
    def test_get_job_results(self):
        """Test job results retrieval."""
        engine = CrawlingEngine()
        
        job = engine.create_job(
            urls=["https://example.com"],
            project_id="test"
        )
        
        # Add some mock results
        result1 = CrawlResult(
            url="https://example.com/page1",
            title="Page 1",
            content="Content 1",
            markdown="# Page 1"
        )
        result2 = CrawlResult(
            url="https://example.com/page2",
            title="Page 2",
            content="Content 2",
            markdown="# Page 2"
        )
        
        job.results = [result1, result2]
        
        results = engine.get_job_results(job.job_id)
        assert len(results) == 2
        assert results[0].title == "Page 1"
        assert results[1].title == "Page 2"
    
    def test_extract_links(self):
        """Test link extraction from content."""
        engine = CrawlingEngine()
        
        markdown_content = """
        # Test Page
        
        Here are some links:
        - [Internal Link](https://example.com/page1)
        - [External Link](https://other.com/page)
        - [Relative Link](/relative)
        
        <a href="https://example.com/page2">Another Link</a>
        """
        
        base_url = "https://example.com"
        links = engine._extract_links(markdown_content, base_url)
        
        assert "https://example.com/page1" in links
        assert "https://other.com/page" in links
        assert "https://example.com/relative" in links
        assert "https://example.com/page2" in links
    
    def test_chunk_content(self):
        """Test content chunking."""
        engine = CrawlingEngine()
        
        long_content = "This is a test sentence. " * 100  # ~2500 characters
        chunks = engine._chunk_content(long_content, chunk_size=1000, chunk_overlap=200)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 1200 for chunk in chunks)  # Allow for overlap
        
        # Test that chunks have overlap
        if len(chunks) > 1:
            # Find common text between chunks
            overlap_found = False
            for i in range(len(chunks) - 1):
                current_chunk = chunks[i]
                next_chunk = chunks[i + 1]
                # Check if there's any overlap
                for j in range(50, min(200, len(current_chunk))):
                    if current_chunk[-j:] in next_chunk:
                        overlap_found = True
                        break
            assert overlap_found, "No overlap found between chunks"
    
    def test_get_active_jobs(self):
        """Test getting all active jobs."""
        engine = CrawlingEngine()
        
        job1 = engine.create_job(["https://example.com"], "proj1")
        job2 = engine.create_job(["https://test.com"], "proj2")
        
        active_jobs = engine.get_active_jobs()
        assert len(active_jobs) == 2
        assert job1.job_id in [job.job_id for job in active_jobs]
        assert job2.job_id in [job.job_id for job in active_jobs]
    
    def test_get_job_history(self):
        """Test getting job history."""
        engine = CrawlingEngine()
        
        # Create and complete a job
        job = engine.create_job(["https://example.com"], "proj")
        job.status = "completed"
        job.completed_at = datetime.utcnow()
        engine.job_history.append(job)
        
        history = engine.get_job_history()
        assert len(history) == 1
        assert history[0].status == "completed"
    
    @pytest.mark.asyncio
    async def test_process_rag_strategies(self):
        """Test RAG strategy processing."""
        engine = CrawlingEngine()
        
        # Mock the RAG processor
        with patch('components.crawling_engine.RAGProcessor') as mock_rag:
            mock_processor = MagicMock()
            mock_rag.return_value = mock_processor
            mock_processor.process_chunks.return_value = [
                {"chunk": "processed chunk 1", "metadata": {}},
                {"chunk": "processed chunk 2", "metadata": {}}
            ]
            
            result = CrawlResult(
                url="https://example.com",
                title="Test",
                content=TestConfig.SAMPLE_CONTENT,
                markdown=TestConfig.SAMPLE_CONTENT
            )
            
            strategies = ["vector_embeddings", "contextual_embeddings"]
            processed = await engine._process_rag_strategies(result, strategies)
            
            assert len(processed) == 2
            assert processed[0]["chunk"] == "processed chunk 1"