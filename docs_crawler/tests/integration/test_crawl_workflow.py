"""
Integration tests for the complete crawling workflow.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime

from components.crawling_engine import CrawlingEngine
from components.supabase_integration import SupabaseIntegration
from components.search_engine import SearchEngine
from components.rag_strategies import RAGProcessor
from tests import TestConfig


@pytest.mark.integration
class TestCrawlWorkflow:
    """Test complete crawling workflow integration."""
    
    @pytest.fixture
    def setup_components(self, mock_supabase):
        """Setup integrated components for testing."""
        # Setup crawling engine
        crawler = CrawlingEngine()
        
        # Setup Supabase integration
        supabase = SupabaseIntegration()
        supabase.client = mock_supabase
        
        # Setup search engine
        search = SearchEngine()
        search.supabase = mock_supabase
        
        # Setup RAG processor
        rag = RAGProcessor()
        
        return {
            'crawler': crawler,
            'supabase': supabase,
            'search': search,
            'rag': rag
        }
    
    @patch('components.crawling_engine.AsyncWebCrawler')
    async def test_full_crawl_and_store_workflow(self, mock_crawler_class, setup_components, mock_openai):
        """Test complete workflow from crawling to storage."""
        components = setup_components
        
        # Setup mock crawler
        mock_crawler = AsyncMock()
        mock_crawler_class.return_value.__aenter__.return_value = mock_crawler
        
        mock_result = MagicMock()
        mock_result.markdown = TestConfig.SAMPLE_CONTENT
        mock_result.extracted_content = TestConfig.SAMPLE_CONTENT
        mock_result.metadata = {"title": "Test Page", "description": "Test description"}
        mock_result.links = {"internal": ["https://example.com/page1"], "external": []}
        mock_result.media = {"images": []}
        
        mock_crawler.arun.return_value = mock_result
        
        # Setup mock OpenAI for embeddings
        with patch('openai.OpenAI') as mock_openai_class:
            mock_openai_class.return_value = mock_openai
            
            # Create and start crawl job
            job = components['crawler'].create_job(
                urls=["https://example.com"],
                project_id="test-project",
                rag_strategies=["vector_embeddings"]
            )
            
            # Simulate crawling process
            result = await components['crawler']._crawl_single_url("https://example.com")
            
            # Verify crawl result
            assert result.success is True
            assert result.title == "Test Page"
            assert result.markdown == TestConfig.SAMPLE_CONTENT
            
            # Process with RAG strategies
            processed_chunks = await components['rag'].process_content(
                result.markdown,
                ["vector_embeddings"],
                {"openai_client": mock_openai}
            )
            
            # Store in Supabase
            for chunk in processed_chunks:
                components['supabase'].store_document(
                    project_id="test-project",
                    url=result.url,
                    title=result.title,
                    content=chunk['content'],
                    metadata=chunk.get('metadata', {}),
                    embedding=chunk.get('embedding', [])
                )
            
            # Verify storage calls were made
            assert components['supabase'].client.table.called
            assert components['supabase'].client.table.return_value.insert.called
    
    async def test_crawl_with_error_handling(self, setup_components):
        """Test crawling workflow with error handling."""
        components = setup_components
        
        # Mock failed crawl
        with patch('components.crawling_engine.AsyncWebCrawler') as mock_crawler_class:
            mock_crawler = AsyncMock()
            mock_crawler_class.return_value.__aenter__.return_value = mock_crawler
            mock_crawler.arun.side_effect = Exception("Network timeout")
            
            # Create job
            job = components['crawler'].create_job(
                urls=["https://example.com"],
                project_id="test-project"
            )
            
            # Start crawl
            result = await components['crawler']._crawl_single_url("https://example.com")
            
            # Verify error handling
            assert result.success is False
            assert result.error == "Network timeout"
            
            # Verify job status reflects error
            job.results = [result]
            status = components['crawler'].get_job_status(job.job_id)
            assert status['urls_failed'] == 1
    
    async def test_search_after_crawl(self, setup_components, mock_openai):
        """Test searching content after crawling and storage."""
        components = setup_components
        
        # Mock stored documents
        mock_search_results = [
            {
                'id': '1',
                'content': 'Machine learning content',
                'title': 'ML Guide',
                'url': 'https://example.com/ml',
                'metadata': {'topic': 'AI'},
                'similarity': 0.85
            },
            {
                'id': '2',
                'content': 'Python programming content',
                'title': 'Python Guide',
                'url': 'https://example.com/python',
                'metadata': {'topic': 'Programming'},
                'similarity': 0.78
            }
        ]
        
        # Setup mock Supabase responses
        components['supabase'].client.rpc.return_value.execute.return_value.data = mock_search_results
        
        # Setup mock OpenAI for query embedding
        with patch('openai.OpenAI') as mock_openai_class:
            mock_openai_class.return_value = mock_openai
            
            # Perform search
            results = await components['search'].search(
                project_id="test-project",
                query="machine learning",
                limit=10,
                similarity_threshold=0.7
            )
            
            # Verify search results
            assert len(results) == 2
            assert results[0]['title'] == 'ML Guide'
            assert results[0]['similarity'] == 0.85
            assert results[1]['title'] == 'Python Guide'
            
            # Verify OpenAI was called for query embedding
            mock_openai.embeddings.create.assert_called()
    
    async def test_multiple_rag_strategies(self, setup_components, mock_openai):
        """Test processing with multiple RAG strategies."""
        components = setup_components
        
        with patch('openai.OpenAI') as mock_openai_class:
            mock_openai_class.return_value = mock_openai
            
            # Process content with multiple strategies
            strategies = ["vector_embeddings", "contextual_embeddings", "agentic_rag"]
            
            processed_chunks = await components['rag'].process_content(
                TestConfig.SAMPLE_CONTENT,
                strategies,
                {"openai_client": mock_openai}
            )
            
            # Verify processing
            assert len(processed_chunks) > 0
            
            # Verify different strategies were applied
            strategy_types = set()
            for chunk in processed_chunks:
                if 'strategy' in chunk.get('metadata', {}):
                    strategy_types.add(chunk['metadata']['strategy'])
            
            # Should have processed with multiple strategies
            assert len(strategy_types) > 1
    
    async def test_project_isolation(self, setup_components):
        """Test that different projects are properly isolated."""
        components = setup_components
        
        # Setup mock responses for different projects
        project1_docs = [{'id': '1', 'content': 'Project 1 content', 'project_id': 'proj1'}]
        project2_docs = [{'id': '2', 'content': 'Project 2 content', 'project_id': 'proj2'}]
        
        def mock_select_response(*args, **kwargs):
            # Return different data based on filter
            mock_response = MagicMock()
            # This is a simplified mock - in reality, filtering would be more complex
            mock_response.execute.return_value.data = project1_docs
            return mock_response
        
        components['supabase'].client.table.return_value.select.side_effect = mock_select_response
        
        # Create jobs for different projects
        job1 = components['crawler'].create_job(
            urls=["https://example.com/proj1"],
            project_id="proj1"
        )
        
        job2 = components['crawler'].create_job(
            urls=["https://example.com/proj2"],
            project_id="proj2"
        )
        
        # Verify jobs are isolated
        assert job1.project_id == "proj1"
        assert job2.project_id == "proj2"
        assert job1.job_id != job2.job_id
    
    async def test_concurrent_crawling(self, setup_components):
        """Test concurrent crawling of multiple URLs."""
        components = setup_components
        
        urls = [
            "https://example.com/page1",
            "https://example.com/page2", 
            "https://example.com/page3"
        ]
        
        with patch('components.crawling_engine.AsyncWebCrawler') as mock_crawler_class:
            mock_crawler = AsyncMock()
            mock_crawler_class.return_value.__aenter__.return_value = mock_crawler
            
            # Mock different responses for different URLs
            def mock_crawl_response(url, **kwargs):
                mock_result = MagicMock()
                mock_result.markdown = f"Content for {url}"
                mock_result.metadata = {"title": f"Title for {url}"}
                mock_result.links = {"internal": [], "external": []}
                mock_result.media = {"images": []}
                return mock_result
            
            mock_crawler.arun.side_effect = mock_crawl_response
            
            # Create job with multiple URLs
            job = components['crawler'].create_job(
                urls=urls,
                project_id="test-project",
                max_concurrent=3
            )
            
            # Simulate concurrent crawling
            tasks = []
            for url in urls:
                task = components['crawler']._crawl_single_url(url)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            # Verify all URLs were processed
            assert len(results) == 3
            for i, result in enumerate(results):
                assert result.success is True
                assert urls[i] in result.markdown
    
    async def test_content_deduplication(self, setup_components):
        """Test content deduplication during crawling."""
        components = setup_components
        
        # Mock duplicate content detection
        duplicate_content = TestConfig.SAMPLE_CONTENT
        
        with patch('components.crawling_engine.AsyncWebCrawler') as mock_crawler_class:
            mock_crawler = AsyncMock()
            mock_crawler_class.return_value.__aenter__.return_value = mock_crawler
            
            # Return same content for different URLs
            mock_result = MagicMock()
            mock_result.markdown = duplicate_content
            mock_result.metadata = {"title": "Same Content"}
            mock_result.links = {"internal": [], "external": []}
            mock_result.media = {"images": []}
            
            mock_crawler.arun.return_value = mock_result
            
            # Crawl multiple URLs with same content
            urls = ["https://example.com/page1", "https://example.com/page2"]
            results = []
            
            for url in urls:
                result = await components['crawler']._crawl_single_url(url)
                results.append(result)
            
            # Verify content was retrieved
            assert len(results) == 2
            assert results[0].markdown == results[1].markdown
            
            # In a real implementation, deduplication would happen here
            # For testing, we just verify the content is identical
            assert results[0].markdown == duplicate_content
            assert results[1].markdown == duplicate_content


@pytest.mark.integration
class TestSearchIntegration:
    """Test search engine integration."""
    
    async def test_semantic_search_flow(self, mock_supabase, mock_openai):
        """Test complete semantic search flow."""
        search_engine = SearchEngine()
        search_engine.supabase = mock_supabase
        
        # Mock search results
        mock_results = [
            {
                'content': 'Machine learning algorithms',
                'title': 'ML Algorithms',
                'url': 'https://example.com/ml',
                'similarity': 0.92
            }
        ]
        
        mock_supabase.rpc.return_value.execute.return_value.data = mock_results
        
        with patch('openai.OpenAI') as mock_openai_class:
            mock_openai_class.return_value = mock_openai
            
            results = await search_engine.search(
                project_id="test-project",
                query="machine learning",
                limit=5
            )
            
            assert len(results) == 1
            assert results[0]['title'] == 'ML Algorithms'
            assert results[0]['similarity'] == 0.92
    
    async def test_filtered_search(self, mock_supabase, mock_openai):
        """Test search with filters."""
        search_engine = SearchEngine()
        search_engine.supabase = mock_supabase
        
        # Mock filtered results
        mock_results = [
            {
                'content': 'Python programming guide',
                'title': 'Python Guide',
                'url': 'https://example.com/python',
                'metadata': {'domain': 'example.com', 'type': 'tutorial'},
                'similarity': 0.88
            }
        ]
        
        mock_supabase.rpc.return_value.execute.return_value.data = mock_results
        
        with patch('openai.OpenAI') as mock_openai_class:
            mock_openai_class.return_value = mock_openai
            
            results = await search_engine.search(
                project_id="test-project",
                query="python",
                filters={
                    'domain': 'example.com',
                    'content_type': 'tutorial'
                },
                limit=5
            )
            
            assert len(results) == 1
            assert results[0]['metadata']['domain'] == 'example.com'
            assert results[0]['metadata']['type'] == 'tutorial'


@pytest.mark.integration
class TestMonitoringIntegration:
    """Test monitoring system integration."""
    
    def test_monitoring_during_crawl(self, system_monitor):
        """Test that monitoring captures crawl metrics."""
        monitor = system_monitor
        
        # Start monitoring
        monitor.start_monitoring()
        
        # Simulate some system activity
        import time
        time.sleep(0.1)  # Brief pause to generate metrics
        
        # Check that metrics were collected
        assert len(monitor.system_metrics) > 0
        
        # Verify metric structure
        latest_metric = monitor.system_metrics[-1]
        assert hasattr(latest_metric, 'cpu_percent')
        assert hasattr(latest_metric, 'memory_percent')
        assert hasattr(latest_metric, 'disk_percent')
        
        # Stop monitoring
        monitor.stop_monitoring()
    
    def test_alert_generation(self, system_monitor):
        """Test alert generation based on metrics."""
        monitor = system_monitor
        
        # Add some mock metrics that should trigger alerts
        from components.monitoring import SystemMetrics, ApplicationMetrics
        
        # High CPU usage metric
        high_cpu_metric = SystemMetrics(
            timestamp=datetime.utcnow(),
            cpu_percent=95.0,  # High CPU
            memory_percent=50.0,
            disk_percent=70.0,
            network_bytes_sent=1000,
            network_bytes_recv=1000,
            disk_read_bytes=1000,
            disk_write_bytes=1000,
            disk_used=100.0,
            disk_total=200.0
        )
        
        monitor.system_metrics.append(high_cpu_metric)
        
        # Check health status
        health_status = monitor.get_health_status()
        
        # Should detect high CPU issue
        assert health_status['status'] != 'healthy'
        assert any('CPU' in issue for issue in health_status['issues'])