"""
Pytest configuration and shared fixtures for Crawl4AI tests.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
from typing import Dict, Any, Generator

# Import test configuration
from tests import TEST_DATA_DIR, TestConfig

# Import components to test
from components.configuration import ApplicationConfig, ConfigurationManager
from components.supabase_integration import SupabaseIntegration
from components.crawling_engine import CrawlingEngine
from components.search_engine import SearchEngine
from components.monitoring import SystemMonitor
from components.api_integration import APIIntegrationManager


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp(prefix="crawl4ai_test_"))
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def test_config() -> ApplicationConfig:
    """Create a test application configuration."""
    return ApplicationConfig(
        openai={"api_key": "test-key", "model": "gpt-3.5-turbo"},
        supabase={"url": "http://localhost:8000", "service_key": "test-key"},
        crawling={"max_concurrent": 5, "timeout": 30},
        rag={"chunk_size": 1000, "chunk_overlap": 200},
        ui={"theme": "light", "items_per_page": 10},
        performance={"cache_ttl": 300, "max_memory_mb": 1024},
        logging={"level": "INFO", "format": "json"},
        security={"rate_limit": 100, "enable_cors": True}
    )


@pytest.fixture
def mock_supabase():
    """Create a mock Supabase client."""
    mock = MagicMock()
    mock.table.return_value.select.return_value.execute.return_value.data = []
    mock.table.return_value.insert.return_value.execute.return_value.data = [{"id": "test-id"}]
    mock.table.return_value.update.return_value.eq.return_value.execute.return_value.data = [{"id": "test-id"}]
    mock.table.return_value.delete.return_value.eq.return_value.execute.return_value.data = []
    return mock


@pytest.fixture
def mock_openai():
    """Create a mock OpenAI client."""
    mock = MagicMock()
    mock.embeddings.create.return_value.data = [
        MagicMock(embedding=[0.1, 0.2, 0.3] * 512)  # 1536-dimensional embedding
    ]
    mock.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="Test response"))
    ]
    return mock


@pytest.fixture
def supabase_integration(mock_supabase) -> SupabaseIntegration:
    """Create a SupabaseIntegration instance with mocked client."""
    integration = SupabaseIntegration()
    integration.client = mock_supabase
    return integration


@pytest.fixture
def crawling_engine() -> CrawlingEngine:
    """Create a CrawlingEngine instance for testing."""
    return CrawlingEngine()


@pytest.fixture
def search_engine(mock_supabase) -> SearchEngine:
    """Create a SearchEngine instance with mocked dependencies."""
    engine = SearchEngine()
    engine.supabase = mock_supabase
    return engine


@pytest.fixture
def system_monitor() -> SystemMonitor:
    """Create a SystemMonitor instance for testing."""
    return SystemMonitor()


@pytest.fixture
def api_manager() -> APIIntegrationManager:
    """Create an APIIntegrationManager instance for testing."""
    return APIIntegrationManager()


@pytest.fixture
def sample_crawl_data() -> Dict[str, Any]:
    """Sample crawl data for testing."""
    return {
        "url": "https://example.com",
        "title": "Example Website",
        "content": TestConfig.SAMPLE_CONTENT,
        "markdown": TestConfig.SAMPLE_CONTENT,
        "metadata": {
            "description": "Example website for testing",
            "keywords": ["test", "example"],
            "author": "Test Author"
        },
        "links": ["https://example.com/page1", "https://example.com/page2"],
        "images": ["https://example.com/image1.jpg"],
        "timestamp": "2025-01-01T00:00:00Z"
    }


@pytest.fixture
def sample_search_results() -> list:
    """Sample search results for testing."""
    return [
        {
            "id": "1",
            "content": "Machine learning is a subset of artificial intelligence",
            "url": "https://example.com/ml",
            "title": "Machine Learning Basics",
            "similarity": 0.85,
            "metadata": {"topic": "AI", "difficulty": "beginner"}
        },
        {
            "id": "2", 
            "content": "Python is a popular programming language for data science",
            "url": "https://example.com/python",
            "title": "Python Programming",
            "similarity": 0.78,
            "metadata": {"topic": "Programming", "difficulty": "intermediate"}
        }
    ]


@pytest.fixture
def mock_crawl4ai():
    """Create a mock Crawl4AI instance."""
    mock = AsyncMock()
    mock.arun.return_value = MagicMock(
        markdown=TestConfig.SAMPLE_CONTENT,
        extracted_content=TestConfig.SAMPLE_CONTENT,
        metadata={"title": "Test Page", "description": "Test description"},
        links={"internal": ["https://example.com/page1"], "external": []},
        media={"images": [], "videos": []}
    )
    return mock


@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    mock = MagicMock()
    mock.get.return_value = None
    mock.set.return_value = True
    mock.exists.return_value = False
    return mock


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "unit: marks tests as unit tests (fast, isolated)"
    )
    config.addinivalue_line(
        "markers", 
        "integration: marks tests as integration tests (slower, requires services)"
    )
    config.addinivalue_line(
        "markers",
        "api: marks tests as API tests (requires running API server)"
    )
    config.addinivalue_line(
        "markers",
        "e2e: marks tests as end-to-end tests (slowest, full system)"
    )


# Test utilities
class MockResponse:
    """Mock HTTP response for testing."""
    
    def __init__(self, json_data: dict, status_code: int = 200):
        self.json_data = json_data
        self.status_code = status_code
        self.text = str(json_data)
        
    def json(self):
        return self.json_data
        
    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


def create_mock_response(data: dict, status: int = 200) -> MockResponse:
    """Create a mock HTTP response."""
    return MockResponse(data, status)