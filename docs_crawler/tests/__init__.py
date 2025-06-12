"""
Test suite for Crawl4AI Standalone Application.

This package contains comprehensive tests for all components:
- Unit tests for individual components
- Integration tests for component interactions
- API tests for REST endpoints
- End-to-end tests for complete workflows
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the parent directory to sys.path so we can import components
test_dir = Path(__file__).parent
project_root = test_dir.parent
sys.path.insert(0, str(project_root))

# Test configuration
TEST_DATABASE_URL = "sqlite:///test.db"
TEST_SUPABASE_URL = os.getenv("TEST_SUPABASE_URL", "http://localhost:8000")
TEST_SUPABASE_KEY = os.getenv("TEST_SUPABASE_KEY", "test-key")
TEST_OPENAI_API_KEY = os.getenv("TEST_OPENAI_API_KEY", "test-key")

# Create temporary directories for test files
TEST_DATA_DIR = Path(tempfile.gettempdir()) / "crawl4ai_tests"
TEST_DATA_DIR.mkdir(exist_ok=True)

# Test utilities
class TestConfig:
    """Test configuration constants."""
    
    SAMPLE_URLS = [
        "https://example.com",
        "https://httpbin.org/html",
        "https://jsonplaceholder.typicode.com"
    ]
    
    SAMPLE_CONTENT = """
    # Sample Content
    
    This is sample content for testing purposes.
    
    ## Features
    - Feature 1
    - Feature 2
    - Feature 3
    
    ```python
    def example_function():
        return "test"
    ```
    """
    
    SAMPLE_SEARCH_QUERIES = [
        "machine learning",
        "python programming",
        "data science",
        "web development"
    ]