# 🧪 Testing Guide

This document provides comprehensive information about testing the Crawl4AI Standalone Application.

## 📋 Overview

The testing suite consists of multiple layers of testing to ensure reliability and quality:

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions and workflows
- **API Tests**: Test REST API endpoints and webhook functionality
- **End-to-End Tests**: Test complete user workflows (planned for future)

## 🚀 Quick Start

### Prerequisites

```bash
# Install testing dependencies
pip install pytest pytest-asyncio pytest-cov pytest-mock httpx requests-mock
```

### Running Tests

```bash
# Run all tests
python test_runner.py all

# Run unit tests only (fastest)
python test_runner.py unit

# Run with verbose output
python test_runner.py all -v

# Generate coverage report
python test_runner.py coverage
```

## 📁 Test Structure

```
tests/
├── __init__.py                 # Test configuration
├── conftest.py                 # Shared fixtures and setup
├── pytest.ini                 # Pytest configuration
├── unit/                       # Unit tests
│   ├── __init__.py
│   ├── test_configuration.py   # Configuration management tests
│   ├── test_crawling_engine.py # Crawling engine tests
│   ├── test_supabase_integration.py
│   ├── test_search_engine.py
│   ├── test_monitoring.py
│   └── test_api_integration.py
├── integration/                # Integration tests
│   ├── __init__.py
│   ├── test_crawl_workflow.py  # Full crawling workflow
│   ├── test_search_integration.py
│   └── test_monitoring_integration.py
├── api/                        # API tests
│   ├── __init__.py
│   └── test_rest_api.py        # REST endpoint tests
└── coverage_html/              # Coverage reports (generated)
```

## 🏷️ Test Categories

### Unit Tests (`@pytest.mark.unit`)

Test individual functions and classes in isolation using mocks for external dependencies.

**Characteristics:**
- Fast execution (< 1 second per test)
- No external dependencies
- High test coverage
- Easy to debug

**Example:**
```python
@pytest.mark.unit
def test_configuration_validation():
    config = OpenAIConfig(api_key="sk-test", model="gpt-4")
    config.validate()  # Should not raise
```

### Integration Tests (`@pytest.mark.integration`)

Test component interactions and workflows with realistic mock services.

**Characteristics:**
- Medium execution time (1-10 seconds per test)
- Tests component integration
- May use real services in controlled environment
- Tests complete workflows

**Example:**
```python
@pytest.mark.integration
async def test_crawl_and_store_workflow(setup_components):
    # Test complete crawl → process → store workflow
    result = await crawler.crawl_url("https://example.com")
    processed = await rag.process(result.content)
    success = await storage.store(processed)
    assert success
```

### API Tests (`@pytest.mark.api`)

Test REST API endpoints using FastAPI TestClient.

**Characteristics:**
- Fast execution with TestClient
- Tests HTTP interfaces
- Validates request/response formats
- Tests authentication and authorization

**Example:**
```python
@pytest.mark.api
def test_create_project_endpoint(client, auth_headers):
    response = client.post("/projects", json=project_data, headers=auth_headers)
    assert response.status_code == 201
```

## 🛠️ Test Configuration

### Environment Variables

Tests use isolated environment variables:

```bash
# Required for some tests
TEST_SUPABASE_URL=http://localhost:8000
TEST_SUPABASE_KEY=test-key
TEST_OPENAI_API_KEY=test-key

# Optional
TEST_MODE=true
ENVIRONMENT=test
```

### Fixtures

Common fixtures are defined in `conftest.py`:

- `test_config`: Application configuration for testing
- `mock_supabase`: Mocked Supabase client
- `mock_openai`: Mocked OpenAI client
- `temp_dir`: Temporary directory for file operations
- `sample_crawl_data`: Sample data for testing

## 🎯 Running Specific Tests

### By Category

```bash
# Unit tests only
python test_runner.py unit

# Integration tests only
python test_runner.py integration

# API tests only
python test_runner.py api
```

### By File

```bash
# Specific test file
python test_runner.py specific --test-path tests/unit/test_configuration.py

# Specific test function
python test_runner.py specific --test-path tests/unit/test_configuration.py::TestOpenAIConfig::test_validation
```

### By Marker

```bash
# Tests that require internet
pytest -m "requires_internet"

# Tests that require OpenAI API
pytest -m "requires_openai"

# Exclude slow tests
pytest -m "not slow"
```

## 📊 Coverage Reports

### Generate Coverage

```bash
# HTML report (recommended)
python test_runner.py coverage

# Terminal report
pytest --cov=components --cov-report=term-missing

# XML report (for CI/CD)
pytest --cov=components --cov-report=xml:coverage.xml
```

### Coverage Targets

- **Overall Coverage**: > 85%
- **Critical Components**: > 90%
  - Configuration management
  - Crawling engine
  - Search engine
  - API integration
- **UI Components**: > 70%

### Viewing Reports

```bash
# Open HTML coverage report
open tests/coverage_html/index.html
```

## 🔧 Writing Tests

### Unit Test Template

```python
"""
Unit tests for [component_name].
"""

import pytest
from unittest.mock import MagicMock, patch

from components.[component_name] import [ClassName]


@pytest.mark.unit
class Test[ClassName]:
    """Test [ClassName] functionality."""
    
    def test_creation(self):
        """Test instance creation."""
        instance = [ClassName]()
        assert instance is not None
    
    def test_functionality(self):
        """Test specific functionality."""
        # Arrange
        instance = [ClassName]()
        
        # Act
        result = instance.some_method()
        
        # Assert
        assert result == expected_value
    
    @patch('components.[component_name].external_dependency')
    def test_with_mock(self, mock_dependency):
        """Test with mocked external dependency."""
        mock_dependency.return_value = "mocked_value"
        
        instance = [ClassName]()
        result = instance.method_using_dependency()
        
        assert result == "expected_result"
        mock_dependency.assert_called_once()
```

### Integration Test Template

```python
"""
Integration tests for [workflow_name].
"""

import pytest
from unittest.mock import patch

from components.component1 import Component1
from components.component2 import Component2


@pytest.mark.integration
class Test[WorkflowName]:
    """Test [workflow_name] integration."""
    
    @pytest.fixture
    def setup_components(self):
        """Setup integrated components."""
        return {
            'comp1': Component1(),
            'comp2': Component2()
        }
    
    async def test_workflow(self, setup_components):
        """Test complete workflow."""
        components = setup_components
        
        # Test workflow steps
        result1 = await components['comp1'].step1()
        result2 = await components['comp2'].step2(result1)
        
        assert result2.success is True
```

### API Test Template

```python
"""
API tests for [endpoint_group].
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from components.api_integration import app


@pytest.mark.api
class Test[EndpointGroup]:
    """Test [endpoint_group] API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_endpoint(self, client, auth_headers):
        """Test specific endpoint."""
        response = client.get("/endpoint", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "expected_field" in data
```

## 🐛 Debugging Tests

### Running with Debug

```bash
# Verbose output
python test_runner.py all -v

# Drop into debugger on failure
pytest --pdb

# Show local variables on failure
pytest --tb=long
```

### Common Issues

**Import Errors:**
```bash
# Ensure you're in the project directory
cd docs_crawler

# Install in development mode
pip install -e .
```

**Mock Issues:**
```python
# Use patch correctly
@patch('components.module.ClassName')  # Full path
def test_function(self, mock_class):
    pass
```

**Async Test Issues:**
```python
# Mark async tests correctly
@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result is not None
```

## 🚀 Continuous Integration

### GitHub Actions (Example)

```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: python test_runner.py all --no-coverage
      - run: python test_runner.py coverage
      - uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Setup hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## 📈 Performance Testing

### Load Testing (Future)

```python
# Example load test
@pytest.mark.slow
def test_api_load():
    """Test API under load."""
    import concurrent.futures
    
    def make_request():
        response = client.get("/health")
        return response.status_code == 200
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request) for _ in range(100)]
        results = [f.result() for f in futures]
    
    assert all(results)
```

### Memory Testing

```python
# Example memory test
@pytest.mark.slow
def test_memory_usage():
    """Test memory usage under load."""
    import psutil
    import gc
    
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    # Perform memory-intensive operations
    for _ in range(1000):
        large_data = "x" * 10000
        process_data(large_data)
    
    gc.collect()
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # Should not increase by more than 100MB
    assert memory_increase < 100 * 1024 * 1024
```

## 🎯 Test Maintenance

### Regular Tasks

1. **Update Dependencies**: Keep testing dependencies current
2. **Review Coverage**: Ensure coverage targets are met
3. **Clean Up Tests**: Remove obsolete tests
4. **Update Documentation**: Keep testing docs current

### Test Quality Checklist

- [ ] Tests are isolated and independent
- [ ] Tests have clear, descriptive names
- [ ] Tests follow AAA pattern (Arrange, Act, Assert)
- [ ] Mocks are used appropriately
- [ ] Edge cases are covered
- [ ] Error conditions are tested
- [ ] Tests run quickly (< 10s for full suite)

## 🆘 Troubleshooting

### Common Test Failures

**Configuration Issues:**
```bash
# Reset test environment
python test_runner.py --setup-only
```

**Database Conflicts:**
```bash
# Use isolated test database
export TEST_DATABASE_URL="sqlite:///test.db"
```

**Dependency Issues:**
```bash
# Reinstall test dependencies
pip install -r requirements.txt --force-reinstall
```

### Getting Help

1. Check test logs for detailed error messages
2. Run tests with `-v` flag for verbose output
3. Use `--pdb` to debug failing tests
4. Check GitHub issues for known problems
5. Consult the main README.md for setup issues

## 📚 Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [Python unittest.mock](https://docs.python.org/3/library/unittest.mock.html)
- [Coverage.py](https://coverage.readthedocs.io/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)

---

**Happy Testing! 🧪**