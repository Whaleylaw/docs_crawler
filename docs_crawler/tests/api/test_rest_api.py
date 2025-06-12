"""
API tests for REST endpoints.
"""

import pytest
import json
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from components.api_integration import APIIntegrationManager, app


@pytest.mark.api
class TestAPIEndpoints:
    """Test REST API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def api_manager(self):
        """Create API manager instance."""
        return APIIntegrationManager()
    
    @pytest.fixture
    def auth_headers(self, api_manager):
        """Create authentication headers."""
        # Create test API key
        api_key = api_manager.create_api_key("test-user", "Test API Key")
        return {"Authorization": f"Bearer {api_key}"}
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_create_project(self, client, auth_headers):
        """Test project creation endpoint."""
        project_data = {
            "name": "Test Project",
            "description": "Test project description",
            "supabase_url": "https://test.supabase.co",
            "supabase_key": "test-key"
        }
        
        with patch('components.supabase_integration.SupabaseIntegration.create_project') as mock_create:
            mock_create.return_value = "test-project-id"
            
            response = client.post(
                "/projects",
                json=project_data,
                headers=auth_headers
            )
            
            assert response.status_code == 201
            data = response.json()
            assert data["id"] == "test-project-id"
            assert data["name"] == "Test Project"
            assert data["status"] == "created"
    
    def test_create_project_unauthorized(self, client):
        """Test project creation without authentication."""
        project_data = {
            "name": "Test Project",
            "description": "Test description"
        }
        
        response = client.post("/projects", json=project_data)
        assert response.status_code == 401
    
    def test_list_projects(self, client, auth_headers):
        """Test project listing endpoint."""
        mock_projects = [
            {
                "id": "proj1",
                "name": "Project 1",
                "description": "First project",
                "created_at": "2025-01-01T00:00:00Z"
            },
            {
                "id": "proj2", 
                "name": "Project 2",
                "description": "Second project",
                "created_at": "2025-01-02T00:00:00Z"
            }
        ]
        
        with patch('components.supabase_integration.SupabaseIntegration.list_projects') as mock_list:
            mock_list.return_value = mock_projects
            
            response = client.get("/projects", headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2
            assert data[0]["name"] == "Project 1"
            assert data[1]["name"] == "Project 2"
    
    def test_get_project(self, client, auth_headers):
        """Test getting single project."""
        mock_project = {
            "id": "test-proj",
            "name": "Test Project",
            "description": "Test description",
            "documents_count": 42,
            "storage_used": 1024
        }
        
        with patch('components.supabase_integration.SupabaseIntegration.get_project') as mock_get:
            mock_get.return_value = mock_project
            
            response = client.get("/projects/test-proj", headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "test-proj"
            assert data["documents_count"] == 42
    
    def test_get_project_not_found(self, client, auth_headers):
        """Test getting non-existent project."""
        with patch('components.supabase_integration.SupabaseIntegration.get_project') as mock_get:
            mock_get.return_value = None
            
            response = client.get("/projects/nonexistent", headers=auth_headers)
            assert response.status_code == 404
    
    def test_delete_project(self, client, auth_headers):
        """Test project deletion."""
        with patch('components.supabase_integration.SupabaseIntegration.delete_project') as mock_delete:
            mock_delete.return_value = True
            
            response = client.delete("/projects/test-proj", headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "Project deleted successfully"
    
    def test_start_crawl(self, client, auth_headers):
        """Test starting crawl job."""
        crawl_data = {
            "project_id": "test-proj",
            "urls": ["https://example.com", "https://test.com"],
            "max_depth": 2,
            "max_concurrent": 5,
            "rag_strategies": ["vector_embeddings"]
        }
        
        with patch('components.crawling_engine.CrawlingEngine.create_job') as mock_create:
            mock_job = MagicMock()
            mock_job.job_id = "test-job-123"
            mock_job.status = "pending"
            mock_job.to_dict.return_value = {
                "job_id": "test-job-123",
                "status": "pending",
                "urls": crawl_data["urls"]
            }
            mock_create.return_value = mock_job
            
            with patch('components.crawling_engine.CrawlingEngine.start_crawl'):
                response = client.post(
                    "/crawl",
                    json=crawl_data,
                    headers=auth_headers
                )
                
                assert response.status_code == 202
                data = response.json()
                assert data["job_id"] == "test-job-123"
                assert data["status"] == "pending"
    
    def test_get_crawl_status(self, client, auth_headers):
        """Test getting crawl job status."""
        mock_status = {
            "job_id": "test-job-123",
            "status": "running",
            "urls_total": 10,
            "urls_completed": 7,
            "urls_failed": 1,
            "progress": 70.0
        }
        
        with patch('components.crawling_engine.CrawlingEngine.get_job_status') as mock_status_func:
            mock_status_func.return_value = mock_status
            
            response = client.get("/crawl/test-job-123/status", headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            assert data["job_id"] == "test-job-123"
            assert data["progress"] == 70.0
    
    def test_stop_crawl(self, client, auth_headers):
        """Test stopping crawl job."""
        with patch('components.crawling_engine.CrawlingEngine.stop_crawl') as mock_stop:
            with patch('components.crawling_engine.CrawlingEngine.get_job') as mock_get_job:
                mock_job = MagicMock()
                mock_job.status = "cancelled"
                mock_get_job.return_value = mock_job
                
                response = client.post("/crawl/test-job-123/stop", headers=auth_headers)
                
                assert response.status_code == 200
                data = response.json()
                assert data["message"] == "Crawl job stopped successfully"
    
    def test_get_crawl_results(self, client, auth_headers):
        """Test getting crawl results."""
        mock_results = [
            {
                "url": "https://example.com",
                "title": "Example Page",
                "success": True,
                "content_length": 1024
            },
            {
                "url": "https://test.com",
                "title": "Test Page", 
                "success": True,
                "content_length": 2048
            }
        ]
        
        with patch('components.crawling_engine.CrawlingEngine.get_job_results') as mock_results_func:
            mock_results_func.return_value = mock_results
            
            response = client.get("/crawl/test-job-123/results", headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2
            assert data[0]["title"] == "Example Page"
    
    def test_search_documents(self, client, auth_headers):
        """Test document search endpoint."""
        search_data = {
            "project_id": "test-proj",
            "query": "machine learning",
            "limit": 10,
            "similarity_threshold": 0.7
        }
        
        mock_results = [
            {
                "id": "doc1",
                "content": "Machine learning algorithms",
                "title": "ML Guide",
                "url": "https://example.com/ml",
                "similarity": 0.92
            }
        ]
        
        with patch('components.search_engine.SearchEngine.search') as mock_search:
            mock_search.return_value = mock_results
            
            response = client.post(
                "/search",
                json=search_data,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["title"] == "ML Guide"
            assert data[0]["similarity"] == 0.92
    
    def test_create_webhook(self, client, auth_headers):
        """Test webhook creation."""
        webhook_data = {
            "url": "https://example.com/webhook",
            "events": ["crawl.completed", "crawl.failed"],
            "description": "Test webhook"
        }
        
        mock_webhook = {
            "id": "webhook-123",
            "url": "https://example.com/webhook",
            "events": ["crawl.completed", "crawl.failed"],
            "status": "active"
        }
        
        with patch('components.api_integration.APIIntegrationManager.create_webhook') as mock_create:
            mock_create.return_value = mock_webhook
            
            response = client.post(
                "/webhooks",
                json=webhook_data,
                headers=auth_headers
            )
            
            assert response.status_code == 201
            data = response.json()
            assert data["id"] == "webhook-123"
            assert data["status"] == "active"
    
    def test_list_webhooks(self, client, auth_headers):
        """Test webhook listing."""
        mock_webhooks = [
            {
                "id": "webhook-1",
                "url": "https://example.com/webhook1",
                "events": ["crawl.completed"],
                "status": "active"
            },
            {
                "id": "webhook-2",
                "url": "https://example.com/webhook2", 
                "events": ["crawl.failed"],
                "status": "inactive"
            }
        ]
        
        with patch('components.api_integration.APIIntegrationManager.list_webhooks') as mock_list:
            mock_list.return_value = mock_webhooks
            
            response = client.get("/webhooks", headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2
            assert data[0]["status"] == "active"
    
    def test_delete_webhook(self, client, auth_headers):
        """Test webhook deletion."""
        with patch('components.api_integration.APIIntegrationManager.delete_webhook') as mock_delete:
            mock_delete.return_value = True
            
            response = client.delete("/webhooks/webhook-123", headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "Webhook deleted successfully"
    
    def test_get_api_usage(self, client, auth_headers):
        """Test API usage statistics."""
        mock_usage = {
            "total_requests": 1000,
            "requests_today": 50,
            "rate_limit_remaining": 950,
            "endpoints": {
                "/projects": 200,
                "/crawl": 300,
                "/search": 500
            }
        }
        
        with patch('components.api_integration.APIIntegrationManager.get_usage_stats') as mock_stats:
            mock_stats.return_value = mock_usage
            
            response = client.get("/usage", headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            assert data["total_requests"] == 1000
            assert data["requests_today"] == 50
    
    def test_rate_limiting(self, client, api_manager):
        """Test rate limiting functionality."""
        # Create API key with low rate limit for testing
        api_key = api_manager.create_api_key("test-user", "Rate Limited Key", rate_limit=2)
        headers = {"Authorization": f"Bearer {api_key}"}
        
        # Make requests up to limit
        for i in range(2):
            response = client.get("/health", headers=headers)
            assert response.status_code == 200
        
        # Next request should be rate limited
        with patch('components.api_integration.APIIntegrationManager.check_rate_limit') as mock_check:
            mock_check.return_value = False  # Simulate rate limit exceeded
            
            response = client.get("/health", headers=headers)
            assert response.status_code == 429
    
    def test_invalid_api_key(self, client):
        """Test invalid API key handling."""
        headers = {"Authorization": "Bearer invalid-key"}
        
        response = client.get("/projects", headers=headers)
        assert response.status_code == 401
        
        data = response.json()
        assert "Invalid API key" in data["detail"]
    
    def test_input_validation(self, client, auth_headers):
        """Test input validation."""
        # Invalid project data
        invalid_project = {
            "name": "",  # Empty name should fail
            "description": "Test"
        }
        
        response = client.post(
            "/projects",
            json=invalid_project,
            headers=auth_headers
        )
        assert response.status_code == 422
        
        # Invalid crawl data
        invalid_crawl = {
            "project_id": "test",
            "urls": [],  # Empty URLs should fail
            "max_depth": -1  # Negative depth should fail
        }
        
        response = client.post(
            "/crawl",
            json=invalid_crawl,
            headers=auth_headers
        )
        assert response.status_code == 422
    
    def test_pagination(self, client, auth_headers):
        """Test pagination in list endpoints."""
        # Mock large project list
        mock_projects = [
            {"id": f"proj{i}", "name": f"Project {i}"}
            for i in range(25)
        ]
        
        with patch('components.supabase_integration.SupabaseIntegration.list_projects') as mock_list:
            mock_list.return_value = mock_projects[:10]  # Return first page
            
            response = client.get(
                "/projects?page=1&limit=10",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 10
    
    def test_error_handling(self, client, auth_headers):
        """Test error handling in API endpoints."""
        # Simulate database error
        with patch('components.supabase_integration.SupabaseIntegration.list_projects') as mock_list:
            mock_list.side_effect = Exception("Database connection failed")
            
            response = client.get("/projects", headers=auth_headers)
            assert response.status_code == 500
            
            data = response.json()
            assert "Internal server error" in data["detail"]


@pytest.mark.api
class TestWebhookDelivery:
    """Test webhook delivery functionality."""
    
    def test_webhook_delivery_success(self, api_manager):
        """Test successful webhook delivery."""
        webhook_data = {
            "url": "https://httpbin.org/post",
            "events": ["crawl.completed"],
            "description": "Test webhook"
        }
        
        webhook = api_manager.create_webhook(**webhook_data)
        
        # Mock successful HTTP response
        with patch('requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"received": True}
            mock_post.return_value = mock_response
            
            # Deliver webhook
            event_data = {
                "event": "crawl.completed",
                "job_id": "test-job-123",
                "timestamp": "2025-01-01T00:00:00Z"
            }
            
            success = api_manager.deliver_webhook(webhook["id"], event_data)
            assert success is True
            
            # Verify request was made
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[1]["json"] == event_data
    
    def test_webhook_delivery_failure(self, api_manager):
        """Test webhook delivery failure and retry."""
        webhook_data = {
            "url": "https://example.com/webhook",
            "events": ["crawl.failed"],  
            "description": "Test webhook"
        }
        
        webhook = api_manager.create_webhook(**webhook_data)
        
        # Mock failed HTTP response
        with patch('requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.raise_for_status.side_effect = Exception("Server error")
            mock_post.return_value = mock_response
            
            event_data = {
                "event": "crawl.failed",
                "job_id": "test-job-456",
                "error": "Network timeout"
            }
            
            success = api_manager.deliver_webhook(webhook["id"], event_data)
            assert success is False
            
            # Should have retried
            assert mock_post.call_count > 1
    
    def test_webhook_event_filtering(self, api_manager):
        """Test webhook event filtering."""
        # Create webhook that only listens to completed events
        webhook_data = {
            "url": "https://example.com/webhook",
            "events": ["crawl.completed"],
            "description": "Completion only webhook"
        }
        
        webhook = api_manager.create_webhook(**webhook_data)
        
        with patch('requests.post') as mock_post:
            # Send completed event - should be delivered
            completed_event = {
                "event": "crawl.completed",
                "job_id": "test-job-123"
            }
            
            api_manager.deliver_webhook(webhook["id"], completed_event)
            
            # Send failed event - should NOT be delivered
            failed_event = {
                "event": "crawl.failed", 
                "job_id": "test-job-456"
            }
            
            api_manager.deliver_webhook(webhook["id"], failed_event)
            
            # Only one call should have been made (for completed event)
            assert mock_post.call_count == 1