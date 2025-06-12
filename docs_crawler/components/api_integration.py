"""
API Integration and Webhooks Module
Provides REST API endpoints and webhook capabilities
"""

import asyncio
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import hmac
import uuid
from pathlib import Path

import requests
import streamlit as st
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

from components.supabase_integration import get_supabase_client
from components.monitoring import get_monitor, record_api_response_time


class WebhookEvent(Enum):
    """Webhook event types"""
    CRAWL_STARTED = "crawl.started"
    CRAWL_COMPLETED = "crawl.completed"
    CRAWL_FAILED = "crawl.failed"
    DOCUMENT_PROCESSED = "document.processed"
    EMBEDDING_GENERATED = "embedding.generated"
    SEARCH_PERFORMED = "search.performed"
    PROJECT_CREATED = "project.created"
    PROJECT_DELETED = "project.deleted"
    ALERT_TRIGGERED = "alert.triggered"
    SYSTEM_ERROR = "system.error"


class WebhookStatus(Enum):
    """Webhook delivery status"""
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class WebhookEndpoint:
    """Webhook endpoint configuration"""
    id: str
    url: str
    events: List[str]
    secret: str
    enabled: bool = True
    created_at: datetime = None
    last_delivery: Optional[datetime] = None
    failure_count: int = 0
    max_retries: int = 3
    retry_delay: int = 60  # seconds
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class WebhookDelivery:
    """Webhook delivery record"""
    id: str
    webhook_id: str
    event: str
    payload: Dict[str, Any]
    status: WebhookStatus
    created_at: datetime
    delivered_at: Optional[datetime] = None
    response_code: Optional[int] = None
    response_body: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0


# Pydantic models for API
class ProjectCreate(BaseModel):
    name: str
    description: str = ""
    initial_urls: List[str] = []


class ProjectResponse(BaseModel):
    id: str
    name: str
    description: str
    created_at: str
    document_count: int
    embedding_count: int
    status: str


class CrawlJobCreate(BaseModel):
    project_id: str
    urls: List[str]
    max_depth: int = 3
    max_pages: int = 100
    rag_strategies: List[str] = ["vector_embeddings"]


class CrawlJobResponse(BaseModel):
    id: str
    project_id: str
    status: str
    urls: List[str]
    created_at: str
    completed_at: Optional[str] = None
    documents_found: int
    documents_processed: int
    error_message: Optional[str] = None


class SearchRequest(BaseModel):
    project_id: str
    query: str
    limit: int = 20
    similarity_threshold: float = 0.7
    filters: Dict[str, Any] = {}


class SearchResult(BaseModel):
    id: str
    title: str
    content: str
    url: str
    similarity_score: float
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_count: int
    execution_time: float


class WebhookCreate(BaseModel):
    url: str
    events: List[str]
    secret: str = ""


class WebhookResponse(BaseModel):
    id: str
    url: str
    events: List[str]
    enabled: bool
    created_at: str
    last_delivery: Optional[str] = None
    failure_count: int


class WebhookDeliveryResponse(BaseModel):
    id: str
    webhook_id: str
    event: str
    status: str
    created_at: str
    delivered_at: Optional[str] = None
    response_code: Optional[int] = None
    error_message: Optional[str] = None


class APIIntegrationManager:
    """Main API integration manager"""
    
    def __init__(self):
        self.webhooks: Dict[str, WebhookEndpoint] = {}
        self.deliveries: Dict[str, WebhookDelivery] = {}
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        self.rate_limits: Dict[str, List[float]] = {}
        
        # FastAPI app
        self.app = None
        self.server_thread = None
        self.server_port = 8000
        
        # Load configuration
        self.load_webhooks()
        self.load_api_keys()
        
        # Setup background tasks
        self.webhook_processor_active = False
        self.start_webhook_processor()
    
    def setup_fastapi_app(self) -> FastAPI:
        """Setup FastAPI application with all endpoints"""
        
        app = FastAPI(
            title="Crawl4AI Standalone API",
            description="REST API for Crawl4AI Standalone Application",
            version="1.0.0"
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Security
        security = HTTPBearer()
        
        def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
            """Validate API key"""
            token = credentials.credentials
            
            if token not in self.api_keys:
                raise HTTPException(status_code=401, detail="Invalid API key")
            
            # Check rate limiting
            if not self.check_rate_limit(token):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            return self.api_keys[token]
        
        # API endpoints
        @app.get("/")
        async def root():
            return {"message": "Crawl4AI Standalone API", "version": "1.0.0"}
        
        @app.get("/health")
        async def health_check():
            """Health check endpoint"""
            monitor = get_monitor()
            health = monitor.get_health_status()
            
            return {
                "status": "healthy" if health["status"] in ["healthy", "warning"] else "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "details": health
            }
        
        # Project endpoints
        @app.post("/projects", response_model=ProjectResponse)
        async def create_project(
            project: ProjectCreate,
            background_tasks: BackgroundTasks,
            user=Depends(get_current_user)
        ):
            """Create a new project"""
            
            with record_api_response_time(time.time()):
                try:
                    supabase = get_supabase_client()
                    
                    # Create project in database
                    project_data = {
                        "name": project.name,
                        "description": project.description,
                        "created_at": datetime.now().isoformat()
                    }
                    
                    result = supabase.table("projects").insert(project_data).execute()
                    project_id = result.data[0]["id"]
                    
                    # Trigger webhook
                    background_tasks.add_task(
                        self.trigger_webhook,
                        WebhookEvent.PROJECT_CREATED,
                        {"project_id": project_id, "project": project_data}
                    )
                    
                    return ProjectResponse(
                        id=project_id,
                        name=project.name,
                        description=project.description,
                        created_at=project_data["created_at"],
                        document_count=0,
                        embedding_count=0,
                        status="active"
                    )
                    
                except Exception as e:
                    logging.error(f"Failed to create project: {e}")
                    raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/projects", response_model=List[ProjectResponse])
        async def list_projects(user=Depends(get_current_user)):
            """List all projects"""
            
            try:
                supabase = get_supabase_client()
                result = supabase.table("projects").select("*").execute()
                
                projects = []
                for project in result.data:
                    # Get document and embedding counts
                    doc_count = len(supabase.table("documents").select("id").eq("project_id", project["id"]).execute().data)
                    embed_count = len(supabase.table("embeddings").select("id").eq("project_id", project["id"]).execute().data)
                    
                    projects.append(ProjectResponse(
                        id=project["id"],
                        name=project["name"],
                        description=project.get("description", ""),
                        created_at=project["created_at"],
                        document_count=doc_count,
                        embedding_count=embed_count,
                        status="active"
                    ))
                
                return projects
                
            except Exception as e:
                logging.error(f"Failed to list projects: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/projects/{project_id}", response_model=ProjectResponse)
        async def get_project(project_id: str, user=Depends(get_current_user)):
            """Get project details"""
            
            try:
                supabase = get_supabase_client()
                result = supabase.table("projects").select("*").eq("id", project_id).execute()
                
                if not result.data:
                    raise HTTPException(status_code=404, detail="Project not found")
                
                project = result.data[0]
                
                # Get counts
                doc_count = len(supabase.table("documents").select("id").eq("project_id", project_id).execute().data)
                embed_count = len(supabase.table("embeddings").select("id").eq("project_id", project_id).execute().data)
                
                return ProjectResponse(
                    id=project["id"],
                    name=project["name"],
                    description=project.get("description", ""),
                    created_at=project["created_at"],
                    document_count=doc_count,
                    embedding_count=embed_count,
                    status="active"
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logging.error(f"Failed to get project: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.delete("/projects/{project_id}")
        async def delete_project(
            project_id: str,
            background_tasks: BackgroundTasks,
            user=Depends(get_current_user)
        ):
            """Delete a project"""
            
            try:
                supabase = get_supabase_client()
                
                # Check if project exists
                result = supabase.table("projects").select("*").eq("id", project_id).execute()
                if not result.data:
                    raise HTTPException(status_code=404, detail="Project not found")
                
                project = result.data[0]
                
                # Delete associated data
                supabase.table("embeddings").delete().eq("project_id", project_id).execute()
                supabase.table("documents").delete().eq("project_id", project_id).execute()
                supabase.table("projects").delete().eq("id", project_id).execute()
                
                # Trigger webhook
                background_tasks.add_task(
                    self.trigger_webhook,
                    WebhookEvent.PROJECT_DELETED,
                    {"project_id": project_id, "project": project}
                )
                
                return {"message": "Project deleted successfully"}
                
            except HTTPException:
                raise
            except Exception as e:
                logging.error(f"Failed to delete project: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Crawl job endpoints
        @app.post("/crawl", response_model=CrawlJobResponse)
        async def start_crawl(
            crawl_job: CrawlJobCreate,
            background_tasks: BackgroundTasks,
            user=Depends(get_current_user)
        ):
            """Start a new crawl job"""
            
            try:
                # Create crawl job
                job_id = str(uuid.uuid4())
                
                job_data = {
                    "id": job_id,
                    "project_id": crawl_job.project_id,
                    "urls": crawl_job.urls,
                    "status": "pending",
                    "created_at": datetime.now().isoformat(),
                    "max_depth": crawl_job.max_depth,
                    "max_pages": crawl_job.max_pages,
                    "rag_strategies": crawl_job.rag_strategies
                }
                
                # Store in session state (in production, use proper job queue)
                if 'api_crawl_jobs' not in st.session_state:
                    st.session_state.api_crawl_jobs = {}
                
                st.session_state.api_crawl_jobs[job_id] = job_data
                
                # Trigger webhook
                background_tasks.add_task(
                    self.trigger_webhook,
                    WebhookEvent.CRAWL_STARTED,
                    {"job_id": job_id, "job": job_data}
                )
                
                # Start background crawl (simplified for demo)
                background_tasks.add_task(self.process_crawl_job, job_id, job_data)
                
                return CrawlJobResponse(
                    id=job_id,
                    project_id=crawl_job.project_id,
                    status="pending",
                    urls=crawl_job.urls,
                    created_at=job_data["created_at"],
                    documents_found=0,
                    documents_processed=0
                )
                
            except Exception as e:
                logging.error(f"Failed to start crawl: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/crawl/{job_id}", response_model=CrawlJobResponse)
        async def get_crawl_status(job_id: str, user=Depends(get_current_user)):
            """Get crawl job status"""
            
            jobs = st.session_state.get('api_crawl_jobs', {})
            
            if job_id not in jobs:
                raise HTTPException(status_code=404, detail="Crawl job not found")
            
            job = jobs[job_id]
            
            return CrawlJobResponse(
                id=job_id,
                project_id=job["project_id"],
                status=job["status"],
                urls=job["urls"],
                created_at=job["created_at"],
                completed_at=job.get("completed_at"),
                documents_found=job.get("documents_found", 0),
                documents_processed=job.get("documents_processed", 0),
                error_message=job.get("error_message")
            )
        
        # Search endpoints
        @app.post("/search", response_model=SearchResponse)
        async def search_documents(
            search_request: SearchRequest,
            user=Depends(get_current_user)
        ):
            """Search documents using vector similarity"""
            
            from components.search_interface import perform_search
            from components.monitoring import record_search_request
            
            start_time = time.time()
            
            try:
                # Record search for monitoring
                record_search_request()
                
                # Perform search
                results = perform_search(
                    search_request.query,
                    search_request.project_id,
                    search_request.limit,
                    search_request.similarity_threshold,
                    search_request.filters
                )
                
                # Convert to API format
                search_results = []
                for result in results:
                    search_results.append(SearchResult(
                        id=result.get("id", ""),
                        title=result.get("title", ""),
                        content=result.get("content", ""),
                        url=result.get("url", ""),
                        similarity_score=result.get("similarity", 0.0),
                        metadata=result.get("metadata", {})
                    ))
                
                execution_time = time.time() - start_time
                
                # Trigger webhook
                asyncio.create_task(self.trigger_webhook(
                    WebhookEvent.SEARCH_PERFORMED,
                    {
                        "query": search_request.query,
                        "project_id": search_request.project_id,
                        "result_count": len(search_results),
                        "execution_time": execution_time
                    }
                ))
                
                return SearchResponse(
                    query=search_request.query,
                    results=search_results,
                    total_count=len(search_results),
                    execution_time=execution_time
                )
                
            except Exception as e:
                logging.error(f"Search failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Webhook management endpoints
        @app.post("/webhooks", response_model=WebhookResponse)
        async def create_webhook(webhook: WebhookCreate, user=Depends(get_current_user)):
            """Create a new webhook endpoint"""
            
            webhook_id = str(uuid.uuid4())
            secret = webhook.secret or self.generate_webhook_secret()
            
            webhook_endpoint = WebhookEndpoint(
                id=webhook_id,
                url=webhook.url,
                events=webhook.events,
                secret=secret
            )
            
            self.webhooks[webhook_id] = webhook_endpoint
            self.save_webhooks()
            
            return WebhookResponse(
                id=webhook_id,
                url=webhook.url,
                events=webhook.events,
                enabled=True,
                created_at=webhook_endpoint.created_at.isoformat()
            )
        
        @app.get("/webhooks", response_model=List[WebhookResponse])
        async def list_webhooks(user=Depends(get_current_user)):
            """List all webhook endpoints"""
            
            webhooks = []
            for webhook in self.webhooks.values():
                webhooks.append(WebhookResponse(
                    id=webhook.id,
                    url=webhook.url,
                    events=webhook.events,
                    enabled=webhook.enabled,
                    created_at=webhook.created_at.isoformat(),
                    last_delivery=webhook.last_delivery.isoformat() if webhook.last_delivery else None,
                    failure_count=webhook.failure_count
                ))
            
            return webhooks
        
        @app.delete("/webhooks/{webhook_id}")
        async def delete_webhook(webhook_id: str, user=Depends(get_current_user)):
            """Delete a webhook endpoint"""
            
            if webhook_id not in self.webhooks:
                raise HTTPException(status_code=404, detail="Webhook not found")
            
            del self.webhooks[webhook_id]
            self.save_webhooks()
            
            return {"message": "Webhook deleted successfully"}
        
        @app.get("/webhooks/{webhook_id}/deliveries", response_model=List[WebhookDeliveryResponse])
        async def get_webhook_deliveries(webhook_id: str, user=Depends(get_current_user)):
            """Get webhook delivery history"""
            
            if webhook_id not in self.webhooks:
                raise HTTPException(status_code=404, detail="Webhook not found")
            
            deliveries = [
                delivery for delivery in self.deliveries.values()
                if delivery.webhook_id == webhook_id
            ]
            
            delivery_responses = []
            for delivery in sorted(deliveries, key=lambda x: x.created_at, reverse=True)[:50]:
                delivery_responses.append(WebhookDeliveryResponse(
                    id=delivery.id,
                    webhook_id=delivery.webhook_id,
                    event=delivery.event,
                    status=delivery.status.value,
                    created_at=delivery.created_at.isoformat(),
                    delivered_at=delivery.delivered_at.isoformat() if delivery.delivered_at else None,
                    response_code=delivery.response_code,
                    error_message=delivery.error_message
                ))
            
            return delivery_responses
        
        self.app = app
        return app
    
    def start_api_server(self, port: int = 8000):
        """Start the API server in a background thread"""
        
        if self.server_thread and self.server_thread.is_alive():
            logging.warning("API server is already running")
            return
        
        self.server_port = port
        self.setup_fastapi_app()
        
        def run_server():
            uvicorn.run(self.app, host="0.0.0.0", port=port, log_level="info")
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        
        logging.info(f"API server started on port {port}")
    
    def stop_api_server(self):
        """Stop the API server"""
        # In a real implementation, you'd need to properly shut down uvicorn
        logging.info("API server stop requested")
    
    def generate_api_key(self, name: str, permissions: List[str] = None) -> str:
        """Generate a new API key"""
        
        api_key = f"crawl4ai_{uuid.uuid4().hex}"
        
        self.api_keys[api_key] = {
            "name": name,
            "permissions": permissions or ["read", "write"],
            "created_at": datetime.now().isoformat(),
            "last_used": None,
            "request_count": 0
        }
        
        self.save_api_keys()
        return api_key
    
    def revoke_api_key(self, api_key: str):
        """Revoke an API key"""
        
        if api_key in self.api_keys:
            del self.api_keys[api_key]
            self.save_api_keys()
    
    def check_rate_limit(self, api_key: str, requests_per_minute: int = 60) -> bool:
        """Check if API key is within rate limits"""
        
        current_time = time.time()
        
        if api_key not in self.rate_limits:
            self.rate_limits[api_key] = []
        
        # Clean old requests (older than 1 minute)
        self.rate_limits[api_key] = [
            req_time for req_time in self.rate_limits[api_key]
            if current_time - req_time < 60
        ]
        
        # Check if under limit
        if len(self.rate_limits[api_key]) >= requests_per_minute:
            return False
        
        # Add current request
        self.rate_limits[api_key].append(current_time)
        
        # Update API key usage
        if api_key in self.api_keys:
            self.api_keys[api_key]["last_used"] = datetime.now().isoformat()
            self.api_keys[api_key]["request_count"] += 1
        
        return True
    
    def generate_webhook_secret(self) -> str:
        """Generate a webhook secret"""
        return f"whsec_{uuid.uuid4().hex}"
    
    def create_webhook_signature(self, payload: str, secret: str) -> str:
        """Create webhook signature for verification"""
        return hmac.new(
            secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    async def trigger_webhook(self, event: WebhookEvent, payload: Dict[str, Any]):
        """Trigger webhook for event"""
        
        webhook_payload = {
            "event": event.value,
            "timestamp": datetime.now().isoformat(),
            "data": payload
        }
        
        # Find webhooks listening for this event
        relevant_webhooks = [
            webhook for webhook in self.webhooks.values()
            if webhook.enabled and event.value in webhook.events
        ]
        
        # Queue deliveries
        for webhook in relevant_webhooks:
            delivery_id = str(uuid.uuid4())
            
            delivery = WebhookDelivery(
                id=delivery_id,
                webhook_id=webhook.id,
                event=event.value,
                payload=webhook_payload,
                status=WebhookStatus.PENDING,
                created_at=datetime.now()
            )
            
            self.deliveries[delivery_id] = delivery
            
            # Async delivery will be handled by background processor
    
    def start_webhook_processor(self):
        """Start background webhook processor"""
        
        if self.webhook_processor_active:
            return
        
        self.webhook_processor_active = True
        
        def webhook_processor():
            while self.webhook_processor_active:
                try:
                    self.process_pending_webhooks()
                    time.sleep(5)  # Check every 5 seconds
                except Exception as e:
                    logging.error(f"Webhook processor error: {e}")
                    time.sleep(30)
        
        thread = threading.Thread(target=webhook_processor, daemon=True)
        thread.start()
    
    def process_pending_webhooks(self):
        """Process pending webhook deliveries"""
        
        pending_deliveries = [
            delivery for delivery in self.deliveries.values()
            if delivery.status in [WebhookStatus.PENDING, WebhookStatus.RETRYING]
        ]
        
        for delivery in pending_deliveries:
            try:
                self.deliver_webhook(delivery)
            except Exception as e:
                logging.error(f"Failed to deliver webhook {delivery.id}: {e}")
    
    def deliver_webhook(self, delivery: WebhookDelivery):
        """Deliver a webhook"""
        
        webhook = self.webhooks.get(delivery.webhook_id)
        if not webhook or not webhook.enabled:
            delivery.status = WebhookStatus.FAILED
            delivery.error_message = "Webhook endpoint disabled or not found"
            return
        
        # Check retry logic
        if delivery.retry_count >= webhook.max_retries:
            delivery.status = WebhookStatus.FAILED
            delivery.error_message = "Max retries exceeded"
            return
        
        # Check retry delay
        if delivery.status == WebhookStatus.RETRYING:
            time_since_last_attempt = (datetime.now() - delivery.created_at).total_seconds()
            if time_since_last_attempt < webhook.retry_delay:
                return
        
        try:
            # Prepare payload
            payload_str = json.dumps(delivery.payload)
            signature = self.create_webhook_signature(payload_str, webhook.secret)
            
            headers = {
                "Content-Type": "application/json",
                "X-Webhook-Signature": f"sha256={signature}",
                "User-Agent": "Crawl4AI-Webhooks/1.0"
            }
            
            # Make request
            response = requests.post(
                webhook.url,
                data=payload_str,
                headers=headers,
                timeout=30
            )
            
            # Update delivery status
            delivery.response_code = response.status_code
            delivery.response_body = response.text[:1000]  # Limit response body size
            
            if 200 <= response.status_code < 300:
                delivery.status = WebhookStatus.DELIVERED
                delivery.delivered_at = datetime.now()
                webhook.last_delivery = datetime.now()
                webhook.failure_count = 0
            else:
                delivery.status = WebhookStatus.RETRYING
                delivery.retry_count += 1
                webhook.failure_count += 1
                
        except Exception as e:
            delivery.status = WebhookStatus.RETRYING
            delivery.retry_count += 1
            delivery.error_message = str(e)
            webhook.failure_count += 1
    
    async def process_crawl_job(self, job_id: str, job_data: Dict[str, Any]):
        """Process a crawl job (simplified for demo)"""
        
        try:
            # Update status
            job_data["status"] = "running"
            
            # Simulate crawling process
            await asyncio.sleep(2)
            
            # Simulate processing results
            job_data["documents_found"] = len(job_data["urls"]) * 5
            job_data["documents_processed"] = job_data["documents_found"]
            job_data["status"] = "completed"
            job_data["completed_at"] = datetime.now().isoformat()
            
            # Trigger completion webhook
            await self.trigger_webhook(
                WebhookEvent.CRAWL_COMPLETED,
                {"job_id": job_id, "job": job_data}
            )
            
        except Exception as e:
            job_data["status"] = "failed"
            job_data["error_message"] = str(e)
            job_data["completed_at"] = datetime.now().isoformat()
            
            # Trigger failure webhook
            await self.trigger_webhook(
                WebhookEvent.CRAWL_FAILED,
                {"job_id": job_id, "job": job_data, "error": str(e)}
            )
    
    def load_webhooks(self):
        """Load webhooks from storage"""
        # In production, load from database
        pass
    
    def save_webhooks(self):
        """Save webhooks to storage"""
        # In production, save to database
        pass
    
    def load_api_keys(self):
        """Load API keys from storage"""
        # In production, load from secure storage
        # For demo, create a default key
        if not self.api_keys:
            default_key = self.generate_api_key("Default API Key")
            logging.info(f"Generated default API key: {default_key}")
    
    def save_api_keys(self):
        """Save API keys to storage"""
        # In production, save to secure storage
        pass
    
    def get_api_stats(self) -> Dict[str, Any]:
        """Get API usage statistics"""
        
        total_requests = sum(key_data["request_count"] for key_data in self.api_keys.values())
        active_keys = len([key for key, data in self.api_keys.items() if data["last_used"]])
        
        return {
            "total_api_keys": len(self.api_keys),
            "active_api_keys": active_keys,
            "total_requests": total_requests,
            "total_webhooks": len(self.webhooks),
            "active_webhooks": len([w for w in self.webhooks.values() if w.enabled]),
            "total_deliveries": len(self.deliveries),
            "successful_deliveries": len([d for d in self.deliveries.values() if d.status == WebhookStatus.DELIVERED])
        }

    def create_webhook_endpoint(self, url: str, events: List[str], secret: str = None) -> str:
        """Create a new webhook endpoint"""
        
        webhook_id = str(uuid.uuid4())
        secret = secret or self.generate_webhook_secret()
        
        webhook_endpoint = WebhookEndpoint(
            id=webhook_id,
            url=url,
            events=events,
            secret=secret
        )
        
        self.webhooks[webhook_id] = webhook_endpoint
        self.save_webhooks()
        
        return webhook_id


# Global API manager instance
_api_manager = None


def get_api_manager() -> APIIntegrationManager:
    """Get the global API manager instance"""
    global _api_manager
    
    if _api_manager is None:
        _api_manager = APIIntegrationManager()
    
    return _api_manager