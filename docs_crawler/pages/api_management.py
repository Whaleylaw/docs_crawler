"""
API Management Page
Interface for managing REST API, webhooks, and integrations
"""

import streamlit as st
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
import plotly.graph_objects as go
import plotly.express as px
import time

from components.api_integration import get_api_manager, WebhookEvent


def show():
    """Display the API management page"""
    
    st.header("🔌 API Management & Integrations")
    st.markdown("Manage REST API access, webhooks, and external integrations")
    
    # Get API manager
    api_manager = get_api_manager()
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔑 API Keys",
        "🔗 Webhooks",
        "📊 Usage Statistics",
        "📖 API Documentation",
        "🔧 Server Management"
    ])
    
    with tab1:
        display_api_keys_management(api_manager)
    
    with tab2:
        display_webhooks_management(api_manager)
    
    with tab3:
        display_usage_statistics(api_manager)
    
    with tab4:
        display_api_documentation()
    
    with tab5:
        display_server_management(api_manager)


def display_api_keys_management(api_manager):
    """Display API keys management interface"""
    
    st.subheader("🔑 API Keys Management")
    st.markdown("Create and manage API keys for programmatic access")
    
    # API Statistics
    api_stats = api_manager.get_api_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total API Keys", api_stats["total_api_keys"])
    
    with col2:
        st.metric("Active Keys", api_stats["active_api_keys"])
    
    with col3:
        st.metric("Total Requests", f"{api_stats['total_requests']:,}")
    
    with col4:
        st.metric("Server Status", "Running" if api_manager.server_thread and api_manager.server_thread.is_alive() else "Stopped")
    
    st.markdown("---")
    
    # Create new API key
    with st.expander("➕ Create New API Key", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            key_name = st.text_input("API Key Name", placeholder="My Application")
            key_description = st.text_area("Description (optional)", placeholder="Description of what this key will be used for")
        
        with col2:
            st.markdown("**Permissions**")
            read_permission = st.checkbox("Read Access", value=True)
            write_permission = st.checkbox("Write Access", value=True)
            admin_permission = st.checkbox("Admin Access", value=False)
        
        if st.button("🔐 Generate API Key", type="primary"):
            if key_name.strip():
                permissions = []
                if read_permission:
                    permissions.append("read")
                if write_permission:
                    permissions.append("write")
                if admin_permission:
                    permissions.append("admin")
                
                new_key = api_manager.generate_api_key(key_name.strip(), permissions)
                
                st.success("✅ API Key created successfully!")
                st.code(new_key, language=None)
                st.warning("⚠️ **Important**: Save this API key now. You won't be able to see it again!")
                
                # Store in session for display
                if 'new_api_key' not in st.session_state:
                    st.session_state.new_api_key = {}
                st.session_state.new_api_key[new_key] = {
                    "name": key_name,
                    "permissions": permissions,
                    "created": datetime.now().isoformat()
                }
            else:
                st.error("Please provide a name for the API key")
    
    # Display existing API keys
    st.markdown("### 🗂️ Existing API Keys")
    
    if api_manager.api_keys:
        # Create DataFrame for display
        keys_data = []
        for api_key, data in api_manager.api_keys.items():
            # Mask the key for security
            masked_key = f"{api_key[:12]}...{api_key[-4:]}"
            
            keys_data.append({
                "Name": data["name"],
                "API Key": masked_key,
                "Permissions": ", ".join(data["permissions"]),
                "Created": data["created_at"][:10],
                "Last Used": data["last_used"][:10] if data["last_used"] else "Never",
                "Requests": data["request_count"],
                "Full Key": api_key  # Hidden for actions
            })
        
        df = pd.DataFrame(keys_data)
        
        # Display table with actions
        for idx, row in df.iterrows():
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 1, 1])
                
                with col1:
                    st.write(f"**{row['Name']}**")
                    st.caption(f"Key: {row['API Key']}")
                
                with col2:
                    st.write(f"Permissions: {row['Permissions']}")
                    st.caption(f"Created: {row['Created']}")
                
                with col3:
                    st.write(f"Last Used: {row['Last Used']}")
                    st.caption(f"Requests: {row['Requests']}")
                
                with col4:
                    if st.button("📋", key=f"copy_{idx}", help="Copy Key"):
                        st.code(row['Full Key'], language=None)
                
                with col5:
                    if st.button("🗑️", key=f"delete_{idx}", help="Delete Key"):
                        api_manager.revoke_api_key(row['Full Key'])
                        st.success("API key deleted")
                        st.rerun()
                
                st.markdown("---")
    else:
        st.info("No API keys created yet. Create your first API key above.")


def display_webhooks_management(api_manager):
    """Display webhooks management interface"""
    
    st.subheader("🔗 Webhooks Management")
    st.markdown("Configure webhooks to receive real-time notifications about events")
    
    # Webhook Statistics
    webhook_stats = api_manager.get_api_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Webhooks", webhook_stats["total_webhooks"])
    
    with col2:
        st.metric("Active Webhooks", webhook_stats["active_webhooks"])
    
    with col3:
        st.metric("Total Deliveries", webhook_stats["total_deliveries"])
    
    with col4:
        st.metric("Successful Deliveries", webhook_stats["successful_deliveries"])
    
    st.markdown("---")
    
    # Create new webhook
    with st.expander("➕ Create New Webhook", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            webhook_url = st.text_input(
                "Webhook URL",
                placeholder="https://example.com/webhook",
                help="The URL where webhook events will be sent"
            )
            
            webhook_secret = st.text_input(
                "Secret (optional)",
                type="password",
                help="Secret key for verifying webhook authenticity"
            )
        
        with col2:
            st.markdown("**Events to Subscribe**")
            
            all_events = [event.value for event in WebhookEvent]
            selected_events = []
            
            # Group events by category
            crawl_events = [e for e in all_events if e.startswith("crawl.")]
            document_events = [e for e in all_events if e.startswith("document.") or e.startswith("embedding.")]
            project_events = [e for e in all_events if e.startswith("project.")]
            system_events = [e for e in all_events if e.startswith("search.") or e.startswith("alert.") or e.startswith("system.")]
            
            if st.checkbox("Crawl Events", value=True):
                selected_events.extend(crawl_events)
                for event in crawl_events:
                    st.caption(f"• {event}")
            
            if st.checkbox("Document Events"):
                selected_events.extend(document_events)
                for event in document_events:
                    st.caption(f"• {event}")
            
            if st.checkbox("Project Events"):
                selected_events.extend(project_events)
                for event in project_events:
                    st.caption(f"• {event}")
            
            if st.checkbox("System Events"):
                selected_events.extend(system_events)
                for event in system_events:
                    st.caption(f"• {event}")
        
        if st.button("🔗 Create Webhook", type="primary"):
            if webhook_url.strip() and selected_events:
                try:
                    # Validate URL format
                    if not webhook_url.startswith(('http://', 'https://')):
                        st.error("Please provide a valid HTTP/HTTPS URL")
                        return
                    
                    webhook_id = api_manager.create_webhook_endpoint(
                        webhook_url.strip(),
                        selected_events,
                        webhook_secret.strip() if webhook_secret.strip() else None
                    )
                    
                    st.success(f"✅ Webhook created successfully!")
                    st.info(f"Webhook ID: {webhook_id}")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Failed to create webhook: {e}")
            else:
                st.error("Please provide a webhook URL and select at least one event")
    
    # Display existing webhooks
    st.markdown("### 🗂️ Existing Webhooks")
    
    if api_manager.webhooks:
        for webhook_id, webhook in api_manager.webhooks.items():
            with st.container():
                col1, col2, col3 = st.columns([3, 2, 1])
                
                with col1:
                    status_icon = "🟢" if webhook.enabled else "🔴"
                    st.write(f"{status_icon} **{webhook.url}**")
                    st.caption(f"Events: {', '.join(webhook.events[:3])}{'...' if len(webhook.events) > 3 else ''}")
                
                with col2:
                    st.write(f"Created: {webhook.created_at.strftime('%Y-%m-%d')}")
                    if webhook.last_delivery:
                        st.caption(f"Last delivery: {webhook.last_delivery.strftime('%Y-%m-%d %H:%M')}")
                    else:
                        st.caption("No deliveries yet")
                    
                    if webhook.failure_count > 0:
                        st.warning(f"⚠️ {webhook.failure_count} failures")
                
                with col3:
                    if st.button("📊", key=f"webhook_stats_{webhook_id}", help="View Deliveries"):
                        st.session_state.selected_webhook = webhook_id
                    
                    if st.button("🗑️", key=f"webhook_delete_{webhook_id}", help="Delete Webhook"):
                        del api_manager.webhooks[webhook_id]
                        st.success("Webhook deleted")
                        st.rerun()
                
                # Show webhook details if selected
                if st.session_state.get('selected_webhook') == webhook_id:
                    with st.expander(f"📊 Delivery History - {webhook.url}", expanded=True):
                        deliveries = [
                            delivery for delivery in api_manager.deliveries.values()
                            if delivery.webhook_id == webhook_id
                        ]
                        
                        if deliveries:
                            # Sort by creation time
                            deliveries.sort(key=lambda x: x.created_at, reverse=True)
                            
                            for delivery in deliveries[:10]:  # Show last 10
                                status_color = {
                                    "delivered": "🟢",
                                    "pending": "🟡",
                                    "retrying": "🟠", 
                                    "failed": "🔴"
                                }.get(delivery.status.value, "⚪")
                                
                                st.write(f"{status_color} **{delivery.event}** - {delivery.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                                
                                if delivery.error_message:
                                    st.caption(f"Error: {delivery.error_message}")
                                elif delivery.response_code:
                                    st.caption(f"HTTP {delivery.response_code}")
                        else:
                            st.info("No deliveries yet")
                
                st.markdown("---")
    else:
        st.info("No webhooks configured yet. Create your first webhook above.")


def display_usage_statistics(api_manager):
    """Display API usage statistics and analytics"""
    
    st.subheader("📊 API Usage Statistics")
    st.markdown("Monitor API usage patterns and performance metrics")
    
    # Get statistics
    stats = api_manager.get_api_stats()
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Requests",
            f"{stats['total_requests']:,}",
            delta="All time"
        )
    
    with col2:
        st.metric(
            "Active API Keys",
            stats['active_api_keys'],
            delta=f"of {stats['total_api_keys']} total"
        )
    
    with col3:
        webhook_success_rate = 0
        if stats['total_deliveries'] > 0:
            webhook_success_rate = (stats['successful_deliveries'] / stats['total_deliveries']) * 100
        
        st.metric(
            "Webhook Success Rate",
            f"{webhook_success_rate:.1f}%",
            delta=f"{stats['successful_deliveries']}/{stats['total_deliveries']}"
        )
    
    with col4:
        avg_requests_per_key = 0
        if stats['active_api_keys'] > 0:
            avg_requests_per_key = stats['total_requests'] / stats['active_api_keys']
        
        st.metric(
            "Avg Requests/Key",
            f"{avg_requests_per_key:.1f}",
            delta="per active key"
        )
    
    # API Key Usage Chart
    if api_manager.api_keys:
        st.markdown("#### 📈 API Key Usage Distribution")
        
        key_usage_data = []
        for api_key, data in api_manager.api_keys.items():
            key_usage_data.append({
                "Name": data["name"],
                "Requests": data["request_count"],
                "Last Used": data["last_used"]
            })
        
        df_usage = pd.DataFrame(key_usage_data)
        
        if not df_usage.empty:
            fig_usage = px.bar(
                df_usage,
                x="Name",
                y="Requests",
                title="Requests per API Key",
                color="Requests",
                color_continuous_scale="blues"
            )
            fig_usage.update_layout(height=400)
            st.plotly_chart(fig_usage, use_container_width=True)
    
    # Webhook Delivery Status Chart
    if api_manager.deliveries:
        st.markdown("#### 🔗 Webhook Delivery Status")
        
        status_counts = {}
        for delivery in api_manager.deliveries.values():
            status = delivery.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        if status_counts:
            fig_webhooks = go.Figure(data=[
                go.Pie(
                    labels=list(status_counts.keys()),
                    values=list(status_counts.values()),
                    hole=0.3,
                    marker_colors=['#2ecc71', '#f39c12', '#e74c3c', '#95a5a6']
                )
            ])
            fig_webhooks.update_layout(
                title="Webhook Delivery Status Distribution",
                height=400
            )
            st.plotly_chart(fig_webhooks, use_container_width=True)
    
    # Recent Activity
    st.markdown("#### 🕒 Recent API Activity")
    
    recent_activity = []
    
    # Add API key usage
    for api_key, data in api_manager.api_keys.items():
        if data["last_used"]:
            recent_activity.append({
                "Time": data["last_used"],
                "Type": "API Request",
                "Details": f"Key: {data['name']} ({data['request_count']} total requests)",
                "Status": "Success"
            })
    
    # Add webhook deliveries
    for delivery in list(api_manager.deliveries.values())[-10:]:  # Last 10 deliveries
        webhook = api_manager.webhooks.get(delivery.webhook_id)
        webhook_url = webhook.url if webhook else "Unknown"
        
        recent_activity.append({
            "Time": delivery.created_at.isoformat(),
            "Type": "Webhook Delivery",
            "Details": f"{delivery.event} → {webhook_url}",
            "Status": delivery.status.value.title()
        })
    
    if recent_activity:
        # Sort by time
        recent_activity.sort(key=lambda x: x["Time"], reverse=True)
        
        # Display as table
        df_activity = pd.DataFrame(recent_activity[:20])  # Show last 20 activities
        df_activity["Time"] = pd.to_datetime(df_activity["Time"]).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        st.dataframe(df_activity, use_container_width=True)
    else:
        st.info("No recent API activity")


def display_api_documentation():
    """Display comprehensive API documentation"""
    
    st.subheader("📖 API Documentation")
    st.markdown("Complete reference for the Crawl4AI REST API")
    
    # API Overview
    with st.expander("🌟 API Overview", expanded=True):
        st.markdown("""
        The Crawl4AI Standalone API provides programmatic access to all crawling and search functionality.
        
        **Base URL:** `http://localhost:8000`
        
        **Authentication:** Bearer token (API Key)
        
        **Content Type:** `application/json`
        """)
        
        st.code("""
# Example request with curl
curl -X GET "http://localhost:8000/projects" \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json"
        """, language="bash")
    
    # Endpoints Documentation
    endpoints = [
        {
            "category": "Projects",
            "endpoints": [
                {
                    "method": "GET",
                    "path": "/projects",
                    "description": "List all projects",
                    "parameters": [],
                    "response": "Array of project objects"
                },
                {
                    "method": "POST",
                    "path": "/projects",
                    "description": "Create a new project",
                    "parameters": ["name", "description", "initial_urls"],
                    "response": "Created project object"
                },
                {
                    "method": "GET",
                    "path": "/projects/{id}",
                    "description": "Get project details",
                    "parameters": ["id (path)"],
                    "response": "Project object with statistics"
                },
                {
                    "method": "DELETE",
                    "path": "/projects/{id}",
                    "description": "Delete a project",
                    "parameters": ["id (path)"],
                    "response": "Success message"
                }
            ]
        },
        {
            "category": "Crawling",
            "endpoints": [
                {
                    "method": "POST",
                    "path": "/crawl",
                    "description": "Start a new crawl job",
                    "parameters": ["project_id", "urls", "max_depth", "max_pages", "rag_strategies"],
                    "response": "Crawl job object"
                },
                {
                    "method": "GET",
                    "path": "/crawl/{job_id}",
                    "description": "Get crawl job status",
                    "parameters": ["job_id (path)"],
                    "response": "Crawl job status and results"
                }
            ]
        },
        {
            "category": "Search",
            "endpoints": [
                {
                    "method": "POST",
                    "path": "/search",
                    "description": "Search documents",
                    "parameters": ["project_id", "query", "limit", "similarity_threshold", "filters"],
                    "response": "Search results array"
                }
            ]
        },
        {
            "category": "Webhooks",
            "endpoints": [
                {
                    "method": "GET",
                    "path": "/webhooks",
                    "description": "List all webhooks",
                    "parameters": [],
                    "response": "Array of webhook objects"
                },
                {
                    "method": "POST",
                    "path": "/webhooks",
                    "description": "Create a new webhook",
                    "parameters": ["url", "events", "secret"],
                    "response": "Created webhook object"
                },
                {
                    "method": "DELETE",
                    "path": "/webhooks/{id}",
                    "description": "Delete a webhook",
                    "parameters": ["id (path)"],
                    "response": "Success message"
                },
                {
                    "method": "GET",
                    "path": "/webhooks/{id}/deliveries",
                    "description": "Get webhook delivery history",
                    "parameters": ["id (path)"],
                    "response": "Array of delivery objects"
                }
            ]
        }
    ]
    
    for category_data in endpoints:
        with st.expander(f"📁 {category_data['category']} Endpoints"):
            for endpoint in category_data["endpoints"]:
                method_color = {
                    "GET": "🟢",
                    "POST": "🔵", 
                    "PUT": "🟡",
                    "DELETE": "🔴"
                }.get(endpoint["method"], "⚪")
                
                st.markdown(f"**{method_color} {endpoint['method']} {endpoint['path']}**")
                st.markdown(f"*{endpoint['description']}*")
                
                if endpoint["parameters"]:
                    st.markdown("**Parameters:**")
                    for param in endpoint["parameters"]:
                        st.markdown(f"• `{param}`")
                
                st.markdown(f"**Response:** {endpoint['response']}")
                st.markdown("---")
    
    # Example Code
    with st.expander("💻 Code Examples"):
        
        tab1, tab2, tab3 = st.tabs(["Python", "JavaScript", "cURL"])
        
        with tab1:
            st.markdown("#### Python Example")
            st.code("""
import requests

# Configuration
API_KEY = "your_api_key_here"
BASE_URL = "http://localhost:8000"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Create a project
project_data = {
    "name": "My Project",
    "description": "Test project via API",
    "initial_urls": ["https://example.com"]
}

response = requests.post(
    f"{BASE_URL}/projects",
    json=project_data,
    headers=headers
)

project = response.json()
print(f"Created project: {project['id']}")

# Start a crawl
crawl_data = {
    "project_id": project['id'],
    "urls": ["https://example.com"],
    "max_depth": 2,
    "rag_strategies": ["vector_embeddings"]
}

crawl_response = requests.post(
    f"{BASE_URL}/crawl",
    json=crawl_data,
    headers=headers
)

crawl_job = crawl_response.json()
print(f"Started crawl job: {crawl_job['id']}")

# Search documents
search_data = {
    "project_id": project['id'],
    "query": "python programming",
    "limit": 10
}

search_response = requests.post(
    f"{BASE_URL}/search",
    json=search_data,
    headers=headers
)

results = search_response.json()
print(f"Found {len(results['results'])} results")
            """, language="python")
        
        with tab2:
            st.markdown("#### JavaScript Example")
            st.code("""
const API_KEY = 'your_api_key_here';
const BASE_URL = 'http://localhost:8000';

const headers = {
    'Authorization': `Bearer ${API_KEY}`,
    'Content-Type': 'application/json'
};

// Create a project
async function createProject() {
    const projectData = {
        name: 'My Project',
        description: 'Test project via API',
        initial_urls: ['https://example.com']
    };
    
    const response = await fetch(`${BASE_URL}/projects`, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify(projectData)
    });
    
    const project = await response.json();
    console.log('Created project:', project.id);
    return project;
}

// Start a crawl
async function startCrawl(projectId) {
    const crawlData = {
        project_id: projectId,
        urls: ['https://example.com'],
        max_depth: 2,
        rag_strategies: ['vector_embeddings']
    };
    
    const response = await fetch(`${BASE_URL}/crawl`, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify(crawlData)
    });
    
    const crawlJob = await response.json();
    console.log('Started crawl job:', crawlJob.id);
    return crawlJob;
}

// Search documents
async function searchDocuments(projectId, query) {
    const searchData = {
        project_id: projectId,
        query: query,
        limit: 10
    };
    
    const response = await fetch(`${BASE_URL}/search`, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify(searchData)
    });
    
    const results = await response.json();
    console.log(`Found ${results.results.length} results`);
    return results;
}
            """, language="javascript")
        
        with tab3:
            st.markdown("#### cURL Examples")
            st.code("""
# Create a project
curl -X POST "http://localhost:8000/projects" \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "name": "My Project",
    "description": "Test project via API",
    "initial_urls": ["https://example.com"]
  }'

# Start a crawl
curl -X POST "http://localhost:8000/crawl" \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "project_id": "PROJECT_ID",
    "urls": ["https://example.com"],
    "max_depth": 2,
    "rag_strategies": ["vector_embeddings"]
  }'

# Search documents
curl -X POST "http://localhost:8000/search" \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "project_id": "PROJECT_ID",
    "query": "python programming",
    "limit": 10
  }'

# Create webhook
curl -X POST "http://localhost:8000/webhooks" \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "url": "https://example.com/webhook",
    "events": ["crawl.completed", "crawl.failed"],
    "secret": "my_webhook_secret"
  }'
            """, language="bash")
    
    # Webhook Events Documentation
    with st.expander("🔔 Webhook Events"):
        st.markdown("#### Available Webhook Events")
        
        webhook_events = [
            ("crawl.started", "Triggered when a crawl job begins"),
            ("crawl.completed", "Triggered when a crawl job completes successfully"),
            ("crawl.failed", "Triggered when a crawl job fails"),
            ("document.processed", "Triggered when a document is processed"),
            ("embedding.generated", "Triggered when embeddings are generated"),
            ("search.performed", "Triggered when a search is performed"),
            ("project.created", "Triggered when a project is created"),
            ("project.deleted", "Triggered when a project is deleted"),
            ("alert.triggered", "Triggered when a system alert is raised"),
            ("system.error", "Triggered when a system error occurs")
        ]
        
        for event, description in webhook_events:
            st.markdown(f"• **`{event}`** - {description}")
        
        st.markdown("#### Webhook Payload Format")
        st.code("""
{
  "event": "crawl.completed",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "job_id": "uuid-here",
    "project_id": "project-uuid",
    "status": "completed",
    "documents_processed": 25,
    "completion_time": "2024-01-15T10:30:00Z"
  }
}
        """, language="json")
        
        st.markdown("#### Webhook Security")
        st.markdown("""
        All webhook deliveries include a signature in the `X-Webhook-Signature` header:
        
        ```
        X-Webhook-Signature: sha256=HMAC_SHA256_SIGNATURE
        ```
        
        Verify the signature using your webhook secret to ensure authenticity.
        """)


def display_server_management(api_manager):
    """Display API server management interface"""
    
    st.subheader("🔧 API Server Management")
    st.markdown("Control the REST API server and monitor its status")
    
    # Server Status
    is_running = api_manager.server_thread and api_manager.server_thread.is_alive()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_color = "🟢" if is_running else "🔴"
        status_text = "Running" if is_running else "Stopped"
        st.metric("Server Status", f"{status_color} {status_text}")
    
    with col2:
        st.metric("Server Port", api_manager.server_port)
    
    with col3:
        if is_running:
            st.metric("Server URL", f"http://localhost:{api_manager.server_port}")
    
    st.markdown("---")
    
    # Server Controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if not is_running:
            port = st.number_input("Port", min_value=1000, max_value=65535, value=8000)
            
            if st.button("🚀 Start API Server", type="primary"):
                try:
                    api_manager.start_api_server(port)
                    st.success(f"API server starting on port {port}")
                    st.info("Please wait a moment for the server to initialize...")
                    time.sleep(2)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to start server: {e}")
        else:
            if st.button("🛑 Stop API Server"):
                try:
                    api_manager.stop_api_server()
                    st.success("API server stop requested")
                    st.info("Server may take a moment to fully stop...")
                except Exception as e:
                    st.error(f"Failed to stop server: {e}")
    
    with col2:
        if is_running:
            if st.button("🔄 Restart Server"):
                try:
                    api_manager.stop_api_server()
                    time.sleep(2)
                    api_manager.start_api_server(api_manager.server_port)
                    st.success("Server restarted")
                except Exception as e:
                    st.error(f"Failed to restart server: {e}")
    
    with col3:
        if st.button("🧪 Test Server"):
            if is_running:
                try:
                    import requests
                    response = requests.get(f"http://localhost:{api_manager.server_port}/health", timeout=5)
                    if response.status_code == 200:
                        st.success("✅ Server is responding correctly")
                        st.json(response.json())
                    else:
                        st.warning(f"⚠️ Server responded with status {response.status_code}")
                except Exception as e:
                    st.error(f"❌ Server test failed: {e}")
            else:
                st.error("Server is not running")
    
    # API Documentation Links
    if is_running:
        st.markdown("---")
        st.markdown("### 📖 Interactive Documentation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **Swagger UI**: [http://localhost:{api_manager.server_port}/docs](http://localhost:{api_manager.server_port}/docs)
            
            Interactive API documentation with the ability to test endpoints directly.
            """)
        
        with col2:
            st.markdown(f"""
            **ReDoc**: [http://localhost:{api_manager.server_port}/redoc](http://localhost:{api_manager.server_port}/redoc)
            
            Alternative documentation format with a clean, readable layout.
            """)
    
    # Server Configuration
    with st.expander("⚙️ Server Configuration"):
        st.markdown("#### Current Configuration")
        
        config_info = {
            "Host": "0.0.0.0",
            "Port": api_manager.server_port,
            "CORS Enabled": "Yes",
            "Authentication": "Bearer Token (API Key)",
            "Rate Limiting": "60 requests/minute per API key",
            "Request Timeout": "30 seconds",
            "Max Request Size": "10 MB"
        }
        
        for key, value in config_info.items():
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write(f"**{key}:**")
            with col2:
                st.write(value)
    
    # Server Logs (if available)
    with st.expander("📋 Server Logs"):
        st.markdown("#### Recent Server Activity")
        st.info("Server logs would be displayed here in a production environment")
        
        # Placeholder for server logs
        sample_logs = [
            "2024-01-15 10:30:15 - INFO - API server started on port 8000",
            "2024-01-15 10:30:20 - INFO - Health check endpoint accessed",
            "2024-01-15 10:30:25 - INFO - Project created via API",
            "2024-01-15 10:30:30 - INFO - Crawl job started via API"
        ]
        
        for log in sample_logs:
            st.code(log, language=None)


if __name__ == "__main__":
    show()