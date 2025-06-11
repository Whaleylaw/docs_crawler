"""
Administration Page
Handles system monitoring, configuration, and management tasks
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from components.ui_components import display_metric_card, create_status_indicator

def show():
    """Display the administration page"""
    
    st.header("⚙️ Administration")
    st.markdown("System monitoring, configuration, and management")
    
    # Admin tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 System Monitoring", "🔧 Configuration", "🔑 API Keys", "📈 Analytics"])
    
    with tab1:
        display_system_monitoring()
    
    with tab2:
        display_configuration_management()
    
    with tab3:
        display_api_key_management()
    
    with tab4:
        display_analytics_dashboard()


def display_system_monitoring():
    """Display system monitoring dashboard"""
    
    st.subheader("🖥️ System Status")
    
    # System health indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_status_indicator("System", "Healthy", "green")
        st.metric("Uptime", "99.7%", "0.2%")
    
    with col2:
        create_status_indicator("Database", "Connected", "green")
        st.metric("Connections", "23/100", "3")
    
    with col3:
        create_status_indicator("API", "Operational", "green")
        st.metric("Response Time", "245ms", "-15ms")
    
    with col4:
        create_status_indicator("Storage", "Normal", "yellow")
        st.metric("Usage", "67%", "5%")
    
    # Resource usage
    st.markdown("---")
    st.subheader("📈 Resource Usage")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CPU and Memory usage chart
        hours = list(range(24))
        cpu_data = [45, 52, 38, 41, 55, 67, 73, 68, 62, 58, 64, 71, 
                   78, 82, 75, 69, 72, 76, 68, 61, 58, 52, 47, 43]
        memory_data = [62, 65, 58, 61, 68, 72, 78, 75, 71, 69, 73, 77,
                      82, 85, 79, 74, 76, 79, 73, 67, 64, 61, 58, 56]
        
        fig_resources = go.Figure()
        fig_resources.add_trace(go.Scatter(x=hours, y=cpu_data, name='CPU %', line=dict(color='blue')))
        fig_resources.add_trace(go.Scatter(x=hours, y=memory_data, name='Memory %', line=dict(color='red')))
        fig_resources.update_layout(
            title="CPU & Memory Usage (24h)",
            xaxis_title="Hour",
            yaxis_title="Usage %",
            height=300
        )
        st.plotly_chart(fig_resources, use_container_width=True)
    
    with col2:
        # API calls and errors
        api_calls = [120, 135, 98, 156, 189, 234, 267, 298, 276, 245, 
                    289, 312, 356, 389, 367, 334, 298, 276, 234, 198, 167, 145, 132, 118]
        api_errors = [2, 3, 1, 4, 6, 8, 12, 11, 9, 7, 10, 13, 15, 18, 14, 12, 9, 8, 6, 4, 3, 2, 2, 1]
        
        fig_api = go.Figure()
        fig_api.add_trace(go.Scatter(x=hours, y=api_calls, name='API Calls', line=dict(color='green')))
        fig_api.add_trace(go.Scatter(x=hours, y=api_errors, name='Errors', line=dict(color='orange'), yaxis='y2'))
        fig_api.update_layout(
            title="API Activity (24h)",
            xaxis_title="Hour",
            yaxis_title="API Calls",
            yaxis2=dict(title="Errors", overlaying='y', side='right'),
            height=300
        )
        st.plotly_chart(fig_api, use_container_width=True)
    
    # Performance metrics
    st.markdown("---")
    st.subheader("⚡ Performance Metrics")
    
    perf_data = {
        "Metric": ["Avg Query Response", "Embedding Generation", "Crawl Speed", "Index Update"],
        "Current": ["0.85s", "2.3s", "45 pages/min", "12s"],
        "Target": ["<3s", "<5s", "50 pages/min", "<15s"],
        "Status": ["✅ Good", "✅ Good", "⚠️ Below Target", "✅ Good"]
    }
    
    df_perf = pd.DataFrame(perf_data)
    st.dataframe(df_perf, use_container_width=True, hide_index=True)
    
    # Error logs
    st.markdown("---")
    st.subheader("🐛 Recent Errors")
    
    error_data = [
        {"timestamp": "2025-01-20 14:30", "level": "WARNING", "component": "Crawler", "message": "Rate limit approaching for docs.example.com"},
        {"timestamp": "2025-01-20 12:15", "level": "ERROR", "component": "Database", "message": "Connection timeout after 30s"},
        {"timestamp": "2025-01-20 09:45", "level": "INFO", "component": "API", "message": "Cache cleared successfully"},
    ]
    
    for error in error_data:
        level_color = {"ERROR": "red", "WARNING": "orange", "INFO": "blue"}[error["level"]]
        st.markdown(f"**{error['timestamp']}** | "
                   f"<span style='color: {level_color}; font-weight: bold;'>{error['level']}</span> | "
                   f"**{error['component']}**: {error['message']}", unsafe_allow_html=True)


def display_configuration_management():
    """Display configuration management interface"""
    
    st.subheader("🔧 System Configuration")
    
    # Model selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### AI Models")
        
        primary_model = st.selectbox(
            "Primary LLM Model",
            ["claude-3-sonnet", "gpt-4", "gpt-3.5-turbo", "gemini-pro"],
            help="Main model for text processing and summarization"
        )
        
        embedding_model = st.selectbox(
            "Embedding Model",
            ["text-embedding-ada-002", "sentence-transformers/all-MiniLM-L6-v2", "local-embedding-model"],
            help="Model for generating text embeddings"
        )
        
        research_model = st.selectbox(
            "Research Model",
            ["perplexity-sonar", "claude-3-sonnet", "gpt-4"],
            help="Model for research-backed operations"
        )
    
    with col2:
        st.markdown("#### Processing Settings")
        
        max_tokens = st.slider("Max Tokens", 1000, 10000, 4000, step=500)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 100, step=50)
        
        enable_caching = st.checkbox("Enable Result Caching", value=True)
        enable_monitoring = st.checkbox("Enable Performance Monitoring", value=True)
    
    # Database configuration
    st.markdown("---")
    st.markdown("#### Database Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        db_pool_size = st.slider("Connection Pool Size", 5, 50, 20)
        query_timeout = st.slider("Query Timeout (seconds)", 10, 60, 30)
    
    with col2:
        auto_vacuum = st.checkbox("Auto Vacuum", value=True)
        backup_enabled = st.checkbox("Automatic Backups", value=True)
        backup_frequency = st.selectbox("Backup Frequency", ["Daily", "Weekly", "Monthly"], index=0)
    
    # Save configuration
    if st.button("💾 Save Configuration", type="primary"):
        st.success("✅ Configuration saved successfully!")
        st.info("Changes will take effect after restart")


def display_api_key_management():
    """Display API key management interface"""
    
    st.subheader("🔑 API Key Management")
    
    # Current API keys status
    api_keys_status = [
        {"service": "OpenAI", "status": "Active", "usage": "75%", "expires": "2025-06-15"},
        {"service": "Anthropic", "status": "Active", "usage": "45%", "expires": "2025-04-20"},
        {"service": "Perplexity", "status": "Warning", "usage": "95%", "expires": "2025-03-10"},
        {"service": "Supabase", "status": "Active", "usage": "30%", "expires": "Never"},
    ]
    
    for key_info in api_keys_status:
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
            
            with col1:
                st.write(f"**{key_info['service']}**")
            
            with col2:
                status_color = {"Active": "green", "Warning": "orange", "Expired": "red"}[key_info['status']]
                st.markdown(f"<span style='color: {status_color};'>●</span> {key_info['status']}", unsafe_allow_html=True)
            
            with col3:
                st.write(key_info['usage'])
            
            with col4:
                st.write(key_info['expires'])
            
            with col5:
                if st.button("🔄 Rotate", key=f"rotate_{key_info['service']}"):
                    st.info(f"Rotating {key_info['service']} API key...")
            
            st.markdown("---")
    
    # Add new API key
    st.markdown("#### Add New API Key")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        new_service = st.selectbox("Service", ["OpenAI", "Anthropic", "Perplexity", "Google", "Custom"])
        new_key = st.text_input("API Key", type="password", placeholder="Enter your API key")
    
    with col2:
        st.markdown("#### Actions")
        if st.button("✅ Add Key", type="primary"):
            if new_key:
                st.success(f"✅ {new_service} API key added successfully!")
            else:
                st.error("Please enter a valid API key")
        
        if st.button("🧪 Test All Keys"):
            st.info("Testing all API keys... This may take a moment.")


def display_analytics_dashboard():
    """Display analytics and usage dashboard"""
    
    st.subheader("📈 Usage Analytics")
    
    # Date range selector
    col1, col2 = st.columns([3, 1])
    
    with col1:
        date_range = st.selectbox(
            "Time Period",
            ["Last 7 days", "Last 30 days", "Last 90 days", "Custom"],
            index=1
        )
    
    with col2:
        if st.button("📥 Export Report"):
            st.success("Analytics report exported!")
    
    # Usage metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Queries", "12,847", "2,156")
    with col2:
        st.metric("Pages Crawled", "45,623", "8,934")
    with col3:
        st.metric("Storage Used", "234 GB", "45 GB")
    with col4:
        st.metric("API Costs", "$127.45", "$23.12")
    
    # Usage trends
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        # Daily usage chart
        days = pd.date_range(start='2025-01-01', end='2025-01-20', freq='D')
        queries = [120, 145, 98, 167, 234, 189, 156, 278, 312, 298, 
                  267, 234, 198, 176, 145, 167, 189, 234, 267, 289]
        
        fig_usage = px.line(x=days, y=queries, title="Daily Query Volume")
        fig_usage.update_layout(height=300)
        st.plotly_chart(fig_usage, use_container_width=True)
    
    with col2:
        # Project usage distribution
        project_data = {"Project": ["Documentation Site", "API Reference", "Blog Content"], 
                       "Queries": [5234, 4567, 2046]}
        fig_projects = px.pie(values=project_data["Queries"], names=project_data["Project"], 
                             title="Queries by Project")
        fig_projects.update_layout(height=300)
        st.plotly_chart(fig_projects, use_container_width=True)
    
    # Top searches
    st.markdown("---")
    st.subheader("🔍 Top Search Queries")
    
    top_queries = [
        {"query": "authentication JWT", "count": 234, "project": "Documentation Site"},
        {"query": "API endpoints", "count": 189, "project": "API Reference"},
        {"query": "database connection", "count": 156, "project": "Documentation Site"},
        {"query": "error handling", "count": 134, "project": "All Projects"},
        {"query": "deployment guide", "count": 98, "project": "Documentation Site"},
    ]
    
    df_queries = pd.DataFrame(top_queries)
    st.dataframe(df_queries, use_container_width=True, hide_index=True)


def create_status_indicator(title: str, status: str, color: str):
    """Create a status indicator with title and colored status"""
    st.markdown(f"**{title}**")
    st.markdown(f"<span style='color: {color}; font-size: 14px;'>● {status}</span>", unsafe_allow_html=True) 