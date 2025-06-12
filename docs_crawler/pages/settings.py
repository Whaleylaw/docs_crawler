"""
Settings and Configuration Page
Provides interface for managing application configuration
"""

import streamlit as st
import json
from typing import Dict, Any
from components.configuration import (
    get_config_manager, get_session_config, update_session_config,
    ConfigCategory, ApplicationConfig, OpenAIConfig, SupabaseConfig,
    CrawlingConfig, RAGConfig, UIConfig, PerformanceConfig,
    LoggingConfig, SecurityConfig
)


def show():
    """Display the settings page"""
    
    st.header("⚙️ Settings & Configuration")
    st.markdown("Manage application settings, API keys, and system preferences")
    
    # Get configuration manager and current config
    config_manager = get_config_manager()
    current_config = get_session_config()
    
    # Validate current configuration
    validation_issues = config_manager.validate_config()
    
    if validation_issues:
        st.error("⚠️ Configuration Issues Found:")
        for issue in validation_issues:
            st.write(f"• {issue}")
        st.markdown("---")
    
    # Configuration tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔑 API Keys",
        "🕷️ Crawling", 
        "🧠 AI & RAG",
        "🎨 Interface",
        "⚡ Performance",
        "🔧 Advanced"
    ])
    
    with tab1:
        display_api_configuration(current_config)
    
    with tab2:
        display_crawling_configuration(current_config)
    
    with tab3:
        display_rag_configuration(current_config)
    
    with tab4:
        display_ui_configuration(current_config)
    
    with tab5:
        display_performance_configuration(current_config)
    
    with tab6:
        display_advanced_configuration(current_config, config_manager)
    
    # Global actions
    st.markdown("---")
    display_global_actions(current_config, config_manager)


def display_api_configuration(config: ApplicationConfig):
    """Display API configuration section"""
    
    st.subheader("🔑 API Keys & Authentication")
    st.markdown("Configure external service credentials")
    
    # OpenAI Configuration
    with st.expander("🤖 OpenAI Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # API Key with masked display
            api_key = st.text_input(
                "API Key",
                value=config.openai.api_key,
                type="password",
                help="Your OpenAI API key"
            )
            
            embedding_model = st.selectbox(
                "Embedding Model",
                ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
                index=["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"].index(config.openai.embedding_model)
            )
            
            chat_model = st.selectbox(
                "Chat Model",
                ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
                index=["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "gpt-4-turbo"].index(config.openai.chat_model) if config.openai.chat_model in ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "gpt-4-turbo"] else 0
            )
        
        with col2:
            max_tokens = st.number_input(
                "Max Tokens",
                min_value=100,
                max_value=4000,
                value=config.openai.max_tokens,
                help="Maximum tokens for AI responses"
            )
            
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=config.openai.temperature,
                step=0.1,
                help="Controls randomness in AI responses"
            )
            
            timeout = st.number_input(
                "Timeout (seconds)",
                min_value=5,
                max_value=120,
                value=config.openai.timeout,
                help="Request timeout for OpenAI API calls"
            )
        
        # Update OpenAI config
        config.openai.api_key = api_key
        config.openai.embedding_model = embedding_model
        config.openai.chat_model = chat_model
        config.openai.max_tokens = max_tokens
        config.openai.temperature = temperature
        config.openai.timeout = timeout
        
        # Test API connection
        if st.button("🧪 Test OpenAI Connection"):
            test_openai_connection(config.openai)
    
    # Supabase Configuration
    with st.expander("🗄️ Supabase Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            supabase_url = st.text_input(
                "Supabase URL",
                value=config.supabase.url,
                help="Your Supabase project URL"
            )
            
            service_key = st.text_input(
                "Service Role Key",
                value=config.supabase.service_key,
                type="password",
                help="Supabase service role key (admin access)"
            )
        
        with col2:
            anon_key = st.text_input(
                "Anonymous Key",
                value=config.supabase.anon_key,
                help="Supabase anonymous key (public access)"
            )
            
            max_connections = st.number_input(
                "Max Connections",
                min_value=1,
                max_value=50,
                value=config.supabase.max_connections,
                help="Maximum database connections"
            )
        
        # Update Supabase config
        config.supabase.url = supabase_url
        config.supabase.service_key = service_key
        config.supabase.anon_key = anon_key
        config.supabase.max_connections = max_connections
        
        # Test Supabase connection
        if st.button("🧪 Test Supabase Connection"):
            test_supabase_connection(config.supabase)


def display_crawling_configuration(config: ApplicationConfig):
    """Display crawling configuration section"""
    
    st.subheader("🕷️ Web Crawling Settings")
    st.markdown("Configure web crawling behavior and performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Performance")
        
        max_concurrent = st.slider(
            "Max Concurrent Crawls",
            min_value=1,
            max_value=50,
            value=config.crawling.max_concurrent,
            help="Number of parallel crawling sessions"
        )
        
        max_depth = st.slider(
            "Max Crawl Depth",
            min_value=1,
            max_value=10,
            value=config.crawling.max_depth,
            help="Maximum recursion depth for link following"
        )
        
        request_timeout = st.number_input(
            "Request Timeout (seconds)",
            min_value=5,
            max_value=120,
            value=config.crawling.request_timeout,
            help="Timeout for individual web requests"
        )
        
        delay_between_requests = st.number_input(
            "Delay Between Requests (seconds)",
            min_value=0.0,
            max_value=10.0,
            value=config.crawling.delay_between_requests,
            step=0.1,
            help="Delay to be respectful to target servers"
        )
    
    with col2:
        st.markdown("#### Content Processing")
        
        chunk_size = st.number_input(
            "Chunk Size (characters)",
            min_value=500,
            max_value=20000,
            value=config.crawling.chunk_size,
            step=500,
            help="Size of text chunks for processing"
        )
        
        chunk_overlap = st.number_input(
            "Chunk Overlap (characters)",
            min_value=0,
            max_value=1000,
            value=config.crawling.chunk_overlap,
            step=50,
            help="Overlap between consecutive chunks"
        )
        
        max_file_size_mb = st.number_input(
            "Max File Size (MB)",
            min_value=1,
            max_value=500,
            value=config.crawling.max_file_size_mb,
            help="Maximum file size to crawl"
        )
        
        respect_robots_txt = st.checkbox(
            "Respect robots.txt",
            value=config.crawling.respect_robots_txt,
            help="Follow robots.txt directives"
        )
    
    with col3:
        st.markdown("#### Headers & User Agent")
        
        user_agent = st.text_input(
            "User Agent",
            value=config.crawling.user_agent,
            help="User agent string for web requests"
        )
        
        st.markdown("**Custom Headers**")
        headers_text = st.text_area(
            "Custom Headers (JSON format)",
            value=json.dumps(config.crawling.default_headers, indent=2),
            height=100,
            help="Custom HTTP headers as JSON"
        )
        
        # Domain filters
        st.markdown("**Domain Filters**")
        allowed_domains = st.text_area(
            "Allowed Domains (one per line)",
            value="\n".join(config.crawling.allowed_domains),
            height=60,
            help="Only crawl these domains (empty = all allowed)"
        )
        
        blocked_domains = st.text_area(
            "Blocked Domains (one per line)",
            value="\n".join(config.crawling.blocked_domains),
            height=60,
            help="Never crawl these domains"
        )
    
    # Update crawling config
    config.crawling.max_concurrent = max_concurrent
    config.crawling.max_depth = max_depth
    config.crawling.request_timeout = request_timeout
    config.crawling.delay_between_requests = delay_between_requests
    config.crawling.chunk_size = chunk_size
    config.crawling.chunk_overlap = chunk_overlap
    config.crawling.max_file_size_mb = max_file_size_mb
    config.crawling.respect_robots_txt = respect_robots_txt
    config.crawling.user_agent = user_agent
    
    # Parse headers JSON
    try:
        config.crawling.default_headers = json.loads(headers_text)
    except json.JSONDecodeError:
        st.error("Invalid JSON format for custom headers")
    
    # Update domain filters
    config.crawling.allowed_domains = [d.strip() for d in allowed_domains.split('\n') if d.strip()]
    config.crawling.blocked_domains = [d.strip() for d in blocked_domains.split('\n') if d.strip()]


def display_rag_configuration(config: ApplicationConfig):
    """Display RAG configuration section"""
    
    st.subheader("🧠 AI & RAG Settings")
    st.markdown("Configure AI-powered content processing strategies")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### RAG Strategies")
        
        use_contextual_embeddings = st.checkbox(
            "Contextual Embeddings",
            value=config.rag.use_contextual_embeddings,
            help="Use AI to enhance chunks with document context"
        )
        
        extract_code_examples = st.checkbox(
            "Code Example Extraction",
            value=config.rag.extract_code_examples,
            help="Extract and analyze code blocks with AI summaries"
        )
        
        use_reranking = st.checkbox(
            "Cross-encoder Reranking",
            value=config.rag.use_reranking,
            help="Re-rank search results for better relevance"
        )
        
        use_hybrid_search = st.checkbox(
            "Hybrid Search (Beta)",
            value=config.rag.use_hybrid_search,
            help="Combine vector and keyword search",
            disabled=True  # Not fully implemented
        )
        
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=config.rag.similarity_threshold,
            step=0.05,
            help="Minimum similarity for search results"
        )
        
        max_results = st.number_input(
            "Max Search Results",
            min_value=5,
            max_value=100,
            value=config.rag.max_results,
            help="Maximum number of search results"
        )
    
    with col2:
        st.markdown("#### Processing Settings")
        
        context_model = st.selectbox(
            "Context Generation Model",
            ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
            index=["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"].index(config.rag.context_model) if config.rag.context_model in ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"] else 0,
            help="Model for generating contextual embeddings"
        )
        
        reranking_model = st.selectbox(
            "Reranking Model",
            [
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "cross-encoder/ms-marco-TinyBERT-L-2-v2",
                "cross-encoder/ms-marco-MiniLM-L-12-v2"
            ],
            index=0,
            help="Cross-encoder model for result reranking"
        )
        
        min_code_length = st.number_input(
            "Min Code Block Length",
            min_value=100,
            max_value=5000,
            value=config.rag.min_code_length,
            help="Minimum characters for code extraction"
        )
        
        parallel_workers = st.slider(
            "Parallel Workers",
            min_value=1,
            max_value=20,
            value=config.rag.parallel_workers,
            help="Number of parallel workers for AI processing"
        )
        
        batch_size = st.slider(
            "Batch Size",
            min_value=5,
            max_value=50,
            value=config.rag.batch_size,
            help="Size of processing batches"
        )
    
    # Update RAG config
    strategies = ["vector_embeddings"]  # Always enabled
    if use_contextual_embeddings:
        strategies.append("contextual_embeddings")
    if extract_code_examples:
        strategies.append("agentic_rag")
    if use_reranking:
        strategies.append("cross_encoder_reranking")
    if use_hybrid_search:
        strategies.append("hybrid_search")
    
    config.rag.enabled_strategies = strategies
    config.rag.use_contextual_embeddings = use_contextual_embeddings
    config.rag.extract_code_examples = extract_code_examples
    config.rag.use_reranking = use_reranking
    config.rag.use_hybrid_search = use_hybrid_search
    config.rag.similarity_threshold = similarity_threshold
    config.rag.max_results = max_results
    config.rag.context_model = context_model
    config.rag.reranking_model = reranking_model
    config.rag.min_code_length = min_code_length
    config.rag.parallel_workers = parallel_workers
    config.rag.batch_size = batch_size


def display_ui_configuration(config: ApplicationConfig):
    """Display UI configuration section"""
    
    st.subheader("🎨 User Interface Settings")
    st.markdown("Customize the application appearance and behavior")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Appearance")
        
        theme = st.selectbox(
            "Theme",
            ["light", "dark", "auto"],
            index=["light", "dark", "auto"].index(config.ui.theme),
            help="Application color theme"
        )
        
        page_title = st.text_input(
            "Page Title",
            value=config.ui.page_title,
            help="Browser tab title"
        )
        
        show_sidebar = st.checkbox(
            "Show Sidebar",
            value=config.ui.show_sidebar,
            help="Display navigation sidebar"
        )
        
        compact_mode = st.checkbox(
            "Compact Mode",
            value=config.ui.compact_mode,
            help="Use compact UI elements"
        )
        
        enable_tooltips = st.checkbox(
            "Enable Tooltips",
            value=config.ui.enable_tooltips,
            help="Show helpful tooltips"
        )
    
    with col2:
        st.markdown("#### Behavior")
        
        default_page = st.selectbox(
            "Default Page",
            ["home", "project_management", "crawl_content", "search_interface", "content_analysis"],
            index=["home", "project_management", "crawl_content", "search_interface", "content_analysis"].index(config.ui.default_page) if config.ui.default_page in ["home", "project_management", "crawl_content", "search_interface", "content_analysis"] else 0,
            help="Page to show on startup"
        )
        
        items_per_page = st.number_input(
            "Items Per Page",
            min_value=5,
            max_value=100,
            value=config.ui.items_per_page,
            help="Number of items to show per page"
        )
        
        show_progress_bars = st.checkbox(
            "Show Progress Bars",
            value=config.ui.show_progress_bars,
            help="Display progress indicators"
        )
        
        auto_refresh_interval = st.number_input(
            "Auto Refresh Interval (seconds)",
            min_value=10,
            max_value=300,
            value=config.ui.auto_refresh_interval,
            help="How often to refresh data automatically"
        )
    
    # Update UI config
    config.ui.theme = theme
    config.ui.page_title = page_title
    config.ui.show_sidebar = show_sidebar
    config.ui.compact_mode = compact_mode
    config.ui.enable_tooltips = enable_tooltips
    config.ui.default_page = default_page
    config.ui.items_per_page = items_per_page
    config.ui.show_progress_bars = show_progress_bars
    config.ui.auto_refresh_interval = auto_refresh_interval


def display_performance_configuration(config: ApplicationConfig):
    """Display performance configuration section"""
    
    st.subheader("⚡ Performance & Optimization")
    st.markdown("Configure performance settings and resource limits")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Caching & Memory")
        
        enable_caching = st.checkbox(
            "Enable Caching",
            value=config.performance.enable_caching,
            help="Cache frequently accessed data"
        )
        
        cache_ttl_minutes = st.number_input(
            "Cache TTL (minutes)",
            min_value=5,
            max_value=1440,
            value=config.performance.cache_ttl_minutes,
            help="How long to keep cached data"
        )
        
        max_memory_mb = st.number_input(
            "Max Memory Usage (MB)",
            min_value=256,
            max_value=8192,
            value=config.performance.max_memory_mb,
            help="Maximum memory usage limit"
        )
        
        enable_compression = st.checkbox(
            "Enable Compression",
            value=config.performance.enable_compression,
            help="Compress stored data"
        )
    
    with col2:
        st.markdown("#### Processing")
        
        async_processing = st.checkbox(
            "Async Processing",
            value=config.performance.async_processing,
            help="Use asynchronous processing"
        )
        
        connection_pool_size = st.number_input(
            "Connection Pool Size",
            min_value=5,
            max_value=100,
            value=config.performance.connection_pool_size,
            help="Database connection pool size"
        )
        
        query_timeout = st.number_input(
            "Query Timeout (seconds)",
            min_value=10,
            max_value=300,
            value=config.performance.query_timeout,
            help="Database query timeout"
        )
        
        enable_metrics = st.checkbox(
            "Enable Metrics",
            value=config.performance.enable_metrics,
            help="Collect performance metrics"
        )
    
    # Update performance config
    config.performance.enable_caching = enable_caching
    config.performance.cache_ttl_minutes = cache_ttl_minutes
    config.performance.max_memory_mb = max_memory_mb
    config.performance.enable_compression = enable_compression
    config.performance.async_processing = async_processing
    config.performance.connection_pool_size = connection_pool_size
    config.performance.query_timeout = query_timeout
    config.performance.enable_metrics = enable_metrics


def display_advanced_configuration(config: ApplicationConfig, config_manager):
    """Display advanced configuration section"""
    
    st.subheader("🔧 Advanced Settings")
    st.markdown("Logging, security, and system configuration")
    
    # Logging Configuration
    with st.expander("📋 Logging Configuration", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            log_level = st.selectbox(
                "Log Level",
                ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                index=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"].index(config.logging.level),
                help="Minimum log level to record"
            )
            
            enable_file_logging = st.checkbox(
                "Enable File Logging",
                value=config.logging.enable_file_logging,
                help="Write logs to file"
            )
            
            log_file_path = st.text_input(
                "Log File Path",
                value=config.logging.log_file_path,
                help="Path to log file"
            )
        
        with col2:
            max_log_size_mb = st.number_input(
                "Max Log File Size (MB)",
                min_value=1,
                max_value=1000,
                value=config.logging.max_log_size_mb,
                help="Maximum size of log files"
            )
            
            log_retention_days = st.number_input(
                "Log Retention (days)",
                min_value=1,
                max_value=365,
                value=config.logging.log_retention_days,
                help="How long to keep log files"
            )
            
            enable_performance_logging = st.checkbox(
                "Performance Logging",
                value=config.logging.enable_performance_logging,
                help="Log performance metrics"
            )
        
        # Update logging config
        config.logging.level = log_level
        config.logging.enable_file_logging = enable_file_logging
        config.logging.log_file_path = log_file_path
        config.logging.max_log_size_mb = max_log_size_mb
        config.logging.log_retention_days = log_retention_days
        config.logging.enable_performance_logging = enable_performance_logging
    
    # Security Configuration
    with st.expander("🔒 Security Configuration", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            encrypt_credentials = st.checkbox(
                "Encrypt Credentials",
                value=config.security.encrypt_credentials,
                help="Encrypt stored credentials"
            )
            
            session_timeout_minutes = st.number_input(
                "Session Timeout (minutes)",
                min_value=30,
                max_value=1440,
                value=config.security.session_timeout_minutes,
                help="User session timeout"
            )
            
            enable_csrf_protection = st.checkbox(
                "CSRF Protection",
                value=config.security.enable_csrf_protection,
                help="Enable CSRF protection"
            )
        
        with col2:
            enable_rate_limiting = st.checkbox(
                "Rate Limiting",
                value=config.security.enable_rate_limiting,
                help="Limit request rate"
            )
            
            rate_limit_requests_per_minute = st.number_input(
                "Rate Limit (requests/minute)",
                min_value=10,
                max_value=1000,
                value=config.security.rate_limit_requests_per_minute,
                help="Maximum requests per minute"
            )
            
            enable_audit_logging = st.checkbox(
                "Audit Logging",
                value=config.security.enable_audit_logging,
                help="Log security events"
            )
        
        # Update security config
        config.security.encrypt_credentials = encrypt_credentials
        config.security.session_timeout_minutes = session_timeout_minutes
        config.security.enable_csrf_protection = enable_csrf_protection
        config.security.enable_rate_limiting = enable_rate_limiting
        config.security.rate_limit_requests_per_minute = rate_limit_requests_per_minute
        config.security.enable_audit_logging = enable_audit_logging
    
    # Environment Information
    with st.expander("📊 Environment Information", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Version", config.version)
            st.metric("Environment", config.environment)
            st.metric("Last Updated", config.last_updated[:19])
        
        with col2:
            # Configuration file info
            st.write(f"**Config File:** {config_manager.config_file}")
            st.write(f"**File Exists:** {config_manager.config_file.exists()}")
            if config_manager.config_file.exists():
                st.write(f"**File Size:** {config_manager.config_file.stat().st_size} bytes")


def display_global_actions(config: ApplicationConfig, config_manager):
    """Display global configuration actions"""
    
    st.subheader("💾 Configuration Management")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("💾 Save Configuration", type="primary", use_container_width=True):
            update_session_config(config)
            if config_manager.save_config():
                st.success("✅ Configuration saved successfully!")
            else:
                st.error("❌ Failed to save configuration")
    
    with col2:
        if st.button("🔄 Reset to Defaults", use_container_width=True):
            if st.session_state.get('confirm_reset', False):
                config_manager.reset_to_defaults()
                update_session_config(config_manager.get_config())
                st.success("✅ Configuration reset to defaults")
                st.session_state.confirm_reset = False
                st.rerun()
            else:
                st.session_state.confirm_reset = True
                st.warning("⚠️ Click again to confirm reset")
    
    with col3:
        if st.button("🔍 Validate Config", use_container_width=True):
            issues = config_manager.validate_config()
            if issues:
                st.error(f"❌ Found {len(issues)} validation issues")
                for issue in issues:
                    st.write(f"• {issue}")
            else:
                st.success("✅ Configuration is valid")
    
    with col4:
        if st.button("📤 Export Config", use_container_width=True):
            export_configuration(config_manager)
    
    with col5:
        if st.button("📥 Import Config", use_container_width=True):
            import_configuration(config_manager)
    
    # Show current config summary
    st.markdown("---")
    st.subheader("📋 Configuration Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("OpenAI Model", config.openai.embedding_model)
        st.metric("Chat Model", config.openai.chat_model)
    
    with col2:
        st.metric("Max Concurrent", config.crawling.max_concurrent)
        st.metric("Chunk Size", f"{config.crawling.chunk_size:,}")
    
    with col3:
        st.metric("RAG Strategies", len(config.rag.enabled_strategies))
        st.metric("Cache Enabled", "Yes" if config.performance.enable_caching else "No")
    
    with col4:
        st.metric("Log Level", config.logging.level)
        st.metric("Theme", config.ui.theme.title())


def test_openai_connection(openai_config: OpenAIConfig):
    """Test OpenAI API connection"""
    
    with st.spinner("Testing OpenAI connection..."):
        try:
            import openai
            
            # Set API key
            openai.api_key = openai_config.api_key
            
            # Test with a simple embedding request
            response = openai.embeddings.create(
                model=openai_config.embedding_model,
                input="test connection"
            )
            
            if response.data:
                st.success("✅ OpenAI connection successful!")
                st.info(f"Model: {openai_config.embedding_model}")
            else:
                st.error("❌ OpenAI connection failed")
                
        except Exception as e:
            st.error(f"❌ OpenAI connection failed: {e}")


def test_supabase_connection(supabase_config: SupabaseConfig):
    """Test Supabase connection"""
    
    with st.spinner("Testing Supabase connection..."):
        try:
            from supabase import create_client
            
            client = create_client(supabase_config.url, supabase_config.service_key)
            
            # Test with a simple query
            response = client.table('projects').select('*').limit(1).execute()
            
            st.success("✅ Supabase connection successful!")
            st.info(f"URL: {supabase_config.url}")
            
        except Exception as e:
            st.error(f"❌ Supabase connection failed: {e}")


def export_configuration(config_manager):
    """Export configuration to file"""
    
    st.markdown("### 📤 Export Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        export_format = st.selectbox("Format", ["JSON", "YAML"])
    
    with col2:
        include_secrets = st.checkbox("Include API Keys", value=False, help="Include sensitive information")
    
    try:
        config_str = config_manager.export_config(export_format.lower())
        
        if not include_secrets:
            # Remove sensitive information
            import json
            config_dict = json.loads(config_str) if export_format == "JSON" else config_str
            if isinstance(config_dict, dict):
                config_dict.get('openai', {}).pop('api_key', None)
                config_dict.get('supabase', {}).pop('service_key', None)
                config_dict.get('supabase', {}).pop('anon_key', None)
            
            if export_format == "JSON":
                config_str = json.dumps(config_dict, indent=2)
        
        st.download_button(
            label=f"📥 Download {export_format} Config",
            data=config_str,
            file_name=f"crawl4ai_config.{export_format.lower()}",
            mime="application/json" if export_format == "JSON" else "text/yaml"
        )
        
        # Show preview
        with st.expander("Preview Configuration"):
            st.code(config_str, language=export_format.lower())
            
    except Exception as e:
        st.error(f"Export failed: {e}")


def import_configuration(config_manager):
    """Import configuration from file"""
    
    st.markdown("### 📥 Import Configuration")
    
    uploaded_file = st.file_uploader(
        "Choose configuration file",
        type=['json', 'yaml', 'yml'],
        help="Upload a configuration file to import settings"
    )
    
    if uploaded_file:
        try:
            content = uploaded_file.read().decode('utf-8')
            file_extension = uploaded_file.name.split('.')[-1].lower()
            format_type = "json" if file_extension == "json" else "yaml"
            
            # Show preview
            st.code(content, language=format_type)
            
            if st.button("Import Configuration"):
                if config_manager.import_config(content, format_type):
                    update_session_config(config_manager.get_config())
                    st.success("✅ Configuration imported successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to import configuration")
                    
        except Exception as e:
            st.error(f"Import failed: {e}")


if __name__ == "__main__":
    show()