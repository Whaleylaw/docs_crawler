"""
Crawl Content Page
Handles web crawling configuration and execution
"""

import streamlit as st
from datetime import datetime
import time
from components.ui_components import create_progress_indicator, display_crawl_status

def show():
    """Display the crawl content page"""
    
    st.header("🕷️ Crawl Content")
    st.markdown("Configure and execute web crawling operations")
    
    # Project selection
    project_options = ["Documentation Site", "API Reference", "Create New Project"]
    selected_project = st.selectbox(
        "Select Project",
        project_options,
        help="Choose an existing project or create a new one"
    )
    
    if selected_project == "Create New Project":
        st.info("👆 Please go to Project Management to create a new project first")
        return
    
    # URL input section
    st.markdown("---")
    st.subheader("🌐 URL Configuration")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        url_input_method = st.radio(
            "URL Input Method",
            ["Single URL", "Multiple URLs", "Upload URL List"],
            horizontal=True
        )
        
        if url_input_method == "Single URL":
            crawl_url = st.text_input(
                "URL to Crawl",
                placeholder="https://example.com/docs",
                help="Enter a single URL to crawl"
            )
        elif url_input_method == "Multiple URLs":
            crawl_urls = st.text_area(
                "URLs to Crawl (one per line)",
                placeholder="https://example.com/docs\nhttps://example.com/api\nhttps://example.com/guide",
                height=150,
                help="Enter multiple URLs, one per line"
            )
        else:  # Upload URL List
            uploaded_file = st.file_uploader(
                "Upload URL List",
                type=['txt', 'csv'],
                help="Upload a text file with URLs (one per line) or CSV with URL column"
            )
    
    with col2:
        st.markdown("### Smart Detection")
        st.info("🔍 Auto-detect:\n- Sitemaps\n- Text files\n- Regular webpages")
        
        detect_sitemaps = st.checkbox("Auto-detect Sitemaps", value=True)
        detect_text_files = st.checkbox("Auto-detect Text Files", value=True)
    
    # Crawling configuration
    st.markdown("---")
    st.subheader("⚙️ Crawling Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Basic Settings")
        max_depth = st.slider("Maximum Depth", 1, 5, 2, help="How many levels deep to crawl")
        max_concurrent = st.slider("Concurrent Sessions", 5, 50, 10, help="Number of parallel crawling sessions")
        chunk_size = st.slider("Chunk Size", 1000, 10000, 4000, step=500, help="Size of text chunks for processing")
    
    with col2:
        st.markdown("#### Filtering")
        include_patterns = st.text_area(
            "Include Patterns (regex)",
            placeholder=".*\\.html$\n.*\\/docs\\/.*",
            height=100,
            help="Regex patterns for URLs to include (one per line)"
        )
        
        exclude_patterns = st.text_area(
            "Exclude Patterns (regex)", 
            placeholder=".*\\.pdf$\n.*\\/private\\/.*",
            height=100,
            help="Regex patterns for URLs to exclude (one per line)"
        )
    
    with col3:
        st.markdown("#### RAG Strategy")
        rag_strategies = st.multiselect(
            "Select RAG Strategies",
            [
                "Contextual Embeddings (slower, higher accuracy)",
                "Hybrid Search (vector + keyword)",
                "Agentic RAG (extracts code examples)",
                "Cross-encoder Reranking (improves relevance)"
            ],
            default=["Contextual Embeddings (slower, higher accuracy)"],
            help="Choose processing strategies for content"
        )
        
        preview_mode = st.checkbox(
            "Preview Mode",
            help="Show sample chunks before full processing"
        )
    
    # Crawl execution
    st.markdown("---")
    st.subheader("🚀 Execute Crawl")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.button("🕷️ Start Crawling", type="primary", use_container_width=True):
            execute_crawl()
    
    with col2:
        if st.button("⏸️ Pause Crawl", use_container_width=True):
            st.info("Crawl paused")
    
    with col3:
        if st.button("🛑 Stop Crawl", use_container_width=True):
            st.warning("Crawl stopped")
    
    # Progress monitoring
    if st.session_state.get('crawling_active', False):
        st.markdown("---")
        st.subheader("📊 Crawl Progress")
        display_crawl_progress()
    
    # Recent crawl history
    st.markdown("---")
    st.subheader("📚 Crawl History")
    display_crawl_history()


def execute_crawl():
    """Execute the crawling operation"""
    st.session_state.crawling_active = True
    
    with st.spinner("Initializing crawl..."):
        time.sleep(2)  # Simulate initialization
    
    st.success("✅ Crawl started successfully!")
    st.rerun()


def display_crawl_progress():
    """Display real-time crawling progress"""
    
    # Mock progress data
    progress_data = {
        'urls_discovered': 150,
        'urls_crawled': 89,
        'urls_failed': 5,
        'pages_processed': 84,
        'chunks_created': 1247
    }
    
    # Progress metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("URLs Discovered", progress_data['urls_discovered'])
    with col2:
        st.metric("URLs Crawled", progress_data['urls_crawled'])
    with col3:
        st.metric("URLs Failed", progress_data['urls_failed'])
    with col4:
        st.metric("Pages Processed", progress_data['pages_processed'])
    with col5:
        st.metric("Chunks Created", progress_data['chunks_created'])
    
    # Progress bar
    progress_pct = progress_data['urls_crawled'] / progress_data['urls_discovered']
    st.progress(progress_pct, text=f"Progress: {progress_pct:.1%}")
    
    # Failed URLs (if any)
    if progress_data['urls_failed'] > 0:
        with st.expander(f"❌ Failed URLs ({progress_data['urls_failed']})"):
            failed_urls = [
                {"url": "https://example.com/page1", "error": "Connection timeout"},
                {"url": "https://example.com/page2", "error": "404 Not Found"},
                {"url": "https://example.com/page3", "error": "403 Forbidden"},
            ]
            
            for failed in failed_urls:
                st.error(f"**{failed['url']}**: {failed['error']}")


def display_crawl_history():
    """Display history of previous crawl operations"""
    
    # Mock crawl history
    history_data = [
        {
            "timestamp": "2025-01-20 14:30",
            "project": "Documentation Site",
            "urls": 125,
            "status": "Completed",
            "duration": "12m 34s",
            "chunks": 1847
        },
        {
            "timestamp": "2025-01-19 09:15",
            "project": "API Reference",
            "urls": 67,
            "status": "Completed",
            "duration": "8m 12s",
            "chunks": 892
        },
        {
            "timestamp": "2025-01-18 16:45",
            "project": "Documentation Site",
            "urls": 23,
            "status": "Failed",
            "duration": "2m 18s",
            "chunks": 0
        }
    ]
    
    # Display as table
    import pandas as pd
    df = pd.DataFrame(history_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Action buttons for each crawl
    for i, crawl in enumerate(history_data):
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        
        with col1:
            st.write(f"**{crawl['project']}** - {crawl['timestamp']}")
        
        with col2:
            if st.button("📄 View Details", key=f"details_{i}"):
                st.info(f"Viewing details for crawl from {crawl['timestamp']}")
        
        with col3:
            if crawl['status'] == "Completed" and st.button("🔄 Re-run", key=f"rerun_{i}"):
                st.info(f"Re-running crawl configuration from {crawl['timestamp']}")
        
        with col4:
            if st.button("🗑️ Delete", key=f"delete_{i}"):
                st.warning(f"Crawl record from {crawl['timestamp']} deleted") 