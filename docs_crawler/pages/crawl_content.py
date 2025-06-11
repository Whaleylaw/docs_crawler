"""
Crawl Content Page
Handles web crawling configuration and execution
"""

import streamlit as st
import asyncio
import pandas as pd
from datetime import datetime
import time
from typing import List, Optional
from components.ui_components import create_progress_indicator, display_crawl_status
from components.supabase_integration import get_supabase_integration, get_project_list
from components.crawling_engine import get_crawling_engine, CrawlStatus, URLProcessor

def show():
    """Display the crawl content page"""
    
    st.header("🕷️ Crawl Content")
    st.markdown("Configure and execute web crawling operations")
    
    # Get available projects
    projects = get_project_list()
    
    if not projects:
        st.warning("⚠️ No projects available. Please create a project first.")
        if st.button("➕ Go to Project Management"):
            st.switch_page("project_management")
        return
    
    # Project selection
    project_options = [f"{p.name} ({p.project_id})" for p in projects]
    
    selected_project_idx = st.selectbox(
        "Select Project",
        range(len(project_options)),
        format_func=lambda x: project_options[x],
        help="Choose an existing project to crawl content for"
    )
    
    selected_project = projects[selected_project_idx]
    
    # Check if there's a selected project from project management page
    if 'selected_project_id' in st.session_state:
        for i, project in enumerate(projects):
            if project.project_id == st.session_state.selected_project_id:
                selected_project_idx = i
                selected_project = project
                break
        # Clear the session state
        del st.session_state.selected_project_id
    
    # Display project info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Project", selected_project.name)
    with col2:
        st.metric("Documents", selected_project.total_documents)
    with col3:
        st.metric("Storage", f"{selected_project.storage_used:.1f} MB")
    
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
        
        urls_to_crawl = []
        
        if url_input_method == "Single URL":
            crawl_url = st.text_input(
                "URL to Crawl",
                placeholder="https://example.com/docs",
                help="Enter a single URL to crawl"
            )
            if crawl_url and URLProcessor.validate_url(crawl_url):
                urls_to_crawl = [crawl_url]
                st.success(f"✅ Valid URL: {URLProcessor.detect_url_type(crawl_url)}")
            elif crawl_url:
                st.error("❌ Invalid URL format")
                
        elif url_input_method == "Multiple URLs":
            crawl_urls_text = st.text_area(
                "URLs to Crawl (one per line)",
                placeholder="https://example.com/docs\nhttps://example.com/api\nhttps://example.com/guide",
                height=150,
                help="Enter multiple URLs, one per line"
            )
            if crawl_urls_text:
                urls_to_crawl = [url.strip() for url in crawl_urls_text.split('\n') if url.strip()]
                valid_urls = [url for url in urls_to_crawl if URLProcessor.validate_url(url)]
                invalid_urls = [url for url in urls_to_crawl if not URLProcessor.validate_url(url)]
                
                if valid_urls:
                    st.success(f"✅ {len(valid_urls)} valid URLs found")
                    # Show URL types
                    url_types = {}
                    for url in valid_urls:
                        url_type = URLProcessor.detect_url_type(url)
                        url_types[url_type] = url_types.get(url_type, 0) + 1
                    
                    type_info = ", ".join([f"{count} {type}" for type, count in url_types.items()])
                    st.info(f"📊 URL types: {type_info}")
                
                if invalid_urls:
                    st.error(f"❌ {len(invalid_urls)} invalid URLs found:")
                    for url in invalid_urls:
                        st.write(f"  - {url}")
                    
                urls_to_crawl = valid_urls
                
        else:  # Upload URL List
            uploaded_file = st.file_uploader(
                "Upload URL List",
                type=['txt', 'csv'],
                help="Upload a text file with URLs (one per line) or CSV with URL column"
            )
            
            if uploaded_file:
                try:
                    if uploaded_file.name.endswith('.txt'):
                        content = uploaded_file.read().decode('utf-8')
                        urls_to_crawl = [url.strip() for url in content.split('\n') if url.strip()]
                    elif uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                        # Try to find URL column
                        url_columns = [col for col in df.columns if 'url' in col.lower()]
                        if url_columns:
                            urls_to_crawl = df[url_columns[0]].dropna().tolist()
                        else:
                            st.error("No URL column found in CSV. Please ensure there's a column with 'url' in the name.")
                    
                    if urls_to_crawl:
                        valid_urls = [url for url in urls_to_crawl if URLProcessor.validate_url(url)]
                        st.success(f"✅ Loaded {len(valid_urls)} valid URLs from file")
                        urls_to_crawl = valid_urls
                        
                except Exception as e:
                    st.error(f"Error reading file: {e}")
    
    with col2:
        st.markdown("### Smart Detection")
        st.info("🔍 Auto-detect:\n- Sitemaps\n- Text files\n- Regular webpages")
        
        if urls_to_crawl:
            # Show detected URL types
            url_type_counts = {}
            for url in urls_to_crawl:
                url_type = URLProcessor.detect_url_type(url)
                url_type_counts[url_type] = url_type_counts.get(url_type, 0) + 1
            
            for url_type, count in url_type_counts.items():
                if url_type == "sitemap":
                    st.info(f"🗺️ {count} sitemap(s)")
                elif url_type == "text_file":
                    st.info(f"📄 {count} text file(s)")
                else:
                    st.info(f"🌐 {count} webpage(s)")
    
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
        include_patterns_text = st.text_area(
            "Include Patterns (regex)",
            placeholder=".*\\.html$\n.*\\/docs\\/.*",
            height=100,
            help="Regex patterns for URLs to include (one per line)"
        )
        
        exclude_patterns_text = st.text_area(
            "Exclude Patterns (regex)", 
            placeholder=".*\\.pdf$\n.*\\/private\\/.*",
            height=100,
            help="Regex patterns for URLs to exclude (one per line)"
        )
        
        # Parse patterns
        include_patterns = [p.strip() for p in include_patterns_text.split('\n') if p.strip()] if include_patterns_text else []
        exclude_patterns = [p.strip() for p in exclude_patterns_text.split('\n') if p.strip()] if exclude_patterns_text else []
    
    with col3:
        st.markdown("#### RAG Strategy")
        rag_strategies = st.multiselect(
            "Select RAG Strategies",
            [
                "vector_embeddings",
                "contextual_embeddings", 
                "hybrid_search",
                "agentic_rag",
                "cross_encoder_reranking"
            ],
            default=["vector_embeddings"],
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
        can_start_crawl = bool(urls_to_crawl)
        if st.button("🕷️ Start Crawling", type="primary", use_container_width=True, disabled=not can_start_crawl):
            if can_start_crawl:
                config = {
                    'max_depth': max_depth,
                    'max_concurrent': max_concurrent,
                    'chunk_size': chunk_size,
                    'include_patterns': include_patterns,
                    'exclude_patterns': exclude_patterns,
                    'rag_strategies': rag_strategies
                }
                execute_crawl(selected_project.project_id, urls_to_crawl, config)
            else:
                st.error("Please provide valid URLs to crawl")
    
    with col2:
        if st.button("⏸️ Pause Active Crawl", use_container_width=True):
            pause_active_crawl(selected_project.project_id)
    
    with col3:
        if st.button("🛑 Stop Active Crawl", use_container_width=True):
            stop_active_crawl(selected_project.project_id)
    
    # Progress monitoring for active crawls
    display_active_crawls(selected_project.project_id)
    
    # Recent crawl history
    st.markdown("---")
    st.subheader("📚 Crawl History")
    display_crawl_history(selected_project.project_id)


def execute_crawl(project_id: str, urls: List[str], config: dict):
    """Execute the crawling operation"""
    
    try:
        crawling_engine = get_crawling_engine()
        
        # Create crawl job
        job = crawling_engine.create_crawl_job(project_id, urls, config)
        
        # Start crawl in background (simulate async execution)
        st.session_state.active_crawl_job = job.job_id
        
        # In a real implementation, you'd start this asynchronously
        # For now, we'll simulate by starting it immediately
        asyncio.create_task(crawling_engine.start_crawl_job(job.job_id))
        
        st.success(f"✅ Crawl job {job.job_id} started successfully!")
        st.info(f"📊 Processing {len(urls)} URLs with {config['max_concurrent']} concurrent sessions")
        
        time.sleep(1)  # Brief pause for user feedback
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Failed to start crawl: {e}")


def pause_active_crawl(project_id: str):
    """Pause active crawl for project"""
    
    crawling_engine = get_crawling_engine()
    active_jobs = [job for job in crawling_engine.list_jobs(project_id) 
                   if job.status == CrawlStatus.RUNNING]
    
    if active_jobs:
        for job in active_jobs:
            if crawling_engine.pause_job(job.job_id):
                st.success(f"⏸️ Crawl job {job.job_id} paused")
    else:
        st.info("No active crawls to pause")


def stop_active_crawl(project_id: str):
    """Stop active crawl for project"""
    
    crawling_engine = get_crawling_engine()
    active_jobs = [job for job in crawling_engine.list_jobs(project_id) 
                   if job.status in [CrawlStatus.RUNNING, CrawlStatus.PAUSED]]
    
    if active_jobs:
        for job in active_jobs:
            if crawling_engine.cancel_job(job.job_id):
                st.warning(f"🛑 Crawl job {job.job_id} cancelled")
        st.rerun()
    else:
        st.info("No active crawls to stop")


def display_active_crawls(project_id: str):
    """Display progress of active crawl jobs"""
    
    crawling_engine = get_crawling_engine()
    active_jobs = [job for job in crawling_engine.list_jobs(project_id) 
                   if job.status in [CrawlStatus.RUNNING, CrawlStatus.PAUSED]]
    
    if active_jobs:
        st.markdown("---")
        st.subheader("📊 Active Crawls")
        
        for job in active_jobs:
            with st.container():
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"**Job {job.job_id}**")
                    status_color = "🟢" if job.status == CrawlStatus.RUNNING else "⏸️"
                    st.markdown(f"Status: {status_color} {job.status.value.title()}")
                
                with col2:
                    st.metric("Progress", f"{job.progress:.1%}")
                
                with col3:
                    if job.status == CrawlStatus.RUNNING and st.button(f"⏸️ Pause", key=f"pause_{job.job_id}"):
                        crawling_engine.pause_job(job.job_id)
                        st.rerun()
                    elif job.status == CrawlStatus.PAUSED and st.button(f"▶️ Resume", key=f"resume_{job.job_id}"):
                        crawling_engine.resume_job(job.job_id)
                        st.rerun()
                
                # Progress bar
                st.progress(job.progress, text=f"Crawling progress: {job.progress:.1%}")
                
                # Job details
                with st.expander(f"Details for {job.job_id}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**URLs to crawl:** {len(job.urls)}")
                        st.write(f"**Max depth:** {job.max_depth}")
                        st.write(f"**Concurrent sessions:** {job.max_concurrent}")
                    with col2:
                        st.write(f"**Started:** {job.started_at.strftime('%H:%M:%S') if job.started_at else 'Not started'}")
                        st.write(f"**Chunk size:** {job.chunk_size}")
                        st.write(f"**RAG strategies:** {', '.join(job.rag_strategies or [])}")
                
                st.markdown("---")


def display_crawl_history(project_id: str):
    """Display history of previous crawl operations"""
    
    crawling_engine = get_crawling_engine()
    completed_jobs = [job for job in crawling_engine.list_jobs(project_id) 
                     if job.status in [CrawlStatus.COMPLETED, CrawlStatus.FAILED, CrawlStatus.CANCELLED]]
    
    if not completed_jobs:
        st.info("No crawl history available for this project")
        return
    
    # Prepare data for display
    history_data = []
    for job in completed_jobs:
        duration = "N/A"
        if job.started_at and job.completed_at:
            duration_seconds = (job.completed_at - job.started_at).total_seconds()
            duration = f"{int(duration_seconds // 60)}m {int(duration_seconds % 60)}s"
        
        status_icon = {
            CrawlStatus.COMPLETED: "✅",
            CrawlStatus.FAILED: "❌", 
            CrawlStatus.CANCELLED: "🛑"
        }.get(job.status, "❓")
        
        history_data.append({
            "Job ID": job.job_id,
            "Timestamp": job.created_at.strftime("%Y-%m-%d %H:%M"),
            "URLs": len(job.urls),
            "Status": f"{status_icon} {job.status.value.title()}",
            "Duration": duration,
            "Results": job.results_summary.get('successful', 0) if job.results_summary else 0
        })
    
    # Display as table
    if history_data:
        df = pd.DataFrame(history_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Action buttons for recent jobs
        st.markdown("#### Actions")
        for i, job in enumerate(completed_jobs[:5]):  # Show actions for last 5 jobs
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                st.write(f"**{job.job_id}** - {job.created_at.strftime('%m-%d %H:%M')}")
            
            with col2:
                if st.button("📄 Details", key=f"details_{job.job_id}"):
                    show_job_details(job)
            
            with col3:
                if job.status == CrawlStatus.COMPLETED and st.button("🔄 Re-run", key=f"rerun_{job.job_id}"):
                    rerun_crawl_job(job)
            
            with col4:
                if st.button("🗑️ Delete", key=f"delete_{job.job_id}"):
                    if crawling_engine.delete_job(job.job_id):
                        st.success(f"Job {job.job_id} deleted")
                        st.rerun()


def show_job_details(job):
    """Show detailed information about a crawl job"""
    
    st.modal("Job Details")
    with st.container():
        st.json({
            "job_id": job.job_id,
            "project_id": job.project_id,
            "status": job.status.value,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "progress": job.progress,
            "urls": job.urls,
            "configuration": {
                "max_depth": job.max_depth,
                "max_concurrent": job.max_concurrent,
                "chunk_size": job.chunk_size,
                "include_patterns": job.include_patterns,
                "exclude_patterns": job.exclude_patterns,
                "rag_strategies": job.rag_strategies
            },
            "results_summary": job.results_summary,
            "error_message": job.error_message
        })


def rerun_crawl_job(job):
    """Re-run a previous crawl job with the same configuration"""
    
    config = {
        'max_depth': job.max_depth,
        'max_concurrent': job.max_concurrent,
        'chunk_size': job.chunk_size,
        'include_patterns': job.include_patterns,
        'exclude_patterns': job.exclude_patterns,
        'rag_strategies': job.rag_strategies
    }
    
    execute_crawl(job.project_id, job.urls, config) 