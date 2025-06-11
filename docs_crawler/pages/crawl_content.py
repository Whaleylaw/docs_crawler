"""
Crawl Content Page
Handles web crawling configuration and execution using the enhanced crawling engine
"""

import streamlit as st
import asyncio
import pandas as pd
from datetime import datetime
import time
from typing import List
from components.ui_components import create_progress_indicator, display_crawl_status
from components.supabase_integration import get_supabase_integration, get_project_list
from components.crawling_engine import get_crawling_engine, CrawlStatus


def show():
    """Display the crawl content page"""
    
    st.header("🕷️ Crawl Content")
    st.markdown("Configure and execute web crawling operations")
    
    # Get available projects
    projects = get_project_list()
    crawling_engine = get_crawling_engine()
    
    if not projects:
        st.warning("No projects available. Please create a project first in Project Management.")
        return
    
    # Project selection
    project_options = [f"{p.name} ({p.project_id})" for p in projects]
    selected_project_idx = st.selectbox(
        "Select Project",
        range(len(project_options)),
        format_func=lambda x: project_options[x],
        help="Choose an existing project for crawling"
    )
    
    selected_project = projects[selected_project_idx]
    
    # Store selected project in session state
    st.session_state.selected_project_id = selected_project.project_id
    
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
            if crawl_url:
                urls_to_crawl = [crawl_url]
        
        elif url_input_method == "Multiple URLs":
            crawl_urls_text = st.text_area(
                "URLs to Crawl (one per line)",
                placeholder="https://example.com/docs\nhttps://example.com/api\nhttps://example.com/guide",
                height=150,
                help="Enter multiple URLs, one per line"
            )
            if crawl_urls_text:
                urls_to_crawl = [url.strip() for url in crawl_urls_text.split('\n') if url.strip()]
        
        else:  # Upload URL List
            uploaded_file = st.file_uploader(
                "Upload URL List",
                type=['txt', 'csv'],
                help="Upload a text file with URLs (one per line) or CSV with URL column"
            )
            if uploaded_file:
                content = uploaded_file.read().decode('utf-8')
                urls_to_crawl = [url.strip() for url in content.split('\n') if url.strip()]
    
    with col2:
        st.markdown("### Smart Detection")
        st.info("🔍 Auto-detect:\n- Sitemaps\n- Text files\n- Regular webpages")
        
        detect_sitemaps = st.checkbox("Auto-detect Sitemaps", value=True)
        detect_text_files = st.checkbox("Auto-detect Text Files", value=True)
    
    # URL validation and preview
    if urls_to_crawl:
        st.markdown("### 📋 URLs to Process")
        with st.expander(f"View {len(urls_to_crawl)} URLs", expanded=len(urls_to_crawl) <= 5):
            for i, url in enumerate(urls_to_crawl[:10]):  # Show first 10
                st.text(f"{i+1}. {url}")
            if len(urls_to_crawl) > 10:
                st.text(f"... and {len(urls_to_crawl) - 10} more URLs")
    
    # Crawling configuration
    st.markdown("---")
    st.subheader("⚙️ Crawling Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Basic Settings")
        max_depth = st.slider("Maximum Depth", 1, 5, 2, help="How many levels deep to crawl")
        max_concurrent = st.slider("Concurrent Sessions", 1, 20, 5, help="Number of parallel crawling sessions")
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
        rag_strategy_options = [
            "contextual",
            "hybrid", 
            "agentic",
            "cross_encoder"
        ]
        
        rag_strategy_labels = [
            "Contextual Embeddings (slower, higher accuracy)",
            "Hybrid Search (vector + keyword)",
            "Agentic RAG (extracts code examples)",
            "Cross-encoder Reranking (improves relevance)"
        ]
        
        selected_strategies = st.multiselect(
            "Select RAG Strategies",
            rag_strategy_options,
            default=["contextual"],
            format_func=lambda x: rag_strategy_labels[rag_strategy_options.index(x)],
            help="Choose processing strategies for content"
        )
        
        preview_mode = st.checkbox(
            "Preview Mode",
            help="Show sample chunks before full processing"
        )
    
    # Crawl execution
    st.markdown("---")
    st.subheader("🚀 Execute Crawl")
    
    # Configuration summary
    config_summary = {
        "max_depth": max_depth,
        "max_concurrent": max_concurrent,
        "chunk_size": chunk_size,
        "include_patterns": include_patterns,
        "exclude_patterns": exclude_patterns,
        "rag_strategies": selected_strategies
    }
    
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        start_crawl_disabled = not urls_to_crawl or any(
            job.status == CrawlStatus.RUNNING 
            for job in crawling_engine.list_jobs(selected_project.project_id)
        )
        
        if st.button("🕷️ Start Crawling", type="primary", disabled=start_crawl_disabled, use_container_width=True):
            if urls_to_crawl:
                execute_crawl(selected_project.project_id, urls_to_crawl, config_summary)
            else:
                st.error("Please provide URLs to crawl")
    
    # Show active jobs controls
    active_jobs = [job for job in crawling_engine.list_jobs(selected_project.project_id) if job.status == CrawlStatus.RUNNING]
    
    if active_jobs:
        active_job = active_jobs[0]  # Assume one job at a time for now
        
        with col2:
            if st.button("⏸️ Pause", use_container_width=True):
                if crawling_engine.pause_job(active_job.job_id):
                    st.success("Crawl paused")
                    st.rerun()
        
        with col3:
            if st.button("▶️ Resume", use_container_width=True):
                if crawling_engine.resume_job(active_job.job_id):
                    st.success("Crawl resumed")
                    st.rerun()
        
        with col4:
            if st.button("🛑 Cancel", use_container_width=True):
                if crawling_engine.cancel_job(active_job.job_id):
                    st.warning("Crawl cancelled")
                    st.rerun()
    
    # Progress monitoring for active jobs
    current_jobs = crawling_engine.list_jobs(selected_project.project_id)
    running_jobs = [job for job in current_jobs if job.status == CrawlStatus.RUNNING]
    
    if running_jobs:
        st.markdown("---")
        st.subheader("📊 Crawl Progress")
        for job in running_jobs:
            display_job_progress(job)
    
    # Crawl history
    st.markdown("---")
    st.subheader("📚 Crawl History")
    display_crawl_history(selected_project.project_id, current_jobs)


def execute_crawl(project_id: str, urls: List[str], config: dict):
    """Execute the crawling operation"""
    
    crawling_engine = get_crawling_engine()
    
    try:
        # Create crawl job
        job = crawling_engine.create_crawl_job(project_id, urls, config)
        
        st.success(f"✅ Crawl job {job.job_id} created successfully!")
        st.info("🚀 Crawling will start in the background. Monitor progress below.")
        
        # Start the job asynchronously
        asyncio.create_task(start_crawl_job_async(job.job_id))
        
        # Store job ID in session state for monitoring
        if 'active_crawl_jobs' not in st.session_state:
            st.session_state.active_crawl_jobs = []
        st.session_state.active_crawl_jobs.append(job.job_id)
        
        st.rerun()
        
    except Exception as e:
        st.error(f"Failed to start crawl: {str(e)}")


async def start_crawl_job_async(job_id: str):
    """Start crawl job asynchronously"""
    crawling_engine = get_crawling_engine()
    success = await crawling_engine.start_crawl_job(job_id)
    return success


def display_job_progress(job):
    """Display real-time progress for a crawl job"""
    
    # Progress metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    discovered_count = len(job.discovered_urls)
    processed_count = len(job.processed_urls)
    failed_count = len(job.failed_urls)
    
    with col1:
        st.metric("URLs Discovered", discovered_count)
    with col2:
        st.metric("URLs Processed", processed_count)
    with col3:
        st.metric("URLs Failed", failed_count)
    with col4:
        st.metric("Progress", f"{job.progress:.1%}")
    with col5:
        st.metric("Job ID", job.job_id)
    
    # Progress bar
    st.progress(job.progress, text=f"Crawling: {job.progress:.1%} complete")
    
    # Job details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Configuration:**")
        st.write(f"- Max Depth: {job.max_depth}")
        st.write(f"- Concurrent: {job.max_concurrent}")
        st.write(f"- Chunk Size: {job.chunk_size}")
    
    with col2:
        st.markdown("**Status:**")
        st.write(f"- Started: {job.started_at.strftime('%H:%M:%S') if job.started_at else 'Not started'}")
        st.write(f"- Status: {job.status.value}")
        
        if job.error_message:
            st.error(f"Error: {job.error_message}")
    
    # Failed URLs (if any)
    if job.failed_urls:
        with st.expander(f"❌ Failed URLs ({len(job.failed_urls)})"):
            for url, error in job.failed_urls.items():
                st.error(f"**{url}**: {error}")


def display_crawl_history(project_id: str, jobs: list):
    """Display history of crawl operations for the project"""
    
    if not jobs:
        st.info("No crawl history for this project yet.")
        return
    
    # Filter completed jobs
    completed_jobs = [job for job in jobs if job.status in [CrawlStatus.COMPLETED, CrawlStatus.FAILED, CrawlStatus.CANCELLED]]
    
    if not completed_jobs:
        st.info("No completed crawl jobs yet.")
        return
    
    # Create history table
    history_data = []
    for job in completed_jobs:
        duration = "N/A"
        if job.started_at and job.completed_at:
            duration_seconds = (job.completed_at - job.started_at).total_seconds()
            if duration_seconds < 60:
                duration = f"{duration_seconds:.0f}s"
            else:
                duration = f"{duration_seconds/60:.1f}m"
        
        history_data.append({
            "Timestamp": job.created_at.strftime("%Y-%m-%d %H:%M"),
            "Job ID": job.job_id,
            "URLs": len(job.urls),
            "Status": job.status.value.title(),
            "Duration": duration,
            "Results": len(job.processed_urls) if job.processed_urls else 0
        })
    
    # Display as table
    if history_data:
        df = pd.DataFrame(history_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Action buttons for each job
        st.markdown("#### Job Actions")
        for i, job in enumerate(completed_jobs):
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                status_color = "🟢" if job.status == CrawlStatus.COMPLETED else "🔴" if job.status == CrawlStatus.FAILED else "🟡"
                st.write(f"{status_color} **{job.job_id}** - {job.created_at.strftime('%Y-%m-%d %H:%M')}")
            
            with col2:
                if st.button("� Results", key=f"results_{job.job_id}"):
                    show_job_results(job)
            
            with col3:
                if job.status == CrawlStatus.COMPLETED and st.button("🔄 Re-run", key=f"rerun_{job.job_id}"):
                    # Re-create job with same config
                    config = {
                        "max_depth": job.max_depth,
                        "max_concurrent": job.max_concurrent,
                        "chunk_size": job.chunk_size,
                        "include_patterns": job.include_patterns,
                        "exclude_patterns": job.exclude_patterns,
                        "rag_strategies": job.rag_strategies
                    }
                    execute_crawl(project_id, job.urls, config)
            
            with col4:
                if st.button("🗑️ Delete", key=f"delete_{job.job_id}"):
                    if get_crawling_engine().delete_job(job.job_id):
                        st.success("Job deleted")
                        st.rerun()


def show_job_results(job):
    """Show detailed results for a completed job"""
    
    st.markdown(f"### 📊 Results for Job {job.job_id}")
    
    if job.results_summary:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total URLs", job.results_summary.get('total_urls', 0))
        with col2:
            st.metric("Successful", job.results_summary.get('successful', 0))
        with col3:
            st.metric("Failed", job.results_summary.get('failed', 0))
        with col4:
            st.metric("Total Chunks", job.results_summary.get('total_chunks', 0))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Words", job.results_summary.get('total_words', 0))
        with col2:
            st.metric("Code Examples", job.results_summary.get('total_code_examples', 0))
        with col3:
            st.metric("Avg Chunks/URL", f"{job.results_summary.get('avg_chunks_per_url', 0):.1f}")
    
    else:
        st.info("No detailed results available for this job.")
    
    # Show configuration used
    with st.expander("🔧 Configuration Used"):
        st.json({
            "max_depth": job.max_depth,
            "max_concurrent": job.max_concurrent,
            "chunk_size": job.chunk_size,
            "include_patterns": job.include_patterns,
            "exclude_patterns": job.exclude_patterns,
            "rag_strategies": job.rag_strategies
        }) 