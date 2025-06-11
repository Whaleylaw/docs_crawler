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
from components.rag_strategies import RAGStrategySelector, RAGConfiguration, RAGStrategy

def show():
    """Display the crawl content page"""
    
    st.header("🕷️ Crawl Content")
    st.markdown("Configure and execute web crawling operations with advanced RAG processing")
    
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
    
    # RAG Strategy Selection
    st.markdown("---")
    rag_config = RAGStrategySelector.render_strategy_selection()
    
    # Show preview if enabled
    preview_samples = []
    if rag_config.preview_mode and urls_to_crawl:
        st.markdown("---")
        st.subheader("👁️ Preview Mode")
        
        if st.button("🔍 Generate Preview", help="Preview how your content will be processed"):
            with st.spinner("Generating preview samples..."):
                preview_samples = generate_content_preview(urls_to_crawl[:3], rag_config)
                
            if preview_samples:
                display_preview_samples(preview_samples)
    
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
        st.markdown("#### Cost Estimation")
        
        # Calculate estimated costs based on configuration
        estimated_urls = len(urls_to_crawl)
        if any(URLProcessor.detect_url_type(url) == "sitemap" for url in urls_to_crawl):
            estimated_urls *= 20  # Rough estimate for sitemap expansion
        
        if max_depth > 1:
            estimated_urls *= max_depth
        
        # OpenAI API cost estimation
        openai_cost = 0
        if RAGStrategy.CONTEXTUAL_EMBEDDINGS in rag_config.enabled_strategies:
            openai_cost += estimated_urls * 0.02  # Rough estimate for context generation
        if RAGStrategy.AGENTIC_RAG in rag_config.enabled_strategies:
            openai_cost += estimated_urls * 0.01  # Rough estimate for code summarization
        
        embedding_cost = estimated_urls * 0.0001  # Embedding cost
        
        st.metric("Estimated URLs", f"{estimated_urls:,}")
        st.metric("Est. OpenAI Cost", f"${openai_cost + embedding_cost:.2f}")
        
        if openai_cost > 5:
            st.warning("⚠️ High cost estimate. Consider reducing scope.")
        elif openai_cost > 0:
            st.info("💡 AI features will incur OpenAI API costs")
    
    # Configuration summary
    with st.expander("📋 Configuration Summary"):
        RAGStrategySelector.display_strategy_summary(rag_config)
        
        st.markdown("#### Crawl Settings")
        st.json({
            "max_depth": max_depth,
            "max_concurrent": max_concurrent,
            "chunk_size": chunk_size,
            "include_patterns": include_patterns,
            "exclude_patterns": exclude_patterns,
            "urls_count": len(urls_to_crawl)
        })
    
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
                    'rag_strategies': [strategy.value for strategy in rag_config.enabled_strategies],
                    'rag_config': {
                        'embedding_model': rag_config.embedding_model,
                        'context_model': rag_config.context_model,
                        'max_context_tokens': rag_config.max_context_tokens,
                        'min_code_length': rag_config.min_code_length,
                        'parallel_workers': rag_config.parallel_workers,
                        'preview_mode': rag_config.preview_mode,
                        'preview_sample_size': rag_config.preview_sample_size,
                        'batch_size': rag_config.batch_size,
                        'max_retries': rag_config.max_retries
                    }
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


def generate_content_preview(urls: List[str], rag_config: RAGConfiguration) -> List[dict]:
    """Generate preview samples for content processing"""
    
    try:
        from components.rag_strategies import get_rag_processor
        
        # Create a RAG processor with the configuration
        rag_processor = get_rag_processor(rag_config)
        
        # Simple preview - just fetch first few URLs and show processing
        preview_samples = []
        
        # Mock content for preview (in real implementation, would fetch actual content)
        mock_contents = [
            "# API Documentation\n\nThis is a comprehensive guide to our REST API.\n\n```python\nimport requests\n\nresponse = requests.get('https://api.example.com/data')\nprint(response.json())\n```",
            "## Authentication\n\nUse JWT tokens for authentication.\n\n```javascript\nconst token = 'your-jwt-token';\nfetch('/api/data', {\n  headers: { 'Authorization': `Bearer ${token}` }\n})\n```",
            "### Error Handling\n\nProper error handling is essential for robust applications.\n\n```python\ntry:\n    result = api_call()\nexcept APIError as e:\n    log.error(f'API call failed: {e}')\n```"
        ]
        
        for i, (url, content) in enumerate(zip(urls[:3], mock_contents[:3])):
            # Process content with RAG strategies
            processed = rag_processor.process_content(content, url, content)
            
            preview_samples.append({
                "url": url,
                "original_content": content,
                "processed_content": processed.get('processed_content', content),
                "strategies_applied": processed.get('strategies_applied', []),
                "code_examples": processed.get('code_examples', []),
                "metadata": processed.get('metadata', {})
            })
        
        return preview_samples
        
    except Exception as e:
        st.error(f"Error generating preview: {e}")
        return []


def display_preview_samples(samples: List[dict]):
    """Display preview samples to the user"""
    
    st.markdown("### 🔍 Content Processing Preview")
    st.info("This preview shows how your content will be processed with the selected RAG strategies.")
    
    for i, sample in enumerate(samples):
        with st.expander(f"Sample {i+1}: {sample['url']}", expanded=i == 0):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Original Content")
                st.code(sample['original_content'][:500] + "..." if len(sample['original_content']) > 500 else sample['original_content'])
            
            with col2:
                st.markdown("#### Processed Content")
                processed_preview = sample['processed_content'][:500] + "..." if len(sample['processed_content']) > 500 else sample['processed_content']
                if sample['processed_content'] != sample['original_content']:
                    st.code(processed_preview)
                else:
                    st.info("No changes (no contextual processing applied)")
            
            # Show applied strategies
            if sample['strategies_applied']:
                st.markdown("#### Applied Strategies")
                for strategy in sample['strategies_applied']:
                    st.write(f"✅ {strategy.replace('_', ' ').title()}")
            
            # Show code examples if found
            if sample['code_examples']:
                st.markdown(f"#### Code Examples Found: {len(sample['code_examples'])}")
                for j, code_ex in enumerate(sample['code_examples'][:2]):  # Show first 2
                    with st.expander(f"Code Example {j+1} ({code_ex.get('language', 'unknown')})"):
                        st.code(code_ex['code'][:300] + "..." if len(code_ex['code']) > 300 else code_ex['code'], 
                                language=code_ex.get('language', 'text'))
                        if code_ex.get('summary'):
                            st.markdown(f"**Summary:** {code_ex['summary']}")
            
            # Show metadata
            if sample['metadata']:
                with st.expander("Metadata"):
                    st.json(sample['metadata'])


def execute_crawl(project_id: str, urls: List[str], config: dict):
    """Execute the crawling operation"""
    
    try:
        crawling_engine = get_crawling_engine()
        
        # Create crawl job
        job = crawling_engine.create_crawl_job(project_id, urls, config)
        
        # Start crawl in background (simulate async execution)
        st.session_state.active_crawl_job = job.job_id
        
        # Show configuration summary
        st.success(f"✅ Crawl job {job.job_id} created successfully!")
        
        with st.expander("Job Configuration", expanded=False):
            st.json({
                "job_id": job.job_id,
                "urls_count": len(urls),
                "max_depth": config['max_depth'],
                "max_concurrent": config['max_concurrent'],
                "rag_strategies": config['rag_strategies'],
                "estimated_processing_time": f"{len(urls) * 2} - {len(urls) * 5} minutes"
            })
        
        # In a real implementation, you'd start this asynchronously
        # For now, we'll simulate by starting it immediately
        if st.button("▶️ Confirm and Start Processing"):
            asyncio.create_task(crawling_engine.start_crawl_job(job.job_id))
            st.info(f"📊 Processing {len(urls)} URLs with {config['max_concurrent']} concurrent sessions")
            time.sleep(1)  # Brief pause for user feedback
            st.rerun()
        
    except Exception as e:
        st.error(f"❌ Failed to create crawl job: {e}")


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
                        st.write(f"**RAG strategies:** {', '.join(job.rag_strategies or [])}")
                    with col2:
                        st.write(f"**Started:** {job.started_at.strftime('%H:%M:%S') if job.started_at else 'Not started'}")
                        st.write(f"**Chunk size:** {job.chunk_size}")
                        if job.rag_config:
                            rag_info = f"Model: {job.rag_config.get('embedding_model', 'default')}"
                            if job.rag_config.get('use_contextual_embeddings'):
                                rag_info += f", Context: {job.rag_config.get('context_model', 'gpt-3.5-turbo')}"
                            st.write(f"**RAG config:** {rag_info}")
                
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
        
        # Show RAG strategies if available
        rag_strategies = job.rag_strategies or ['vector_embeddings']
        rag_display = ", ".join([s.replace('_', ' ').title() for s in rag_strategies])
        
        history_data.append({
            "Job ID": job.job_id,
            "Timestamp": job.created_at.strftime("%Y-%m-%d %H:%M"),
            "URLs": len(job.urls),
            "Status": f"{status_icon} {job.status.value.title()}",
            "Duration": duration,
            "Results": job.results_summary.get('successful', 0) if job.results_summary else 0,
            "RAG Strategies": rag_display
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
        
        # Show results summary if available
        if job.results_summary:
            st.markdown("#### Results Summary")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Successful URLs", job.results_summary.get('successful', 0))
            with col2:
                st.metric("Total Chunks", job.results_summary.get('total_chunks', 0))
            with col3:
                st.metric("Code Examples", job.results_summary.get('total_code_examples', 0))
            
            if job.results_summary.get('rag_strategies_applied'):
                st.markdown("#### RAG Strategies Applied")
                for strategy in job.results_summary['rag_strategies_applied']:
                    st.write(f"✅ {strategy.replace('_', ' ').title()}")
        
        # Show full job configuration
        st.markdown("#### Full Configuration")
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
                "rag_strategies": job.rag_strategies,
                "rag_config": job.rag_config
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
        'rag_strategies': job.rag_strategies,
        'rag_config': job.rag_config
    }
    
    execute_crawl(job.project_id, job.urls, config) 