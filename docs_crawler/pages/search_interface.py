"""
Search Interface Page
Provides semantic search capabilities across crawled content
"""

import streamlit as st
import pandas as pd
import json
import csv
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from components.ui_components import display_search_result
from components.supabase_integration import get_supabase_integration, get_project_list
from components.rag_strategies import get_rag_processor, RAGConfiguration, RAGStrategy

def show():
    """Display the search interface page"""
    
    st.header("🔍 Search Interface")
    st.markdown("Semantic search across your crawled content with AI-powered relevance ranking")
    
    # Get available projects
    projects = get_project_list()
    
    if not projects:
        st.warning("⚠️ No projects available. Please create a project and crawl some content first.")
        if st.button("➕ Go to Project Management"):
            st.switch_page("project_management")
        return
    
    # Project selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        project_options = ["All Projects"] + [f"{p.name} ({p.project_id})" for p in projects]
        selected_project_idx = st.selectbox(
            "Search Scope",
            range(len(project_options)),
            format_func=lambda x: project_options[x],
            help="Choose projects to search across"
        )
        
        selected_project = None if selected_project_idx == 0 else projects[selected_project_idx - 1]
    
    with col2:
        # Quick stats
        if selected_project:
            st.metric("Documents", selected_project.total_documents)
            st.metric("Storage", f"{selected_project.storage_used:.1f} MB")
        else:
            total_docs = sum(p.total_documents for p in projects)
            total_storage = sum(p.storage_used for p in projects)
            st.metric("Total Documents", total_docs)
            st.metric("Total Storage", f"{total_storage:.1f} MB")
    
    # Search configuration
    st.markdown("---")
    st.subheader("🔧 Search Configuration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Search input
        search_query = st.text_input(
            "Search Query",
            placeholder="What are you looking for?",
            help="Enter your search query using natural language"
        )
        
        # Advanced filters
        with st.expander("🎛️ Advanced Filters", expanded=False):
            col1_inner, col2_inner = st.columns(2)
            
            with col1_inner:
                similarity_threshold = st.slider(
                    "Similarity Threshold",
                    0.0, 1.0, 0.7,
                    step=0.05,
                    help="Minimum similarity score for results"
                )
                
                max_results = st.slider(
                    "Maximum Results",
                    5, 100, 20,
                    step=5,
                    help="Maximum number of results to return"
                )
                
                content_types = st.multiselect(
                    "Content Types",
                    ["documentation", "code_examples", "api_reference", "tutorials"],
                    help="Filter by content type"
                )
            
            with col2_inner:
                date_range = st.date_input(
                    "Date Range",
                    value=(datetime.now().date() - timedelta(days=30), datetime.now().date()),
                    help="Filter by crawl date"
                )
                
                source_filter = st.text_input(
                    "Source Filter",
                    placeholder="github.com, docs.python.org",
                    help="Filter by source domain (comma-separated)"
                )
                
                enable_reranking = st.checkbox(
                    "Enable AI Reranking",
                    value=True,
                    help="Use cross-encoder model to improve result relevance"
                )
    
    with col2:
        # RAG Enhancement Configuration
        st.markdown("### 🧠 Search Enhancement")
        
        use_code_search = st.checkbox(
            "Include Code Examples",
            value=False,
            help="Search through extracted code examples with AI summaries"
        )
        
        hybrid_search = st.checkbox(
            "Hybrid Search (Beta)",
            value=False,
            help="Combine vector search with keyword matching",
            disabled=True  # Not fully implemented yet
        )
        
        search_mode = st.radio(
            "Search Mode",
            ["Semantic", "Exact Match", "Mixed"],
            index=0,
            help="Choose search approach"
        )
        
        # Show estimated processing time
        if enable_reranking:
            st.info("🔄 AI reranking enabled\n(+1-2 seconds processing)")
        else:
            st.info("⚡ Fast search mode")
    
    # Search execution
    search_results = []
    code_results = []
    
    if st.button("🔍 Search", type="primary", use_container_width=True) and search_query:
        with st.spinner("Searching..."):
            # Perform the search
            search_results, code_results = perform_search(
                query=search_query,
                project_id=selected_project.project_id if selected_project else None,
                similarity_threshold=similarity_threshold,
                max_results=max_results,
                content_types=content_types,
                date_range=date_range,
                source_filter=source_filter,
                enable_reranking=enable_reranking,
                include_code_search=use_code_search,
                search_mode=search_mode
            )
            
            # Store search in history
            store_search_history(search_query, selected_project.project_id if selected_project else "all", 
                               len(search_results) + len(code_results))
    
    # Display search results
    if search_results or code_results:
        display_search_results(search_query, search_results, code_results, enable_reranking)
    
    # Search history and saved searches
    st.markdown("---")
    display_search_history_section()


def perform_search(query: str, project_id: Optional[str], similarity_threshold: float,
                  max_results: int, content_types: List[str], date_range: Tuple,
                  source_filter: str, enable_reranking: bool, include_code_search: bool,
                  search_mode: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Perform the actual search with all configured options
    """
    
    try:
        supabase_integration = get_supabase_integration()
        
        # Get appropriate client
        if project_id:
            client = supabase_integration.get_supabase_client(project_id)
        else:
            # For "All Projects", we'll need to search across all available projects
            # For now, use the first project's client - in production, you'd implement cross-project search
            client = supabase_integration.get_supabase_client(get_project_list()[0].project_id)
        
        # Prepare search filters
        search_filters = {}
        
        if content_types:
            search_filters["content_types"] = content_types
        
        if source_filter:
            source_domains = [s.strip() for s in source_filter.split(',') if s.strip()]
            search_filters["source_domains"] = source_domains
        
        # Perform semantic search on documents
        search_results = []
        if client:
            search_results = supabase_integration.search_documents(
                project_id=project_id or get_project_list()[0].project_id,
                query=query,
                match_count=max_results,
                similarity_threshold=similarity_threshold,
                filter_metadata=search_filters
            )
        
        # Perform code search if enabled
        code_results = []
        if include_code_search and client:
            code_results = supabase_integration.search_code_examples(
                project_id=project_id or get_project_list()[0].project_id,
                query=query,
                match_count=max_results // 2,  # Half the results for code
                similarity_threshold=similarity_threshold
            )
        
        # Apply reranking if enabled
        if enable_reranking and (search_results or code_results):
            # Initialize RAG processor with reranking enabled
            rag_config = RAGConfiguration(
                enabled_strategies=[RAGStrategy.CROSS_ENCODER_RERANKING],
                use_reranking=True
            )
            rag_processor = get_rag_processor(rag_config)
            
            # Rerank document results
            if search_results:
                search_results = rag_processor.rerank_search_results(
                    query=query,
                    results=search_results,
                    content_key="content"
                )
            
            # Rerank code results
            if code_results:
                code_results = rag_processor.rerank_search_results(
                    query=query,
                    results=code_results,
                    content_key="code_example"
                )
        
        return search_results, code_results
        
    except Exception as e:
        st.error(f"Search failed: {e}")
        return [], []


def display_search_results(query: str, search_results: List[Dict], code_results: List[Dict], 
                         reranked: bool):
    """Display search results with enhanced formatting"""
    
    total_results = len(search_results) + len(code_results)
    
    if total_results == 0:
        st.info("🔍 No results found. Try adjusting your search terms or filters.")
        return
    
    # Results header
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader(f"🎯 Search Results ({total_results})")
        st.markdown(f"Query: *{query}*")
    
    with col2:
        if reranked:
            st.success("🧠 AI Reranked")
        else:
            st.info("⚡ Vector Search")
    
    with col3:
        # Export options
        if st.button("📊 Export Results"):
            export_search_results(query, search_results, code_results)
    
    # Display document results
    if search_results:
        st.markdown("### 📄 Document Results")
        
        for i, result in enumerate(search_results):
            with st.expander(f"Result {i+1}: {result.get('metadata', {}).get('title', 'Untitled')}", expanded=i < 3):
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Show content preview
                    content_preview = result.get('content', '')[:500] + "..." if len(result.get('content', '')) > 500 else result.get('content', '')
                    st.markdown(f"**Content:**")
                    st.write(content_preview)
                    
                    # Highlight search relevance if reranked
                    if reranked and 'rerank_score' in result:
                        st.progress(result['rerank_score'], text=f"Relevance: {result['rerank_score']:.3f}")
                    elif 'similarity' in result:
                        st.progress(result['similarity'], text=f"Similarity: {result['similarity']:.3f}")
                
                with col2:
                    # Metadata
                    st.markdown("**Metadata:**")
                    metadata = result.get('metadata', {})
                    
                    if 'title' in metadata:
                        st.write(f"**Title:** {metadata['title']}")
                    
                    st.write(f"**URL:** [{result.get('url', 'Unknown')}]({result.get('url', '#')})")
                    
                    if 'word_count' in metadata:
                        st.write(f"**Words:** {metadata['word_count']}")
                    
                    if 'crawled_at' in metadata:
                        st.write(f"**Crawled:** {metadata['crawled_at'][:10]}")  # Show date only
                    
                    if 'rag_strategies' in metadata and metadata['rag_strategies']:
                        st.write(f"**RAG:** {', '.join([s.replace('_', ' ').title() for s in metadata['rag_strategies']])}")
                    
                    # Action buttons
                    if st.button("🔗 Open URL", key=f"open_{i}"):
                        st.link_button("Open", result.get('url', '#'))
                    
                    if st.button("📋 Copy Content", key=f"copy_{i}"):
                        st.session_state[f"copied_content_{i}"] = result.get('content', '')
                        st.success("Content copied!")
    
    # Display code example results
    if code_results:
        st.markdown("### 💻 Code Example Results")
        
        for i, result in enumerate(code_results):
            with st.expander(f"Code Example {i+1}: {result.get('summary', 'Code snippet')}", expanded=i < 2):
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Show code with syntax highlighting
                    code_content = result.get('code_example', result.get('content', ''))
                    language = result.get('metadata', {}).get('language', 'text')
                    
                    st.code(code_content, language=language)
                    
                    # Show AI-generated summary
                    if result.get('summary'):
                        st.markdown("**AI Summary:**")
                        st.info(result['summary'])
                    
                    # Show relevance score
                    if reranked and 'rerank_score' in result:
                        st.progress(result['rerank_score'], text=f"Code Relevance: {result['rerank_score']:.3f}")
                    elif 'similarity' in result:
                        st.progress(result['similarity'], text=f"Similarity: {result['similarity']:.3f}")
                
                with col2:
                    # Code metadata
                    st.markdown("**Code Info:**")
                    
                    metadata = result.get('metadata', {})
                    
                    if 'language' in metadata:
                        st.write(f"**Language:** {metadata['language']}")
                    
                    st.write(f"**Source:** [{result.get('url', 'Unknown')}]({result.get('url', '#')})")
                    
                    if 'char_count' in metadata:
                        st.write(f"**Characters:** {metadata['char_count']}")
                    
                    if 'word_count' in metadata:
                        st.write(f"**Words:** {metadata['word_count']}")
                    
                    # Action buttons
                    if st.button("🔗 View Source", key=f"code_open_{i}"):
                        st.link_button("Open", result.get('url', '#'))
                    
                    if st.button("📋 Copy Code", key=f"code_copy_{i}"):
                        st.session_state[f"copied_code_{i}"] = code_content
                        st.success("Code copied!")


def export_search_results(query: str, search_results: List[Dict], code_results: List[Dict]):
    """Export search results to CSV/JSON"""
    
    # Prepare data for export
    export_data = []
    
    # Add document results
    for result in search_results:
        export_data.append({
            "type": "document",
            "query": query,
            "url": result.get('url', ''),
            "title": result.get('metadata', {}).get('title', ''),
            "content": result.get('content', ''),
            "similarity": result.get('similarity', 0),
            "rerank_score": result.get('rerank_score', ''),
            "word_count": result.get('metadata', {}).get('word_count', 0),
            "crawled_at": result.get('metadata', {}).get('crawled_at', ''),
            "rag_strategies": ', '.join(result.get('metadata', {}).get('rag_strategies', []))
        })
    
    # Add code results
    for result in code_results:
        export_data.append({
            "type": "code_example",
            "query": query,
            "url": result.get('url', ''),
            "title": result.get('summary', ''),
            "content": result.get('code_example', result.get('content', '')),
            "similarity": result.get('similarity', 0),
            "rerank_score": result.get('rerank_score', ''),
            "word_count": result.get('metadata', {}).get('word_count', 0),
            "language": result.get('metadata', {}).get('language', ''),
            "crawled_at": result.get('metadata', {}).get('crawled_at', '')
        })
    
    # Create downloadable files
    if export_data:
        # CSV export
        df = pd.DataFrame(export_data)
        csv_data = df.to_csv(index=False)
        
        # JSON export
        json_data = json.dumps(export_data, indent=2)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"search_results_{query.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=f"search_results_{query.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )


def store_search_history(query: str, project_id: str, result_count: int):
    """Store search query in history"""
    
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    
    # Add to history (keep last 50 searches)
    history_entry = {
        "query": query,
        "project_id": project_id,
        "result_count": result_count,
        "timestamp": datetime.now().isoformat(),
        "search_id": len(st.session_state.search_history) + 1
    }
    
    st.session_state.search_history.insert(0, history_entry)
    st.session_state.search_history = st.session_state.search_history[:50]


def display_search_history_section():
    """Display search history and saved searches"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📚 Recent Searches")
        
        if 'search_history' not in st.session_state or not st.session_state.search_history:
            st.info("No search history yet")
        else:
            # Show last 10 searches
            recent_searches = st.session_state.search_history[:10]
            
            for search in recent_searches:
                with st.container():
                    col1_inner, col2_inner, col3_inner = st.columns([3, 1, 1])
                    
                    with col1_inner:
                        st.write(f"**{search['query']}**")
                        timestamp = datetime.fromisoformat(search['timestamp'])
                        st.caption(f"{timestamp.strftime('%m/%d %H:%M')} • {search['result_count']} results")
                    
                    with col2_inner:
                        if st.button("🔄", key=f"repeat_{search['search_id']}", help="Repeat search"):
                            st.session_state.repeat_search = search['query']
                            st.rerun()
                    
                    with col3_inner:
                        if st.button("⭐", key=f"save_{search['search_id']}", help="Save search"):
                            save_search(search)
    
    with col2:
        st.subheader("⭐ Saved Searches")
        
        if 'saved_searches' not in st.session_state or not st.session_state.saved_searches:
            st.info("No saved searches yet")
        else:
            saved_searches = st.session_state.saved_searches
            
            for search in saved_searches:
                with st.container():
                    col1_inner, col2_inner, col3_inner = st.columns([3, 1, 1])
                    
                    with col1_inner:
                        st.write(f"**{search['query']}**")
                        st.caption(f"Saved on {search['saved_at'][:10]}")
                    
                    with col2_inner:
                        if st.button("🔍", key=f"run_saved_{search['search_id']}", help="Run search"):
                            st.session_state.repeat_search = search['query']
                            st.rerun()
                    
                    with col3_inner:
                        if st.button("🗑️", key=f"delete_saved_{search['search_id']}", help="Delete"):
                            remove_saved_search(search['search_id'])
                            st.rerun()
    
    # Handle repeat search
    if 'repeat_search' in st.session_state:
        st.info(f"🔄 Repeating search: {st.session_state.repeat_search}")
        # You could auto-fill the search box or trigger the search here
        del st.session_state.repeat_search


def save_search(search_entry: Dict):
    """Save a search query for later use"""
    
    if 'saved_searches' not in st.session_state:
        st.session_state.saved_searches = []
    
    # Check if already saved
    existing = [s for s in st.session_state.saved_searches if s['query'] == search_entry['query']]
    if existing:
        st.warning(f"Search '{search_entry['query']}' is already saved")
        return
    
    saved_entry = {
        **search_entry,
        "saved_at": datetime.now().isoformat()
    }
    
    st.session_state.saved_searches.append(saved_entry)
    st.success(f"Search '{search_entry['query']}' saved!")


def remove_saved_search(search_id: int):
    """Remove a saved search"""
    
    if 'saved_searches' in st.session_state:
        st.session_state.saved_searches = [
            s for s in st.session_state.saved_searches 
            if s['search_id'] != search_id
        ]
        st.success("Saved search removed!") 