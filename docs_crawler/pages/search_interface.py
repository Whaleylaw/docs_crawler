"""
Search Interface Page
Handles search queries and result display
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from components.ui_components import display_search_result, create_filter_sidebar
from components.supabase_integration import get_supabase_integration, get_project_list

def show():
    """Display the search interface page"""
    
    st.header("🔍 Search Interface")
    st.markdown("Search through your crawled content using AI-powered vector search")
    
    # Get available projects
    projects = get_project_list()
    
    if not projects:
        st.warning("⚠️ No projects available. Please create a project and crawl some content first.")
        if st.button("➕ Go to Project Management"):
            st.switch_page("project_management")
        return
    
    # Project selection
    project_options = [f"{p.name} ({p.project_id})" for p in projects] + ["All Projects"]
    
    selected_project_idx = st.selectbox(
        "Search in Project",
        range(len(project_options)),
        format_func=lambda x: project_options[x],
        help="Choose which project to search in"
    )
    
    # Determine selected project
    if selected_project_idx < len(projects):
        selected_project = projects[selected_project_idx]
        search_all_projects = False
    else:
        selected_project = None
        search_all_projects = True
    
    # Check if there's a selected project from project management page
    if 'selected_project_id' in st.session_state and not search_all_projects:
        for i, project in enumerate(projects):
            if project.project_id == st.session_state.selected_project_id:
                selected_project_idx = i
                selected_project = project
                break
        # Clear the session state
        del st.session_state.selected_project_id
    
    # Display project info
    if not search_all_projects:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Project", selected_project.name)
        with col2:
            st.metric("Documents", selected_project.total_documents)
        with col3:
            st.metric("Last Updated", selected_project.last_crawled.strftime("%m-%d %H:%M") if selected_project.last_crawled else "Never")
    
    # Search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "",
            placeholder="Enter your search query... (e.g., 'authentication with JWT tokens')",
            help="Search for content, code examples, or specific topics using natural language",
            label_visibility="collapsed",
            key="main_search_input"
        )
        
        # Quick search buttons
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            if st.button("🔍 Search", type="primary", use_container_width=True):
                if search_query:
                    perform_search(search_query, selected_project, search_all_projects)
                else:
                    st.warning("Please enter a search query")
        
        with col_b:
            if st.button("🧹 Clear", use_container_width=True):
                clear_search_results()
        
        with col_c:
            if st.button("💾 Save Query", use_container_width=True):
                save_search_query(search_query)
        
        with col_d:
            if st.button("📝 Code Search", use_container_width=True):
                st.session_state.search_mode = "Code Examples"
                st.rerun()
    
    with col2:
        search_mode = st.radio(
            "Search Mode",
            ["Simple Search", "Advanced Search", "Code Examples"],
            index=["Simple Search", "Advanced Search", "Code Examples"].index(st.session_state.get('search_mode', 'Simple Search')),
            help="Choose your search approach"
        )
        st.session_state.search_mode = search_mode
    
    # Advanced filters (expandable)
    advanced_filters = {}
    if search_mode == "Advanced Search":
        with st.expander("🔧 Advanced Filters", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Get unique source domains from projects
                all_sources = []
                if search_all_projects:
                    for project in projects:
                        # This would ideally come from querying the database
                        all_sources.extend([f"{project.name}.domain.com"])
                else:
                    all_sources = [f"{selected_project.name}.domain.com"]
                
                source_domains = st.multiselect(
                    "Source Domains",
                    all_sources,
                    help="Filter by source domain"
                )
                advanced_filters['source_domains'] = source_domains
                
                content_types = st.multiselect(
                    "Content Types",
                    ["Documentation", "API Reference", "Blog Posts", "Code Examples"],
                    help="Filter by content type"
                )
                advanced_filters['content_types'] = content_types
            
            with col2:
                similarity_threshold = st.slider(
                    "Similarity Threshold",
                    0.0, 1.0, 0.3, 0.1,
                    help="Minimum similarity score for results"
                )
                advanced_filters['similarity_threshold'] = similarity_threshold
                
                result_count = st.slider(
                    "Max Results",
                    5, 50, 10,
                    help="Maximum number of results to return"
                )
                advanced_filters['result_count'] = result_count
            
            with col3:
                date_range = st.date_input(
                    "Date Range",
                    value=None,
                    help="Filter by content creation date"
                )
                advanced_filters['date_range'] = date_range
                
                language_filter = st.selectbox(
                    "Programming Language",
                    ["All", "Python", "JavaScript", "Java", "C++", "Go", "Rust"],
                    help="Filter code examples by language"
                )
                advanced_filters['language_filter'] = language_filter
    else:
        # Default values for simple search
        advanced_filters = {
            'result_count': 10,
            'similarity_threshold': 0.3
        }
    
    # Code search interface
    if search_mode == "Code Examples":
        st.markdown("---")
        st.subheader("💻 Code Example Search")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            code_query = st.text_area(
                "Describe the code you're looking for",
                placeholder="e.g., 'function to authenticate users with JWT tokens'",
                height=100,
                help="Describe what kind of code pattern you need in natural language"
            )
            
            if st.button("🔍 Search Code", type="primary"):
                if code_query:
                    perform_code_search(code_query, selected_project, search_all_projects, advanced_filters)
                else:
                    st.warning("Please describe the code you're looking for")
        
        with col2:
            st.markdown("#### Filters")
            code_language = st.selectbox(
                "Language",
                ["All", "Python", "JavaScript", "TypeScript", "Java", "C++", "Go"],
                help="Filter by programming language"
            )
            advanced_filters['code_language'] = code_language
            
            include_context = st.checkbox(
                "Include Documentation Context",
                value=True,
                help="Show surrounding documentation with code"
            )
            advanced_filters['include_context'] = include_context
    
    # Search results
    if st.session_state.get('search_results'):
        st.markdown("---")
        display_search_results()
    
    # Search history
    st.markdown("---")
    st.subheader("📚 Search History")
    display_search_history()


def perform_search(query: str, project, search_all_projects: bool, filters: Dict = None):
    """Execute search and store results"""
    
    if filters is None:
        filters = {'result_count': 10, 'similarity_threshold': 0.3}
    
    try:
        supabase_integration = get_supabase_integration()
        all_results = []
        
        with st.spinner("🔍 Searching through your content..."):
            if search_all_projects:
                # Search across all projects
                projects = get_project_list()
                for proj in projects:
                    try:
                        results = supabase_integration.search_documents(
                            project_id=proj.project_id,
                            query=query,
                            match_count=filters.get('result_count', 10),
                            filter_metadata=build_metadata_filter(filters)
                        )
                        # Add project info to results
                        for result in results:
                            result['project_name'] = proj.name
                            result['project_id'] = proj.project_id
                        all_results.extend(results)
                    except Exception as e:
                        st.warning(f"Could not search project {proj.name}: {e}")
                        continue
            else:
                # Search in specific project
                results = supabase_integration.search_documents(
                    project_id=project.project_id,
                    query=query,
                    match_count=filters.get('result_count', 10),
                    filter_metadata=build_metadata_filter(filters)
                )
                # Add project info to results
                for result in results:
                    result['project_name'] = project.name
                    result['project_id'] = project.project_id
                all_results = results
        
        # Filter by similarity threshold
        similarity_threshold = filters.get('similarity_threshold', 0.3)
        filtered_results = [r for r in all_results if r.get('similarity', 0) >= similarity_threshold]
        
        # Sort by similarity score
        filtered_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        
        # Store results in session state
        st.session_state.search_results = filtered_results
        st.session_state.current_query = query
        st.session_state.search_timestamp = datetime.now()
        st.session_state.search_project = "All Projects" if search_all_projects else project.name
        
        # Add to search history
        add_to_search_history(query, st.session_state.search_project, len(filtered_results))
        
        if filtered_results:
            st.success(f"✅ Found {len(filtered_results)} results for: '{query}'")
        else:
            st.info(f"No results found for: '{query}'. Try adjusting your search terms or filters.")
        
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Search failed: {e}")


def perform_code_search(query: str, project, search_all_projects: bool, filters: Dict):
    """Execute code-specific search"""
    
    # For now, use the same search but with code-specific prompting
    enhanced_query = f"code example: {query}"
    perform_search(enhanced_query, project, search_all_projects, filters)


def build_metadata_filter(filters: Dict) -> Optional[Dict]:
    """Build metadata filter from advanced search options"""
    
    metadata_filter = {}
    
    if filters.get('content_types'):
        metadata_filter['content_type'] = filters['content_types']
    
    if filters.get('language_filter') and filters['language_filter'] != "All":
        metadata_filter['language'] = filters['language_filter']
    
    if filters.get('date_range'):
        # Convert date range to filter
        pass  # TODO: Implement date filtering
    
    return metadata_filter if metadata_filter else None


def display_search_results():
    """Display search results with relevance scoring"""
    
    results = st.session_state.get('search_results', [])
    query = st.session_state.get('current_query', '')
    search_project = st.session_state.get('search_project', '')
    
    if not results:
        return
    
    st.subheader(f"🎯 Search Results for: '{query}'")
    st.markdown(f"*Searching in: {search_project}*")
    
    # Results summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Results", len(results))
    with col2:
        avg_similarity = sum(r.get('similarity', 0) for r in results) / len(results) if results else 0
        st.metric("Avg Similarity", f"{avg_similarity:.3f}")
    with col3:
        search_time = st.session_state.get('search_timestamp')
        if search_time:
            time_ago = datetime.now() - search_time
            st.metric("Search Time", f"{time_ago.total_seconds():.1f}s ago")
    with col4:
        unique_sources = len(set(r.get('url', '') for r in results))
        st.metric("Unique Sources", unique_sources)
    
    # Display results
    for i, result in enumerate(results):
        with st.container():
            # Result header
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                # Extract title from content or URL
                title = result.get('metadata', {}).get('title', '') or f"Result from {result.get('url', 'Unknown source')}"
                st.markdown(f"### {title}")
            
            with col2:
                # Similarity score with color coding
                similarity = result.get('similarity', 0)
                score_color = "green" if similarity > 0.8 else "orange" if similarity > 0.6 else "red"
                st.markdown(f"<span style='color: {score_color}; font-weight: bold;'>📊 {similarity:.3f}</span>", unsafe_allow_html=True)
            
            with col3:
                project_name = result.get('project_name', 'Unknown')
                st.markdown(f"**📁 {project_name}**")
            
            # Content preview
            content = result.get('content', '')
            if content:
                # Check if it's code content
                if '```' in content or result.get('metadata', {}).get('content_type') == 'code':
                    # Try to detect language
                    language = result.get('metadata', {}).get('language', 'text')
                    st.code(content[:1000] + "..." if len(content) > 1000 else content, language=language.lower())
                else:
                    # Regular text content
                    st.markdown(content[:500] + "..." if len(content) > 500 else content)
            
            # Metadata
            metadata = result.get('metadata', {})
            if metadata:
                with st.expander("ℹ️ Metadata"):
                    col1, col2 = st.columns(2)
                    with col1:
                        if 'word_count' in metadata:
                            st.write(f"**Word Count:** {metadata['word_count']}")
                        if 'headers' in metadata and metadata['headers']:
                            st.write(f"**Headers:** {metadata['headers']}")
                    with col2:
                        if 'crawled_at' in metadata:
                            st.write(f"**Crawled:** {metadata['crawled_at']}")
                        if 'chunk_size' in metadata:
                            st.write(f"**Chunk Size:** {metadata['chunk_size']}")
            
            # Source and actions
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if result.get('url'):
                    st.markdown(f"🔗 [View Source]({result['url']})")
            with col2:
                if st.button("📋 Copy Content", key=f"copy_{i}"):
                    st.session_state[f'copied_{i}'] = True
                    st.success("Content copied!")
            with col3:
                if st.button("🔗 Share Result", key=f"share_{i}"):
                    share_url = f"?query={query}&result={i}"
                    st.info(f"Share URL: {share_url}")
            with col4:
                if st.button("🔄 Similar", key=f"similar_{i}"):
                    # Find similar content based on this result
                    similar_query = content[:100] + "..."
                    st.info(f"Searching for similar content...")
            
            st.markdown("---")
    
    # Export options
    if results:
        st.markdown("### 📤 Export Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Export as CSV", use_container_width=True):
                export_results_csv(results, query)
        
        with col2:
            if st.button("📋 Export as JSON", use_container_width=True):
                export_results_json(results, query)
        
        with col3:
            if st.button("📝 Create Summary", use_container_width=True):
                create_search_summary(results, query)


def export_results_csv(results: List[Dict], query: str):
    """Export search results as CSV"""
    
    export_data = []
    for result in results:
        export_data.append({
            'Query': query,
            'Similarity': result.get('similarity', 0),
            'Content': result.get('content', '')[:500],  # Truncate for CSV
            'URL': result.get('url', ''),
            'Project': result.get('project_name', ''),
            'Title': result.get('metadata', {}).get('title', ''),
            'Word Count': result.get('metadata', {}).get('word_count', ''),
        })
    
    df = pd.DataFrame(export_data)
    csv = df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )


def export_results_json(results: List[Dict], query: str):
    """Export search results as JSON"""
    
    export_data = {
        'query': query,
        'timestamp': datetime.now().isoformat(),
        'result_count': len(results),
        'results': results
    }
    
    json_data = json.dumps(export_data, indent=2, default=str)
    
    st.download_button(
        label="📥 Download JSON",
        data=json_data,
        file_name=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )


def create_search_summary(results: List[Dict], query: str):
    """Create a summary of search results"""
    
    st.info("📝 Summary generation coming soon! This will use AI to summarize the search results.")


def clear_search_results():
    """Clear current search results"""
    
    if 'search_results' in st.session_state:
        del st.session_state.search_results
    if 'current_query' in st.session_state:
        del st.session_state.current_query
    st.rerun()


def add_to_search_history(query: str, project: str, result_count: int):
    """Add search to history"""
    
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    
    # Add new search to beginning of history
    new_search = {
        'timestamp': datetime.now(),
        'query': query,
        'project': project,
        'results': result_count
    }
    
    st.session_state.search_history.insert(0, new_search)
    
    # Keep only last 20 searches
    st.session_state.search_history = st.session_state.search_history[:20]


def display_search_history():
    """Display previous search queries"""
    
    history = st.session_state.get('search_history', [])
    
    if history:
        # Display as expandable items
        for i, item in enumerate(history):
            timestamp_str = item['timestamp'].strftime("%Y-%m-%d %H:%M")
            with st.expander(f"🔍 '{item['query']}' - {timestamp_str} ({item['results']} results)"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Project:** {item['project']}")
                    st.write(f"**Results:** {item['results']}")
                    time_ago = datetime.now() - item['timestamp']
                    if time_ago.days > 0:
                        st.write(f"**Time:** {time_ago.days}d ago")
                    else:
                        st.write(f"**Time:** {time_ago.seconds//3600}h ago")
                
                with col2:
                    if st.button("🔄 Repeat Search", key=f"repeat_{i}"):
                        # Repeat the search
                        projects = get_project_list()
                        if item['project'] == "All Projects":
                            perform_search(item['query'], None, True)
                        else:
                            # Find the project
                            project = next((p for p in projects if p.name == item['project']), None)
                            if project:
                                perform_search(item['query'], project, False)
                            else:
                                st.error("Project not found")
                
                with col3:
                    if st.button("🗑️ Delete", key=f"delete_history_{i}"):
                        st.session_state.search_history.pop(i)
                        st.rerun()
    else:
        st.info("No search history yet. Start searching to build your history!")


def save_search_query(query: str):
    """Save search query for later use"""
    
    if query:
        if 'saved_queries' not in st.session_state:
            st.session_state.saved_queries = []
        
        if query not in st.session_state.saved_queries:
            st.session_state.saved_queries.append(query)
            st.success(f"✅ Query '{query}' saved to your favorites!")
        else:
            st.info(f"Query '{query}' is already in your favorites!")
    else:
        st.warning("Please enter a query to save") 