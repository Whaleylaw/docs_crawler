"""
Search Interface Page
Handles search queries and result display
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from components.ui_components import display_search_result, create_filter_sidebar

def show():
    """Display the search interface page"""
    
    st.header("🔍 Search Interface")
    st.markdown("Search through your crawled content and code examples")
    
    # Project selection
    project_options = ["Documentation Site", "API Reference", "All Projects"]
    selected_project = st.selectbox(
        "Search in Project",
        project_options,
        help="Choose which project to search in"
    )
    
    # Search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "",
            placeholder="Enter your search query...",
            help="Search for content, code examples, or specific topics",
            label_visibility="collapsed"
        )
        
        # Quick search buttons
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            if st.button("🔍 Search", type="primary", use_container_width=True):
                if search_query:
                    perform_search(search_query, selected_project)
                else:
                    st.warning("Please enter a search query")
        
        with col_b:
            if st.button("🧹 Clear", use_container_width=True):
                st.rerun()
        
        with col_c:
            if st.button("💾 Save Query", use_container_width=True):
                save_search_query(search_query)
        
        with col_d:
            if st.button("📝 Code Search", use_container_width=True):
                st.session_state.code_search_mode = True
                st.rerun()
    
    with col2:
        search_mode = st.radio(
            "Search Mode",
            ["Simple Search", "Advanced Search", "Code Examples"],
            help="Choose your search approach"
        )
    
    # Advanced filters (expandable)
    if search_mode == "Advanced Search":
        with st.expander("🔧 Advanced Filters", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                source_domains = st.multiselect(
                    "Source Domains",
                    ["docs.example.com", "api.example.com", "blog.example.com"],
                    help="Filter by source domain"
                )
                
                content_types = st.multiselect(
                    "Content Types",
                    ["Documentation", "API Reference", "Blog Posts", "Code Examples"],
                    help="Filter by content type"
                )
            
            with col2:
                similarity_threshold = st.slider(
                    "Similarity Threshold",
                    0.0, 1.0, 0.7, 0.1,
                    help="Minimum similarity score for results"
                )
                
                result_count = st.slider(
                    "Max Results",
                    5, 50, 10,
                    help="Maximum number of results to return"
                )
            
            with col3:
                date_range = st.date_input(
                    "Date Range",
                    value=None,
                    help="Filter by content date range"
                )
                
                language_filter = st.selectbox(
                    "Programming Language",
                    ["All", "Python", "JavaScript", "Java", "C++", "Go", "Rust"],
                    help="Filter code examples by language"
                )
    
    # Code search interface
    elif search_mode == "Code Examples":
        st.markdown("---")
        st.subheader("💻 Code Example Search")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            code_query = st.text_area(
                "Describe the code you're looking for",
                placeholder="e.g., 'function to authenticate users with JWT tokens'",
                height=100,
                help="Describe what kind of code pattern you need"
            )
        
        with col2:
            st.markdown("#### Filters")
            code_language = st.selectbox(
                "Language",
                ["All", "Python", "JavaScript", "TypeScript", "Java", "C++", "Go"],
                help="Filter by programming language"
            )
            
            include_context = st.checkbox(
                "Include Documentation Context",
                value=True,
                help="Show surrounding documentation with code"
            )
    
    # Search results
    if st.session_state.get('search_performed', False):
        st.markdown("---")
        display_search_results()
    
    # Search history
    st.markdown("---")
    st.subheader("📚 Search History")
    display_search_history()


def perform_search(query: str, project: str):
    """Execute search and store results"""
    st.session_state.search_performed = True
    st.session_state.current_query = query
    st.session_state.current_project = project
    
    with st.spinner("Searching..."):
        # Simulate search delay
        import time
        time.sleep(1)
    
    st.success(f"✅ Search completed for: '{query}'")
    st.rerun()


def display_search_results():
    """Display search results with relevance scoring"""
    
    query = st.session_state.get('current_query', '')
    project = st.session_state.get('current_project', '')
    
    st.subheader(f"🎯 Search Results for: '{query}'")
    st.markdown(f"*Searching in: {project}*")
    
    # Mock search results
    search_results = [
        {
            "title": "User Authentication with JWT",
            "content": "This guide explains how to implement JWT authentication in your application. JWT (JSON Web Tokens) provide a secure way to authenticate users...",
            "source": "https://docs.example.com/auth/jwt",
            "similarity": 0.94,
            "content_type": "Documentation",
            "date": "2025-01-15"
        },
        {
            "title": "Authentication Middleware",
            "content": "```python\ndef authenticate_token(token):\n    try:\n        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])\n        return payload\n    except jwt.ExpiredSignatureError:\n        return None\n```",
            "source": "https://api.example.com/middleware",
            "similarity": 0.87,
            "content_type": "Code Example",
            "date": "2025-01-12"
        },
        {
            "title": "Setting up Authentication Headers",
            "content": "When making API requests, you need to include the authentication token in the headers. Here's how to properly configure headers for different HTTP clients...",
            "source": "https://docs.example.com/api/headers",
            "similarity": 0.82,
            "content_type": "API Reference",
            "date": "2025-01-10"
        }
    ]
    
    # Results summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Results", len(search_results))
    with col2:
        avg_similarity = sum(r['similarity'] for r in search_results) / len(search_results)
        st.metric("Avg Similarity", f"{avg_similarity:.2f}")
    with col3:
        response_time = "0.85s"  # Mock response time
        st.metric("Response Time", response_time)
    with col4:
        st.metric("Sources", len(set(r['source'] for r in search_results)))
    
    # Display results
    for i, result in enumerate(search_results):
        with st.container():
            # Result header
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"### {result['title']}")
            
            with col2:
                # Similarity score with color coding
                score_color = "green" if result['similarity'] > 0.8 else "orange" if result['similarity'] > 0.6 else "red"
                st.markdown(f"<span style='color: {score_color}; font-weight: bold;'>📊 {result['similarity']:.2f}</span>", unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"**{result['content_type']}**")
            
            # Content preview
            if result['content_type'] == "Code Example":
                st.code(result['content'], language='python')
            else:
                st.markdown(result['content'])
            
            # Source and metadata
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"🔗 [Source]({result['source']})")
            with col2:
                st.markdown(f"📅 {result['date']}")
            with col3:
                if st.button("📋 Copy", key=f"copy_{i}"):
                    st.info("Content copied to clipboard!")
            with col4:
                if st.button("🔗 Share", key=f"share_{i}"):
                    st.info("Share link generated!")
            
            st.markdown("---")
    
    # Export options
    st.markdown("### 📤 Export Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export as CSV", use_container_width=True):
            # Create CSV download
            df = pd.DataFrame(search_results)
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📋 Export as JSON", use_container_width=True):
            import json
            json_data = json.dumps(search_results, indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col3:
        if st.button("📧 Email Results", use_container_width=True):
            st.info("Email feature coming soon!")


def display_search_history():
    """Display previous search queries"""
    
    # Mock search history
    history_data = [
        {"timestamp": "2025-01-20 14:30", "query": "JWT authentication", "project": "Documentation Site", "results": 8},
        {"timestamp": "2025-01-20 09:15", "query": "API endpoints", "project": "API Reference", "results": 15},
        {"timestamp": "2025-01-19 16:45", "query": "database connection", "project": "All Projects", "results": 23},
    ]
    
    if history_data:
        # Display as expandable items
        for i, item in enumerate(history_data):
            with st.expander(f"🔍 '{item['query']}' - {item['timestamp']} ({item['results']} results)"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Project:** {item['project']}")
                    st.write(f"**Results:** {item['results']}")
                
                with col2:
                    if st.button("🔄 Repeat Search", key=f"repeat_{i}"):
                        st.session_state.search_query = item['query']
                        perform_search(item['query'], item['project'])
                
                with col3:
                    if st.button("🗑️ Delete", key=f"delete_history_{i}"):
                        st.success("Search history item deleted")
    else:
        st.info("No search history yet. Start searching to build your history!")


def save_search_query(query: str):
    """Save search query for later use"""
    if query:
        st.success(f"✅ Query '{query}' saved to your favorites!")
    else:
        st.warning("Please enter a query to save") 