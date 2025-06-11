"""
UI Components
Reusable Streamlit UI components for the Crawl4AI application
"""

import streamlit as st
from datetime import datetime
from typing import Dict, Any, List


def create_project_card(project: Dict[str, Any], key: str = None):
    """Create a project card display"""
    
    with st.container():
        # Project header
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"### {project['name']}")
            st.markdown(f"*Created: {project['created']}*")
        
        with col2:
            status_color = {"Active": "green", "Paused": "orange", "Inactive": "red"}
            color = status_color.get(project['status'], "gray")
            st.markdown(f"<span style='color: {color}; font-weight: bold;'>● {project['status']}</span>", 
                       unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"**ID:** {project['id']}")
        
        # Project metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Documents", f"{project['documents']:,}")
        with col2:
            st.metric("Storage", f"{project['storage_mb']:.1f} MB")
        with col3:
            st.metric("Last Crawled", project['last_crawled'])
        
        # Action buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🕷️ Crawl", key=f"crawl_{key}", use_container_width=True):
                st.info(f"Starting crawl for {project['name']}")
        
        with col2:
            if st.button("🔍 Search", key=f"search_{key}", use_container_width=True):
                st.info(f"Opening search for {project['name']}")
        
        with col3:
            if st.button("📤 Export", key=f"export_{key}", use_container_width=True):
                st.info(f"Exporting data for {project['name']}")
        
        with col4:
            if st.button("⚙️ Settings", key=f"settings_{key}", use_container_width=True):
                st.info(f"Opening settings for {project['name']}")
        
        st.markdown("---")


def display_status_badge(status: str, size: str = "small") -> str:
    """Create a colored status badge"""
    
    status_colors = {
        "active": "#28a745",
        "pending": "#ffc107", 
        "inactive": "#dc3545",
        "completed": "#28a745",
        "failed": "#dc3545",
        "running": "#007bff"
    }
    
    color = status_colors.get(status.lower(), "#6c757d")
    font_size = "12px" if size == "small" else "14px"
    
    return f'<span style="background-color: {color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: {font_size}; font-weight: bold;">{status}</span>'


def create_progress_indicator(current: int, total: int, label: str = "Progress"):
    """Create a progress indicator with metrics"""
    
    progress = current / total if total > 0 else 0
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.progress(progress, text=f"{label}: {current}/{total} ({progress:.1%})")
    
    with col2:
        st.metric("Remaining", total - current)


def display_crawl_status(crawl_data: Dict[str, Any]):
    """Display crawling status information"""
    
    status_color = {
        "running": "blue",
        "completed": "green", 
        "failed": "red",
        "paused": "orange"
    }
    
    color = status_color.get(crawl_data.get('status', '').lower(), "gray")
    
    st.markdown(f"**Status:** <span style='color: {color};'>● {crawl_data.get('status', 'Unknown')}</span>", 
                unsafe_allow_html=True)
    
    if 'start_time' in crawl_data:
        st.markdown(f"**Started:** {crawl_data['start_time']}")
    
    if 'duration' in crawl_data:
        st.markdown(f"**Duration:** {crawl_data['duration']}")


def display_search_result(result: Dict[str, Any], index: int):
    """Display a single search result"""
    
    with st.container():
        # Result header
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"### {result.get('title', 'Untitled')}")
        
        with col2:
            similarity = result.get('similarity', 0)
            score_color = "green" if similarity > 0.8 else "orange" if similarity > 0.6 else "red"
            st.markdown(f"<span style='color: {score_color}; font-weight: bold;'>📊 {similarity:.2f}</span>", 
                       unsafe_allow_html=True)
        
        # Content preview
        content = result.get('content', '')
        if len(content) > 300:
            content = content[:300] + "..."
        
        st.markdown(content)
        
        # Metadata
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"🔗 [Source]({result.get('source', '#')})")
        
        with col2:
            st.markdown(f"📅 {result.get('date', 'Unknown')}")
        
        with col3:
            content_type = result.get('content_type', 'Unknown')
            st.markdown(f"📋 {content_type}")
        
        st.markdown("---")


def create_filter_sidebar():
    """Create a filter sidebar for search"""
    
    with st.sidebar:
        st.markdown("### Filters")
        
        # Date range filter
        date_range = st.date_input(
            "Date Range",
            value=None,
            help="Filter by content date"
        )
        
        # Content type filter
        content_types = st.multiselect(
            "Content Types",
            ["Documentation", "Code Examples", "API Reference", "Blog Posts"],
            help="Filter by content type"
        )
        
        # Source filter
        sources = st.multiselect(
            "Sources",
            ["docs.example.com", "api.example.com", "blog.example.com"],
            help="Filter by source domain"
        )
        
        # Similarity threshold
        similarity_threshold = st.slider(
            "Min Similarity",
            0.0, 1.0, 0.5, 0.1,
            help="Minimum similarity score"
        )
        
        return {
            "date_range": date_range,
            "content_types": content_types,
            "sources": sources,
            "similarity_threshold": similarity_threshold
        }


def display_metric_card(title: str, value: str, change: str = None, change_color: str = "normal"):
    """Display a metric card with optional change indicator"""
    
    with st.container():
        st.markdown(f"### {title}")
        st.markdown(f"**{value}**")
        
        if change:
            color = {"positive": "green", "negative": "red", "normal": "gray"}[change_color]
            st.markdown(f"<span style='color: {color};'>{change}</span>", unsafe_allow_html=True)


def create_status_indicator(title: str, status: str, color: str):
    """Create a status indicator with title and colored status"""
    st.markdown(f"**{title}**")
    st.markdown(f"<span style='color: {color}; font-size: 14px;'>● {status}</span>", unsafe_allow_html=True)


def create_notification(message: str, type: str = "info"):
    """Create a notification message"""
    
    if type == "success":
        st.success(message)
    elif type == "warning":
        st.warning(message)
    elif type == "error":
        st.error(message)
    else:
        st.info(message)


def create_loading_spinner(message: str = "Loading..."):
    """Create a loading spinner with message"""
    
    with st.spinner(message):
        import time
        time.sleep(1)  # Simulate loading


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_names[i]}"


def format_duration(seconds: int) -> str:
    """Format duration in human readable format"""
    
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds}s"
    else:
        hours = seconds // 3600
        remaining_minutes = (seconds % 3600) // 60
        return f"{hours}h {remaining_minutes}m"


def create_data_table(data: List[Dict], columns: List[str] = None, searchable: bool = True):
    """Create an interactive data table"""
    
    import pandas as pd
    
    if not data:
        st.info("No data to display")
        return
    
    df = pd.DataFrame(data)
    
    if columns:
        df = df[columns]
    
    if searchable:
        search_term = st.text_input("Search in table:", key="table_search")
        if search_term:
            mask = df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
            df = df[mask]
    
    st.dataframe(df, use_container_width=True, hide_index=True)


def create_export_button(data: Any, filename: str, format: str = "csv"):
    """Create an export button for data"""
    
    if format == "csv":
        import pandas as pd
        if isinstance(data, list):
            df = pd.DataFrame(data)
            csv_data = df.to_csv(index=False)
        else:
            csv_data = str(data)
        
        st.download_button(
            label=f"📥 Export as CSV",
            data=csv_data,
            file_name=f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    elif format == "json":
        import json
        json_data = json.dumps(data, indent=2, default=str)
        
        st.download_button(
            label=f"📥 Export as JSON",
            data=json_data,
            file_name=f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        ) 