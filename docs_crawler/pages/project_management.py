"""
Project Management Page
Handles project creation, viewing, and management functionality
"""

import streamlit as st
import pandas as pd
import asyncio
import json
import csv
from datetime import datetime
from urllib.parse import urlparse
from components.ui_components import create_project_card, display_status_badge
from components.supabase_integration import (
    get_supabase_integration, 
    create_new_project, 
    get_project_list
)


def validate_url(url):
    """Validate URL format and accessibility"""
    try:
        result = urlparse(url)
        # Check if URL has scheme and netloc
        if not all([result.scheme, result.netloc]):
            return False, "Invalid URL format"
        
        # Check if scheme is http or https
        if result.scheme not in ['http', 'https']:
            return False, "URL must start with http:// or https://"
        
        # Check for common invalid patterns
        if '..' in url or result.netloc == 'localhost' and 'localhost' not in st.secrets.get('allowed_hosts', []):
            return False, "URL contains invalid patterns"
        
        return True, "Valid URL"
    except Exception as e:
        return False, f"URL validation error: {str(e)}"


def show():
    """Display the project management page"""
    
    st.header("📁 Project Management")
    st.markdown("Create and manage your Crawl4AI projects")
    
    # Get Supabase integration instance
    supabase = get_supabase_integration()
    
    # Project creation section
    with st.expander("🆕 Create New Project", expanded=False):
        st.subheader("Project Creation Workflow")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            project_name = st.text_input(
                "Project Name",
                placeholder="Enter project name or leave blank for auto-generation",
                help="Will be auto-generated from domain if left empty"
            )
            
            project_url = st.text_input(
                "Initial URL to Crawl",
                placeholder="https://example.com",
                help="Starting URL for your crawling project"
            )
            
            # Real-time URL validation
            if project_url:
                is_valid, message = validate_url(project_url)
                if is_valid:
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")
            
            project_description = st.text_area(
                "Project Description (Optional)",
                placeholder="Describe the purpose of this project...",
                height=100
            )
        
        with col2:
            st.markdown("### Configuration")
            st.info("🔢 Supabase project will be created automatically via MCP server")
            
            # Configuration options
            max_depth = st.slider("Maximum Crawl Depth", 1, 5, 2)
            max_concurrent = st.slider("Concurrent Sessions", 5, 50, 10)
            chunk_size = st.slider("Chunk Size", 1000, 10000, 4000, step=500)
            
            # Store config in session state
            if 'crawl_config' not in st.session_state:
                st.session_state.crawl_config = {}
            
            st.session_state.crawl_config.update({
                'max_depth': max_depth,
                'max_concurrent': max_concurrent,
                'chunk_size': chunk_size
            })
            
            # Estimated cost display
            st.markdown("### Estimated Resources")
            st.metric("Est. Storage", f"{(chunk_size / 1000) * 10:.1f} MB per 1000 pages")
            st.metric("Est. Processing Time", f"{100 / max_concurrent:.1f} min per 100 pages")
        
        if st.button("🚀 Create Project", type="primary"):
            if project_url:
                is_valid, message = validate_url(project_url)
                if not is_valid:
                    st.error(message)
                    return
                
                # Generate project name from URL if not provided
                if not project_name:
                    domain = urlparse(project_url).netloc
                    project_name = domain.replace('www.', '').replace('.', '_')
                
                # Create progress container for real-time updates
                progress_container = st.container()
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Step 1: Validate project name
                        progress_bar.progress(20)
                        status_text.text("🔍 Validating project name...")
                        
                        # Step 2: Create Supabase project
                        progress_bar.progress(40)
                        status_text.text("🏗️ Creating Supabase project via MCP server...")
                        
                        # Run async function in sync context
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        project_config = loop.run_until_complete(
                            create_new_project(project_name, project_url)
                        )
                        
                        # Step 3: Initialize database
                        progress_bar.progress(60)
                        status_text.text("🗄️ Initializing database schema...")
                        
                        # Step 4: Test connection
                        progress_bar.progress(80)
                        status_text.text("🔗 Testing database connection...")
                        
                        if supabase.test_connection(project_config.project_id):
                            progress_bar.progress(100)
                            status_text.text("✅ Project created successfully!")
                            
                            # Store initial URL and config
                            st.session_state.initial_url = project_url
                            st.session_state.project_description = project_description
                            st.session_state.selected_project_id = project_config.project_id
                            
                            st.success(f"✅ Project '{project_config.name}' created successfully!")
                            st.info(f"Project ID: {project_config.project_id}")
                            st.info(f"Supabase URL: {project_config.supabase_url}")
                            
                            st.balloons()
                            st.rerun()
                        else:
                            st.error("Failed to verify database connection")
                            
                    except Exception as e:
                        st.error(f"Error creating project: {str(e)}")
                        progress_bar.empty()
                        status_text.empty()
            else:
                st.error("Please provide a valid URL")
    
    # Project dashboard section
    st.markdown("---")
    st.subheader("📊 Project Dashboard")
    
    # Get real projects from Supabase integration
    projects = get_project_list()
    
    if projects:
        # Project statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Projects", len(projects))
        with col2:
            active_projects = len([p for p in projects if p.total_documents > 0])
            st.metric("Active Projects", active_projects)
        with col3:
            total_docs = sum(p.total_documents for p in projects)
            st.metric("Total Documents", f"{total_docs:,}")
        with col4:
            total_storage = sum(p.storage_used for p in projects)
            st.metric("Storage Used", f"{total_storage:.1f} MB")
        
        st.markdown("---")
        
        # Export all projects data
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("📤 Export All Projects", type="secondary"):
                export_projects_data(projects)
        
        # Project cards
        for i, project in enumerate(projects):
            # Convert ProjectConfig to dict for display
            project_dict = {
                "id": project.project_id,
                "name": project.name,
                "status": "Active" if project.total_documents > 0 else "Ready",
                "created": project.created_at.strftime("%Y-%m-%d"),
                "documents": project.total_documents,
                "storage_mb": project.storage_used,
                "last_crawled": project.last_crawled.strftime("%Y-%m-%d %H:%M") if project.last_crawled else "Never",
                "supabase_url": project.supabase_url,
                "rag_strategies": project.rag_strategies
            }
            
            # Create expandable card for each project
            with st.expander(f"📁 {project.name}", expanded=False):
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.markdown(f"**Project ID:** `{project.project_id}`")
                    st.markdown(f"**Created:** {project_dict['created']}")
                    st.markdown(f"**Last Crawled:** {project_dict['last_crawled']}")
                    
                with col2:
                    st.markdown(f"**Documents:** {project_dict['documents']:,}")
                    st.markdown(f"**Storage:** {project_dict['storage_mb']:.1f} MB")
                    st.markdown(f"**Status:** {display_status_badge(project_dict['status'])}")
                
                with col3:
                    st.markdown("**Actions**")
                    
                    if st.button("🔗 Test", key=f"test_{project.project_id}", help="Test database connection"):
                        with st.spinner("Testing connection..."):
                            if supabase.test_connection(project.project_id):
                                st.success("✅ Connection successful!")
                            else:
                                st.error("❌ Connection failed!")
                    
                    if st.button("🕷️ Crawl", key=f"crawl_{project.project_id}", help="Start crawling"):
                        st.session_state.selected_project_id = project.project_id
                        st.session_state.page = "Crawl Content"
                        st.info("Navigate to 'Crawl Content' page to start crawling")
                    
                    if st.button("🔍 Search", key=f"search_{project.project_id}", help="Search content"):
                        st.session_state.selected_project_id = project.project_id
                        st.session_state.page = "Search Interface"
                        st.info("Navigate to 'Search Interface' page to search")
                    
                    if st.button("📤 Export", key=f"export_{project.project_id}", help="Export project data"):
                        export_single_project_data(project)
                    
                    if st.button("🗑️ Delete", key=f"delete_{project.project_id}", help="Delete project"):
                        st.warning("⚠️ Click again to confirm deletion")
                        if st.button("Confirm Delete", key=f"confirm_delete_{project.project_id}"):
                            with st.spinner("Deleting project..."):
                                if supabase.delete_project(project.project_id):
                                    st.success(f"Project '{project.name}' deleted successfully!")
                                    st.rerun()
                
                # Show RAG strategies
                st.markdown("**RAG Strategies:**")
                for strategy in project.rag_strategies:
                    st.markdown(f"- {strategy}")
                    
                # Show crawl configuration if stored
                if 'crawl_config' in st.session_state and st.session_state.get('selected_project_id') == project.project_id:
                    st.markdown("**Crawl Configuration:**")
                    config = st.session_state.crawl_config
                    st.markdown(f"- Max Depth: {config.get('max_depth', 'N/A')}")
                    st.markdown(f"- Concurrent Sessions: {config.get('max_concurrent', 'N/A')}")
                    st.markdown(f"- Chunk Size: {config.get('chunk_size', 'N/A')}")
    
    else:
        st.info("🎯 No projects yet. Create your first project above!")
        
    # Recent activity section
    st.markdown("---")
    st.subheader("📈 Recent Activity")
    
    if projects:
        # Show activity summary
        activity_summary = []
        for project in projects:
            if project.last_crawled:
                activity_summary.append({
                    "Timestamp": project.last_crawled.strftime("%Y-%m-%d %H:%M"),
                    "Project": project.name,
                    "Activity": "Last crawled",
                    "Details": f"{project.total_documents} documents indexed"
                })
        
        if activity_summary:
            df = pd.DataFrame(activity_summary)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No recent activity to display")
    else:
        st.info("No projects to show activity for")


def export_projects_data(projects):
    """Export all projects data to JSON"""
    export_data = []
    for project in projects:
        export_data.append({
            "id": project.project_id,
            "name": project.name,
            "created": project.created_at.isoformat(),
            "last_crawled": project.last_crawled.isoformat() if project.last_crawled else None,
            "documents": project.total_documents,
            "storage_mb": project.storage_used,
            "rag_strategies": project.rag_strategies
        })
    
    json_data = json.dumps(export_data, indent=2)
    st.download_button(
        label="📥 Download Projects Data (JSON)",
        data=json_data,
        file_name=f"crawl4ai_projects_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )


def export_single_project_data(project):
    """Export single project data"""
    export_data = {
        "project_info": {
            "id": project.project_id,
            "name": project.name,
            "created": project.created_at.isoformat(),
            "last_crawled": project.last_crawled.isoformat() if project.last_crawled else None,
            "documents": project.total_documents,
            "storage_mb": project.storage_used,
            "rag_strategies": project.rag_strategies
        },
        "configuration": {
            "supabase_url": project.supabase_url,
            "mcp_server_config": project.mcp_server_config
        }
    }
    
    json_data = json.dumps(export_data, indent=2)
    st.download_button(
        label="📥 Download Project Data",
        data=json_data,
        file_name=f"{project.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        key=f"download_{project.project_id}"
    )


def handle_project_action(project_id: str, action: str):
    """Handle project actions (crawl, search, export, etc.)"""
    
    if action == "crawl":
        st.info(f"🕷️ Starting crawl for project {project_id}")
        # Navigate to crawl content page
        
    elif action == "search":
        st.info(f"🔍 Opening search for project {project_id}")
        # Navigate to search interface page
        
    elif action == "export":
        st.info(f"📤 Exporting data for project {project_id}")
        # Handle export functionality
        
    elif action == "delete":
        if st.confirm(f"Are you sure you want to delete project {project_id}?"):
            st.success(f"🗑️ Project {project_id} deleted successfully")
            st.rerun() 