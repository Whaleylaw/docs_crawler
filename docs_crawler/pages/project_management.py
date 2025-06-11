"""
Project Management Page
Handles project creation, viewing, and management functionality
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from components.ui_components import create_project_card, display_status_badge

def show():
    """Display the project management page"""
    
    st.header("📁 Project Management")
    st.markdown("Create and manage your Crawl4AI projects")
    
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
            
            project_description = st.text_area(
                "Project Description (Optional)",
                placeholder="Describe the purpose of this project...",
                height=100
            )
        
        with col2:
            st.markdown("### Cost Estimation")
            st.info("🔢 Estimated costs will appear here based on your configuration")
            
            # Configuration options
            max_depth = st.slider("Maximum Crawl Depth", 1, 5, 2)
            max_concurrent = st.slider("Concurrent Sessions", 5, 50, 10)
            chunk_size = st.slider("Chunk Size", 1000, 10000, 4000, step=500)
        
        if st.button("🚀 Create Project", type="primary"):
            if project_url:
                with st.spinner("Creating project..."):
                    # Placeholder for project creation logic
                    st.success(f"✅ Project created successfully!")
                    st.rerun()
            else:
                st.error("Please provide a valid URL")
    
    # Project dashboard section
    st.markdown("---")
    st.subheader("📊 Project Dashboard")
    
    # Sample projects for demonstration
    sample_projects = [
        {
            "id": "proj_001",
            "name": "Documentation Site",
            "status": "Active",
            "created": "2025-01-15",
            "documents": 1250,
            "storage_mb": 45.2,
            "last_crawled": "2025-01-20"
        },
        {
            "id": "proj_002", 
            "name": "API Reference",
            "status": "Paused",
            "created": "2025-01-10",
            "documents": 890,
            "storage_mb": 32.7,
            "last_crawled": "2025-01-18"
        }
    ]
    
    if sample_projects:
        # Project statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Projects", len(sample_projects))
        with col2:
            active_projects = len([p for p in sample_projects if p["status"] == "Active"])
            st.metric("Active Projects", active_projects)
        with col3:
            total_docs = sum(p["documents"] for p in sample_projects)
            st.metric("Total Documents", f"{total_docs:,}")
        with col4:
            total_storage = sum(p["storage_mb"] for p in sample_projects)
            st.metric("Storage Used", f"{total_storage:.1f} MB")
        
        st.markdown("---")
        
        # Project cards
        for i, project in enumerate(sample_projects):
            create_project_card(project, key=f"project_{i}")
    
    else:
        st.info("🎯 No projects yet. Create your first project above!")
        
    # Recent activity section
    st.markdown("---")
    st.subheader("📈 Recent Activity")
    
    # Sample activity data
    activity_data = [
        {"timestamp": "2025-01-20 14:30", "project": "Documentation Site", "action": "Crawl completed", "details": "125 new pages indexed"},
        {"timestamp": "2025-01-20 09:15", "project": "API Reference", "action": "Search query", "details": "User searched for 'authentication'"},
        {"timestamp": "2025-01-19 16:45", "project": "Documentation Site", "action": "Project created", "details": "Initial setup completed"},
    ]
    
    if activity_data:
        df = pd.DataFrame(activity_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No recent activity to display")


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