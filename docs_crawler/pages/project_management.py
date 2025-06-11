"""
Project Management Page
Handles project creation, viewing, and management functionality
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Optional
from components.ui_components import create_project_card, display_status_badge
from components.supabase_integration import (
    get_supabase_integration, 
    create_new_project, 
    get_project_list,
    ProjectConfig
)

def show():
    """Display the project management page"""
    
    st.header("📁 Project Management")
    st.markdown("Create and manage your Crawl4AI projects")
    
    # Get the Supabase integration instance
    supabase_integration = get_supabase_integration()
    
    # Project creation section
    with st.expander("🆕 Create New Project", expanded=False):
        st.subheader("Project Creation Workflow")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            project_name = st.text_input(
                "Project Name",
                placeholder="Enter project name",
                help="Name for your crawling project"
            )
            
            # Supabase credentials section
            st.markdown("#### Supabase Connection")
            supabase_url = st.text_input(
                "Supabase URL",
                placeholder="https://your-project.supabase.co",
                help="Your Supabase project URL"
            )
            
            supabase_key = st.text_input(
                "Supabase Service Key",
                placeholder="Your service role key",
                type="password",
                help="Your Supabase service role key (starts with 'eyJ...')"
            )
            
            # Validation feedback
            if supabase_url and supabase_key:
                if supabase_integration.validate_supabase_credentials(supabase_url, supabase_key):
                    st.success("✅ Supabase credentials appear valid")
                else:
                    st.error("❌ Invalid Supabase credentials")
            
            project_url = st.text_input(
                "Initial URL to Crawl (Optional)",
                placeholder="https://example.com",
                help="Starting URL for your crawling project"
            )
            
            project_description = st.text_area(
                "Project Description (Optional)",
                placeholder="Describe the purpose of this project...",
                height=100
            )
        
        with col2:
            st.markdown("### Configuration")
            
            # Configuration options
            max_depth = st.slider("Maximum Crawl Depth", 1, 5, 2)
            max_concurrent = st.slider("Concurrent Sessions", 5, 50, 10)
            chunk_size = st.slider("Chunk Size", 1000, 10000, 4000, step=500)
            
            st.markdown("### Cost Estimation")
            st.info("💡 Estimated costs depend on:\n- Number of pages crawled\n- OpenAI API usage for embeddings\n- Supabase storage usage")
        
        if st.button("🚀 Create Project", type="primary"):
            if project_name and supabase_url and supabase_key:
                with st.spinner("Creating project and setting up database..."):
                    try:
                        # Create the project
                        config = create_new_project(
                            name=project_name,
                            supabase_url=supabase_url,
                            supabase_key=supabase_key,
                            initial_url=project_url
                        )
                        
                        st.success(f"✅ Project '{project_name}' created successfully!")
                        st.success(f"📊 Project ID: {config.project_id}")
                        st.info("🔄 Database schema has been initialized. You can now start crawling content.")
                        
                        # Clear the form
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Failed to create project: {str(e)}")
                        if "credentials" in str(e).lower():
                            st.error("Please check your Supabase URL and service key")
                        elif "schema" in str(e).lower():
                            st.error("Database schema initialization failed. Check Supabase permissions.")
            else:
                st.error("Please provide project name, Supabase URL, and service key")
    
    # Project dashboard section
    st.markdown("---")
    st.subheader("📊 Project Dashboard")
    
    # Get actual projects
    projects = get_project_list()
    
    if projects:
        # Project statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Projects", len(projects))
        with col2:
            active_projects = len([p for p in projects if p.last_crawled is not None])
            st.metric("Active Projects", active_projects)
        with col3:
            total_docs = sum(p.total_documents for p in projects)
            st.metric("Total Documents", f"{total_docs:,}")
        with col4:
            total_storage = sum(p.storage_used for p in projects)
            st.metric("Storage Used", f"{total_storage:.1f} MB")
        
        st.markdown("---")
        
        # Project cards
        for i, project in enumerate(projects):
            create_project_card_enhanced(project, key=f"project_{i}")
    
    else:
        st.info("🎯 No projects yet. Create your first project above!")
        
    # Recent activity section - for now showing placeholder
    # TODO: Implement actual activity tracking in Task 8
    if projects:
        st.markdown("---")
        st.subheader("📈 Recent Activity")
        
        # Generate sample activity from existing projects
        activity_data = []
        for project in projects[-3:]:  # Show last 3 projects
            if project.last_crawled:
                activity_data.append({
                    "timestamp": project.last_crawled.strftime("%Y-%m-%d %H:%M"),
                    "project": project.name,
                    "action": "Last crawl activity",
                    "details": f"{project.total_documents} documents indexed"
                })
            activity_data.append({
                "timestamp": project.created_at.strftime("%Y-%m-%d %H:%M"),
                "project": project.name,
                "action": "Project created",
                "details": "Initial setup completed"
            })
        
        if activity_data:
            df = pd.DataFrame(activity_data)
            df = df.sort_values('timestamp', ascending=False)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No recent activity to display")


def create_project_card_enhanced(project: ProjectConfig, key: str):
    """Create an enhanced project card with real data and actions"""
    
    with st.container():
        # Card styling
        st.markdown(f"""
        <div style="
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        ">
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"### 🗂️ {project.name}")
            st.markdown(f"**Project ID:** `{project.project_id}`")
            st.markdown(f"**Created:** {project.created_at.strftime('%Y-%m-%d')}")
            
            # Status badge
            status = "Active" if project.last_crawled else "Created"
            color = "🟢" if project.last_crawled else "🟡"
            st.markdown(f"**Status:** {color} {status}")
        
        with col2:
            st.markdown("#### 📊 Metrics")
            st.metric("Documents", f"{project.total_documents:,}")
            st.metric("Storage", f"{project.storage_used:.1f} MB")
            
            if project.last_crawled:
                days_ago = (datetime.now() - project.last_crawled).days
                st.metric("Last Crawl", f"{days_ago} days ago" if days_ago > 0 else "Today")
        
        with col3:
            st.markdown("#### ⚡ Actions")
            
            # Action buttons
            col_a, col_b = st.columns(2)
            
            with col_a:
                if st.button("🕷️ Crawl", key=f"crawl_{key}"):
                    handle_project_action(project.project_id, "crawl")
                
                if st.button("🔍 Search", key=f"search_{key}"):
                    handle_project_action(project.project_id, "search")
            
            with col_b:
                if st.button("📤 Export", key=f"export_{key}"):
                    handle_project_action(project.project_id, "export")
                
                if st.button("🗑️ Delete", key=f"delete_{key}"):
                    handle_project_action(project.project_id, "delete")
            
            # Connection test
            if st.button("🔗 Test Connection", key=f"test_{key}"):
                supabase_integration = get_supabase_integration()
                if supabase_integration.test_connection(project.project_id):
                    st.success("✅ Connection successful")
                else:
                    st.error("❌ Connection failed")
        
        st.markdown("</div>", unsafe_allow_html=True)


def handle_project_action(project_id: str, action: str):
    """Handle project actions (crawl, search, export, etc.)"""
    
    supabase_integration = get_supabase_integration()
    
    if action == "crawl":
        st.info(f"🕷️ Redirecting to crawl interface for project {project_id}")
        # Store the selected project in session state for the crawl page
        st.session_state.selected_project_id = project_id
        # TODO: Implement page navigation (requires updating main app routing)
        
    elif action == "search":
        st.info(f"🔍 Redirecting to search interface for project {project_id}")
        # Store the selected project in session state for the search page  
        st.session_state.selected_project_id = project_id
        # TODO: Implement page navigation
        
    elif action == "export":
        st.info(f"📤 Export functionality for project {project_id}")
        # TODO: Implement export functionality in later tasks
        
    elif action == "delete":
        # Confirmation dialog
        st.warning(f"⚠️ Are you sure you want to delete project {project_id}?")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Confirm Delete", key=f"confirm_delete_{project_id}"):
                if supabase_integration.delete_project(project_id):
                    st.success(f"🗑️ Project {project_id} deleted successfully")
                    st.rerun()
                else:
                    st.error("Failed to delete project")
        
        with col2:
            if st.button("❌ Cancel", key=f"cancel_delete_{project_id}"):
                st.info("Delete cancelled")
                st.rerun() 