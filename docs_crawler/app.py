"""
Crawl4AI Standalone Application
Main Streamlit application entry point
"""

import streamlit as st
from streamlit_option_menu import option_menu
import os
import sys

# Add the current directory to the Python path for component imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main application entry point"""
    
    # Configure page settings
    st.set_page_config(
        page_title="Crawl4AI Standalone",
        page_icon="🕷️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sidebar-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .status-badge {
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        font-weight: bold;
    }
    .status-active { background-color: #d4edda; color: #155724; }
    .status-inactive { background-color: #f8d7da; color: #721c24; }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">🕷️ Crawl4AI Standalone</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown('<div class="sidebar-title">Navigation</div>', unsafe_allow_html=True)
        
        selected = option_menu(
            menu_title="Main Menu",
            options=["Project Management", "Crawl Content", "Search Interface", "Administration"],
            icons=["folder", "download", "search", "gear"],
            menu_icon="list",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "#1f77b4", "font-size": "18px"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "#1f77b4"},
            }
        )
        
        # System status in sidebar
        st.markdown("---")
        st.markdown("### System Status")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<span class="status-badge status-active">Online</span>', unsafe_allow_html=True)
        with col2:
            st.markdown('<span class="status-badge status-active">Ready</span>', unsafe_allow_html=True)
    
    # Route to selected page
    if selected == "Project Management":
        from pages import project_management
        project_management.show()
    elif selected == "Crawl Content":
        from pages import crawl_content
        crawl_content.show()
    elif selected == "Search Interface":
        from pages import search_interface
        search_interface.show()
    elif selected == "Administration":
        from pages import administration
        administration.show()

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.8rem;'>
        <p>Crawl4AI Standalone Application | Built with Streamlit | 
        <a href="https://github.com/unclecode/crawl4ai" target="_blank">Powered by Crawl4AI</a></p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 