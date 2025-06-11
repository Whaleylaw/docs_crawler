"""
Crawl4AI Standalone Application
Main Streamlit application entry point
"""

import streamlit as st
from pages import home, project_management, crawl_content, search_interface, content_analysis
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
        padding: 1rem 0;
        border-bottom: 2px solid #f0f2f6;
        margin-bottom: 2rem;
    }
    
    .sidebar .sidebar-content {
        padding-top: 1rem;
    }
    
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
    }
    
    .success-banner {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    .warning-banner {
        background-color: #fff3cd;
        border-color: #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    .info-card {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .nav-button {
        width: 100%;
        margin: 0.25rem 0;
        padding: 0.5rem;
        text-align: left;
        border: none;
        background: transparent;
        cursor: pointer;
        border-radius: 0.25rem;
    }
    
    .nav-button:hover {
        background-color: #f0f2f6;
    }
    
    .nav-button.active {
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">🕷️ Crawl4AI Standalone</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar navigation
    with st.sidebar:
        st.title("🕷️ Crawl4AI Standalone")
        st.markdown("---")
        
        # Navigation menu
        pages = {
            "🏠 Home": "home",
            "📁 Project Management": "project_management", 
            "🕷️ Crawl Content": "crawl_content",
            "🔍 Search Interface": "search_interface",
            "📊 Content Analysis": "content_analysis"
        }
        
        # Get current page from session state
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'home'
        
        # Create navigation buttons
        for page_name, page_key in pages.items():
            if st.button(page_name, key=f"nav_{page_key}", use_container_width=True):
                st.session_state.current_page = page_key
                st.rerun()
        
        st.markdown("---")
        
        # Application info
        st.markdown("### 📋 Quick Info")
        st.info("""
        **Crawl4AI Standalone**
        
        A complete web crawling and RAG solution built with:
        • 🚀 Streamlit UI
        • 🕷️ Crawl4AI Engine  
        • 🗄️ Supabase Storage
        • 🧠 OpenAI Integration
        • 📊 Advanced Analytics
        """)
        
        # Version info
        st.caption("v1.0.0 | Built with ❤️ for developers")
    
    # Main content area
    current_page = st.session_state.current_page
    
    # Route to appropriate page
    if current_page == 'home':
        home.show()
    elif current_page == 'project_management':
        project_management.show()
    elif current_page == 'crawl_content':
        crawl_content.show()
    elif current_page == 'search_interface':
        search_interface.show()
    elif current_page == 'content_analysis':
        content_analysis.show()
    else:
        # Default to home
        home.show()

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