"""
Content Analysis Page
Provides comprehensive content analysis, preview, and quality metrics
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from components.content_analysis import get_content_analyzer, ContentVisualization, ANALYSIS_DEPENDENCIES_AVAILABLE
from components.supabase_integration import get_supabase_integration, get_project_list


def show():
    """Display the content analysis page"""
    
    st.header("📊 Content Analysis & Preview")
    st.markdown("Analyze content quality, detect duplicates, and gain insights from your crawled data")
    
    # Get available projects
    projects = get_project_list()
    
    if not projects:
        st.warning("⚠️ No projects available. Please create a project and crawl some content first.")
        if st.button("➕ Go to Project Management"):
            st.switch_page("project_management")
        return
    
    # Project selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        project_options = [f"{p.name} ({p.project_id})" for p in projects]
        selected_project_idx = st.selectbox(
            "Select Project to Analyze",
            range(len(project_options)),
            format_func=lambda x: project_options[x],
            help="Choose a project to analyze its content"
        )
        
        selected_project = projects[selected_project_idx]
    
    with col2:
        st.metric("Total Documents", selected_project.total_documents)
        if st.button("🔄 Refresh Data"):
            st.rerun()
    
    # Analysis tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Overview", 
        "🔍 Document Explorer", 
        "📊 Quality Analysis", 
        "🔄 Duplicate Detection",
        "📈 Analytics Dashboard"
    ])
    
    # Load documents for analysis
    documents = load_project_documents(selected_project.project_id)
    analyzer = get_content_analyzer()
    
    with tab1:
        display_content_overview(documents, analyzer, selected_project)
    
    with tab2:
        display_document_explorer(documents, analyzer, selected_project)
    
    with tab3:
        display_quality_analysis(documents, analyzer, selected_project)
    
    with tab4:
        display_duplicate_detection(documents, analyzer, selected_project)
    
    with tab5:
        display_analytics_dashboard(documents, analyzer, selected_project)


def load_project_documents(project_id: str) -> List[Dict[str, Any]]:
    """Load documents from the selected project"""
    
    try:
        supabase_integration = get_supabase_integration()
        
        # Get all documents for the project
        client = supabase_integration.get_supabase_client(project_id)
        
        if client:
            # Fetch documents with metadata
            response = client.table('crawled_pages').select('*').execute()
            return response.data if response.data else []
        else:
            return []
            
    except Exception as e:
        st.error(f"Failed to load documents: {e}")
        return []


def display_content_overview(documents: List[Dict], analyzer, project):
    """Display high-level content overview"""
    
    st.subheader("📋 Content Overview")
    
    if not documents:
        st.info("No documents found in this project. Start by crawling some content!")
        return
    
    # Calculate metrics
    with st.spinner("Analyzing content corpus..."):
        metrics = analyzer.analyze_content_corpus(documents)
    
    # Key metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Documents",
            f"{metrics.total_documents:,}",
            help="Total number of crawled documents"
        )
    
    with col2:
        st.metric(
            "Total Words",
            f"{metrics.total_words:,}",
            help="Total word count across all documents"
        )
    
    with col3:
        st.metric(
            "Avg Document Length",
            f"{metrics.avg_document_length:.0f} words",
            help="Average length per document"
        )
    
    with col4:
        quality_color = "normal"
        if metrics.readability_score > 0.7:
            quality_color = "normal"
        elif metrics.readability_score < 0.3:
            quality_color = "inverse"
        
        st.metric(
            "Readability Score",
            f"{metrics.readability_score:.2f}",
            delta_color=quality_color,
            help="Content readability (0-1, higher is better)"
        )
    
    # Distribution charts
    st.markdown("---")
    st.subheader("📊 Content Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if metrics.content_type_distribution:
            st.markdown("#### Content Types")
            df_types = pd.DataFrame(
                list(metrics.content_type_distribution.items()),
                columns=['Content Type', 'Count']
            )
            fig = px.pie(df_types, values='Count', names='Content Type', 
                        title='Content Type Distribution')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No content type data available")
    
    with col2:
        if metrics.source_distribution:
            st.markdown("#### Top Sources")
            source_items = list(metrics.source_distribution.items())[:10]
            df_sources = pd.DataFrame(source_items, columns=['Source', 'Count'])
            fig = px.bar(df_sources, x='Count', y='Source', orientation='h',
                        title='Top 10 Content Sources')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No source distribution data available")
    
    # Content freshness
    if metrics.content_freshness:
        st.markdown("#### Content Freshness")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Last 24h", metrics.content_freshness.get('last_24h', 0))
        with col2:
            st.metric("Last Week", metrics.content_freshness.get('last_week', 0))
        with col3:
            st.metric("Last Month", metrics.content_freshness.get('last_month', 0))
        with col4:
            st.metric("Older", metrics.content_freshness.get('older', 0))
    
    # Quick actions
    st.markdown("---")
    st.subheader("🚀 Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📄 Export Overview", use_container_width=True):
            export_overview_report(metrics, project)
    
    with col2:
        if st.button("🔍 Find Duplicates", use_container_width=True):
            st.session_state.active_tab = "🔄 Duplicate Detection"
            st.rerun()
    
    with col3:
        if st.button("📊 Quality Report", use_container_width=True):
            st.session_state.active_tab = "📊 Quality Analysis"
            st.rerun()
    
    with col4:
        if st.button("📈 View Analytics", use_container_width=True):
            st.session_state.active_tab = "📈 Analytics Dashboard"
            st.rerun()


def display_document_explorer(documents: List[Dict], analyzer, project):
    """Display individual document explorer with previews"""
    
    st.subheader("🔍 Document Explorer")
    
    if not documents:
        st.info("No documents to explore")
        return
    
    # Search and filter controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_query = st.text_input(
            "Search Documents",
            placeholder="Search by title, URL, or content...",
            help="Filter documents by keywords"
        )
    
    with col2:
        sort_options = ["Latest", "Quality Score", "Word Count", "Title"]
        sort_by = st.selectbox("Sort By", sort_options)
    
    with col3:
        content_type_filter = st.selectbox(
            "Content Type",
            ["All"] + list(set(analyzer._detect_content_type(doc.get('content', '')) for doc in documents))
        )
    
    # Filter documents
    filtered_docs = documents
    
    if search_query:
        search_lower = search_query.lower()
        filtered_docs = [
            doc for doc in filtered_docs
            if (search_lower in doc.get('content', '').lower() or
                search_lower in doc.get('url', '').lower() or
                search_lower in doc.get('metadata', {}).get('title', '').lower())
        ]
    
    if content_type_filter != "All":
        filtered_docs = [
            doc for doc in filtered_docs
            if analyzer._detect_content_type(doc.get('content', '')) == content_type_filter
        ]
    
    # Sort documents
    if sort_by == "Latest":
        filtered_docs.sort(key=lambda x: x.get('metadata', {}).get('crawled_at', ''), reverse=True)
    elif sort_by == "Quality Score":
        # Calculate quality scores for sorting
        for doc in filtered_docs:
            doc['_quality_score'] = analyzer._calculate_quality_score(
                doc.get('content', ''), doc.get('metadata', {})
            )
        filtered_docs.sort(key=lambda x: x.get('_quality_score', 0), reverse=True)
    elif sort_by == "Word Count":
        filtered_docs.sort(key=lambda x: x.get('metadata', {}).get('word_count', 0), reverse=True)
    elif sort_by == "Title":
        filtered_docs.sort(key=lambda x: x.get('metadata', {}).get('title', '').lower())
    
    st.info(f"Showing {len(filtered_docs)} of {len(documents)} documents")
    
    # Display documents with pagination
    docs_per_page = 10
    total_pages = (len(filtered_docs) + docs_per_page - 1) // docs_per_page
    
    if total_pages > 1:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            page = st.selectbox("Page", range(1, total_pages + 1), format_func=lambda x: f"Page {x}")
        
        start_idx = (page - 1) * docs_per_page
        end_idx = start_idx + docs_per_page
        page_docs = filtered_docs[start_idx:end_idx]
    else:
        page_docs = filtered_docs[:docs_per_page]
    
    # Display documents
    for i, doc in enumerate(page_docs):
        with st.expander(f"📄 {doc.get('metadata', {}).get('title', f'Document {i+1}')}", expanded=False):
            
            # Generate preview
            preview = analyzer.generate_content_preview(doc)
            
            # Document info
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**URL:** [{doc.get('url', 'N/A')}]({doc.get('url', '#')})")
                st.markdown(f"**Content Type:** {preview['content_type']}")
                st.markdown(f"**Reading Time:** ~{preview['estimated_reading_time']} min")
                
                # Content preview
                st.markdown("**Preview:**")
                st.write(preview['summary'])
                
                # Headers
                if preview['headers']:
                    st.markdown("**Headers:**")
                    for header in preview['headers'][:5]:
                        st.write(f"• {header}")
                
                # Key phrases
                if preview['key_phrases']:
                    st.markdown("**Key Phrases:**")
                    for phrase in preview['key_phrases'][:3]:
                        st.code(f'"{phrase}"', language="text")
            
            with col2:
                # Metrics
                st.metric("Words", f"{preview['word_count']:,}")
                
                # Quality analysis
                quality_score = analyzer._calculate_quality_score(
                    doc.get('content', ''), doc.get('metadata', {})
                )
                st.metric("Quality Score", f"{quality_score:.2f}")
                
                # Links and code blocks
                if preview['links']:
                    st.metric("Links Found", len(preview['links']))
                
                if preview['code_blocks']:
                    st.metric("Code Blocks", len(preview['code_blocks']))
                    
                    # Show code block languages
                    languages = list(set(block['language'] for block in preview['code_blocks']))
                    if languages:
                        st.markdown("**Languages:**")
                        for lang in languages:
                            st.code(lang, language="text")
                
                # Actions
                col1_action, col2_action = st.columns(2)
                
                with col1_action:
                    if st.button("🔗 Open", key=f"open_doc_{i}"):
                        st.link_button("Open URL", doc.get('url', '#'))
                
                with col2_action:
                    if st.button("📊 Analyze", key=f"analyze_doc_{i}"):
                        analyze_single_document(doc, analyzer)


def analyze_single_document(document: Dict, analyzer):
    """Perform detailed analysis of a single document"""
    
    with st.spinner("Analyzing document..."):
        analysis = analyzer.analyze_single_document(document)
    
    st.markdown("### 📊 Document Analysis Results")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Word Count", f"{analysis.word_count:,}")
    with col2:
        st.metric("Readability", f"{analysis.readability_score:.2f}")
    with col3:
        st.metric("Complexity", f"{analysis.complexity_score:.2f}")
    with col4:
        st.metric("Quality Score", f"{analysis.quality_score:.2f}")
    
    # Key topics
    if analysis.key_topics:
        st.markdown("**Key Topics:**")
        topic_cols = st.columns(min(5, len(analysis.key_topics)))
        for i, topic in enumerate(analysis.key_topics[:5]):
            with topic_cols[i % len(topic_cols)]:
                st.code(topic)
    
    # Analysis summary
    st.json({
        "analysis_timestamp": analysis.analysis_timestamp.isoformat(),
        "content_hash": analysis.content_hash,
        "url": analysis.url
    })


def display_quality_analysis(documents: List[Dict], analyzer, project):
    """Display comprehensive quality analysis"""
    
    st.subheader("📊 Quality Analysis")
    
    if not documents:
        st.info("No documents to analyze")
        return
    
    # Generate quality report
    with st.spinner("Generating quality report..."):
        quality_report = analyzer.create_quality_report(documents)
    
    if 'error' in quality_report:
        st.error(quality_report['error'])
        return
    
    # Quality overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_quality = quality_report['average_quality']
        color = "normal" if avg_quality > 0.6 else "inverse"
        st.metric("Average Quality", f"{avg_quality:.2f}", delta_color=color)
    
    with col2:
        st.metric("Total Issues", quality_report['total_issues'])
    
    with col3:
        excellent_count = quality_report['quality_distribution']['excellent']
        st.metric("Excellent Docs", excellent_count)
    
    with col4:
        poor_count = quality_report['quality_distribution']['poor']
        st.metric("Poor Quality Docs", poor_count)
    
    # Quality distribution chart
    st.markdown("---")
    st.subheader("📈 Quality Distribution")
    
    quality_dist = quality_report['quality_distribution']
    df_quality = pd.DataFrame([
        {'Quality Level': 'Excellent (0.8+)', 'Count': quality_dist['excellent']},
        {'Quality Level': 'Good (0.6-0.8)', 'Count': quality_dist['good']},
        {'Quality Level': 'Fair (0.4-0.6)', 'Count': quality_dist['fair']},
        {'Quality Level': 'Poor (<0.4)', 'Count': quality_dist['poor']}
    ])
    
    fig = px.bar(df_quality, x='Quality Level', y='Count', 
                title='Document Quality Distribution',
                color='Count', color_continuous_scale='RdYlGn')
    st.plotly_chart(fig, use_container_width=True)
    
    # Quality issues
    if quality_report['issues_by_document']:
        st.markdown("---")
        st.subheader("⚠️ Quality Issues")
        
        # Group issues by type
        issue_summary = {}
        for doc_issue in quality_report['issues_by_document']:
            for issue in doc_issue['issues']:
                if issue not in issue_summary:
                    issue_summary[issue] = []
                issue_summary[issue].append(doc_issue)
        
        # Display issue summary
        for issue_type, affected_docs in issue_summary.items():
            with st.expander(f"{issue_type} (affects {len(affected_docs)} documents)"):
                for doc in affected_docs[:10]:  # Show first 10
                    st.write(f"• [{doc['title']}]({doc['url']})")
                if len(affected_docs) > 10:
                    st.write(f"... and {len(affected_docs) - 10} more")
    
    # Recommendations
    if quality_report['recommendations']:
        st.markdown("---")
        st.subheader("💡 Recommendations")
        
        for recommendation in quality_report['recommendations']:
            st.info(recommendation)
    
    # Export quality report
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Export Quality Report", use_container_width=True):
            export_quality_report(quality_report, project)
    
    with col2:
        if st.button("🔧 Fix Common Issues", use_container_width=True):
            st.info("🚧 Auto-fix functionality coming soon!")


def display_duplicate_detection(documents: List[Dict], analyzer, project):
    """Display duplicate detection results"""
    
    st.subheader("🔄 Duplicate Detection")
    
    if not documents:
        st.info("No documents to analyze for duplicates")
        return
    
    # Duplicate detection settings
    col1, col2 = st.columns([2, 1])
    
    with col1:
        similarity_threshold = st.slider(
            "Similarity Threshold",
            0.5, 1.0, 0.85, 0.05,
            help="Higher values detect only very similar documents"
        )
    
    with col2:
        detection_method = st.selectbox(
            "Detection Method",
            ["Content Hash", "Text Similarity"],
            help="Method for duplicate detection"
        )
    
    # Detect duplicates
    if st.button("🔍 Detect Duplicates", type="primary"):
        with st.spinner("Detecting duplicate content..."):
            duplicate_groups = analyzer.detect_duplicates(documents, similarity_threshold)
        
        if not duplicate_groups:
            st.success("🎉 No duplicates found!")
        else:
            st.warning(f"Found {len(duplicate_groups)} groups of duplicate documents")
            
            total_duplicates = sum(len(group) - 1 for group in duplicate_groups)
            duplicate_percentage = (total_duplicates / len(documents)) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Duplicate Groups", len(duplicate_groups))
            with col2:
                st.metric("Total Duplicates", total_duplicates)
            with col3:
                st.metric("Duplicate %", f"{duplicate_percentage:.1f}%")
            
            # Display duplicate groups
            st.markdown("---")
            st.subheader("📄 Duplicate Document Groups")
            
            for i, group in enumerate(duplicate_groups):
                with st.expander(f"Group {i+1}: {len(group)} duplicates", expanded=i < 3):
                    
                    for j, doc in enumerate(group):
                        col1, col2, col3 = st.columns([3, 1, 1])
                        
                        with col1:
                            title = doc.get('metadata', {}).get('title', f'Document {j+1}')
                            st.write(f"**{title}**")
                            st.caption(doc.get('url', 'No URL'))
                        
                        with col2:
                            word_count = doc.get('metadata', {}).get('word_count', 0)
                            st.metric("Words", word_count)
                        
                        with col3:
                            if j > 0 and st.button(f"🗑️ Remove", key=f"remove_{i}_{j}"):
                                st.info("🚧 Duplicate removal coming soon!")
                    
                    # Group actions
                    st.markdown("**Group Actions:**")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button(f"📊 Compare All", key=f"compare_{i}"):
                            compare_duplicate_group(group)
                    
                    with col2:
                        if st.button(f"📄 Export Group", key=f"export_{i}"):
                            export_duplicate_group(group, i+1)
                    
                    with col3:
                        if st.button(f"🔗 Merge Docs", key=f"merge_{i}"):
                            st.info("🚧 Document merging coming soon!")


def compare_duplicate_group(group: List[Dict]):
    """Compare documents in a duplicate group"""
    
    st.markdown("### 📊 Document Comparison")
    
    comparison_data = []
    for i, doc in enumerate(group):
        metadata = doc.get('metadata', {})
        comparison_data.append({
            'Document': f"Doc {i+1}",
            'Title': metadata.get('title', 'Untitled'),
            'URL': doc.get('url', ''),
            'Word Count': metadata.get('word_count', 0),
            'Char Count': metadata.get('char_count', 0),
            'Crawled At': metadata.get('crawled_at', '')
        })
    
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True)


def display_analytics_dashboard(documents: List[Dict], analyzer, project):
    """Display comprehensive analytics dashboard"""
    
    st.subheader("📈 Analytics Dashboard")
    
    if not documents:
        st.info("No documents available for analytics")
        return
    
    # Calculate metrics
    with st.spinner("Generating analytics..."):
        metrics = analyzer.analyze_content_corpus(documents)
    
    # Time-based analysis
    if metrics.crawl_frequency:
        st.markdown("#### 📅 Crawling Activity Timeline")
        
        # Convert to DataFrame for plotting
        dates = list(metrics.crawl_frequency.keys())
        counts = list(metrics.crawl_frequency.values())
        
        df_timeline = pd.DataFrame({
            'Date': pd.to_datetime(dates),
            'Documents': counts
        })
        
        fig = px.line(df_timeline, x='Date', y='Documents', 
                     title='Documents Crawled Over Time',
                     markers=True)
        st.plotly_chart(fig, use_container_width=True)
    
    # Advanced visualizations
    if ANALYSIS_DEPENDENCIES_AVAILABLE:
        # Word cloud
        try:
            st.markdown("#### ☁️ Content Word Cloud")
            word_cloud_fig = ContentVisualization.create_word_cloud(documents)
            if word_cloud_fig:
                st.pyplot(word_cloud_fig)
            else:
                st.info("Word cloud generation not available")
        except:
            st.info("Word cloud generation failed")
    
    # Content complexity analysis
    st.markdown("#### 🧠 Content Complexity Analysis")
    
    complexity_scores = []
    readability_scores = []
    
    for doc in documents:
        content = doc.get('content', '')
        complexity = analyzer._calculate_document_complexity(content)
        readability = analyzer._calculate_readability(content) if ANALYSIS_DEPENDENCIES_AVAILABLE else 0.5
        
        complexity_scores.append(complexity)
        readability_scores.append(readability)
    
    df_complexity = pd.DataFrame({
        'Document': range(1, len(documents) + 1),
        'Complexity': complexity_scores,
        'Readability': readability_scores
    })
    
    fig = px.scatter(df_complexity, x='Readability', y='Complexity',
                    title='Document Complexity vs Readability',
                    hover_data=['Document'])
    st.plotly_chart(fig, use_container_width=True)
    
    # Export analytics
    st.markdown("---")
    if st.button("📊 Export Analytics Report", use_container_width=True):
        export_analytics_report(metrics, project)


def export_overview_report(metrics, project):
    """Export content overview report"""
    
    report_data = {
        'project': project.name,
        'generated_at': datetime.now().isoformat(),
        'metrics': {
            'total_documents': metrics.total_documents,
            'total_words': metrics.total_words,
            'avg_document_length': metrics.avg_document_length,
            'readability_score': metrics.readability_score,
            'complexity_score': metrics.complexity_score,
            'duplicate_percentage': metrics.duplicate_percentage
        },
        'distributions': {
            'content_types': metrics.content_type_distribution,
            'sources': metrics.source_distribution,
            'languages': metrics.language_distribution
        }
    }
    
    import json
    report_json = json.dumps(report_data, indent=2, default=str)
    
    st.download_button(
        label="📥 Download Overview Report",
        data=report_json,
        file_name=f"content_overview_{project.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )


def export_quality_report(quality_report, project):
    """Export quality analysis report"""
    
    report_data = {
        'project': project.name,
        'generated_at': datetime.now().isoformat(),
        'quality_report': quality_report
    }
    
    import json
    report_json = json.dumps(report_data, indent=2, default=str)
    
    st.download_button(
        label="📥 Download Quality Report",
        data=report_json,
        file_name=f"quality_report_{project.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )


def export_duplicate_group(group, group_number):
    """Export duplicate group data"""
    
    group_data = {
        'group_number': group_number,
        'total_documents': len(group),
        'documents': group,
        'exported_at': datetime.now().isoformat()
    }
    
    import json
    group_json = json.dumps(group_data, indent=2, default=str)
    
    st.download_button(
        label="📥 Download Duplicate Group",
        data=group_json,
        file_name=f"duplicate_group_{group_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )


def export_analytics_report(metrics, project):
    """Export full analytics report"""
    
    analytics_data = {
        'project': project.name,
        'generated_at': datetime.now().isoformat(),
        'comprehensive_metrics': {
            'basic_metrics': {
                'total_documents': metrics.total_documents,
                'total_words': metrics.total_words,
                'total_characters': metrics.total_characters,
                'avg_document_length': metrics.avg_document_length
            },
            'quality_metrics': {
                'readability_score': metrics.readability_score,
                'complexity_score': metrics.complexity_score,
                'duplicate_percentage': metrics.duplicate_percentage
            },
            'content_distributions': {
                'language_distribution': metrics.language_distribution,
                'content_type_distribution': metrics.content_type_distribution,
                'source_distribution': metrics.source_distribution
            },
            'temporal_metrics': {
                'crawl_frequency': metrics.crawl_frequency,
                'content_freshness': metrics.content_freshness
            }
        }
    }
    
    import json
    analytics_json = json.dumps(analytics_data, indent=2, default=str)
    
    st.download_button(
        label="📥 Download Analytics Report",
        data=analytics_json,
        file_name=f"analytics_report_{project.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )


if __name__ == "__main__":
    show()