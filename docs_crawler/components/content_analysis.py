"""
Content Analysis Module
Provides content preview, analytics, quality metrics, and analysis tools
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import hashlib
import re
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from dataclasses import dataclass
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import dependencies
try:
    from textstat import flesch_reading_ease, flesch_kincaid_grade, automated_readability_index
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    ANALYSIS_DEPENDENCIES_AVAILABLE = True
except ImportError:
    ANALYSIS_DEPENDENCIES_AVAILABLE = False


@dataclass
class ContentMetrics:
    """Container for content analysis metrics"""
    
    # Basic metrics
    total_documents: int
    total_words: int
    total_characters: int
    avg_document_length: float
    
    # Quality metrics
    readability_score: float
    complexity_score: float
    duplicate_percentage: float
    
    # Content distribution
    language_distribution: Dict[str, int]
    content_type_distribution: Dict[str, int]
    source_distribution: Dict[str, int]
    
    # Temporal metrics
    crawl_frequency: Dict[str, int]
    content_freshness: Dict[str, int]


@dataclass
class DocumentAnalysis:
    """Analysis results for a single document"""
    
    url: str
    title: str
    word_count: int
    char_count: int
    readability_score: float
    complexity_score: float
    quality_score: float
    key_topics: List[str]
    similar_documents: List[str]
    content_hash: str
    analysis_timestamp: datetime


class ContentAnalyzer:
    """Main class for content analysis operations"""
    
    def __init__(self):
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
    
    def analyze_content_corpus(self, documents: List[Dict[str, Any]]) -> ContentMetrics:
        """Analyze a corpus of documents and return comprehensive metrics"""
        
        if not documents:
            return self._empty_metrics()
        
        # Basic metrics
        total_docs = len(documents)
        total_words = sum(doc.get('metadata', {}).get('word_count', 0) for doc in documents)
        total_chars = sum(doc.get('metadata', {}).get('char_count', 0) for doc in documents)
        avg_length = total_words / total_docs if total_docs > 0 else 0
        
        # Content analysis
        all_content = ' '.join([doc.get('content', '') for doc in documents])
        
        # Quality metrics
        readability = self._calculate_readability(all_content) if ANALYSIS_DEPENDENCIES_AVAILABLE else 0.5
        complexity = self._calculate_complexity(documents)
        duplicate_pct = self._calculate_duplicate_percentage(documents)
        
        # Distribution analysis
        language_dist = self._analyze_language_distribution(documents)
        content_type_dist = self._analyze_content_type_distribution(documents)
        source_dist = self._analyze_source_distribution(documents)
        
        # Temporal analysis
        crawl_freq = self._analyze_crawl_frequency(documents)
        content_fresh = self._analyze_content_freshness(documents)
        
        return ContentMetrics(
            total_documents=total_docs,
            total_words=total_words,
            total_characters=total_chars,
            avg_document_length=avg_length,
            readability_score=readability,
            complexity_score=complexity,
            duplicate_percentage=duplicate_pct,
            language_distribution=language_dist,
            content_type_distribution=content_type_dist,
            source_distribution=source_dist,
            crawl_frequency=crawl_freq,
            content_freshness=content_fresh
        )
    
    def analyze_single_document(self, document: Dict[str, Any], similar_docs: List[Dict[str, Any]] = None) -> DocumentAnalysis:
        """Perform detailed analysis of a single document"""
        
        content = document.get('content', '')
        metadata = document.get('metadata', {})
        
        # Basic metrics
        word_count = metadata.get('word_count', len(content.split()))
        char_count = len(content)
        
        # Quality metrics
        readability = self._calculate_readability(content) if ANALYSIS_DEPENDENCIES_AVAILABLE else 0.5
        complexity = self._calculate_document_complexity(content)
        quality = self._calculate_quality_score(content, metadata)
        
        # Content analysis
        topics = self._extract_key_topics(content)
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        # Similar documents
        similar_urls = []
        if similar_docs:
            similar_urls = [doc.get('url', '') for doc in similar_docs[:5]]
        
        return DocumentAnalysis(
            url=document.get('url', ''),
            title=metadata.get('title', 'Untitled'),
            word_count=word_count,
            char_count=char_count,
            readability_score=readability,
            complexity_score=complexity,
            quality_score=quality,
            key_topics=topics,
            similar_documents=similar_urls,
            content_hash=content_hash,
            analysis_timestamp=datetime.now()
        )
    
    def detect_duplicates(self, documents: List[Dict[str, Any]], similarity_threshold: float = 0.85) -> List[List[Dict[str, Any]]]:
        """Detect duplicate or near-duplicate documents"""
        
        duplicate_groups = []
        processed_hashes = set()
        
        # Create content hashes
        doc_hashes = {}
        for doc in documents:
            content = doc.get('content', '')
            # Create hash based on content (ignoring whitespace)
            normalized_content = re.sub(r'\s+', ' ', content.strip().lower())
            content_hash = hashlib.md5(normalized_content.encode()).hexdigest()
            
            if content_hash not in doc_hashes:
                doc_hashes[content_hash] = []
            doc_hashes[content_hash].append(doc)
        
        # Group duplicates
        for content_hash, docs in doc_hashes.items():
            if len(docs) > 1:
                duplicate_groups.append(docs)
        
        # For near-duplicates, use text similarity (simplified approach)
        if ANALYSIS_DEPENDENCIES_AVAILABLE:
            near_duplicates = self._find_near_duplicates(documents, similarity_threshold)
            duplicate_groups.extend(near_duplicates)
        
        return duplicate_groups
    
    def generate_content_preview(self, document: Dict[str, Any], preview_length: int = 500) -> Dict[str, Any]:
        """Generate a structured preview of document content"""
        
        content = document.get('content', '')
        metadata = document.get('metadata', {})
        
        # Extract different types of content
        preview = {
            'title': metadata.get('title', 'Untitled Document'),
            'url': document.get('url', ''),
            'summary': content[:preview_length] + '...' if len(content) > preview_length else content,
            'word_count': metadata.get('word_count', len(content.split())),
            'headers': self._extract_headers(content),
            'code_blocks': self._extract_code_blocks_preview(content),
            'links': self._extract_links(content),
            'key_phrases': self._extract_key_phrases(content),
            'content_type': self._detect_content_type(content),
            'estimated_reading_time': self._estimate_reading_time(content)
        }
        
        return preview
    
    def create_quality_report(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a comprehensive quality report for the document corpus"""
        
        if not documents:
            return {'error': 'No documents provided'}
        
        quality_scores = []
        issues = []
        recommendations = []
        
        for doc in documents:
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
            
            # Calculate quality score
            quality = self._calculate_quality_score(content, metadata)
            quality_scores.append(quality)
            
            # Identify issues
            doc_issues = self._identify_content_issues(content, metadata)
            if doc_issues:
                issues.extend([{
                    'url': doc.get('url', ''),
                    'title': metadata.get('title', 'Untitled'),
                    'issues': doc_issues
                }])
        
        # Generate recommendations
        recommendations = self._generate_quality_recommendations(quality_scores, issues)
        
        avg_quality = np.mean(quality_scores) if quality_scores else 0
        quality_distribution = {
            'excellent': len([q for q in quality_scores if q >= 0.8]),
            'good': len([q for q in quality_scores if 0.6 <= q < 0.8]),
            'fair': len([q for q in quality_scores if 0.4 <= q < 0.6]),
            'poor': len([q for q in quality_scores if q < 0.4])
        }
        
        return {
            'average_quality': avg_quality,
            'quality_distribution': quality_distribution,
            'total_issues': len(issues),
            'issues_by_document': issues,
            'recommendations': recommendations,
            'quality_scores': quality_scores
        }
    
    def export_analysis_report(self, analysis_data: Dict[str, Any], format: str = 'json') -> str:
        """Export analysis results in various formats"""
        
        if format.lower() == 'json':
            return json.dumps(analysis_data, indent=2, default=str)
        elif format.lower() == 'csv':
            return self._export_to_csv(analysis_data)
        else:
            return str(analysis_data)
    
    # Private helper methods
    
    def _empty_metrics(self) -> ContentMetrics:
        """Return empty metrics for edge cases"""
        return ContentMetrics(
            total_documents=0,
            total_words=0,
            total_characters=0,
            avg_document_length=0,
            readability_score=0,
            complexity_score=0,
            duplicate_percentage=0,
            language_distribution={},
            content_type_distribution={},
            source_distribution={},
            crawl_frequency={},
            content_freshness={}
        )
    
    def _calculate_readability(self, content: str) -> float:
        """Calculate readability score using textstat"""
        try:
            if not content.strip():
                return 0.0
            
            # Use Flesch Reading Ease (0-100, higher is easier)
            score = flesch_reading_ease(content)
            # Normalize to 0-1 scale
            return max(0, min(1, score / 100))
        except:
            return 0.5  # Default middle score
    
    def _calculate_complexity(self, documents: List[Dict[str, Any]]) -> float:
        """Calculate overall content complexity"""
        
        complexity_scores = []
        
        for doc in documents:
            content = doc.get('content', '')
            doc_complexity = self._calculate_document_complexity(content)
            complexity_scores.append(doc_complexity)
        
        return np.mean(complexity_scores) if complexity_scores else 0.5
    
    def _calculate_document_complexity(self, content: str) -> float:
        """Calculate complexity score for a single document"""
        
        if not content.strip():
            return 0.0
        
        # Multiple complexity indicators
        sentences = content.count('.') + content.count('!') + content.count('?')
        words = len(content.split())
        
        if sentences == 0 or words == 0:
            return 0.0
        
        # Average sentence length (complexity indicator)
        avg_sentence_length = words / sentences
        
        # Code blocks increase complexity
        code_blocks = content.count('```')
        
        # Technical terms (simplified detection)
        technical_indicators = ['function', 'class', 'import', 'def', 'return', 'variable', 'array', 'object']
        technical_count = sum(content.lower().count(term) for term in technical_indicators)
        
        # Normalize complexity score (0-1)
        complexity = min(1.0, (avg_sentence_length / 20) + (code_blocks * 0.1) + (technical_count / words * 10))
        
        return complexity
    
    def _calculate_duplicate_percentage(self, documents: List[Dict[str, Any]]) -> float:
        """Calculate percentage of duplicate content"""
        
        if len(documents) <= 1:
            return 0.0
        
        duplicate_groups = self.detect_duplicates(documents)
        duplicate_count = sum(len(group) - 1 for group in duplicate_groups)  # Don't count originals
        
        return (duplicate_count / len(documents)) * 100
    
    def _analyze_language_distribution(self, documents: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze programming language distribution in code blocks"""
        
        languages = Counter()
        
        for doc in documents:
            content = doc.get('content', '')
            # Find code blocks with language specifiers
            code_pattern = r'```(\w+)\n'
            matches = re.findall(code_pattern, content)
            languages.update(matches)
        
        return dict(languages.most_common())
    
    def _analyze_content_type_distribution(self, documents: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze content type distribution"""
        
        content_types = Counter()
        
        for doc in documents:
            content = doc.get('content', '')
            content_type = self._detect_content_type(content)
            content_types[content_type] += 1
        
        return dict(content_types)
    
    def _analyze_source_distribution(self, documents: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze source domain distribution"""
        
        sources = Counter()
        
        for doc in documents:
            url = doc.get('url', '')
            if url:
                from urllib.parse import urlparse
                domain = urlparse(url).netloc
                sources[domain] += 1
        
        return dict(sources.most_common())
    
    def _analyze_crawl_frequency(self, documents: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze crawling frequency over time"""
        
        frequency = Counter()
        
        for doc in documents:
            metadata = doc.get('metadata', {})
            crawled_at = metadata.get('crawled_at', '')
            if crawled_at:
                try:
                    date = datetime.fromisoformat(crawled_at.replace('Z', '+00:00'))
                    date_key = date.strftime('%Y-%m-%d')
                    frequency[date_key] += 1
                except:
                    continue
        
        return dict(frequency)
    
    def _analyze_content_freshness(self, documents: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze content freshness distribution"""
        
        now = datetime.now()
        freshness = {
            'last_24h': 0,
            'last_week': 0,
            'last_month': 0,
            'older': 0
        }
        
        for doc in documents:
            metadata = doc.get('metadata', {})
            crawled_at = metadata.get('crawled_at', '')
            if crawled_at:
                try:
                    date = datetime.fromisoformat(crawled_at.replace('Z', '+00:00'))
                    age = now - date.replace(tzinfo=None)
                    
                    if age.days < 1:
                        freshness['last_24h'] += 1
                    elif age.days < 7:
                        freshness['last_week'] += 1
                    elif age.days < 30:
                        freshness['last_month'] += 1
                    else:
                        freshness['older'] += 1
                except:
                    freshness['older'] += 1
            else:
                freshness['older'] += 1
        
        return freshness
    
    def _extract_key_topics(self, content: str, max_topics: int = 10) -> List[str]:
        """Extract key topics from content using simple frequency analysis"""
        
        # Simple keyword extraction (in a real implementation, you might use TF-IDF or NLP)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', content.lower())
        word_freq = Counter(word for word in words if word not in self.stop_words)
        
        return [word for word, _ in word_freq.most_common(max_topics)]
    
    def _find_near_duplicates(self, documents: List[Dict[str, Any]], threshold: float) -> List[List[Dict[str, Any]]]:
        """Find near-duplicate documents using text similarity"""
        # Simplified implementation - in production, you'd use more sophisticated similarity measures
        return []  # Placeholder for now
    
    def _extract_headers(self, content: str) -> List[str]:
        """Extract markdown headers from content"""
        headers = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
        return headers[:10]  # Limit to first 10 headers
    
    def _extract_code_blocks_preview(self, content: str) -> List[Dict[str, str]]:
        """Extract code blocks with language and snippet"""
        code_blocks = []
        
        # Find code blocks
        pattern = r'```(\w*)\n(.*?)\n```'
        matches = re.findall(pattern, content, re.DOTALL)
        
        for language, code in matches[:5]:  # Limit to first 5
            code_blocks.append({
                'language': language or 'text',
                'snippet': code[:200] + '...' if len(code) > 200 else code,
                'line_count': len(code.split('\n'))
            })
        
        return code_blocks
    
    def _extract_links(self, content: str) -> List[str]:
        """Extract links from markdown content"""
        # Find markdown links
        links = re.findall(r'\[([^\]]*)\]\(([^)]+)\)', content)
        return [url for _, url in links[:10]]  # Limit to first 10
    
    def _extract_key_phrases(self, content: str) -> List[str]:
        """Extract key phrases from content"""
        # Simple approach - find quoted text and emphasized text
        phrases = []
        
        # Quoted text
        quotes = re.findall(r'"([^"]{10,100})"', content)
        phrases.extend(quotes)
        
        # Bold/emphasized text
        bold = re.findall(r'\*\*([^*]{5,50})\*\*', content)
        phrases.extend(bold)
        
        return phrases[:10]
    
    def _detect_content_type(self, content: str) -> str:
        """Detect the type of content"""
        
        # Count different content indicators
        code_blocks = content.count('```')
        api_indicators = len(re.findall(r'\b(GET|POST|PUT|DELETE|API|endpoint)\b', content, re.IGNORECASE))
        tutorial_indicators = len(re.findall(r'\b(step|tutorial|guide|how to|example)\b', content, re.IGNORECASE))
        
        if code_blocks >= 3:
            return 'code_heavy'
        elif api_indicators >= 5:
            return 'api_documentation'
        elif tutorial_indicators >= 3:
            return 'tutorial'
        else:
            return 'general_documentation'
    
    def _estimate_reading_time(self, content: str) -> int:
        """Estimate reading time in minutes"""
        words = len(content.split())
        # Average reading speed is 200-250 words per minute
        return max(1, words // 200)
    
    def _calculate_quality_score(self, content: str, metadata: Dict[str, Any]) -> float:
        """Calculate overall quality score for content"""
        
        if not content.strip():
            return 0.0
        
        score_components = []
        
        # Length component (not too short, not too long)
        word_count = len(content.split())
        if 100 <= word_count <= 5000:
            length_score = 1.0
        elif word_count < 100:
            length_score = word_count / 100
        else:
            length_score = max(0.5, 1.0 - (word_count - 5000) / 10000)
        score_components.append(length_score)
        
        # Structure component (headers, formatting)
        headers = len(re.findall(r'^#+\s+', content, re.MULTILINE))
        structure_score = min(1.0, headers / 5)  # Ideal: 5+ headers
        score_components.append(structure_score)
        
        # Code examples component
        code_blocks = content.count('```')
        code_score = min(1.0, code_blocks / 3)  # Ideal: 3+ code blocks
        score_components.append(code_score)
        
        # Readability component
        if ANALYSIS_DEPENDENCIES_AVAILABLE:
            readability = self._calculate_readability(content)
            score_components.append(readability)
        
        # Metadata completeness
        metadata_score = 0
        if metadata.get('title'):
            metadata_score += 0.5
        if metadata.get('headers'):
            metadata_score += 0.5
        score_components.append(metadata_score)
        
        return np.mean(score_components)
    
    def _identify_content_issues(self, content: str, metadata: Dict[str, Any]) -> List[str]:
        """Identify potential content quality issues"""
        
        issues = []
        
        # Too short
        if len(content.split()) < 50:
            issues.append('Content too short (< 50 words)')
        
        # No structure
        if not re.search(r'^#+\s+', content, re.MULTILINE):
            issues.append('No headers found')
        
        # No examples
        if '```' not in content and 'example' not in content.lower():
            issues.append('No code examples or examples mentioned')
        
        # Missing title
        if not metadata.get('title'):
            issues.append('No title in metadata')
        
        # Broken formatting
        if content.count('```') % 2 != 0:
            issues.append('Unmatched code block delimiters')
        
        return issues
    
    def _generate_quality_recommendations(self, quality_scores: List[float], issues: List[Dict]) -> List[str]:
        """Generate recommendations based on quality analysis"""
        
        recommendations = []
        
        avg_quality = np.mean(quality_scores) if quality_scores else 0
        
        if avg_quality < 0.5:
            recommendations.append('Overall content quality is low. Consider reviewing and improving documentation.')
        
        # Count common issues
        issue_types = Counter()
        for doc_issues in issues:
            for issue in doc_issues['issues']:
                issue_types[issue] += 1
        
        for issue, count in issue_types.most_common(3):
            recommendations.append(f'Common issue: {issue} (affects {count} documents)')
        
        return recommendations
    
    def _export_to_csv(self, analysis_data: Dict[str, Any]) -> str:
        """Export analysis data to CSV format"""
        # Simplified CSV export
        lines = ['Metric,Value']
        
        if 'average_quality' in analysis_data:
            lines.append(f'Average Quality,{analysis_data["average_quality"]:.3f}')
        
        if 'total_issues' in analysis_data:
            lines.append(f'Total Issues,{analysis_data["total_issues"]}')
        
        return '\n'.join(lines)


class ContentVisualization:
    """Class for creating content analysis visualizations"""
    
    @staticmethod
    def create_quality_distribution_chart(quality_scores: List[float]) -> go.Figure:
        """Create a histogram of quality score distribution"""
        
        fig = go.Figure(data=[
            go.Histogram(x=quality_scores, nbinsx=20, name='Quality Scores')
        ])
        
        fig.update_layout(
            title='Content Quality Distribution',
            xaxis_title='Quality Score',
            yaxis_title='Number of Documents',
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_content_type_pie_chart(content_types: Dict[str, int]) -> go.Figure:
        """Create a pie chart of content type distribution"""
        
        fig = go.Figure(data=[
            go.Pie(labels=list(content_types.keys()), values=list(content_types.values()))
        ])
        
        fig.update_layout(title='Content Type Distribution')
        
        return fig
    
    @staticmethod
    def create_source_distribution_bar_chart(source_dist: Dict[str, int]) -> go.Figure:
        """Create a bar chart of source distribution"""
        
        # Limit to top 10 sources
        top_sources = dict(sorted(source_dist.items(), key=lambda x: x[1], reverse=True)[:10])
        
        fig = go.Figure(data=[
            go.Bar(x=list(top_sources.keys()), y=list(top_sources.values()))
        ])
        
        fig.update_layout(
            title='Top 10 Content Sources',
            xaxis_title='Source Domain',
            yaxis_title='Number of Documents',
            xaxis_tickangle=-45
        )
        
        return fig
    
    @staticmethod
    def create_crawl_timeline(crawl_frequency: Dict[str, int]) -> go.Figure:
        """Create a timeline of crawling activity"""
        
        dates = sorted(crawl_frequency.keys())
        counts = [crawl_frequency[date] for date in dates]
        
        fig = go.Figure(data=[
            go.Scatter(x=dates, y=counts, mode='lines+markers', name='Crawl Activity')
        ])
        
        fig.update_layout(
            title='Crawling Activity Over Time',
            xaxis_title='Date',
            yaxis_title='Documents Crawled'
        )
        
        return fig
    
    @staticmethod
    def create_word_cloud(documents: List[Dict[str, Any]]) -> Optional[plt.Figure]:
        """Create a word cloud from document content"""
        
        if not ANALYSIS_DEPENDENCIES_AVAILABLE:
            return None
        
        # Combine all content
        all_text = ' '.join([doc.get('content', '') for doc in documents])
        
        # Create word cloud
        try:
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('Content Word Cloud')
            
            return fig
        except:
            return None


# Global analyzer instance
content_analyzer = ContentAnalyzer()


def get_content_analyzer() -> ContentAnalyzer:
    """Get the global content analyzer instance"""
    return content_analyzer