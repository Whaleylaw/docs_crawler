"""
Search Engine Module
Handles search operations, query processing, and result ranking
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import streamlit as st


class SearchType(Enum):
    """Search type enumeration"""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    CODE = "code"


@dataclass
class SearchResult:
    """Single search result"""
    id: str
    title: str
    content: str
    url: str
    similarity_score: float
    content_type: str
    source_domain: str
    date_crawled: datetime
    metadata: Dict[str, Any]
    highlights: List[str] = None


@dataclass
class SearchQuery:
    """Search query configuration"""
    query_text: str
    search_type: SearchType
    project_id: Optional[str]
    filters: Dict[str, Any]
    limit: int = 10
    similarity_threshold: float = 0.0
    include_code: bool = False


class SearchEngine:
    """Main search engine for content retrieval"""
    
    def __init__(self):
        self.query_history = []
        self.query_counter = 0
    
    def search(self, query: SearchQuery) -> List[SearchResult]:
        """Execute a search query and return results"""
        
        self.query_counter += 1
        query_id = f"query_{self.query_counter:06d}"
        
        # Log query
        self.query_history.append({
            "id": query_id,
            "query": query.query_text,
            "search_type": query.search_type.value,
            "project_id": query.project_id,
            "timestamp": datetime.now(),
            "filters": query.filters
        })
        
        # Execute search based on type
        if query.search_type == SearchType.SEMANTIC:
            results = self._semantic_search(query)
        elif query.search_type == SearchType.KEYWORD:
            results = self._keyword_search(query)
        elif query.search_type == SearchType.HYBRID:
            results = self._hybrid_search(query)
        elif query.search_type == SearchType.CODE:
            results = self._code_search(query)
        else:
            results = []
        
        # Apply filters and ranking
        filtered_results = self._apply_filters(results, query.filters)
        ranked_results = self._rank_results(filtered_results, query)
        
        # Limit results
        return ranked_results[:query.limit]
    
    def _semantic_search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform semantic vector search"""
        
        # TODO: Implement actual semantic search
        # - Generate query embedding
        # - Search vector database
        # - Return similar content chunks
        
        # Placeholder mock results
        mock_results = [
            SearchResult(
                id="result_001",
                title="JWT Authentication Guide",
                content="Complete guide to implementing JWT authentication in modern applications...",
                url="https://docs.example.com/auth/jwt",
                similarity_score=0.94,
                content_type="Documentation",
                source_domain="docs.example.com",
                date_crawled=datetime.now(),
                metadata={"word_count": 1250, "language": "en"}
            ),
            SearchResult(
                id="result_002",
                title="Authentication Best Practices",
                content="Security best practices for user authentication and authorization...",
                url="https://docs.example.com/security/auth",
                similarity_score=0.87,
                content_type="Documentation",
                source_domain="docs.example.com",
                date_crawled=datetime.now(),
                metadata={"word_count": 890, "language": "en"}
            )
        ]
        
        return mock_results
    
    def _keyword_search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform keyword-based search"""
        
        # TODO: Implement actual keyword search
        # - Parse query terms
        # - Search indexed content
        # - Calculate TF-IDF scores
        
        # Placeholder implementation
        return self._semantic_search(query)  # Use semantic as fallback
    
    def _hybrid_search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform hybrid semantic + keyword search"""
        
        # TODO: Implement hybrid search
        # - Combine semantic and keyword results
        # - Apply fusion ranking algorithm
        # - Normalize scores
        
        semantic_results = self._semantic_search(query)
        keyword_results = self._keyword_search(query)
        
        # Simple fusion: combine and deduplicate
        combined_results = {}
        
        for result in semantic_results + keyword_results:
            if result.id in combined_results:
                # Average the scores
                existing = combined_results[result.id]
                existing.similarity_score = (existing.similarity_score + result.similarity_score) / 2
            else:
                combined_results[result.id] = result
        
        return list(combined_results.values())
    
    def _code_search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform specialized code example search"""
        
        # TODO: Implement code-specific search
        # - Search code_examples table
        # - Apply programming language filters
        # - Include surrounding context
        
        # Mock code results
        mock_code_results = [
            SearchResult(
                id="code_001",
                title="JWT Token Validation Function",
                content="""```python
def validate_jwt_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        return {'valid': True, 'payload': payload}
    except jwt.ExpiredSignatureError:
        return {'valid': False, 'error': 'Token expired'}
    except jwt.InvalidTokenError:
        return {'valid': False, 'error': 'Invalid token'}
```""",
                url="https://api.example.com/auth/validation",
                similarity_score=0.91,
                content_type="Code Example",
                source_domain="api.example.com",
                date_crawled=datetime.now(),
                metadata={"language": "python", "function_name": "validate_jwt_token"}
            )
        ]
        
        return mock_code_results
    
    def _apply_filters(self, results: List[SearchResult], filters: Dict[str, Any]) -> List[SearchResult]:
        """Apply search filters to results"""
        
        filtered_results = results
        
        # Content type filter
        if filters.get('content_types'):
            content_types = filters['content_types']
            filtered_results = [r for r in filtered_results if r.content_type in content_types]
        
        # Source domain filter
        if filters.get('source_domains'):
            domains = filters['source_domains']
            filtered_results = [r for r in filtered_results if r.source_domain in domains]
        
        # Date range filter
        if filters.get('date_range'):
            start_date, end_date = filters['date_range']
            filtered_results = [r for r in filtered_results 
                              if start_date <= r.date_crawled.date() <= end_date]
        
        # Similarity threshold filter
        if filters.get('similarity_threshold'):
            threshold = filters['similarity_threshold']
            filtered_results = [r for r in filtered_results if r.similarity_score >= threshold]
        
        # Programming language filter (for code)
        if filters.get('programming_language') and filters['programming_language'] != "All":
            language = filters['programming_language'].lower()
            filtered_results = [r for r in filtered_results 
                              if r.content_type == "Code Example" and 
                              r.metadata.get('language', '').lower() == language]
        
        return filtered_results
    
    def _rank_results(self, results: List[SearchResult], query: SearchQuery) -> List[SearchResult]:
        """Rank and sort search results"""
        
        # Primary sort by similarity score (descending)
        ranked_results = sorted(results, key=lambda x: x.similarity_score, reverse=True)
        
        # Secondary ranking factors could include:
        # - Recency (newer content scored higher)
        # - Source authority
        # - Content length
        # - User interaction metrics
        
        return ranked_results
    
    def get_query_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent query history"""
        return self.query_history[-limit:]
    
    def get_search_suggestions(self, partial_query: str, limit: int = 5) -> List[str]:
        """Get search suggestions based on query history"""
        
        suggestions = []
        partial_lower = partial_query.lower()
        
        # Find queries that start with the partial query
        for query_data in self.query_history:
            query_text = query_data['query']
            if query_text.lower().startswith(partial_lower) and query_text not in suggestions:
                suggestions.append(query_text)
                if len(suggestions) >= limit:
                    break
        
        return suggestions
    
    def analyze_query(self, query_text: str) -> Dict[str, Any]:
        """Analyze query to suggest search improvements"""
        
        analysis = {
            "query_length": len(query_text.split()),
            "has_special_chars": any(char in query_text for char in ['(', ')', '[', ']', '{', '}']),
            "suggested_search_type": SearchType.SEMANTIC,
            "estimated_results": "10-50",
            "suggestions": []
        }
        
        # Detect code-related queries
        code_keywords = ['function', 'class', 'method', 'code', 'example', 'implementation']
        if any(keyword in query_text.lower() for keyword in code_keywords):
            analysis["suggested_search_type"] = SearchType.CODE
            analysis["suggestions"].append("Try the Code Examples search mode for better results")
        
        # Detect very short queries
        if analysis["query_length"] < 2:
            analysis["suggestions"].append("Try adding more specific terms to your query")
        
        # Detect very long queries  
        if analysis["query_length"] > 10:
            analysis["suggestions"].append("Consider breaking down your query into smaller, focused searches")
        
        return analysis


class SearchCache:
    """Cache for search results to improve performance"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}
    
    def get(self, query_hash: str) -> Optional[List[SearchResult]]:
        """Get cached results for a query"""
        
        if query_hash in self.cache:
            self.access_times[query_hash] = datetime.now()
            return self.cache[query_hash]
        
        return None
    
    def set(self, query_hash: str, results: List[SearchResult]):
        """Cache results for a query"""
        
        # Simple LRU eviction if cache is full
        if len(self.cache) >= self.max_size:
            oldest_query = min(self.access_times.keys(), 
                              key=lambda k: self.access_times[k])
            del self.cache[oldest_query]
            del self.access_times[oldest_query]
        
        self.cache[query_hash] = results
        self.access_times[query_hash] = datetime.now()
    
    def clear(self):
        """Clear the cache"""
        self.cache.clear()
        self.access_times.clear()


class QueryProcessor:
    """Utility class for query processing and enhancement"""
    
    @staticmethod
    def normalize_query(query: str) -> str:
        """Normalize query text for better matching"""
        
        # Remove extra whitespace
        normalized = ' '.join(query.split())
        
        # Convert to lowercase for case-insensitive search
        normalized = normalized.lower()
        
        # Remove special characters that might interfere with search
        import re
        normalized = re.sub(r'[^\w\s\-\.]', ' ', normalized)
        
        return normalized.strip()
    
    @staticmethod
    def extract_keywords(query: str) -> List[str]:
        """Extract important keywords from query"""
        
        # Simple keyword extraction (can be enhanced with NLP)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        words = query.lower().split()
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    @staticmethod
    def detect_intent(query: str) -> str:
        """Detect search intent from query"""
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['how', 'tutorial', 'guide', 'step']):
            return "tutorial"
        elif any(word in query_lower for word in ['what', 'definition', 'meaning']):
            return "definition"
        elif any(word in query_lower for word in ['example', 'sample', 'demo']):
            return "example"
        elif any(word in query_lower for word in ['error', 'fix', 'problem', 'issue']):
            return "troubleshooting"
        else:
            return "general"


# Global instances
search_engine = SearchEngine()
search_cache = SearchCache()


def get_search_engine() -> SearchEngine:
    """Get the global search engine instance"""
    return search_engine


def get_search_cache() -> SearchCache:
    """Get the global search cache instance"""
    return search_cache 