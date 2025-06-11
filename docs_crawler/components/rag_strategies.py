"""
RAG Strategies Module
Implements different RAG (Retrieval-Augmented Generation) processing strategies
"""

import os
import re
import json
import time
import concurrent.futures
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import streamlit as st

# Import dependencies
try:
    import openai
    from sentence_transformers import CrossEncoder
    RAG_DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    st.error(f"RAG dependencies not available: {e}")
    RAG_DEPENDENCIES_AVAILABLE = False


class RAGStrategy(Enum):
    """Available RAG processing strategies"""
    VECTOR_EMBEDDINGS = "vector_embeddings"
    CONTEXTUAL_EMBEDDINGS = "contextual_embeddings"
    HYBRID_SEARCH = "hybrid_search"
    AGENTIC_RAG = "agentic_rag"
    CROSS_ENCODER_RERANKING = "cross_encoder_reranking"


@dataclass
class RAGConfiguration:
    """Configuration for RAG processing strategies"""
    
    # Strategy selection
    enabled_strategies: List[RAGStrategy]
    
    # Vector embeddings settings
    embedding_model: str = "text-embedding-3-small"
    chunk_overlap: int = 100
    
    # Contextual embeddings settings
    use_contextual_embeddings: bool = False
    context_model: str = "gpt-3.5-turbo"
    max_context_tokens: int = 200
    
    # Agentic RAG settings
    extract_code_examples: bool = False
    min_code_length: int = 1000
    code_context_chars: int = 1000
    parallel_workers: int = 10
    
    # Cross-encoder reranking settings
    use_reranking: bool = False
    reranking_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 50
    
    # Hybrid search settings
    use_hybrid_search: bool = False
    vector_weight: float = 0.7
    keyword_weight: float = 0.3
    
    # Preview and batch settings
    preview_mode: bool = False
    preview_sample_size: int = 3
    batch_size: int = 20
    max_retries: int = 3


class RAGProcessor:
    """Main class for processing content with different RAG strategies"""
    
    def __init__(self, config: RAGConfiguration):
        self.config = config
        self.reranking_model = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models based on configuration"""
        
        if not RAG_DEPENDENCIES_AVAILABLE:
            return
        
        # Initialize OpenAI API key
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            openai.api_key = openai_key
        
        # Initialize reranking model if enabled
        if self.config.use_reranking:
            try:
                self.reranking_model = CrossEncoder(self.config.reranking_model)
                st.success(f"✅ Loaded reranking model: {self.config.reranking_model}")
            except Exception as e:
                st.warning(f"Failed to load reranking model: {e}")
                self.reranking_model = None
    
    def process_content(self, content: str, url: str, full_document: str = None) -> Dict[str, Any]:
        """
        Process content with selected RAG strategies
        
        Args:
            content: The content chunk to process
            url: Source URL
            full_document: Full document content for contextual embeddings
            
        Returns:
            Processed content with embeddings and metadata
        """
        
        result = {
            "original_content": content,
            "processed_content": content,
            "url": url,
            "strategies_applied": [],
            "metadata": {},
            "code_examples": [],
            "embeddings": {}
        }
        
        # Apply contextual embeddings if enabled
        if (RAGStrategy.CONTEXTUAL_EMBEDDINGS in self.config.enabled_strategies and 
            self.config.use_contextual_embeddings and full_document):
            
            try:
                contextual_content, success = self.generate_contextual_embedding(full_document, content)
                if success:
                    result["processed_content"] = contextual_content
                    result["strategies_applied"].append("contextual_embeddings")
                    result["metadata"]["contextual_embedding"] = True
            except Exception as e:
                st.warning(f"Contextual embedding failed: {e}")
        
        # Extract code examples if agentic RAG is enabled
        if (RAGStrategy.AGENTIC_RAG in self.config.enabled_strategies and 
            self.config.extract_code_examples):
            
            try:
                code_blocks = self.extract_code_blocks(content)
                if code_blocks:
                    # Generate summaries for code blocks
                    code_summaries = self.generate_code_summaries(code_blocks)
                    
                    for block, summary in zip(code_blocks, code_summaries):
                        result["code_examples"].append({
                            "code": block["code"],
                            "language": block["language"],
                            "summary": summary,
                            "context_before": block["context_before"],
                            "context_after": block["context_after"]
                        })
                    
                    result["strategies_applied"].append("agentic_rag")
                    result["metadata"]["code_examples_count"] = len(code_blocks)
            except Exception as e:
                st.warning(f"Code extraction failed: {e}")
        
        # Generate embeddings
        try:
            content_to_embed = result["processed_content"]
            embedding = self.create_embedding(content_to_embed)
            result["embeddings"]["content"] = embedding
            
            # Generate separate embeddings for code examples
            for i, code_example in enumerate(result["code_examples"]):
                code_text = f"{code_example['code']}\n\nSummary: {code_example['summary']}"
                code_embedding = self.create_embedding(code_text)
                result["embeddings"][f"code_{i}"] = code_embedding
                
        except Exception as e:
            st.error(f"Embedding generation failed: {e}")
            result["embeddings"]["content"] = [0.0] * 1536  # Fallback zero embedding
        
        return result
    
    def generate_contextual_embedding(self, full_document: str, chunk: str) -> Tuple[str, bool]:
        """Generate contextual information for a chunk within a document"""
        
        if not openai.api_key:
            return chunk, False
        
        try:
            # Create the prompt for generating contextual information
            prompt = f"""<document> 
{full_document[:25000]} 
</document>
Here is the chunk we want to situate within the whole document 
<chunk> 
{chunk}
</chunk> 
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

            # Call the OpenAI API to generate contextual information
            response = openai.chat.completions.create(
                model=self.config.context_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides concise contextual information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=self.config.max_context_tokens
            )
            
            # Extract the generated context
            context = response.choices[0].message.content.strip()
            
            # Combine the context with the original chunk
            contextual_text = f"{context}\n---\n{chunk}"
            
            return contextual_text, True
        
        except Exception as e:
            print(f"Error generating contextual embedding: {e}. Using original chunk instead.")
            return chunk, False
    
    def extract_code_blocks(self, markdown_content: str) -> List[Dict[str, Any]]:
        """Extract code blocks from markdown content along with context"""
        
        code_blocks = []
        
        # Skip if content starts with triple backticks (edge case)
        content = markdown_content.strip()
        start_offset = 0
        if content.startswith('```'):
            start_offset = 3
        
        # Find all occurrences of triple backticks
        backtick_positions = []
        pos = start_offset
        while True:
            pos = markdown_content.find('```', pos)
            if pos == -1:
                break
            backtick_positions.append(pos)
            pos += 3
        
        # Process pairs of backticks
        i = 0
        while i < len(backtick_positions) - 1:
            start_pos = backtick_positions[i]
            end_pos = backtick_positions[i + 1]
            
            # Extract the content between backticks
            code_section = markdown_content[start_pos+3:end_pos]
            
            # Check if there's a language specifier on the first line
            lines = code_section.split('\n', 1)
            if len(lines) > 1:
                # Check if first line is a language specifier
                first_line = lines[0].strip()
                if first_line and not ' ' in first_line and len(first_line) < 20:
                    language = first_line
                    code_content = lines[1].strip() if len(lines) > 1 else ""
                else:
                    language = ""
                    code_content = code_section.strip()
            else:
                language = ""
                code_content = code_section.strip()
            
            # Skip if code block is too short
            if len(code_content) < self.config.min_code_length:
                i += 2
                continue
            
            # Extract context before and after
            context_chars = self.config.code_context_chars
            context_start = max(0, start_pos - context_chars)
            context_before = markdown_content[context_start:start_pos].strip()
            
            context_end = min(len(markdown_content), end_pos + 3 + context_chars)
            context_after = markdown_content[end_pos + 3:context_end].strip()
            
            code_blocks.append({
                'code': code_content,
                'language': language,
                'context_before': context_before,
                'context_after': context_after,
                'full_context': f"{context_before}\n\n{code_content}\n\n{context_after}"
            })
            
            i += 2
        
        return code_blocks
    
    def generate_code_summaries(self, code_blocks: List[Dict[str, Any]]) -> List[str]:
        """Generate summaries for code blocks using parallel processing"""
        
        if not openai.api_key:
            return ["Code example for demonstration purposes."] * len(code_blocks)
        
        # Process code examples in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            # Prepare arguments for parallel processing
            summary_args = [(block['code'], block['context_before'], block['context_after']) 
                          for block in code_blocks]
            
            # Generate summaries in parallel
            summaries = list(executor.map(self._generate_single_code_summary, summary_args))
        
        return summaries
    
    def _generate_single_code_summary(self, args: Tuple[str, str, str]) -> str:
        """Generate a summary for a single code example"""
        
        code, context_before, context_after = args
        
        # Create the prompt
        prompt = f"""<context_before>
{context_before[-500:] if len(context_before) > 500 else context_before}
</context_before>

<code_example>
{code[:1500] if len(code) > 1500 else code}
</code_example>

<context_after>
{context_after[:500] if len(context_after) > 500 else context_after}
</context_after>

Based on the code example and its surrounding context, provide a concise summary (2-3 sentences) that describes what this code example demonstrates and its purpose. Focus on the practical application and key concepts illustrated.
"""
        
        try:
            response = openai.chat.completions.create(
                model=self.config.context_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides concise code example summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=100
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"Error generating code example summary: {e}")
            return "Code example for demonstration purposes."
    
    def create_embedding(self, text: str) -> List[float]:
        """Create an embedding for text using OpenAI's API"""
        
        if not openai.api_key:
            return [0.0] * 1536  # Return zero embedding
        
        try:
            response = openai.embeddings.create(
                model=self.config.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error creating embedding: {e}")
            return [0.0] * 1536
    
    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for multiple texts in a single API call"""
        
        if not texts or not openai.api_key:
            return [[0.0] * 1536] * len(texts)
        
        max_retries = self.config.max_retries
        retry_delay = 1.0
        
        for retry in range(max_retries):
            try:
                response = openai.embeddings.create(
                    model=self.config.embedding_model,
                    input=texts
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"Error creating batch embeddings (attempt {retry + 1}/{max_retries}): {e}")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print(f"Failed to create batch embeddings after {max_retries} attempts: {e}")
                    return [[0.0] * 1536] * len(texts)
    
    def rerank_search_results(self, query: str, results: List[Dict[str, Any]], 
                            content_key: str = "content") -> List[Dict[str, Any]]:
        """Rerank search results using cross-encoder model"""
        
        if not self.reranking_model or not results:
            return results
        
        try:
            # Extract content from results
            texts = [result.get(content_key, "") for result in results]
            
            # Create pairs of [query, document] for the cross-encoder
            pairs = [[query, text] for text in texts]
            
            # Get relevance scores from the cross-encoder
            scores = self.reranking_model.predict(pairs)
            
            # Add scores to results and sort by score (descending)
            for i, result in enumerate(results):
                result["rerank_score"] = float(scores[i])
            
            # Sort by rerank score
            reranked = sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)
            
            return reranked
        except Exception as e:
            print(f"Error during reranking: {e}")
            return results
    
    def create_preview_samples(self, content_list: List[str], urls: List[str]) -> List[Dict[str, Any]]:
        """Create preview samples to show before full processing"""
        
        if not content_list:
            return []
        
        # Select sample content based on configuration
        sample_size = min(self.config.preview_sample_size, len(content_list))
        step_size = max(1, len(content_list) // sample_size)
        
        samples = []
        for i in range(0, len(content_list), step_size):
            if len(samples) >= sample_size:
                break
            
            content = content_list[i]
            url = urls[i] if i < len(urls) else "Unknown"
            
            # Process sample with all enabled strategies
            processed = self.process_content(content, url)
            
            samples.append({
                "index": i,
                "url": url,
                "original_length": len(content),
                "processed_length": len(processed["processed_content"]),
                "strategies_applied": processed["strategies_applied"],
                "code_examples_found": len(processed["code_examples"]),
                "preview_content": content[:500] + "..." if len(content) > 500 else content,
                "processed_preview": processed["processed_content"][:500] + "..." if len(processed["processed_content"]) > 500 else processed["processed_content"]
            })
        
        return samples


class RAGStrategySelector:
    """UI component for selecting and configuring RAG strategies"""
    
    @staticmethod
    def render_strategy_selection() -> RAGConfiguration:
        """Render the RAG strategy selection interface"""
        
        st.subheader("🧠 RAG Processing Strategies")
        st.markdown("Choose how to process and enhance your crawled content")
        
        # Strategy selection
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Core Strategies")
            
            vector_embeddings = st.checkbox(
                "Vector Embeddings",
                value=True,
                help="Standard embedding-based search (always recommended)",
                disabled=True  # Always enabled
            )
            
            contextual_embeddings = st.checkbox(
                "Contextual Embeddings",
                value=False,
                help="AI-enhanced chunks with document context (slower, higher accuracy)"
            )
            
            agentic_rag = st.checkbox(
                "Agentic RAG (Code Extraction)",
                value=False,
                help="Extract and analyze code examples with AI summaries"
            )
        
        with col2:
            st.markdown("#### 🔍 Search Enhancements")
            
            cross_encoder_reranking = st.checkbox(
                "Cross-encoder Reranking",
                value=False,
                help="Re-rank search results for improved relevance"
            )
            
            hybrid_search = st.checkbox(
                "Hybrid Search",
                value=False,
                help="Combine vector and keyword search (coming soon)",
                disabled=True  # Not implemented yet
            )
        
        # Advanced configuration
        with st.expander("⚙️ Advanced Configuration", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("##### Embeddings")
                embedding_model = st.selectbox(
                    "Embedding Model",
                    ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
                    help="OpenAI embedding model to use"
                )
                
                chunk_overlap = st.slider(
                    "Chunk Overlap",
                    0, 500, 100,
                    help="Character overlap between chunks"
                )
            
            with col2:
                st.markdown("##### Contextual Processing")
                context_model = st.selectbox(
                    "Context Model",
                    ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
                    help="Model for generating contextual information"
                )
                
                max_context_tokens = st.slider(
                    "Max Context Tokens",
                    50, 500, 200,
                    help="Maximum tokens for context generation"
                )
            
            with col3:
                st.markdown("##### Code Processing")
                min_code_length = st.slider(
                    "Min Code Length",
                    100, 5000, 1000,
                    help="Minimum characters for code extraction"
                )
                
                parallel_workers = st.slider(
                    "Parallel Workers",
                    1, 20, 10,
                    help="Number of parallel workers for processing"
                )
        
        # Preview and batch settings
        with st.expander("👁️ Preview & Batch Settings", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                preview_mode = st.checkbox(
                    "Enable Preview Mode",
                    value=False,
                    help="Show sample processed content before full crawl"
                )
                
                preview_sample_size = st.slider(
                    "Preview Sample Size",
                    1, 10, 3,
                    help="Number of samples to show in preview"
                )
            
            with col2:
                batch_size = st.slider(
                    "Batch Size",
                    5, 50, 20,
                    help="Number of items to process in each batch"
                )
                
                max_retries = st.slider(
                    "Max Retries",
                    1, 10, 3,
                    help="Maximum retry attempts for failed operations"
                )
        
        # Build configuration
        enabled_strategies = [RAGStrategy.VECTOR_EMBEDDINGS]  # Always enabled
        
        if contextual_embeddings:
            enabled_strategies.append(RAGStrategy.CONTEXTUAL_EMBEDDINGS)
        if agentic_rag:
            enabled_strategies.append(RAGStrategy.AGENTIC_RAG)
        if cross_encoder_reranking:
            enabled_strategies.append(RAGStrategy.CROSS_ENCODER_RERANKING)
        if hybrid_search:
            enabled_strategies.append(RAGStrategy.HYBRID_SEARCH)
        
        config = RAGConfiguration(
            enabled_strategies=enabled_strategies,
            embedding_model=embedding_model,
            chunk_overlap=chunk_overlap,
            use_contextual_embeddings=contextual_embeddings,
            context_model=context_model,
            max_context_tokens=max_context_tokens,
            extract_code_examples=agentic_rag,
            min_code_length=min_code_length,
            parallel_workers=parallel_workers,
            use_reranking=cross_encoder_reranking,
            use_hybrid_search=hybrid_search,
            preview_mode=preview_mode,
            preview_sample_size=preview_sample_size,
            batch_size=batch_size,
            max_retries=max_retries
        )
        
        return config
    
    @staticmethod
    def display_strategy_summary(config: RAGConfiguration):
        """Display a summary of selected strategies"""
        
        st.markdown("#### 📋 Selected Strategies")
        
        strategy_names = {
            RAGStrategy.VECTOR_EMBEDDINGS: "Vector Embeddings",
            RAGStrategy.CONTEXTUAL_EMBEDDINGS: "Contextual Embeddings",
            RAGStrategy.AGENTIC_RAG: "Agentic RAG",
            RAGStrategy.CROSS_ENCODER_RERANKING: "Cross-encoder Reranking",
            RAGStrategy.HYBRID_SEARCH: "Hybrid Search"
        }
        
        for strategy in config.enabled_strategies:
            st.write(f"✅ {strategy_names[strategy]}")
        
        # Show configuration summary
        with st.expander("Configuration Details"):
            st.json({
                "embedding_model": config.embedding_model,
                "context_model": config.context_model if config.use_contextual_embeddings else "Not used",
                "extract_code_examples": config.extract_code_examples,
                "use_reranking": config.use_reranking,
                "preview_mode": config.preview_mode,
                "batch_size": config.batch_size
            })


# Global instance
rag_processor = None


def get_rag_processor(config: RAGConfiguration = None) -> RAGProcessor:
    """Get a configured RAG processor instance"""
    
    global rag_processor
    
    if config is None:
        # Use default configuration
        config = RAGConfiguration(enabled_strategies=[RAGStrategy.VECTOR_EMBEDDINGS])
    
    if rag_processor is None or rag_processor.config != config:
        rag_processor = RAGProcessor(config)
    
    return rag_processor