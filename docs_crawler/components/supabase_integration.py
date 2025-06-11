"""
Supabase Integration Module
Handles Supabase project creation, database connections, and data operations
"""

import os
import time
import json
import concurrent.futures
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from urllib.parse import urlparse
import streamlit as st

# Import required libraries
try:
    from supabase import create_client, Client
    import openai
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    st.error(f"Required dependencies not available: {e}")
    DEPENDENCIES_AVAILABLE = False


@dataclass
class ProjectConfig:
    """Configuration for a Crawl4AI project"""
    project_id: str
    name: str
    supabase_url: str
    supabase_key: str  # Encrypted
    created_at: datetime
    last_crawled: Optional[datetime]
    total_documents: int
    storage_used: int  # in MB
    rag_strategies: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        # Convert datetime objects to strings
        data['created_at'] = self.created_at.isoformat()
        data['last_crawled'] = self.last_crawled.isoformat() if self.last_crawled else None
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProjectConfig':
        """Create from dictionary"""
        # Convert string timestamps back to datetime
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data['last_crawled']:
            data['last_crawled'] = datetime.fromisoformat(data['last_crawled'])
        return cls(**data)


class SupabaseIntegration:
    """Main class for Supabase operations"""
    
    def __init__(self):
        self.projects = {}
        self.active_connection = None
        self._load_projects()
    
    def _load_projects(self):
        """Load existing projects from persistent storage"""
        # TODO: Implement persistent storage (e.g., local file, session state)
        # For now, use Streamlit session state
        if 'projects' in st.session_state:
            for project_id, project_data in st.session_state.projects.items():
                self.projects[project_id] = ProjectConfig.from_dict(project_data)
    
    def _save_projects(self):
        """Save projects to persistent storage"""
        # Save to Streamlit session state
        if 'projects' not in st.session_state:
            st.session_state.projects = {}
        
        for project_id, config in self.projects.items():
            st.session_state.projects[project_id] = config.to_dict()
    
    def create_project(self, name: str, supabase_url: str, supabase_key: str, initial_url: str = None) -> ProjectConfig:
        """Create a new project with actual Supabase connection"""
        
        if not DEPENDENCIES_AVAILABLE:
            raise Exception("Required dependencies not available")
        
        # Validate credentials first
        if not self.validate_supabase_credentials(supabase_url, supabase_key):
            raise ValueError("Invalid Supabase credentials")
        
        # Test connection
        try:
            test_client = create_client(supabase_url, supabase_key)
            # Simple test query
            test_client.table('_realtime_subscription').select('*').limit(1).execute()
        except Exception as e:
            raise ValueError(f"Failed to connect to Supabase: {e}")
        
        # Generate project ID
        project_id = f"proj_{len(self.projects) + 1:03d}"
        
        config = ProjectConfig(
            project_id=project_id,
            name=name,
            supabase_url=supabase_url,
            supabase_key=self.encrypt_credentials(supabase_key),
            created_at=datetime.now(),
            last_crawled=None,
            total_documents=0,
            storage_used=0,
            rag_strategies=["Vector Embeddings"]
        )
        
        self.projects[project_id] = config
        
        # Initialize database schema
        success = self._initialize_database_schema(config, supabase_key)
        if not success:
            # Remove project if schema initialization failed
            del self.projects[project_id]
            raise Exception("Failed to initialize database schema")
        
        self._save_projects()
        return config
    
    def _initialize_database_schema(self, config: ProjectConfig, raw_supabase_key: str) -> bool:
        """Initialize the database schema for a new project"""
        
        try:
            client = create_client(config.supabase_url, raw_supabase_key)
            
            # Read and execute the schema SQL
            schema_path = os.path.join(os.path.dirname(__file__), '..', 'mcp-crawl4ai-rag-main', 'crawled_pages.sql')
            
            if os.path.exists(schema_path):
                with open(schema_path, 'r') as f:
                    schema_sql = f.read()
                
                # Split SQL into individual statements and execute
                statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
                
                for statement in statements:
                    try:
                        # Use raw SQL execution for DDL statements
                        client.rpc('exec_sql', {'sql': statement}).execute()
                    except Exception as e:
                        # Some statements might fail if objects already exist, that's ok
                        print(f"SQL statement execution note: {e}")
                
                return True
            else:
                st.warning("Database schema file not found, using minimal schema")
                # Create minimal required tables
                self._create_minimal_schema(client)
                return True
        
        except Exception as e:
            st.error(f"Failed to initialize database schema: {e}")
            return False
    
    def _create_minimal_schema(self, client: Client) -> bool:
        """Create minimal required schema if SQL file is not available"""
        try:
            # Enable pgvector extension
            client.rpc('exec_sql', {'sql': 'CREATE EXTENSION IF NOT EXISTS vector;'}).execute()
            
            # Create sources table
            sources_sql = """
            CREATE TABLE IF NOT EXISTS sources (
                id SERIAL PRIMARY KEY,
                source_id VARCHAR(255) UNIQUE NOT NULL,
                summary TEXT,
                word_count INTEGER DEFAULT 0,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            """
            client.rpc('exec_sql', {'sql': sources_sql}).execute()
            
            # Create crawled_pages table
            pages_sql = """
            CREATE TABLE IF NOT EXISTS crawled_pages (
                id SERIAL PRIMARY KEY,
                url TEXT NOT NULL,
                chunk_number INTEGER NOT NULL,
                content TEXT NOT NULL,
                metadata JSONB,
                source_id VARCHAR(255),
                embedding vector(1536),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                FOREIGN KEY (source_id) REFERENCES sources(source_id)
            );
            """
            client.rpc('exec_sql', {'sql': pages_sql}).execute()
            
            # Create code_examples table
            code_sql = """
            CREATE TABLE IF NOT EXISTS code_examples (
                id SERIAL PRIMARY KEY,
                url TEXT NOT NULL,
                chunk_number INTEGER NOT NULL,
                code_example TEXT NOT NULL,
                summary TEXT,
                metadata JSONB,
                source_id VARCHAR(255),
                embedding vector(1536),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                FOREIGN KEY (source_id) REFERENCES sources(source_id)
            );
            """
            client.rpc('exec_sql', {'sql': code_sql}).execute()
            
            return True
        except Exception as e:
            print(f"Error creating minimal schema: {e}")
            return False
    
    def get_supabase_client(self, project_id: str) -> Client:
        """Get Supabase client for a specific project"""
        
        if not DEPENDENCIES_AVAILABLE:
            raise Exception("Required dependencies not available")
        
        config = self.get_project(project_id)
        if not config:
            raise ValueError(f"Project {project_id} not found")
        
        # Decrypt the key
        raw_key = self.decrypt_credentials(config.supabase_key)
        
        return create_client(config.supabase_url, raw_key)
    
    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for multiple texts in a single API call"""
        
        if not texts:
            return []
        
        # Get OpenAI API key from environment
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            st.error("OpenAI API key not found in environment variables")
            return [[0.0] * 1536] * len(texts)  # Return zero embeddings
        
        openai.api_key = openai_key
        
        max_retries = 3
        retry_delay = 1.0
        
        for retry in range(max_retries):
            try:
                response = openai.embeddings.create(
                    model="text-embedding-3-small",
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
                    return [[0.0] * 1536] * len(texts)  # Return zero embeddings
    
    def add_documents_to_supabase(
        self,
        project_id: str,
        urls: List[str], 
        chunk_numbers: List[int],
        contents: List[str], 
        metadatas: List[Dict[str, Any]],
        batch_size: int = 20
    ) -> bool:
        """Add documents to the Supabase crawled_pages table in batches"""
        
        try:
            client = self.get_supabase_client(project_id)
            
            # Get unique URLs to delete existing records
            unique_urls = list(set(urls))
            
            # Delete existing records for these URLs
            if unique_urls:
                try:
                    client.table("crawled_pages").delete().in_("url", unique_urls).execute()
                except Exception as e:
                    print(f"Error deleting existing records: {e}")
            
            # Process in batches
            for i in range(0, len(contents), batch_size):
                batch_end = min(i + batch_size, len(contents))
                
                batch_urls = urls[i:batch_end]
                batch_chunk_numbers = chunk_numbers[i:batch_end]
                batch_contents = contents[i:batch_end]
                batch_metadatas = metadatas[i:batch_end]
                
                # Create embeddings for the batch
                batch_embeddings = self.create_embeddings_batch(batch_contents)
                
                batch_data = []
                for j in range(len(batch_contents)):
                    # Extract source_id from URL
                    parsed_url = urlparse(batch_urls[j])
                    source_id = parsed_url.netloc or parsed_url.path
                    
                    data = {
                        "url": batch_urls[j],
                        "chunk_number": batch_chunk_numbers[j],
                        "content": batch_contents[j],
                        "metadata": {
                            "chunk_size": len(batch_contents[j]),
                            **batch_metadatas[j]
                        },
                        "source_id": source_id,
                        "embedding": batch_embeddings[j]
                    }
                    batch_data.append(data)
                
                # Insert batch with retry logic
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        client.table("crawled_pages").insert(batch_data).execute()
                        break
                    except Exception as e:
                        if retry < max_retries - 1:
                            print(f"Error inserting batch (attempt {retry + 1}): {e}")
                            time.sleep(1.0 * (retry + 1))
                        else:
                            print(f"Failed to insert batch after {max_retries} attempts: {e}")
                            return False
            
            # Update project document count
            self.update_project(project_id, {
                'total_documents': self.get_document_count(project_id),
                'last_crawled': datetime.now()
            })
            
            return True
            
        except Exception as e:
            st.error(f"Error adding documents to Supabase: {e}")
            return False
    
    def search_documents(
        self,
        project_id: str,
        query: str, 
        match_count: int = 10, 
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for documents using vector similarity"""
        
        try:
            client = self.get_supabase_client(project_id)
            
            # Create embedding for the query
            query_embeddings = self.create_embeddings_batch([query])
            if not query_embeddings:
                return []
            
            query_embedding = query_embeddings[0]
            
            # Execute the search using RPC function
            params = {
                'query_embedding': query_embedding,
                'match_count': match_count
            }
            
            if filter_metadata:
                params['filter'] = filter_metadata
            
            result = client.rpc('match_crawled_pages', params).execute()
            return result.data
            
        except Exception as e:
            st.error(f"Error searching documents: {e}")
            return []
    
    def get_document_count(self, project_id: str) -> int:
        """Get total document count for a project"""
        try:
            client = self.get_supabase_client(project_id)
            result = client.table("crawled_pages").select("id", count="exact").execute()
            return result.count or 0
        except Exception as e:
            print(f"Error getting document count: {e}")
            return 0
    
    def get_project(self, project_id: str) -> Optional[ProjectConfig]:
        """Get project configuration by ID"""
        return self.projects.get(project_id)
    
    def list_projects(self) -> List[ProjectConfig]:
        """List all available projects"""
        return list(self.projects.values())
    
    def update_project(self, project_id: str, updates: Dict[str, Any]) -> bool:
        """Update project configuration"""
        
        if project_id not in self.projects:
            return False
        
        config = self.projects[project_id]
        
        # Update allowed fields
        for key, value in updates.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        self._save_projects()
        return True
    
    def delete_project(self, project_id: str) -> bool:
        """Delete a project and all its data"""
        
        if project_id not in self.projects:
            return False
        
        try:
            client = self.get_supabase_client(project_id)
            
            # Delete all data from tables
            client.table("crawled_pages").delete().neq("id", 0).execute()
            client.table("code_examples").delete().neq("id", 0).execute()  
            client.table("sources").delete().neq("id", 0).execute()
            
        except Exception as e:
            print(f"Error deleting project data: {e}")
        
        # Remove from local storage
        del self.projects[project_id]
        self._save_projects()
        return True
    
    def test_connection(self, project_id: str) -> bool:
        """Test connection to a project's database"""
        
        try:
            client = self.get_supabase_client(project_id)
            # Simple test query
            client.table('crawled_pages').select('id').limit(1).execute()
            return True
        except Exception as e:
            st.error(f"Connection test failed: {e}")
            return False
    
    @staticmethod
    def validate_supabase_credentials(url: str, key: str) -> bool:
        """Validate Supabase credentials"""
        
        if not url or not key:
            return False
        
        if not url.startswith("https://"):
            return False
        
        if ".supabase.co" not in url and "localhost" not in url:
            return False
        
        if len(key) < 50:  # Supabase keys are typically quite long
            return False
        
        return True
    
    @staticmethod
    def encrypt_credentials(credentials: str) -> str:
        """Encrypt sensitive credentials for storage"""
        # TODO: Implement proper encryption
        # For now, just use base64 encoding as basic obfuscation
        import base64
        return base64.b64encode(credentials.encode()).decode()
    
    @staticmethod
    def decrypt_credentials(encrypted: str) -> str:
        """Decrypt stored credentials"""
        # TODO: Implement proper decryption
        import base64
        try:
            return base64.b64decode(encrypted.encode()).decode()
        except Exception:
            return encrypted


# Global instance
supabase_integration = SupabaseIntegration()


def get_supabase_integration() -> SupabaseIntegration:
    """Get the global Supabase integration instance"""
    return supabase_integration


def create_new_project(name: str, supabase_url: str, supabase_key: str, initial_url: str = None) -> ProjectConfig:
    """Convenience function to create a new project"""
    return supabase_integration.create_project(name, supabase_url, supabase_key, initial_url)


def get_project_list() -> List[ProjectConfig]:
    """Convenience function to get all projects"""
    return supabase_integration.list_projects() 