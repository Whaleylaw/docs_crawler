"""
Supabase Integration Module
Handles Supabase project creation, database connections, and data operations
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import streamlit as st
from supabase import create_client, Client
from cryptography.fernet import Fernet
import psycopg2
from psycopg2.extras import RealDictCursor
import hashlib
import base64


@dataclass
class ProjectConfig:
    """Configuration for a Crawl4AI project"""
    project_id: str
    name: str
    supabase_url: str
    supabase_key: str  # Encrypted
    database_url: str  # Direct database connection URL (encrypted)
    created_at: datetime
    last_crawled: Optional[datetime]
    total_documents: int
    storage_used: int  # in MB
    rag_strategies: List[str]
    mcp_server_config: Optional[Dict[str, Any]] = None


class CredentialManager:
    """Manages secure storage and retrieval of credentials"""
    
    def __init__(self):
        # Generate or load encryption key
        self.key_file = os.path.join(os.path.expanduser("~"), ".crawl4ai", "secret.key")
        self.ensure_key()
    
    def ensure_key(self):
        """Ensure encryption key exists"""
        os.makedirs(os.path.dirname(self.key_file), exist_ok=True)
        
        if os.path.exists(self.key_file):
            with open(self.key_file, 'rb') as f:
                self.key = f.read()
        else:
            self.key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(self.key)
            os.chmod(self.key_file, 0o600)  # Restrict access
    
    def encrypt(self, data: str) -> str:
        """Encrypt sensitive data"""
        f = Fernet(self.key)
        return f.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        f = Fernet(self.key)
        return f.decrypt(encrypted_data.encode()).decode()
    
    def hash_project_name(self, name: str) -> str:
        """Generate unique project ID from name"""
        return f"proj_{hashlib.sha256(name.encode()).hexdigest()[:8]}"


class MCPServerInterface:
    """Interface for MCP server operations"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.base_url = self.config.get('base_url', 'http://localhost:8000')
    
    async def create_supabase_project(self, project_name: str, region: str = 'us-east-1') -> Dict[str, Any]:
        """Create a new Supabase project via MCP server"""
        # In a real implementation, this would make an API call to the MCP server
        # For now, we'll simulate the response
        import aiohttp
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "name": project_name,
                    "region": region,
                    "database_password": self._generate_secure_password()
                }
                
                # Simulated response for development
                # In production, this would be: 
                # async with session.post(f"{self.base_url}/create-project", json=payload) as resp:
                #     return await resp.json()
                
                return {
                    "project_id": hashlib.sha256(project_name.encode()).hexdigest()[:8],
                    "supabase_url": f"https://{project_name.lower().replace(' ', '-')}.supabase.co",
                    "anon_key": self._generate_mock_key("anon"),
                    "service_key": self._generate_mock_key("service"),
                    "database_url": f"postgresql://postgres:password@db.{project_name.lower().replace(' ', '-')}.supabase.co:5432/postgres"
                }
        except Exception as e:
            st.error(f"MCP Server Error: {e}")
            raise
    
    def _generate_secure_password(self) -> str:
        """Generate a secure password for database"""
        import secrets
        import string
        alphabet = string.ascii_letters + string.digits + string.punctuation
        return ''.join(secrets.choice(alphabet) for i in range(32))
    
    def _generate_mock_key(self, key_type: str) -> str:
        """Generate mock API key for development"""
        prefix = "eyJ" if key_type == "anon" else "eyK"
        return f"{prefix}{base64.b64encode(f'{key_type}_key_mock'.encode()).decode()}"


class SupabaseIntegration:
    """Main class for Supabase operations"""
    
    def __init__(self):
        self.projects = {}
        self.active_connection = None
        self.credential_manager = CredentialManager()
        self.mcp_interface = MCPServerInterface()
        self._load_projects()
    
    def _load_projects(self):
        """Load saved projects from persistent storage"""
        config_file = os.path.join(os.path.expanduser("~"), ".crawl4ai", "projects.json")
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    data = json.load(f)
                    for proj_id, proj_data in data.items():
                        # Convert datetime strings back to datetime objects
                        proj_data['created_at'] = datetime.fromisoformat(proj_data['created_at'])
                        if proj_data.get('last_crawled'):
                            proj_data['last_crawled'] = datetime.fromisoformat(proj_data['last_crawled'])
                        self.projects[proj_id] = ProjectConfig(**proj_data)
            except Exception as e:
                st.error(f"Error loading projects: {e}")
    
    def _save_projects(self):
        """Save projects to persistent storage"""
        config_file = os.path.join(os.path.expanduser("~"), ".crawl4ai", "projects.json")
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        
        data = {}
        for proj_id, config in self.projects.items():
            proj_dict = asdict(config)
            # Convert datetime objects to strings
            proj_dict['created_at'] = config.created_at.isoformat()
            if config.last_crawled:
                proj_dict['last_crawled'] = config.last_crawled.isoformat()
            data[proj_id] = proj_dict
        
        with open(config_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    async def create_project(self, name: str, initial_url: str = None) -> ProjectConfig:
        """Create a new Supabase project via MCP server"""
        
        # Generate project ID
        project_id = self.credential_manager.hash_project_name(name)
        
        # Check if project already exists
        if project_id in self.projects:
            st.warning(f"Project '{name}' already exists")
            return self.projects[project_id]
        
        # Create project via MCP server
        with st.spinner(f"Creating Supabase project '{name}'..."):
            project_info = await self.mcp_interface.create_supabase_project(name)
        
        # Encrypt sensitive credentials
        encrypted_key = self.credential_manager.encrypt(project_info['service_key'])
        encrypted_db_url = self.credential_manager.encrypt(project_info['database_url'])
        
        config = ProjectConfig(
            project_id=project_id,
            name=name,
            supabase_url=project_info['supabase_url'],
            supabase_key=encrypted_key,
            database_url=encrypted_db_url,
            created_at=datetime.now(),
            last_crawled=None,
            total_documents=0,
            storage_used=0,
            rag_strategies=["Contextual Embeddings", "Semantic Chunking"],
            mcp_server_config={
                "project_ref": project_info['project_id'],
                "region": "us-east-1"
            }
        )
        
        self.projects[project_id] = config
        self._save_projects()
        
        # Initialize database schema
        await self._initialize_database_schema(config)
        
        return config
    
    async def _initialize_database_schema(self, config: ProjectConfig):
        """Initialize the database schema for a new project"""
        
        # Decrypt database URL
        db_url = self.credential_manager.decrypt(config.database_url)
        
        # Read SQL schema file
        schema_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sql", "crawled_pages.sql")
        
        if not os.path.exists(schema_path):
            st.error(f"Schema file not found: {schema_path}")
            return False
        
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
        
        # Execute schema
        try:
            conn = psycopg2.connect(db_url)
            cursor = conn.cursor()
            
            # Execute the schema SQL
            cursor.execute(schema_sql)
            
            # Initialize project statistics
            cursor.execute(
                "SELECT update_project_statistics(%s)",
                (config.project_id,)
            )
            
            conn.commit()
            cursor.close()
            conn.close()
            
            st.success(f"Database schema initialized for project '{config.name}'")
            return True
            
        except Exception as e:
            st.error(f"Error initializing database schema: {e}")
            return False
    
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
        allowed_fields = ['last_crawled', 'total_documents', 'storage_used', 'rag_strategies']
        for key, value in updates.items():
            if key in allowed_fields and hasattr(config, key):
                setattr(config, key, value)
        
        self._save_projects()
        return True
    
    def delete_project(self, project_id: str) -> bool:
        """Delete a project and all its data"""
        
        if project_id not in self.projects:
            return False
        
        config = self.projects[project_id]
        
        try:
            # Get database connection
            db_url = self.credential_manager.decrypt(config.database_url)
            conn = psycopg2.connect(db_url)
            cursor = conn.cursor()
            
            # Delete all project data
            tables = ['search_history', 'content_chunks', 'code_examples', 'crawled_pages', 'sources', 'project_statistics']
            for table in tables:
                cursor.execute(f"DELETE FROM {table} WHERE project_id = %s", (project_id,))
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            st.error(f"Error deleting project data: {e}")
            return False
        
        del self.projects[project_id]
        self._save_projects()
        return True
    
    def get_connection(self, project_id: str) -> Client:
        """Get Supabase client connection for a specific project"""
        
        config = self.get_project(project_id)
        if not config:
            raise ValueError(f"Project {project_id} not found")
        
        # Decrypt credentials
        supabase_key = self.credential_manager.decrypt(config.supabase_key)
        
        return create_client(config.supabase_url, supabase_key)
    
    def get_db_connection(self, project_id: str) -> psycopg2.extensions.connection:
        """Get direct database connection for a specific project"""
        
        config = self.get_project(project_id)
        if not config:
            raise ValueError(f"Project {project_id} not found")
        
        # Decrypt database URL
        db_url = self.credential_manager.decrypt(config.database_url)
        
        return psycopg2.connect(db_url, cursor_factory=RealDictCursor)
    
    def test_connection(self, project_id: str) -> bool:
        """Test connection to a project's database"""
        
        try:
            # Test Supabase client
            client = self.get_connection(project_id)
            # Simple test query
            result = client.table('sources').select('id').limit(1).execute()
            
            # Test direct database connection
            conn = self.get_db_connection(project_id)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            conn.close()
            
            return True
        except Exception as e:
            st.error(f"Connection test failed: {e}")
            return False


# Global instance
supabase_integration = SupabaseIntegration()


def get_supabase_integration() -> SupabaseIntegration:
    """Get the global Supabase integration instance"""
    return supabase_integration


async def create_new_project(name: str, initial_url: str = None) -> ProjectConfig:
    """Convenience function to create a new project"""
    return await supabase_integration.create_project(name, initial_url)


def get_project_list() -> List[ProjectConfig]:
    """Convenience function to get all projects"""
    return supabase_integration.list_projects()


def validate_supabase_credentials(url: str, key: str) -> bool:
    """Validate Supabase credentials"""
    
    if not url or not key:
        return False
    
    if not url.startswith("https://"):
        return False
    
    if ".supabase.co" not in url:
        return False
    
    try:
        # Try to create a client with the credentials
        client = create_client(url, key)
        # Simple test query
        client.table('test').select('*').limit(1).execute()
        return True
    except:
        return False 