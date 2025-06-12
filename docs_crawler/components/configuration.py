"""
Configuration Management System
Handles application settings, environment variables, and user preferences
"""

import os
import json
import yaml
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
import streamlit as st
from enum import Enum


class ConfigCategory(Enum):
    """Configuration categories"""
    GENERAL = "general"
    SUPABASE = "supabase" 
    OPENAI = "openai"
    CRAWLING = "crawling"
    RAG = "rag"
    UI = "ui"
    PERFORMANCE = "performance"
    LOGGING = "logging"
    SECURITY = "security"


class ModelProvider(Enum):
    """Supported AI model providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"


@dataclass
class OpenAIConfig:
    """OpenAI API configuration"""
    api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-3.5-turbo"
    max_tokens: int = 1000
    temperature: float = 0.3
    timeout: int = 30
    max_retries: int = 3
    organization: str = ""
    

@dataclass
class SupabaseConfig:
    """Supabase configuration"""
    url: str = ""
    service_key: str = ""
    anon_key: str = ""
    timeout: int = 30
    max_connections: int = 10
    enable_realtime: bool = False


@dataclass
class CrawlingConfig:
    """Web crawling configuration"""
    max_concurrent: int = 10
    max_depth: int = 3
    chunk_size: int = 4000
    chunk_overlap: int = 200
    request_timeout: int = 30
    delay_between_requests: float = 1.0
    user_agent: str = "Crawl4AI-Standalone/1.0"
    respect_robots_txt: bool = True
    max_file_size_mb: int = 50
    allowed_domains: List[str] = field(default_factory=list)
    blocked_domains: List[str] = field(default_factory=list)
    default_headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class RAGConfig:
    """RAG processing configuration"""
    enabled_strategies: List[str] = field(default_factory=lambda: ["vector_embeddings"])
    use_contextual_embeddings: bool = False
    extract_code_examples: bool = False
    use_reranking: bool = False
    use_hybrid_search: bool = False
    similarity_threshold: float = 0.7
    max_results: int = 20
    context_model: str = "gpt-3.5-turbo"
    reranking_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    min_code_length: int = 1000
    parallel_workers: int = 10
    batch_size: int = 20


@dataclass
class UIConfig:
    """User interface configuration"""
    theme: str = "light"
    page_title: str = "Crawl4AI Standalone"
    show_sidebar: bool = True
    default_page: str = "home"
    items_per_page: int = 10
    show_progress_bars: bool = True
    auto_refresh_interval: int = 30
    enable_tooltips: bool = True
    compact_mode: bool = False


@dataclass
class PerformanceConfig:
    """Performance and optimization settings"""
    enable_caching: bool = True
    cache_ttl_minutes: int = 60
    max_memory_mb: int = 1024
    enable_compression: bool = True
    async_processing: bool = True
    connection_pool_size: int = 10
    query_timeout: int = 30
    enable_metrics: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    enable_file_logging: bool = True
    log_file_path: str = "logs/app.log"
    max_log_size_mb: int = 100
    log_retention_days: int = 30
    enable_error_tracking: bool = True
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    enable_performance_logging: bool = False


@dataclass
class SecurityConfig:
    """Security configuration"""
    encrypt_credentials: bool = True
    session_timeout_minutes: int = 480  # 8 hours
    enable_csrf_protection: bool = True
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    enable_rate_limiting: bool = True
    rate_limit_requests_per_minute: int = 100
    enable_audit_logging: bool = True
    password_min_length: int = 8


@dataclass
class ApplicationConfig:
    """Complete application configuration"""
    
    # Core configuration sections
    general: Dict[str, Any] = field(default_factory=dict)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    supabase: SupabaseConfig = field(default_factory=SupabaseConfig)
    crawling: CrawlingConfig = field(default_factory=CrawlingConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Metadata
    version: str = "1.0.0"
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    environment: str = "development"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self)
    
    def save_to_file(self, file_path: str):
        """Save configuration to file"""
        config_dict = self.to_dict()
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
        elif file_path.suffix.lower() in ['.yml', '.yaml']:
            with open(file_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'ApplicationConfig':
        """Load configuration from file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return cls()  # Return default config
        
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'r') as f:
                config_dict = json.load(f)
        elif file_path.suffix.lower() in ['.yml', '.yaml']:
            with open(file_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ApplicationConfig':
        """Create configuration from dictionary"""
        config = cls()
        
        # Update each section if present
        if 'openai' in config_dict:
            config.openai = OpenAIConfig(**config_dict['openai'])
        if 'supabase' in config_dict:
            config.supabase = SupabaseConfig(**config_dict['supabase'])
        if 'crawling' in config_dict:
            config.crawling = CrawlingConfig(**config_dict['crawling'])
        if 'rag' in config_dict:
            config.rag = RAGConfig(**config_dict['rag'])
        if 'ui' in config_dict:
            config.ui = UIConfig(**config_dict['ui'])
        if 'performance' in config_dict:
            config.performance = PerformanceConfig(**config_dict['performance'])
        if 'logging' in config_dict:
            config.logging = LoggingConfig(**config_dict['logging'])
        if 'security' in config_dict:
            config.security = SecurityConfig(**config_dict['security'])
        
        # Update metadata
        config.version = config_dict.get('version', config.version)
        config.environment = config_dict.get('environment', config.environment)
        config.general = config_dict.get('general', {})
        
        return config


class ConfigurationManager:
    """Main configuration management class"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = Path(config_file)
        self.config = ApplicationConfig()
        self._load_config()
        self._load_environment_variables()
    
    def _load_config(self):
        """Load configuration from file"""
        try:
            if self.config_file.exists():
                self.config = ApplicationConfig.load_from_file(str(self.config_file))
        except Exception as e:
            print(f"Error loading config file: {e}")
            # Use default configuration
            self.config = ApplicationConfig()
    
    def _load_environment_variables(self):
        """Load configuration from environment variables"""
        
        # OpenAI configuration
        if os.getenv("OPENAI_API_KEY"):
            self.config.openai.api_key = os.getenv("OPENAI_API_KEY")
        if os.getenv("OPENAI_EMBEDDING_MODEL"):
            self.config.openai.embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL")
        if os.getenv("OPENAI_CHAT_MODEL"):
            self.config.openai.chat_model = os.getenv("OPENAI_CHAT_MODEL")
        
        # Supabase configuration
        if os.getenv("SUPABASE_URL"):
            self.config.supabase.url = os.getenv("SUPABASE_URL")
        if os.getenv("SUPABASE_SERVICE_KEY"):
            self.config.supabase.service_key = os.getenv("SUPABASE_SERVICE_KEY")
        if os.getenv("SUPABASE_ANON_KEY"):
            self.config.supabase.anon_key = os.getenv("SUPABASE_ANON_KEY")
        
        # Environment
        if os.getenv("ENVIRONMENT"):
            self.config.environment = os.getenv("ENVIRONMENT")
        
        # Performance settings
        if os.getenv("MAX_CONCURRENT"):
            self.config.crawling.max_concurrent = int(os.getenv("MAX_CONCURRENT", "10"))
        if os.getenv("CHUNK_SIZE"):
            self.config.crawling.chunk_size = int(os.getenv("CHUNK_SIZE", "4000"))
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            self.config.last_updated = datetime.now().isoformat()
            self.config.save_to_file(str(self.config_file))
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    def get_config(self) -> ApplicationConfig:
        """Get current configuration"""
        return self.config
    
    def update_config(self, category: ConfigCategory, updates: Dict[str, Any]):
        """Update configuration for a specific category"""
        
        if category == ConfigCategory.OPENAI:
            for key, value in updates.items():
                if hasattr(self.config.openai, key):
                    setattr(self.config.openai, key, value)
        elif category == ConfigCategory.SUPABASE:
            for key, value in updates.items():
                if hasattr(self.config.supabase, key):
                    setattr(self.config.supabase, key, value)
        elif category == ConfigCategory.CRAWLING:
            for key, value in updates.items():
                if hasattr(self.config.crawling, key):
                    setattr(self.config.crawling, key, value)
        elif category == ConfigCategory.RAG:
            for key, value in updates.items():
                if hasattr(self.config.rag, key):
                    setattr(self.config.rag, key, value)
        elif category == ConfigCategory.UI:
            for key, value in updates.items():
                if hasattr(self.config.ui, key):
                    setattr(self.config.ui, key, value)
        elif category == ConfigCategory.PERFORMANCE:
            for key, value in updates.items():
                if hasattr(self.config.performance, key):
                    setattr(self.config.performance, key, value)
        elif category == ConfigCategory.LOGGING:
            for key, value in updates.items():
                if hasattr(self.config.logging, key):
                    setattr(self.config.logging, key, value)
        elif category == ConfigCategory.SECURITY:
            for key, value in updates.items():
                if hasattr(self.config.security, key):
                    setattr(self.config.security, key, value)
        elif category == ConfigCategory.GENERAL:
            self.config.general.update(updates)
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Validate OpenAI configuration
        if not self.config.openai.api_key:
            issues.append("OpenAI API key is not configured")
        
        # Validate models
        valid_embedding_models = [
            "text-embedding-3-small", "text-embedding-3-large", 
            "text-embedding-ada-002"
        ]
        if self.config.openai.embedding_model not in valid_embedding_models:
            issues.append(f"Invalid embedding model: {self.config.openai.embedding_model}")
        
        # Validate numeric ranges
        if self.config.openai.temperature < 0 or self.config.openai.temperature > 2:
            issues.append("OpenAI temperature must be between 0 and 2")
        
        if self.config.crawling.max_concurrent < 1 or self.config.crawling.max_concurrent > 100:
            issues.append("Max concurrent crawls must be between 1 and 100")
        
        if self.config.crawling.chunk_size < 100 or self.config.crawling.chunk_size > 20000:
            issues.append("Chunk size must be between 100 and 20000")
        
        # Validate file paths
        if self.config.logging.enable_file_logging:
            log_dir = Path(self.config.logging.log_file_path).parent
            if not log_dir.exists():
                try:
                    log_dir.mkdir(parents=True, exist_ok=True)
                except Exception:
                    issues.append(f"Cannot create log directory: {log_dir}")
        
        return issues
    
    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        self.config = ApplicationConfig()
        self._load_environment_variables()
    
    def export_config(self, format: str = "json") -> str:
        """Export configuration as string"""
        config_dict = self.config.to_dict()
        
        if format.lower() == "json":
            return json.dumps(config_dict, indent=2, default=str)
        elif format.lower() == "yaml":
            return yaml.dump(config_dict, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def import_config(self, config_str: str, format: str = "json"):
        """Import configuration from string"""
        try:
            if format.lower() == "json":
                config_dict = json.loads(config_str)
            elif format.lower() == "yaml":
                config_dict = yaml.safe_load(config_str)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.config = ApplicationConfig.from_dict(config_dict)
            return True
        except Exception as e:
            print(f"Error importing config: {e}")
            return False
    
    def get_environment_config(self) -> Dict[str, str]:
        """Get configuration as environment variables"""
        env_vars = {}
        
        env_vars["OPENAI_API_KEY"] = self.config.openai.api_key
        env_vars["OPENAI_EMBEDDING_MODEL"] = self.config.openai.embedding_model
        env_vars["OPENAI_CHAT_MODEL"] = self.config.openai.chat_model
        env_vars["SUPABASE_URL"] = self.config.supabase.url
        env_vars["SUPABASE_SERVICE_KEY"] = self.config.supabase.service_key
        env_vars["ENVIRONMENT"] = self.config.environment
        env_vars["MAX_CONCURRENT"] = str(self.config.crawling.max_concurrent)
        env_vars["CHUNK_SIZE"] = str(self.config.crawling.chunk_size)
        
        return env_vars


# Global configuration manager instance
_config_manager = None


def get_config_manager() -> ConfigurationManager:
    """Get the global configuration manager instance"""
    global _config_manager
    
    if _config_manager is None:
        config_file = os.getenv("CONFIG_FILE", "config.json")
        _config_manager = ConfigurationManager(config_file)
    
    return _config_manager


def get_config() -> ApplicationConfig:
    """Get the current application configuration"""
    return get_config_manager().get_config()


# Session state configuration cache
def get_session_config() -> ApplicationConfig:
    """Get configuration from Streamlit session state"""
    if 'app_config' not in st.session_state:
        st.session_state.app_config = get_config()
    return st.session_state.app_config


def update_session_config(config: ApplicationConfig):
    """Update configuration in Streamlit session state"""
    st.session_state.app_config = config
    # Also update the global config manager
    config_manager = get_config_manager()
    config_manager.config = config