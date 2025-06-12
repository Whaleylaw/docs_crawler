"""
Unit tests for the configuration management component.
"""

import pytest
import json
import yaml
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

from components.configuration import (
    OpenAIConfig, SupabaseConfig, CrawlingConfig, RAGConfig,
    UIConfig, PerformanceConfig, LoggingConfig, SecurityConfig,
    ApplicationConfig, ConfigurationManager
)


@pytest.mark.unit
class TestOpenAIConfig:
    """Test OpenAI configuration."""
    
    def test_creation(self):
        """Test OpenAI config creation."""
        config = OpenAIConfig(api_key="test-key", model="gpt-4")
        assert config.api_key == "test-key"
        assert config.model == "gpt-4"
        assert config.embedding_model == "text-embedding-3-small"
    
    def test_default_values(self):
        """Test default values are set correctly."""
        config = OpenAIConfig(api_key="test-key")
        assert config.model == "gpt-3.5-turbo"
        assert config.temperature == 0.7
        assert config.max_tokens == 2000
    
    def test_validation(self):
        """Test configuration validation."""
        # Valid config should not raise
        config = OpenAIConfig(api_key="sk-test123", model="gpt-4")
        config.validate()
        
        # Invalid API key should raise
        invalid_config = OpenAIConfig(api_key="invalid", model="gpt-4")
        with pytest.raises(ValueError, match="API key must start with 'sk-'"):
            invalid_config.validate()


@pytest.mark.unit
class TestSupabaseConfig:
    """Test Supabase configuration."""
    
    def test_creation(self):
        """Test Supabase config creation."""
        config = SupabaseConfig(
            url="https://test.supabase.co",
            service_key="test-key"
        )
        assert config.url == "https://test.supabase.co"
        assert config.service_key == "test-key"
    
    def test_validation(self):
        """Test configuration validation."""
        # Valid config should not raise
        config = SupabaseConfig(
            url="https://test.supabase.co",
            service_key="eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.test"
        )
        config.validate()
        
        # Invalid URL should raise
        invalid_config = SupabaseConfig(url="invalid-url", service_key="test")
        with pytest.raises(ValueError, match="URL must be a valid HTTPS URL"):
            invalid_config.validate()


@pytest.mark.unit
class TestApplicationConfig:
    """Test main application configuration."""
    
    def test_creation_with_dicts(self):
        """Test creation from dictionary data."""
        config = ApplicationConfig(
            openai={"api_key": "sk-test", "model": "gpt-4"},
            supabase={"url": "https://test.supabase.co", "service_key": "test"},
            crawling={"max_concurrent": 10}
        )
        
        assert isinstance(config.openai, OpenAIConfig)
        assert config.openai.api_key == "sk-test"
        assert config.crawling.max_concurrent == 10
    
    def test_creation_with_objects(self):
        """Test creation from config objects."""
        openai_config = OpenAIConfig(api_key="sk-test", model="gpt-4")
        config = ApplicationConfig(openai=openai_config)
        
        assert config.openai == openai_config
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = ApplicationConfig(
            openai={"api_key": "sk-test", "model": "gpt-4"},
            crawling={"max_concurrent": 5}
        )
        
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["openai"]["api_key"] == "sk-test"
        assert config_dict["crawling"]["max_concurrent"] == 5
    
    def test_save_json(self, temp_dir):
        """Test saving configuration to JSON."""
        config = ApplicationConfig(
            openai={"api_key": "sk-test", "model": "gpt-4"}
        )
        
        json_path = temp_dir / "config.json"
        config.save_json(json_path)
        
        assert json_path.exists()
        with open(json_path) as f:
            saved_data = json.load(f)
        
        assert saved_data["openai"]["api_key"] == "sk-test"
    
    def test_load_json(self, temp_dir):
        """Test loading configuration from JSON."""
        config_data = {
            "openai": {"api_key": "sk-test", "model": "gpt-4"},
            "crawling": {"max_concurrent": 15}
        }
        
        json_path = temp_dir / "config.json"
        with open(json_path, 'w') as f:
            json.dump(config_data, f)
        
        config = ApplicationConfig.load_json(json_path)
        assert config.openai.api_key == "sk-test"
        assert config.crawling.max_concurrent == 15
    
    def test_save_yaml(self, temp_dir):
        """Test saving configuration to YAML."""
        config = ApplicationConfig(
            openai={"api_key": "sk-test", "model": "gpt-4"}
        )
        
        yaml_path = temp_dir / "config.yaml"
        config.save_yaml(yaml_path)
        
        assert yaml_path.exists()
        with open(yaml_path) as f:
            saved_data = yaml.safe_load(f)
        
        assert saved_data["openai"]["api_key"] == "sk-test"
    
    def test_load_yaml(self, temp_dir):
        """Test loading configuration from YAML."""
        config_data = {
            "openai": {"api_key": "sk-test", "model": "gpt-4"},
            "crawling": {"max_concurrent": 20}
        }
        
        yaml_path = temp_dir / "config.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(config_data, f)
        
        config = ApplicationConfig.load_yaml(yaml_path)
        assert config.openai.api_key == "sk-test"
        assert config.crawling.max_concurrent == 20


@pytest.mark.unit
class TestConfigurationManager:
    """Test configuration manager."""
    
    def test_creation(self):
        """Test manager creation."""
        manager = ConfigurationManager()
        assert manager.config is not None
        assert isinstance(manager.config, ApplicationConfig)
    
    def test_load_from_env(self):
        """Test loading configuration from environment variables."""
        manager = ConfigurationManager()
        
        with patch.dict('os.environ', {
            'OPENAI_API_KEY': 'sk-env-test',
            'OPENAI_MODEL': 'gpt-4',
            'SUPABASE_URL': 'https://env.supabase.co',
            'MAX_CONCURRENT': '25'
        }):
            manager.load_from_env()
        
        assert manager.config.openai.api_key == 'sk-env-test'
        assert manager.config.openai.model == 'gpt-4'
        assert manager.config.supabase.url == 'https://env.supabase.co'
        assert manager.config.crawling.max_concurrent == 25
    
    def test_validate_all(self):
        """Test validation of all configuration sections."""
        manager = ConfigurationManager()
        manager.config = ApplicationConfig(
            openai={"api_key": "sk-test", "model": "gpt-4"},
            supabase={"url": "https://test.supabase.co", "service_key": "test"}
        )
        
        # Should not raise with valid config
        manager.validate()
    
    def test_validate_invalid(self):
        """Test validation with invalid configuration."""
        manager = ConfigurationManager()
        manager.config = ApplicationConfig(
            openai={"api_key": "invalid-key", "model": "gpt-4"}
        )
        
        with pytest.raises(ValueError):
            manager.validate()
    
    def test_export_config(self, temp_dir):
        """Test exporting configuration."""
        manager = ConfigurationManager()
        manager.config = ApplicationConfig(
            openai={"api_key": "sk-test"},
            crawling={"max_concurrent": 10}
        )
        
        export_path = temp_dir / "export.json"
        manager.export_config(export_path)
        
        assert export_path.exists()
        with open(export_path) as f:
            exported = json.load(f)
        
        assert exported["openai"]["api_key"] == "sk-test"
        assert exported["crawling"]["max_concurrent"] == 10
    
    def test_import_config(self, temp_dir):
        """Test importing configuration."""
        config_data = {
            "openai": {"api_key": "sk-imported", "model": "gpt-4"},
            "crawling": {"max_concurrent": 30}
        }
        
        import_path = temp_dir / "import.json" 
        with open(import_path, 'w') as f:
            json.dump(config_data, f)
        
        manager = ConfigurationManager()
        manager.import_config(import_path)
        
        assert manager.config.openai.api_key == "sk-imported"
        assert manager.config.crawling.max_concurrent == 30
    
    def test_get_section(self):
        """Test getting specific configuration sections."""
        manager = ConfigurationManager()
        manager.config = ApplicationConfig(
            openai={"api_key": "sk-test", "model": "gpt-4"}
        )
        
        openai_section = manager.get_section("openai")
        assert isinstance(openai_section, OpenAIConfig)
        assert openai_section.api_key == "sk-test"
        
        # Non-existent section should return None
        missing_section = manager.get_section("nonexistent")
        assert missing_section is None
    
    def test_update_section(self):
        """Test updating specific configuration sections."""
        manager = ConfigurationManager()
        
        manager.update_section("openai", {"api_key": "sk-updated", "model": "gpt-4"})
        assert manager.config.openai.api_key == "sk-updated"
        assert manager.config.openai.model == "gpt-4"
        
        manager.update_section("crawling", {"max_concurrent": 50})
        assert manager.config.crawling.max_concurrent == 50


@pytest.mark.unit
class TestConfigValidation:
    """Test configuration validation edge cases."""
    
    def test_crawling_config_validation(self):
        """Test crawling configuration validation."""
        # Valid config
        config = CrawlingConfig(max_concurrent=10, timeout=30)
        config.validate()
        
        # Invalid max_concurrent
        invalid_config = CrawlingConfig(max_concurrent=0, timeout=30)
        with pytest.raises(ValueError, match="max_concurrent must be positive"):
            invalid_config.validate()
        
        # Invalid timeout
        invalid_config = CrawlingConfig(max_concurrent=10, timeout=0)
        with pytest.raises(ValueError, match="timeout must be positive"):
            invalid_config.validate()
    
    def test_rag_config_validation(self):
        """Test RAG configuration validation."""
        # Valid config
        config = RAGConfig(chunk_size=1000, chunk_overlap=200)
        config.validate()
        
        # Invalid chunk_size
        invalid_config = RAGConfig(chunk_size=0, chunk_overlap=200)
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            invalid_config.validate()
        
        # Overlap too large
        invalid_config = RAGConfig(chunk_size=1000, chunk_overlap=1000)
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            invalid_config.validate()