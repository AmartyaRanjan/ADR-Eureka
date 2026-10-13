"""Unit tests for ConfigurationManager.

Tests the load_env_config method with various valid and invalid scenarios
to ensure proper parsing, validation, and error handling.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7**
"""

import pytest
import tempfile
import os
from pathlib import Path

from src.config import ConfigurationManager, EnvConfig
from src.exceptions import ConfigurationError


class TestLoadEnvConfig:
    """Test suite for ConfigurationManager.load_env_config method."""
    
    def test_load_valid_env_config(self):
        """Test loading a valid .env file with API key and model."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            f.write('GEMINI_API_KEY=test_api_key_12345\n')
            f.write('LLM_MODEL=gemini-1.5-pro\n')
            env_path = f.name
        
        try:
            manager = ConfigurationManager()
            config = manager.load_env_config(env_path)
            
            assert isinstance(config, EnvConfig)
            assert config.gemini_api_key == 'test_api_key_12345'
            assert config.llm_model == 'gemini-1.5-pro'
        finally:
            os.unlink(env_path)
    
    def test_load_env_config_with_default_model(self):
        """Test loading .env file without LLM_MODEL uses default."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            f.write('GEMINI_API_KEY=test_api_key_12345\n')
            env_path = f.name
        
        try:
            manager = ConfigurationManager()
            config = manager.load_env_config(env_path)
            
            assert config.gemini_api_key == 'test_api_key_12345'
            assert config.llm_model == 'gemini-1.5-pro'  # Default value
        finally:
            os.unlink(env_path)
    
    def test_load_env_config_with_quoted_values(self):
        """Test loading .env file with quoted values."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            f.write('GEMINI_API_KEY="test_api_key_12345"\n')
            f.write('LLM_MODEL=\'gemini-1.5-flash\'\n')
            env_path = f.name
        
        try:
            manager = ConfigurationManager()
            config = manager.load_env_config(env_path)
            
            assert config.gemini_api_key == 'test_api_key_12345'
            assert config.llm_model == 'gemini-1.5-flash'
        finally:
            os.unlink(env_path)
    
    def test_load_env_config_with_comments_and_blank_lines(self):
        """Test loading .env file with comments and blank lines."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            f.write('# This is a comment\n')
            f.write('\n')
            f.write('GEMINI_API_KEY=test_api_key_12345\n')
            f.write('  \n')
            f.write('# Another comment\n')
            f.write('LLM_MODEL=gemini-1.5-pro\n')
            env_path = f.name
        
        try:
            manager = ConfigurationManager()
            config = manager.load_env_config(env_path)
            
            assert config.gemini_api_key == 'test_api_key_12345'
            assert config.llm_model == 'gemini-1.5-pro'
        finally:
            os.unlink(env_path)
    
    def test_load_env_config_with_spaces_around_equals(self):
        """Test loading .env file with spaces around the equals sign."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            f.write('GEMINI_API_KEY = test_api_key_12345\n')
            f.write('LLM_MODEL  =  gemini-1.5-pro\n')
            env_path = f.name
        
        try:
            manager = ConfigurationManager()
            config = manager.load_env_config(env_path)
            
            assert config.gemini_api_key == 'test_api_key_12345'
            assert config.llm_model == 'gemini-1.5-pro'
        finally:
            os.unlink(env_path)
    
    def test_missing_env_file(self):
        """Test that missing .env file raises ConfigurationError."""
        manager = ConfigurationManager()
        
        with pytest.raises(ConfigurationError) as exc_info:
            manager.load_env_config('/nonexistent/path/.env')
        
        assert 'not found' in str(exc_info.value).lower()
    
    def test_env_path_is_directory(self):
        """Test that directory path instead of file raises ConfigurationError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ConfigurationManager()
            
            with pytest.raises(ConfigurationError) as exc_info:
                manager.load_env_config(tmpdir)
            
            assert 'not a file' in str(exc_info.value).lower()
    
    def test_unreadable_env_file(self):
        """Test that unreadable .env file raises ConfigurationError."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            f.write('GEMINI_API_KEY=test_api_key_12345\n')
            env_path = f.name
        
        try:
            # Remove read permissions
            os.chmod(env_path, 0o000)
            
            manager = ConfigurationManager()
            
            with pytest.raises(ConfigurationError) as exc_info:
                manager.load_env_config(env_path)
            
            assert 'failed to read' in str(exc_info.value).lower()
        finally:
            # Restore permissions before cleanup
            os.chmod(env_path, 0o644)
            os.unlink(env_path)
    
    def test_missing_gemini_api_key(self):
        """Test that missing GEMINI_API_KEY raises ConfigurationError."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            f.write('LLM_MODEL=gemini-1.5-pro\n')
            env_path = f.name
        
        try:
            manager = ConfigurationManager()
            
            with pytest.raises(ConfigurationError) as exc_info:
                manager.load_env_config(env_path)
            
            assert 'GEMINI_API_KEY' in str(exc_info.value)
            assert 'missing or empty' in str(exc_info.value).lower()
        finally:
            os.unlink(env_path)
    
    def test_empty_gemini_api_key(self):
        """Test that empty GEMINI_API_KEY raises ConfigurationError."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            f.write('GEMINI_API_KEY=\n')
            f.write('LLM_MODEL=gemini-1.5-pro\n')
            env_path = f.name
        
        try:
            manager = ConfigurationManager()
            
            with pytest.raises(ConfigurationError) as exc_info:
                manager.load_env_config(env_path)
            
            assert 'GEMINI_API_KEY' in str(exc_info.value)
            assert 'missing or empty' in str(exc_info.value).lower()
        finally:
            os.unlink(env_path)
    
    def test_whitespace_only_gemini_api_key(self):
        """Test that whitespace-only GEMINI_API_KEY raises ConfigurationError."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            f.write('GEMINI_API_KEY=   \n')
            f.write('LLM_MODEL=gemini-1.5-pro\n')
            env_path = f.name
        
        try:
            manager = ConfigurationManager()
            
            with pytest.raises(ConfigurationError) as exc_info:
                manager.load_env_config(env_path)
            
            assert 'GEMINI_API_KEY' in str(exc_info.value)
            assert 'missing or empty' in str(exc_info.value).lower()
        finally:
            os.unlink(env_path)
    
    def test_invalid_env_file_format(self):
        """Test that invalid .env file format raises ConfigurationError."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            f.write('GEMINI_API_KEY=test_api_key_12345\n')
            f.write('INVALID_LINE_WITHOUT_EQUALS\n')
            env_path = f.name
        
        try:
            manager = ConfigurationManager()
            
            with pytest.raises(ConfigurationError) as exc_info:
                manager.load_env_config(env_path)
            
            assert 'invalid format' in str(exc_info.value).lower()
            assert 'KEY=value' in str(exc_info.value)
        finally:
            os.unlink(env_path)
    
    def test_empty_llm_model_uses_default(self):
        """Test that empty LLM_MODEL uses default value."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            f.write('GEMINI_API_KEY=test_api_key_12345\n')
            f.write('LLM_MODEL=\n')
            env_path = f.name
        
        try:
            manager = ConfigurationManager()
            config = manager.load_env_config(env_path)
            
            assert config.gemini_api_key == 'test_api_key_12345'
            assert config.llm_model == 'gemini-1.5-pro'  # Default value
        finally:
            os.unlink(env_path)
    
    def test_default_env_path(self):
        """Test loading from default .env path in current directory."""
        # Create temporary directory and .env file
        original_dir = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)
                
                # Create .env file in temp directory
                env_path = Path(tmpdir) / '.env'
                with open(env_path, 'w') as f:
                    f.write('GEMINI_API_KEY=test_api_key_12345\n')
                    f.write('LLM_MODEL=gemini-1.5-flash\n')
                
                manager = ConfigurationManager()
                config = manager.load_env_config()  # No path specified
                
                assert config.gemini_api_key == 'test_api_key_12345'
                assert config.llm_model == 'gemini-1.5-flash'
            finally:
                os.chdir(original_dir)
