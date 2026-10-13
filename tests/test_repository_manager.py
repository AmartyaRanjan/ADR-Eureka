"""Unit tests for Repository Manager.

This module tests the repository cloning and setup functionality,
including directory conflict detection and network failure handling.

**Validates: Requirements 1.1, 1.2, 1.3**
"""

import os
import subprocess
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from src.exceptions import RepositorySetupError
from src.utils.repository_manager import RepositoryManager
from src.utils.models import SetupResult


class TestRepositoryManager:
    """Test suite for RepositoryManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.repo_manager = RepositoryManager()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up temp directory if it exists
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_successful_repository_clone(self):
        """Test successful repository cloning.
        
        **Validates: Requirement 1.1**
        """
        target_dir = os.path.join(self.temp_dir, "isaaclab_test")
        
        # Mock subprocess.run to simulate successful clone
        mock_result = mock.MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        
        with mock.patch('subprocess.run', return_value=mock_result) as mock_run:
            result = self.repo_manager.setup_repository(target_dir)
            
            # Verify result
            assert result.success is True
            assert result.error_message is None
            assert result.repository_path == str(Path(target_dir).resolve())
            
            # Verify git clone was called with correct arguments
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert args[0] == "git"
            assert args[1] == "clone"
            assert args[2] == RepositoryManager.REPOSITORY_URL
            assert Path(args[3]) == Path(target_dir).resolve()
    
    def test_directory_conflict_error(self):
        """Test that existing directory causes conflict error.
        
        **Validates: Requirement 1.3**
        """
        # Create an existing directory
        existing_dir = os.path.join(self.temp_dir, "existing")
        os.makedirs(existing_dir, exist_ok=True)
        
        # Try to clone into existing directory
        with pytest.raises(RepositorySetupError) as exc_info:
            self.repo_manager.setup_repository(existing_dir)
        
        # Verify error message mentions directory conflict
        error_message = str(exc_info.value)
        assert "Directory conflict" in error_message
        assert "already exists" in error_message
        assert existing_dir in error_message
    
    def test_network_failure_handling(self):
        """Test network failure error handling.
        
        **Validates: Requirement 1.2**
        """
        target_dir = os.path.join(self.temp_dir, "network_fail_test")
        
        # Mock subprocess.run to simulate network failure
        mock_result = mock.MagicMock()
        mock_result.returncode = 128
        mock_result.stderr = "fatal: unable to access 'https://github.com/...': Could not resolve host"
        
        with mock.patch('subprocess.run', return_value=mock_result):
            with pytest.raises(RepositorySetupError) as exc_info:
                self.repo_manager.setup_repository(target_dir)
            
            # Verify error message indicates network failure
            error_message = str(exc_info.value)
            assert "Network failure" in error_message
            assert "internet connection" in error_message
    
    def test_connection_timeout_handling(self):
        """Test connection timeout error handling.
        
        **Validates: Requirement 1.2**
        """
        target_dir = os.path.join(self.temp_dir, "timeout_test")
        
        # Mock subprocess.run to simulate connection timeout
        mock_result = mock.MagicMock()
        mock_result.returncode = 128
        mock_result.stderr = "fatal: unable to access 'https://github.com/...': Failed to connect to github.com port 443: Connection timed out"
        
        with mock.patch('subprocess.run', return_value=mock_result):
            with pytest.raises(RepositorySetupError) as exc_info:
                self.repo_manager.setup_repository(target_dir)
            
            # Verify error message indicates network failure
            error_message = str(exc_info.value)
            assert "Network failure" in error_message
    
    def test_ssl_certificate_error_handling(self):
        """Test SSL certificate error handling.
        
        **Validates: Requirement 1.2**
        """
        target_dir = os.path.join(self.temp_dir, "ssl_test")
        
        # Mock subprocess.run to simulate SSL error
        mock_result = mock.MagicMock()
        mock_result.returncode = 128
        mock_result.stderr = "fatal: unable to access 'https://github.com/...': SSL certificate problem: certificate verify failed"
        
        with mock.patch('subprocess.run', return_value=mock_result):
            with pytest.raises(RepositorySetupError) as exc_info:
                self.repo_manager.setup_repository(target_dir)
            
            # Verify error message indicates network failure
            error_message = str(exc_info.value)
            assert "Network failure" in error_message
    
    def test_generic_clone_failure(self):
        """Test generic clone failure (non-network error).
        
        **Validates: Requirement 1.2**
        """
        target_dir = os.path.join(self.temp_dir, "generic_fail_test")
        
        # Mock subprocess.run to simulate generic failure
        mock_result = mock.MagicMock()
        mock_result.returncode = 128
        mock_result.stderr = "fatal: repository 'https://github.com/...' not found"
        
        with mock.patch('subprocess.run', return_value=mock_result):
            with pytest.raises(RepositorySetupError) as exc_info:
                self.repo_manager.setup_repository(target_dir)
            
            # Verify error message
            error_message = str(exc_info.value)
            assert "Failed to clone repository" in error_message
    
    def test_subprocess_timeout(self):
        """Test subprocess timeout during clone.
        
        **Validates: Requirement 1.2**
        """
        target_dir = os.path.join(self.temp_dir, "subprocess_timeout_test")
        
        # Mock subprocess.run to raise TimeoutExpired
        with mock.patch('subprocess.run', side_effect=subprocess.TimeoutExpired("git", 300)):
            with pytest.raises(RepositorySetupError) as exc_info:
                self.repo_manager.setup_repository(target_dir)
            
            # Verify error message mentions timeout
            error_message = str(exc_info.value)
            assert "timed out" in error_message.lower()
            assert "5 minutes" in error_message
    
    def test_git_not_installed(self):
        """Test error when git command is not found.
        
        **Validates: Requirement 1.2**
        """
        target_dir = os.path.join(self.temp_dir, "git_not_found_test")
        
        # Mock subprocess.run to raise FileNotFoundError
        with mock.patch('subprocess.run', side_effect=FileNotFoundError()):
            with pytest.raises(RepositorySetupError) as exc_info:
                self.repo_manager.setup_repository(target_dir)
            
            # Verify error message mentions git installation
            error_message = str(exc_info.value)
            assert "Git command not found" in error_message
            assert "Git is installed" in error_message
    
    def test_parent_directory_creation(self):
        """Test that parent directories are created if they don't exist.
        
        **Validates: Requirement 1.1**
        """
        # Create a deeply nested target path
        target_dir = os.path.join(self.temp_dir, "level1", "level2", "isaaclab")
        
        # Mock subprocess.run to simulate successful clone
        mock_result = mock.MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        
        with mock.patch('subprocess.run', return_value=mock_result):
            result = self.repo_manager.setup_repository(target_dir)
            
            # Verify parent directories were created
            assert result.success is True
            assert os.path.exists(os.path.join(self.temp_dir, "level1", "level2"))
    
    def test_unexpected_error_handling(self):
        """Test handling of unexpected errors during setup.
        
        **Validates: Requirement 1.2**
        """
        target_dir = os.path.join(self.temp_dir, "unexpected_error_test")
        
        # Mock subprocess.run to raise unexpected exception
        with mock.patch('subprocess.run', side_effect=RuntimeError("Unexpected error")):
            with pytest.raises(RepositorySetupError) as exc_info:
                self.repo_manager.setup_repository(target_dir)
            
            # Verify error message mentions unexpected error
            error_message = str(exc_info.value)
            assert "Unexpected error" in error_message
