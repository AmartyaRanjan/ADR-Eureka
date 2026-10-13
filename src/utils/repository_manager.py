"""Repository Manager for cloning and setting up IsaacLabEurekai repository.

This module handles cloning the IsaacLabEurekai repository from GitHub,
installing dependencies, and verifying the installation.

**Validates: Requirements 1.1, 1.2, 1.3**
"""

import os
import subprocess
from pathlib import Path
from typing import Optional

from src.exceptions import RepositorySetupError
from src.utils.models import SetupResult


class RepositoryManager:
    """Manages repository cloning and setup for IsaacLabEurekai.
    
    This class handles the complete setup process for the IsaacLabEurekai
    repository, including cloning from GitHub, checking for directory conflicts,
    and handling network failures with descriptive error messages.
    """
    
    REPOSITORY_URL = "https://github.com/NVlabs/IsaacLabEureka"
    
    def __init__(self):
        """Initialize the RepositoryManager."""
        pass
    
    def setup_repository(self, target_dir: str) -> SetupResult:
        """
        Clone and setup IsaacLabEurekai repository.
        
        This method performs the following operations:
        1. Check for directory conflicts before cloning
        2. Clone the repository from GitHub
        3. Handle network failures with descriptive error messages
        
        Args:
            target_dir: Directory to clone repository into
            
        Returns:
            SetupResult with status and error message if failed
            
        Raises:
            RepositorySetupError: If clone or setup fails
            
        **Validates: Requirements 1.1, 1.2, 1.3**
        """
        try:
            # Convert target_dir to Path for easier manipulation
            target_path = Path(target_dir).resolve()
            
            # Check if directory already exists (Requirement 1.3)
            if target_path.exists():
                error_msg = (
                    f"Directory conflict: '{target_path}' already exists. "
                    f"Please remove it or choose a different target directory."
                )
                raise RepositorySetupError(error_msg)
            
            # Ensure parent directory exists
            parent_dir = target_path.parent
            if not parent_dir.exists():
                parent_dir.mkdir(parents=True, exist_ok=True)
            
            # Clone the repository (Requirement 1.1)
            try:
                result = subprocess.run(
                    ["git", "clone", self.REPOSITORY_URL, str(target_path)],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout for cloning
                )
                
                if result.returncode != 0:
                    # Extract error message from stderr
                    error_output = result.stderr.strip()
                    
                    # Check for common network-related errors (Requirement 1.2)
                    if any(keyword in error_output.lower() for keyword in 
                           ['network', 'connection', 'timeout', 'could not resolve host', 
                            'failed to connect', 'ssl', 'certificate']):
                        error_msg = (
                            f"Network failure while cloning repository: {error_output}. "
                            f"Please check your internet connection and try again."
                        )
                    else:
                        error_msg = f"Failed to clone repository: {error_output}"
                    
                    raise RepositorySetupError(error_msg)
                
            except subprocess.TimeoutExpired:
                error_msg = (
                    "Repository clone operation timed out after 5 minutes. "
                    "This may indicate a slow network connection. Please try again."
                )
                raise RepositorySetupError(error_msg)
            
            except FileNotFoundError:
                error_msg = (
                    "Git command not found. Please ensure Git is installed and "
                    "available in your system PATH."
                )
                raise RepositorySetupError(error_msg)
            
            # Return success result
            return SetupResult(
                success=True,
                error_message=None,
                repository_path=str(target_path)
            )
        
        except RepositorySetupError:
            # Re-raise RepositorySetupError as-is
            raise
        
        except Exception as e:
            # Wrap unexpected errors
            error_msg = f"Unexpected error during repository setup: {str(e)}"
            raise RepositorySetupError(error_msg) from e
