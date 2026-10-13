"""Core exception classes for the LEAP Hand Eureka Training System.

This module defines all custom exceptions used throughout the system,
providing clear error categorization for setup, configuration, API,
initialization, and checkpoint operations.
"""


class RepositorySetupError(Exception):
    """Raised when repository cloning or setup operations fail.
    
    This includes failures in:
    - Cloning the IsaacLabEurekai repository
    - Directory conflicts
    - Dependency installation failures
    - Installation verification failures
    """
    pass


class ConfigurationError(Exception):
    """Raised when configuration loading or validation fails.
    
    This includes failures in:
    - Missing or unreadable configuration files
    - Invalid configuration format
    - Missing required configuration fields
    - Invalid configuration values
    """
    pass


class GeminiAPIError(Exception):
    """Raised when Gemini API calls fail.
    
    This includes failures in:
    - Network errors (after retries exhausted)
    - Invalid API responses
    - Rate limiting errors
    - Authentication failures
    """
    pass


class GeminiTimeoutError(GeminiAPIError):
    """Raised when Gemini API calls exceed the timeout.
    
    This is a specialized GeminiAPIError for timeout-specific cases.
    """
    pass


class InitializationError(Exception):
    """Raised when robot or environment initialization fails.
    
    This includes failures in:
    - Robot model initialization
    - Environment setup
    - Rubik's cube spawning
    - Simulation component verification
    """
    pass


class CheckpointError(Exception):
    """Raised when checkpoint operations fail.
    
    This includes failures in:
    - Checkpoint loading
    - Checkpoint corruption or integrity issues
    - Invalid checkpoint format
    - Missing checkpoint data
    """
    pass


class SetupError(Exception):
    """Raised when overall system setup fails.
    
    This is a general setup error for failures that don't fit
    other specific categories or involve multiple component failures.
    """
    pass
