"""Utility data model classes.

This module defines utility data classes used across the system
for setup results and scene data.

**Validates: Requirements 10.1, 10.2, 10.3**
"""

from dataclasses import dataclass
from typing import Optional, Any, Dict


@dataclass
class SetupResult:
    """Result from repository setup operation.
    
    Contains the status and any error information from setting up
    the IsaacLabEurekai repository.
    
    Attributes:
        success: Whether the setup was successful
        error_message: Error message if setup failed, None if successful
        repository_path: Path to the cloned repository (if successful)
    """
    success: bool
    error_message: Optional[str] = None
    repository_path: Optional[str] = None


@dataclass
class SceneData:
    """Scene state data for visualization.
    
    Contains the current state of the simulation scene for rendering.
    
    Attributes:
        robot_state: Dictionary containing robot joint positions and other state
        cube_state: Dictionary containing cube position, orientation, and other state
        environment_state: Dictionary containing environment and other object states
        timestamp: Timestamp of the scene state
    """
    robot_state: Dict[str, Any]
    cube_state: Dict[str, Any]
    environment_state: Dict[str, Any]
    timestamp: float
