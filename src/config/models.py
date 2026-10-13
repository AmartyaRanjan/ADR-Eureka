"""Configuration data model classes.

This module defines all configuration data classes using Python dataclasses.
These models represent environment configuration, training hyperparameters,
and system settings.

**Validates: Requirements 10.1, 10.2, 10.3**
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EnvConfig:
    """Environment configuration from .env file.
    
    Contains API credentials and LLM model specification.
    
    Attributes:
        gemini_api_key: Gemini API authentication key
        llm_model: Language model identifier (default: gemini-1.5-pro)
    """
    gemini_api_key: str
    llm_model: str = "gemini-1.5-pro"


@dataclass
class PPOConfig:
    """PPO (Proximal Policy Optimization) hyperparameters.
    
    Defines all parameters for the PPO reinforcement learning algorithm.
    
    Attributes:
        batch_size: Number of samples per training batch (64-4096)
        num_epochs: Number of policy update epochs per iteration (3-15)
        learning_rate: Optimizer learning rate
        gamma: Discount factor for future rewards
        gae_lambda: GAE (Generalized Advantage Estimation) lambda parameter
        clip_epsilon: PPO clipping parameter for policy updates
    """
    batch_size: int = 2048
    num_epochs: int = 10
    learning_rate: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    max_training_iterations: int = 500
    num_envs: int = 32
    rollout_horizon: int = 16


@dataclass
class EurekaConfig:
    """Eureka reward generation configuration.
    
    Parameters controlling the Eureka automated reward function generation process.
    
    Attributes:
        num_iterations: Number of Eureka iterations to run (1-50)
        evaluation_episodes: Episodes to evaluate each reward function (default: 100)
        improvement_threshold: Minimum improvement to update best reward (default: 5%)
        generation_timeout: Timeout for reward generation API calls (seconds)
    """
    num_iterations: int = 10
    evaluation_episodes: int = 100
    improvement_threshold: float = 0.05
    generation_timeout: int = 60


@dataclass
class TaskConfig:
    """Rubik's cube rotation task configuration.
    
    Parameters defining the manipulation task setup and success criteria.
    
    Attributes:
        max_episode_steps: Maximum timesteps per episode
        rotation_tolerance: Tolerance for successful rotation (degrees)
        cube_distance_min: Minimum cube distance from robot base (meters)
        cube_distance_max: Maximum cube distance from robot base (meters)
        drop_threshold: Distance threshold for detecting dropped cube (meters)
    """
    max_episode_steps: int = 500
    rotation_tolerance: float = 5.0
    cube_distance_min: float = 0.3
    cube_distance_max: float = 0.5
    drop_threshold: float = 0.5


@dataclass
class VisualizationConfig:
    """Visualization system configuration.
    
    Settings for rendering training progress to the display.
    
    Attributes:
        enabled: Whether visualization is enabled
        fps: Target frames per second for rendering (minimum 30)
        update_frequency: Frequency for metrics updates (Hz, minimum 1)
    """
    enabled: bool = True
    fps: int = 30
    update_frequency: int = 1


@dataclass
class CheckpointConfig:
    """Checkpoint management configuration.
    
    Settings for saving and managing training checkpoints.
    
    Attributes:
        enabled: Whether checkpointing is enabled
        interval: Checkpoint save interval (training iterations)
        save_timeout: Maximum time for checkpoint save operation (seconds)
        max_checkpoints: Maximum number of checkpoints to keep
    """
    enabled: bool = True
    interval: int = 1000
    save_timeout: int = 10
    max_checkpoints: int = 5


@dataclass
class LoggingConfig:
    """Logging system configuration.
    
    Settings for training metrics logging and export.
    
    Attributes:
        output_dir: Directory for log files
        csv_export: Whether to support CSV export
        precision: Timestamp precision ('millisecond' or 'second')
    """
    output_dir: str = "./logs"
    csv_export: bool = True
    precision: str = "millisecond"


@dataclass
class TrainingConfig:
    """Complete training system configuration.
    
    Top-level configuration combining all subsystem configurations.
    
    Attributes:
        robot_type: Robot configuration ('leap_hand' or 'arm_system')
        training_mode: Training mode ('ppo' or 'eureka')
        ppo: PPO hyperparameters
        eureka: Eureka configuration
        task: Task configuration
        visualization: Visualization settings
        checkpoint: Checkpoint settings
        logging: Logging settings
    """
    robot_type: str
    training_mode: str
    ppo: PPOConfig = field(default_factory=PPOConfig)
    eureka: EurekaConfig = field(default_factory=EurekaConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
