"""Training data model classes.

This module defines data classes for training metrics, evaluation results,
checkpoint data, and training summaries.

**Validates: Requirements 10.1, 10.2, 10.3**
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class TrainingMetrics:
    """Metrics from one training iteration.
    
    Contains performance metrics collected during a training iteration,
    including rewards, success rates, and loss values.
    
    Attributes:
        iteration: Training iteration number
        episode_reward_mean: Mean episode reward across episodes
        episode_reward_std: Standard deviation of episode rewards
        success_rate: Fraction of successful episodes (0.0-1.0)
        policy_loss: Policy network loss value
        value_loss: Value network loss value
        learning_rate: Current learning rate
        timestamp: ISO 8601 timestamp with millisecond precision
    """
    iteration: int
    episode_reward_mean: float
    episode_reward_std: float
    success_rate: float
    policy_loss: float
    value_loss: float
    learning_rate: float
    timestamp: str


@dataclass
class RewardEvaluationResult:
    """Results from evaluating a reward function.
    
    Contains performance metrics from evaluating a generated reward function
    over multiple episodes.
    
    Attributes:
        mean_cumulative_reward: Mean cumulative reward across episodes
        std_cumulative_reward: Standard deviation of cumulative rewards
        success_rate: Fraction of successful episodes (0.0-1.0)
        num_crashes: Number of episodes that crashed or raised exceptions
        is_valid: Whether the reward function is valid for use
    """
    mean_cumulative_reward: float
    std_cumulative_reward: float
    success_rate: float
    num_crashes: int
    is_valid: bool


@dataclass
class EurekaTrainingResult:
    """Results from complete Eureka training.
    
    Contains the final results of an Eureka training run, including
    the best reward function discovered and overall performance.
    
    Attributes:
        best_reward_code: Source code of the best reward function
        best_reward_performance: Performance metric (mean cumulative reward)
        best_policy_path: File path to saved best policy weights
        total_iterations: Total number of Eureka iterations completed
        successful_generations: Number of successfully generated valid reward functions
        training_duration_seconds: Total training duration in seconds
    """
    best_reward_code: str
    best_reward_performance: float
    best_policy_path: str
    total_iterations: int
    successful_generations: int
    training_duration_seconds: float


@dataclass
class CheckpointData:
    """Data loaded from a training checkpoint.
    
    Contains all state information needed to resume training from a checkpoint.
    
    Attributes:
        version: Checkpoint format version
        iteration: Training iteration number at checkpoint
        robot_type: Robot configuration ('leap_hand' or 'arm_system')
        training_mode: Training mode ('ppo' or 'eureka')
        policy_state: Policy network state dictionary
        optimizer_state: Optimizer state dictionary
        training_stats: Training statistics dictionary
        reward_function_code: Reward function code (for Eureka mode, None for PPO)
    """
    version: str
    iteration: int
    robot_type: str
    training_mode: str
    policy_state: Dict[str, Any]
    optimizer_state: Dict[str, Any]
    training_stats: Dict[str, Any]
    reward_function_code: Optional[str] = None


@dataclass
class TrainingSummary:
    """Final training summary report.
    
    Contains aggregated metrics and results from a complete training run.
    
    Attributes:
        robot_type: Robot configuration used ('leap_hand' or 'arm_system')
        training_mode: Training mode used ('ppo' or 'eureka')
        total_iterations: Total number of training iterations completed
        final_mean_reward: Final mean episode reward
        final_success_rate: Final success rate (0.0-1.0)
        training_duration_seconds: Total training duration in seconds
        checkpoints_saved: Number of checkpoints saved during training
        best_reward_code: Best reward function code (for Eureka mode, None for PPO)
    """
    robot_type: str
    training_mode: str
    total_iterations: int
    final_mean_reward: float
    final_success_rate: float
    training_duration_seconds: float
    checkpoints_saved: int
    best_reward_code: Optional[str] = None
