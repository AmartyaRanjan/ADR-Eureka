# Implementation Plan: LEAP Hand Eureka Training System

## Overview

This implementation plan breaks down the LEAP Hand Eureka Training System into discrete coding tasks. The system enables training of robotic systems (LEAP hand and 2DOF arm with LEAP hand) to perform Rubik's cube rotation tasks using PPO or Eureka-based reward generation with Gemini API integration.

## Tasks

- [x] 1. Set up project structure and core interfaces
  - Create directory structure: `src/`, `src/config/`, `src/training/`, `src/eureka/`, `src/utils/`, `tests/`
  - Define core exception classes: `RepositorySetupError`, `ConfigurationError`, `GeminiAPIError`, `GeminiTimeoutError`, `InitializationError`, `CheckpointError`, `SetupError`
  - Create data model classes: `EnvConfig`, `PPOConfig`, `EurekaConfig`, `TaskConfig`, `VisualizationConfig`, `CheckpointConfig`, `LoggingConfig`, `TrainingConfig`
  - Create result data classes: `TrainingMetrics`, `RewardEvaluationResult`, `EurekaTrainingResult`, `CheckpointData`, `TrainingSummary`, `SetupResult`, `SceneData`
  - _Requirements: 10.1, 10.2, 10.3_

- [ ]* 1.1 Write unit tests for data model classes
  - Test data class initialization and validation
  - Test default values for configuration classes
  - Test serialization/deserialization of result classes
  - _Requirements: 10.1, 10.2_

- [-] 2. Implement Repository Manager
  - [-] 2.1 Create RepositoryManager class with setup_repository method
    - Implement repository cloning from https://github.com/NVlabs/IsaacLabEurekai
    - Check for directory conflicts before cloning
    - Handle network failures with descriptive error messages
    - _Requirements: 1.1, 1.2, 1.3_

  - [~] 2.2 Implement dependency installation in setup_repository
    - Parse repository configuration for dependencies
    - Install dependencies using appropriate package manager
    - Report specific dependency that failed if installation fails
    - _Requirements: 1.4, 1.5_

  - [~] 2.3 Implement verify_installation method
    - Verify Isaac Lab Python package is importable
    - Verify Eureka configuration directory exists
    - Return descriptive error for missing components
    - _Requirements: 1.6, 1.7, 1.8_

  - [ ]* 2.4 Write integration tests for Repository Manager
    - Test successful repository setup with mock git operations
    - Test directory conflict handling
    - Test network failure scenarios
    - Test dependency installation failure handling
    - Test verification of installed components
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8_

- [-] 3. Implement Configuration Manager
  - [ ] 3.1 Create ConfigurationManager class with load_env_config method
    - Load .env file and parse GEMINI_API_KEY and LLM_MODEL fields
    - Raise ConfigurationError if file is missing or unreadable
    - Validate GEMINI_API_KEY is present and non-empty
    - Return EnvConfig object with credentials
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7_

  - [~] 3.2 Implement create_default_env_file method
    - Create .env file with GEMINI_API_KEY and LLM_MODEL placeholder fields
    - Use default value "gemini-1.5-pro" for LLM_MODEL
    - _Requirements: 2.1, 2.2, 2.3_

  - [~] 3.3 Implement load_training_config method
    - Support YAML and JSON configuration file formats
    - Parse all configuration sections: robot_type, training_mode, ppo, eureka, task, visualization, checkpoint, logging
    - Return TrainingConfig object with all parameters
    - Raise ConfigurationError for missing required parameters
    - _Requirements: 10.1, 10.2, 10.3_

  - [~] 3.4 Implement validate_config method
    - Validate robot_type is "leap_hand" or "arm_system"
    - Validate training_mode is "ppo" or "eureka"
    - Validate PPO hyperparameters are within acceptable ranges (batch_size: 64-4096, num_epochs: 3-15)
    - Validate Eureka parameters (num_iterations: 1-50, evaluation_episodes > 0)
    - Return list of validation error messages
    - _Requirements: 10.2, 10.5_

  - [~] 3.5 Implement command-line argument override support
    - Parse command-line arguments
    - Override configuration file settings with command-line values
    - _Requirements: 10.4_

  - [ ]* 3.6 Write unit tests for Configuration Manager
    - Test valid .env configuration loading
    - Test missing API key validation
    - Test default .env file creation
    - Test YAML and JSON training config loading
    - Test configuration validation for valid and invalid inputs
    - Test command-line override functionality
    - Test missing required parameter detection
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 10.1, 10.2, 10.3, 10.4, 10.5_

- [~] 4. Checkpoint - Ensure core infrastructure is working
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 5. Implement Gemini Integration Layer
  - [~] 5.1 Create GeminiClient class with initialization
    - Accept api_key, model, and timeout parameters
    - Configure Gemini API client with authentication
    - Set default timeout to 30 seconds
    - _Requirements: 3.1, 3.2_

  - [~] 5.2 Implement generate_reward_function method
    - Format prompt with context (task description, state variables, success criteria)
    - Send request to Gemini API with configured timeout
    - Parse API response to extract Python code
    - Implement timeout handling (30 seconds)
    - Raise GeminiTimeoutError if timeout exceeded
    - _Requirements: 3.1, 3.2, 3.4, 3.5, 3.6_

  - [~] 5.3 Implement _retry_with_backoff method
    - Implement exponential backoff: initial 1s, multiplier 2x, max 8s
    - Retry up to 3 times on network errors
    - Log errors on each retry attempt
    - Raise exception after all retries exhausted
    - _Requirements: 3.7, 3.8_

  - [ ]* 5.4 Write unit tests for Gemini Integration Layer (with mocked API)
    - Test successful API call and response parsing
    - Test timeout handling
    - Test retry logic with exponential backoff
    - Test retry exhaustion after 3 attempts
    - Test invalid response format handling
    - _Requirements: 3.1, 3.2, 3.4, 3.5, 3.6, 3.7, 3.8_

- [ ] 6. Implement PPO Training Engine
  - [~] 6.1 Create PPOTrainer class for LEAP Hand
    - Initialize with TrainingConfig and robot_type="leap_hand"
    - Store configuration parameters (batch_size, num_epochs, learning_rate, gamma, etc.)
    - _Requirements: 5.1, 6.1_

  - [~] 6.2 Implement initialize_environment for LEAP Hand
    - Initialize LEAP Hand model in IsaacLab simulation
    - Spawn Rubik's cube at distance 0.3-0.5m from robot base
    - Verify LEAP Hand joints are controllable
    - Raise InitializationError if setup fails
    - _Requirements: 5.2, 5.3, 7.1, 7.3_

  - [~] 6.3 Implement observation and action collection for LEAP Hand
    - Collect observations: joint positions (16D), joint velocities (16D), fingertip forces (5D), cube orientation (4D quaternion), cube position (3D)
    - Apply actions to LEAP Hand joint actuators (16D continuous)
    - _Requirements: 5.4, 5.5, 7.4_

  - [~] 6.4 Implement train_iteration method for LEAP Hand
    - Execute one training iteration with PPO algorithm
    - Use configured batch_size (64-4096) and num_epochs (3-15)
    - Compute policy and value losses
    - Return TrainingMetrics with rewards, success rate, losses
    - Support optional custom reward function parameter
    - _Requirements: 5.4, 5.5, 5.6, 5.7_

  - [~] 6.5 Implement episode termination logic
    - Mark episode successful when cube face rotates 90° ± 5°
    - Terminate episode at 1000 timesteps
    - Terminate episode if cube moves > 0.5m from initial position (dropped)
    - _Requirements: 7.2, 7.6, 7.7, 7.8_

  - [~] 6.6 Implement reward computation and logging
    - Calculate progress as angular displacement toward 90° rotation
    - Log episode reward, success rate, policy loss to training log
    - Handle logging failures gracefully (continue training)
    - _Requirements: 5.10, 5.11, 7.5_

  - [ ]* 6.7 Write unit tests for LEAP Hand PPO training
    - Test environment initialization
    - Test observation space dimensions (16+16+5+4+3 = 44D)
    - Test action space dimensions (16D)
    - Test episode termination conditions
    - Test reward computation logic
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8_

- [ ] 7. Implement PPO Training Engine for Arm System
  - [~] 7.1 Extend PPOTrainer for robot_type="arm_system"
    - Initialize with TrainingConfig and robot_type="arm_system"
    - _Requirements: 6.1_

  - [~] 7.2 Implement initialize_environment for Arm System
    - Initialize 2DOF arm and LEAP Hand in IsaacLab simulation
    - Spawn Rubik's cube at distance 0.3-0.5m from robot base
    - Verify both arm joints and LEAP Hand joints are controllable
    - Raise InitializationError if setup fails
    - _Requirements: 6.2, 6.3, 6.11_

  - [~] 7.3 Implement observation and action collection for Arm System
    - Collect observations: arm joint positions (2D), arm joint velocities (2D), LEAP hand joint positions (16D), LEAP hand joint velocities (16D), fingertip forces (5D), cube orientation (4D), cube position (3D)
    - Apply actions to arm and LEAP Hand joint actuators (18D continuous)
    - _Requirements: 6.4, 6.5_

  - [~] 7.4 Implement train_iteration for Arm System
    - Execute one training iteration with PPO algorithm
    - Use configured batch_size (64-4096) and num_epochs (3-15)
    - Compute policy and value losses
    - Return TrainingMetrics with rewards, success rate, losses
    - Support optional custom reward function parameter
    - _Requirements: 6.4, 6.5, 6.6, 6.7_

  - [~] 7.5 Implement reward computation and logging for Arm System
    - Calculate progress as angular displacement toward 90° rotation
    - Log episode reward, success rate, policy loss to training log
    - Handle logging failures gracefully (continue training)
    - _Requirements: 6.10_

  - [ ]* 7.6 Write unit tests for Arm System PPO training
    - Test environment initialization with 2DOF arm and LEAP hand
    - Test observation space dimensions (2+2+16+16+5+4+3 = 48D)
    - Test action space dimensions (18D)
    - Test joint controllability verification
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.11_

- [ ] 8. Implement policy save/load functionality
  - [~] 8.1 Implement save_policy method
    - Save policy weights to disk in standard checkpoint format
    - Retry save operation once if it fails
    - Log error if both save attempts fail
    - Return True if successful, False otherwise
    - _Requirements: 5.8, 5.9, 6.8, 6.9_

  - [~] 8.2 Implement load_policy method
    - Load policy weights from disk
    - Raise CheckpointError if checkpoint is invalid or corrupted
    - _Requirements: 5.8, 6.8_

  - [~] 8.3 Implement evaluate_policy method
    - Run policy for specified number of episodes
    - Support optional custom reward function
    - Return mean cumulative reward over episodes
    - _Requirements: 5.4, 5.5, 6.4, 6.5_

  - [ ]* 8.4 Write unit tests for policy save/load
    - Test successful policy save and load
    - Test save retry logic
    - Test corrupted checkpoint detection
    - Test evaluate_policy with different reward functions
    - _Requirements: 5.8, 5.9, 6.8, 6.9_

- [~] 9. Checkpoint - Ensure PPO training engine is working
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 10. Implement Eureka Reward Generator
  - [~] 10.1 Create EurekaModule class with initialization
    - Accept GeminiClient, PPOTrainer, and EurekaConfig
    - Store configuration (num_iterations, evaluation_episodes, improvement_threshold, generation_timeout)
    - _Requirements: 8.1, 8.9, 9.1, 9.10_

  - [~] 10.2 Implement generate_reward_function method
    - Build context with task description, state variables, success criteria
    - Include performance feedback from previous iterations
    - Call Gemini API with 60-second timeout
    - Parse response to extract Python code
    - Return None if generation fails or times out
    - Log timeout error if timeout exceeded
    - _Requirements: 8.1, 8.2, 9.1, 9.2_

  - [~] 10.3 Implement validate_reward_function method
    - Parse Python code string
    - Check for required function signature: reward_function(observations, actions, next_observations, cube_rotation, cube_dropped, timestep)
    - Compile code and catch syntax errors
    - Verify function accepts correct arguments
    - Return (is_valid, compiled_function) tuple
    - Log invalid code error if validation fails
    - _Requirements: 8.3, 8.4, 9.3, 9.4_

  - [~] 10.4 Implement evaluate_reward_function method for LEAP Hand
    - Run PPO training for specified number of episodes (default 100)
    - Use provided reward function for training
    - Catch runtime exceptions during evaluation
    - Compute mean cumulative reward over episodes
    - Return RewardEvaluationResult with performance metrics
    - Mark as invalid if episodes crash or raise exceptions
    - _Requirements: 8.4, 8.5, 8.6, 8.7_

  - [~] 10.5 Extend evaluate_reward_function for Arm System
    - Run PPO training for 100 episodes with Arm System
    - Include both arm and LEAP Hand state variables in reward computation context
    - Catch runtime exceptions during evaluation
    - Compute mean cumulative reward over episodes
    - Return RewardEvaluationResult with performance metrics
    - _Requirements: 9.5, 9.6, 9.7_

  - [~] 10.6 Implement run_eureka_training method
    - Iterate for configured number of iterations (1-50, default 10)
    - Generate reward function at each iteration
    - Validate and evaluate each generated function
    - Compare performance against current best (threshold: 5% improvement)
    - Update active reward function when improvement >= 5%
    - Track best reward function code and performance
    - Return EurekaTrainingResult with best reward and policy
    - _Requirements: 8.8, 8.9, 8.10, 9.8, 9.9, 9.10, 9.11_

  - [ ]* 10.7 Write unit tests for Eureka Reward Generator
    - Test reward function generation with mocked Gemini API
    - Test reward function validation (valid and invalid syntax)
    - Test function signature verification
    - Test reward function evaluation
    - Test performance comparison and selection logic (5% threshold)
    - Test full Eureka training loop
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 8.10, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 9.10, 9.11_

- [ ] 11. Implement Visualization System
  - [~] 11.1 Create VisualizationSystem class with initialization
    - Accept VisualizationConfig
    - Store configuration (enabled, fps, update_frequency)
    - _Requirements: 4.2_

  - [~] 11.2 Implement initialize_renderer method
    - Initialize display renderer for laptop display
    - Return True if successful, False if display unavailable
    - Continue training without rendering if display unavailable
    - _Requirements: 4.1, 4.7_

  - [~] 11.3 Implement render_frame method
    - Render robot model, Rubik's cube, and environment
    - Update display at minimum 30 FPS when enabled
    - Handle rendering failures gracefully (continue training, log error)
    - _Requirements: 4.1, 4.2, 4.3, 4.8_

  - [~] 11.4 Implement update_metrics method
    - Display episode reward (numeric) and success rate (percentage)
    - Update metrics overlay at minimum 1 Hz
    - _Requirements: 4.5, 4.6_

  - [~] 11.5 Implement toggle_rendering method
    - Enable or disable rendering within 100ms
    - _Requirements: 4.4_

  - [ ]* 11.6 Write unit tests for Visualization System
    - Test renderer initialization with and without display
    - Test frame rendering at target framerate
    - Test metrics overlay update frequency
    - Test toggle response time
    - Test graceful fallback when display unavailable
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8_

- [ ] 12. Implement Checkpoint Manager
  - [~] 12.1 Create CheckpointManager class with initialization
    - Accept CheckpointConfig
    - Store configuration (enabled, interval, save_timeout, max_checkpoints)
    - _Requirements: 11.1_

  - [~] 12.2 Implement save_checkpoint method
    - Save policy_state, optimizer_state, training_stats to checkpoint file
    - Include metadata: version, timestamp, iteration, robot_type, training_mode
    - Complete save within 10 seconds (configured timeout)
    - Keep last 5 checkpoints (configurable)
    - Retry once if save fails
    - Return True if successful, False otherwise
    - _Requirements: 11.2, 11.3_

  - [~] 12.3 Implement verify_checkpoint method
    - Verify checkpoint file exists and is readable
    - Validate checkpoint format and required fields
    - Return True if valid, False otherwise
    - _Requirements: 11.6_

  - [~] 12.4 Implement load_checkpoint method
    - Load checkpoint from disk
    - Parse checkpoint data into CheckpointData object
    - Verify checkpoint integrity before loading
    - Raise CheckpointError if checkpoint is corrupted or invalid
    - _Requirements: 11.4, 11.6_

  - [ ]* 12.5 Write unit tests for Checkpoint Manager
    - Test checkpoint save with all metadata
    - Test checkpoint save timeout handling
    - Test checkpoint save retry logic
    - Test checkpoint verification (valid and corrupted)
    - Test checkpoint loading and parsing
    - Test max_checkpoints cleanup
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.6_

- [ ] 13. Implement Logging System
  - [~] 13.1 Create LoggingSystem class with initialization
    - Accept LoggingConfig
    - Store configuration (output_dir, csv_export, precision)
    - Create output directory if it doesn't exist
    - _Requirements: 12.1_

  - [~] 13.2 Implement log_training_iteration method
    - Record episode rewards, success rates, policy loss, value loss, learning rate
    - Timestamp all log entries with millisecond precision
    - Write logs to structured log files (JSON format)
    - _Requirements: 12.2, 12.6_

  - [~] 13.3 Implement log_eureka_generation method
    - Record generated reward function code
    - Record evaluation performance (mean cumulative reward)
    - Record whether function is new best
    - _Requirements: 12.3_

  - [~] 13.4 Implement export_csv method
    - Export training logs to CSV format
    - Include all metrics: iteration, rewards, success rate, losses, learning rate
    - _Requirements: 12.4_

  - [~] 13.5 Implement generate_summary method
    - Generate TrainingSummary with final performance metrics
    - Include training duration, final mean reward, final success rate
    - Include best reward function code if applicable
    - _Requirements: 12.5_

  - [ ]* 13.6 Write unit tests for Logging System
    - Test log directory creation
    - Test training iteration logging with correct timestamps
    - Test Eureka generation logging
    - Test CSV export functionality
    - Test summary report generation
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6_

- [~] 14. Checkpoint - Ensure support systems are working
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 15. Implement Training Manager
  - [~] 15.1 Create TrainingManager class with initialization
    - Accept TrainingConfig
    - Initialize all component references (ConfigurationManager, RepositoryManager, PPOTrainer, etc.)
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

  - [~] 15.2 Implement setup method
    - Initialize Configuration Manager and load configs
    - Verify Repository Manager installation
    - Initialize PPOTrainer with correct robot_type
    - Initialize Visualization System if enabled
    - Initialize Checkpoint Manager if enabled
    - Initialize Logging System
    - Raise SetupError if any component fails
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

  - [~] 15.3 Implement run_ppo_training method
    - Execute standard PPO training loop
    - Call train_iteration repeatedly
    - Save checkpoints at configured intervals
    - Update visualization if enabled
    - Log metrics after each iteration
    - Handle interruptions gracefully
    - _Requirements: 5.1, 5.4, 5.5, 5.6, 5.7, 5.10, 6.1, 6.4, 6.5, 6.6, 6.7, 6.10_

  - [~] 15.4 Implement run_eureka_training method
    - Initialize GeminiClient with API credentials
    - Initialize EurekaModule with GeminiClient and PPOTrainer
    - Execute Eureka training loop via run_eureka_training
    - Save best reward function and policy after completion
    - Log Eureka generation attempts and results
    - Save checkpoints at configured intervals
    - Update visualization if enabled
    - _Requirements: 8.1, 8.9, 8.10, 9.1, 9.10, 9.11_

  - [~] 15.5 Implement resume_from_checkpoint method
    - Load checkpoint using CheckpointManager
    - Restore policy weights to PPOTrainer
    - Restore optimizer state
    - Resume training from saved iteration counter
    - _Requirements: 11.4, 11.5_

  - [ ]* 15.6 Write integration tests for Training Manager
    - Test setup method with valid configuration
    - Test setup method with invalid configuration (expect SetupError)
    - Test PPO training workflow (LEAP Hand and Arm System)
    - Test Eureka training workflow (LEAP Hand and Arm System)
    - Test checkpoint resumption
    - Test graceful handling of component failures
    - _Requirements: 5.1, 5.4, 5.5, 5.6, 5.7, 6.1, 6.4, 6.5, 6.6, 6.7, 8.1, 8.9, 8.10, 9.1, 9.10, 9.11, 10.1, 10.2, 10.3, 10.4, 10.5, 11.4, 11.5_

- [ ] 16. Implement Command-Line Interface
  - [~] 16.1 Create CLI entry point script
    - Parse command-line arguments: config file path, robot type, training mode, checkpoint path
    - Support --resume-from-checkpoint flag
    - Support configuration overrides via command-line flags
    - _Requirements: 10.4_

  - [~] 16.2 Implement main execution flow
    - Create default .env file if it doesn't exist (prompt user for API key)
    - Load configuration from file and apply CLI overrides
    - Initialize TrainingManager with configuration
    - Call setup method
    - Call run_ppo_training or run_eureka_training based on training_mode
    - Handle errors and display user-friendly messages
    - _Requirements: 2.1, 10.1, 10.2, 10.3, 10.4, 10.5_

  - [ ]* 16.3 Write end-to-end smoke tests for CLI
    - Test CLI with default configuration
    - Test CLI with custom configuration file
    - Test CLI with command-line overrides
    - Test CLI with checkpoint resumption
    - Test CLI error handling (missing config, invalid API key, etc.)
    - _Requirements: 2.1, 10.1, 10.2, 10.3, 10.4, 10.5_

- [ ] 17. Final Integration and Testing
  - [~] 17.1 Create example configuration files
    - Create example YAML config for LEAP Hand PPO training
    - Create example YAML config for LEAP Hand Eureka training
    - Create example YAML config for Arm System PPO training
    - Create example YAML config for Arm System Eureka training
    - Include comments explaining each parameter
    - _Requirements: 10.1, 10.2, 10.3_

  - [~] 17.2 Create README documentation
    - Document installation steps (repository cloning, dependency installation)
    - Document configuration setup (.env file and training config)
    - Document how to run PPO and Eureka training
    - Document checkpoint resumption workflow
    - Include troubleshooting section for common errors
    - _Requirements: 1.1, 2.1, 10.1, 10.2, 10.3, 10.4_

  - [ ]* 17.3 Run end-to-end integration tests
    - Test complete LEAP Hand PPO training workflow
    - Test complete LEAP Hand Eureka training workflow
    - Test complete Arm System PPO training workflow
    - Test complete Arm System Eureka training workflow
    - Test checkpoint save and resume workflow
    - Test visualization enable/disable
    - Test error recovery scenarios
    - _Requirements: All requirements_

- [~] 18. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation of major components
- Integration tests validate component interactions with mocked external dependencies (Gemini API, IsaacLab simulation)
- The system uses Python with type hints and dataclasses for clean, maintainable code
- All external dependencies (Gemini API, IsaacLab simulation, file system) should be mocked in unit tests
- Focus on infrastructure setup, configuration management, and orchestration logic
- The design explicitly states that property-based testing is not appropriate for this infrastructure-heavy system

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["2.1", "3.1", "3.2"] },
    { "id": 2, "tasks": ["2.2", "3.3"] },
    { "id": 3, "tasks": ["2.3", "2.4", "3.4", "3.5", "3.6"] },
    { "id": 4, "tasks": ["5.1"] },
    { "id": 5, "tasks": ["5.2", "5.3"] },
    { "id": 6, "tasks": ["5.4", "6.1"] },
    { "id": 7, "tasks": ["6.2", "7.1"] },
    { "id": 8, "tasks": ["6.3", "7.2"] },
    { "id": 9, "tasks": ["6.4", "7.3"] },
    { "id": 10, "tasks": ["6.5", "7.4"] },
    { "id": 11, "tasks": ["6.6", "7.5", "8.1"] },
    { "id": 12, "tasks": ["6.7", "7.6", "8.2", "8.3"] },
    { "id": 13, "tasks": ["8.4", "10.1"] },
    { "id": 14, "tasks": ["10.2", "11.1", "12.1", "13.1"] },
    { "id": 15, "tasks": ["10.3", "11.2", "12.2", "13.2"] },
    { "id": 16, "tasks": ["10.4", "11.3", "12.3", "13.3"] },
    { "id": 17, "tasks": ["10.5", "11.4", "12.4", "13.4"] },
    { "id": 18, "tasks": ["10.6", "11.5", "12.5", "13.5"] },
    { "id": 19, "tasks": ["10.7", "11.6", "13.6"] },
    { "id": 20, "tasks": ["15.1"] },
    { "id": 21, "tasks": ["15.2"] },
    { "id": 22, "tasks": ["15.3", "15.4", "15.5"] },
    { "id": 23, "tasks": ["15.6", "16.1"] },
    { "id": 24, "tasks": ["16.2"] },
    { "id": 25, "tasks": ["16.3", "17.1", "17.2"] },
    { "id": 26, "tasks": ["17.3"] }
  ]
}
```
