# Requirements Document

## Introduction

This document specifies the requirements for training robotic systems (LEAP hand and 2DOF arm with LEAP hand) to perform Rubik's cube rotation tasks using the IsaacLabEurekai framework. The system will support both plain PPO (Proximal Policy Optimization) training and Eureka-based automated reward function generation using the Gemini LLM API.

## Glossary

- **Training_System**: The complete software system for training robotic systems on Rubik's cube rotation tasks
- **IsaacLabEurekai**: The simulation and training framework combining IsaacLab with Eureka reward generation
- **LEAP_Hand**: The robotic hand gripper used for manipulation tasks
- **Arm_System**: The 2 degree of freedom robotic arm with attached LEAP hand
- **PPO_Trainer**: The Proximal Policy Optimization reinforcement learning trainer
- **Eureka_Module**: The LLM-based automated reward function generation system
- **Gemini_API**: The Google Gemini large language model API service
- **Visualization_System**: The component that renders training progress on the laptop display
- **Environment_Configuration**: The .env file containing API credentials and configuration settings
- **Repository_Manager**: The component responsible for cloning and setting up the IsaacLabEurekai repository

## Requirements

### Requirement 1: Repository Setup

**User Story:** As a robotics researcher, I want to set up the IsaacLabEurekai repository, so that I have the foundation for training robotic systems.

#### Acceptance Criteria

1. THE Repository_Manager SHALL clone the IsaacLabEurekai repository from https://github.com/NVlabs/IsaacLabEurekai to the local filesystem
2. IF the clone operation fails, THEN THE Repository_Manager SHALL terminate with an error message indicating the failure reason
3. IF a directory with the same name already exists, THEN THE Repository_Manager SHALL terminate with an error message indicating the directory conflict
4. WHEN the repository is cloned successfully, THE Repository_Manager SHALL install all dependencies specified in the repository configuration
5. IF dependency installation fails, THEN THE Repository_Manager SHALL terminate with an error message indicating which dependency failed to install
6. WHEN dependencies are installed successfully, THE Repository_Manager SHALL verify that the Isaac Lab Python package is importable
7. WHEN dependencies are installed successfully, THE Repository_Manager SHALL verify that the Eureka configuration directory exists in the repository structure
8. IF verification fails, THEN THE Repository_Manager SHALL terminate with an error message indicating which component is missing

### Requirement 2: Environment Configuration

**User Story:** As a robotics researcher, I want to configure API credentials securely, so that the Gemini API can be used for reward generation.

#### Acceptance Criteria

1. WHEN the Training_System starts, IF the Environment_Configuration file does not exist, THEN THE Training_System SHALL create an Environment_Configuration file in .env format
2. THE Environment_Configuration file SHALL contain a field named GEMINI_API_KEY for storing the Gemini API key value
3. THE Environment_Configuration file SHALL contain a field named LLM_MODEL for specifying the language model identifier
4. WHEN the Training_System starts, THE Training_System SHALL load credentials from the Environment_Configuration file
5. IF the Environment_Configuration file is missing or unreadable at startup, THEN THE Training_System SHALL terminate with an error message indicating the configuration file could not be accessed
6. WHEN the Training_System loads credentials, THE Training_System SHALL validate that the GEMINI_API_KEY field is present and contains a non-empty string value
7. IF the GEMINI_API_KEY validation fails, THEN THE Training_System SHALL terminate with an error message indicating the API key is missing or invalid before starting Eureka-based training

### Requirement 3: Gemini LLM Integration

**User Story:** As a robotics researcher, I want to use Gemini instead of the default LLM in Eureka, so that I can leverage Gemini's capabilities for reward function generation.

#### Acceptance Criteria

1. THE Eureka_Module SHALL use Gemini API as the LLM backend for reward function generation
2. WHEN Eureka_Module is initialized, THE Eureka_Module SHALL read the Gemini API configuration from the Environment_Configuration file
3. IF the Gemini API configuration is invalid or missing required fields, THEN THE Eureka_Module SHALL terminate with an error message indicating the invalid configuration
4. WHEN generating reward functions, THE Eureka_Module SHALL send prompts to the Gemini API with a timeout of 30 seconds
5. WHEN Gemini API returns a response, THE Eureka_Module SHALL parse the response into executable reward function code
6. IF the Gemini API response cannot be parsed into executable code, THEN THE Eureka_Module SHALL log the parsing failure and mark the generation attempt as failed
7. IF the Gemini API call fails, THEN THE Eureka_Module SHALL log the error and retry up to 3 times with exponential backoff starting at 1 second, multiplying by 2 each retry, with a maximum backoff of 8 seconds
8. IF all retry attempts are exhausted, THEN THE Eureka_Module SHALL terminate the reward generation process and report the failure

### Requirement 4: Visualization Setup

**User Story:** As a robotics researcher, I want to visualize training progress on my laptop screen, so that I can monitor the robot's learning in real-time.

#### Acceptance Criteria

1. THE Visualization_System SHALL render the simulation environment to the laptop display
2. WHEN training begins, THE Visualization_System SHALL update the display at a minimum of 30 frames per second
3. THE Visualization_System SHALL display the robot model, Rubik's cube, and environment
4. WHEN the user issues a toggle command, THE Visualization_System SHALL enable or disable rendering within 100 milliseconds
5. WHILE visualization is enabled, THE Visualization_System SHALL update training metrics at a minimum of 1 Hz
6. WHILE visualization is enabled, THE Visualization_System SHALL display episode reward as a numeric value and success rate as a percentage
7. IF the laptop display is unavailable at startup, THEN THE Visualization_System SHALL continue training without rendering
8. IF rendering fails during training, THEN THE Visualization_System SHALL continue training and log an error message indicating the rendering failure

### Requirement 5: LEAP Hand PPO Training

**User Story:** As a robotics researcher, I want to train the LEAP hand alone to rotate a Rubik's cube using PPO, so that I can establish a baseline performance.

#### Acceptance Criteria

1. THE Training_System SHALL provide a configuration option to select LEAP_Hand-only training mode
2. WHEN LEAP_Hand training mode is selected and training is initiated, THE PPO_Trainer SHALL initialize the LEAP_Hand model in the IsaacLab simulation environment with a Rubik's cube object
3. IF the LEAP_Hand or Rubik's cube fails to initialize, THEN THE PPO_Trainer SHALL terminate with an error message indicating the initialization failure
4. WHEN a training episode executes, THE PPO_Trainer SHALL collect state observations including LEAP_Hand joint positions, joint velocities, and fingertip contact forces
5. WHEN a training episode executes, THE PPO_Trainer SHALL apply actions to the LEAP_Hand joint actuators at each simulation timestep
6. THE PPO_Trainer SHALL update the LEAP_Hand policy using the PPO algorithm with batch size between 64 and 4096 samples
7. THE PPO_Trainer SHALL perform between 3 and 15 policy update epochs per training iteration
8. WHEN a training run completes or is terminated, THE PPO_Trainer SHALL save the trained policy weights to disk in a standard checkpoint file format
9. IF the policy weights fail to save, THEN THE PPO_Trainer SHALL retry the save operation once and log an error if both attempts fail
10. WHEN a policy update iteration completes, THE Training_System SHALL log episode reward, success rate, and policy loss to the training log
11. IF logging fails, THEN THE Training_System SHALL continue training and report the logging failure to the console

### Requirement 6: 2DOF Arm with LEAP Hand PPO Training

**User Story:** As a robotics researcher, I want to train the 2DOF arm with LEAP hand to rotate a Rubik's cube using PPO, so that I can evaluate the combined system performance.

#### Acceptance Criteria

1. THE Training_System SHALL provide a configuration option to select Arm_System training mode with 2DOF arm and LEAP_Hand
2. WHEN Arm_System training mode is selected and training is initiated, THE PPO_Trainer SHALL initialize the Arm_System model in the IsaacLab simulation environment with a Rubik's cube object
3. WHEN the Arm_System initializes, THE PPO_Trainer SHALL verify that both the 2DOF arm joints and LEAP_Hand joints are present and controllable
4. WHEN a training episode executes, THE PPO_Trainer SHALL collect state observations including 2DOF arm joint positions, arm joint velocities, LEAP_Hand joint positions, LEAP_Hand joint velocities, and fingertip contact forces
5. WHEN a training episode executes, THE PPO_Trainer SHALL apply actions to both the 2DOF arm joint actuators and LEAP_Hand joint actuators at each simulation timestep
6. THE PPO_Trainer SHALL update the Arm_System policy using the PPO algorithm with batch size between 64 and 4096 samples
7. THE PPO_Trainer SHALL perform between 3 and 15 policy update epochs per training iteration
8. WHEN a training run completes or is terminated, THE PPO_Trainer SHALL save the trained policy weights to disk in a standard checkpoint file format
9. IF the policy weights fail to save, THEN THE PPO_Trainer SHALL retry the save operation once and log an error if both attempts fail
10. WHEN a policy update iteration completes, THE Training_System SHALL log episode reward, success rate, and policy loss to the training log
11. IF the Arm_System fails to initialize, THEN THE PPO_Trainer SHALL terminate with an error message indicating the initialization failure

### Requirement 7: Rubik's Cube Rotation Task

**User Story:** As a robotics researcher, I want the robot to perform Rubik's cube rotation as the training task, so that the system learns dexterous manipulation skills.

#### Acceptance Criteria

1. THE Training_System SHALL spawn a Rubik's cube object in the simulation environment at the start of each training episode
2. THE Training_System SHALL define the task objective as rotating a face of the Rubik's cube by 90 degrees within a tolerance of ±5 degrees
3. WHEN a training episode starts, THE Training_System SHALL place the Rubik's cube at a distance between 0.3 and 0.5 meters from the robot base in a reachable position
4. THE Training_System SHALL provide state observations including cube orientation as a quaternion, cube position as 3D coordinates, and robot joint positions as a vector
5. WHEN computing rewards during an episode, THE Training_System SHALL calculate progress as the angular displacement of the target cube face toward the 90-degree rotation goal
6. WHEN the target face rotation reaches 90 degrees within ±5 degrees tolerance, THE Training_System SHALL mark the episode as successful
7. WHEN an episode reaches 1000 timesteps, THE Training_System SHALL terminate the episode
8. WHEN the cube position moves more than 0.5 meters from its initial position, THE Training_System SHALL mark the cube as dropped and terminate the episode

### Requirement 8: Eureka Integration for LEAP Hand

**User Story:** As a robotics researcher, I want to use Eureka to automatically generate reward functions for the LEAP hand training, so that I can improve performance beyond manually designed rewards.

#### Acceptance Criteria

1. WHERE Eureka mode is enabled for LEAP_Hand, THE Eureka_Module SHALL generate reward function code using the Gemini API with a timeout of 60 seconds per generation request
2. IF the Gemini API call exceeds the timeout, THEN THE Eureka_Module SHALL log a timeout error and mark the generation attempt as failed
3. IF the generated code cannot be executed as a valid Python function, THEN THE Eureka_Module SHALL log the invalid code error and discard the generated function
4. WHEN the Eureka_Module generates a valid reward function, THE Eureka_Module SHALL evaluate the function by running 100 training episodes
5. IF an evaluation episode crashes or raises an exception, THEN THE Eureka_Module SHALL log the error and mark the reward function as invalid
6. THE Eureka_Module SHALL compute the performance of each reward function as the mean cumulative reward over the 100 evaluation episodes
7. THE Eureka_Module SHALL compare the performance of each generated reward function against the baseline reward function using mean cumulative reward
8. WHEN a generated reward function achieves at least 5% higher mean cumulative reward than the current best reward function, THE Eureka_Module SHALL update the active reward function
9. THE Eureka_Module SHALL iterate reward function generation for a number of iterations between 1 and 50, with a default of 10 iterations
10. WHEN Eureka training completes all iterations, THE Training_System SHALL save the best reward function code and corresponding policy weights to disk

### Requirement 9: Eureka Integration for 2DOF Arm System

**User Story:** As a robotics researcher, I want to use Eureka to automatically generate reward functions for the 2DOF arm with LEAP hand training, so that I can optimize the combined system performance.

#### Acceptance Criteria

1. WHERE Eureka mode is enabled for Arm_System, THE Eureka_Module SHALL generate reward function code using the Gemini API with a timeout of 60 seconds per generation request
2. IF the Gemini API call exceeds the timeout, THEN THE Eureka_Module SHALL log a timeout error and mark the generation attempt as failed
3. WHEN the Eureka_Module generates reward function code, THE Eureka_Module SHALL include both 2DOF arm joint state variables and LEAP_Hand joint state variables in the reward computation context
4. IF the generated code cannot be executed as a valid Python function, THEN THE Eureka_Module SHALL log the invalid code error and discard the generated function
5. WHEN the Eureka_Module generates a valid reward function, THE Eureka_Module SHALL evaluate the function by running 100 training episodes with the Arm_System
6. IF an evaluation episode crashes or raises an exception, THEN THE Eureka_Module SHALL log the error and mark the reward function as invalid
7. THE Eureka_Module SHALL compute the performance of each reward function as the mean cumulative reward over the 100 evaluation episodes
8. THE Eureka_Module SHALL compare the performance of each generated reward function against the current best reward function using mean cumulative reward
9. WHEN a generated reward function achieves at least 5% higher mean cumulative reward than the current best reward function, THE Eureka_Module SHALL update the active reward function
10. THE Eureka_Module SHALL iterate reward function generation for a number of iterations between 1 and 50, with a default of 10 iterations
11. WHEN Eureka training completes all iterations, THE Training_System SHALL save the best reward function code and corresponding policy weights to disk

### Requirement 10: Training Configuration Management

**User Story:** As a robotics researcher, I want to manage different training configurations, so that I can easily switch between LEAP hand only, Arm system, PPO mode, and Eureka mode.

#### Acceptance Criteria

1. THE Training_System SHALL support configuration files that specify robot type, training mode, and hyperparameters
2. THE Training_System SHALL validate configuration files before starting training
3. WHEN a configuration file is loaded, THE Training_System SHALL apply all specified settings
4. THE Training_System SHALL support command-line arguments to override configuration file settings
5. IF a required configuration parameter is missing, THEN THE Training_System SHALL report the missing parameter and terminate with an error message

### Requirement 11: Training Checkpointing and Resumption

**User Story:** As a robotics researcher, I want to save and resume training progress, so that I can continue training after interruptions.

#### Acceptance Criteria

1. WHILE training is active, THE Training_System SHALL save checkpoints at configurable intervals
2. THE Training_System SHALL save policy weights, optimizer state, and training statistics in each checkpoint
3. WHEN a checkpoint save is requested, THE Training_System SHALL complete the save within 10 seconds
4. THE Training_System SHALL support resuming training from a saved checkpoint
5. WHEN resuming from a checkpoint, THE Training_System SHALL restore policy weights, optimizer state, and training iteration counter
6. THE Training_System SHALL verify checkpoint integrity before loading

### Requirement 12: Performance Monitoring and Logging

**User Story:** As a robotics researcher, I want comprehensive logging of training performance, so that I can analyze and compare different training runs.

#### Acceptance Criteria

1. THE Training_System SHALL log training metrics to structured log files
2. THE Training_System SHALL record episode rewards, success rates, policy loss, value loss, and learning rate at each training iteration
3. WHERE Eureka mode is enabled, THE Training_System SHALL log each generated reward function and its evaluation performance
4. THE Training_System SHALL support exporting logs in CSV format
5. WHEN training completes, THE Training_System SHALL generate a summary report including final performance metrics and training duration
6. THE Training_System SHALL timestamp all log entries with millisecond precision
