"""Configuration management for the LEAP Hand Eureka Training System.

This module provides the ConfigurationManager class for loading and validating
environment and training configurations from .env files and config files.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7**
"""

import os
from pathlib import Path
from typing import Optional

import json
from typing import Any, Dict, List, Optional

from ..exceptions import ConfigurationError
from .models import (
    CheckpointConfig,
    EnvConfig,
    EurekaConfig,
    LoggingConfig,
    PPOConfig,
    TaskConfig,
    TrainingConfig,
    VisualizationConfig,
)


class ConfigurationManager:
    """Manager for loading and validating configuration files.
    
    Handles loading of environment configuration (.env files) and
    training configuration (YAML/JSON files), with validation and
    command-line override support.
    """
    
    def load_env_config(self, env_path: str = ".env") -> EnvConfig:
        """Load environment configuration from .env file.
        
        Parses the .env file to extract GEMINI_API_KEY and LLM_MODEL fields.
        Validates that GEMINI_API_KEY is present and non-empty.
        
        Args:
            env_path: Path to .env file (default: ".env")
            
        Returns:
            EnvConfig object with API credentials and model identifier
            
        Raises:
            ConfigurationError: If file is missing, unreadable, or validation fails
            
        **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7**
        """
        env_file = Path(env_path)
        
        # Check if file exists and is readable
        if not env_file.exists():
            raise ConfigurationError(
                f"Environment configuration file not found: {env_path}"
            )
        
        if not env_file.is_file():
            raise ConfigurationError(
                f"Environment configuration path is not a file: {env_path}"
            )
        
        try:
            with open(env_file, 'r') as f:
                content = f.read()
        except (OSError, IOError) as e:
            raise ConfigurationError(
                f"Failed to read environment configuration file {env_path}: {e}"
            )
        
        # Parse .env file format (KEY=value lines)
        config_vars = {}
        for line_num, line in enumerate(content.splitlines(), 1):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Parse KEY=value format
            if '=' not in line:
                raise ConfigurationError(
                    f"Invalid format in {env_path} at line {line_num}: "
                    f"expected KEY=value format"
                )
            
            key, _, value = line.partition('=')
            key = key.strip()
            value = value.strip()
            
            # Remove quotes from value if present
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            
            config_vars[key] = value
        
        # Validate GEMINI_API_KEY is present and non-empty
        gemini_api_key = config_vars.get('GEMINI_API_KEY', '').strip()
        if not gemini_api_key:
            raise ConfigurationError(
                f"GEMINI_API_KEY is missing or empty in {env_path}. "
                "Please add a valid API key to the .env file."
            )
        
        # Get LLM_MODEL with default value
        llm_model = config_vars.get('LLM_MODEL', 'gemini-1.5-pro').strip()
        if not llm_model:
            llm_model = 'gemini-1.5-pro'
        
        return EnvConfig(
            gemini_api_key=gemini_api_key,
            llm_model=llm_model
        )

    def create_default_env_file(self, env_path: str = ".env") -> None:
        """Create a default .env file with placeholder Gemini credentials."""
        env_file = Path(env_path)
        if env_file.exists():
            return
        env_file.write_text(
            "GEMINI_API_KEY=\n"
            "LLM_MODEL=gemini-2.0-flash\n",
            encoding="utf-8",
        )

    def load_training_config(self, config_path: str) -> TrainingConfig:
        """Load YAML or JSON training configuration."""
        path = Path(config_path)
        if not path.exists():
            raise ConfigurationError(f"Training configuration file not found: {config_path}")
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigurationError(f"Failed to read training configuration {config_path}: {exc}") from exc

        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise ConfigurationError("PyYAML is required to load YAML training configs.") from exc
            data = yaml.safe_load(raw) or {}
        elif path.suffix.lower() == ".json":
            data = json.loads(raw)
        else:
            raise ConfigurationError(f"Unsupported configuration format: {path.suffix}")

        if not isinstance(data, dict):
            raise ConfigurationError("Training configuration must be a mapping.")

        try:
            config = TrainingConfig(
                robot_type=data["robot_type"],
                training_mode=data["training_mode"],
                ppo=PPOConfig(**data.get("ppo", {})),
                eureka=EurekaConfig(**data.get("eureka", {})),
                task=TaskConfig(**data.get("task", {})),
                visualization=VisualizationConfig(**data.get("visualization", {})),
                checkpoint=CheckpointConfig(**data.get("checkpoint", {})),
                logging=LoggingConfig(**data.get("logging", {})),
            )
        except KeyError as exc:
            raise ConfigurationError(f"Missing required configuration parameter: {exc}") from exc
        except TypeError as exc:
            raise ConfigurationError(f"Invalid training configuration: {exc}") from exc

        errors = self.validate_config(config)
        if errors:
            raise ConfigurationError("Invalid training configuration:\n" + "\n".join(errors))
        return config

    def validate_config(self, config: TrainingConfig) -> List[str]:
        errors: List[str] = []
        if config.robot_type not in {"leap_hand", "arm_system"}:
            errors.append("robot_type must be 'leap_hand' or 'arm_system'")
        if config.training_mode not in {"ppo", "eureka"}:
            errors.append("training_mode must be 'ppo' or 'eureka'")
        if not 64 <= config.ppo.batch_size <= 4096:
            errors.append("ppo.batch_size must be between 64 and 4096")
        if not 3 <= config.ppo.num_epochs <= 15:
            errors.append("ppo.num_epochs must be between 3 and 15")
        if not 1 <= config.eureka.num_iterations <= 50:
            errors.append("eureka.num_iterations must be between 1 and 50")
        if config.eureka.evaluation_episodes <= 0:
            errors.append("eureka.evaluation_episodes must be > 0")
        if config.task.max_episode_steps <= 0:
            errors.append("task.max_episode_steps must be > 0")
        return errors

    def apply_cli_overrides(self, config: TrainingConfig, overrides: Optional[Dict[str, Any]] = None) -> TrainingConfig:
        if not overrides:
            return config
        for key, value in overrides.items():
            if value is None:
                continue
            if key == "robot_type":
                config.robot_type = value
            elif key == "training_mode":
                config.training_mode = value
            elif key == "max_training_iterations":
                setattr(config.ppo, "max_training_iterations", value)
            elif key == "max_episode_steps":
                config.task.max_episode_steps = value
        errors = self.validate_config(config)
        if errors:
            raise ConfigurationError("Invalid training configuration after CLI overrides:\n" + "\n".join(errors))
        return config
