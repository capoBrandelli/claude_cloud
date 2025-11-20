"""
Configuration loader for TradeAI
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

from tradeai.utils.logger import get_logger

logger = get_logger(__name__)


class ConfigLoader:
    """Load and manage configuration from YAML files and environment variables."""

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize ConfigLoader.

        Args:
            config_dir: Path to configuration directory. If None, uses default.
        """
        if config_dir is None:
            config_dir = Path(__file__).parent.parent.parent / "config"

        self.config_dir = Path(config_dir)
        self._configs: Dict[str, Any] = {}

        # Load environment variables
        load_dotenv()

        logger.info(f"ConfigLoader initialized with config_dir: {self.config_dir}")

    def load(self, config_name: str) -> Dict[str, Any]:
        """
        Load a configuration file.

        Args:
            config_name: Name of the configuration file (without .yaml extension)

        Returns:
            Configuration dictionary
        """
        if config_name in self._configs:
            return self._configs[config_name]

        config_path = self.config_dir / f"{config_name}.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        logger.debug(f"Loading configuration from: {config_path}")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Replace environment variables
        config = self._replace_env_vars(config)

        self._configs[config_name] = config
        logger.info(f"Configuration '{config_name}' loaded successfully")

        return config

    def _replace_env_vars(self, config: Any) -> Any:
        """
        Recursively replace ${ENV_VAR} placeholders with environment variable values.

        Args:
            config: Configuration object (dict, list, or value)

        Returns:
            Configuration with environment variables replaced
        """
        if isinstance(config, dict):
            return {k: self._replace_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._replace_env_vars(item) for item in config]
        elif isinstance(config, str):
            # Replace ${VAR_NAME} with environment variable value
            if config.startswith("${") and config.endswith("}"):
                env_var = config[2:-1]
                value = os.getenv(env_var)
                if value is None:
                    logger.warning(
                        f"Environment variable '{env_var}' not set, using placeholder"
                    )
                    return config
                return value
            return config
        else:
            return config

    def get(self, config_name: str, key_path: Optional[str] = None, default: Any = None) -> Any:
        """
        Get a configuration value by key path.

        Args:
            config_name: Name of the configuration file
            key_path: Dot-separated path to the key (e.g., 'app.name')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        config = self.load(config_name)

        if key_path is None:
            return config

        keys = key_path.split(".")
        value = config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                logger.debug(f"Key path '{key_path}' not found, returning default")
                return default

        return value

    def reload(self, config_name: str) -> Dict[str, Any]:
        """
        Reload a configuration file.

        Args:
            config_name: Name of the configuration file

        Returns:
            Reloaded configuration dictionary
        """
        if config_name in self._configs:
            del self._configs[config_name]
        return self.load(config_name)

    def clear_cache(self) -> None:
        """Clear all cached configurations."""
        self._configs.clear()
        logger.info("Configuration cache cleared")


# Global config loader instance
_config_loader = ConfigLoader()


def load_config(config_name: str) -> Dict[str, Any]:
    """
    Load a configuration file using the global ConfigLoader instance.

    Args:
        config_name: Name of the configuration file (without .yaml extension)

    Returns:
        Configuration dictionary
    """
    return _config_loader.load(config_name)


def get_config(config_name: str, key_path: Optional[str] = None, default: Any = None) -> Any:
    """
    Get a configuration value using the global ConfigLoader instance.

    Args:
        config_name: Name of the configuration file
        key_path: Dot-separated path to the key
        default: Default value if key not found

    Returns:
        Configuration value
    """
    return _config_loader.get(config_name, key_path, default)
