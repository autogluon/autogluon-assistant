import logging
import re
from copy import deepcopy
from importlib.resources import files
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf


def _get_default_config_path(
    presets: str,
) -> Path:
    """
    Get default config folder under package root
    Returns Path to the config.yaml file
    """
    try:
        # Look directly in the assistant package for configs
        config_path = files("autogluon.assistant") / "configs" / f"{presets}.yaml"

        if not config_path.exists():
            raise ValueError(
                f"Config file not found at expected location: {config_path}\n"
                "Please ensure the config files are properly installed in the configs directory."
            )
        return config_path
    except Exception:
        # Fallback for development environment
        package_root = Path(__file__).parent.parent
        config_path = package_root / "configs" / f"{presets}.yaml"
        if not config_path.exists():
            raise ValueError(f"Config file not found: {config_path}")
        return config_path


def parse_override(override: str) -> tuple:
    """
    Parse a single override string in the format 'key=value' or 'key.nested=value'

    Args:
        override: String in format "key=value" or "key.nested=value"

    Returns:
        Tuple of (key, value)

    Raises:
        ValueError: If override string is not in correct format
    """
    if "=" not in override:
        raise ValueError(f"Invalid override format: {override}. Must be in format 'key=value' or 'key.nested=value'")
    key, value = override.split("=", 1)
    return key, value


def apply_overrides(config: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """
    Apply command-line overrides to config
    Args:
        config: Base configuration
        overrides: List of overrides in format ["key1=value1", "key2.nested=value2"]
    Returns:
        Updated configuration
    """
    if not overrides:
        return config

    # Convert overrides to nested dict
    override_conf = {}
    overrides = ",".join(overrides)
    # Split by comma but preserve commas inside square brackets
    overrides = re.split(r",(?![^\[]*\])", overrides)

    for override in overrides:
        override = override.strip()  # Remove any whitespace
        key, value = parse_override(override)

        # Handle list values enclosed in square brackets
        if value.startswith("[") and value.endswith("]"):
            # Extract items between brackets and split by comma
            items = value[1:-1].split(",")
            # Clean up each item and convert to list
            value = [item.strip() for item in items if item.strip()]
        else:
            # Try to convert value to appropriate type for non-list values
            try:
                value = eval(value)
            except:
                # Keep as string if eval fails
                pass

        # Handle nested keys
        current = override_conf
        key_parts = key.split(".")
        for part in key_parts[:-1]:
            current = current.setdefault(part, {})
        current[key_parts[-1]] = value

    # Convert override dict to OmegaConf and merge
    override_conf = OmegaConf.create(override_conf)
    return OmegaConf.merge(config, override_conf)


def load_config(
    presets: str, config_path: Optional[str] = None, overrides: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Load configuration from yaml file, merging with default config and applying overrides

    Args:
        config_path: Optional path to config file. If provided, will merge with and override default config
        overrides: Optional list of command-line overrides in format ["key1=value1", "key2.nested=value2"]

    Returns:
        Loaded and merged configuration

    Raises:
        ValueError: If config file not found or invalid
    """
    # Load default config
    default_config_path = _get_default_config_path(presets="default")
    logging.info(f"Loading default config from: {default_config_path}")
    config = OmegaConf.load(default_config_path)

    # Apply Presets
    presets_config_path = _get_default_config_path(presets=presets)
    presets_config = OmegaConf.load(presets_config_path)
    logging.info(f"Merging {presets} config from: {presets_config_path}")
    config = OmegaConf.merge(config, presets_config)

    # If custom config provided, merge it
    if config_path:
        custom_config_path = Path(config_path)
        if not custom_config_path.is_file():
            raise ValueError(f"Custom config file not found at: {custom_config_path}")

        logging.info(f"Loading custom config from: {custom_config_path}")
        custom_config = OmegaConf.load(custom_config_path)
        config = OmegaConf.merge(config, custom_config)
        logging.info("Successfully merged custom config with default config")

    # Apply command-line overrides if any
    if overrides:
        logging.info(f"Applying command-line overrides: {overrides}")
        config = apply_overrides(config, overrides)
        logging.info("Successfully applied command-line overrides")

    return config


def get_feature_transformers_config(config: OmegaConf) -> Optional[List[Dict[str, Any]]]:
    """
    Retrieve the configuration of feature transformers based on enabled models.
    Returns None if no models are enabled.

    Args:
        config (OmegaConf): The configuration object loaded from YAML

    Returns:
        Optional[List[Dict[str, Any]]]: List of transformer configurations,
                                      or None if no models are enabled
    """
    # Get list of enabled models
    enabled_models = config.feature_transformers.enabled_models

    # Convert string to list if single model string
    if isinstance(enabled_models, str):
        enabled_models = [enabled_models]

    # Return None if no models are enabled
    if not enabled_models:
        return None

    # Get all available model configurations
    all_models_config = config.feature_transformers.models

    # Create list of configurations for enabled models
    transformers_config = [
        OmegaConf.to_container(all_models_config[model_name], resolve=True)
        for model_name in enabled_models
        if model_name in all_models_config
    ]

    # Return None if no valid configurations were found
    return transformers_config if transformers_config else None


def unpack_omega_config(config):
    temp_config = deepcopy(config)
    dict_config = OmegaConf.to_container(temp_config, resolve=True)
    return dict_config
