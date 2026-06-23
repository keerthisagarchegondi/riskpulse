"""Centralized configuration loader for RiskPulse platform.

Loads configuration from YAML files with environment-specific overrides
and environment variable substitution. Follows the hierarchy:
  1. config/settings.yaml (base defaults)
  2. config/environments/{env}.yaml (environment overrides)
  3. Environment variables (highest priority, prefix: RISKPULSE_)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_env_vars(config: Any) -> Any:
    """Recursively resolve ${ENV_VAR} and ${ENV_VAR:-default} patterns in string values."""
    if isinstance(config, dict):
        return {k: _resolve_env_vars(v) for k, v in config.items()}
    if isinstance(config, list):
        return [_resolve_env_vars(item) for item in config]
    if isinstance(config, str) and "${" in config:
        import re

        def _replace(match: re.Match) -> str:
            expr = match.group(1)
            if ":-" in expr:
                var_name, default = expr.split(":-", 1)
                return os.environ.get(var_name, default)
            return os.environ.get(expr, match.group(0))

        return re.sub(r"\$\{([^}]+)\}", _replace, config)
    return config


def _get_project_root() -> Path:
    """Find project root by looking for pyproject.toml."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parent.parent


class Settings:
    """Application settings loaded from YAML configuration files."""

    def __init__(self, environment: str | None = None) -> None:
        self._environment = environment or os.environ.get("RISKPULSE_ENV", "dev")
        self._config: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """Load configuration from YAML files with environment overrides."""
        root = _get_project_root()
        config_dir = root / "config"

        # Load base settings
        base_path = config_dir / "settings.yaml"
        if base_path.exists():
            with open(base_path, "r") as f:
                self._config = yaml.safe_load(f) or {}

        # Load environment-specific overrides
        env_path = config_dir / "environments" / f"{self._environment}.yaml"
        if env_path.exists():
            with open(env_path, "r") as f:
                env_config = yaml.safe_load(f) or {}
                self._config = _deep_merge(self._config, env_config)

        # Resolve environment variable references
        self._config = _resolve_env_vars(self._config)

        # Apply direct environment variable overrides (RISKPULSE_ prefix)
        self._apply_env_overrides()

    def _apply_env_overrides(self) -> None:
        """Apply RISKPULSE_ prefixed environment variables as overrides.

        Converts RISKPULSE_DATABASE__POOL_SIZE=30 to config["database"]["pool_size"] = 30
        """
        prefix = "RISKPULSE_"
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            config_path = key[len(prefix):].lower().split("__")
            target = self._config
            for part in config_path[:-1]:
                target = target.setdefault(part, {})
            # Attempt type coercion
            target[config_path[-1]] = self._coerce_value(value)

    @staticmethod
    def _coerce_value(value: str) -> Any:
        """Coerce string environment variable to appropriate Python type."""
        if value.lower() in ("true", "1", "yes"):
            return True
        if value.lower() in ("false", "0", "no"):
            return False
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value

    @property
    def environment(self) -> str:
        return self._environment

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get a config value using dot-notation path.

        Args:
            key_path: Dot-separated path, e.g. "database.pool_size"
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key_path.split(".")
        value = self._config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        return value

    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access to top-level config sections."""
        return self._config[key]

    def __contains__(self, key: str) -> bool:
        return key in self._config

    @property
    def database_url(self) -> str:
        """Construct PostgreSQL database URL from config."""
        host = os.environ.get("RISKPULSE_DB_HOST", "localhost")
        port = os.environ.get("RISKPULSE_DB_PORT", "5432")
        name = os.environ.get("RISKPULSE_DB_NAME", "riskpulse")
        user = os.environ.get("RISKPULSE_DB_USER", "riskpulse")
        password = os.environ.get("RISKPULSE_DB_PASSWORD", "riskpulse")
        return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{name}"

    @property
    def database_url_sync(self) -> str:
        """Construct synchronous PostgreSQL URL (for migrations, scripts)."""
        host = os.environ.get("RISKPULSE_DB_HOST", "localhost")
        port = os.environ.get("RISKPULSE_DB_PORT", "5432")
        name = os.environ.get("RISKPULSE_DB_NAME", "riskpulse")
        user = os.environ.get("RISKPULSE_DB_USER", "riskpulse")
        password = os.environ.get("RISKPULSE_DB_PASSWORD", "riskpulse")
        return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"

    @property
    def redis_url(self) -> str:
        """Construct Redis URL from environment."""
        host = os.environ.get("RISKPULSE_REDIS_HOST", "localhost")
        port = os.environ.get("RISKPULSE_REDIS_PORT", "6379")
        db = os.environ.get("RISKPULSE_REDIS_DB", "0")
        return f"redis://{host}:{port}/{db}"

    @property
    def kafka_bootstrap_servers(self) -> str:
        """Kafka bootstrap servers from environment."""
        return os.environ.get("RISKPULSE_KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

    @property
    def is_debug(self) -> bool:
        """Whether debug mode is enabled."""
        return self.get("app.debug", False)

    def as_dict(self) -> dict[str, Any]:
        """Return full configuration as dictionary (for debugging)."""
        return self._config.copy()


@lru_cache(maxsize=1)
def get_settings(environment: str | None = None) -> Settings:
    """Get cached Settings instance (singleton per environment)."""
    return Settings(environment=environment)
