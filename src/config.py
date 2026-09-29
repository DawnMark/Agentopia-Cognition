# src/config.py

import json
from pathlib import Path
from typing import Dict, Any

# This global variable holds all configuration data.
_CONFIG: Dict[str, Any] | None = None

# Keys whose values may hold secrets; redacted before logging the config.
_SENSITIVE_KEYS = {"api_key", "credentials_file", "token", "secret", "password"}


def _redact_secrets(obj: Any) -> Any:
    """Return a deep copy of obj with sensitive values masked, for safe printing."""
    if isinstance(obj, dict):
        redacted: Dict[str, Any] = {}
        for k, v in obj.items():
            if k in _SENSITIVE_KEYS and isinstance(v, str) and v not in ("", "EMPTY"):
                redacted[k] = "***REDACTED***"
            else:
                redacted[k] = _redact_secrets(v)
        return redacted
    if isinstance(obj, list):
        return [_redact_secrets(v) for v in obj]
    return obj


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load the main config.json file and store it for global access.

    This should be called once by run_world.py at application startup.
    """
    global _CONFIG

    root_cfg = Path(config_path)

    if not root_cfg.exists():
        raise FileNotFoundError(f"config.json not found at {root_cfg}")

    with root_cfg.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("config.json must be a JSON object")

    # Store the loaded data in the global variable.
    _CONFIG = data
    print(
        "=" * 50
        + f"\nConfiguration loaded from {root_cfg}.\nConfig:{_redact_secrets(data)}\n"
        + "=" * 50
    )
    return data


def get_config() -> Dict[str, Any]:
    """
    Return the full loaded configuration.
    """
    if _CONFIG is None:
        # Safeguard to ensure load_config() is always called first.
        CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"
        print(
            "=" * 50
            + f"\nConfiguration is not loaded. Call load_config() and load default config from {CONFIG_PATH}.\n"
            + "=" * 50
        )
        load_config(CONFIG_PATH)
    return _CONFIG


def get_world_config() -> Dict[str, Any]:
    """
    Helper function for quickly fetching the 'world' sub-configuration.
    """
    config = get_config()
    if "world" not in config:
        raise ValueError("config.json must contain a 'world' field")
    return config["world"]
