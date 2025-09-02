import logging
from pathlib import Path
from shutil import copyfile

import yaml

__all__ = ["ConfigTests"]

CONFIG_DIR = Path(__file__).parent


class ConfigTests:
    """Class that parses the YAML config."""

    def __init__(self, config_name="config.yaml"):
        self._config_path = CONFIG_DIR / config_name
        self._cabaret_config_path = CONFIG_DIR / "config_cabaret.yaml"
        self._DEFAULT_PATH = CONFIG_DIR / "template.yaml"
        self._DEFAULT_CABARET_PATH = CONFIG_DIR / "template_cabaret.yaml"
        self._config = None

    def get(self) -> dict:
        """Return the loaded config. Loads on first access."""
        if self._config is None:
            self._config = self._ensure_and_load()
        return self._config

    def _ensure_and_load(self):
        if not CONFIG_DIR.exists():
            raise FileNotFoundError(f"Config directory not found: {CONFIG_DIR!s}")

        if not self._config_path.exists():
            self._initialise_with_default_config()

        if not self._cabaret_config_path.exists():
            self._initialise_with_default_config_cabaret()

        with open(self._config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        with open(self._cabaret_config_path, encoding="utf-8") as f:
            config["cabaret"] = yaml.safe_load(f)

        for key, value in config.items():
            if isinstance(value, str) and value.startswith("/path/to/"):
                config[key] = None
                logging.info(f"Config key '{key}' set to None as it contains a placeholder path.")

        return config

    def _initialise_with_default_config(self):
        """Copy default config to config.yaml if not present"""
        if not self._DEFAULT_PATH.exists():
            raise FileNotFoundError(f"Default config not found at {self._DEFAULT_PATH!s}")

        copyfile(self._DEFAULT_PATH, self._config_path)

    def _initialise_with_default_config_cabaret(self):
        """Copy default cabaret config to config_cabaret.yaml if not present"""
        if not self._DEFAULT_CABARET_PATH.exists():
            raise FileNotFoundError(f"Default cabaret config not found at {self._DEFAULT_CABARET_PATH!s}")

        copyfile(self._DEFAULT_CABARET_PATH, self._cabaret_config_path)

    def __getitem__(self, key):
        return self.get()[key]

    def reload(self):
        """Force reload from disk"""
        self._config = self._ensure_and_load()
        return self._config
