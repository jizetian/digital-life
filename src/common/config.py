"""Load config/settings.yaml and optional environment overrides."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = ROOT / "config" / "settings.yaml"

_DEFAULTS = {
    "database": {"path": "data/database/cleaned_messages.db"},
    "import": {"batch_size": 1000, "raw_dir": "data/raw/moect_export"},
}


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@lru_cache(maxsize=1)
def load_settings() -> dict:
    _load_dotenv(ROOT / ".env")
    data = dict(_DEFAULTS)
    if SETTINGS_PATH.is_file():
        loaded = yaml.safe_load(SETTINGS_PATH.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            for key, value in loaded.items():
                if isinstance(value, dict) and isinstance(data.get(key), dict):
                    merged = dict(data[key])
                    merged.update(value)
                    data[key] = merged
                else:
                    data[key] = value
    return data


def default_db_path() -> Path:
    override = os.environ.get("CHATTWIN_DB_PATH")
    if override:
        return Path(override).expanduser()
    configured = load_settings()["database"]["path"]
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path


def default_batch_size() -> int:
    try:
        size = int(load_settings()["import"]["batch_size"])
    except (TypeError, ValueError, KeyError):
        return 1000
    return size if size > 0 else 1000
