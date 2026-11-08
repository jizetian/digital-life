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
    "segmentation": {
        "gap_minutes": 30,
        "max_messages": 50,
        "overlap_messages": 4,
        "overlap_on_time_gap": False,
        "page_size": 2000,
    },
    "distillation": {
        "batch_size": 8,
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "qwen2.5:7b",
        "model_version": "",
        "timeout_seconds": 120,
        "temperature": 0,
        "max_attempts": 3,
        "max_chars_per_message": 4000,
    },
    "personality": {
        "batch_size": 20,
        "contradict_batch_size": 30,
        "max_attempts": 3,
        "min_independent": 2,
        "max_chars_per_summary": 500,
        "output_dir": "personality",
    },
    "memory": {
        "batch_size": 4,
        "max_attempts": 3,
        "max_chars_per_message": 2000,
        "max_existing": 40,
    },
    "retrieval": {
        "embedding_provider": "openai",
        "embedding_model": "bge-m3",
        "embedding_base_url": "",
        "embedding_dimension": 1024,
        "embedding_batch_size": 16,
        "embedding_timeout_seconds": 120,
        "qdrant_path": "data/qdrant",
        "qdrant_url": "",
        "qdrant_api_key": "",
        "vector_backend": "qdrant",
        "rerank_url": "",
        "rrf_k": 60,
        "default_limit": 10,
        "candidate_limit": 40,
        "max_chars": 8000,
    },
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


def _merged_section(name: str) -> dict:
    section = load_settings().get(name) or {}
    if not isinstance(section, dict):
        return {}
    return dict(section)


def _apply_str(cfg: dict, env_name: str, key: str) -> None:
    raw = os.environ.get(env_name)
    if raw:
        cfg[key] = raw


def _apply_int(cfg: dict, env_name: str, key: str) -> None:
    raw = os.environ.get(env_name)
    if raw:
        cfg[key] = int(raw)


def segmentation_config() -> dict:
    """Segmentation settings with optional environment overrides.

    Environment is applied here, not inside the cached YAML load, so a shell
    override wins over ``config/settings.yaml``.
    """
    cfg = _merged_section("segmentation")
    _apply_str(cfg, "CHATTWIN_SEGMENT_GAP_MINUTES", "gap_minutes")
    _apply_int(cfg, "CHATTWIN_SEGMENT_MAX_MESSAGES", "max_messages")
    _apply_int(cfg, "CHATTWIN_SEGMENT_OVERLAP", "overlap_messages")
    flag = os.environ.get("CHATTWIN_SEGMENT_OVERLAP_ON_TIME_GAP")
    if flag is not None and flag != "":
        cfg["overlap_on_time_gap"] = flag.strip().lower() in {"1", "true", "yes", "on"}
    return cfg


def distillation_config() -> dict:
    """L1 distillation settings with optional environment overrides."""
    cfg = _merged_section("distillation")
    _apply_str(cfg, "CHATTWIN_LLM_BASE_URL", "base_url")
    _apply_str(cfg, "CHATTWIN_LLM_MODEL", "model")
    _apply_str(cfg, "CHATTWIN_MODEL_VERSION", "model_version")
    _apply_int(cfg, "CHATTWIN_DISTILL_BATCH_SIZE", "batch_size")
    _apply_int(cfg, "CHATTWIN_DISTILL_MAX_ATTEMPTS", "max_attempts")
    return cfg


def memory_config() -> dict:
    """Long-term memory settings. The model endpoint is the shared LLM client."""
    cfg = _merged_section("memory")
    _apply_int(cfg, "CHATTWIN_MEMORY_BATCH_SIZE", "batch_size")
    _apply_int(cfg, "CHATTWIN_MEMORY_MAX_ATTEMPTS", "max_attempts")
    _apply_int(cfg, "CHATTWIN_MEMORY_MAX_EXISTING", "max_existing")
    _apply_str(cfg, "CHATTWIN_MEMORY_PROMPT_VERSION", "prompt_version")
    return cfg


def retrieval_config() -> dict:
    """Hybrid retrieval settings. The embedding endpoint is separate from the chat model."""
    cfg = _merged_section("retrieval")
    _apply_str(cfg, "CHATTWIN_EMBEDDING_PROVIDER", "embedding_provider")
    _apply_str(cfg, "CHATTWIN_EMBEDDING_MODEL", "embedding_model")
    _apply_str(cfg, "CHATTWIN_EMBEDDING_BASE_URL", "embedding_base_url")
    _apply_int(cfg, "CHATTWIN_EMBEDDING_DIMENSION", "embedding_dimension")
    _apply_int(cfg, "CHATTWIN_EMBEDDING_BATCH_SIZE", "embedding_batch_size")
    _apply_int(cfg, "CHATTWIN_EMBEDDING_TIMEOUT", "embedding_timeout_seconds")
    _apply_str(cfg, "CHATTWIN_QDRANT_PATH", "qdrant_path")
    _apply_str(cfg, "CHATTWIN_QDRANT_URL", "qdrant_url")
    _apply_str(cfg, "CHATTWIN_QDRANT_API_KEY", "qdrant_api_key")
    _apply_str(cfg, "CHATTWIN_VECTOR_BACKEND", "vector_backend")
    _apply_str(cfg, "CHATTWIN_RERANK_URL", "rerank_url")
    _apply_int(cfg, "CHATTWIN_RRF_K", "rrf_k")
    _apply_int(cfg, "CHATTWIN_SEARCH_LIMIT", "default_limit")
    _apply_int(cfg, "CHATTWIN_SEARCH_CANDIDATES", "candidate_limit")
    _apply_int(cfg, "CHATTWIN_RETRIEVAL_MAX_CHARS", "max_chars")
    return cfg


def qdrant_path() -> Path:
    configured = retrieval_config().get("qdrant_path") or "data/qdrant"
    path = Path(str(configured)).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path


def personality_config() -> dict:
    """L2/L3 personality settings. The model endpoint is the distillation client."""
    cfg = _merged_section("personality")
    _apply_int(cfg, "CHATTWIN_PERSONALITY_BATCH_SIZE", "batch_size")
    _apply_int(cfg, "CHATTWIN_PERSONALITY_CONTRADICT_BATCH_SIZE", "contradict_batch_size")
    _apply_int(cfg, "CHATTWIN_PERSONALITY_MAX_ATTEMPTS", "max_attempts")
    _apply_int(cfg, "CHATTWIN_PERSONALITY_MIN_INDEPENDENT", "min_independent")
    _apply_str(cfg, "CHATTWIN_PERSONALITY_OUTPUT_DIR", "output_dir")
    _apply_str(cfg, "CHATTWIN_L2_PROMPT_VERSION", "prompt_version")
    return cfg
