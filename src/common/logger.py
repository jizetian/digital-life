"""Logging setup. Message bodies are never written to the log."""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format=_FORMAT)
