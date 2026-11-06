"""Accept the Moect export shape and reject anything else."""

from __future__ import annotations


class SchemaError(Exception):
    """The file is not a Moect export we know how to read."""


def detect_schema(metadata: dict) -> str:
    if not isinstance(metadata, dict):
        raise SchemaError("metadata must be an object")
    application = metadata.get("application")
    if application == "Moect":
        return "moect"
    raise SchemaError(f"unsupported export application: {application!r}")
