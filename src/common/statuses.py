"""Shared pipeline statuses for segments and distillation tasks."""

from __future__ import annotations

PENDING = "pending"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
RETRYING = "retrying"
SKIPPED = "skipped"

PIPELINE_STATUSES = (PENDING, RUNNING, COMPLETED, FAILED, RETRYING, SKIPPED)
