"""Conversation segmentation over local SQLite messages."""

from segmentation.service import SegmentationConflictError, run_segmentation
from segmentation.splitter import SegmentParams

__all__ = ["SegmentParams", "SegmentationConflictError", "run_segmentation"]
