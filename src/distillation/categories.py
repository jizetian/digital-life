"""L1 observation categories from the ChatTwin distillation spec.

These are evidence-level notes, not a personality model (L2/L3).
"""

from __future__ import annotations

L1_CATEGORIES = (
    "communication",
    "emotion",
    "opinion",
    "value",
    "thinking",
    "decision",
    "social",
    "interest",
    "event",
    "fact",
)

L1_CATEGORY_SET = frozenset(L1_CATEGORIES)
