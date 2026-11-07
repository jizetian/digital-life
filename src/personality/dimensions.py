"""Six L3 personality dimensions from the ChatTwin distillation spec.

L1 categories that are events, facts, or interests stay context notes.
They are not promoted into the personality profile.
"""

from __future__ import annotations

DIMENSIONS = (
    "communication",
    "emotion",
    "thinking",
    "values",
    "social",
    "decision",
)

# L1 category -> L3 dimension. ``value`` is the L1 name; ``values`` is the profile dimension.
L1_TO_DIMENSION = {
    "communication": "communication",
    "emotion": "emotion",
    "opinion": "thinking",
    "thinking": "thinking",
    "value": "values",
    "social": "social",
    "decision": "decision",
}

CONTEXT_CATEGORIES = ("interest", "event", "fact")

DIMENSION_META = {
    "communication": {
        "title": "Communication style",
        "heading": "Language and communication style",
        "file": "communication_style.md",
    },
    "emotion": {
        "title": "Emotional patterns",
        "heading": "Emotion",
        "file": "emotional_patterns.md",
    },
    "thinking": {
        "title": "Thinking patterns",
        "heading": "Thinking",
        "file": "thinking_patterns.md",
    },
    "values": {
        "title": "Values",
        "heading": "Values",
        "file": "values.md",
    },
    "social": {
        "title": "Social patterns",
        "heading": "Social",
        "file": "social_patterns.md",
    },
    "decision": {
        "title": "Decision patterns",
        "heading": "Decision",
        "file": "decision_patterns.md",
    },
}

PROFILE_FILES = (
    "personality_core.md",
    *(DIMENSION_META[name]["file"] for name in DIMENSIONS),
    "personality_evidence.json",
)


def dimension_for_category(category: str) -> str | None:
    return L1_TO_DIMENSION.get(category)
