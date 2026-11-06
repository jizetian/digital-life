"""Map Moect ``is_sender`` onto owner / target_person.

``is_sender`` true is the export owner. ``is_sender`` false is the target person.
A missing flag stays ``unknown`` — display names are not used to guess.
"""

from __future__ import annotations

from common.schemas import ROLE_OWNER, ROLE_TARGET, ROLE_UNKNOWN


def resolve_identity(raw: dict) -> tuple[str, bool | None]:
    if "is_sender" not in raw:
        return ROLE_UNKNOWN, None
    flag = raw.get("is_sender")
    if flag is True:
        return ROLE_OWNER, True
    if flag is False:
        return ROLE_TARGET, False
    return ROLE_UNKNOWN, None


def sender_is_missing(sender_id: str | None, sender_name: str | None) -> bool:
    return sender_id is None and sender_name is None
