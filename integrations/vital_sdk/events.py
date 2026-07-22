"""Event helpers, types, and category constants for the Vital event system."""

from typing import Optional


class EventCategory:
    """Well-known event categories matching vital-core's event store."""

    HEALTH = "health"
    EDUCATION = "education"
    IDENTITY = "identity"
    SECURITY = "security"
    AI = "ai"
    SYSTEM = "system"


class EventBusChannel:
    """Redis Pub/Sub channel names used by vital-core's EventBus."""

    HEALTH = "vital.health"
    EDUCATION = "vital.education"
    IDENTITY = "vital.identity"
    SECURITY = "vital.security"
    SYSTEM = "vital.system"
    AI = "vital.ai"
    ENERGY = "vital.energy"

    @staticmethod
    def for_category(category: str) -> str:
        """Return the Redis channel name for a given category."""
        return f"vital.{category}"


# -- Codking-specific event types -------------------------------------------

CODKING_EVENT_TYPES = {
    "threat.detected": EventCategory.SECURITY,
    "threat.batch_analyzed": EventCategory.SECURITY,
    "classification.completed": EventCategory.AI,
    "log_file.analyzed": EventCategory.SECURITY,
    "model.loaded": EventCategory.AI,
}


def create_event(
    category: str,
    action: str,
    source: str,
    event_type: str = "sdk",
    payload: Optional[dict] = None,
    metadata: Optional[dict] = None,
    tags: Optional[list[str]] = None,
    subcategory: Optional[str] = None,
) -> dict:
    """Build an event dict matching vital-core's ``EventCreate`` schema.

    Returns a dict ready to POST to ``/api/v1/events``.
    """
    return {
        "category": category,
        "subcategory": subcategory,
        "source": source,
        "action": action,
        "event_type": event_type,
        "payload": payload or {},
        "metadata": metadata or {},
        "tags": tags or [],
    }
