from typing import Any, Dict


class Datapoint:
    """Simple container for a datapoint and its metadata."""

    def __init__(self, id: int, metadata: Dict[str, Any] = None):
        self.id = id
        self.metadata = metadata or {}

    def add_metadata(self, key: str, value: Any) -> None:
        """Add or update a metadata value."""
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value with optional default."""
        return self.metadata.get(key, default)

    def has_metadata(self, key: str) -> bool:
        """Check if metadata key exists."""
        return key in self.metadata
