from typing import Any, Dict


class MetadataStore:
    """Container for a datapoint's metadata.

    MetadataStore objects store metadata associated with a datapoint which is referenced by ID.
    """

    def __init__(self, id: int, metadata: Dict[str, Any] = None):
        """Initialize a datapoint metadata container with an ID.

        Parameters
        ----------
        id : int
            Unique identifier for the datapoint.
        metadata : Dict[str, Any], optional
            Dictionary of metadata key-value pairs, by default None.
        """
        self.id = id
        self.metadata = metadata or {}

    def add_metadata(self, key: str, value: Any) -> None:
        """Add or update a metadata value.

        Parameters
        ----------
        key : str
            Metadata key to add or update.
        value : Any
            Value to associate with the key.
        """
        self.metadata[key] = value

    def get_metadata(self, key: str) -> Any:
        """Get metadata value.

        Parameters
        ----------
        key : str
            Metadata key to retrieve.

        Returns
        -------
        Any
            The metadata value.
        """
        return self.metadata[key]
