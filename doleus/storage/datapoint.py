from typing import Any, Dict


class Datapoint:
    """Container for a datapoint and its metadata.

    Datapoints are used to store data samples along with their associated metadata.
    """

    def __init__(self, id: int, metadata: Dict[str, Any] = None):
        """Initialize a datapoint with an ID and optional metadata.

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

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value with optional default.

        Parameters
        ----------
        key : str
            Metadata key to retrieve.
        default : Any, optional
            Default value to return if key is not found, by default None.

        Returns
        -------
        Any
            The metadata value or default if not found.
        """
        return self.metadata.get(key, default)

    def has_metadata(self, key: str) -> bool:
        """Check if metadata key exists.

        Parameters
        ----------
        key : str
            Metadata key to check.

        Returns
        -------
        bool
            True if the key exists in the metadata, False otherwise.
        """
        return key in self.metadata
