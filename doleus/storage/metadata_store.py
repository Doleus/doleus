from typing import Any, Dict, List


class MetadataStore:
    """Container for dataset metadata.

    MetadataStore objects store metadata associated with datapoints in a dataset.
    Each datapoint's metadata is stored as a dictionary at the corresponding index.
    """

    def __init__(self, metadata: List[Dict[str, Any]] = None):
        """Initialize a metadata container.

        Parameters
        ----------
        metadata : List[Dict[str, Any]], optional
            List of metadata dictionaries, one per datapoint, by default None.
        """
        self.metadata: List[Dict[str, Any]] = metadata or []

    def add_metadata(self, datapoint_idx: int, key: str, value: Any) -> None:
        """Add or update a metadata value for a specific datapoint.

        Parameters
        ----------
        datapoint_idx : int
            Index of the datapoint to add metadata for.
        key : str
            Metadata key to add or update.
        value : Any
            Value to associate with the key.
        """
        while len(self.metadata) <= datapoint_idx:
            self.metadata.append({})
        self.metadata[datapoint_idx][key] = value

    def get_metadata(self, datapoint_idx: int, key: str) -> Any:
        """Get metadata value for a specific datapoint.

        Parameters
        ----------
        datapoint_idx : int
            Index of the datapoint to get metadata for.
        key : str
            Metadata key to retrieve.

        Returns
        -------
        Any
            The metadata value.

        Raises
        ------
        KeyError
            If the metadata key doesn't exist for the datapoint.
        IndexError
            If the datapoint index is out of range.
        """
        return self.metadata[datapoint_idx][key]

    def get_all_metadata(self, datapoint_idx: int) -> Dict[str, Any]:
        """Get all metadata for a specific datapoint.

        Parameters
        ----------
        datapoint_idx : int
            Index of the datapoint to get metadata for.

        Returns
        -------
        Dict[str, Any]
            Dictionary of all metadata for the datapoint.

        Raises
        ------
        IndexError
            If the datapoint index is out of range.
        """
        return self.metadata[datapoint_idx]
