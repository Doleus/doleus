from typing import Any, Dict, List, Optional


class MetadataStore:
    """Container for storing metadata for a dataset.

    MetadataStore objects store metadata associated with datapoints in a dataset.
    Each datapoint's metadata is stored as a dictionary at the corresponding index.
    """

    def __init__(
        self, num_datapoints: int, metadata: Optional[List[Dict[str, Any]]] = None
    ):
        """Initialize a metadata container.

        Parameters
        ----------
        num_datapoints : int
            Number of datapoints to initialize the metadata list with.
        metadata : Optional[List[Dict[str, Any]]], optional
            List of metadata dictionaries, one per datapoint, by default None.
        """
        if metadata:
            if len(metadata) != num_datapoints:
                raise ValueError(
                    f"Metadata list has {len(metadata)} entries but dataset has {num_datapoints} datapoints"
                )
            self.metadata = metadata
        else:
            self.metadata = [{} for _ in range(num_datapoints)]

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

        """
        return self.metadata[datapoint_idx][key]

    def get_subset(self, indices: List[int]) -> List[Dict[str, Any]]:
        """Get a subset of metadata based on indices.

        Parameters
        ----------
        indices : List[int]
            List of indices to get metadata for.

        Returns
        -------
        List[Dict[str, Any]]
            List of metadata dictionaries for the specified indices.
        """
        return [self.metadata[i] for i in indices]
