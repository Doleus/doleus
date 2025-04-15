from typing import List

from doleus.datasets.base import Doleus


class Slice:
    """A subset of a Doleus dataset containing only selected datapoints."""

    def __init__(
        self,
        name: str,
        root_dataset: Doleus,
        indices: List[int],
    ):
        """Initialize a Slice instance.

        Parameters
        ----------
        name : str
            Name of the slice.
        root_dataset : Doleus
            The parent dataset this slice is created from.
        indices : List[int]
            List of indices from the parent dataset to include in this slice.
        """
        self.name = name
        self.root_dataset = root_dataset
        self.indices = indices
        self.datapoints = [root_dataset.datapoints[i] for i in indices]

    def __len__(self) -> int:
        """Get the number of datapoints in the slice.

        Returns
        -------
        int
            Number of datapoints in the slice.
        """
        return len(self.indices)

    def __getitem__(self, idx: int):
        """Get a datapoint from the slice by index.

        Parameters
        ----------
        idx : int
            Index in the slice.

        Returns
        -------
        Any
            The datapoint from the parent dataset corresponding to the slice index.
        """
        root_idx = self.indices[idx]
        return self.root_dataset.dataset[root_idx]

    def __getattr__(self, attr: str):
        """Get an attribute from the parent dataset if not found in the slice.

        Parameters
        ----------
        attr : str
            Name of the attribute to get.

        Returns
        -------
        Any
            The attribute value from the parent dataset.
        """
        return getattr(self.root_dataset, attr)
