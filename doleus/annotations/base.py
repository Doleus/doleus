from typing import List


class Annotation:
    """Base annotation class."""

    def __init__(self, datapoint_number: int):
        """Initialize an Annotation instance.

        Parameters
        ----------
        datapoint_number : int
            ID corresponding to a sample in the dataset.
        """
        self.datapoint_number = datapoint_number


class Annotations:
    """Container for annotation objects.

    This class provides a container for Labels or BoundingBoxes annotations.
    It is used as a base class for both Labels and BoundingBoxes.
    """

    def __init__(self, annotations: List[Annotation] = None):
        """Initialize an Annotations container.

        Parameters
        ----------
        annotations : List[Annotation], optional
            List of annotation objects to store.
        """
        self.annotations = annotations or []
        self.datapoint_number_to_annotation_index = {}
        for idx, annotation in enumerate(self.annotations):
            dp_num = annotation.datapoint_number
            self.datapoint_number_to_annotation_index[dp_num] = idx

    def __getitem__(self, datapoint_number: int) -> Annotation:
        """Get annotation by datapoint number.

        Parameters
        ----------
        datapoint_number : int
            The ID of the sample in the dataset.

        Returns
        -------
        Annotation
            The annotation corresponding to the datapoint number.
        """
        index = self.datapoint_number_to_annotation_index.get(datapoint_number)
        if index is None:
            raise KeyError(
                f"No annotation found for datapoint_number: {datapoint_number}"
            )
        return self.annotations[index]

    def add(self, annotation: Annotation):
        """Add a new annotation.

        Parameters
        ----------
        annotation : Annotation
            An annotation of type Labels or BoundingBoxes.
        """
        dp_num = annotation.datapoint_number
        self.annotations.append(annotation)
        self.datapoint_number_to_annotation_index[dp_num] = len(self.annotations) - 1
