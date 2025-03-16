"""Base annotation classes for handling model predictions and ground truths."""

from typing import List


class Annotation:
    """Base annotation class for storing datapoint identifiers.

    This class serves as a base for both classification and detection annotations,
    providing a unified interface for datapoint identification.
    """

    def __init__(self, datapoint_number: int):
        """Initialize an Annotation instance.

        Parameters
        ----------
        datapoint_number : int
            Integer index or ID corresponding to a sample in the dataset.
        """
        self.datapoint_number = datapoint_number


class Annotations:
    """Container for managing annotation objects.

    This class provides a container for Labels or BoundingBoxes annotations,
    allowing indexing by datapoint number and iteration over all annotations.
    It can be used to store both predictions and ground truths.
    """

    def __init__(self, dataset=None, annotations: List[Annotation] = None):
        """Initialize an Annotations container.

        Parameters
        ----------
        dataset : optional
            Reference to the dataset (e.g., Doleus object), by default None.
        annotations : List[Annotation], optional
            Initial list of annotation objects to store, by default None.
        """
        self.dataset = dataset
        self.annotations = annotations if annotations is not None else []
        # Map from datapoint_number -> index in self.annotations
        self.datapoint_number_to_annotation_index = {}
        for idx, annotation in enumerate(self.annotations):
            dp_num = annotation.datapoint_number
            self.datapoint_number_to_annotation_index[dp_num] = idx

    def add(self, annotation: Annotation):
        """Add a new annotation to the container.

        Parameters
        ----------
        annotation : Annotation
            An annotation of type Labels or BoundingBoxes.

        Raises
        ------
        TypeError
            If annotation is not an instance of the base Annotation class.
        KeyError
            If an annotation for the datapoint already exists.
        """
        if not isinstance(annotation, Annotation):
            raise TypeError(
                "annotation must be an instance of the base Annotation class."
            )

        dp_num = annotation.datapoint_number
        if dp_num in self.datapoint_number_to_annotation_index:
            raise KeyError(f"Annotation for datapoint {dp_num} already exists.")

        self.annotations.append(annotation)
        self.datapoint_number_to_annotation_index[dp_num] = len(self.annotations) - 1

    def get(self, datapoint_number: int) -> Annotation:
        """Retrieve the annotation object for a given datapoint.

        Parameters
        ----------
        datapoint_number : int
            The ID of the sample in the dataset.

        Returns
        -------
        Annotation
            The annotation (Labels or BoundingBoxes) for the datapoint.

        Raises
        ------
        KeyError
            If no annotation is found for the datapoint number.
        """
        index = self.datapoint_number_to_annotation_index.get(datapoint_number)
        if index is None:
            raise KeyError(f"No annotation found for datapoint {datapoint_number}.")
        return self.annotations[index]

    def get_datapoint_ids(self) -> List[int]:
        """Get all datapoint IDs with annotations.

        Returns
        -------
        List[int]
            A list of all datapoint IDs (keys) for which annotations exist.
        """
        return list(self.datapoint_number_to_annotation_index.keys())

    def __getitem__(self, datapoint_number: int) -> Annotation:
        """Get annotation by datapoint number using indexing syntax.

        Parameters
        ----------
        datapoint_number : int
            The ID of the sample in the dataset.

        Returns
        -------
        Annotation
            The annotation for the datapoint.
        """
        return self.get(datapoint_number)

    def __len__(self) -> int:
        """Get the number of annotations.

        Returns
        -------
        int
            The total number of annotations in the container.
        """
        return len(self.annotations)

    def __iter__(self):
        """Get an iterator over all annotations.

        Returns
        -------
        iterator
            Iterator over all annotations in the container.
        """
        return iter(self.annotations)
