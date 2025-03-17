from typing import List


class Annotation:
    """Base annotation class for storing datapoint identifiers."""

    def __init__(self, datapoint_number: int):
        """Initialize an Annotation instance.

        Parameters
        ----------
        datapoint_number : int
            ID corresponding to a sample in the dataset.
        """
        self.datapoint_number = datapoint_number


class Annotations:
    """Container for managing annotation objects.

    This class provides a container for Labels or BoundingBoxes annotations.
    It can be used to store both predictions and ground truths.
    """

    def __init__(self, annotations: List[Annotation] = None):
        """Initialize an Annotations container.

        Parameters
        ----------
        annotations : List[Annotation], optional
            Initial list of annotation objects to store, by default None.
        """
        self.annotations = annotations if annotations is not None else []
        self.datapoint_number_to_annotation_index = {}
        for idx, annotation in enumerate(self.annotations):
            dp_num = annotation.datapoint_number
            self.datapoint_number_to_annotation_index[dp_num] = idx

    def add(self, annotation: Annotation):
        """Add a new annotation.

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
        """Get annotation by datapoint number.

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
