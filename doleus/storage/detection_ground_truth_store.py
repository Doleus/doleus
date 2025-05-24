import torch
from typing import Any

from doleus.storage.base_store import BaseGroundTruthStore
from doleus.annotations import Annotations
from doleus.annotations.detection import BoundingBoxes


class DetectionGroundTruthStore(BaseGroundTruthStore):
    """Ground truth store for detection tasks."""

    def __init__(self, dataset: Any):
        """
        Initialize the detection ground truth store.

        Parameters
        ----------
        dataset : Any
            The raw PyTorch dataset object.
        """
        super().__init__(dataset)  # This will call _process_groundtruths

    def _process_groundtruths(self) -> Annotations:
        """
        Process raw ground truth data from the dataset for detection tasks.

        Returns
        -------
        Annotations
            Processed ground truths in standard annotation format.
        
        Raises
        ------
        ValueError
            If ground truth data is in an invalid format.
        """
        processed_annotations = Annotations()

        for idx, data in enumerate(self.dataset):
            # Assuming standard (image, bounding_boxes, labels) structure for dataset items
            if not (isinstance(data, (list, tuple)) and len(data) == 3):
                raise ValueError(
                    f"Dataset item at index {idx} is not in the expected format (image, bounding_boxes, labels). "
                    f"Got {len(data)} elements of type: {type(data)}"
                )
            
            _, raw_boxes, raw_labels = data

            # Convert to tensors
            if not isinstance(raw_boxes, torch.Tensor):
                try:
                    bounding_boxes = torch.tensor(raw_boxes, dtype=torch.float32)
                except Exception as e:
                    raise ValueError(f"Could not convert bounding_boxes for item {idx} to tensor: {raw_boxes}. Error: {e}")
            else:
                bounding_boxes = raw_boxes.float() # Ensure correct dtype

            if not isinstance(raw_labels, torch.Tensor):
                try:
                    labels = torch.tensor(raw_labels, dtype=torch.long)
                except Exception as e:
                    raise ValueError(f"Could not convert labels for item {idx} to tensor: {raw_labels}. Error: {e}")
            else:
                labels = raw_labels.long() # Ensure correct dtype

            # Validate shapes
            # Assuming M is the number of detected objects for this datapoint
            # Bounding boxes should be (M, 4)
            if bounding_boxes.ndim != 2 or bounding_boxes.shape[1] != 4:
                raise ValueError(
                    f"Bounding boxes for item {idx} must have shape (M, 4). Got shape: {bounding_boxes.shape}"
                )
            
            # Labels should be (M,)
            num_detections = bounding_boxes.shape[0]
            if not (labels.ndim == 1 and labels.shape[0] == num_detections):
                raise ValueError(
                    f"Labels for item {idx} must have shape (M,). Got shape: {labels.shape}, expected M={num_detections}"
                )

            ann = BoundingBoxes(
                datapoint_number=idx, 
                boxes_xyxy=bounding_boxes, 
                labels=labels, 
                scores=None
            )
            processed_annotations.add(ann)

        return processed_annotations 