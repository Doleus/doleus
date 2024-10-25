import pickle
import random
import json
import pydicom
from pathlib import Path
import pandas as pd
import random
from typing import List
import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset

from moonwatcher.dataset.dataset import MoonwatcherDataset
from moonwatcher.model.model import MoonwatcherModel
from moonwatcher.utils.data import TaskType, Task
from moonwatcher.check import Check, CheckSuite

# Variables
EXAMS_PATH = "/Users/hendrik/Studium/Master/Thesis/Data/physionet.org/files/vindr-mammo/1.0.0/images"
BREAST_LEVEL_ANNOTATIONS_PATH = "/Users/hendrik/Studium/Master/Thesis/Data/physionet.org/files/vindr-mammo/1.0.0/breast-level_annotations.csv"


"""
A function that combines a BI-RADS score ['BI-RADS 1', 'BI-RADS 2', 'BI-RADS 3', 'BI-RADS 4','BI-RADS 5'] and a view ['LL-C', 'R-CC', 'L-MLO', 'R-MLO'] to create a label
The left and right view should be combined to create one label
The labels should be in the following format:
"""


# Utility functions
def get_img_view(img_filename, annotations):
    """
    Find image_id ("id") laterality ("R" or "L") and view_position ("CC" or "MLO")
    :param img_filename: image filename without extension
    :param annotations: DataFrame containing the annotations
    :return: view in the format "laterality-view_position"
    """
    try:
        row = annotations.loc[annotations['image_id'] ==
                              img_filename, ["laterality", "view_position"]]
        laterality = row['laterality'].values[0]
        view_position = row['view_position'].values[0]
        return f"{laterality}-{view_position}"
    except IndexError:
        print(f"Annotations for image {img_filename} not found.")
        return None
    except Exception as e:
        print(f"Error while getting view for {img_filename}: {e}")
        return None


def create_pkl_exam_list(exams_folder_path, breast_level_annotations_file_path):
    """
    Create a pickle file containing a list of exam dictionaries from a folder of DICOM images and a CSV of breast level annotations.

    This function reads DICOM images from exam folders and creates a dictionary for each exam. Each dictionary contains keys
    for different views of the exam and lists of filenames for those views. The function also reads breast level annotations
    from a CSV file to determine the view of each image. The resulting list of exam dictionaries is then saved as a pickle file.

    Args:
        exams_folder_path (str or Path): The path to the folder containing exam subdirectories. Each subdirectory represents an
                                         individual exam and contains DICOM image files.
        breast_level_annotations_file_path (str or Path): The path to the CSV file containing breast level annotations.
                                                           The CSV file should have columns indicating the filename of
                                                           the DICOM images and their corresponding view.

    Returns:
        None: The function does not return any value. It saves a pickle file named "exam_list_before_cropping.pkl"
              in the current working directory.

    Raises:
        ValueError: If the view cannot be determined for an image or if the view is unrecognized.
        Exception: If there is an error reading the annotations file or saving the pickle file.
    """
    # Create pathlib folder and check if folder exists.
    exams_folder_path = Path(exams_folder_path)
    if not exams_folder_path.exists():
        print(f"Couldn't find exams folder path: {exams_folder_path}")
        return
    if not Path(breast_level_annotations_file_path).exists():
        print(f"Couldn't find breast level annotations file path: {
              breast_level_annotations_file_path}")
        return

    # Create a pandas dataframe from the breast level annotations csv file
    try:
        breast_level_annotations = pd.read_csv(
            breast_level_annotations_file_path)
    except Exception as e:
        print(f"Error reading annotations file: {e}")
        return

    # Create pickle file
    exams_list = []
    for exam in exams_folder_path.iterdir():
        # if exam is not a folder, skip
        if not (exam.is_dir()):
            continue
        # Dict for exam. Structure according to breast cancer classifier sample
        exam_dict = {
            'horizontal_flip': 'NO',
            'L-CC': [],
            'L-MLO': [],
            'R-MLO': [],
            'R-CC': []
        }

        # Iterate over images in exam folder
        for img in exam.glob('*.dicom'):
            img_filename = img.stem
            view = get_img_view(img_filename, breast_level_annotations)

            # Raise an error if view is None
            if view is None:
                raise ValueError(f"Could not find view for {img_filename}")
            elif view in ['L-CC', 'L-MLO', 'R-MLO', 'R-CC']:
                exam_dict[view].append(img_filename)
            else:
                raise ValueError(f"Unrecognized view: {view}")

        # By now the exam dictionary should have been created and we can add it to the exam list
        exams_list.append(exam_dict)

    # Save exam list
    try:
        with open("exam_list_before_cropping.pkl", 'wb') as f:
            pickle.dump(exams_list, f)
        print("Exam list created successfully.")
    except Exception as e:
        print(f"Error while saving pickle file: {e}")


def convert_annotations_to_prediction_format(annotations: List[str]) -> List[int]:
    """
    Convert BI-RADS annotations to a binary prediction format.

    The function takes a list of four BI-RADS scores (two for each breast) and converts them
    into a binary format representing the presence or absence of benign and malignant findings.

    Input:
    - annotations: List of four BI-RADS scores as strings, in the order:
      [left CC, left MLO, right CC, right MLO]
      Each score is one of: 'BI-RADS 1', 'BI-RADS 2', 'BI-RADS 3', 'BI-RADS 4', 'BI-RADS 5'

    Output:
    - A list of four binary values (0 or 1) in the order:
      [left benign, right benign, left malignant, right malignant]

    Mapping:
    - BI-RADS 1: Considered negative for both benign and malignant (no effect on output)
    - BI-RADS 2 or 3: Mapped to benign (1 for benign, 0 for malignant)
    - BI-RADS 4 or 5: Mapped to malignant (0 for benign, 1 for malignant)

    For each breast, the highest BI-RADS score between CC and MLO views is used for the mapping.

    Example:
    Input: ['BI-RADS 2', 'BI-RADS 1', 'BI-RADS 4', 'BI-RADS 3']
    Output: [1, 0, 0, 1]
    Explanation: Left breast benign (BI-RADS 2), Right breast malignant (BI-RADS 4)
    """

    # Map each birad score to an int
    birads_to_int = {
        'BI-RADS 1': 1,
        'BI-RADS 2': 2,
        'BI-RADS 3': 3,
        'BI-RADS 4': 4,
        'BI-RADS 5': 5
    }

    # Convert BI-RADS scores to integers
    int_scores = [birads_to_int[score] for score in annotations]

    # Initialize the result list
    result = [0, 0, 0, 0]

    # Process left breast (first two scores in annotations)
    # BI-RADS 1 is mapped to 0, so it has no effect on the prediction
    left_max_score = max(int_scores[:2])
    if left_max_score in [2, 3]:
        result[0] = 1  # Left benign
    elif left_max_score in [4, 5]:
        result[2] = 1  # Left malignant

    # Process right breast (last two scores in annotations)
    # BI-RADS 1 is mapped to 0, so it has no effect on the prediction
    right_max_score = max(int_scores[2:])
    if right_max_score in [2, 3]:
        result[1] = 1  # Right benign
    elif right_max_score in [4, 5]:
        result[3] = 1  # Right malignant

    return result

# Create the VinDrMammo dataset class


class VinDrMammoDataset(Dataset):

    def __init__(self, exam_list_before_cropping_file, breast_level_annotations_file, exams_path,
                 transform=None):
        with open(exam_list_before_cropping_file, "rb") as f:
            self.exam_list_before_cropping = pickle.load(f)
        # with open(exam_labels_path, "rb") as f:
        #     self.exam_labels = pickle.load(f)
        self.breast_level_annotations = pd.read_csv(
            breast_level_annotations_file)
        # The images folder in the vinDrMamoDataset contains one folder for each exam. Each exam folder contains 4 images. One image per view.
        self.exams_path = exams_path
        self.transform = transform

    def __len__(self):
        return len(self.exam_list_before_cropping)

    def __getitem__(self, idx):
        exam = self.exam_list_before_cropping[idx]
        images = []
        labels = []
        annotations = self.breast_level_annotations

        for view in ['L-CC', 'R-CC', 'L-MLO', 'R-MLO']:
            if exam[view]:
                # The images path points to the image folder in the VinDrMamo Dataset
                # The exam id points to the folder for the exam that contains the 4 images
                # Each image is stored as a dicom file
                img_id = exam[view][0]
                # Find exam id that corresponds to the image id by searching in the annotations file
                exam_id_row = annotations[annotations['image_id']
                                          == img_id]["study_id"]
                assert len(
                    exam_id_row) == 1, f"There should only be one exam id for each image id {img_id} but there are {len(exam_id_row)}"
                exam_id = exam_id_row.iloc[0]

                img_path = Path(self.exams_path) / \
                    str(exam_id) / f"{img_id}.dicom"

                # Load dicom image
                dicom = pydicom.read_file(img_path)
                image = dicom.pixel_array

                # Apply transform if any
                if self.transform:
                    image = self.transform(image)
                images.append(image)

                # Get the label for the image
                # Select row with corresponding image id
                row = annotations.loc[annotations['image_id']
                                      == img_id, ["breast_birads"]]
                # Make sure that there's only one row because the img id should be unique
                assert len(row) == 1, f"Expecting one row for image id {
                    img_id}, because ids are unique"
                label = row["breast_birads"].iloc[0]

                # Check if label is birads score
                assert label in ['BI-RADS 1', 'BI-RADS 2', 'BI-RADS 3', 'BI-RADS 4',
                                 'BI-RADS 5'], f'expecting label to be bi-rads score to be between 1 and 5 but label is: {label}'

                # Add label to labels
                labels.append(label)

        # Convert annotations to match the prediction format of the breast cancer classifier
        labels = convert_annotations_to_prediction_format(labels)

        # Labels and images should have the same length
        assert len(labels) == len(images)

        return images, labels


# Create the exam list before cropping pickle file
create_pkl_exam_list(exams_folder_path=EXAMS_PATH,
                     breast_level_annotations_file_path=BREAST_LEVEL_ANNOTATIONS_PATH)

# Instantiate the dataset
dataset = VinDrMammoDataset(
    'exam_list_before_cropping.pkl',
    BREAST_LEVEL_ANNOTATIONS_PATH,
    EXAMS_PATH
)

# Load dummy predictions
# Generate dummy predictions for each item in the VinDrMammoDataset
dummy_predictions = []
for _ in range(dataset.__len__()):  # Using the dataset's length
    # Generate a random number between 0 and 1 for per class.
    # Class are left benign, right benign, left malignant, right malignant in this order
    prediction = [random.uniform(0, 1) for _ in range(4)]
    dummy_predictions.append(prediction)

# Now dummy_predictions contains predictions for each item in the dataset
print(f"Generated {len(dummy_predictions)} dummy predictions.")
print("Sample prediction:", dummy_predictions[0])

# Define dataset output transformation


def dataset_output_transform(datapoint):
    images, labels = datapoint
    # Labels is an array of 4 elements, each being either 0 or 1.
    # The elemnts are in the following order: [left benign, right benign, left malignant, right malignant]
    # Make sure that the label is a 1-d int tensor
    labels = torch.tensor(labels, dtype=torch.int64)

    return images, labels


# Create moonwatcher dataset
mw_dataset = MoonwatcherDataset(
    name="VinDrMammo",
    dataset=dataset,
    num_classes=4,
    task_type=TaskType.CLASSIFICATION.value,
    task=Task.MULTILABEL.value,
    output_transform=dataset_output_transform,
)

# Instantiate a model
dummy_model = MoonwatcherModel(
    name="DummyModel",
    task_type=TaskType.CLASSIFICATION.value,
    task=Task.MULTILABEL.value,
    predictions=dummy_predictions
    # TODO: Add prediction probs
)

# Accuracy check for the model
accuracy_check = Check(
    name="Accuracy",
    dataset_or_slice=mw_dataset,
    metric="Accuracy",
    operator=">",
    value=0.8,
    model=dummy_model
)

check_suite = CheckSuite(
    name="TestCheck",
    checks=[accuracy_check],
)

result = check_suite()
print(result)
