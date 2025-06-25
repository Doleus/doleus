<div align="center">
  <img alt="Doleus Logo" src="Doleus_Logo.png" width="400">
</div>
<h1 align="center" weight='300'>Doleus: Test Your Image-based AI Models on Data Slices</h1>

<div align="center">

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/doleus/doleus/blob/main/LICENSE)
[![Doleus on Discord](https://img.shields.io/discord/1230407128842506251?label=Discord)](https://discord.gg/B8SzfbGRA9)

</div>

## Table of Contents

- [What is Doleus?](#what-is-doleus)
- [Quick Start](#quick-start)
- [Why It Matters: Real-World Examples](#why-it-matters-real-world-examples)
- [Core Concepts](#core-concepts)
- [Prediction Format Requirements](#prediction-format-requirements)
- [Tips](#tips)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## What is Doleus?

Doleus is a PyTorch-based testing framework for image-based AI models. It helps you understand how your models perform on different subsets of your data, allowing you to quantify performance gaps and identify failure modes that aggregate metrics miss.

**The workflow is simple:**

1. Add metadata to your dataset (patient demographics, weather conditions, manufacturing specs, etc.)
2. Create slices of your dataset based on this metadata (e.g., weather = sunny, weather = cloudy, weather = foggy)
3. Run tests on these slices to find performance gaps (e.g., model accuracy drops from 95% in sunny conditions to 73% in foggy conditions)

This approach surfaces hidden failure modes that aggregate metrics miss.

> [!NOTE]
> **Task Types**: Doleus works reliably for object detection and classification tasks. If you work on different tasks and would like to see them implemented, please submit a feature request or start contributing yourself🤗 .

## Quick Start (Classification)

```sh
pip install git+https://github.com/doleus/doleus.git
```
### Try the demo to see how Doleus works end to end
Want to try a complete working example before diving into the details?  
Run [`examples/demos/demo_classification.py`](examples/demos/demo_classification.py) to see the full workflow in action.

### Demo

```python
from doleus.datasets import DoleusClassification
from doleus.checks import Check, CheckSuite

# Wrap your PyTorch dataset
doleus_dataset = DoleusClassification(
    name="product_inspection",
    dataset=your_pytorch_dataset,
    task="multiclass",
    num_classes=5  # defect types
)

# Add your domain-specific metadata
# You can add metadata from a list, from a dataframe, from a custom function applied to each image in the dataset and from our pre-defined metadata functions
metadata_list = [
    {"surface_type": "matte", "lighting": "bright", "defect_size_mm": 0.8},
    {"surface_type": "glossy", "lighting": "dim", "defect_size_mm": 2.1},
    # ... one dict per image
]
doleus_dataset.add_metadata_from_list(metadata_list)

# Add model predictions
doleus_dataset.add_model_predictions(predictions, model_id="v1")

# Create slice and test
glossy_surface = doleus_dataset.slice_by_value("surface_type", "==", "glossy")

check = Check(
    name="glossy_surface_accuracy",
    dataset=glossy_surface,
    model_id="v1",
    metric="Accuracy",
    operator=">",
    value=0.95
)

# Run test
result = check.run(show=True)
```

**Output:**

```
❌ glossy_surface_accuracy           0.87 > 0.95    (Accuracy on product_surface_type_eq_glossy)
```

> [!TIP]
> **Storing Results**: You can save check results to JSON files by setting `save_report=True`:
>
> ```python
> result = check.run(show=True, save_report=True)
> # Creates: check_glossy_surface_accuracy_report.json
> ```

> [!TIP]
> **Multiple Model Predictions**: You can add predictions from different model versions to the same dataset:
>
> ```python
> doleus_dataset.add_model_predictions(predictions_v1, model_id="model_v1")
> doleus_dataset.add_model_predictions(predictions_v2, model_id="model_v2")
> # Now you can test both models on the same slices
> ```

> [!IMPORTANT]
> **Prediction Inheritance**: Add predictions to your dataset **before** creating slices. Slices automatically inherit predictions from their parent dataset, but only if the predictions exist when the slice is created.

> [!TIP]
> **Ways to add metadata**: Doleus offers a variety of ways to add metadata to your data set. Find all supported functions in `doleus.dataset.base.py` under "METADATA FUNCTIONS":

> [!TIP]
> **Available Metrics**: Find all supported metrics in `doleus.metrics.METRIC_FUNCTIONS`. Common ones include:
>
> - Classification: `Accuracy`, `Precision`, `Recall`, `F1_Score`
> - Detection: `mAP`, `IntersectionOverUnion`, `CompleteIntersectionOverUnion`

## Why It Matters: Real-World Examples

<details open>
<summary>
<b>Medical Imaging</b> - Ensure your model works across all patient demographics
</summary> <br />

```python
# Problem: Your mammography AI performs well overall but might fail silently on dense breast tissue or for young patients.
# Solution: Test performance across breast density categories and age.

# Add metadata from your medical annotations
metadata_list = [
    {"patient_age": 45, "breast_density": 4, "scanner": "GE_Senographe"},
    {"patient_age": 52, "breast_density": 2, "scanner": "Hologic_3D"},
    # ... one dict per image
]
doleus_dataset.add_metadata_from_list(metadata_list)

# Create test suite for high-risk categories
dense_tissue = doleus_dataset.slice_by_value("breast_density", ">=", 3)
older_patients = doleus_dataset.slice_by_value("patient_age", "<=", 45)

suite = CheckSuite(name="mammography_safety", checks=[
    Check("dense_tissue_sensitivity", dense_tissue, "model_v2", "Recall", ">", 0.95),
    Check("younger_patient_accuracy", older_patients, "model_v2", "Accuracy", ">", 0.90),
])
results = suite.run_all(show=True)
```

**Output:**

```
❌ mammography_safety
    ❌ dense_tissue_sensitivity           0.82 > 0.95    (Recall on mammo_breast_density_ge_3)
    ✅ younger_patient_accuracy             0.91 > 0.90    (Accuracy on mammo_patient_age_le_45)
```

**Finding:** Model underperforms on dense breast tissue but does not on younger patients.

</details>

<details>
<summary>
<b>Autonomous Driving</b> - Test model performance across varying weather conditions
</summary> <br />

```python
# Problem: Your object detection model misses pedestrians in foggy conditions
# Solution: Test detection performance across weather and visibility conditions

# Add weather and visibility metadata
doleus_dataset.add_metadata("weather_condition", detect_weather_condition)  # Your weather detection function
doleus_dataset.add_metadata("visibility_meters", estimate_visibility_distance)  # Visibility estimation

# Test safety-critical scenarios
foggy_weather = doleus_dataset.slice_by_value("weather_condition", "==", "fog")
low_visibility = doleus_dataset.slice_by_value("visibility_meters", "<", 50)

suite = CheckSuite(name="weather_safety", checks=[
    Check("fog_pedestrian_detection", foggy_weather, "model_v3", "Recall", ">", 0.90),
    Check("low_visibility_detection", low_visibility, "model_v3", "mAP", ">", 0.85),
])
results = suite.run_all(show=True)
```

**Output:**

```
❌ weather_safety
    ❌ fog_pedestrian_detection           0.73 > 0.90    (Recall on driving_weather_condition_eq_fog)
    ❌ low_visibility_detection           0.81 > 0.85    (mAP on driving_visibility_meters_lt_50)
```

**Finding:** Model dangerously underperforms in fog. Might need additional training data.

</details>

<details>
<summary>
<b>Manufacturing Quality Control</b> - Catch defects across specific product variations
</summary> <br />

```python
# Problem: Tiny scratches on reflective aluminum surfaces go undetected
# Solution: Test defect detection across material types and defect sizes

# Add production metadata
import pandas as pd
production_metadata = pd.DataFrame({
    "material": ["aluminum", "steel", "plastic", ...],
    "surface_reflectivity": [0.95, 0.60, 0.20, ...],  # 0-1 scale
    "defect_type": ["scratch", "dent", "discoloration", ...],
    "defect_area_mm2": [0.5, 2.1, 0.3, ...]
})
doleus_dataset.add_metadata_from_dataframe(production_metadata)

# Test challenging conditions
reflective_aluminum = doleus_dataset.slice_by_value("material", "==", "aluminum")
tiny_defects = doleus_dataset.slice_by_percentile("defect_area_mm2", "<=", 10) # Smallest 10% of defects

suite = CheckSuite(name="quality_assurance", checks=[
    Check("reflective_aluminum_detection", reflective_aluminum, "qc_model", "Precision", ">", 0.98),
    Check("tiny_defect_detection", tiny_defects, "qc_model", "Recall", ">", 0.95),
])
results = suite.run_all(show=True)
```

**Output:**

```
❌ quality_assurance
    ❌ reflective_aluminum_detection      0.91 > 0.98    (Precision on product_material_eq_aluminum)
    ❌ tiny_defect_detection             0.88 > 0.95    (Recall on product_defect_area_mm2_le_10)
```

**Finding:** Reflective surfaces and tiny defects cause false positives. Consider updating product guidelines for defect detection.

</details>

<details>
<summary>
<b>Security & Surveillance</b> - Verify face recognition works in challenging real-world conditions
</summary> <br />

```python
# Problem: Face recognition fails for people wearing masks at oblique angles
# Solution: Test recognition across face occlusions and camera angles

# Add surveillance metadata
doleus_dataset.add_metadata("face_occlusion_percent", detect_face_occlusion)  # % of face covered
doleus_dataset.add_metadata("camera_angle_degrees", estimate_camera_angle)  # Angle from frontal
doleus_dataset.add_metadata("lighting_lux", measure_scene_brightness)  # Light level in lux

# Test real-world scenarios
masked_faces = doleus_dataset.slice_by_value("face_occlusion_percent", ">", 50)
oblique_angles = doleus_dataset.slice_by_value("camera_angle_degrees", ">", 45)
low_light = doleus_dataset.slice_by_value("lighting_lux", "<", 50)

suite = CheckSuite(name="surveillance_reliability", checks=[
    Check("masked_face_recognition", masked_faces, "face_model_v2", "Top5_Accuracy", ">", 0.85),
    Check("oblique_angle_recognition", oblique_angles, "face_model_v2", "Top1_Accuracy", ">", 0.75),
    Check("low_light_recognition", low_light, "face_model_v2", "Top5_Accuracy", ">", 0.80),
])
results = suite.run_all(show=True)
```

**Output:**

```
❌ surveillance_reliability
    ❌ masked_face_recognition            0.72 > 0.85    (Top5_Accuracy on surveillance_face_occlusion_percent_gt_50)
    ✅ oblique_angle_recognition          0.78 > 0.75    (Top1_Accuracy on surveillance_camera_angle_degrees_gt_45)
    ❌ low_light_recognition              0.69 > 0.80    (Top5_Accuracy on surveillance_lighting_lux_lt_50)
```

**Finding:** System unreliable for masked individuals and in low light conditions. Note that each scenario is tested separately - combining multiple conditions (e.g., masked faces in low light) would require creating additional slices for those specific combinations.

</details>

<details>
<summary>
<b>Agriculture & Food Safety</b> - Detect crop diseases across varying field conditions
</summary> <br />

```python
# Problem: Disease detection AI misses early-stage infections in drought-stressed crops
# Solution: Test disease detection across crop stress levels and disease stages

# Add agricultural metadata
import pandas as pd
field_metadata = pd.DataFrame({
    "crop_moisture_stress": ["none", "mild", "severe", ...],  # From NDVI/sensors
    "disease_stage": ["early", "mid", "late", ...],
    "leaf_coverage_percent": [95, 70, 85, ...],  # How much of image shows leaves
    "shadow_percent": [10, 45, 5, ...]  # Shadow coverage in image
})
doleus_dataset.add_metadata_from_dataframe(field_metadata)

# Test critical scenarios separately
stressed_crops = doleus_dataset.slice_by_value("crop_moisture_stress", "==", "severe")
early_disease = doleus_dataset.slice_by_value("disease_stage", "==", "early")
shadowed_crops = doleus_dataset.slice_by_value("shadow_percent", ">", 30)

suite = CheckSuite(name="crop_disease_detection", checks=[
    Check("stressed_crop_detection", stressed_crops, "disease_model", "Recall", ">", 0.90),
    Check("early_disease_detection", early_disease, "disease_model", "Recall", ">", 0.85),
    Check("shadowed_area_detection", shadowed_crops, "disease_model", "F1", ">", 0.85),
])
results = suite.run_all(show=True)
```

**Output:**

```
❌ crop_disease_detection
    ❌ stressed_crop_detection            0.76 > 0.90    (Recall on crops_crop_moisture_stress_eq_severe)
    ❌ early_disease_detection            0.72 > 0.85    (Recall on crops_disease_stage_eq_early)
    ✅ shadowed_area_detection            0.87 > 0.85    (F1 on crops_shadow_percent_gt_30)
```

**Finding:** Early disease detection fails in drought-stressed crops - critical for preventive treatment

</details>

## Core Concepts

### **Metadata**

Attributes you add to your dataset:

- **Custom**: Any domain-specific attributes (patient age, weather conditions, defect sizes)
- **Predefined**: `brightness`, `contrast`, `saturation`, `resolution` (auto-computed)

### **Slices**

Subsets of your data filtered by metadata:

- `slice_by_percentile("defect_area_mm2", "<=", 10)` → Smallest 10% of defects
- `slice_by_value("weather_condition", "==", "fog")` → Only foggy conditions
- `slice_by_groundtruth_class(class_names=["pedestrian", "cyclist"])` → Specific object classes

> [!NOTE]
> **Slicing Method**: Use `slice_by_value("metadata_key", "==", "value")` for categorical filtering. In theory, all comparison operators are supported: `>`, `<`, `>=`, `<=`, `==`, `!=`.

### **Checks**

Tests that compute metrics on slices:

- **Pass/fail tests**: `Check("test_name", slice, "model_id", "Accuracy", ">", 0.9)`
- **Evaluation only**: `Check("test_name", slice, "model_id", "mAP")`

Checks become tests when you add pass/fail conditions (operator and value). Without these conditions, checks simply evaluate and report metric values.

> [!NOTE]
> **Prediction Format**: Doleus uses [torchmetrics](https://torchmetrics.readthedocs.io/) to compute metrics and expects the same prediction formats that torchmetrics functions require.

> [!IMPORTANT]
> **Macro Averaging Default**: Doleus uses **macro averaging** as the default for classification metrics (Accuracy, Precision, Recall, F1) to avoid known bugs in torchmetrics' micro averaging implementation (see [GitHub issue #2280](https://github.com/Lightning-AI/torchmetrics/issues/2280)).
>
> You can override this by setting `metric_parameters={"average": "micro"}` in your checks if needed.

### **CheckSuites**

Groups of related checks that run together:

- Organize tests by concern (safety, accuracy, edge cases)
- Run all checks and get a summary report

> [!NOTE]
> **Aggregation Logic**: A CheckSuite succeeds if **no individual check fails**. Checks without pass/fail criteria (info-only) don't affect the suite's success status.

## Prediction Format Requirements

Doleus interprets model predictions differently based on data type and task. Understanding this behavior is crucial for correct metric computation.

### **Classification Tasks**

**Binary Classification:**

```python
# Integer predictions → stored as class labels (0 or 1)
predictions = torch.tensor([0, 1, 0, 1], dtype=torch.long)

# Float predictions → stored as scores for positive class
predictions = torch.tensor([0.2, 0.8, 0.3, 0.9], dtype=torch.float32)
# Note: [0.0, 1.0] would be treated as scores, not labels
```

**Multiclass Classification:**

```python
# 1D integer → class indices (labels)
predictions = torch.tensor([0, 1, 2, 0], dtype=torch.long)

# 2D float → logits/probabilities (converted to labels via argmax)
predictions = torch.tensor([
    [0.8, 0.1, 0.1],  # Class 0
    [0.2, 0.7, 0.1],  # Class 1
], dtype=torch.float32)
```

**Multilabel Classification:**

```python
# 2D integer → multi-hot encoding (labels)
predictions = torch.tensor([
    [1, 0, 1],  # Labels 0 and 2 active
    [0, 1, 1],  # Labels 1 and 2 active
], dtype=torch.long)

# 2D float → probabilities/logits (scores)
predictions = torch.tensor([
    [0.9, 0.1, 0.8],  # Probabilities for each label
    [0.2, 0.7, 0.9],
], dtype=torch.float32)
```

### **Detection Tasks**

Detection predictions use a list of dictionaries format:

```python
predictions = [
    {
        "boxes": [[x1, y1, x2, y2], ...],      # Bounding boxes
        "labels": [class_id1, class_id2, ...], # Class IDs
        "scores": [conf1, conf2, ...]          # Confidence scores
    },
    # ... one dict per image
]
```

### **Threshold Control in Checks**

For float predictions (scores/probabilities), use `metric_parameters` to control thresholding:

```python
# Control binary classification threshold
check = Check(
    name="high_threshold_test",
    dataset=my_slice,
    model_id="model_v1",
    metric="Accuracy",
    metric_parameters={"threshold": 0.8}  # Passed to torchmetrics
)

# Control multiclass top-k accuracy
check = Check(
    name="top3_accuracy",
    dataset=my_slice,
    model_id="model_v1",
    metric="Accuracy",
    metric_parameters={"top_k": 3}
)
```

> [!IMPORTANT]
> **Data Type Matters**: The distinction between integer and float predictions determines how Doleus processes your data:
>
> - **Integer tensors** → Treated as final class decisions (labels)
> - **Float tensors** → Treated as scores/probabilities that may need thresholding
>
> This means `torch.tensor([0.0, 1.0])` is treated as **scores**, not labels. Cast to integer if you intend them as class labels: `torch.tensor([0, 1], dtype=torch.long)`.

## Tips

> [!IMPORTANT]
> **Order Matters**: Always add predictions to your dataset **before** creating slices. Slices inherit predictions from their parent dataset only at creation time.

```python
# ✅ Correct order
doleus_dataset.add_model_predictions(predictions, model_id="model_v1")
high_quality_slice = doleus_dataset.slice_by_value("quality", "==", "high")

# ❌ Wrong order - slice won't have predictions
high_quality_slice = doleus_dataset.slice_by_value("quality", "==", "high")
doleus_dataset.add_model_predictions(predictions, model_id="model_v1")
```

> [!TIP]
> **Finding Available Metrics**:
>
> ```python
> from doleus.metrics import METRIC_FUNCTIONS
> print(list(METRIC_FUNCTIONS.keys()))
> # ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'mAP', ...]
> ```

> [!CAUTION]
> **Task-Metric Compatibility**: Not all metrics work with all task types. Use classification metrics (`Accuracy`, `F1_Score`) with classification datasets and detection metrics (`mAP`, `IntersectionOverUnion`) with detection datasets.

## Examples

- **[Image Classification](examples/demos/demo_classification.py)** - Test classification models

## Contributing

We welcome contributions! 🎉

**Quick Links:**

- 📖 **[Full Contributing Guide](CONTRIBUTING.md)** - Complete guidelines and setup instructions
- 🐛 **[Report a Bug](https://github.com/doleus/doleus/issues/new?labels=bug)**
- 💡 **[Request a Feature](https://github.com/doleus/doleus/issues/new?labels=feature-request)**
- 💬 **[Join our Discord](https://discord.gg/B8SzfbGRA9)** - Get help and discuss ideas

For detailed setup instructions, development guidelines, and contribution workflow, see our **[Contributing Guide](CONTRIBUTING.md)**.

Not sure where to start? Join our [Discord](https://discord.gg/B8SzfbGRA9) and we'll help you get started!

## License

Apache 2.0. See [LICENSE](LICENSE).

---

Questions? Join our [Discord](https://discord.gg/B8SzfbGRA9) or open an issue.

---

Doleus is a successor to the Moonwatcher project: https://github.com/moonwatcher-ai/moonwatcher
