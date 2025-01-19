# check.py

import json
from pathlib import Path
from typing import Optional, List, Union, Dict, Any

import torch

from moonwatcher.utils.data import OPERATOR_DICT, TaskType
from moonwatcher.dataset.dataset import Moonwatcher, Slice
from moonwatcher.dataset.metadata import ATTRIBUTE_FUNCTIONS
from moonwatcher.metric import calculate_metric
from moonwatcher.utils.helpers import get_current_timestamp


def visualize_report(report: Dict[str, Any]):
    """
    Entry point for printing or formatting the check/checksuite report.
    """
    visualizer = ReportVisualizer()
    visualizer.visualize(report)


class ReportVisualizer:
    """
    Helper class to pretty-print check (and checksuite) reports to the console.
    """

    def __init__(self):
        self.pass_emoji = "\u2705"
        self.fail_emoji = "\u274C"
        self.BOLD = "\033[1m"
        self.END = "\033[0m"
        self.UNDERLINE = "\033[4m"
        self.RED = "\033[91m"
        self.GREEN = "\033[92m"

    def visualize(self, report: Dict[str, Any], spacing: str = ""):
        """
        Recursively print check or checksuite reports in a hierarchical manner.
        """
        if "checks" in report:
            # This is a CheckSuite
            self._print_checksuite(report, spacing)
        else:
            # This is a single Check
            self._print_check(report, spacing)

    def _print_checksuite(self, suite_report: Dict[str, Any], spacing: str):
        # Prepare line for top-level suite
        success_symbol = self.pass_emoji if suite_report["success"] else self.fail_emoji
        suite_name = suite_report["checksuite_name"]
        line_str = f"{spacing}{success_symbol} {suite_name}"
        print(line_str)

        # Indent further for sub-checks
        new_spacing = spacing + "    "
        for sub_report in suite_report["checks"]:
            self.visualize(sub_report, new_spacing)

    def _print_check(self, check_report: Dict[str, Any], spacing: str):
        # Decide success symbol (or blank if no thresholding)
        if check_report["success"] is not None:
            success_symbol = self.pass_emoji if check_report["success"] else self.fail_emoji
        else:
            success_symbol = "   "

        check_name = check_report["check_name"]
        line_str = f"{spacing}{success_symbol} {check_name}"

        # If there's a threshold test, show it
        operator = check_report["operator"]
        value = check_report["value"]
        result_val = check_report["result"]
        dataset_name = check_report["slice_name"] or check_report["dataset_name"]
        metric_name = check_report["metric"]

        if operator is not None and value is not None:
            # Pretty-print operator
            op_symbol = operator
            if op_symbol == ">=":
                op_symbol = "≥"
            elif op_symbol == "<=":
                op_symbol = "≤"
            elif op_symbol == "==":
                op_symbol = "="
            elif op_symbol == "!=":
                op_symbol = "≠"

            # Format success/fail color
            color = self.GREEN if check_report["success"] else self.RED
            # 5 decimal places
            result_str = f"{color}{result_val:.5f}{self.END}"
            comparison_str = f"{op_symbol} {value}"

            appendix = f"({metric_name} on {dataset_name})"
            print(
                f"{line_str.ljust(40)} {result_str} {comparison_str.ljust(10)} {appendix}"
            )
        else:
            # No threshold test
            # Just print the metric result
            print(
                f"{line_str.ljust(40)} {result_val:.5f}  ({metric_name} on {dataset_name})")


class Check:
    """
    A Check encapsulates:
      - Which dataset/slice to evaluate
      - A metric to compute (classification/detection, etc.)
      - Optional operator & value for threshold comparison
    """

    def __init__(
        self,
        name: str,
        dataset_or_slice: Union[Moonwatcher, Slice],
        predictions: Union[torch.Tensor, List[Dict[str, Any]]],
        metric: str,
        metric_parameters: Optional[Dict] = None,
        metric_class: Optional[Union[str, int]] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        operator: Optional[str] = None,
        value: Optional[float] = None,
    ):
        """
        :param name: Name of the check
        :param dataset_or_slice: The dataset or slice on which to evaluate
        :param predictions: Raw predictions (tensor for classification, list[dict] for detection)
        :param metric: Metric name (e.g. "Accuracy", "mAP", etc.)
        :param metric_parameters: Optional parameters for torchmetrics
        :param metric_class: Optional class name or ID for per-class metrics
        :param description: Optional description
        :param metadata: Optional dict of metadata/tags
        :param operator: For threshold checks (">", ">=", "<", "<=", "==", "!=")
        :param value: Numeric threshold to compare against
        """
        self.name = name
        self.dataset_or_slice = dataset_or_slice
        self.predictions = predictions
        self.metric = metric
        self.metric_parameters = metric_parameters or {}
        self.metric_class = metric_class
        self.description = description
        self.metadata = metadata or {}

        # Threshold comparison
        self.operator = operator
        self.value = value
        self.testing = bool(operator is not None and value is not None)

    def run(self, show: bool = False, save_report: bool = False) -> Dict[str, Any]:
        """
        Execute the check (compute the metric, optionally compare to threshold).
        """
        # 1) Compute metric
        result_value = calculate_metric(
            dataset_or_slice=self.dataset_or_slice,
            predictions=self.predictions,
            metric=self.metric,
            metric_parameters=self.metric_parameters,
            metric_class=self.metric_class,
        )

        # 2) Evaluate threshold if applicable
        success = None
        if self.testing:
            op_func = OPERATOR_DICT[self.operator]
            success = op_func(result_value, self.value)

        # 3) Build report dict
        if isinstance(self.dataset_or_slice, Slice):
            ds_name = self.dataset_or_slice.dataset_name
            slice_name = self.dataset_or_slice.name
        else:
            ds_name = self.dataset_or_slice.name
            slice_name = None

        report = {
            "check_name": self.name,
            "dataset_name": ds_name,
            "slice_name": slice_name,
            "metric": self.metric,
            "operator": self.operator,
            "value": self.value,
            "result": result_value,
            "success": success,
            "timestamp": get_current_timestamp(),
        }

        # 4) Optional output
        if show:
            visualize_report(report)

        if save_report:
            with open(f"check_{self.name}_report.json", "w", encoding="utf-8") as f:
                json.dump(report, f, indent=4)

        return report

    def __call__(self, show=False, save_report=False) -> Dict[str, Any]:
        """
        Convenience call to run the check.
        """
        return self.run(show=show, save_report=save_report)


class CheckSuite:
    """
    A CheckSuite is a collection of checks that can be run together, producing
    an aggregate success/fail (if each check uses thresholding).
    """

    def __init__(
        self,
        name: str,
        checks: List[Check],
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        show: bool = False,
    ):
        """
        :param name: Name of the checksuite
        :param checks: List of Check objects
        :param description: Optional description
        :param metadata: Optional dict of metadata/tags
        :param show: If True, prints each check's result to console
        """
        self.name = name
        self.checks = checks
        self.description = description
        self.metadata = metadata or {}
        self.show = show

    def run_all(self, show: Optional[bool] = None, save_report: bool = False) -> Dict[str, Any]:
        """
        Execute all checks in the suite, compile results,
        optionally show them and/or save to JSON.
        """
        show = self.show if show is None else show
        check_reports = []

        # Gather each check's report
        for check in self.checks:
            report = check.run(show=False, save_report=False)
            check_reports.append(report)

        # Evaluate overall success: only relevant if checks have thresholds
        # We say "true" if all threshold-based checks passed or if no thresholds used at all
        threshold_checks = [
            r for r in check_reports if r["success"] is not None]
        all_threshold_success = all(
            r["success"] for r in threshold_checks) if threshold_checks else True

        checksuite_report = {
            "checksuite_name": self.name,
            "timestamp": get_current_timestamp(),
            "success": all_threshold_success,
            "checks": check_reports,
        }

        # Optionally show & save
        if show:
            visualize_report(checksuite_report)
        if save_report:
            with open(f"checksuite_{self.name}_report.json", "w", encoding="utf-8") as f:
                json.dump(checksuite_report, f, indent=4)

        return checksuite_report

    def __call__(self, show: Optional[bool] = None, save_report: bool = False) -> Dict[str, Any]:
        """
        Convenience call to `run_all`.
        """
        return self.run_all(show=show, save_report=save_report)


# -----------------------------------------------------------------------------
# Optional Utility: "automated_checking" for demonstration or config-driven runs
# -----------------------------------------------------------------------------

def automated_checking(
    mw_dataset: Moonwatcher,
    predictions: Union[torch.Tensor, List[Dict[str, Any]]],
    metadata_keys: List[str] = None,
    metadata_list: List[Dict[str, Any]] = None,
    slicing_conditions: List[Dict[str, Any]] = None,
    checks: List[Dict[str, Any]] = None,
    demo: bool = True,
) -> Dict[str, Any]:
    """
    Example function for automating checks based on a config-driven approach.
    Optionally loads a "demo" config JSON to define metadata keys, slicing conditions, checks, etc.
    """
    import os

    # 1) Possibly load a config file if demo=True
    if demo:
        if mw_dataset.task_type == TaskType.CLASSIFICATION.value:
            filename = "demo_classification.json"
        elif mw_dataset.task_type == TaskType.DETECTION.value:
            filename = "demo_detection.json"
        else:
            raise ValueError(f"Unsupported task type: {mw_dataset.task_type}")

        base_path = Path(os.path.abspath(__file__)).parent
        with open(base_path / "configs" / filename, "r", encoding="utf-8") as file:
            config = json.load(file)

        metadata_keys = config.get("metadata_keys", [])
        slicing_conditions = config.get("slicing_conditions", [])
        checks = config.get("checks", [])

    # 2) Add metadata
    if metadata_keys:
        for key in metadata_keys:
            if key in ATTRIBUTE_FUNCTIONS:
                mw_dataset.add_predefined_metadata(key)
            else:
                mw_dataset.add_metadata_from_groundtruths(key)

    if metadata_list:
        mw_dataset.add_metadata_from_list(metadata_list)

    # 3) Create slices
    mw_slices = []
    if slicing_conditions:
        for cond in slicing_conditions:
            if cond["type"] == "threshold":
                mw_slice = mw_dataset.slice_by_threshold(
                    cond["key"], cond["operator"], cond["value"]
                )
                mw_slices.append(mw_slice)
            elif cond["type"] == "percentile":
                mw_slice = mw_dataset.slice_by_percentile(
                    cond["key"], cond["operator"], cond["value"]
                )
                mw_slices.append(mw_slice)
            elif cond["type"] == "class":
                # e.g. slice_by_class
                slices = mw_dataset.slice_by_class(cond["key"])
                mw_slices.extend(slices)
            else:
                raise ValueError(
                    f"Unknown slicing condition type: {cond['type']}")

    # 4) Build and run checks for each slice
    all_test_results = {}

    for mw_slice in mw_slices:
        check_objects = []
        for c in checks:
            check_name = f"{c['name']}_{mw_slice.name}"
            new_check = Check(
                name=check_name,
                dataset_or_slice=mw_slice,
                predictions=predictions,
                metric=c["metric"],
                metric_parameters=c.get("metric_parameters", None),
                metric_class=c.get("metric_class", None),
                operator=c.get("operator", None),
                value=c.get("value", None),
            )
            check_objects.append(new_check)

        check_suite = CheckSuite(
            name=f"Test_{mw_slice.name}", checks=check_objects)
        test_results = check_suite(show=True)
        all_test_results[mw_slice.name] = test_results

    # 5) Build and run checks on entire dataset
    check_objects = []
    for c in checks:
        check_name = f"{c['name']}_{mw_dataset.name}"
        new_check = Check(
            name=check_name,
            dataset_or_slice=mw_dataset,
            predictions=predictions,
            metric=c["metric"],
            metric_parameters=c.get("metric_parameters", None),
            metric_class=c.get("metric_class", None),
            operator=c.get("operator", None),
            value=c.get("value", None),
        )
        check_objects.append(new_check)

    entire_dataset_check_suite = CheckSuite(
        name=f"Test_{mw_dataset.name}", checks=check_objects)
    entire_dataset_test_results = entire_dataset_check_suite(show=True)
    all_test_results[mw_dataset.name] = entire_dataset_test_results

    return all_test_results
