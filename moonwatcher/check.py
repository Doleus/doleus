"""Check and CheckSuite classes for evaluating model performance metrics."""

import json
from typing import Any, Dict, List, Optional, Union

from moonwatcher.dataset.dataset import (Moonwatcher, Slice,
                                         get_original_indices)
from moonwatcher.metric import calculate_metric
from moonwatcher.utils.data import OPERATOR_DICT
from moonwatcher.utils.helpers import get_current_timestamp


def visualize_report(report: Dict[str, Any]):
    """Visualize a check or checksuite report.

    Parameters
    ----------
    report : Dict[str, Any]
        The report dictionary containing check or checksuite results.
    """
    visualizer = ReportVisualizer()
    visualizer.visualize(report)


class ReportVisualizer:
    """Helper class to pretty-print check and checksuite reports to the console."""

    def __init__(self):
        """Initialize the ReportVisualizer with formatting constants."""
        self.pass_emoji = "\u2705"
        self.fail_emoji = "\u274C"
        self.BOLD = "\033[1m"
        self.END = "\033[0m"
        self.UNDERLINE = "\033[4m"
        self.RED = "\033[91m"
        self.GREEN = "\033[92m"

    def visualize(self, report: Dict[str, Any], spacing: str = ""):
        """Recursively print check or checksuite reports in a hierarchical manner.

        Parameters
        ----------
        report : Dict[str, Any]
            The report dictionary containing check or checksuite results.
        spacing : str, optional
            Indentation spacing for nested reports, by default "".
        """
        if "checks" in report:
            # This is a CheckSuite
            self._print_checksuite(report, spacing)
        else:
            # This is a single Check
            self._print_check(report, spacing)

    def _print_checksuite(self, suite_report: Dict[str, Any], spacing: str):
        """Print a checksuite report with nested checks.

        Parameters
        ----------
        suite_report : Dict[str, Any]
            The checksuite report dictionary.
        spacing : str
            Indentation spacing for nested reports.
        """
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
        """Print a single check report.

        Parameters
        ----------
        check_report : Dict[str, Any]
            The check report dictionary.
        spacing : str
            Indentation spacing for the report.
        """
        # Decide success symbol (or blank if no thresholding)
        if check_report["success"] is not None:
            success_symbol = (
                self.pass_emoji if check_report["success"] else self.fail_emoji
            )
        else:
            success_symbol = "   "

        check_name = check_report["check_name"]
        line_str = f"{spacing}{success_symbol} {check_name}"

        # If there's a threshold test, show it
        operator = check_report["operator"]
        value = check_report["value"]
        result_val = check_report["result"]
        root_dataset_id = check_report["slice_name"] or check_report["root_dataset_id"]
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

            appendix = f"({metric_name} on {root_dataset_id})"
            print(
                f"{line_str.ljust(40)} {result_str} {comparison_str.ljust(10)} {appendix}"
            )
        else:
            # No threshold test
            # Just print the metric result
            print(
                f"{line_str.ljust(40)} {result_val:.5f}  ({metric_name} on {root_dataset_id})"
            )


class Check:
    """A Check encapsulates dataset evaluation with optional threshold comparison.

    A Check contains:
    - Which dataset/slice to evaluate
    - A metric to compute (classification/detection, etc.)
    - Optional operator & value for threshold comparison
    """

    def __init__(
        self,
        name: str,
        dataset: Union[Moonwatcher, Slice],
        model_id: str,
        metric: str,
        metric_parameters: Optional[Dict] = None,
        metric_class: Optional[Union[str, int]] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        operator: Optional[str] = None,
        value: Optional[float] = None,
    ):
        """Initialize a Check instance.

        Parameters
        ----------
        name : str
            Name of the check.
        dataset : Union[Moonwatcher, Slice]
            The dataset or slice on which to evaluate.
        model_id : str
            The name of the model associated with the predictions.
        metric : str
            Metric name (e.g. "Accuracy", "mAP", etc.).
        metric_parameters : Optional[Dict], optional
            Optional parameters for torchmetrics, by default None.
        metric_class : Optional[Union[str, int]], optional
            Optional class name or ID for per-class metrics, by default None.
        description : Optional[str], optional
            Optional description of the check, by default None.
        metadata : Optional[Dict[str, Any]], optional
            Optional dict of metadata/tags, by default None.
        operator : Optional[str], optional
            For threshold checks (">", ">=", "<", "<=", "==", "!="), by default None.
        value : Optional[float], optional
            Numeric threshold to compare against, by default None.
        """
        self.name = name
        self.dataset = dataset
        self.model_id = model_id
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
        """Execute the check (compute the metric, optionally compare to threshold).

        Parameters
        ----------
        show : bool, optional
            If True, prints the check results to console, by default False.
        save_report : bool, optional
            If True, saves the report to a JSON file, by default False.

        Returns
        -------
        Dict[str, Any]
            The check report dictionary containing results and metadata.
        """
        # Get root dataset and predictions
        root_dataset = (
            self.dataset.root_dataset
            if isinstance(self.dataset, Slice)
            else self.dataset
        )
        predictions = root_dataset.prediction_store.get_predictions(
            dataset_id=root_dataset.name, model_id=self.model_id
        )

        # Add predictions to dataset (temporary)
        root_dataset._set_predictions(predictions)

        indices = get_original_indices(dataset=self.dataset)

        # 1) Compute metric
        result_value = calculate_metric(
            dataset=root_dataset,
            indices=indices,
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
        if isinstance(self.dataset, Slice):
            ds_id = self.dataset.root_dataset.name
            slice_name = self.dataset.name
        else:
            ds_id = self.dataset.name
            slice_name = None

        report = {
            "check_name": self.name,
            "root_dataset_id": ds_id,
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
        """Convenience call to run the check.

        Parameters
        ----------
        show : bool, optional
            If True, prints the check results to console, by default False.
        save_report : bool, optional
            If True, saves the report to a JSON file, by default False.

        Returns
        -------
        Dict[str, Any]
            The check report dictionary containing results and metadata.
        """
        return self.run(show=show, save_report=save_report)


class CheckSuite:
    """A collection of checks that can be run together.

    A CheckSuite produces an aggregate success/fail result if each check uses
    thresholding.
    """

    def __init__(
        self,
        name: str,
        checks: List[Check],
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        show: bool = False,
    ):
        """Initialize a CheckSuite instance.

        Parameters
        ----------
        name : str
            Name of the checksuite.
        checks : List[Check]
            List of Check objects to be run.
        description : Optional[str], optional
            Optional description of the checksuite, by default None.
        metadata : Optional[Dict[str, Any]], optional
            Optional dict of metadata/tags, by default None.
        show : bool, optional
            If True, prints each check's result to console, by default False.
        """
        self.name = name
        self.checks = checks
        self.description = description
        self.metadata = metadata or {}
        self.show = show

    def run_all(
        self, show: Optional[bool] = None, save_report: bool = False
    ) -> Dict[str, Any]:
        """Run all checks in the suite.

        Parameters
        ----------
        show : Optional[bool], optional
            If True, prints each check's result to console. If None, uses the
            instance's show value, by default None.
        save_report : bool, optional
            If True, saves the report to a JSON file, by default False.

        Returns
        -------
        Dict[str, Any]
            The checksuite report dictionary containing all check results.
        """
        # Use instance-level show if not overridden
        show = self.show if show is None else show

        # Run all checks
        check_reports = []
        for check in self.checks:
            report = check.run(show=False)  # Don't show individual reports yet
            check_reports.append(report)

        # Determine overall success (if all checks have thresholds)
        all_have_thresholds = all(
            report["success"] is not None for report in check_reports
        )
        overall_success = None
        if all_have_thresholds:
            overall_success = all(report["success"] for report in check_reports)

        # Build suite report
        suite_report = {
            "checksuite_name": self.name,
            "success": overall_success,
            "checks": check_reports,
            "timestamp": get_current_timestamp(),
        }

        # Optional output
        if show:
            visualize_report(suite_report)

        if save_report:
            with open(
                f"checksuite_{self.name}_report.json", "w", encoding="utf-8"
            ) as f:
                json.dump(suite_report, f, indent=4)

        return suite_report

    def __call__(
        self, show: Optional[bool] = None, save_report: bool = False
    ) -> Dict[str, Any]:
        """Convenience call to run all checks in the suite.

        Parameters
        ----------
        show : Optional[bool], optional
            If True, prints each check's result to console. If None, uses the
            instance's show value, by default None.
        save_report : bool, optional
            If True, saves the report to a JSON file, by default False.

        Returns
        -------
        Dict[str, Any]
            The checksuite report dictionary containing all check results.
        """
        return self.run_all(show=show, save_report=save_report)


# -----------------------------------------------------------------------------
# Optional Utility: "automated_checking" for demonstration or config-driven runs
# -----------------------------------------------------------------------------

# def load_config(task_type: str) -> Dict[str, Any]:
#     """Load configuration based on task type"""

# def create_slices(dataset: Moonwatcher, conditions: List[Dict]) -> List[Slice]:
#     """Create slices based on conditions"""

# def create_checks(dataset: Union[Moonwatcher, Slice], check_configs: List[Dict]) -> List[Check]:
#     """Create checks from configurations"""

# def automated_checking(mw_dataset: Moonwatcher, **kwargs):
#     """Main function orchestrating the automated checking process"""
#     config = load_config(mw_dataset.task_type) if kwargs.get('demo') else kwargs
#     slices = create_slices(mw_dataset, config.get('slicing_conditions', []))
#     checks = create_checks(mw_dataset, config.get('checks', []))
#     return run_checks(mw_dataset, slices, checks)
