"""Check and CheckSuite classes for evaluating model performance metrics."""

import json
from typing import Any, Dict, List, Optional, Union

from moonwatcher.dataset.dataset import (Moonwatcher, Slice,
                                         get_original_indices)
from moonwatcher.metric import calculate_metric
from moonwatcher.utils.data import OPERATOR_DICT
from moonwatcher.utils.helpers import get_current_timestamp

# At the top of the file with other imports
REPORT_FORMATTING = {
    "PASS_EMOJI": "✅",  # \u2705
    "FAIL_EMOJI": "❌",  # \u274C
    "BOLD": "\033[1m",
    "END": "\033[0m",
    "UNDERLINE": "\033[4m",
    "RED": "\033[91m",
    "GREEN": "\033[92m",
}

OPERATOR_SYMBOLS = {">=": "≥", "<=": "≤", "==": "=", "!=": "≠"}


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
        self.pass_emoji = REPORT_FORMATTING["PASS_EMOJI"]
        self.fail_emoji = REPORT_FORMATTING["FAIL_EMOJI"]
        self.BOLD = REPORT_FORMATTING["BOLD"]
        self.END = REPORT_FORMATTING["END"]
        self.UNDERLINE = REPORT_FORMATTING["UNDERLINE"]
        self.RED = REPORT_FORMATTING["RED"]
        self.GREEN = REPORT_FORMATTING["GREEN"]

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
            self._print_checksuite(report, spacing)
        else:
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
            # Use lookup table instead of multiple if/elif
            op_symbol = OPERATOR_SYMBOLS.get(operator, operator)
            color = self.GREEN if check_report["success"] else self.RED
            result_str = f"{color}{result_val:.5f}{self.END}"
            comparison_str = f"{op_symbol} {value}"

            appendix = f"({metric_name} on {root_dataset_id})"
            print(
                f"{line_str.ljust(40)} {result_str} {comparison_str.ljust(10)} {appendix}"
            )
        else:
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
        show: bool = False,
    ):
        """Initialize a CheckSuite instance.

        Parameters
        ----------
        name : str
            Name of the checksuite.
        checks : List[Check]
            List of Check objects to be run.
        show : bool, optional
            If True, prints each check's result to console, by default False.
        """
        self.name = name
        self.checks = checks
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

        check_reports = []
        for check in self.checks:
            report = check.run(show=False)  # Don't show individual reports yet
            check_reports.append(report)

        all_have_thresholds = all(
            report["success"] is not None for report in check_reports
        )
        overall_success = None
        if all_have_thresholds:
            overall_success = all(report["success"] for report in check_reports)

        suite_report = {
            "checksuite_name": self.name,
            "success": overall_success,
            "checks": check_reports,
            "timestamp": get_current_timestamp(),
        }

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
