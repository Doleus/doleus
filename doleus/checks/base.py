import json
from typing import Any, Dict, List, Optional, Union

from doleus.checks.visualization import visualize_report
from doleus.datasets import Doleus, Slice, get_original_indices
from doleus.metrics import calculate_metric
from doleus.utils.data import OPERATOR_DICT
from doleus.utils.utils import get_current_timestamp


class Check:
    """A Check encapsulates a specific metric evaluation on a dataset."""

    def __init__(
        self,
        name: str,
        dataset: Union[Doleus, Slice],
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
        dataset : Union[Doleus, Slice]
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
        self.operator = operator
        self.value = value
        self.testing = operator is not None and value is not None

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

        # Compute metric
        result_value = calculate_metric(
            dataset=root_dataset,
            indices=indices,
            metric=self.metric,
            metric_parameters=self.metric_parameters,
            metric_class=self.metric_class,
        )

        # Evaluate threshold if applicable
        success = None
        if self.testing:
            op_func = OPERATOR_DICT[self.operator]
            success = op_func(result_value, self.value)

        # Build report dict
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
        show = self.show if show is None else show

        check_reports = []
        for check in self.checks:
            report = check.run(show=False)
            check_reports.append(report)

        if not check_reports:
            overall_success = True  # An empty suite passes by default
        else:
            overall_success = all(report["success"] is not False for report in check_reports)

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
