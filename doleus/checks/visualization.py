"""Visualization utilities for check and checksuite reports."""

from typing import Any, Dict

# Formatting constants for report visualization
REPORT_FORMATTING = {
    "PASS_EMOJI": "✅",  # \u2705
    "FAIL_EMOJI": "❌",  # \u274C
    "BOLD": "\033[1m",
    "END": "\033[0m",
    "UNDERLINE": "\033[4m",
    "RED": "\033[91m",
    "GREEN": "\033[92m",
}

# Symbol mappings for operators
OPERATOR_SYMBOLS = {
    ">=": "≥",
    "<=": "≤",
    "==": "=",
    "!=": "≠"
}


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