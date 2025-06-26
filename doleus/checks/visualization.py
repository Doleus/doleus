# SPDX-FileCopyrightText: 2025 Doleus contributors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict

REPORT_FORMATTING = {
    "PASS_EMOJI": "\u2705",
    "FAIL_EMOJI": "\u274c",
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
    """Helper class to visualize a check or checksuite report."""

    def __init__(self):
        self.pass_emoji = REPORT_FORMATTING["PASS_EMOJI"]
        self.fail_emoji = REPORT_FORMATTING["FAIL_EMOJI"]
        self.BOLD = REPORT_FORMATTING["BOLD"]
        self.END = REPORT_FORMATTING["END"]
        self.UNDERLINE = REPORT_FORMATTING["UNDERLINE"]
        self.RED = REPORT_FORMATTING["RED"]
        self.GREEN = REPORT_FORMATTING["GREEN"]

    def visualize(self, report: Dict[str, Any]):
        """Print a check or checksuite report.

        Parameters
        ----------
        report : Dict[str, Any]
            The report dictionary containing check or checksuite results.
        """
        if "checks" in report:
            self._print_checksuite(report)
        else:
            self._print_check(report)

    def _print_checksuite(self, report: Dict[str, Any]):
        """Print a checksuite report with nested checks.

        Parameters
        ----------
        report : Dict[str, Any]
            The checksuite report dictionary.
        """
        # Print line for the top-level suite
        success_symbol = self.pass_emoji if report["success"] else self.fail_emoji
        suite_name = report["checksuite_name"]
        line_str = f"{success_symbol} {suite_name}"
        print(line_str)

        # Print sub-checks with indentation
        for sub_report in report["checks"]:
            self._print_check(sub_report, indent=True)

    def _print_check(self, report: Dict[str, Any], indent: bool = False):
        """Print a single check report.

        Parameters
        ----------
        report : Dict[str, Any]
            The check report dictionary.
        indent : bool, optional
            Whether to indent this check (for checks inside a checksuite), by default False.
        """
        if report["success"] is not None:
            success_symbol = self.pass_emoji if report["success"] else self.fail_emoji
        else:
            success_symbol = "   "

        check_name = report["check_name"]
        spacing = "    " if indent else ""
        line_str = f"{spacing}{success_symbol} {check_name}"

        operator = report["operator"]
        value = report["value"]
        result_val = report["result"]
        dataset_id = report["dataset_id"]
        metric_name = report["metric"]

        if operator is not None and value is not None:
            op_symbol = OPERATOR_SYMBOLS.get(operator, operator)
            color = self.GREEN if report["success"] else self.RED
            result_str = f"{color}{result_val:.5f}{self.END}"
            comparison_str = f"{op_symbol} {value}"

            appendix = f"({metric_name} on {dataset_id})"
            print(
                f"{line_str.ljust(40)} {result_str} {comparison_str.ljust(10)} {appendix}"
            )
        else:
            print(
                f"{line_str.ljust(40)} {result_val:.5f}  ({metric_name} on {dataset_id})"
            )
