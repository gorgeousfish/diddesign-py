"""diddesign: Double Difference-in-Differences for Python.

This package implements the multiple-pre-treatment DID estimator proposed by
Egami and Yamauchi (2023, Political Analysis). It combines standard DID and
sequential DID via efficient GMM weighting, extending to K-DID for panels
with three or more pre-treatment periods and to staggered-adoption designs
with lead-specific estimates.

The public interface consists of two estimation functions—:func:`did` for
treatment effect estimation and :func:`did_check` for pre-treatment
diagnostics—together with immutable result objects (:class:`DidResult`,
:class:`DidCheckResult`) whose frame accessors return pandas DataFrames
for downstream analysis, plotting, and LaTeX export.
"""

from importlib.metadata import PackageNotFoundError, version as _package_version

from .core.data_contracts import DataContractError, DidDataError
from .diagnostics import DidCheckDiagnosticRow, DidCheckPatternRow, DidCheckResult, DidCheckTrendRow, did_check
from .diagnostics_reporter import DiagnosticsReporter
from .errors import (
    DidError,
    DidRuntimeError,
    DidValueError,
    DidWarning,
    ErrorCode,
    WarningCode,
    did_warn,
)
from .estimators import did
from .formula import DidFormulaSpec, did_formula
from .plotting import check, fit
from .results import (
    DidBootstrapDraw,
    DidBootstrapDrawK,
    DidEstimateRow,
    DidGmmAuditRow,
    DidGmmRow,
    DidResult,
    DidWeightRow,
    format_summary,
    summary,
)
from .visualization import plot_diagnostics, plot_estimates, plot_pattern, plot_placebo, plot_trends

DIDResult = DidResult

try:
    __version__ = _package_version("diddesign")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "__version__",
    "DataContractError",
    "DiagnosticsReporter",
    "DidDataError",
    "DidError",
    "DidFormulaSpec",
    "DidRuntimeError",
    "DidValueError",
    "DidWarning",
    "ErrorCode",
    "WarningCode",
    "did_warn",
    "DidBootstrapDraw",
    "DidBootstrapDrawK",
    "DidCheckDiagnosticRow",
    "DidCheckPatternRow",
    "DidCheckResult",
    "DidCheckTrendRow",
    "DidEstimateRow",
    "DidGmmRow",
    "DidResult",
    "DIDResult",
    "DidWeightRow",
    "check",
    "did",
    "did_check",
    "did_formula",
    "fit",
    "format_summary",
    "plot_diagnostics",
    "plot_estimates",
    "plot_pattern",
    "plot_placebo",
    "plot_trends",
    "summary",
]
