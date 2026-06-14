"""Public DIDdesign package interface."""

from importlib.metadata import PackageNotFoundError, version as _package_version

from .core.data_contracts import DataContractError, DidDataError
from .diagnostics import DidCheckDiagnosticRow, DidCheckPatternRow, DidCheckResult, DidCheckTrendRow, did_check
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
    "DidDataError",
    "DidFormulaSpec",
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
