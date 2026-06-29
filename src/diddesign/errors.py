"""DIDdesign structured error and warning system.

Provides error codes, warning codes, and structured exceptions that parallel
the Stata DIDdesign package's E001-E018 / W001-W010 diagnostic system.  Each
error carries a machine-readable code and a context dictionary with the
diagnostic quantities a user needs to identify the root cause.
"""

from __future__ import annotations

import warnings
from enum import IntEnum
from typing import Any, Dict, Optional


class ErrorCode(IntEnum):
    """Standard error codes matching Stata's E001-E018 system."""

    # Parameter / option errors
    E001 = 1   # Missing required parameter or variable
    E002 = 2   # Invalid parameter value
    # Data errors
    E003 = 3   # Data constraint violation (no obs, non-binary, unbalanced panel, etc.)
    # Computation errors
    E004 = 4   # Point estimate failed
    E005 = 5   # Covariance matrix computation failed
    E006 = 6   # GMM weight computation failed
    # Design constraints
    E007 = 7   # SA design-specific constraint violation
    E008 = 8   # Panel structure violation
    E009 = 9   # Bootstrap failure
    E010 = 10  # kmax out of range
    E011 = 11  # Insufficient moment conditions
    # Validation errors
    E012 = 12  # Treatment not binary (not 0/1)
    E013 = 13  # Treatment not absorbing (irreversible)
    E014 = 14  # Standard DID requires common treatment adoption time
    E015 = 15  # SA design requires >= 3 time periods
    E016 = 16  # RCS post indicator inconsistent with time
    E017 = 17  # Time column contains non-finite values
    E018 = 18  # RCS requires cluster variable
    E019 = 19  # Ambiguous string time ordering
    E020 = 20  # Invalid design/data_type parameter


class WarningCode(IntEnum):
    """Standard warning codes matching Stata's W001-W010 system."""

    W001 = 1   # Automatic variable type conversion
    W002 = 2   # Factor variable expansion
    W003 = 3   # Duplicate covariate removal
    W004 = 4   # Parallel option not implemented
    W005 = 5   # Bootstrap partial iteration failure
    W006 = 6   # Covariance matrix singular (rank reduction)
    W007 = 7   # Condition number too large
    W008 = 8   # GMM weights do not sum to 1
    W009 = 9   # GMM weights outside [0, 1]
    W010 = 10  # Bootstrap effective sample insufficient
    W011 = 11  # J-test disabled: insufficient moment conditions (kmax <= 2)
    W012 = 12  # Bootstrap iterations below recommended threshold


# ---------------------------------------------------------------------------
# Fix Suggestions for Common Errors
# ---------------------------------------------------------------------------

ERROR_SUGGESTIONS: dict[str, str] = {
    "E001": "Check that all required parameters (unit_id for panel, post for RCS) are provided.",
    "E002": "When using the formula interface, do not pass outcome, treatment, post, or covariates as separate parameters. For non-formula calls, verify that lead values are non-negative integers and kmax is 1-8.",
    "E003": "Ensure the treatment column contains only 0 and 1 values (binary indicator).",
    "E007": "SA design requires absorbing treatment: once treated, a unit stays treated.",
    "E008": "K-DID (kmax>2) requires panel data. Set data_type='panel' and provide unit_id.",
    "E009": "Bootstrap needs at least 2 clusters. Check id_cluster or unit_id has >=2 unique values.",
    "E014": "SA design is not supported for RCS data. Use design='did' or switch to panel data.",
    "E016": "Cannot specify both id() and post(). Use id/unit_id for panel, post for RCS.",
    "E018": "RCS design requires cluster(). Specify id_cluster for bootstrap inference.",
    "E020": "design must be 'did' or 'sa'; data_type must be 'panel' or 'rcs'.",
}


# ---------------------------------------------------------------------------
# Structured Exceptions
# ---------------------------------------------------------------------------


class DidDataError(ValueError):
    """Raised when input data cannot satisfy the DID data requirements.

    This is the backward-compatible base class.  ``DataContractError`` is an
    alias for this class.
    """


class DidError(Exception):
    """Base DIDdesign structured error with code and context."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.code = code
        self.message = message
        self.context = context or {}
        # Build detailed multi-line message
        code_str = f"E{int(code):03d}"
        detail_parts = [f"[{code_str}] {message}"]
        suggestion = ERROR_SUGGESTIONS.get(code_str, "")
        if suggestion:
            detail_parts.append(f"  Suggestion: {suggestion}")
        if self.context:
            detail_parts.append("  Context:")
            for k, v in self.context.items():
                detail_parts.append(f"    {k} = {v}")
        self.detailed_message = "\n".join(detail_parts)
        super().__init__(self.detailed_message)


class DidValueError(DidError, DidDataError):
    """Structured DIDdesign error for parameter / data validation failures.

    Inherits from both :class:`DidError` and :class:`DidDataError` so that
    existing ``except ValueError`` and ``except DataContractError`` handlers
    continue to catch these errors.
    """


class DidRuntimeError(DidError, RuntimeError):
    """Structured DIDdesign error for computation failures.

    Inherits from both :class:`DidError` and :class:`RuntimeError` so that
    existing ``except RuntimeError`` handlers continue to catch these errors.
    """


# ---------------------------------------------------------------------------
# Structured Warnings
# ---------------------------------------------------------------------------


class DidWarning(UserWarning):
    """DIDdesign structured warning with code and context."""

    def __init__(
        self,
        code: WarningCode,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.code = code
        self.context = context or {}
        detail = f"[W{int(code):03d}] {message}"
        if self.context:
            ctx_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            detail += f" ({ctx_str})"
        super().__init__(detail)


# ---------------------------------------------------------------------------
# Convenience emitter
# ---------------------------------------------------------------------------


def did_warn(
    code: WarningCode,
    message: str,
    context: Optional[Dict[str, Any]] = None,
    stacklevel: int = 2,
) -> None:
    """Emit a structured DIDdesign warning."""
    warnings.warn(
        DidWarning(code, message, context),
        stacklevel=stacklevel,
    )
