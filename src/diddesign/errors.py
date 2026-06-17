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


# ---------------------------------------------------------------------------
# Structured Exceptions
# ---------------------------------------------------------------------------


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
        detail_parts = [f"[E{int(code):03d}] {message}"]
        if self.context:
            detail_parts.append("  Context:")
            for k, v in self.context.items():
                detail_parts.append(f"    {k} = {v}")
        self.detailed_message = "\n".join(detail_parts)
        super().__init__(self.detailed_message)


class DidValueError(DidError, ValueError):
    """Structured DIDdesign error for parameter / data validation failures.

    Inherits from both :class:`DidError` and :class:`ValueError` so that
    existing ``except ValueError`` handlers continue to catch these errors.
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
