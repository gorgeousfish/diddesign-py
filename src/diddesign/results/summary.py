from __future__ import annotations

import math
from collections.abc import Sequence
from collections.abc import Mapping
from operator import index as _index
from typing import Any

import pandas as pd

from .objects import DidResult


_SUMMARY_COLUMNS = (
    "estimator",
    "lead",
    "estimate",
    "std.error",
    "statistic",
    "p_value",
    "ci.low",
    "ci.high",
)
_DIAGNOSTIC_SUMMARY_COLUMNS = (
    "lag",
    "estimate_raw",
    "std_error_raw",
    "eqci95_lb_std",
    "eqci95_ub_std",
)
_SUPPORTED_ESTIMATORS = frozenset(
    {
        "Double-DID",
        "DID",
        "sDID",
        "SA-Double-DID",
        "SA-DID",
        "SA-sDID",
    }
)
_ESTIMATOR_SUMMARY_RENDER_COLUMNS = (
    ("estimator", "Estimator"),
    ("lead", "Lead"),
    ("estimate", "Estimate"),
    ("std.error", "Std. Error"),
    ("statistic", "z"),
    ("p_value", "p>|z|"),
    ("ci.low", "95% CI Low"),
    ("ci.high", "95% CI High"),
)
_DIAGNOSTIC_SUMMARY_RENDER_COLUMNS = (
    ("lag", "Lag"),
    ("estimate_raw", "Estimate"),
    ("std_error_raw", "Std. Error"),
    ("eqci95_lb_std", "Eq. CI Low"),
    ("eqci95_ub_std", "Eq. CI High"),
)


def _normal_two_sided_p_value(statistic: float | None) -> float | None:
    if statistic is None:
        return None
    return math.erfc(abs(statistic) / math.sqrt(2.0))


def _z_statistic(estimate: float, std_error: float | None) -> float | None:
    if std_error is None:
        return None
    if std_error == 0:
        if estimate > 0:
            return math.inf
        if estimate < 0:
            return -math.inf
        return None
    return estimate / std_error


def _normalize_estimator_filter(estimator: str | Sequence[str] | None) -> tuple[str, ...] | None:
    if estimator is None:
        return None
    if isinstance(estimator, str):
        requested = (estimator,)
    elif isinstance(estimator, Sequence):
        requested = tuple(estimator)
    else:
        raise TypeError("estimator must be a string, a sequence of strings, or None.")

    if not requested:
        raise ValueError("estimator sequence must not be empty.")

    if any(not isinstance(label, str) for label in requested):
        raise TypeError("estimator must be a string, a sequence of strings, or None.")

    if len(set(requested)) != len(requested):
        raise ValueError("estimator sequence must not contain duplicate labels.")

    unsupported = tuple(label for label in requested if label not in _SUPPORTED_ESTIMATORS)
    if unsupported:
        raise ValueError(f"Not supported estimator(s): {', '.join(unsupported)}")
    return requested


def _require_as_frame_bool(as_frame: Any) -> bool:
    if isinstance(as_frame, bool):
        return as_frame
    if type(as_frame).__module__ == "numpy" and type(as_frame).__name__ == "bool":
        return bool(as_frame)
    if type(as_frame).__module__ == "numpy" and type(as_frame).__name__ == "bool_":
        return bool(as_frame)
    if not isinstance(as_frame, bool):
        raise TypeError("as_frame must be a boolean.")
    return as_frame


def _require_nonnegative_digits(digits: Any) -> int:
    if isinstance(digits, bool):
        raise TypeError("digits must be a non-negative integer.")
    try:
        parsed = _index(digits)
    except TypeError:
        raise TypeError("digits must be a non-negative integer.") from None
    if parsed < 0:
        raise TypeError("digits must be a non-negative integer.")
    return parsed


def _format_summary_value(value: Any, *, digits: int, p_value: bool = False) -> str:
    if value is None:
        return "."
    if isinstance(value, float):
        if math.isnan(value):
            return "."
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        if p_value and 0 < value < 10**-digits:
            return f"<{10**-digits:.{digits}f}"
        return f"{value:.{digits}f}"
    return str(value)


def _render_summary_table(
    *,
    title: str,
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[tuple[str, str]],
    digits: int,
) -> str:
    if not rows:
        return f"{title}\n{'=' * len(title)}\n(no rows)"

    rendered_rows = [
        [
            _format_summary_value(
                row.get(key),
                digits=digits,
                p_value=(key == "p_value"),
            )
            for key, _ in columns
        ]
        for row in rows
    ]
    headers = [label for _, label in columns]
    widths = [
        max(len(header), *(len(row[index]) for row in rendered_rows))
        for index, header in enumerate(headers)
    ]
    numeric_keys = {
        "lead",
        "lag",
        "estimate",
        "std.error",
        "std_error_raw",
        "statistic",
        "p_value",
        "ci.low",
        "ci.high",
        "estimate_raw",
        "eqci95_lb_std",
        "eqci95_ub_std",
    }

    def render_line(values: Sequence[str], *, header: bool = False) -> str:
        cells: list[str] = []
        for index, value in enumerate(values):
            key = columns[index][0]
            if not header and key in numeric_keys:
                cells.append(value.rjust(widths[index]))
            else:
                cells.append(value.ljust(widths[index]))
        return "  ".join(cells).rstrip()

    header_line = render_line(headers, header=True)
    separator = "  ".join("-" * width for width in widths)
    body = "\n".join(render_line(row) for row in rendered_rows)
    return f"{title}\n{'=' * len(title)}\n{header_line}\n{separator}\n{body}"


def summary(
    result: DidResult | Any,
    estimator: str | Sequence[str] | None = None,
    *,
    as_frame: bool = False,
) -> tuple[dict[str, Any], ...] | pd.DataFrame:
    """Project estimator or diagnostic rows into public summary output."""

    from diddesign.diagnostics import DidCheckResult

    as_frame = _require_as_frame_bool(as_frame)
    if isinstance(result, DidCheckResult):
        if estimator is not None:
            raise TypeError("summary() only supports estimator= for DidResult instances.")
        rows = tuple(result.summary_rows())
        if as_frame:
            return pd.DataFrame.from_records(rows, columns=_DIAGNOSTIC_SUMMARY_COLUMNS)
        return rows
    if not isinstance(result, DidResult):
        raise TypeError("summary() expects a DidResult or DidCheckResult instance.")

    requested_estimators = _normalize_estimator_filter(estimator)
    rows: list[dict[str, Any]] = []
    for estimate_row in result.estimate_rows():
        if requested_estimators is not None and estimate_row.estimator not in requested_estimators:
            continue
        statistic = _z_statistic(estimate_row.estimate, estimate_row.std_error)
        rows.append(
            {
                "estimator": estimate_row.estimator,
                "lead": estimate_row.lead,
                "estimate": estimate_row.estimate,
                "std.error": estimate_row.std_error,
                "statistic": statistic,
                "p_value": _normal_two_sided_p_value(statistic),
                "ci.low": estimate_row.ci_lo,
                "ci.high": estimate_row.ci_hi,
            }
        )
    summary_rows = tuple(rows)
    if as_frame:
        return pd.DataFrame.from_records(summary_rows, columns=_SUMMARY_COLUMNS)
    return summary_rows


def format_summary(
    result: DidResult | Any,
    estimator: str | Sequence[str] | None = None,
    *,
    digits: int = 3,
) -> str:
    """Render public summary rows as a stable, readable text table."""

    from diddesign.diagnostics import DidCheckResult

    digits = _require_nonnegative_digits(digits)
    rows = summary(result, estimator=estimator)
    if isinstance(result, DidCheckResult):
        return _render_summary_table(
            title="DIDdesign Diagnostic Summary",
            rows=rows,
            columns=_DIAGNOSTIC_SUMMARY_RENDER_COLUMNS,
            digits=digits,
        )
    return _render_summary_table(
        title="DIDdesign Summary",
        rows=rows,
        columns=_ESTIMATOR_SUMMARY_RENDER_COLUMNS,
        digits=digits,
    )


__all__ = ["format_summary", "summary"]
