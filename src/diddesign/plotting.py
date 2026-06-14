from __future__ import annotations

from collections.abc import Mapping
import math
from statistics import NormalDist
from typing import Any

import pandas as pd

from .diagnostics import DidCheckResult
from .results.objects import DidResult


_Z_90 = NormalDist().inv_cdf(0.95)
_FIT_COLUMNS = (
    "source",
    "estimator",
    "lag",
    "lead",
    "time_to_treat",
    "estimate",
    "std_error",
    "ci90_lb",
    "ci90_ub",
)
_FIT_SOURCE_ORDER = {"check": 0, "fit": 1}


def _ci90_bounds(estimate: float, std_error: float) -> tuple[float, float]:
    return estimate - _Z_90 * std_error, estimate + _Z_90 * std_error


def _require_finite_fit_value(value: Any, *, field_name: str) -> float:
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"fit() requires finite {field_name}.") from None
    if not math.isfinite(coerced):
        raise ValueError(f"fit() requires finite {field_name}.")
    return coerced


def _require_nonnegative_fit_std_error(value: Any, *, field_name: str) -> float:
    coerced = _require_finite_fit_value(value, field_name=field_name)
    if coerced < 0:
        raise ValueError(f"fit() requires non-negative {field_name}.")
    return coerced


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


def _fit_row_sort_key(row: Mapping[str, Any]) -> tuple[float, int, int]:
    lag_or_lead = row["lag"] if row["source"] == "check" else row["lead"]
    return (
        row["time_to_treat"],
        _FIT_SOURCE_ORDER[row["source"]],
        -1 if lag_or_lead is None else lag_or_lead,
    )


def _metadata_value(metadata: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = metadata.get(key)
        if value is not None:
            return value
    return None


def _canonical_clustvar(metadata: Mapping[str, Any]) -> Any:
    cluster_label = _metadata_value(metadata, "clustvar", "cluster_column", "cluster_default")
    if cluster_label is not None:
        return cluster_label
    if _metadata_value(metadata, "cluster_mode") == "observation":
        return "_observation"
    return None


def _require_fit_compatibility(result: DidResult, check_fit: DidCheckResult) -> None:
    mismatches: list[str] = []
    comparable_fields = (
        ("design", ("design",), ("design",)),
        ("data_type", ("data_type", "datatype"), ("data_type", "datatype")),
        ("outcome", ("outcome",), ("outcome",)),
        ("treatment", ("treatment",), ("treatment",)),
        ("time", ("time",), ("time",)),
        ("unit_id", ("unit_id",), ("unit_id",)),
        ("post", ("post",), ("post",)),
        ("time_order", ("time_order",), ("time_order",)),
        ("covariates", ("covariates",), ("covariates",)),
        ("thres", ("thres",), ("thres",)),
    )
    for label, result_keys, check_keys in comparable_fields:
        result_value = _metadata_value(result.metadata, *result_keys)
        check_value = _metadata_value(check_fit.metadata, *check_keys)
        if result_value is None or check_value is None:
            continue
        if result_value != check_value:
            mismatches.append(label)

    result_clustvar = _canonical_clustvar(result.metadata)
    check_clustvar = _canonical_clustvar(check_fit.metadata)
    if result_clustvar is not None and check_clustvar is not None and result_clustvar != check_clustvar:
        mismatches.append("clustvar")

    if mismatches:
        fields = ", ".join(mismatches)
        raise ValueError(f"check_fit metadata is incompatible with did() result for: {fields}.")


def check(
    result: DidCheckResult,
    *,
    as_frame: bool = False,
) -> dict[str, tuple[dict[str, Any], ...]] | dict[str, pd.DataFrame]:
    """Return named plotting rows for did_check diagnostics."""

    if not isinstance(result, DidCheckResult):
        raise TypeError("check() expects a DidCheckResult instance.")
    as_frame = _require_as_frame_bool(as_frame)

    plot_rows = result.named_plot_rows()
    if as_frame:
        frames: dict[str, pd.DataFrame] = {"placebo": result.to_placebo_frame()}
        if "pattern" in plot_rows:
            frames["pattern"] = result.to_pattern_frame()
            return frames
        frames["trends"] = result.to_trends_frame()
        return frames
    return plot_rows


def fit(
    result: DidResult,
    check_fit: DidCheckResult | None = None,
    *,
    as_frame: bool = False,
) -> tuple[dict[str, Any], ...] | pd.DataFrame:
    """Return fit rows using raw placebo rows and Double-DID estimates."""

    if not isinstance(result, DidResult):
        raise TypeError("fit() expects a DidResult instance.")
    if check_fit is not None and not isinstance(check_fit, DidCheckResult):
        raise TypeError("fit() expects check_fit to be a DidCheckResult instance.")
    as_frame = _require_as_frame_bool(as_frame)

    rows: list[dict[str, Any]] = []
    if check_fit is not None:
        _require_fit_compatibility(result, check_fit)
        result_design = _metadata_value(result.metadata, "design")
        if result_design == "sa" and not check_fit.pattern_table:
            raise ValueError("fit() requires staggered-adoption check_fit pattern diagnostics for SA results.")
        if check_fit.pattern_table and result_design != "sa":
            raise ValueError("fit() cannot overlay staggered-adoption pattern diagnostics onto a non-SA result.")
        for diagnostic_row in check_fit.diagnostic_table:
            estimate_raw = _require_finite_fit_value(
                diagnostic_row.estimate_raw,
                field_name="raw placebo estimate",
            )
            std_error_raw = _require_nonnegative_fit_std_error(
                diagnostic_row.std_error_raw,
                field_name="raw placebo std_error",
            )
            ci90_lb, ci90_ub = _ci90_bounds(estimate_raw, std_error_raw)
            rows.append(
                {
                    "source": "check",
                    "estimator": None,
                    "lag": diagnostic_row.lag,
                    "lead": None,
                    "time_to_treat": -diagnostic_row.lag,
                    "estimate": estimate_raw,
                    "std_error": std_error_raw,
                    "ci90_lb": ci90_lb,
                    "ci90_ub": ci90_ub,
                }
            )

    fit_rows = []
    for estimate_row in result.estimate_rows():
        if not estimate_row.estimator.endswith("Double-DID") or estimate_row.std_error is None:
            continue
        ci90_lb, ci90_ub = _ci90_bounds(estimate_row.estimate, estimate_row.std_error)
        fit_rows.append(
            {
                "source": "fit",
                "estimator": estimate_row.estimator,
                "lag": None,
                "lead": estimate_row.lead,
                "time_to_treat": estimate_row.lead,
                "estimate": estimate_row.estimate,
                "std_error": estimate_row.std_error,
                "ci90_lb": ci90_lb,
                "ci90_ub": ci90_ub,
            }
        )
    if not fit_rows:
        raise ValueError("fit() requires Double-DID estimate rows to be available.")

    rows.extend(fit_rows)
    fit_rows = tuple(sorted(rows, key=_fit_row_sort_key))
    if as_frame:
        return pd.DataFrame.from_records(fit_rows, columns=_FIT_COLUMNS)
    return fit_rows


__all__ = ["check", "fit"]
