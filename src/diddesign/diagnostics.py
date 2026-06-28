from __future__ import annotations

import math
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from operator import index as _index
from statistics import NormalDist
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from .core.data_contracts import normalize_design_data
from .core.encoding import auto_encode_string_columns
from .core.validation import require_column
from .formula import (
    DidFormulaSpec,
    did_formula as parse_did_formula,
    parse_covariate_expression,
    parse_covariate_term,
)
from .estimators.did import (
    _bootstrap_sa_frame,
    _prepare_sa_frame,
    _sa_periods_to_use,
    _sa_status_matrix,
    _sa_subject_ids,
    _sa_time_weights,
    _validate_thres,
    _with_outcome_delta,
)


_Z_90 = NormalDist().inv_cdf(0.95)
_MISSING = object()
_OPTION_IGNORED_KEYS = frozenset({"parallel", "se_boot", "stdz", "lead"})
_DIAGNOSTIC_COLUMNS = (
    "lag",
    "estimate_std",
    "std_error_std",
    "estimate_raw",
    "std_error_raw",
    "eqci95_lb_std",
    "eqci95_ub_std",
)
_SUMMARY_COLUMNS = (
    "lag",
    "estimate_raw",
    "std_error_raw",
    "eqci95_lb_std",
    "eqci95_ub_std",
)
_PLACEBO_COLUMNS = (
    "lag",
    "time_to_treat",
    "estimate_std",
    "std_error_std",
    "eqci95_lb_std",
    "eqci95_ub_std",
)
_TREND_COLUMNS = (
    "time_to_treat",
    "group",
    "outcome_mean",
    "outcome_sd",
    "ci90_lb",
    "ci90_ub",
    "n_obs",
)
_PATTERN_COLUMNS = ("id_time", "unit_order", "status")


def _merge_user_metadata_without_generated_overrides(
    generated_metadata: Mapping[str, Any],
    user_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(generated_metadata)
    if user_metadata is None:
        return merged
    reserved = tuple(sorted(set(user_metadata) & set(generated_metadata)))
    if reserved:
        raise ValueError(
            "metadata cannot override data-driven did_check fields: "
            + ", ".join(reserved)
            + "."
        )
    merged.update(user_metadata)
    return merged


def _is_bool_like(value: Any) -> bool:
    return isinstance(value, bool) or (
        type(value).__module__ == "numpy"
        and type(value).__name__ in {"bool", "bool_"}
    )


def _lookup_mapping_value(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return _MISSING


def _require_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping.")
    return value


def _coerce_optional_float(value: Any, *, field_name: str) -> float | None:
    if value is _MISSING or value is None:
        return None
    if _is_bool_like(value):
        raise ValueError("numeric diagnostic fields must be finite numbers, not booleans.")
    if isinstance(value, str):
        value = value.strip()
        if not value or value.upper() == "NA":
            return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be finite.") from None
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite.")
    return parsed


def _coerce_required_float(value: Any, *, field_name: str) -> float:
    parsed = _coerce_optional_float(value, field_name=field_name)
    if parsed is None:
        raise ValueError(f"{field_name} is required.")
    return parsed


def _coerce_required_int(value: Any, *, field_name: str) -> int:
    parsed = _coerce_required_float(value, field_name=field_name)
    if not parsed.is_integer():
        raise ValueError(f"{field_name} must be an integer.")
    return int(parsed)


def _coerce_optional_int(value: Any, *, field_name: str) -> int | None:
    parsed = _coerce_optional_float(value, field_name=field_name)
    if parsed is None:
        return None
    if not parsed.is_integer():
        raise ValueError(f"{field_name} must be an integer.")
    return int(parsed)


def _coerce_numeric(value: Any, *, field_name: str) -> int | float:
    parsed = _coerce_required_float(value, field_name=field_name)
    if parsed.is_integer():
        return int(parsed)
    return parsed


def _freeze_metadata_value(value: Any) -> Any:
    if _is_bool_like(value):
        return bool(value)
    if type(value).__module__ == "numpy" and type(value).__name__.startswith(
        ("int", "uint")
    ):
        return int(value)
    if type(value).__module__ == "numpy" and type(value).__name__.startswith("float"):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("metadata value must be finite.")
        return parsed
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metadata value must be finite.")
        return value
    if type(value).__module__ == "numpy" and type(value).__name__ == "ndarray":
        return _freeze_metadata_value(value.tolist())
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                key: _freeze_metadata_value(nested_value)
                for key, nested_value in value.items()
            }
        )
    if isinstance(value, tuple):
        return tuple(_freeze_metadata_value(item) for item in value)
    if isinstance(value, list):
        return tuple(_freeze_metadata_value(item) for item in value)
    return value


def _freeze_metadata(metadata: Mapping[str, Any]) -> MappingProxyType:
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping.")
    return MappingProxyType(
        {
            key: _freeze_metadata_value(value)
            for key, value in metadata.items()
        }
    )


def _thaw_metadata_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _thaw_metadata_value(nested_value)
            for key, nested_value in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_thaw_metadata_value(item) for item in value)
    return value


def _require_finite_numeric(value: Any, *, field_name: str) -> int | float:
    if _is_bool_like(value):
        raise ValueError(f"{field_name} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be finite.") from None
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite.")
    if parsed.is_integer():
        return int(parsed)
    return parsed


def _require_finite_float(value: Any, *, field_name: str) -> float:
    parsed = _require_finite_numeric(value, field_name=field_name)
    return float(parsed)


def _require_optional_finite_float(value: Any, *, field_name: str) -> float | None:
    if value is None:
        return None
    return _require_finite_float(value, field_name=field_name)


def _reject_observed_non_finite_numeric(series: pd.Series, *, label: str) -> None:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    invalid = ~np.isnan(values) & ~np.isfinite(values)
    if invalid.any():
        raise ValueError(f"{label} must contain only finite numeric values.")


def _require_nonnegative_int(value: Any, *, field_name: str) -> int:
    if _is_bool_like(value):
        raise ValueError(f"{field_name} must be an integer.")
    try:
        parsed = _index(value)
    except TypeError:
        raise ValueError(f"{field_name} must be an integer.") from None
    if parsed < 0:
        raise ValueError(f"{field_name} must be non-negative.")
    return parsed


def _normalize_group_label(value: Any) -> str:
    if value is _MISSING or value is None:
        raise ValueError("group is required.")
    if isinstance(value, str):
        stripped = value.strip()
    else:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = None
        if numeric is not None:
            if not math.isfinite(numeric):
                raise ValueError("group must be finite when numeric.")
            if numeric == 0:
                return "Control"
            if numeric == 1:
                return "Treated"
            raise ValueError("numeric group labels must be 0/1.")
        stripped = str(value)
    if stripped == "":
        raise ValueError("group is required.")
    normalized = stripped.lower()
    if normalized in {"0", "control"}:
        return "Control"
    if normalized in {"1", "treated"}:
        return "Treated"
    raise ValueError("group must be 'Control', 'Treated', 0, or 1.")


def _normalize_pattern_status(value: Any) -> str:
    if value is _MISSING or value is None:
        raise ValueError("status is required.")
    if isinstance(value, str):
        stripped = value.strip().lower()
    else:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = None
        if numeric is not None and math.isfinite(numeric):
            if numeric == 0:
                return "control"
            if numeric == 1:
                return "treated"
        stripped = str(value)
    if stripped in {"0", "control"}:
        return "control"
    if stripped in {"1", "treated"}:
        return "treated"
    raise ValueError("status must be either 'control' or 'treated'.")


@dataclass(frozen=True)
class _CovariateSpec:
    source: str
    categorical: bool = False


@dataclass(frozen=True)
class _InteractionCovariateSpec:
    """Interaction covariate: product of two or more atomic terms."""
    terms: tuple[_AnyCovarSpec, ...]


_AnyCovarSpec = _CovariateSpec | _InteractionCovariateSpec


def _covar_spec_sources(spec: _AnyCovarSpec) -> list[str]:
    """Return all source column names referenced by a covariate spec."""
    if isinstance(spec, _InteractionCovariateSpec):
        return [t.source for t in spec.terms]
    return [spec.source]


def _covar_spec_label(spec: _AnyCovarSpec) -> str:
    """Return a human-readable label for a covariate spec."""
    if isinstance(spec, _InteractionCovariateSpec):
        parts = []
        for t in spec.terms:
            parts.append(f"factor({t.source})" if t.categorical else t.source)
        return ":".join(parts)
    return f"factor({spec.source})" if spec.categorical else spec.source


def _parse_covariates(covariates: Sequence[str] | None) -> tuple[_AnyCovarSpec, ...]:
    if isinstance(covariates, (str, bytes)):
        raise TypeError("covariates must be a sequence of column names or factor(...) terms.")
    specs: list[_AnyCovarSpec] = []
    seen_sources: set[str] = set()
    for entry in covariates or ():
        try:
            interaction_specs = parse_covariate_expression(entry)
        except (TypeError, ValueError) as exc:
            msg = str(exc)
            if "blank entries" in msg or "must name a column" in msg or "must be a sequence" in msg:
                raise
            raise ValueError("covariates must contain only column names or factor(column) terms.") from None
        entry_new_sources: set[str] = set()
        is_star_expansion = len(interaction_specs) > 1
        for ispec in interaction_specs:
            if len(ispec.terms) == 1:
                comp = ispec.terms[0]
                if comp.source in seen_sources:
                    if is_star_expansion:
                        continue
                    raise ValueError("covariates must not contain duplicate column names.")
                seen_sources.add(comp.source)
                entry_new_sources.add(comp.source)
                specs.append(_CovariateSpec(source=comp.source, categorical=comp.categorical))
            else:
                for comp in ispec.terms:
                    if comp.source in seen_sources and comp.source not in entry_new_sources and not is_star_expansion:
                        raise ValueError("covariates must not contain duplicate column names.")
                    seen_sources.add(comp.source)
                    entry_new_sources.add(comp.source)
                terms = tuple(
                    _CovariateSpec(source=comp.source, categorical=comp.categorical)
                    for comp in ispec.terms
                )
                specs.append(_InteractionCovariateSpec(terms=terms))
    return tuple(specs)


def _resolve_option_value(
    *,
    option: Mapping[str, Any] | None,
    key: str,
    explicit_value: Any,
    default_value: Any,
) -> Any:
    if option is None or key not in option:
        return explicit_value
    option_value = option[key]
    if explicit_value != default_value and explicit_value != option_value:
        raise ValueError(f"{key} cannot differ between top-level arguments and option.")
    return option_value if explicit_value == default_value else explicit_value


def _normalize_runtime_surface(
    *,
    option: Mapping[str, Any] | None,
    data_type: str,
    is_panel: bool | None,
    lag: int | Sequence[int],
    thres: int | None,
    n_boot: int,
    id_cluster: str | None,
) -> tuple[str, int | Sequence[int], int | None, int, str | None]:
    data_type = _validate_surface_choice(
        data_type,
        field_name="data_type",
        supported={"panel", "rcs"},
    )
    if option is not None:
        if not isinstance(option, Mapping):
            raise TypeError("option must be a mapping when provided.")
        unknown_keys = sorted(set(option) - {"lag", "thres", "n_boot", "id_cluster"} - _OPTION_IGNORED_KEYS)
        if unknown_keys:
            raise ValueError(f"Unsupported option key(s): {', '.join(unknown_keys)}")
    resolved_data_type = data_type
    if is_panel is not None:
        if not _is_bool_like(is_panel):
            raise TypeError("is_panel must be a boolean when provided.")
        is_panel = bool(is_panel)
        implied_data_type = "panel" if is_panel else "rcs"
        if data_type != "panel" and data_type != implied_data_type:
            raise ValueError("is_panel conflicts with data_type.")
        resolved_data_type = implied_data_type

    resolved_lag = _resolve_option_value(
        option=option,
        key="lag",
        explicit_value=lag,
        default_value=1,
    )
    resolved_thres = _resolve_option_value(
        option=option,
        key="thres",
        explicit_value=thres,
        default_value=None,
    )
    resolved_n_boot = _resolve_option_value(
        option=option,
        key="n_boot",
        explicit_value=n_boot,
        default_value=30,
    )
    resolved_id_cluster = _resolve_option_value(
        option=option,
        key="id_cluster",
        explicit_value=id_cluster,
        default_value=None,
    )
    resolved_id_cluster = _validate_optional_column_name(
        resolved_id_cluster,
        field_name="id_cluster",
    )
    return resolved_data_type, resolved_lag, resolved_thres, resolved_n_boot, resolved_id_cluster


def _resolve_formula_surface(
    *,
    formula: str | DidFormulaSpec | None,
    outcome: str | None,
    treatment: str | None,
    post: str | None,
    covariates: Sequence[str] | None,
    design: str,
    data_type: str,
) -> tuple[str | None, str | None, str | None, Sequence[str] | None]:
    if formula is None:
        return outcome, treatment, post, covariates

    if any(value is not None for value in (outcome, treatment, post, covariates)):
        raise ValueError("formula cannot be combined with outcome, treatment, post, or covariates.")

    expected_panel = design == "sa" or data_type != "rcs"
    if isinstance(formula, DidFormulaSpec):
        spec = formula
    elif isinstance(formula, str):
        spec = parse_did_formula(formula, is_panel=expected_panel)
    else:
        raise TypeError("formula must be a string or DidFormulaSpec.")

    if expected_panel and spec.var_post is not None:
        raise ValueError("Panel and SA formulas must use outcome ~ treatment syntax.")
    if not expected_panel and spec.var_post is None:
        raise ValueError("Repeated cross-section formulas must specify treatment + post in that order.")

    return spec.var_outcome, spec.var_treat, spec.var_post, spec.covariate_terms or None


def _coerce_lags(lag: int | Sequence[int]) -> tuple[int, ...]:
    raw_values = tuple(lag) if isinstance(lag, Sequence) and not isinstance(lag, (str, bytes)) else (lag,)
    parsed: list[int] = []
    for value in raw_values:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError("lag must be an integer or a sequence of integers.")
        lag_value = int(value)
        if lag_value < 0:
            raise ValueError("lag must contain only non-negative diagnostic lags.")
        if lag_value in parsed:
            raise ValueError("lag sequence must not contain duplicate values.")
        parsed.append(lag_value)
    if not parsed:
        raise ValueError("lag must contain at least one non-negative diagnostic lag.")
    return tuple(parsed)


def _validate_n_boot(n_boot: int) -> int:
    if isinstance(n_boot, bool) or not isinstance(n_boot, (int, np.integer)) or n_boot < 2:
        raise ValueError("n_boot must be an integer greater than or equal to 2.")
    return int(n_boot)


def _validate_random_seed(random_seed: Any) -> int | None:
    if random_seed is None:
        return None
    if _is_bool_like(random_seed):
        raise ValueError("random_seed must be an integer between 0 and 2**32 - 1.")
    try:
        seed = _index(random_seed)
    except TypeError:
        raise ValueError("random_seed must be an integer between 0 and 2**32 - 1.") from None
    if seed < 0 or seed > 2**32 - 1:
        raise ValueError("random_seed must be an integer between 0 and 2**32 - 1.")
    return int(seed)


def _validate_optional_column_name(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a column name string when provided.")
    if not value.strip():
        raise ValueError(f"{field_name} must be a non-empty column name.")
    return value


def _validate_required_column_name(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a column name string.")
    if not value.strip():
        raise ValueError(f"{field_name} must be a non-empty column name.")
    return value


def _validate_surface_choice(value: Any, *, field_name: str, supported: set[str]) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if value not in supported:
        supported_values = " or ".join(f"'{entry}'" for entry in sorted(supported))
        raise ValueError(f"{field_name} must be {supported_values}.")
    return value


def _materialize_input_rows(data: Iterable[Mapping[str, Any]] | pd.DataFrame) -> list[dict[str, Any]]:
    if isinstance(data, pd.DataFrame):
        return data.to_dict(orient="records")
    return [dict(row) for row in data]


def _reject_missing_cluster_values(series: pd.Series, *, field_name: str) -> None:
    missing = pd.isna(series)
    if missing.any():
        first_missing = int(np.flatnonzero(missing.to_numpy())[0])
        raise ValueError(
            f"{field_name} must not contain missing values; found missing value in row {first_missing}."
        )


def _require_model_columns(
    rows: list[dict[str, Any]],
    *,
    covariates: tuple[_AnyCovarSpec, ...],
    id_cluster: str | None,
) -> None:
    for spec in covariates:
        for src in _covar_spec_sources(spec):
            require_column(
                rows,
                src,
                allow_missing=True,
                field_name=f"covariate '{src}'",
            )
    if id_cluster is not None:
        require_column(rows, id_cluster, field_name="id_cluster")


def _validate_covariates_disjoint_from_roles(
    covariates: tuple[_AnyCovarSpec, ...],
    *,
    outcome: str,
    treatment: str,
    time: str,
    unit_id: str | None,
    post: str | None,
) -> None:
    role_columns = {
        column
        for column in (outcome, treatment, time, unit_id, post)
        if column is not None
    }
    for spec in covariates:
        for src in _covar_spec_sources(spec):
            if src in role_columns:
                raise ValueError(
                    "covariates must not reuse outcome, treatment, time, unit_id, or post columns."
                )


def _validate_cluster_column_disjoint_from_roles(
    id_cluster: str | None,
    *,
    outcome: str,
    treatment: str,
    time: str,
    post: str | None,
) -> None:
    if id_cluster is None:
        return
    role_columns = {outcome, treatment, time}
    if post is not None:
        role_columns.add(post)
    if id_cluster in role_columns:
        raise ValueError("id_cluster must not reuse outcome, treatment, time, or post columns.")


def _build_prepared_data(
    rows: list[dict[str, Any]],
    *,
    outcome: str,
    treatment: str,
    time: str,
    unit_id: str | None,
    post: str | None,
    covariates: tuple[_AnyCovarSpec, ...],
    data_type: str,
    id_cluster: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    contract = normalize_design_data(
        rows,
        outcome=outcome,
        treatment=treatment,
        time=time,
        unit_id=unit_id,
        post=post,
        design="did",
        data_type=data_type,
    )
    cluster_column = id_cluster or contract.cluster_default
    _require_model_columns(rows, covariates=covariates, id_cluster=id_cluster)
    if id_cluster is not None:
        cluster_mode = "explicit"
    elif contract.cluster_default is not None:
        cluster_mode = "unit"
    else:
        cluster_mode = "observation"
    selected_columns = [outcome, treatment, time]
    if unit_id is not None:
        selected_columns.append(unit_id)
    if post is not None:
        selected_columns.append(post)
    for spec in covariates:
        for src in _covar_spec_sources(spec):
            selected_columns.append(src)
    if cluster_column is not None:
        selected_columns.append(cluster_column)
    frame = pd.DataFrame(rows)[list(dict.fromkeys(selected_columns))].copy()
    _reject_observed_non_finite_numeric(frame[outcome], label="outcome")
    for spec in covariates:
        if isinstance(spec, _CovariateSpec) and not spec.categorical and pd.api.types.is_numeric_dtype(frame[spec.source]):
            _reject_observed_non_finite_numeric(frame[spec.source], label=f"covariate '{spec.source}'")
        elif isinstance(spec, _InteractionCovariateSpec):
            for term in spec.terms:
                if not term.categorical and pd.api.types.is_numeric_dtype(frame[term.source]):
                    _reject_observed_non_finite_numeric(frame[term.source], label=f"covariate '{term.source}'")
    if cluster_mode != "observation":
        _reject_missing_cluster_values(
            frame[cluster_column if cluster_column is not None else unit_id],
            field_name="id_cluster" if cluster_mode == "explicit" else "unit_id",
        )
    time_order = {value: index + 1 for index, value in enumerate(contract.time_order)}
    frame["id_time"] = frame[time].map(time_order).astype(float)

    if data_type == "panel":
        if unit_id is None:
            raise ValueError("unit_id is required for panel data.")
        treat_year = float(frame.loc[frame[treatment] == 1, "id_time"].min())
        frame["Gi"] = frame.groupby(unit_id, sort=False)[treatment].transform("max").astype(float)
    else:
        if post is None:
            raise ValueError("post is required for repeated cross-section data.")
        treat_year = float(frame.loc[frame[post] == 1, "id_time"].min())
        frame["Gi"] = frame[treatment].astype(float)
    frame["id_time_std"] = frame["id_time"] - treat_year
    frame["outcome"] = pd.to_numeric(frame[outcome], errors="coerce")
    if data_type == "panel":
        frame["cluster_id"] = frame[cluster_column if cluster_column is not None else unit_id]
    elif cluster_column is not None:
        frame["cluster_id"] = frame[cluster_column]
    else:
        frame["cluster_id"] = np.arange(len(frame))
    effective_cluster_column = cluster_column or "_observation"
    metadata = contract.as_metadata()
    metadata.update(
        {
            "cluster_column": effective_cluster_column,
            "cluster_mode": cluster_mode,
            "clustvar": effective_cluster_column,
            "datatype": data_type,
            "n_clusters": int(pd.Series(frame["cluster_id"]).nunique(dropna=True)),
        }
    )
    return frame, metadata


def _placebo_subset(frame: pd.DataFrame, lag: int) -> pd.DataFrame:
    subset = frame.loc[frame["id_time_std"].isin((-lag - 1, -lag))].copy()
    subset["It"] = (subset["id_time_std"] >= -lag).astype(float)
    return subset


def _compute_interaction_product(columns: list[np.ndarray]) -> np.ndarray:
    """Compute cross-product of multiple column matrices."""
    result = columns[0]
    for i in range(1, len(columns)):
        next_cols = columns[i]
        n_rows = result.shape[0]
        n_new = result.shape[1] * next_cols.shape[1]
        new_result = np.empty((n_rows, n_new), dtype=float)
        col_idx = 0
        for j in range(result.shape[1]):
            for k in range(next_cols.shape[1]):
                new_result[:, col_idx] = result[:, j] * next_cols[:, k]
                col_idx += 1
        result = new_result
    return result


def _build_covariate_columns_diag(
    working: pd.DataFrame,
    spec: _AnyCovarSpec,
) -> np.ndarray:
    """Build design matrix columns for a single covariate spec."""
    if isinstance(spec, _InteractionCovariateSpec):
        parts: list[np.ndarray] = []
        for term in spec.terms:
            series = working[term.source]
            if term.categorical or not pd.api.types.is_numeric_dtype(series):
                dummies = pd.get_dummies(series.astype("string"), prefix=term.source, drop_first=True, dtype=float)
                if dummies.empty:
                    return np.empty((len(working), 0), dtype=float)
                parts.append(dummies.to_numpy(dtype=float))
            else:
                parts.append(pd.to_numeric(series, errors="coerce").to_numpy(dtype=float).reshape(-1, 1))
        if not parts:
            return np.empty((len(working), 0), dtype=float)
        return _compute_interaction_product(parts)
    # Plain covariate
    series = working[spec.source]
    if spec.categorical or not pd.api.types.is_numeric_dtype(series):
        dummies = pd.get_dummies(series.astype("string"), prefix=spec.source, drop_first=True, dtype=float)
        if dummies.empty:
            return np.empty((len(working), 0), dtype=float)
        return dummies.to_numpy(dtype=float)
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float).reshape(-1, 1)


def _build_design_matrix(
    subset: pd.DataFrame,
    *,
    covariates: tuple[_AnyCovarSpec, ...],
) -> tuple[np.ndarray, np.ndarray]:
    working = subset.copy()
    columns_to_keep = ["outcome", "Gi", "It"]
    for spec in covariates:
        columns_to_keep.extend(_covar_spec_sources(spec))
    columns_to_keep = list(dict.fromkeys(columns_to_keep))
    working = working[columns_to_keep].dropna().reset_index(drop=True)
    interaction = working["Gi"].astype(float) * working["It"].astype(float)
    matrix_parts = [
        np.ones((len(working), 1), dtype=float),
        working["Gi"].to_numpy(dtype=float).reshape(-1, 1),
        working["It"].to_numpy(dtype=float).reshape(-1, 1),
        interaction.to_numpy(dtype=float).reshape(-1, 1),
    ]
    for spec in covariates:
        cols = _build_covariate_columns_diag(working, spec)
        if cols.shape[1] > 0:
            matrix_parts.append(cols)
    matrix = np.hstack(matrix_parts)
    outcome = working["outcome"].to_numpy(dtype=float)
    return matrix, outcome


def _solve_interaction_effect(subset: pd.DataFrame, *, covariates: tuple[_AnyCovarSpec, ...]) -> float | None:
    if subset.empty:
        return None
    matrix, outcome = _build_design_matrix(subset, covariates=covariates)
    if len(outcome) == 0 or matrix.shape[0] < matrix.shape[1]:
        return None
    coefficients, _, rank, _ = np.linalg.lstsq(matrix, outcome, rcond=None)
    if rank < matrix.shape[1]:
        return None
    return float(coefficients[3])


def _compute_placebo_estimates(
    subset: pd.DataFrame,
    *,
    covariates: tuple[_AnyCovarSpec, ...],
) -> tuple[float | None, float | None]:
    estimate_raw = _solve_interaction_effect(subset, covariates=covariates)
    control_outcomes = subset.loc[(subset["It"] == 0) & (subset["Gi"] == 0), "outcome"].dropna()
    if len(control_outcomes) < 2:
        return estimate_raw, None
    control_sd = float(control_outcomes.std(ddof=1))
    if math.isnan(control_sd) or control_sd == 0:
        return estimate_raw, None
    standardized = subset.copy()
    standardized["outcome"] = (standardized["outcome"] - float(control_outcomes.mean())) / control_sd
    estimate_std = _solve_interaction_effect(standardized, covariates=covariates)
    return estimate_raw, estimate_std


def _safe_sample_sd(values: Sequence[float | None]) -> float | None:
    finite = np.asarray([value for value in values if value is not None and math.isfinite(value)], dtype=float)
    if len(finite) < 2:
        return None
    return float(np.std(finite, ddof=1))


def _weighted_average(weighted_values: Sequence[tuple[float, float | None]]) -> float | None:
    valid_pairs = [(weight, value) for weight, value in weighted_values if value is not None and math.isfinite(value)]
    if not valid_pairs:
        return None
    total_weight = sum(weight for weight, _ in valid_pairs)
    return sum(weight * value for weight, value in valid_pairs) / total_weight


def _build_sa_placebo_frame(frame: pd.DataFrame, *, period: int) -> pd.DataFrame:
    working = frame.loc[frame["id_time"] <= period].copy()
    working["Gi"] = working.groupby("id_subject", sort=False)["treatment"].transform("max").astype(float)
    working["It"] = (working["id_time"] >= period).astype(float)
    working["id_time_std"] = working["id_time"] - period
    working["_delta_unit_id"] = working["id_subject"]
    return _with_outcome_delta(working, data_type="panel")


def _compute_sa_placebo_results(
    frame: pd.DataFrame,
    *,
    requested_lags: tuple[int, ...],
    covariates: tuple[_AnyCovarSpec, ...],
    thres: int,
) -> tuple[dict[int, dict[str, float | None]], tuple[int, ...], tuple[int, ...], tuple[int, ...], pd.DataFrame]:
    status_matrix = _sa_status_matrix(frame)
    periods = _sa_periods_to_use(status_matrix, thres=thres)
    if not periods:
        raise ValueError(
            "No valid periods found for the SA design after applying thres(). Try reducing the threshold value."
        )

    subject_ids = _sa_subject_ids(status_matrix, periods)
    time_weights = _sa_time_weights(status_matrix, periods)
    raw_paths: dict[int, list[tuple[float, float | None]]] = {lag: [] for lag in requested_lags}
    std_paths: dict[int, list[tuple[float, float | None]]] = {lag: [] for lag in requested_lags}
    feasible_lags: set[int] = set()

    for period, period_subject_ids, time_weight in zip(periods, subject_ids, time_weights, strict=True):
        period_frame = _build_sa_placebo_frame(
            frame.loc[frame["id_subject"].isin(period_subject_ids)].copy(),
            period=period,
        )
        max_lag = int(abs(period_frame["id_time_std"].min()))
        period_feasible_lags = tuple(lag for lag in requested_lags if lag < max_lag)
        feasible_lags.update(period_feasible_lags)

        for lag in period_feasible_lags:
            try:
                estimate_raw, estimate_std = _compute_placebo_estimates(
                    _placebo_subset(period_frame, lag),
                    covariates=covariates,
                )
            except ValueError:
                estimate_raw, estimate_std = None, None
            raw_paths[lag].append((time_weight, estimate_raw))
            std_paths[lag].append((time_weight, estimate_std))

    weighted_rows = {
        lag: {
            "raw": _weighted_average(raw_paths[lag]),
            "std": _weighted_average(std_paths[lag]),
        }
        for lag in requested_lags
    }
    identified_lags = tuple(lag for lag in requested_lags if weighted_rows[lag]["raw"] is not None)
    unidentified_lags = tuple(lag for lag in requested_lags if lag not in identified_lags)
    feasible_lags_tuple = tuple(lag for lag in requested_lags if lag in feasible_lags)
    return weighted_rows, identified_lags, unidentified_lags, feasible_lags_tuple, status_matrix


def _compute_sa_bootstrap_placebo_draws(
    frame: pd.DataFrame,
    *,
    lags: tuple[int, ...],
    covariates: tuple[_AnyCovarSpec, ...],
    n_boot: int,
    random_seed: int | None,
    thres: int,
    cluster_mode: str,
) -> tuple[dict[int, list[float | None]], dict[int, list[float | None]]]:
    rng = np.random.RandomState(random_seed)
    raw_draws = {lag: [] for lag in lags}
    std_draws = {lag: [] for lag in lags}
    successful_iterations = 0
    attempts = 0
    max_attempts = max(n_boot * 20, 100)

    while successful_iterations < n_boot:
        attempts += 1
        if attempts > max_attempts:
            raise ValueError("At least two valid bootstrap samples are required for SA clustered placebo inference.")
        boot_frame = _bootstrap_sa_frame(frame, rng=rng, cluster_mode=cluster_mode)
        try:
            weighted_rows, _, _, _, _ = _compute_sa_placebo_results(
                boot_frame,
                requested_lags=lags,
                covariates=covariates,
                thres=thres,
            )
        except ValueError:
            continue

        successful_iterations += 1
        for lag in lags:
            raw_draws[lag].append(weighted_rows[lag]["raw"])
            std_draws[lag].append(weighted_rows[lag]["std"])

    return raw_draws, std_draws


def _build_sa_pattern_rows(frame: pd.DataFrame, status_matrix: pd.DataFrame) -> tuple["DidCheckPatternRow", ...]:
    first_treatment_period: dict[Any, float] = {}
    for subject in status_matrix.index:
        treated_periods = tuple(int(period) for period in status_matrix.columns if int(status_matrix.loc[subject, period]) == 1)
        first_treatment_period[subject] = float(min(treated_periods)) if treated_periods else math.inf

    ordered_subjects = sorted(status_matrix.index.tolist(), key=lambda subject: first_treatment_period[subject], reverse=True)
    unit_order = {subject: index + 1 for index, subject in enumerate(ordered_subjects)}
    plot_frame = frame.assign(unit_order=frame["id_subject"].map(unit_order)).sort_values(
        ["unit_order", "id_time"],
        kind="stable",
    )
    return tuple(
        DidCheckPatternRow(
            id_time=int(row.id_time),
            unit_order=int(row.unit_order),
            status="treated" if int(row.treatment) == 1 else "control",
        )
        for row in plot_frame.itertuples(index=False)
    )


def _coerce_metadata_lag_tuple(metadata: Mapping[str, Any], key: str) -> tuple[int, ...] | None:
    if key not in metadata:
        return None
    raw_value = metadata[key]
    raw_values = (
        tuple(raw_value)
        if isinstance(raw_value, Sequence) and not isinstance(raw_value, (str, bytes))
        else (raw_value,)
    )
    parsed = tuple(_require_nonnegative_int(value, field_name=f"metadata['{key}']") for value in raw_values)
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"metadata['{key}'] must not contain duplicate lag values.")
    return parsed


def _validate_lag_metadata(metadata: Mapping[str, Any], diagnostic_lags: tuple[int, ...]) -> None:
    diagnostic_lag_set = set(diagnostic_lags)
    requested_lags = _coerce_metadata_lag_tuple(metadata, "requested_lags")
    identified_lags = _coerce_metadata_lag_tuple(metadata, "identified_lags")
    unidentified_lags = _coerce_metadata_lag_tuple(metadata, "unidentified_lags")
    raw_only_lags = _coerce_metadata_lag_tuple(metadata, "raw_only_lags")

    if not diagnostic_lags and any(
        key in metadata for key in ("requested_lags", "identified_lags", "unidentified_lags")
    ):
        raise ValueError("lag metadata requires diagnostic_table rows.")
    if requested_lags is not None and not diagnostic_lag_set.issubset(set(requested_lags)):
        raise ValueError("metadata['requested_lags'] must include every diagnostic lag.")
    if identified_lags is not None and identified_lags != diagnostic_lags:
        raise ValueError("metadata['identified_lags'] must match diagnostic_table lag rows.")
    if unidentified_lags is not None:
        if requested_lags is None:
            raise ValueError("metadata['unidentified_lags'] requires metadata['requested_lags'].")
        if not set(unidentified_lags).issubset(set(requested_lags)):
            raise ValueError("metadata['unidentified_lags'] must be a subset of requested_lags.")
        if diagnostic_lag_set.intersection(unidentified_lags):
            raise ValueError("metadata['unidentified_lags'] must not include diagnostic lag rows.")
    if identified_lags is not None and unidentified_lags is not None:
        if set(identified_lags).intersection(unidentified_lags):
            raise ValueError("metadata['identified_lags'] and metadata['unidentified_lags'] must be disjoint.")
    if requested_lags is not None and identified_lags is not None and unidentified_lags is not None:
        if set(identified_lags).union(unidentified_lags) != set(requested_lags):
            raise ValueError("metadata['requested_lags'] must equal identified_lags plus unidentified_lags.")
    if raw_only_lags is not None and not set(raw_only_lags).issubset(diagnostic_lag_set):
        raise ValueError("metadata['raw_only_lags'] must be a subset of diagnostic lag rows.")


def _validate_raw_only_lag_metadata(
    metadata: Mapping[str, Any],
    diagnostic_table: tuple["DidCheckDiagnosticRow", ...],
) -> None:
    raw_only_lags = _coerce_metadata_lag_tuple(metadata, "raw_only_lags")
    actual_raw_only_lags = tuple(row.lag for row in diagnostic_table if row.estimate_std is None)
    if raw_only_lags is None:
        if actual_raw_only_lags:
            raise ValueError(
                "metadata['raw_only_lags'] is required when diagnostic rows have missing standardized estimates."
            )
        return

    if raw_only_lags != actual_raw_only_lags:
        raise ValueError("metadata['raw_only_lags'] must match diagnostic rows with missing standardized estimates.")


def _validate_bootstrap_metadata(metadata: Mapping[str, Any]) -> None:
    n_boot = metadata.get("n_boot")
    n_boot_requested = metadata.get("n_boot_requested")
    n_boot_realized = metadata.get("n_boot_realized")
    parsed_n_boot = None
    parsed_n_boot_requested = None
    parsed_n_boot_realized = None
    if n_boot is not None:
        parsed_n_boot = _require_nonnegative_int(
            n_boot,
            field_name="n_boot",
        )
    if n_boot_requested is not None:
        parsed_n_boot_requested = _require_nonnegative_int(
            n_boot_requested,
            field_name="n_boot_requested",
        )
    if n_boot_realized is not None:
        parsed_n_boot_realized = _require_nonnegative_int(
            n_boot_realized,
            field_name="n_boot_realized",
        )
    if (
        parsed_n_boot_requested is not None
        and parsed_n_boot_realized is not None
        and parsed_n_boot_requested < parsed_n_boot_realized
    ):
        raise ValueError("n_boot_requested must be greater than or equal to n_boot_realized.")
    if (
        parsed_n_boot is not None
        and parsed_n_boot_requested is not None
        and parsed_n_boot != parsed_n_boot_requested
    ):
        raise ValueError("n_boot must match n_boot_requested when both are present.")
    if (
        parsed_n_boot is not None
        and parsed_n_boot_realized is not None
        and parsed_n_boot < parsed_n_boot_realized
    ):
        raise ValueError("n_boot must be greater than or equal to n_boot_realized.")
    if parsed_n_boot_realized is None:
        requested_bootstrap_count = max(
            parsed_n_boot or 0,
            parsed_n_boot_requested or 0,
        )
        if requested_bootstrap_count > 0:
            raise ValueError(
                "n_boot_realized is required when bootstrap metadata requests positive draws."
            )


def _validate_diagnostic_family_metadata(
    metadata: Mapping[str, Any],
    *,
    trends_table: tuple["DidCheckTrendRow", ...],
    pattern_table: tuple["DidCheckPatternRow", ...],
) -> None:
    design = metadata.get("design")
    if design is not None:
        if not isinstance(design, str):
            raise TypeError("design metadata must be a string when present.")
        if design not in {"did", "sa"}:
            raise ValueError("design metadata must be 'did' or 'sa' when present.")

    branch = metadata.get("branch")
    if branch is not None:
        if not isinstance(branch, str):
            raise TypeError("branch metadata must be a string when present.")
        if branch.startswith("sa-") and design is None:
            raise ValueError("SA diagnostic branch metadata requires design metadata to be 'sa'.")
        if design == "sa" and not branch.startswith("sa-"):
            raise ValueError("branch metadata must match diagnostic design.")
        if design == "did" and branch.startswith("sa-"):
            raise ValueError("branch metadata must match diagnostic design.")

    if pattern_table:
        if design != "sa":
            raise ValueError("pattern_table requires design metadata to be 'sa'.")
        return

    if design == "sa":
        raise ValueError("SA diagnostic metadata requires pattern_table.")


def _metadata_with_inferred_raw_only_lags(
    metadata: Mapping[str, Any] | None,
    diagnostic_table: tuple["DidCheckDiagnosticRow", ...],
) -> dict[str, Any]:
    merged = dict(metadata or {})
    if "raw_only_lags" not in merged and all(
        isinstance(row, DidCheckDiagnosticRow) for row in diagnostic_table
    ):
        merged["raw_only_lags"] = tuple(row.lag for row in diagnostic_table if row.estimate_std is None)
    return merged


def _compute_bootstrap_draws(
    frame: pd.DataFrame,
    *,
    lags: tuple[int, ...],
    covariates: tuple[_AnyCovarSpec, ...],
    n_boot: int,
    random_seed: int | None,
) -> tuple[dict[int, list[float | None]], dict[int, list[float | None]]]:
    clusters = [group.copy() for _, group in frame.groupby("cluster_id", sort=False)]
    if not clusters:
        return {lag: [] for lag in lags}, {lag: [] for lag in lags}
    rng = np.random.RandomState(random_seed)
    raw_draws = {lag: [] for lag in lags}
    std_draws = {lag: [] for lag in lags}
    for _ in range(n_boot):
        sampled_ids = rng.randint(0, len(clusters), size=len(clusters))
        boot_frame = pd.concat([clusters[index] for index in sampled_ids], ignore_index=True)
        for lag in lags:
            raw_estimate, std_estimate = _compute_placebo_estimates(
                _placebo_subset(boot_frame, lag),
                covariates=covariates,
            )
            raw_draws[lag].append(raw_estimate)
            std_draws[lag].append(std_estimate)
    return raw_draws, std_draws


def _compute_trend_rows(frame: pd.DataFrame) -> tuple[DidCheckTrendRow, ...]:
    grouped = (
        frame.groupby(["id_time_std", "Gi"], sort=True)["outcome"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    rows: list[DidCheckTrendRow] = []
    for row in grouped.itertuples(index=False):
        rows.append(
            DidCheckTrendRow(
                time_to_treat=row.id_time_std,
                group="Treated" if row.Gi == 1 else "Control",
                outcome_mean=float(row.mean),
                outcome_sd=0.0 if pd.isna(row.std) else float(row.std),
                n_obs=int(row.count),
            )
        )
    return tuple(rows)


def _data_first_sa_did_check(
    data: Iterable[Mapping[str, Any]] | pd.DataFrame,
    *,
    outcome: str,
    treatment: str,
    time: str,
    unit_id: str,
    covariates: Sequence[str] | None,
    id_cluster: str | None,
    lag: int | Sequence[int],
    n_boot: int,
    random_seed: int | None,
    thres: int,
    metadata: dict[str, Any] | None,
) -> "DidCheckResult":
    rows = _materialize_input_rows(data)
    n_boot = _validate_n_boot(n_boot)
    covariate_specs = _parse_covariates(covariates)
    covariate_labels = tuple(_covar_spec_label(spec) for spec in covariate_specs)
    requested_lags = _coerce_lags(lag)
    frame, contract_metadata = _prepare_sa_frame(
        rows,
        outcome=outcome,
        treatment=treatment,
        time=time,
        unit_id=unit_id,
        covariates=covariate_specs,
        id_cluster=id_cluster,
    )
    weighted_rows, identified_lags, unidentified_lags, feasible_lags, status_matrix = _compute_sa_placebo_results(
        frame,
        requested_lags=requested_lags,
        covariates=covariate_specs,
        thres=thres,
    )
    if not identified_lags:
        if feasible_lags:
            raise ValueError("No identifiable lag() values remain for placebo tests.")
        raise ValueError("No feasible lag() values remain for placebo tests.")

    raw_draws, std_draws = _compute_sa_bootstrap_placebo_draws(
        frame,
        lags=requested_lags,
        covariates=covariate_specs,
        n_boot=n_boot,
        random_seed=random_seed,
        thres=thres,
        cluster_mode=str(contract_metadata["cluster_mode"]),
    )
    diagnostic_rows: list[DidCheckDiagnosticRow] = []
    bootstrap_identified_lags: list[int] = []
    raw_only_lags: list[int] = []
    for lag_value in identified_lags:
        std_error_raw = _safe_sample_sd(raw_draws[lag_value])
        if std_error_raw is None:
            continue
        bootstrap_identified_lags.append(lag_value)
        std_error_std = _safe_sample_sd(std_draws[lag_value])
        estimate_std = weighted_rows[lag_value]["std"] if std_error_std is not None else None
        estimate_raw = weighted_rows[lag_value]["raw"]
        eqci95_lb_std = None
        eqci95_ub_std = None
        if estimate_std is not None and std_error_std is not None:
            radius = max(
                abs(estimate_std - _Z_90 * std_error_std),
                abs(estimate_std + _Z_90 * std_error_std),
            )
            eqci95_lb_std = -radius
            eqci95_ub_std = radius
        else:
            raw_only_lags.append(lag_value)

        diagnostic_rows.append(
            DidCheckDiagnosticRow(
                lag=lag_value,
                estimate_std=estimate_std,
                std_error_std=std_error_std,
                estimate_raw=float(estimate_raw),
                std_error_raw=std_error_raw,
                eqci95_lb_std=eqci95_lb_std,
                eqci95_ub_std=eqci95_ub_std,
            )
        )
    if not bootstrap_identified_lags:
        raise ValueError(
            "No identifiable lag() values remain for placebo tests: fewer than two valid raw bootstrap draws are available for every requested lag."
        )
    bootstrap_identified_lags_tuple = tuple(bootstrap_identified_lags)
    bootstrap_unidentified_lags = tuple(lag for lag in requested_lags if lag not in bootstrap_identified_lags_tuple)

    merged_metadata = dict(contract_metadata)
    merged_metadata.update(
        {
            "branch": "sa-panel-check",
            "design": "sa",
            "outcome": outcome,
            "data_type": "panel",
            "requested_lags": requested_lags,
            "identified_lags": bootstrap_identified_lags_tuple,
            "unidentified_lags": bootstrap_unidentified_lags,
            "raw_only_lags": tuple(raw_only_lags),
            "random_seed": random_seed,
            "n_boot_requested": n_boot,
            "n_boot_realized": n_boot,
            "n_boot": n_boot,
            "covariates": covariate_labels,
            "thres": thres,
        }
    )
    merged_metadata = _merge_user_metadata_without_generated_overrides(
        merged_metadata,
        metadata,
    )

    return DidCheckResult(
        diagnostic_table=tuple(diagnostic_rows),
        trends_table=(),
        metadata=_freeze_metadata(merged_metadata),
        pattern_table=_build_sa_pattern_rows(frame, status_matrix),
    )


def _data_first_did_check(
    data: Iterable[Mapping[str, Any]] | pd.DataFrame,
    *,
    outcome: str,
    treatment: str,
    time: str,
    unit_id: str | None,
    post: str | None,
    covariates: Sequence[str] | None,
    data_type: str,
    id_cluster: str | None,
    lag: int | Sequence[int],
    n_boot: int,
    random_seed: int | None,
    metadata: dict[str, Any] | None,
) -> DidCheckResult:
    rows = _materialize_input_rows(data)
    n_boot = _validate_n_boot(n_boot)
    covariate_specs = _parse_covariates(covariates)
    covariate_labels = tuple(_covar_spec_label(spec) for spec in covariate_specs)
    frame, contract_metadata = _build_prepared_data(
        rows,
        outcome=outcome,
        treatment=treatment,
        time=time,
        unit_id=unit_id,
        post=post,
        covariates=covariate_specs,
        data_type=data_type,
        id_cluster=id_cluster,
    )
    requested_lags = _coerce_lags(lag)
    max_lag = int(abs(frame["id_time_std"].min()))
    feasible_lags = tuple(current_lag for current_lag in requested_lags if current_lag < max_lag)
    if not feasible_lags:
        raise ValueError("No feasible lag() values remain for placebo tests.")
    raw_draws, std_draws = _compute_bootstrap_draws(
        frame,
        lags=feasible_lags,
        covariates=covariate_specs,
        n_boot=n_boot,
        random_seed=random_seed,
    )
    diagnostic_rows: list[DidCheckDiagnosticRow] = []
    identified_lags: list[int] = []
    point_identified_lags: list[int] = []
    raw_only_lags: list[int] = []
    for current_lag in feasible_lags:
        subset = _placebo_subset(frame, current_lag)
        estimate_raw, estimate_std = _compute_placebo_estimates(subset, covariates=covariate_specs)
        if estimate_raw is None:
            continue
        point_identified_lags.append(current_lag)
        std_error_raw = _safe_sample_sd(raw_draws[current_lag])
        if std_error_raw is None:
            continue
        identified_lags.append(current_lag)
        std_error_std = _safe_sample_sd(std_draws[current_lag])
        eqci95_lb_std = None
        eqci95_ub_std = None
        if estimate_std is not None and std_error_std is not None:
            radius = max(
                abs(estimate_std - _Z_90 * std_error_std),
                abs(estimate_std + _Z_90 * std_error_std),
            )
            eqci95_lb_std = -radius
            eqci95_ub_std = radius
        else:
            raw_only_lags.append(current_lag)
        diagnostic_rows.append(
            DidCheckDiagnosticRow(
                lag=current_lag,
                estimate_std=estimate_std if std_error_std is not None else None,
                std_error_std=std_error_std,
                estimate_raw=estimate_raw,
                std_error_raw=std_error_raw,
                eqci95_lb_std=eqci95_lb_std,
                eqci95_ub_std=eqci95_ub_std,
            )
        )
    if not identified_lags:
        if not point_identified_lags:
            raise ValueError("No identifiable lag() values remain for placebo tests.")
        raise ValueError(
            "No identifiable lag() values remain for placebo tests: fewer than two valid raw bootstrap draws are available for every requested lag."
        )
    identified_lags_tuple = tuple(identified_lags)
    unidentified_lags = tuple(lag for lag in requested_lags if lag not in identified_lags_tuple)
    merged_metadata = dict(contract_metadata)
    merged_metadata.update(
        {
            "outcome": outcome,
            "branch": f"did-{data_type}-check",
            "design": "did",
            "data_type": data_type,
            "requested_lags": requested_lags,
            "identified_lags": identified_lags_tuple,
            "unidentified_lags": unidentified_lags,
            "raw_only_lags": tuple(raw_only_lags),
            "random_seed": random_seed,
            "n_boot_requested": n_boot,
            "n_boot_realized": n_boot,
            "n_boot": n_boot,
            "covariates": covariate_labels,
        }
    )
    merged_metadata = _merge_user_metadata_without_generated_overrides(
        merged_metadata,
        metadata,
    )
    return DidCheckResult(
        diagnostic_table=tuple(diagnostic_rows),
        trends_table=_compute_trend_rows(frame),
        metadata=_freeze_metadata(merged_metadata),
    )


@dataclass(frozen=True)
class DidCheckDiagnosticRow:
    """Truth-carrying diagnostic row with raw and standardized channels."""

    lag: int
    estimate_std: float | None
    std_error_std: float | None
    estimate_raw: float
    std_error_raw: float
    eqci95_lb_std: float | None
    eqci95_ub_std: float | None

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
        *,
        lag: int | None = None,
    ) -> DidCheckDiagnosticRow:
        mapping = _require_mapping(mapping, field_name="mapping")
        lag_value = lag if lag is not None else _coerce_required_int(
            _lookup_mapping_value(mapping, "lag"),
            field_name="lag",
        )
        return cls(
            lag=lag_value,
            estimate_std=_coerce_optional_float(
                _lookup_mapping_value(mapping, "estimate_std", "std_estimate", "estimate"),
                field_name="estimate_std",
            ),
            std_error_std=_coerce_optional_float(
                _lookup_mapping_value(mapping, "std_error_std", "std_error", "std.error"),
                field_name="std_error_std",
            ),
            estimate_raw=_coerce_required_float(
                _lookup_mapping_value(mapping, "estimate_raw", "raw_estimate", "estimate_orig"),
                field_name="estimate_raw",
            ),
            std_error_raw=_coerce_required_float(
                _lookup_mapping_value(mapping, "std_error_raw", "raw_std_error", "std_error_orig", "std.error_orig"),
                field_name="std_error_raw",
            ),
            eqci95_lb_std=_coerce_optional_float(
                _lookup_mapping_value(mapping, "eqci95_lb_std", "eqci95_lb", "EqCI95_LB"),
                field_name="eqci95_lb_std",
            ),
            eqci95_ub_std=_coerce_optional_float(
                _lookup_mapping_value(mapping, "eqci95_ub_std", "eqci95_ub", "EqCI95_UB"),
                field_name="eqci95_ub_std",
            ),
        )

    def __post_init__(self) -> None:
        try:
            lag = _require_nonnegative_int(self.lag, field_name="lag")
        except ValueError as exc:
            if str(exc) == "lag must be non-negative.":
                raise ValueError("lag must be a non-negative diagnostic lag.") from None
            raise
        estimate_std = _require_optional_finite_float(self.estimate_std, field_name="estimate_std")
        std_error_std = _require_optional_finite_float(self.std_error_std, field_name="std_error_std")
        estimate_raw = _require_finite_float(self.estimate_raw, field_name="estimate_raw")
        std_error_raw = _require_finite_float(self.std_error_raw, field_name="std_error_raw")
        if std_error_std is not None and std_error_std < 0:
            raise ValueError("std_error_std must be non-negative.")
        if std_error_raw < 0:
            raise ValueError("std_error_raw must be non-negative.")
        eqci95_lb_std = _require_optional_finite_float(self.eqci95_lb_std, field_name="eqci95_lb_std")
        eqci95_ub_std = _require_optional_finite_float(self.eqci95_ub_std, field_name="eqci95_ub_std")
        standardized_channel = (estimate_std, std_error_std)
        if any(value is None for value in standardized_channel) and any(value is not None for value in standardized_channel):
            raise ValueError("Standardized estimate and std_error must be jointly present or jointly missing.")
        eqci_bounds = (eqci95_lb_std, eqci95_ub_std)
        if any(bound is None for bound in eqci_bounds) and any(bound is not None for bound in eqci_bounds):
            raise ValueError("EqCI95 bounds must be jointly present or jointly missing.")
        if any(bound is not None for bound in eqci_bounds) and any(value is None for value in standardized_channel):
            raise ValueError("EqCI95 bounds require standardized estimate and std_error.")
        if all(value is not None for value in standardized_channel) and any(bound is None for bound in eqci_bounds):
            raise ValueError("Standardized estimate and std_error require EqCI95 bounds.")
        if eqci95_lb_std is not None and eqci95_ub_std is not None:
            expected_radius = max(
                abs(estimate_std - _Z_90 * std_error_std),  # type: ignore[operator]
                abs(estimate_std + _Z_90 * std_error_std),  # type: ignore[operator]
            )
            if not math.isclose(eqci95_lb_std, -expected_radius) or not math.isclose(
                eqci95_ub_std,
                expected_radius,
            ):
                raise ValueError("EqCI95 bounds must match the standardized estimate and std_error.")
        object.__setattr__(self, "lag", lag)
        object.__setattr__(self, "estimate_std", estimate_std)
        object.__setattr__(self, "std_error_std", std_error_std)
        object.__setattr__(self, "estimate_raw", estimate_raw)
        object.__setattr__(self, "std_error_raw", std_error_raw)
        object.__setattr__(self, "eqci95_lb_std", eqci95_lb_std)
        object.__setattr__(self, "eqci95_ub_std", eqci95_ub_std)

    def summary_row(self) -> dict[str, float | int | None]:
        return {
            "lag": self.lag,
            "estimate_raw": self.estimate_raw,
            "std_error_raw": self.std_error_raw,
            "eqci95_lb_std": self.eqci95_lb_std,
            "eqci95_ub_std": self.eqci95_ub_std,
        }

    def diagnostic_row(self) -> dict[str, float | int | None]:
        return {
            "lag": self.lag,
            "estimate_std": self.estimate_std,
            "std_error_std": self.std_error_std,
            "estimate_raw": self.estimate_raw,
            "std_error_raw": self.std_error_raw,
            "eqci95_lb_std": self.eqci95_lb_std,
            "eqci95_ub_std": self.eqci95_ub_std,
        }

    def placebo_plot_row(self) -> dict[str, float | int] | None:
        if (
            self.estimate_std is None
            or self.std_error_std is None
            or self.eqci95_lb_std is None
            or self.eqci95_ub_std is None
        ):
            return None

        return {
            "lag": self.lag,
            "time_to_treat": -self.lag,
            "estimate_std": self.estimate_std,
            "std_error_std": self.std_error_std,
            "eqci95_lb_std": self.eqci95_lb_std,
            "eqci95_ub_std": self.eqci95_ub_std,
        }


@dataclass(frozen=True)
class DidCheckTrendRow:
    """Trend plotting row matching the R/Stata data-first output."""

    time_to_treat: float | int
    group: str
    outcome_mean: float
    outcome_sd: float
    n_obs: int | None = None

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> DidCheckTrendRow:
        mapping = _require_mapping(mapping, field_name="mapping")
        return cls(
            time_to_treat=_coerce_numeric(
                _lookup_mapping_value(mapping, "time_to_treat"),
                field_name="time_to_treat",
            ),
            group=_normalize_group_label(_lookup_mapping_value(mapping, "group", "Gi")),
            outcome_mean=_coerce_required_float(
                _lookup_mapping_value(mapping, "outcome_mean", "plot_mean"),
                field_name="outcome_mean",
            ),
            outcome_sd=_coerce_required_float(
                _lookup_mapping_value(mapping, "outcome_sd", "plot_sd", "std.error"),
                field_name="outcome_sd",
            ),
            n_obs=_coerce_optional_int(_lookup_mapping_value(mapping, "n_obs"), field_name="n_obs"),
        )

    def __post_init__(self) -> None:
        time_to_treat = _require_finite_numeric(self.time_to_treat, field_name="time_to_treat")
        group = _normalize_group_label(self.group)
        outcome_mean = _require_finite_float(self.outcome_mean, field_name="outcome_mean")
        outcome_sd = _require_finite_float(self.outcome_sd, field_name="outcome_sd")
        if outcome_sd < 0:
            raise ValueError("outcome_sd must be non-negative.")
        n_obs = None if self.n_obs is None else _require_nonnegative_int(self.n_obs, field_name="n_obs")
        if n_obs is not None and n_obs < 1:
            raise ValueError("n_obs must be positive.")
        object.__setattr__(self, "time_to_treat", time_to_treat)
        object.__setattr__(self, "group", group)
        object.__setattr__(self, "outcome_mean", outcome_mean)
        object.__setattr__(self, "outcome_sd", outcome_sd)
        object.__setattr__(self, "n_obs", n_obs)

    def plot_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "time_to_treat": self.time_to_treat,
            "group": self.group,
            "outcome_mean": self.outcome_mean,
            "outcome_sd": self.outcome_sd,
            "ci90_lb": self.outcome_mean - _Z_90 * self.outcome_sd,
            "ci90_ub": self.outcome_mean + _Z_90 * self.outcome_sd,
        }
        if self.n_obs is not None:
            row["n_obs"] = self.n_obs
        return row


@dataclass(frozen=True)
class DidCheckPatternRow:
    """Pattern row for staggered-adoption treatment timing heatmaps."""

    id_time: int | float
    unit_order: int
    status: str

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "DidCheckPatternRow":
        mapping = _require_mapping(mapping, field_name="mapping")
        return cls(
            id_time=_coerce_numeric(_lookup_mapping_value(mapping, "id_time", "time"), field_name="id_time"),
            unit_order=_coerce_required_int(_lookup_mapping_value(mapping, "unit_order"), field_name="unit_order"),
            status=_normalize_pattern_status(_lookup_mapping_value(mapping, "status", "treatment")),
        )

    def __post_init__(self) -> None:
        id_time = _require_finite_numeric(self.id_time, field_name="id_time")
        unit_order = _require_nonnegative_int(self.unit_order, field_name="unit_order")
        if unit_order < 1:
            raise ValueError("unit_order must be positive.")
        status = _normalize_pattern_status(self.status)
        object.__setattr__(self, "id_time", id_time)
        object.__setattr__(self, "unit_order", unit_order)
        object.__setattr__(self, "status", status)

    def plot_row(self) -> dict[str, Any]:
        return {
            "id_time": self.id_time,
            "unit_order": self.unit_order,
            "status": self.status,
        }


@dataclass(frozen=True)
class DidCheckResult:
    """Diagnostic result for summary tables and plotting rows."""

    diagnostic_table: tuple[DidCheckDiagnosticRow, ...]
    trends_table: tuple[DidCheckTrendRow, ...]
    metadata: MappingProxyType
    pattern_table: tuple[DidCheckPatternRow, ...] = ()

    def __post_init__(self) -> None:
        diagnostic_table = tuple(self.diagnostic_table)
        trends_table = tuple(self.trends_table)
        pattern_table = tuple(self.pattern_table)
        metadata = _freeze_metadata(self.metadata)
        if any(not isinstance(row, DidCheckDiagnosticRow) for row in diagnostic_table):
            raise TypeError("diagnostic_table must contain DidCheckDiagnosticRow instances.")
        if any(not isinstance(row, DidCheckTrendRow) for row in trends_table):
            raise TypeError("trends_table must contain DidCheckTrendRow instances.")
        if any(not isinstance(row, DidCheckPatternRow) for row in pattern_table):
            raise TypeError("pattern_table must contain DidCheckPatternRow instances.")
        if trends_table and pattern_table:
            raise ValueError("DidCheckResult cannot mix trend and pattern plotting rows.")
        _validate_diagnostic_family_metadata(
            metadata,
            trends_table=trends_table,
            pattern_table=pattern_table,
        )
        diagnostic_lags: set[int] = set()
        for row in diagnostic_table:
            if row.lag in diagnostic_lags:
                raise ValueError("diagnostic_table must not contain duplicate lag rows.")
            diagnostic_lags.add(row.lag)
        trend_keys: set[tuple[int | float, str]] = set()
        for row in trends_table:
            key = (row.time_to_treat, row.group)
            if key in trend_keys:
                raise ValueError("trends_table must not contain duplicate time/group rows.")
            trend_keys.add(key)
        pattern_keys: set[tuple[int | float, int]] = set()
        for row in pattern_table:
            key = (row.id_time, row.unit_order)
            if key in pattern_keys:
                raise ValueError("pattern_table must not contain duplicate time/unit rows.")
            pattern_keys.add(key)
        _validate_lag_metadata(metadata, tuple(row.lag for row in diagnostic_table))
        _validate_raw_only_lag_metadata(metadata, diagnostic_table)
        _validate_bootstrap_metadata(metadata)
        object.__setattr__(self, "diagnostic_table", diagnostic_table)
        object.__setattr__(self, "trends_table", trends_table)
        object.__setattr__(self, "pattern_table", pattern_table)
        object.__setattr__(self, "metadata", metadata)

    def diagnostic_rows(self) -> tuple[dict[str, float | int | None], ...]:
        return tuple(row.diagnostic_row() for row in self.diagnostic_table)

    def summary_rows(self) -> tuple[dict[str, float | int | None], ...]:
        return tuple(row.summary_row() for row in self.diagnostic_table)

    def named_plot_rows(self) -> dict[str, tuple[dict[str, Any], ...]]:
        placebo_rows = tuple(
            row for row in (diagnostic_row.placebo_plot_row() for diagnostic_row in self.diagnostic_table) if row
        )
        plot_rows: dict[str, tuple[dict[str, Any], ...]] = {"placebo": placebo_rows}
        if self.pattern_table:
            plot_rows["pattern"] = tuple(pattern_row.plot_row() for pattern_row in self.pattern_table)
            return plot_rows

        plot_rows["trends"] = tuple(trend_row.plot_row() for trend_row in self.trends_table)
        return plot_rows

    def named_plot_payloads(self) -> dict[str, tuple[dict[str, Any], ...]]:
        return self.named_plot_rows()

    def to_serialized_result(self) -> dict[str, Any]:
        """Return detached serialized diagnostic, plot, and metadata records."""

        return {
            "diagnostics": self.diagnostic_rows(),
            "summary": self.summary_rows(),
            "plots": self.named_plot_rows(),
            "metadata": _thaw_metadata_value(self.metadata),
        }

    def as_payload(self) -> dict[str, Any]:
        """Compatibility alias for :meth:`to_serialized_result`."""

        return self.to_serialized_result()

    def to_diagnostics_frame(self) -> pd.DataFrame:
        """Return detached diagnostic rows as a stable pandas DataFrame."""

        return pd.DataFrame.from_records(
            self.diagnostic_rows(),
            columns=_DIAGNOSTIC_COLUMNS,
        )

    def to_summary_frame(self) -> pd.DataFrame:
        """Return detached diagnostic summary rows as a stable pandas DataFrame."""

        return pd.DataFrame.from_records(
            self.summary_rows(),
            columns=_SUMMARY_COLUMNS,
        )

    def to_placebo_frame(self) -> pd.DataFrame:
        """Return detached standardized placebo plot rows as a stable DataFrame."""

        return pd.DataFrame.from_records(
            self.named_plot_rows()["placebo"],
            columns=_PLACEBO_COLUMNS,
        )

    def to_trends_frame(self) -> pd.DataFrame:
        """Return detached trend plot rows as a stable pandas DataFrame."""

        return pd.DataFrame.from_records(
            (row.plot_row() for row in self.trends_table),
            columns=_TREND_COLUMNS,
        )

    def to_pattern_frame(self) -> pd.DataFrame:
        """Return detached staggered-adoption pattern rows as a stable DataFrame."""

        return pd.DataFrame.from_records(
            (row.plot_row() for row in self.pattern_table),
            columns=_PATTERN_COLUMNS,
        )


def did_check(
    *,
    diagnostic_rows: list[DidCheckDiagnosticRow] | tuple[DidCheckDiagnosticRow, ...] | None = None,
    trends_rows: list[DidCheckTrendRow] | tuple[DidCheckTrendRow, ...] | None = None,
    pattern_rows: list[DidCheckPatternRow] | tuple[DidCheckPatternRow, ...] | None = None,
    metadata: dict[str, Any] | None = None,
    data: Iterable[Mapping[str, Any]] | pd.DataFrame | None = None,
    formula: str | DidFormulaSpec | None = None,
    outcome: str | None = None,
    treatment: str | None = None,
    time: str | None = None,
    unit_id: str | None = None,
    post: str | None = None,
    design: str = "did",
    covariates: Sequence[str] | None = None,
    data_type: str = "panel",
    id_cluster: str | None = None,
    lag: int | Sequence[int] = 1,
    thres: int | None = None,
    n_boot: int = 30,
    random_seed: int | None = None,
    verbose: int = 1,
    option: Mapping[str, Any] | None = None,
    is_panel: bool | None = None,
) -> DidCheckResult:
    # Validate verbose parameter
    if not isinstance(verbose, int) or verbose < 0 or verbose > 2:
        verbose = 1  # fallback to default

    # verbose=0: suppress DidWarning via context manager
    from .errors import DidWarning
    _warn_ctx = None
    if verbose == 0:
        _warn_ctx = warnings.catch_warnings()
        _warn_ctx.__enter__()
        warnings.filterwarnings("ignore", category=DidWarning)

    try:
        return _did_check_inner(
            diagnostic_rows=diagnostic_rows,
            trends_rows=trends_rows,
            pattern_rows=pattern_rows,
            metadata=metadata,
            data=data,
            formula=formula,
            outcome=outcome,
            treatment=treatment,
            time=time,
            unit_id=unit_id,
            post=post,
            design=design,
            covariates=covariates,
            data_type=data_type,
            id_cluster=id_cluster,
            lag=lag,
            thres=thres,
            n_boot=n_boot,
            random_seed=random_seed,
            option=option,
            is_panel=is_panel,
        )
    finally:
        if _warn_ctx is not None:
            _warn_ctx.__exit__(None, None, None)


def _did_check_inner(
    *,
    diagnostic_rows: list[DidCheckDiagnosticRow] | tuple[DidCheckDiagnosticRow, ...] | None = None,
    trends_rows: list[DidCheckTrendRow] | tuple[DidCheckTrendRow, ...] | None = None,
    pattern_rows: list[DidCheckPatternRow] | tuple[DidCheckPatternRow, ...] | None = None,
    metadata: dict[str, Any] | None = None,
    data: Iterable[Mapping[str, Any]] | pd.DataFrame | None = None,
    formula: str | DidFormulaSpec | None = None,
    outcome: str | None = None,
    treatment: str | None = None,
    time: str | None = None,
    unit_id: str | None = None,
    post: str | None = None,
    design: str = "did",
    covariates: Sequence[str] | None = None,
    data_type: str = "panel",
    id_cluster: str | None = None,
    lag: int | Sequence[int] = 1,
    thres: int | None = None,
    n_boot: int = 30,
    random_seed: int | None = None,
    option: Mapping[str, Any] | None = None,
    is_panel: bool | None = None,
) -> DidCheckResult:
    """Inner implementation of did_check() without verbose wrapper."""
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping.")
    data_type, lag, thres, n_boot, id_cluster = _normalize_runtime_surface(
        option=option,
        data_type=data_type,
        is_panel=is_panel,
        lag=lag,
        thres=thres,
        n_boot=n_boot,
        id_cluster=id_cluster,
    )
    design = _validate_surface_choice(design, field_name="design", supported={"did", "sa"})
    validated_thres = _validate_thres(thres, design=design)
    validated_random_seed = _validate_random_seed(random_seed)
    outcome, treatment, post, covariates = _resolve_formula_surface(
        formula=formula,
        outcome=outcome,
        treatment=treatment,
        post=post,
        covariates=covariates,
        design=design,
        data_type=data_type,
    )
    if data is not None:
        lag = _coerce_lags(lag)
        if outcome is None or treatment is None or time is None:
            raise ValueError("outcome, treatment, and time are required for data-driven did_check.")
        outcome = _validate_required_column_name(outcome, field_name="outcome")
        treatment = _validate_required_column_name(treatment, field_name="treatment")
        time = _validate_required_column_name(time, field_name="time")
        unit_id = _validate_optional_column_name(unit_id, field_name="unit_id")
        post = _validate_optional_column_name(post, field_name="post")
        # Auto-encode string columns before validation
        if isinstance(data, pd.DataFrame):
            data, _encoding_metadata = auto_encode_string_columns(
                data,
                unit_id=unit_id,
                id_cluster=id_cluster,
            )
        covariate_specs = _parse_covariates(covariates)
        _validate_covariates_disjoint_from_roles(
            covariate_specs,
            outcome=outcome,
            treatment=treatment,
            time=time,
            unit_id=unit_id,
            post=post,
        )
        _validate_cluster_column_disjoint_from_roles(
            id_cluster,
            outcome=outcome,
            treatment=treatment,
            time=time,
            post=post,
        )
        if design == "sa":
            if data_type != "panel":
                raise ValueError("data_type must be 'panel' for design='sa'.")
            if unit_id is None:
                raise ValueError("unit_id is required for design='sa'.")
            return _data_first_sa_did_check(
                data,
                outcome=outcome,
                treatment=treatment,
                time=time,
                unit_id=unit_id,
                covariates=tuple(_covar_spec_label(spec) for spec in covariate_specs),
                id_cluster=id_cluster,
                lag=lag,
                n_boot=n_boot,
                random_seed=validated_random_seed,
                thres=validated_thres,
                metadata=metadata,
            )
        if data_type not in {"panel", "rcs"}:
            raise ValueError("data_type must be either 'panel' or 'rcs'.")
        return _data_first_did_check(
            data,
            outcome=outcome,
            treatment=treatment,
            time=time,
            unit_id=unit_id,
            post=post,
            covariates=tuple(_covar_spec_label(spec) for spec in covariate_specs),
            data_type=data_type,
            id_cluster=id_cluster,
            lag=lag,
            n_boot=n_boot,
            random_seed=validated_random_seed,
            metadata=metadata,
        )
    if diagnostic_rows is None or (trends_rows is None and pattern_rows is None):
        raise ValueError("diagnostic_rows and at least one of trends_rows or pattern_rows are required when data is not provided.")
    diagnostic_table = tuple(diagnostic_rows)
    return DidCheckResult(
        diagnostic_table=diagnostic_table,
        trends_table=tuple(trends_rows or ()),
        metadata=_freeze_metadata(_metadata_with_inferred_raw_only_lags(metadata, diagnostic_table)),
        pattern_table=tuple(pattern_rows or ()),
    )


__all__ = [
    "DidCheckDiagnosticRow",
    "DidCheckPatternRow",
    "DidCheckResult",
    "DidCheckTrendRow",
    "did_check",
]
