from __future__ import annotations

import math
import os
import warnings
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from operator import index as _index
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from ..core.data_contracts import normalize_design_data
from ..core.validation import require_column
from ..errors import (
    DidValueError,
    DidRuntimeError,
    DidWarning,
    ErrorCode,
    WarningCode,
    did_warn,
)
from ..formula import (
    DidFormulaSpec,
    InteractionSpec,
    did_formula as parse_did_formula,
    parse_covariate_expression,
    parse_covariate_term,
)
from ..results.objects import DidBootstrapDraw, DidBootstrapDrawK, DidEstimateRow, DidResult


_Z_975 = NormalDist().inv_cdf(0.975)


def _validate_level(level: int) -> int:
    if not isinstance(level, int) or level < 50 or level > 99:
        raise DidValueError(
            ErrorCode.E002,
            "level must be an integer between 50 and 99.",
            context={"level": level},
        )
    return level


def _is_bool_like(value: Any) -> bool:
    return isinstance(value, bool) or (
        type(value).__module__ == "numpy"
        and type(value).__name__ in {"bool", "bool_"}
    )


@dataclass(frozen=True)
class _CovariateSpec:
    source: str
    categorical: bool = False


@dataclass(frozen=True)
class _InteractionCovariateSpec:
    """Interaction covariate: product of two or more atomic terms."""
    terms: tuple[_CovariateSpec, ...]


# Union type for covariate specifications
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


def _build_covariate_labels(specs: tuple[_AnyCovarSpec, ...]) -> tuple[str, ...]:
    """Build human-readable labels for all covariate specs."""
    return tuple(_covar_spec_label(spec) for spec in specs)


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
            raise DidValueError(
                ErrorCode.E002,
                "covariates must contain only column names or factor(column) terms.",
                context={"entry": entry},
            ) from None
        # Track sources introduced by this single entry (for * expansion dedup)
        entry_new_sources: set[str] = set()
        is_star_expansion = len(interaction_specs) > 1
        for ispec in interaction_specs:
            if len(ispec.terms) == 1:
                comp = ispec.terms[0]
                if comp.source in seen_sources:
                    if is_star_expansion:
                        # Silently skip duplicate main effects from * expansion
                        continue
                    raise DidValueError(
                        ErrorCode.E002,
                        "covariates must not contain duplicate column names.",
                        context={"duplicate": comp.source},
                    )
                seen_sources.add(comp.source)
                entry_new_sources.add(comp.source)
                specs.append(_CovariateSpec(source=comp.source, categorical=comp.categorical))
            else:
                for comp in ispec.terms:
                    if comp.source in seen_sources and comp.source not in entry_new_sources and not is_star_expansion:
                        raise DidValueError(
                            ErrorCode.E002,
                            "covariates must not contain duplicate column names.",
                            context={"duplicate": comp.source},
                        )
                    seen_sources.add(comp.source)
                    entry_new_sources.add(comp.source)
                terms = tuple(
                    _CovariateSpec(source=comp.source, categorical=comp.categorical)
                    for comp in ispec.terms
                )
                specs.append(_InteractionCovariateSpec(terms=terms))
    return tuple(specs)


_OPTION_IGNORED_KEYS = frozenset({"parallel"})


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
        raise DidValueError(
            ErrorCode.E002,
            f"{key} cannot differ between top-level arguments and option.",
            context={"key": key, "explicit_value": explicit_value, "option_value": option_value},
        )
    return option_value if explicit_value == default_value else explicit_value


def _normalize_runtime_surface(
    *,
    option: Mapping[str, Any] | None,
    data_type: str,
    is_panel: bool | None,
    lead: int | Sequence[int],
    thres: int | None,
    n_boot: int,
    se_boot: bool | None,
    id_cluster: str | None,
) -> tuple[str, int | Sequence[int], int | None, int, bool | None, str | None]:
    data_type = _validate_surface_choice(
        data_type,
        field_name="data_type",
        supported={"panel", "rcs"},
    )
    if option is not None:
        if not isinstance(option, Mapping):
            raise TypeError("option must be a mapping when provided.")
        unknown_keys = sorted(set(option) - {"lead", "thres", "n_boot", "se_boot", "id_cluster"} - _OPTION_IGNORED_KEYS)
        if unknown_keys:
            raise DidValueError(
                ErrorCode.E002,
                f"Unsupported option key(s): {', '.join(unknown_keys)}",
                context={"unsupported_keys": unknown_keys},
            )
    resolved_data_type = data_type
    if is_panel is not None:
        if not _is_bool_like(is_panel):
            raise TypeError("is_panel must be a boolean when provided.")
        is_panel = bool(is_panel)
        implied_data_type = "panel" if is_panel else "rcs"
        if data_type != "panel" and data_type != implied_data_type:
            raise DidValueError(
                ErrorCode.E002,
                "is_panel conflicts with data_type.",
                context={"is_panel": is_panel, "data_type": data_type, "implied_data_type": implied_data_type},
            )
        resolved_data_type = implied_data_type

    resolved_lead = _resolve_option_value(
        option=option,
        key="lead",
        explicit_value=lead,
        default_value=0,
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
    resolved_se_boot = _resolve_option_value(
        option=option,
        key="se_boot",
        explicit_value=se_boot,
        default_value=None,
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
    return (
        resolved_data_type,
        resolved_lead,
        resolved_thres,
        resolved_n_boot,
        resolved_se_boot,
        resolved_id_cluster,
    )


def _resolve_formula_surface(
    *,
    formula: str | DidFormulaSpec | None,
    outcome: str | None,
    treatment: str | None,
    post: str | None,
    covariates: Sequence[str] | None,
    design: str,
    data_type: str,
) -> tuple[str, str, str | None, Sequence[str] | None]:
    if formula is None:
        if outcome is None or treatment is None:
            raise TypeError("outcome and treatment are required when formula is not provided.")
        return outcome, treatment, post, covariates

    if any(value is not None for value in (outcome, treatment, post, covariates)):
        raise DidValueError(
            ErrorCode.E002,
            "formula cannot be combined with outcome, treatment, post, or covariates.",
            context={"formula": formula},
        )

    expected_panel = design == "sa" or data_type != "rcs"
    if isinstance(formula, DidFormulaSpec):
        spec = formula
    elif isinstance(formula, str):
        spec = parse_did_formula(formula, is_panel=expected_panel)
    else:
        raise TypeError("formula must be a string or DidFormulaSpec.")

    if expected_panel and spec.var_post is not None:
        raise DidValueError(
            ErrorCode.E002,
            "Panel and SA formulas must use outcome ~ treatment syntax.",
            context={"formula": str(formula), "data_type": data_type},
        )
    if not expected_panel and spec.var_post is None:
        raise DidValueError(
            ErrorCode.E002,
            "Repeated cross-section formulas must specify treatment + post in that order.",
            context={"formula": str(formula), "data_type": data_type},
        )

    return spec.var_outcome, spec.var_treat, spec.var_post, spec.covariate_terms or None


def _validate_requested_leads(lead: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(lead, Sequence) and not isinstance(lead, (str, bytes)):
        parsed = tuple(_validate_single_lead(value) for value in lead)
    else:
        parsed = (_validate_single_lead(lead),)

    requested: list[int] = []
    for current_lead in parsed:
        if current_lead < 0:
            raise DidValueError(
                ErrorCode.E002,
                "lead must be non-negative.",
                context={"lead": current_lead},
            )
        if current_lead in requested:
            raise DidValueError(
                ErrorCode.E002,
                "lead sequence must not contain duplicate values.",
                context={"lead": current_lead, "seen": list(requested)},
            )
        requested.append(current_lead)
    if not requested:
        raise DidValueError(
            ErrorCode.E002,
            "lead must contain at least one non-negative lead.",
        )
    return tuple(requested)


def _validate_single_lead(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise DidValueError(
            ErrorCode.E002,
            "lead must be an integer or a sequence of integers.",
            context={"value": value, "type": type(value).__name__},
        )
    return int(value)


def _validate_n_boot(n_boot: int) -> int:
    if isinstance(n_boot, bool) or not isinstance(n_boot, (int, np.integer)) or n_boot < 2:
        raise DidValueError(
            ErrorCode.E002,
            "n_boot must be an integer greater than or equal to 2.",
            context={"n_boot": n_boot},
        )
    return int(n_boot)


def _validate_random_seed(random_seed: Any) -> int | None:
    if random_seed is None:
        return None
    if _is_bool_like(random_seed):
        raise DidValueError(
            ErrorCode.E002,
            "random_seed must be an integer between 0 and 2**32 - 1.",
            context={"random_seed": random_seed},
        )
    try:
        seed = _index(random_seed)
    except TypeError:
        raise DidValueError(
            ErrorCode.E002,
            "random_seed must be an integer between 0 and 2**32 - 1.",
            context={"random_seed": random_seed},
        ) from None
    if seed < 0 or seed > 2**32 - 1:
        raise DidValueError(
            ErrorCode.E002,
            "random_seed must be an integer between 0 and 2**32 - 1.",
            context={"random_seed": random_seed},
        )
    return int(seed)


def _validate_parallel(parallel: bool, n_cores: int | None) -> tuple[bool, int | None]:
    """Validate parallel and n_cores parameters."""
    if not isinstance(parallel, bool):
        raise TypeError("parallel must be a boolean.")
    if n_cores is not None:
        if not isinstance(n_cores, int) or isinstance(n_cores, bool) or n_cores < 1:
            raise DidValueError(
                ErrorCode.E002,
                "n_cores must be a positive integer.",
                context={"n_cores": n_cores},
            )
    return parallel, n_cores


def _distribute_iterations(n_boot: int, n_workers: int) -> list[int]:
    """Distribute n_boot iterations evenly among n_workers workers."""
    base = n_boot // n_workers
    remainder = n_boot % n_workers
    return [base + (1 if i < remainder else 0) for i in range(n_workers)]


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


def _validate_se_boot(se_boot: bool | None, *, design: str) -> bool:
    if se_boot is None:
        return design == "sa"
    if not _is_bool_like(se_boot):
        raise DidValueError(
            ErrorCode.E002,
            "se_boot must be a boolean.",
            context={"se_boot": se_boot},
        )
    se_boot = bool(se_boot)
    if design == "sa" and not se_boot:
        raise DidValueError(
            ErrorCode.E002,
            "se_boot=False is not supported for design='sa'; SA inference uses bootstrap.",
            context={"se_boot": se_boot, "design": design},
        )
    return se_boot


def _validate_thres(thres: int | None, *, design: str) -> int | None:
    if design != "sa":
        if thres is not None:
            raise DidValueError(
                ErrorCode.E002,
                "thres is only supported for design='sa'.",
                context={"design": design, "thres": thres},
            )
        return None

    if thres is None:
        return 2
    if isinstance(thres, bool) or not isinstance(thres, (int, np.integer)) or thres < 1:
        raise DidValueError(
            ErrorCode.E002,
            "thres must be an integer greater than or equal to 1 for design='sa'.",
            context={"thres": thres},
        )
    return int(thres)


def _materialize_input_rows(data: Iterable[Mapping[str, Any]] | pd.DataFrame) -> list[dict[str, Any]]:
    if isinstance(data, pd.DataFrame):
        return data.to_dict(orient="records")
    return [dict(row) for row in data]


def _reject_observed_non_finite_numeric(series: pd.Series, *, label: str) -> None:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    invalid = ~np.isnan(values) & ~np.isfinite(values)
    if invalid.any():
        raise ValueError(f"{label} must contain only finite numeric values.")


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


def _finite_numeric_array(series: pd.Series, *, label: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} must contain only finite numeric values.")
    return values


def _prepare_frame(
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
    cluster_column = id_cluster or contract.cluster_default or "_observation"
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
    if cluster_mode != "observation":
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
            frame[cluster_column],
            field_name="id_cluster" if cluster_mode == "explicit" else "unit_id",
        )

    time_order = {value: index + 1 for index, value in enumerate(contract.time_order)}
    frame["id_time"] = frame[time].map(time_order).astype(float)

    if data_type == "panel":
        if unit_id is None:
            raise DidValueError(
                ErrorCode.E001,
                "unit_id is required for panel data.",
                context={"data_type": "panel"},
            )
        treat_years = frame.loc[frame[treatment] == 1, "id_time"]
        if treat_years.empty:
            raise DidValueError(
                ErrorCode.E003,
                "At least one treated observation is required for did().",
                context={"n_treated": 0, "n_total": len(frame)},
            )
        treat_year = float(treat_years.min())
        frame["Gi"] = frame.groupby(unit_id, sort=False)[treatment].transform("max").astype(float)
        frame["It"] = (frame["id_time"] >= treat_year).astype(float)
    else:
        if post is None:
            raise DidValueError(
                ErrorCode.E001,
                "post is required for repeated cross-section data.",
                context={"data_type": "rcs"},
            )
        treat_years = frame.loc[frame[post] == 1, "id_time"]
        if treat_years.empty:
            raise DidValueError(
                ErrorCode.E003,
                "At least one post-treatment observation is required for did().",
                context={"n_post_treated": 0, "n_total": len(frame)},
            )
        treat_year = float(treat_years.min())
        frame["Gi"] = frame[treatment].astype(float)
        frame["It"] = frame[post].astype(float)

    frame["id_time_std"] = frame["id_time"] - treat_year
    frame["outcome"] = pd.to_numeric(frame[outcome], errors="coerce")

    if cluster_mode == "observation":
        frame["cluster_id"] = np.arange(len(frame))
    else:
        cluster_values = frame[cluster_column]
        if pd.api.types.is_object_dtype(cluster_values) or pd.api.types.is_string_dtype(cluster_values):
            frame["cluster_id"] = pd.factorize(cluster_values, sort=True)[0]
        else:
            frame["cluster_id"] = cluster_values

    metadata = contract.as_metadata()
    metadata.update(
        {
            "cluster_column": cluster_column,
            "cluster_mode": cluster_mode,
            "clustvar": cluster_column,
            "datatype": data_type,
        }
    )
    metadata["n_clusters"] = int(pd.Series(frame["cluster_id"]).nunique(dropna=True))
    if data_type == "panel":
        frame["_delta_unit_id"] = frame[unit_id]
    return _with_outcome_delta(frame, data_type=data_type), metadata


def _prepare_sa_frame(
    rows: list[dict[str, Any]],
    *,
    outcome: str,
    treatment: str,
    time: str,
    unit_id: str,
    covariates: tuple[_AnyCovarSpec, ...],
    id_cluster: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    contract = normalize_design_data(
        rows,
        outcome=outcome,
        treatment=treatment,
        time=time,
        unit_id=unit_id,
        design="sa",
        data_type="panel",
    )
    cluster_column = id_cluster or contract.cluster_default or unit_id
    _require_model_columns(rows, covariates=covariates, id_cluster=id_cluster)
    cluster_mode = "explicit" if id_cluster is not None else "unit"

    selected_columns = [outcome, treatment, time, unit_id]
    for spec in covariates:
        for src in _covar_spec_sources(spec):
            selected_columns.append(src)
    if cluster_mode == "explicit":
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
    _reject_missing_cluster_values(
        frame[cluster_column],
        field_name="id_cluster" if cluster_mode == "explicit" else "unit_id",
    )

    time_order = {value: index + 1 for index, value in enumerate(contract.time_order)}
    unit_values = frame[unit_id]
    if pd.api.types.is_object_dtype(unit_values) or pd.api.types.is_string_dtype(unit_values):
        frame["id_subject"] = pd.factorize(unit_values, sort=True)[0]
    else:
        frame["id_subject"] = unit_values
    frame["id_time"] = frame[time].map(time_order).astype(int)
    frame["outcome"] = pd.to_numeric(frame[outcome], errors="coerce")
    frame["treatment"] = pd.to_numeric(frame[treatment], errors="coerce").astype(float)
    if cluster_mode == "unit":
        frame["cluster_source"] = frame["id_subject"]
    else:
        cluster_src = frame[cluster_column]
        if pd.api.types.is_object_dtype(cluster_src) or pd.api.types.is_string_dtype(cluster_src):
            frame["cluster_source"] = pd.factorize(cluster_src, sort=True)[0]
        else:
            frame["cluster_source"] = cluster_src
    frame["cluster_id"] = frame["cluster_source"]
    frame = frame.sort_values(["id_subject", "id_time"], kind="stable").reset_index(drop=True)

    metadata = contract.as_metadata()
    metadata.update(
        {
            "cluster_column": cluster_column,
            "cluster_mode": cluster_mode,
            "clustvar": cluster_column,
            "datatype": "panel",
        }
    )
    metadata["n_clusters"] = int(pd.Series(frame["cluster_source"]).nunique(dropna=True))
    return frame, metadata


def _with_group_mean_outcome_delta(frame: pd.DataFrame) -> pd.DataFrame:
    lag_y = (
        frame.groupby(["Gi", "id_time_std"], sort=False)["outcome"]
        .agg(Ymean=lambda values: values.mean(skipna=False))
        .reset_index()
    )
    lag_y["id_time_std"] = lag_y["id_time_std"] + 1
    with_lag = frame.drop(columns=["Ymean", "outcome_delta"], errors="ignore").merge(
        lag_y,
        on=["Gi", "id_time_std"],
        how="left",
    )
    with_lag["outcome_delta"] = with_lag["outcome"] - with_lag["Ymean"]
    return with_lag


def _with_outcome_delta(frame: pd.DataFrame, *, data_type: str) -> pd.DataFrame:
    if data_type in {"panel", "rcs"}:
        return _with_group_mean_outcome_delta(frame)
    raise DidValueError(
        ErrorCode.E002,
        "data_type must be 'panel' or 'rcs'.",
        context={"data_type": data_type},
    )


def _compute_interaction_product(columns: list[np.ndarray]) -> np.ndarray:
    """Compute cross-product of multiple column matrices.

    For example: a matrix with 3 columns crossed with a matrix with 2 columns
    yields a matrix with 6 columns (all pairwise products).
    """
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


def _build_covariate_columns(
    working: pd.DataFrame,
    spec: _AnyCovarSpec,
) -> np.ndarray:
    """Build design matrix columns for a single covariate spec (plain or interaction)."""
    if isinstance(spec, _InteractionCovariateSpec):
        parts: list[np.ndarray] = []
        for term in spec.terms:
            series = working[term.source]
            if term.categorical or not pd.api.types.is_numeric_dtype(series):
                dummies = pd.get_dummies(
                    series.astype("string"),
                    prefix=term.source,
                    drop_first=True,
                    dtype=float,
                )
                if dummies.empty:
                    return np.empty((len(working), 0), dtype=float)
                parts.append(dummies.to_numpy(dtype=float))
            else:
                parts.append(
                    _finite_numeric_array(series, label=f"covariate '{term.source}'").reshape(-1, 1)
                )
        if not parts:
            return np.empty((len(working), 0), dtype=float)
        return _compute_interaction_product(parts)
    # Plain covariate
    series = working[spec.source]
    if spec.categorical or not pd.api.types.is_numeric_dtype(series):
        dummies = pd.get_dummies(
            series.astype("string"),
            prefix=spec.source,
            drop_first=True,
            dtype=float,
        )
        if dummies.empty:
            return np.empty((len(working), 0), dtype=float)
        return dummies.to_numpy(dtype=float)
    return _finite_numeric_array(series, label=f"covariate '{spec.source}'").reshape(-1, 1)


def _build_design_matrix(
    subset: pd.DataFrame,
    *,
    dependent: str,
    covariates: tuple[_AnyCovarSpec, ...],
) -> tuple[np.ndarray, np.ndarray]:
    columns_to_keep = [dependent, "Gi", "It"]
    for spec in covariates:
        columns_to_keep.extend(_covar_spec_sources(spec))
    columns_to_keep = list(dict.fromkeys(columns_to_keep))
    working = subset[columns_to_keep].dropna().reset_index(drop=True)
    if working.empty:
        raise DidValueError(
            ErrorCode.E003,
            f"No rows remain after filtering for {dependent}.",
            context={"dependent": dependent},
        )

    matrix_parts = [
        np.ones((len(working), 1), dtype=float),
        working["Gi"].to_numpy(dtype=float).reshape(-1, 1),
        working["It"].to_numpy(dtype=float).reshape(-1, 1),
        (working["Gi"].astype(float) * working["It"].astype(float)).to_numpy(dtype=float).reshape(-1, 1),
    ]
    for spec in covariates:
        cols = _build_covariate_columns(working, spec)
        if cols.shape[1] > 0:
            matrix_parts.append(cols)

    matrix = np.hstack(matrix_parts)
    outcome = _finite_numeric_array(working[dependent], label=dependent)
    return matrix, outcome


def _solve_interaction_effect(
    subset: pd.DataFrame,
    *,
    dependent: str,
    covariates: tuple[_AnyCovarSpec, ...],
) -> float:
    matrix, outcome = _build_design_matrix(subset, dependent=dependent, covariates=covariates)
    if len(outcome) < matrix.shape[1]:
        raise DidValueError(
            ErrorCode.E004,
            f"Insufficient observations to estimate {dependent}.",
            context={"n_obs": len(outcome), "n_params": matrix.shape[1], "dependent": dependent},
        )
    coefficients, _, rank, _ = np.linalg.lstsq(matrix, outcome, rcond=None)
    if rank < matrix.shape[1]:
        raise DidValueError(
            ErrorCode.E004,
            f"Insufficient variation to estimate {dependent}.",
            context={"rank": rank, "n_params": matrix.shape[1], "dependent": dependent},
        )
    return float(coefficients[3])


# ---------------------------------------------------------------------------
# K-DID generalized transformation (Egami & Yamauchi 2023, Section 3.3.2)
# ---------------------------------------------------------------------------


def _compute_M_coeff(ell: int, s: int) -> float:
    """Compute M^ell_s = C(s+ell-1, ell-1) = prod_{j=1}^{ell-1} (s+j)/j."""
    if ell < 2:
        return 1.0
    result = 1.0
    for j in range(1, ell):
        result = result * (s + j) / j
    return result


def _compute_kdid_pre_coefficients(k: int, s: int) -> tuple[float, ...]:
    """Compute the coefficient vector (alpha_0, ..., alpha_{k-1}) for k-th order transformation.

    Parameters
    ----------
    k : int
        Order of the DID transformation (k=1 is standard DID).
    s : int
        Lead index (distance from last pre-treatment to post-treatment period minus 1).

    Returns
    -------
    tuple of float
        Coefficients (alpha_0, alpha_1, ..., alpha_{k-1}).
    """
    if k < 1:
        raise ValueError("k must be at least 1.")
    if k == 1:
        return (1.0,)

    # alpha_0 = 1 + sum_{j=1}^{k-1} M^{j+1}_s
    alpha_0 = 1.0
    for j in range(1, k):
        alpha_0 += _compute_M_coeff(j + 1, s)

    # alpha_p = sum_{j=p}^{k-1} M^{j+1}_s * (-1)^p * C(j, p)  for p=1..k-1
    alphas = [alpha_0]
    for p in range(1, k):
        val = 0.0
        for j in range(p, k):
            # C(j, p) = j! / (p! * (j-p)!)
            comb = 1.0
            for i in range(1, p + 1):
                comb = comb * (j - i + 1) / i
            val += _compute_M_coeff(j + 1, s) * ((-1) ** p) * comb
        alphas.append(val)

    return tuple(alphas)


def _kdid_transformed_outcome(
    frame: pd.DataFrame,
    *,
    lead: int,
    k: int,
) -> pd.Series:
    """Construct k-th order transformed outcome column.

    For k=1 returns the original outcome (standard DID).
    For k>=2 transforms pre-period outcomes using the K-DID coefficients.

    Parameters
    ----------
    frame : pd.DataFrame
        Must contain columns: outcome, Gi, id_time_std.
    lead : int
        The post-treatment lead being estimated.
    k : int
        Order of the DID transformation.

    Returns
    -------
    pd.Series
        Transformed outcome aligned with frame index.
    """
    if k == 1:
        return frame["outcome"].copy()

    s = lead - 1 if lead >= 1 else 0
    alphas = _compute_kdid_pre_coefficients(k, s)

    transformed = frame["outcome"].copy()

    # Compute group-period means for pre-periods needed: -(2), -(3), ..., -(k)
    # We need periods id_time_std == -(p+1) for p=1..k-1
    needed_periods = [-(p + 1) for p in range(1, k)]

    # Compute group means for each needed period
    group_means: dict[tuple[int, int], float] = {}
    for period in needed_periods:
        period_data = frame.loc[frame["id_time_std"] == period]
        if period_data.empty:
            # Cannot compute transformation - return NaN series
            return pd.Series(np.nan, index=frame.index, name="outcome")
        for gi_val, gi_group in period_data.groupby("Gi"):
            group_means[(int(gi_val), period)] = gi_group["outcome"].mean()

    # Check all needed group means exist
    gi_values = frame["Gi"].unique()
    for gi_val in gi_values:
        for period in needed_periods:
            if (int(gi_val), period) not in group_means:
                return pd.Series(np.nan, index=frame.index, name="outcome")

    # Transform pre-period rows (id_time_std == -1)
    pre_mask = frame["id_time_std"] == -1
    for idx in frame.index[pre_mask]:
        gi_val = int(frame.at[idx, "Gi"])
        y_minus1 = frame.at[idx, "outcome"]
        # alpha_0 * Y_{i,-1} + sum_{p=1}^{k-1}(alpha_p * mean(Y_{g, -(p+1)}))
        new_val = alphas[0] * y_minus1
        for p in range(1, k):
            period = -(p + 1)
            new_val += alphas[p] * group_means[(gi_val, period)]
        transformed.at[idx] = new_val

    return transformed


def _compute_k_component_estimates(
    frame: pd.DataFrame,
    *,
    lead: int,
    kmax: int,
    covariates: tuple[_AnyCovarSpec, ...],
) -> dict[int, float]:
    """Compute component estimates for k=1..K_init.

    Parameters
    ----------
    frame : pd.DataFrame
        Panel data with columns: outcome, Gi, id_time_std, It, plus covariates.
    lead : int
        Post-treatment lead period.
    kmax : int
        Maximum k to attempt.
    covariates : tuple
        Covariate specifications for regression.

    Returns
    -------
    dict mapping k -> tau_k estimate.
    """
    # Determine K_init based on available pre-treatment periods
    pre_periods = sorted(
        [int(t) for t in frame["id_time_std"].unique() if int(t) < 0]
    )
    # We need periods -1, -2, ..., -k for k-th order transformation
    k_init = min(kmax, len(pre_periods))

    subset = frame.loc[frame["id_time_std"].isin((-1, lead))].copy()
    if subset["id_time_std"].nunique() != 2:
        return {}

    results: dict[int, float] = {}

    for k in range(1, k_init + 1):
        if k == 1:
            # Standard DID regression
            try:
                tau = _solve_interaction_effect(
                    subset, dependent="outcome", covariates=covariates
                )
                results[k] = tau
            except ValueError:
                break
        else:
            # K-DID transformed outcome
            transformed = _kdid_transformed_outcome(frame, lead=lead, k=k)
            if transformed.isna().all():
                break
            # Build subset with transformed outcome
            work = subset.copy()
            work["outcome_kdid"] = transformed.reindex(work.index)
            if work["outcome_kdid"].isna().any():
                break
            try:
                tau = _solve_interaction_effect(
                    work, dependent="outcome_kdid", covariates=covariates
                )
                results[k] = tau
            except ValueError:
                break

    return results

# ---------------------------------------------------------------------------
# K-DID Bootstrap, J-test, and GMM combination (kmax > 2)
# ---------------------------------------------------------------------------


def _chi2_sf(x: float, df: int) -> float:
    """Survival function (1 - CDF) of chi-squared distribution."""
    if x <= 0 or df <= 0:
        return 1.0
    a = df / 2.0
    z = x / 2.0
    if z < a + 1:
        return 1.0 - _gamma_inc_series(a, z)
    else:
        return _gamma_inc_cf(a, z)


def _gamma_inc_series(a: float, x: float, max_iter: int = 200, eps: float = 1e-15) -> float:
    """Regularized lower incomplete gamma P(a, x) via series expansion."""
    if x == 0:
        return 0.0
    ap = a
    s = 1.0 / a
    delta = s
    for _ in range(max_iter):
        ap += 1
        delta *= x / ap
        s += delta
        if abs(delta) < abs(s) * eps:
            break
    return s * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_inc_cf(a: float, x: float, max_iter: int = 200, eps: float = 1e-15) -> float:
    """Upper incomplete gamma Q(a, x) = 1 - P(a, x) via continued fraction."""
    b = x + 1 - a
    c = 1.0 / 1e-30
    d = 1.0 / b
    h = d
    for i in range(1, max_iter + 1):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        if abs(d) < 1e-30:
            d = 1e-30
        c = b + an / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _validate_kmax(kmax: int) -> int:
    """Validate kmax parameter."""
    if not isinstance(kmax, int) or isinstance(kmax, bool) or kmax < 1:
        raise DidValueError(
            ErrorCode.E010,
            "kmax must be an integer >= 1.",
            context={"kmax": kmax},
        )
    if kmax > 8:
        raise DidValueError(
            ErrorCode.E010,
            "kmax must be <= 8 to avoid numerical instability in high-order differencing.",
            context={"kmax": kmax, "max_allowed": 8},
        )
    return kmax


def _identify_requested_k_leads(
    frame: pd.DataFrame,
    *,
    requested_leads: tuple[int, ...],
    kmax: int,
    covariates: tuple[_AnyCovarSpec, ...],
) -> tuple[dict[int, dict[int, float]], tuple[int, ...], tuple[int, ...]]:
    """Compute K-dimensional component estimates for each requested lead.

    Returns
    -------
    tuple of (estimates_by_lead, identified_leads, unidentified_leads)
        estimates_by_lead[lead] = {1: tau_1, 2: tau_2, ..., K: tau_K}
    """
    estimates_by_lead: dict[int, dict[int, float]] = {}
    identified_leads: list[int] = []
    unidentified_leads: list[int] = []
    last_failure: ValueError | None = None

    for current_lead in requested_leads:
        try:
            k_estimates = _compute_k_component_estimates(
                frame,
                lead=current_lead,
                kmax=kmax,
                covariates=covariates,
            )
            if not k_estimates:
                unidentified_leads.append(current_lead)
                continue
            estimates_by_lead[current_lead] = k_estimates
            identified_leads.append(current_lead)
        except ValueError as exc:
            last_failure = exc
            unidentified_leads.append(current_lead)
            continue

    if not identified_leads:
        reason = f" Last failure: {last_failure.message if hasattr(last_failure, "message") else last_failure}" if last_failure is not None else ""
        raise DidValueError(
            ErrorCode.E004,
            f"No identifiable lead values remain for K-DID.{reason}",
            context={"requested_leads": list(requested_leads), "unidentified": unidentified_leads},
        )
    return estimates_by_lead, tuple(identified_leads), tuple(unidentified_leads)


def _compute_k_bootstrap_draws(
    frame: pd.DataFrame,
    *,
    leads: tuple[int, ...],
    kmax: int,
    covariates: tuple[_AnyCovarSpec, ...],
    n_boot: int,
    random_seed: int | None,
) -> tuple[DidBootstrapDrawK, ...]:
    """K-DID bootstrap: each resample computes K component estimates.

    Block bootstrap by cluster, same structure as _compute_bootstrap_draws
    but calls _compute_k_component_estimates and returns DidBootstrapDrawK.
    """
    clusters = [group.copy() for _, group in frame.groupby("cluster_id", sort=True)]
    if not clusters:
        raise DidValueError(
            ErrorCode.E009,
            "Bootstrap requires at least one cluster.",
            context={"n_clusters": 0},
        )

    rng = np.random.RandomState(random_seed)
    draws: list[DidBootstrapDrawK] = []
    attempts = 0
    successful_iterations = 0
    max_attempts = max(n_boot * 20, 100)

    while successful_iterations < n_boot:
        attempts += 1
        if attempts > max_attempts:
            raise DidRuntimeError(
                ErrorCode.E009,
                "Bootstrap could not produce enough valid draws for K-DID.",
                context={"n_succeeded": successful_iterations, "n_total": n_boot, "max_attempts": max_attempts},
            )
        sampled_ids = rng.randint(0, len(clusters), size=len(clusters))
        sampled_frames = []
        for draw_position, cluster_index in enumerate(sampled_ids, start=1):
            sampled_frame = clusters[cluster_index].copy()
            if "_delta_unit_id" in sampled_frame:
                sampled_frame["_delta_unit_id"] = sampled_frame["_delta_unit_id"].map(
                    lambda unit, dp=draw_position: (dp, unit)
                )
            sampled_frames.append(sampled_frame)
        boot_raw = pd.concat(sampled_frames, ignore_index=True)
        data_type = "panel" if "_delta_unit_id" in boot_raw else "rcs"
        boot_frame = _with_outcome_delta(boot_raw, data_type=data_type)

        # Compute K component estimates for all leads
        valid_iteration = True
        lead_estimates: dict[int, dict[int, float]] = {}
        for current_lead in leads:
            try:
                k_est = _compute_k_component_estimates(
                    boot_frame,
                    lead=current_lead,
                    kmax=kmax,
                    covariates=covariates,
                )
                if not k_est or len(k_est) < 2:
                    valid_iteration = False
                    break
                lead_estimates[current_lead] = k_est
            except ValueError:
                valid_iteration = False
                break

        if not valid_iteration:
            continue

        successful_iterations += 1
        for current_lead in leads:
            components = tuple(
                lead_estimates[current_lead][k]
                for k in sorted(lead_estimates[current_lead].keys())
            )
            draws.append(
                DidBootstrapDrawK(
                    iteration=successful_iterations,
                    lead=current_lead,
                    components=components,
                )
            )
    return tuple(draws)


def _jtest_moment_selection(
    component_estimates: dict[int, float],
    vcov: np.ndarray,
) -> tuple[dict[int, float], np.ndarray, float, int, float, tuple[int, ...]]:
    """J-test overidentification test with nested moment deletion.

    Implements the Stata-consistent nested deletion algorithm: starting from
    the lowest-order moment (k=1, standard parallel trends), iteratively
    removes moments until the J-test no longer rejects at the 5% level.

    Parameters
    ----------
    component_estimates : dict
        Mapping k -> tau_k point estimates.
    vcov : np.ndarray
        K x K covariance matrix of component estimates.

    Returns
    -------
    tuple of (retained_estimates, final_vcov, J_stat, J_df, J_pval, dropped_moments)
        dropped_moments is a tuple of int indicating which k values were removed.
    """
    all_keys = sorted(component_estimates.keys())
    active_keys = list(all_keys)
    dropped: list[int] = []

    # Initial J-test on full set
    J_stat, J_df, J_pval = _compute_jstat(active_keys, component_estimates, vcov, all_keys)

    if J_pval >= 0.05 or len(active_keys) <= 1:
        # J-test does not reject: keep all moments
        active_idx = [all_keys.index(k) for k in active_keys]
        sub_vcov = vcov[np.ix_(active_idx, active_idx)]
        return dict(component_estimates), sub_vcov, J_stat, J_df, J_pval, tuple(dropped)

    # Nested deletion: remove from lowest-order (k=1 first, then k=2, ...)
    # This mirrors Stata's jtest_select() which preserves higher-order moments.
    for _step in range(len(all_keys) - 1):
        # Remove the lowest remaining key
        key_to_remove = active_keys[0]
        dropped.append(key_to_remove)
        active_keys = active_keys[1:]

        if len(active_keys) <= 1:
            # Only 1 moment remaining, cannot compute J further
            break

        J_stat, J_df, J_pval = _compute_jstat(
            active_keys, component_estimates, vcov, all_keys
        )

        if J_pval >= 0.05:
            # J-test no longer rejects
            break

    # Build final results
    active_idx = [all_keys.index(k) for k in active_keys]
    final_vcov = vcov[np.ix_(active_idx, active_idx)]
    active_estimates = {k: component_estimates[k] for k in active_keys}

    return active_estimates, final_vcov, J_stat, J_df, J_pval, tuple(dropped)


def _compute_jstat(
    active_keys: list[int],
    component_estimates: dict[int, float],
    vcov: np.ndarray,
    all_keys: list[int],
) -> tuple[float, int, float]:
    """Compute J-statistic for a subset of moment conditions.

    Returns (J_stat, J_df, J_pval). Returns (0.0, 0, 1.0) on numerical failure.
    """
    K = len(active_keys)
    if K <= 1:
        return 0.0, 0, 1.0

    theta = np.array([component_estimates[k] for k in active_keys], dtype=float)
    active_idx = [all_keys.index(k) for k in active_keys]
    sub_vcov = vcov[np.ix_(active_idx, active_idx)]

    # Numerical stability checks
    diag = np.diag(sub_vcov)
    if np.any(diag < 0):
        return 0.0, 0, 1.0
    cond = np.linalg.cond(sub_vcov)
    if cond > 1e12 or not np.isfinite(cond):
        return 0.0, 0, 1.0

    try:
        W = np.linalg.inv(sub_vcov)
    except np.linalg.LinAlgError:
        return 0.0, 0, 1.0

    if not np.isfinite(W).all():
        return 0.0, 0, 1.0

    # GMM estimate: tau_hat = (1'W1)^{-1} 1'W theta
    ones = np.ones(K, dtype=float)
    denom = float(ones @ W @ ones)
    if abs(denom) < 1e-12 or not np.isfinite(denom):
        return 0.0, 0, 1.0
    tau_hat = float(ones @ W @ theta) / denom

    # J statistic: (theta - tau_hat*1)' W (theta - tau_hat*1)
    residual = theta - tau_hat * ones
    J_stat = float(residual @ W @ residual)
    J_df = K - 1

    if J_stat < 0 or not np.isfinite(J_stat):
        return 0.0, 0, 1.0

    J_pval = _chi2_sf(J_stat, J_df)
    return J_stat, J_df, J_pval


def _compute_kdid_row(
    *,
    lead: int,
    component_estimates: dict[int, float],
    draws: tuple[DidBootstrapDrawK, ...],
    se_boot: bool,
    jtest: bool = False,
    level: int = 95,
) -> tuple[DidEstimateRow | None, dict]:
    """K-dimensional GMM optimal combination.

    Returns
    -------
    tuple of (DidEstimateRow or None, weights_dict)
    """
    if not draws or not component_estimates:
        return None, {
            "w_k": (), "W": None, "vcov_gmm": None,
            "k_init": len(component_estimates), "k_final": 0,
            "jtest_stat": None, "jtest_df": None, "jtest_pval": None,
            "dropped_moments": (),
        }

    # Extract bootstrap matrix (n_boot x K)
    k_keys = sorted(component_estimates.keys())
    K_init = len(k_keys)
    draw_matrix = np.array(
        [list(d.components[:K_init]) for d in draws],
        dtype=float,
    )

    if draw_matrix.shape[0] < 2:
        return None, {
            "w_k": (), "W": None, "vcov_gmm": None,
            "k_init": K_init, "k_final": 0,
            "jtest_stat": None, "jtest_df": None, "jtest_pval": None,
            "dropped_moments": (),
        }

    # Compute K x K covariance matrix
    vcov = np.cov(draw_matrix.T, ddof=1)
    if vcov.ndim == 0:
        vcov = vcov.reshape(1, 1)

    # J-test moment selection if requested
    jtest_stat = None
    jtest_df = None
    jtest_pval = None
    dropped_moments: tuple[int, ...] = ()
    active_estimates = dict(component_estimates)
    active_vcov = vcov

    if jtest and K_init > 1:
        active_estimates, active_vcov, jtest_stat, jtest_df, jtest_pval, dropped_moments = (
            _jtest_moment_selection(component_estimates, vcov)
        )

    # Determine active keys after potential moment selection
    active_keys = sorted(active_estimates.keys())
    K_final = len(active_keys)

    # Rebuild vcov for active components
    if K_final < K_init:
        active_idx = [k_keys.index(k) for k in active_keys]
        active_vcov = vcov[np.ix_(active_idx, active_idx)]

    # Singularity degradation: reduce K until vcov is invertible
    while K_final > 1:
        try:
            if not np.isfinite(active_vcov).all():
                raise np.linalg.LinAlgError("non-finite")
            W = np.linalg.inv(active_vcov)
            if not np.isfinite(W).all():
                raise np.linalg.LinAlgError("non-finite inverse")
            total = float(W.sum())
            if total <= 0 or not np.isfinite(total):
                raise np.linalg.LinAlgError("non-positive total")
            break
        except np.linalg.LinAlgError:
            # Remove last key and try again
            active_keys = active_keys[:-1]
            active_estimates = {k: active_estimates[k] for k in active_keys}
            K_final = len(active_keys)
            if K_final >= 1:
                active_idx = [k_keys.index(k) for k in active_keys]
                active_vcov = vcov[np.ix_(active_idx, active_idx)]
    else:
        # K_final == 1: pure DID (single component)
        if K_final == 1:
            single_key = active_keys[0]
            estimate = active_estimates[single_key]
            comp_idx = k_keys.index(single_key)
            comp_draws = draw_matrix[:, comp_idx]
            std_error = float(np.std(comp_draws, ddof=1))
            alpha = 1 - level / 100
            if se_boot:
                ci_lo = float(np.quantile(comp_draws, alpha / 2))
                ci_hi = float(np.quantile(comp_draws, 1 - alpha / 2))
            else:
                ci_lo, ci_hi = _normal_ci_bounds(estimate, std_error, level)
            row = DidEstimateRow(
                estimator="K-DID",
                lead=lead,
                estimate=estimate,
                std_error=std_error,
                ci_lo=ci_lo,
                ci_hi=ci_hi,
                weight=None,
            )
            return row, {
                "w_k": (1.0,),
                "W": None,
                "vcov_gmm": tuple(
                    tuple(float(v) for v in r) for r in active_vcov.tolist()
                ) if active_vcov.size > 0 else None,
                "k_init": K_init,
                "k_final": 1,
                "jtest_stat": jtest_stat,
                "jtest_df": jtest_df,
                "jtest_pval": jtest_pval,
                "dropped_moments": dropped_moments,
            }

    # Compute GMM weights: w_k = sum(W[k,:]) / sum(W_all)
    W = np.linalg.inv(active_vcov)
    total = float(W.sum())
    weight_vector = W.sum(axis=1) / total
    weights_tuple = tuple(float(w) for w in weight_vector)

    # Final estimate
    theta = np.array([active_estimates[k] for k in active_keys], dtype=float)
    estimate = float(theta @ weight_vector)

    # Variance = 1 / sum(W_all)
    gmm_variance = 1.0 / total

    # Standard error and CI
    alpha = 1 - level / 100
    if se_boot:
        active_idx = [k_keys.index(k) for k in active_keys]
        active_draw_matrix = draw_matrix[:, active_idx]
        combined_draws = active_draw_matrix @ weight_vector
        std_error = float(np.std(combined_draws, ddof=1))
        ci_lo = float(np.quantile(combined_draws, alpha / 2))
        ci_hi = float(np.quantile(combined_draws, 1 - alpha / 2))
    else:
        std_error = math.sqrt(gmm_variance)
        ci_lo, ci_hi = _normal_ci_bounds(estimate, std_error, level)

    row = DidEstimateRow(
        estimator="K-DID",
        lead=lead,
        estimate=estimate,
        std_error=std_error,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        weight=None,
    )

    return row, {
        "w_k": weights_tuple,
        "W": tuple(tuple(float(v) for v in r) for r in W.tolist()),
        "vcov_gmm": tuple(tuple(float(v) for v in r) for r in active_vcov.tolist()),
        "k_init": K_init,
        "k_final": K_final,
        "jtest_stat": jtest_stat,
        "jtest_df": jtest_df,
        "jtest_pval": jtest_pval,
        "dropped_moments": dropped_moments,
    }




def _compute_component_estimates(
    frame: pd.DataFrame,
    *,
    lead: int,
    covariates: tuple[_AnyCovarSpec, ...],
) -> dict[str, float]:
    subset = frame.loc[frame["id_time_std"].isin((-1, lead))].copy()
    if subset["id_time_std"].nunique() != 2:
        raise DidValueError(
            ErrorCode.E004,
            "The requested lead requires identifiable last-pre and post-treatment periods.",
            context={"lead": lead, "n_unique_periods": int(subset["id_time_std"].nunique())},
        )
    if subset.dropna(subset=["outcome_delta"])["id_time_std"].nunique() != 2:
        raise DidValueError(
            ErrorCode.E004,
            "Sequential DID requires an additional pre-treatment period so outcome deltas are available.",
            context={"lead": lead},
        )
    return {
        "DID": _solve_interaction_effect(subset, dependent="outcome", covariates=covariates),
        "sDID": _solve_interaction_effect(subset, dependent="outcome_delta", covariates=covariates),
    }


def _sa_status_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    periods = tuple(sorted(frame["id_time"].unique()))
    status_rows: dict[Any, dict[int, int]] = {}
    for subject, subject_rows in frame.groupby("id_subject", sort=False):
        ordered = subject_rows.sort_values("id_time", kind="stable")
        max_time = int(ordered["id_time"].max())
        treatment_sum = int(ordered["treatment"].sum())
        treatment_start = max_time - treatment_sum + 1
        status_rows[subject] = {
            period: 0 if treatment_start > period else 1 if treatment_start == period else -1
            for period in periods
        }
    return pd.DataFrame.from_dict(status_rows, orient="index").loc[:, periods]


def _sa_periods_to_use(status_matrix: pd.DataFrame, *, thres: int) -> tuple[int, ...]:
    treated_counts = (status_matrix == 1).sum(axis=0)
    periods = tuple(int(period) for period, count in treated_counts.items() if int(count) >= thres)
    if periods and int(treated_counts.loc[list(periods)].sum()) == len(status_matrix.index):
        periods = periods[:-1]
    return periods


def _sa_subject_ids(status_matrix: pd.DataFrame, periods: tuple[int, ...]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        tuple(status_matrix.index[status_matrix[period] >= 0].tolist())
        for period in periods
    )


def _sa_time_weights(status_matrix: pd.DataFrame, periods: tuple[int, ...]) -> tuple[float, ...]:
    treated_counts = np.asarray([(status_matrix[period] == 1).sum() for period in periods], dtype=float)
    normalized = treated_counts / treated_counts.sum()
    return tuple(float(value) for value in normalized)


def _compute_standard_panel_components(
    subset: pd.DataFrame,
    *,
    lead: int,
    covariates: tuple[_AnyCovarSpec, ...],
) -> dict[str, float]:
    estimates = _compute_standard_panel_component_estimates(
        subset,
        lead=lead,
        covariates=covariates,
        allow_partial=False,
    )
    if "DID" not in estimates or "sDID" not in estimates:
        raise DidValueError(
            ErrorCode.E004,
            "DID and sDID components must both be identifiable.",
            context={"identified_components": list(estimates.keys())},
        )
    return estimates


def _compute_standard_panel_component_estimates(
    subset: pd.DataFrame,
    *,
    lead: int,
    covariates: tuple[_AnyCovarSpec, ...],
    allow_partial: bool,
) -> dict[str, float]:
    treat_years = subset.loc[subset["treatment"] == 1, "id_time"]
    if treat_years.empty:
        raise DidValueError(
            ErrorCode.E003,
            "At least one treated observation is required for design(sa).",
            context={"n_treated": 0, "n_total": len(subset)},
        )

    treat_year = int(treat_years.min())
    working = subset.copy()
    working["Gi"] = working.groupby("id_subject", sort=False)["treatment"].transform("max").astype(float)
    working["It"] = (working["id_time"] >= treat_year).astype(float)
    working["id_time_std"] = working["id_time"] - treat_year
    working["outcome"] = pd.to_numeric(working["outcome"], errors="coerce")
    working["_delta_unit_id"] = working["id_subject"]
    prepared = _with_outcome_delta(working, data_type="panel")
    subset_for_lead = prepared.loc[prepared["id_time_std"].isin((-1, lead))].copy()
    if subset_for_lead["id_time_std"].nunique() != 2:
        raise DidValueError(
            ErrorCode.E004,
            "The requested lead requires identifiable last-pre and post-treatment periods.",
            context={"lead": lead, "n_unique_periods": int(subset_for_lead["id_time_std"].nunique())},
        )

    estimates: dict[str, float] = {}
    first_error: ValueError | None = None
    for estimator, dependent in (("DID", "outcome"), ("sDID", "outcome_delta")):
        if dependent == "outcome_delta" and subset_for_lead.dropna(subset=[dependent])["id_time_std"].nunique() != 2:
            error = ValueError(
                "Sequential DID requires an additional pre-treatment period so outcome deltas are available."
            )
            if not allow_partial:
                raise error
            first_error = first_error or error
            continue
        try:
            estimates[estimator] = _solve_interaction_effect(
                subset_for_lead,
                dependent=dependent,
                covariates=covariates,
            )
        except ValueError as exc:
            if not allow_partial:
                raise
            first_error = first_error or exc

    if not estimates and first_error is not None:
        raise first_error
    return estimates


def _sa_period_component_estimates(
    frame: pd.DataFrame,
    *,
    period: int,
    subject_ids: tuple[Any, ...],
    requested_leads: tuple[int, ...],
    covariates: tuple[_AnyCovarSpec, ...],
) -> dict[int, dict[str, float]]:
    max_time = int(frame["id_time"].max())
    feasible_leads = tuple(current_lead for current_lead in requested_leads if period + current_lead <= max_time)
    if period < 3 or not feasible_leads:
        return {}

    subset = frame.loc[
        frame["id_subject"].isin(subject_ids) & frame["id_time"].isin((period, period - 1, period - 2))
    ].copy()
    if max(feasible_leads) > 0:
        treatment_info = frame.loc[
            frame["id_subject"].isin(subject_ids) & (frame["id_time"] == period),
            ["id_subject", "treatment"],
        ]
        lead_rows = frame.loc[
            frame["id_subject"].isin(subject_ids)
            & (frame["id_time"] >= period + 1)
            & (frame["id_time"] <= period + max(feasible_leads))
        ].drop(columns=["treatment"])
        subset = pd.concat(
            [subset, lead_rows.merge(treatment_info, on="id_subject", how="left")],
            ignore_index=True,
        )

    estimates_by_lead: dict[int, dict[str, float]] = {}
    for current_lead in feasible_leads:
        try:
            estimates_by_lead[current_lead] = _compute_standard_panel_component_estimates(
                subset,
                lead=current_lead,
                covariates=covariates,
                allow_partial=True,
            )
        except ValueError:
            continue
    return estimates_by_lead


# ---------------------------------------------------------------------------
# SA + K-DID generalized support (kmax > 2)
# ---------------------------------------------------------------------------


def _sa_period_data_window_k(
    frame: pd.DataFrame,
    *,
    period_t: int,
    lead_s: int,
    kmax: int,
    unit_id_col: str = "id_subject",
    time_col: str = "id_time",
) -> pd.DataFrame:
    """Construct data window for K-DID in SA design.

    Selects rows with time points {t-kmax, t-kmax+1, ..., t-1, t+s} and
    retains only units observed at all required time points.

    Parameters
    ----------
    frame : pd.DataFrame
        Full panel data.
    period_t : int
        Treatment adoption period.
    lead_s : int
        Lead value (post-treatment distance).
    kmax : int
        Maximum differencing order.
    unit_id_col : str
        Column identifying units.
    time_col : str
        Column identifying time periods.

    Returns
    -------
    pd.DataFrame
        Subset containing valid units across the full time window.
    """
    # Required time points: {t-kmax, ..., t-1, t+s}
    pre_times = [period_t - p for p in range(1, kmax + 1)]
    post_time = period_t + lead_s
    required_times = set(pre_times + [post_time])

    # Filter to required time points
    window = frame.loc[frame[time_col].isin(required_times)].copy()
    if window.empty:
        return window

    # Keep only units that appear at ALL required time points
    n_required = len(required_times)
    unit_time_counts = window.groupby(unit_id_col)[time_col].nunique()
    valid_units = unit_time_counts[unit_time_counts == n_required].index
    return window.loc[window[unit_id_col].isin(valid_units)].copy()


def _compute_kth_diff_component(
    frame: pd.DataFrame,
    *,
    k: int,
    period_t: int,
    lead_s: int,
    outcome_col: str = "outcome",
    treatment_col: str = "treatment",
    time_col: str = "id_time",
    unit_id_col: str = "id_subject",
    covariates: tuple[_AnyCovarSpec, ...] = (),
) -> float | None:
    """Compute k-th order differenced DID component for SA period.

    Uses the K-DID transformed outcome approach: applies k-th order
    transformation coefficients to pre-treatment outcomes, then estimates
    the interaction effect (Gi * It) via OLS on {t-1, t+s} subset.

    Parameters
    ----------
    frame : pd.DataFrame
        Data window containing time points {t-k, ..., t-1, t+s}.
    k : int
        Differencing order (1=standard DID, 2=sDID, >=3 higher-order).
    period_t : int
        Treatment adoption period.
    lead_s : int
        Lead value.
    outcome_col : str
        Outcome variable column.
    treatment_col : str
        Treatment indicator column.
    time_col : str
        Time period column.
    unit_id_col : str
        Unit identifier column.
    covariates : tuple
        Covariate specifications for regression adjustment.

    Returns
    -------
    float or None
        The k-th order component estimate, or None if not identifiable.
    """
    if k < 1:
        return None

    # Prepare working frame with standard columns
    working = frame.copy()
    working["Gi"] = working.groupby(unit_id_col, sort=False)[treatment_col].transform("max").astype(float)
    working["id_time_std"] = working[time_col] - period_t
    working["It"] = (working[time_col] >= period_t).astype(float)
    working["outcome"] = pd.to_numeric(working[outcome_col], errors="coerce")

    # For k=1: standard DID on {t-1, t+s}
    if k == 1:
        subset = working.loc[working["id_time_std"].isin((-1, lead_s))].copy()
        if subset["id_time_std"].nunique() != 2:
            return None
        try:
            return _solve_interaction_effect(
                subset, dependent="outcome", covariates=covariates
            )
        except ValueError:
            return None

    # For k >= 2: use K-DID transformed outcome
    # Need pre-periods -1, -2, ..., -k in id_time_std
    needed_pre = [-p for p in range(1, k + 1)]
    available_pre = sorted(int(t) for t in working["id_time_std"].unique() if int(t) < 0)
    for p in needed_pre:
        if p not in available_pre:
            return None

    # Compute transformed outcome using _kdid_transformed_outcome
    transformed = _kdid_transformed_outcome(working, lead=lead_s, k=k)
    if transformed.isna().all():
        return None

    # Build subset for regression on {t-1, t+s}
    subset = working.loc[working["id_time_std"].isin((-1, lead_s))].copy()
    if subset["id_time_std"].nunique() != 2:
        return None

    subset["outcome_kdid"] = transformed.reindex(subset.index)
    if subset["outcome_kdid"].isna().any():
        return None

    try:
        return _solve_interaction_effect(
            subset, dependent="outcome_kdid", covariates=covariates
        )
    except ValueError:
        return None


def _validate_kmax_sa_feasibility(
    periods_to_use: tuple[int, ...],
    kmax: int,
    all_periods: tuple[int, ...],
) -> tuple[tuple[int, ...], int]:
    """Validate SA feasibility for kmax and return feasible periods.

    For each valid period t, checks whether kmax pre-treatment periods
    {t-kmax, ..., t-1} all exist in the data. If no period satisfies
    the full kmax, automatically downgrades to the maximum feasible order.

    Parameters
    ----------
    periods_to_use : tuple of int
        Candidate treatment adoption periods from SA threshold selection.
    kmax : int
        Requested maximum differencing order.
    all_periods : tuple of int
        All time periods present in the data.

    Returns
    -------
    tuple of (feasible_periods, effective_kmax)
        feasible_periods: periods satisfying the effective kmax requirement.
        effective_kmax: possibly downgraded kmax.
    """
    all_periods_set = set(all_periods)

    def _period_feasible(t: int, k: int) -> bool:
        """Check if period t has k pre-treatment periods available."""
        return all((t - p) in all_periods_set for p in range(1, k + 1))

    # Try full kmax first
    feasible = tuple(t for t in periods_to_use if _period_feasible(t, kmax))
    if feasible:
        return feasible, kmax

    # Downgrade: find maximum feasible kmax'
    for effective_k in range(kmax - 1, 0, -1):
        feasible = tuple(t for t in periods_to_use if _period_feasible(t, effective_k))
        if feasible:
            did_warn(
                WarningCode.W007,
                f"kmax={kmax} is infeasible for SA design (insufficient pre-treatment periods). "
                f"Automatically downgraded to kmax={effective_k}.",
                context={"kmax_requested": kmax, "kmax_effective": effective_k},
                stacklevel=3,
            )
            return feasible, effective_k

    # No period feasible even for k=1
    did_warn(
        WarningCode.W007,
        f"kmax={kmax} is infeasible for SA design. No valid periods found even for kmax=1.",
        context={"kmax_requested": kmax, "n_valid_periods": 0},
        stacklevel=3,
    )
    return (), 0


def _sa_period_component_estimates_k(
    frame: pd.DataFrame,
    *,
    period: int,
    subject_ids: tuple[Any, ...],
    requested_leads: tuple[int, ...],
    kmax: int,
    covariates: tuple[_AnyCovarSpec, ...],
) -> dict[int, dict[int, float]]:
    """Compute K-dimensional component estimates for a single SA period.

    For each feasible lead s at period t, computes components k=1..kmax
    using the K-DID transformed outcome approach.

    Parameters
    ----------
    frame : pd.DataFrame
        Full panel data with SA columns.
    period : int
        Treatment adoption period t.
    subject_ids : tuple
        Valid subject IDs for this period.
    requested_leads : tuple of int
        Lead values to estimate.
    kmax : int
        Maximum differencing order.
    covariates : tuple
        Covariate specifications.

    Returns
    -------
    dict mapping lead -> {k: tau_k}
        For each lead, a dict of component estimates indexed by k=1..kmax.
        Only leads where ALL k=1..kmax components are identified are returned.
    """
    max_time = int(frame["id_time"].max())
    feasible_leads = tuple(
        s for s in requested_leads if period + s <= max_time
    )
    # Need at least kmax+1 periods minimum (kmax pre + 1 post)
    if period < kmax + 1 or not feasible_leads:
        return {}

    # Subset to relevant subjects
    subject_frame = frame.loc[frame["id_subject"].isin(subject_ids)].copy()
    if subject_frame.empty:
        return {}

    estimates_by_lead: dict[int, dict[int, float]] = {}

    for current_lead in feasible_leads:
        # Get data window for this period/lead/kmax
        window = _sa_period_data_window_k(
            subject_frame,
            period_t=period,
            lead_s=current_lead,
            kmax=kmax,
            unit_id_col="id_subject",
            time_col="id_time",
        )
        if window.empty:
            continue

        # Compute all K components
        components: dict[int, float] = {}
        all_identified = True
        for k in range(1, kmax + 1):
            tau_k = _compute_kth_diff_component(
                window,
                k=k,
                period_t=period,
                lead_s=current_lead,
                outcome_col="outcome",
                treatment_col="treatment",
                time_col="id_time",
                unit_id_col="id_subject",
                covariates=covariates,
            )
            if tau_k is None:
                all_identified = False
                break
            components[k] = tau_k

        # Common-target constraint: only keep if ALL components identified
        if all_identified and len(components) == kmax:
            estimates_by_lead[current_lead] = components

    return estimates_by_lead


def _identify_requested_sa_leads_k(
    frame: pd.DataFrame,
    *,
    requested_leads: tuple[int, ...],
    covariates: tuple[_AnyCovarSpec, ...],
    thres: int,
    kmax: int,
) -> tuple[dict[int, dict[int, float]], tuple[int, ...], tuple[int, ...], int]:
    """SA main loop with K-DID support (kmax > 2).

    Aggregates K-dimensional component estimates across adoption periods
    using time weights (proportional to treated counts).

    Parameters
    ----------
    frame : pd.DataFrame
        SA-prepared panel data.
    requested_leads : tuple of int
        Lead values to estimate.
    covariates : tuple
        Covariate specifications.
    thres : int
        Minimum treated units threshold.
    kmax : int
        Maximum differencing order.

    Returns
    -------
    tuple of (estimates_by_lead, identified_leads, unidentified_leads, effective_kmax)
        estimates_by_lead[lead] = {k: aggregated_tau_k}
    """
    status_matrix = _sa_status_matrix(frame)
    periods = _sa_periods_to_use(status_matrix, thres=thres)
    if not periods:
        raise DidValueError(
            ErrorCode.E007,
            "No valid periods found for the SA design after applying thres(). "
            "Try reducing the threshold value.",
            context={"thres": thres},
        )

    # Validate kmax feasibility
    all_periods = tuple(sorted(int(t) for t in frame["id_time"].unique()))
    feasible_periods, effective_kmax = _validate_kmax_sa_feasibility(
        periods, kmax, all_periods
    )
    if effective_kmax == 0 or not feasible_periods:
        raise DidValueError(
            ErrorCode.E007,
            f"kmax={kmax} is not feasible for any SA period. "
            "Insufficient pre-treatment history.",
            context={"kmax": kmax, "effective_kmax": effective_kmax},
        )

    # Recompute subject IDs and time weights for feasible periods only
    subject_ids = _sa_subject_ids(status_matrix, feasible_periods)
    time_weights = _sa_time_weights(status_matrix, feasible_periods)

    # Accumulate per-period K-component estimates
    component_paths: dict[int, dict[int, list[float]]] = {
        current_lead: {k: [] for k in range(1, effective_kmax + 1)}
        for current_lead in requested_leads
    }
    weight_paths: dict[int, list[float]] = {
        current_lead: [] for current_lead in requested_leads
    }

    for period, period_subject_ids, time_weight in zip(
        feasible_periods, subject_ids, time_weights, strict=True
    ):
        try:
            estimates_by_lead = _sa_period_component_estimates_k(
                frame,
                period=period,
                subject_ids=period_subject_ids,
                requested_leads=requested_leads,
                kmax=effective_kmax,
                covariates=covariates,
            )
        except ValueError:
            continue

        for current_lead, components in estimates_by_lead.items():
            # Common-target: only append if ALL K components present
            if len(components) == effective_kmax:
                for k, tau_k in components.items():
                    component_paths[current_lead][k].append(tau_k)
                weight_paths[current_lead].append(time_weight)

    # Aggregate with renormalized time weights
    aggregated: dict[int, dict[int, float]] = {}
    identified_leads: list[int] = []
    unidentified_leads: list[int] = []

    for current_lead in requested_leads:
        weights_arr = np.asarray(weight_paths[current_lead], dtype=float)
        if weights_arr.size == 0:
            unidentified_leads.append(current_lead)
            continue

        # Renormalize weights to sum to 1 on common-support periods
        normalized_weights = weights_arr / weights_arr.sum()

        lead_estimates: dict[int, float] = {}
        for k in range(1, effective_kmax + 1):
            values = np.asarray(component_paths[current_lead][k], dtype=float)
            if values.size != normalized_weights.size:
                break
            lead_estimates[k] = float(values @ normalized_weights)

        if len(lead_estimates) == effective_kmax:
            aggregated[current_lead] = lead_estimates
            identified_leads.append(current_lead)
        else:
            unidentified_leads.append(current_lead)

    if not identified_leads:
        raise DidValueError(
            ErrorCode.E004,
            "No identifiable lead values remain for SA K-DID estimation.",
            context={"requested_leads": list(requested_leads), "unidentified": unidentified_leads},
        )
    return aggregated, tuple(identified_leads), tuple(unidentified_leads), effective_kmax


def _compute_sa_bootstrap_draws_k(
    frame: pd.DataFrame,
    *,
    leads: tuple[int, ...],
    covariates: tuple[_AnyCovarSpec, ...],
    n_boot: int,
    random_seed: int | None,
    thres: int,
    cluster_mode: str,
    kmax: int,
) -> tuple[DidBootstrapDrawK, ...]:
    """SA bootstrap producing K-dimensional component draws.

    Each bootstrap resample computes K-dimensional aggregated estimates
    for each lead, using the same SA + K-DID pipeline.

    Parameters
    ----------
    frame : pd.DataFrame
        SA-prepared panel data.
    leads : tuple of int
        Identified leads.
    covariates : tuple
        Covariate specifications.
    n_boot : int
        Number of bootstrap iterations.
    random_seed : int or None
        Random seed.
    thres : int
        SA threshold.
    cluster_mode : str
        Bootstrap cluster mode.
    kmax : int
        Maximum differencing order.

    Returns
    -------
    tuple of DidBootstrapDrawK
    """
    rng = np.random.RandomState(random_seed)
    draws: list[DidBootstrapDrawK] = []
    attempts = 0
    successful_iterations = 0
    max_attempts = max(n_boot * 20, 100)

    while successful_iterations < n_boot:
        attempts += 1
        if attempts > max_attempts:
            raise DidRuntimeError(
                ErrorCode.E009,
                "Bootstrap could not produce enough valid draws for SA K-DID.",
                context={"n_succeeded": successful_iterations, "n_total": n_boot, "max_attempts": max_attempts},
            )
        boot_frame = _bootstrap_sa_frame(frame, rng=rng, cluster_mode=cluster_mode)
        try:
            estimates_by_lead, identified_leads, _, _ = _identify_requested_sa_leads_k(
                boot_frame,
                requested_leads=leads,
                covariates=covariates,
                thres=thres,
                kmax=kmax,
            )
        except ValueError:
            continue
        if identified_leads != leads:
            continue

        successful_iterations += 1
        for current_lead in leads:
            components = tuple(
                estimates_by_lead[current_lead][k]
                for k in sorted(estimates_by_lead[current_lead].keys())
            )
            draws.append(
                DidBootstrapDrawK(
                    iteration=successful_iterations,
                    lead=current_lead,
                    components=components,
                )
            )
    return tuple(draws)


def _compute_sa_bootstrap_draws_k_parallel(
    frame: pd.DataFrame,
    *,
    leads: tuple[int, ...],
    covariates: tuple[_AnyCovarSpec, ...],
    n_boot: int,
    random_seed: int | None,
    thres: int,
    cluster_mode: str,
    kmax: int,
    n_cores: int | None,
) -> tuple[DidBootstrapDrawK, ...]:
    """Parallel version of _compute_sa_bootstrap_draws_k."""
    effective_cores = min(n_cores or max(os.cpu_count() - 1, 1), n_boot)
    if effective_cores < 2:
        return _compute_sa_bootstrap_draws_k(
            frame, leads=leads, covariates=covariates,
            n_boot=n_boot, random_seed=random_seed,
            thres=thres, cluster_mode=cluster_mode, kmax=kmax,
        )

    chunk_sizes = _distribute_iterations(n_boot, effective_cores)

    if random_seed is not None:
        base_seq = np.random.SeedSequence(random_seed)
        child_seqs = base_seq.spawn(effective_cores)
        worker_seeds: list[int | None] = [
            int(child.generate_state(1)[0]) for child in child_seqs
        ]
    else:
        worker_seeds = [None] * effective_cores

    def worker_task(chunk_size: int, seed: int | None) -> list[DidBootstrapDrawK]:
        draws_chunk = _compute_sa_bootstrap_draws_k(
            frame, leads=leads, covariates=covariates,
            n_boot=chunk_size, random_seed=seed,
            thres=thres, cluster_mode=cluster_mode, kmax=kmax,
        )
        return list(draws_chunk)

    try:
        all_draws: list[DidBootstrapDrawK] = []
        with ThreadPoolExecutor(max_workers=effective_cores) as executor:
            futures = [
                executor.submit(worker_task, chunk_size, seed)
                for chunk_size, seed in zip(chunk_sizes, worker_seeds)
            ]
            for future in futures:
                all_draws.extend(future.result())

        # Renumber iterations
        renumbered: list[DidBootstrapDrawK] = []
        iteration_counter = 0
        for chunk_idx, chunk_size in enumerate(chunk_sizes):
            chunk_start = sum(chunk_sizes[:chunk_idx]) * len(leads)
            chunk_end = chunk_start + chunk_size * len(leads)
            chunk_draws = all_draws[chunk_start:chunk_end]
            for i in range(chunk_size):
                iteration_counter += 1
                for j, current_lead in enumerate(leads):
                    src = chunk_draws[i * len(leads) + j]
                    renumbered.append(DidBootstrapDrawK(
                        iteration=iteration_counter,
                        lead=current_lead,
                        components=src.components,
                    ))
        return tuple(renumbered)
    except Exception as exc:
        did_warn(
            WarningCode.W004,
            f"Parallel SA K-DID bootstrap failed ({exc}); falling back to sequential.",
            context={"error": str(exc)},
            stacklevel=2,
        )
        return _compute_sa_bootstrap_draws_k(
            frame, leads=leads, covariates=covariates,
            n_boot=n_boot, random_seed=random_seed,
            thres=thres, cluster_mode=cluster_mode, kmax=kmax,
        )


def _identify_requested_sa_leads(
    frame: pd.DataFrame,
    *,
    requested_leads: tuple[int, ...],
    covariates: tuple[_AnyCovarSpec, ...],
    thres: int,
) -> tuple[dict[int, dict[str, float]], tuple[int, ...], tuple[int, ...]]:
    status_matrix = _sa_status_matrix(frame)
    periods = _sa_periods_to_use(status_matrix, thres=thres)
    if not periods:
        raise DidValueError(
            ErrorCode.E007,
            "No valid periods found for the SA design after applying thres(). Try reducing the threshold value.",
            context={"thres": thres},
        )

    subject_ids = _sa_subject_ids(status_matrix, periods)
    time_weights = _sa_time_weights(status_matrix, periods)
    component_paths = {
        current_lead: {"DID": [], "sDID": [], "DID_weights": [], "sDID_weights": []}
        for current_lead in requested_leads
    }
    last_failure: ValueError | None = None

    for period, period_subject_ids, time_weight in zip(periods, subject_ids, time_weights, strict=True):
        try:
            estimates_by_lead = _sa_period_component_estimates(
                frame,
                period=period,
                subject_ids=period_subject_ids,
                requested_leads=requested_leads,
                covariates=covariates,
            )
        except ValueError as exc:
            last_failure = exc
            continue
        for current_lead, component_estimates in estimates_by_lead.items():
            if "DID" in component_estimates:
                component_paths[current_lead]["DID"].append(component_estimates["DID"])
                component_paths[current_lead]["DID_weights"].append(time_weight)
            if "sDID" in component_estimates:
                component_paths[current_lead]["sDID"].append(component_estimates["sDID"])
                component_paths[current_lead]["sDID_weights"].append(time_weight)

    estimates_by_lead: dict[int, dict[str, float]] = {}
    identified_leads: list[int] = []
    unidentified_leads: list[int] = []
    for current_lead in requested_leads:
        did_weights = np.asarray(component_paths[current_lead]["DID_weights"], dtype=float)
        sdid_weights = np.asarray(component_paths[current_lead]["sDID_weights"], dtype=float)
        if did_weights.size == 0 or sdid_weights.size == 0:
            unidentified_leads.append(current_lead)
            continue
        normalized_did_weights = did_weights / did_weights.sum()
        normalized_sdid_weights = sdid_weights / sdid_weights.sum()
        did_values = np.asarray(component_paths[current_lead]["DID"], dtype=float)
        sdid_values = np.asarray(component_paths[current_lead]["sDID"], dtype=float)
        estimates_by_lead[current_lead] = {
            "DID": float(did_values @ normalized_did_weights),
            "sDID": float(sdid_values @ normalized_sdid_weights),
        }
        identified_leads.append(current_lead)

    if not identified_leads:
        reason = f" Last failure: {last_failure.message if hasattr(last_failure, "message") else last_failure}" if last_failure is not None else ""
        raise DidValueError(
            ErrorCode.E004,
            f"No identifiable lead values remain for did().{reason}",
            context={"requested_leads": list(requested_leads), "unidentified": unidentified_leads},
        )
    return estimates_by_lead, tuple(identified_leads), tuple(unidentified_leads)


def _identify_requested_leads(
    frame: pd.DataFrame,
    *,
    requested_leads: tuple[int, ...],
    covariates: tuple[_AnyCovarSpec, ...],
) -> tuple[dict[int, dict[str, float]], tuple[int, ...], tuple[int, ...]]:
    estimates_by_lead: dict[int, dict[str, float]] = {}
    identified_leads: list[int] = []
    unidentified_leads: list[int] = []
    last_failure: ValueError | None = None

    for current_lead in requested_leads:
        try:
            estimates_by_lead[current_lead] = _compute_component_estimates(
                frame,
                lead=current_lead,
                covariates=covariates,
            )
        except ValueError as exc:
            last_failure = exc
            unidentified_leads.append(current_lead)
            continue
        identified_leads.append(current_lead)

    if not identified_leads:
        reason = f" Last failure: {last_failure.message if hasattr(last_failure, "message") else last_failure}" if last_failure is not None else ""
        raise DidValueError(
            ErrorCode.E004,
            f"No identifiable lead values remain for did().{reason}",
            context={"requested_leads": list(requested_leads), "unidentified": unidentified_leads},
        )
    return estimates_by_lead, tuple(identified_leads), tuple(unidentified_leads)


def _compute_bootstrap_draws(
    frame: pd.DataFrame,
    *,
    leads: tuple[int, ...],
    covariates: tuple[_AnyCovarSpec, ...],
    n_boot: int,
    random_seed: int | None,
) -> tuple[DidBootstrapDraw, ...]:
    clusters = [group.copy() for _, group in frame.groupby("cluster_id", sort=True)]
    if not clusters:
        raise DidValueError(
            ErrorCode.E009,
            "Bootstrap requires at least one cluster.",
            context={"n_clusters": 0},
        )

    rng = np.random.RandomState(random_seed)
    draws: list[DidBootstrapDraw] = []
    attempts = 0
    max_attempts = max(n_boot * 20, 100)
    while len(draws) < n_boot * len(leads):
        attempts += 1
        if attempts > max_attempts:
            raise DidRuntimeError(
                ErrorCode.E009,
                "Bootstrap could not produce enough valid draws for DID and sDID.",
                context={"n_succeeded": len(draws) // len(leads), "n_total": n_boot, "max_attempts": max_attempts},
            )
        sampled_ids = rng.randint(0, len(clusters), size=len(clusters))
        sampled_frames = []
        for draw_position, cluster_index in enumerate(sampled_ids, start=1):
            sampled_frame = clusters[cluster_index].copy()
            if "_delta_unit_id" in sampled_frame:
                sampled_frame["_delta_unit_id"] = sampled_frame["_delta_unit_id"].map(
                    lambda unit: (draw_position, unit)
                )
            sampled_frames.append(sampled_frame)
        boot_raw = pd.concat(sampled_frames, ignore_index=True)
        data_type = "panel" if "_delta_unit_id" in boot_raw else "rcs"
        boot_frame = _with_outcome_delta(boot_raw, data_type=data_type)
        iteration = len(draws) // len(leads) + 1
        lead_estimates: dict[int, dict[str, float]] = {}
        valid_iteration = True
        for current_lead in leads:
            try:
                lead_estimates[current_lead] = _compute_component_estimates(
                    boot_frame,
                    lead=current_lead,
                    covariates=covariates,
                )
            except ValueError:
                valid_iteration = False
                break
        if not valid_iteration:
            continue

        for current_lead in leads:
            draws.append(
                DidBootstrapDraw(
                    iteration=iteration,
                    lead=current_lead,
                    did=lead_estimates[current_lead]["DID"],
                    sdid=lead_estimates[current_lead]["sDID"],
                )
            )
    return tuple(draws)


def _compute_bootstrap_draws_parallel(
    frame: pd.DataFrame,
    *,
    leads: tuple[int, ...],
    covariates: tuple[_AnyCovarSpec, ...],
    n_boot: int,
    random_seed: int | None,
    n_cores: int | None,
) -> tuple[DidBootstrapDraw, ...]:
    """Parallel version of _compute_bootstrap_draws using ThreadPoolExecutor."""
    effective_cores = min(n_cores or max(os.cpu_count() - 1, 1), n_boot)
    if effective_cores < 2:
        return _compute_bootstrap_draws(
            frame, leads=leads, covariates=covariates,
            n_boot=n_boot, random_seed=random_seed,
        )

    chunk_sizes = _distribute_iterations(n_boot, effective_cores)

    # Generate deterministic child seeds using SeedSequence
    if random_seed is not None:
        base_seq = np.random.SeedSequence(random_seed)
        child_seqs = base_seq.spawn(effective_cores)
        worker_seeds: list[int | None] = [
            int(child.generate_state(1)[0]) for child in child_seqs
        ]
    else:
        worker_seeds = [None] * effective_cores

    def worker_task(chunk_size: int, seed: int | None) -> list[DidBootstrapDraw]:
        draws_chunk = _compute_bootstrap_draws(
            frame, leads=leads, covariates=covariates,
            n_boot=chunk_size, random_seed=seed,
        )
        return list(draws_chunk)

    try:
        all_draws: list[DidBootstrapDraw] = []
        with ThreadPoolExecutor(max_workers=effective_cores) as executor:
            futures = [
                executor.submit(worker_task, chunk_size, seed)
                for chunk_size, seed in zip(chunk_sizes, worker_seeds)
            ]
            for future in futures:
                all_draws.extend(future.result())

        # Renumber iterations to be consecutive 1..n_boot
        renumbered: list[DidBootstrapDraw] = []
        iteration_counter = 0
        for chunk_idx, chunk_size in enumerate(chunk_sizes):
            chunk_start = sum(chunk_sizes[:chunk_idx]) * len(leads)
            chunk_end = chunk_start + chunk_size * len(leads)
            chunk_draws = all_draws[chunk_start:chunk_end]
            for i in range(chunk_size):
                iteration_counter += 1
                for j, current_lead in enumerate(leads):
                    src = chunk_draws[i * len(leads) + j]
                    renumbered.append(DidBootstrapDraw(
                        iteration=iteration_counter,
                        lead=current_lead,
                        did=src.did,
                        sdid=src.sdid,
                    ))
        return tuple(renumbered)
    except Exception as exc:
        did_warn(
            WarningCode.W004,
            f"Parallel bootstrap failed ({exc}); falling back to sequential.",
            context={"error": str(exc)},
            stacklevel=2,
        )
        return _compute_bootstrap_draws(
            frame, leads=leads, covariates=covariates,
            n_boot=n_boot, random_seed=random_seed,
        )


def _compute_sa_bootstrap_draws_parallel(
    frame: pd.DataFrame,
    *,
    leads: tuple[int, ...],
    covariates: tuple[_AnyCovarSpec, ...],
    n_boot: int,
    random_seed: int | None,
    thres: int,
    cluster_mode: str,
    n_cores: int | None,
) -> tuple[DidBootstrapDraw, ...]:
    """Parallel version of _compute_sa_bootstrap_draws using ThreadPoolExecutor."""
    effective_cores = min(n_cores or max(os.cpu_count() - 1, 1), n_boot)
    if effective_cores < 2:
        return _compute_sa_bootstrap_draws(
            frame, leads=leads, covariates=covariates,
            n_boot=n_boot, random_seed=random_seed,
            thres=thres, cluster_mode=cluster_mode,
        )

    chunk_sizes = _distribute_iterations(n_boot, effective_cores)

    if random_seed is not None:
        base_seq = np.random.SeedSequence(random_seed)
        child_seqs = base_seq.spawn(effective_cores)
        worker_seeds: list[int | None] = [
            int(child.generate_state(1)[0]) for child in child_seqs
        ]
    else:
        worker_seeds = [None] * effective_cores

    def worker_task(chunk_size: int, seed: int | None) -> list[DidBootstrapDraw]:
        draws_chunk = _compute_sa_bootstrap_draws(
            frame, leads=leads, covariates=covariates,
            n_boot=chunk_size, random_seed=seed,
            thres=thres, cluster_mode=cluster_mode,
        )
        return list(draws_chunk)

    try:
        all_draws: list[DidBootstrapDraw] = []
        with ThreadPoolExecutor(max_workers=effective_cores) as executor:
            futures = [
                executor.submit(worker_task, chunk_size, seed)
                for chunk_size, seed in zip(chunk_sizes, worker_seeds)
            ]
            for future in futures:
                all_draws.extend(future.result())

        # Renumber iterations
        renumbered: list[DidBootstrapDraw] = []
        iteration_counter = 0
        for chunk_idx, chunk_size in enumerate(chunk_sizes):
            chunk_start = sum(chunk_sizes[:chunk_idx]) * len(leads)
            chunk_end = chunk_start + chunk_size * len(leads)
            chunk_draws = all_draws[chunk_start:chunk_end]
            for i in range(chunk_size):
                iteration_counter += 1
                for j, current_lead in enumerate(leads):
                    src = chunk_draws[i * len(leads) + j]
                    renumbered.append(DidBootstrapDraw(
                        iteration=iteration_counter,
                        lead=current_lead,
                        did=src.did,
                        sdid=src.sdid,
                    ))
        return tuple(renumbered)
    except Exception as exc:
        did_warn(
            WarningCode.W004,
            f"Parallel SA bootstrap failed ({exc}); falling back to sequential.",
            context={"error": str(exc)},
            stacklevel=2,
        )
        return _compute_sa_bootstrap_draws(
            frame, leads=leads, covariates=covariates,
            n_boot=n_boot, random_seed=random_seed,
            thres=thres, cluster_mode=cluster_mode,
        )


def _compute_k_bootstrap_draws_parallel(
    frame: pd.DataFrame,
    *,
    leads: tuple[int, ...],
    kmax: int,
    covariates: tuple[_AnyCovarSpec, ...],
    n_boot: int,
    random_seed: int | None,
    n_cores: int | None,
) -> tuple[DidBootstrapDrawK, ...]:
    """Parallel version of _compute_k_bootstrap_draws using ThreadPoolExecutor."""
    effective_cores = min(n_cores or max(os.cpu_count() - 1, 1), n_boot)
    if effective_cores < 2:
        return _compute_k_bootstrap_draws(
            frame, leads=leads, kmax=kmax, covariates=covariates,
            n_boot=n_boot, random_seed=random_seed,
        )

    chunk_sizes = _distribute_iterations(n_boot, effective_cores)

    if random_seed is not None:
        base_seq = np.random.SeedSequence(random_seed)
        child_seqs = base_seq.spawn(effective_cores)
        worker_seeds: list[int | None] = [
            int(child.generate_state(1)[0]) for child in child_seqs
        ]
    else:
        worker_seeds = [None] * effective_cores

    def worker_task(chunk_size: int, seed: int | None) -> list[DidBootstrapDrawK]:
        draws_chunk = _compute_k_bootstrap_draws(
            frame, leads=leads, kmax=kmax, covariates=covariates,
            n_boot=chunk_size, random_seed=seed,
        )
        return list(draws_chunk)

    try:
        all_draws: list[DidBootstrapDrawK] = []
        with ThreadPoolExecutor(max_workers=effective_cores) as executor:
            futures = [
                executor.submit(worker_task, chunk_size, seed)
                for chunk_size, seed in zip(chunk_sizes, worker_seeds)
            ]
            for future in futures:
                all_draws.extend(future.result())

        # Renumber iterations
        renumbered: list[DidBootstrapDrawK] = []
        iteration_counter = 0
        for chunk_idx, chunk_size in enumerate(chunk_sizes):
            chunk_start = sum(chunk_sizes[:chunk_idx]) * len(leads)
            chunk_end = chunk_start + chunk_size * len(leads)
            chunk_draws = all_draws[chunk_start:chunk_end]
            for i in range(chunk_size):
                iteration_counter += 1
                for j, current_lead in enumerate(leads):
                    src = chunk_draws[i * len(leads) + j]
                    renumbered.append(DidBootstrapDrawK(
                        iteration=iteration_counter,
                        lead=current_lead,
                        components=src.components,
                    ))
        return tuple(renumbered)
    except Exception as exc:
        did_warn(
            WarningCode.W004,
            f"Parallel K-DID bootstrap failed ({exc}); falling back to sequential.",
            context={"error": str(exc)},
            stacklevel=2,
        )
        return _compute_k_bootstrap_draws(
            frame, leads=leads, kmax=kmax, covariates=covariates,
            n_boot=n_boot, random_seed=random_seed,
        )


def _summarize_bootstrap_draws(
    draws: tuple[DidBootstrapDraw, ...],
    level: int = 95,
) -> dict[tuple[int, str], tuple[float | None, float | None, float | None]]:
    alpha = 1 - level / 100
    summary: dict[tuple[int, str], tuple[float | None, float | None, float | None]] = {}
    for current_lead in sorted({draw.lead for draw in draws}):
        lead_draws = tuple(draw for draw in draws if draw.lead == current_lead)
        for estimator in ("DID", "sDID"):
            values = np.asarray(
                [draw.did if estimator == "DID" else draw.sdid for draw in lead_draws],
                dtype=float,
            )
            if len(values) < 2:
                summary[(current_lead, estimator)] = (None, None, None)
                continue
            summary[(current_lead, estimator)] = (
                float(np.std(values, ddof=1)),
                float(np.quantile(values, alpha / 2)),
                float(np.quantile(values, 1 - alpha / 2)),
            )
    return summary


def _normal_ci_bounds(estimate: float, std_error: float | None, level: int = 95) -> tuple[float | None, float | None]:
    if std_error is None:
        return None, None
    alpha = 1 - level / 100
    z = NormalDist().inv_cdf(1 - alpha / 2)
    return (
        estimate - z * std_error,
        estimate + z * std_error,
    )


def _serialize_square_matrix(matrix: np.ndarray | None) -> tuple[tuple[float, ...], ...] | None:
    if matrix is None:
        return None
    return tuple(tuple(float(value) for value in row) for row in matrix.tolist())


def _gmm_vcov_identifiable(vcov: np.ndarray, draw_matrix: np.ndarray) -> bool:
    draw_scale = float(np.max(np.abs(draw_matrix))) if draw_matrix.size else 0.0
    vcov_tolerance = np.finfo(float).eps * (draw_scale**2 if draw_scale > 0 else 1.0)
    if np.linalg.norm(vcov, ord=np.inf) <= vcov_tolerance:
        return False
    return int(np.linalg.matrix_rank(vcov, tol=vcov_tolerance)) == vcov.shape[0]


def _gmm_weight_matrix_usable(weight_matrix: np.ndarray) -> bool:
    if weight_matrix.shape != (2, 2) or not np.isfinite(weight_matrix).all():
        return False
    if not np.allclose(weight_matrix, weight_matrix.T, rtol=1e-9, atol=1e-9):
        return False
    determinant = float(weight_matrix[0, 0] * weight_matrix[1, 1] - weight_matrix[0, 1] ** 2)
    return bool(weight_matrix[0, 0] > 0 and weight_matrix[1, 1] > 0 and determinant > 0)


def _bootstrap_sa_frame(
    frame: pd.DataFrame,
    *,
    rng: np.random.RandomState,
    cluster_mode: str,
) -> pd.DataFrame:
    if cluster_mode == "unit":
        sampled_subjects = rng.choice(
            tuple(sorted(dict.fromkeys(frame["id_subject"].tolist()))),
            size=frame["id_subject"].nunique(),
            replace=True,
        )
        sampled_frames = []
        for new_subject, sampled_subject in enumerate(sampled_subjects, start=1):
            sampled_frame = frame.loc[frame["id_subject"] == sampled_subject].copy()
            sampled_frame["id_subject"] = new_subject
            sampled_frame["cluster_id"] = new_subject
            sampled_frame["cluster_source"] = new_subject
            sampled_frames.append(sampled_frame)
        return pd.concat(sampled_frames, ignore_index=True).sort_values(
            ["id_subject", "id_time"],
            kind="stable",
        )

    sampled_clusters = rng.choice(
        tuple(sorted(dict.fromkeys(frame["cluster_source"].tolist()))),
        size=frame["cluster_source"].nunique(),
        replace=True,
    )
    sampled_frames = []
    for new_cluster, sampled_cluster in enumerate(sampled_clusters, start=1):
        sampled_frame = frame.loc[frame["cluster_source"] == sampled_cluster].copy()
        sampled_frame["cluster_id"] = new_cluster
        sampled_frame["cluster_source"] = new_cluster
        sampled_frame["id_subject"] = sampled_frame["id_subject"].map(
            lambda subject: f"{new_cluster}_{subject}"
        )
        sampled_frames.append(sampled_frame)

    boot_frame = pd.concat(sampled_frames, ignore_index=True)
    boot_frame["id_subject"] = pd.factorize(boot_frame["id_subject"], sort=False)[0] + 1
    return boot_frame.sort_values(["id_subject", "id_time"], kind="stable")


def _compute_double_did_row(
    *,
    lead: int,
    component_estimates: dict[str, float],
    draws: tuple[DidBootstrapDraw, ...],
    se_boot: bool,
    level: int = 95,
) -> tuple[DidEstimateRow | None, dict[str, float | tuple[tuple[float, ...], ...] | None]]:
    if any(draw.lead != lead for draw in draws):
        raise ValueError("Double-DID bootstrap draws must match the requested lead.")
    draw_matrix = np.asarray([(draw.did, draw.sdid) for draw in draws], dtype=float)
    if draw_matrix.shape[0] < 2:
        return None, {"w_did": None, "w_sdid": None, "W": None, "vcov_gmm": None}

    vcov = np.cov(draw_matrix.T, ddof=1)
    if vcov.shape != (2, 2) or not np.isfinite(vcov).all():
        return None, {"w_did": None, "w_sdid": None, "W": None, "vcov_gmm": None}

    if not _gmm_vcov_identifiable(vcov, draw_matrix):
        return None, {
            "w_did": None,
            "w_sdid": None,
            "W": None,
            "vcov_gmm": _serialize_square_matrix(vcov),
        }

    inverse_vcov = np.linalg.inv(vcov)
    if not _gmm_weight_matrix_usable(inverse_vcov):
        return None, {
            "w_did": None,
            "w_sdid": None,
            "W": None,
            "vcov_gmm": _serialize_square_matrix(vcov),
        }
    weight_vector = inverse_vcov @ np.ones(2, dtype=float)
    denominator = float(weight_vector.sum())
    if not np.isfinite(denominator) or denominator <= 0 or abs(denominator) < 1e-12:
        return None, {
            "w_did": None,
            "w_sdid": None,
            "W": None,
            "vcov_gmm": _serialize_square_matrix(vcov),
        }

    normalized_weights = weight_vector / denominator
    if not np.isfinite(normalized_weights).all():
        return None, {
            "w_did": None,
            "w_sdid": None,
            "W": None,
            "vcov_gmm": _serialize_square_matrix(vcov),
        }
    did_weight = float(normalized_weights[0])
    sdid_weight = float(normalized_weights[1])
    ddid_draws = draw_matrix @ normalized_weights
    estimate = did_weight * component_estimates["DID"] + sdid_weight * component_estimates["sDID"]
    alpha = 1 - level / 100
    if se_boot:
        std_error = float(np.std(ddid_draws, ddof=1))
        ci_lo = float(np.quantile(ddid_draws, alpha / 2))
        ci_hi = float(np.quantile(ddid_draws, 1 - alpha / 2))
    else:
        std_error = math.sqrt(1.0 / denominator)
        ci_lo, ci_hi = _normal_ci_bounds(estimate, std_error, level)

    return (
        DidEstimateRow(
            estimator="Double-DID",
            lead=lead,
            estimate=estimate,
            std_error=std_error,
            ci_lo=ci_lo,
            ci_hi=ci_hi,
            weight=None,
        ),
        {
            "w_did": did_weight,
            "w_sdid": sdid_weight,
            "W": _serialize_square_matrix(inverse_vcov),
            "vcov_gmm": _serialize_square_matrix(vcov),
        },
    )


def _compute_sa_bootstrap_draws(
    frame: pd.DataFrame,
    *,
    leads: tuple[int, ...],
    covariates: tuple[_AnyCovarSpec, ...],
    n_boot: int,
    random_seed: int | None,
    thres: int,
    cluster_mode: str,
) -> tuple[DidBootstrapDraw, ...]:
    rng = np.random.RandomState(random_seed)
    draws: list[DidBootstrapDraw] = []
    attempts = 0
    successful_iterations = 0
    max_attempts = max(n_boot * 20, 100)
    while successful_iterations < n_boot:
        attempts += 1
        if attempts > max_attempts:
            raise DidRuntimeError(
                ErrorCode.E009,
                "Bootstrap could not produce enough valid draws for the SA design.",
                context={"n_succeeded": successful_iterations, "n_total": n_boot, "max_attempts": max_attempts},
            )
        boot_frame = _bootstrap_sa_frame(frame, rng=rng, cluster_mode=cluster_mode)
        try:
            estimates_by_lead, identified_leads, _ = _identify_requested_sa_leads(
                boot_frame,
                requested_leads=leads,
                covariates=covariates,
                thres=thres,
            )
        except ValueError:
            continue
        if identified_leads != leads:
            continue

        successful_iterations += 1
        for current_lead in leads:
            draws.append(
                DidBootstrapDraw(
                    iteration=successful_iterations,
                    lead=current_lead,
                    did=estimates_by_lead[current_lead]["DID"],
                    sdid=estimates_by_lead[current_lead]["sDID"],
                )
            )
    return tuple(draws)


def _compute_sa_double_did_row(
    *,
    lead: int,
    component_estimates: dict[str, float],
    draws: tuple[DidBootstrapDraw, ...],
    se_boot: bool,
    level: int = 95,
) -> tuple[DidEstimateRow | None, dict[str, float | tuple[tuple[float, ...], ...] | None]]:
    if any(draw.lead != lead for draw in draws):
        raise ValueError("SA-Double-DID bootstrap draws must match the requested lead.")
    draw_matrix = np.asarray([(draw.did, draw.sdid) for draw in draws], dtype=float)
    if draw_matrix.shape[0] < 2:
        return None, {"w_did": None, "w_sdid": None, "W": None, "vcov_gmm": None}

    vcov = np.cov(draw_matrix.T, ddof=1)
    if vcov.shape != (2, 2) or not np.isfinite(vcov).all():
        return None, {"w_did": None, "w_sdid": None, "W": None, "vcov_gmm": None}

    if not _gmm_vcov_identifiable(vcov, draw_matrix):
        return None, {
            "w_did": None,
            "w_sdid": None,
            "W": None,
            "vcov_gmm": _serialize_square_matrix(vcov),
        }

    inverse_vcov = np.linalg.inv(vcov)
    if not _gmm_weight_matrix_usable(inverse_vcov):
        return None, {
            "w_did": None,
            "w_sdid": None,
            "W": None,
            "vcov_gmm": _serialize_square_matrix(vcov),
        }
    weight_vector = inverse_vcov @ np.ones(2, dtype=float)
    denominator = float(weight_vector.sum())
    if not np.isfinite(denominator) or denominator <= 0 or abs(denominator) < 1e-12:
        return None, {
            "w_did": None,
            "w_sdid": None,
            "W": None,
            "vcov_gmm": _serialize_square_matrix(vcov),
        }

    normalized_weights = weight_vector / denominator
    if not np.isfinite(normalized_weights).all():
        return None, {
            "w_did": None,
            "w_sdid": None,
            "W": None,
            "vcov_gmm": _serialize_square_matrix(vcov),
        }

    did_weight = float(normalized_weights[0])
    sdid_weight = float(normalized_weights[1])
    ddid_draws = draw_matrix @ normalized_weights
    estimate = did_weight * component_estimates["DID"] + sdid_weight * component_estimates["sDID"]
    alpha = 1 - level / 100
    if se_boot:
        std_error = float(np.std(ddid_draws, ddof=1))
        ci_lo = float(np.quantile(ddid_draws, alpha / 2))
        ci_hi = float(np.quantile(ddid_draws, 1 - alpha / 2))
    else:
        std_error = math.sqrt(1.0 / denominator)
        ci_lo, ci_hi = _normal_ci_bounds(estimate, std_error, level)

    return (
        DidEstimateRow(
            estimator="SA-Double-DID",
            lead=lead,
            estimate=estimate,
            std_error=std_error,
            ci_lo=ci_lo,
            ci_hi=ci_hi,
            weight=None,
        ),
        {
            "w_did": did_weight,
            "w_sdid": sdid_weight,
            "W": _serialize_square_matrix(inverse_vcov),
            "vcov_gmm": _serialize_square_matrix(vcov),
        },
    )


def did(
    data: Iterable[Mapping[str, Any]] | pd.DataFrame,
    *,
    formula: str | DidFormulaSpec | None = None,
    outcome: str | None = None,
    treatment: str | None = None,
    time: str,
    unit_id: str | None = None,
    post: str | None = None,
    design: str = "did",
    data_type: str = "panel",
    covariates: Sequence[str] | None = None,
    lead: int | Sequence[int] = 0,
    thres: int | None = None,
    n_boot: int = 30,
    se_boot: bool | None = None,
    level: int = 95,
    id_cluster: str | None = None,
    random_seed: int | None = None,
    parallel: bool = False,
    n_cores: int | None = None,
    option: Mapping[str, Any] | None = None,
    is_panel: bool | None = None,
    kmax: int = 2,
    jtest: bool = False,
) -> DidResult:
    """Estimate DID, Double-DID, K-DID, or SA variants.

    Parameters
    ----------
    data : DataFrame or iterable of mappings
        Panel or repeated cross-section data.
    formula : str or DidFormulaSpec, optional
        Formula specification (e.g. ``"y ~ treat"``).
    outcome : str, optional
        Outcome column name (alternative to formula).
    treatment : str, optional
        Treatment column name (alternative to formula).
    time : str
        Time column name.
    unit_id : str, optional
        Unit identifier column. Required for design='sa'.
    post : str, optional
        Post-treatment indicator column.
    design : str, default 'did'
        Design type: 'did' or 'sa' (staggered adoption).
    data_type : str, default 'panel'
        Data structure: 'panel' or 'rcs'.
    covariates : sequence of str, optional
        Covariate column names.
    lead : int or sequence of int, default 0
        Post-treatment lead period(s) to estimate.
    thres : int, optional
        Minimum observations threshold.
    n_boot : int, default 30
        Number of bootstrap replications.
    se_boot : bool, optional
        Whether to compute bootstrap standard errors.
    level : int, default 95
        Confidence level for intervals.
    id_cluster : str, optional
        Cluster variable for bootstrap resampling.
    random_seed : int, optional
        Random seed for reproducibility.
    parallel : bool, default False
        Whether to run bootstrap in parallel.
    n_cores : int, optional
        Number of cores for parallel bootstrap.
    option : mapping, optional
        Legacy option mapping for backward compatibility.
    is_panel : bool, optional
        Deprecated alias for data_type='panel'.
    kmax : int, default 2
        Maximum K-DID order. Must satisfy 1 <= kmax <= 8.
        Values greater than 8 raise ValueError to avoid numerical instability
        in high-order differencing.
    jtest : bool, default False
        Whether to run J-test for overidentification (requires kmax >= 3).

    Returns
    -------
    DidResult
        Immutable result object with estimates, bootstrap draws, and metadata.
    """
    data_type, lead, thres, n_boot, se_boot, id_cluster = _normalize_runtime_surface(
        option=option,
        data_type=data_type,
        is_panel=is_panel,
        lead=lead,
        thres=thres,
        n_boot=n_boot,
        se_boot=se_boot,
        id_cluster=id_cluster,
    )
    design = _validate_surface_choice(design, field_name="design", supported={"did", "sa"})
    if design == "sa" and data_type != "panel":
        raise DidValueError(
            ErrorCode.E002,
            "data_type must be 'panel' for design='sa'.",
            context={"data_type": data_type, "design": "sa"},
        )
    requested_leads = _validate_requested_leads(lead)
    validated_n_boot = _validate_n_boot(n_boot)
    validated_random_seed = _validate_random_seed(random_seed)
    validated_parallel, validated_n_cores = _validate_parallel(parallel, n_cores)
    validated_se_boot = _validate_se_boot(se_boot, design=design)
    validated_level = _validate_level(level)
    validated_thres = _validate_thres(thres, design=design)
    outcome, treatment, post, covariates = _resolve_formula_surface(
        formula=formula,
        outcome=outcome,
        treatment=treatment,
        post=post,
        covariates=covariates,
        design=design,
        data_type=data_type,
    )
    covariate_specs = _parse_covariates(covariates)
    covariate_labels = _build_covariate_labels(covariate_specs)
    outcome = _validate_required_column_name(outcome, field_name="outcome")
    treatment = _validate_required_column_name(treatment, field_name="treatment")
    time = _validate_required_column_name(time, field_name="time")
    unit_id = _validate_optional_column_name(unit_id, field_name="unit_id")
    post = _validate_optional_column_name(post, field_name="post")
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
    rows = _materialize_input_rows(data)
    if design == "sa":
        if unit_id is None:
            raise DidValueError(
                ErrorCode.E001,
                "unit_id is required for design(sa).",
                context={"design": "sa"},
            )
        frame, metadata = _prepare_sa_frame(
            rows,
            outcome=outcome,
            treatment=treatment,
            time=time,
            unit_id=unit_id,
            covariates=covariate_specs,
            id_cluster=id_cluster,
        )

        validated_kmax = _validate_kmax(kmax)

        # ===== SA + K-DID path (kmax > 2) =====
        if validated_kmax > 2:
            if data_type != "panel":
                raise DidValueError(
                    ErrorCode.E008,
                    "K-DID with kmax > 2 requires data_type='panel'.",
                    context={"kmax": kmax, "data_type": data_type},
                )

            k_estimates_by_lead, k_identified_leads, k_unidentified_leads, effective_kmax = (
                _identify_requested_sa_leads_k(
                    frame,
                    requested_leads=requested_leads,
                    covariates=covariate_specs,
                    thres=validated_thres,
                    kmax=validated_kmax,
                )
            )

            k_draws = _compute_sa_bootstrap_draws_k_parallel(
                frame,
                leads=k_identified_leads,
                covariates=covariate_specs,
                n_boot=validated_n_boot,
                random_seed=validated_random_seed,
                thres=validated_thres,
                cluster_mode=str(metadata["cluster_mode"]),
                kmax=effective_kmax,
                n_cores=validated_n_cores,
            ) if validated_parallel and validated_n_boot > 1 else _compute_sa_bootstrap_draws_k(
                frame,
                leads=k_identified_leads,
                covariates=covariate_specs,
                n_boot=validated_n_boot,
                random_seed=validated_random_seed,
                thres=validated_thres,
                cluster_mode=str(metadata["cluster_mode"]),
                kmax=effective_kmax,
            )

            k_weights_by_lead: dict[int, dict] = {}
            k_estimate_rows: list[DidEstimateRow] = []

            for current_lead in k_identified_leads:
                lead_draws = tuple(d for d in k_draws if d.lead == current_lead)
                kdid_row, weights = _compute_kdid_row(
                    lead=current_lead,
                    component_estimates=k_estimates_by_lead[current_lead],
                    draws=lead_draws,
                    se_boot=validated_se_boot,
                    jtest=jtest,
                    level=validated_level,
                )
                k_weights_by_lead[current_lead] = weights
                if kdid_row is not None:
                    # Label as SA-K-DID
                    k_estimate_rows.append(DidEstimateRow(
                        estimator="SA-K-DID",
                        lead=current_lead,
                        estimate=kdid_row.estimate,
                        std_error=kdid_row.std_error,
                        ci_lo=kdid_row.ci_lo,
                        ci_hi=kdid_row.ci_hi,
                        weight=kdid_row.weight,
                    ))

                # Add component estimate rows
                k_keys_sorted = sorted(k_estimates_by_lead[current_lead].keys())
                for k_comp in k_keys_sorted:
                    tau_k = k_estimates_by_lead[current_lead][k_comp]
                    estimator_label = {1: "SA-DID", 2: "SA-sDID"}.get(k_comp, f"SA-kDID-{k_comp}")
                    comp_draws_list = [
                        d.components[k_comp - 1]
                        for d in lead_draws
                        if len(d.components) >= k_comp
                    ]
                    if len(comp_draws_list) >= 2:
                        comp_arr = np.asarray(comp_draws_list, dtype=float)
                        comp_se = float(np.std(comp_arr, ddof=1))
                        alpha_k = 1 - validated_level / 100
                        if validated_se_boot:
                            ci_lo = float(np.quantile(comp_arr, alpha_k / 2))
                            ci_hi = float(np.quantile(comp_arr, 1 - alpha_k / 2))
                        else:
                            ci_lo, ci_hi = _normal_ci_bounds(tau_k, comp_se, validated_level)
                    else:
                        comp_se = ci_lo = ci_hi = None
                    if comp_se is not None:
                        k_estimate_rows.append(DidEstimateRow(
                            estimator=estimator_label,
                            lead=current_lead,
                            estimate=tau_k,
                            std_error=comp_se,
                            ci_lo=ci_lo,
                            ci_hi=ci_hi,
                            weight=None,
                        ))

            metadata.update({
                "requested_leads": requested_leads,
                "identified_leads": k_identified_leads,
                "filtered_leads": k_unidentified_leads,
                "unidentified_leads": k_unidentified_leads,
                "requested_lead": requested_leads,
                "identified_lead": k_identified_leads,
                "filtered_lead": k_unidentified_leads,
                "unidentified_lead": k_unidentified_leads,
                "n_boot_requested": validated_n_boot,
                "n_boot_realized": validated_n_boot,
                "n_boot": validated_n_boot,
                "n_clusters": int(pd.Series(frame["cluster_source"]).nunique(dropna=True)),
                "datatype": metadata["data_type"],
                "clustvar": metadata["cluster_column"],
                "ci_method": "bootstrap" if validated_se_boot else "asymptotic",
                "se_boot": validated_se_boot,
                "level": validated_level,
                "random_seed": validated_random_seed,
                "covariates": covariate_labels,
                "thres": validated_thres,
                "kmax": validated_kmax,
                "effective_kmax": effective_kmax,
                "jtest": jtest,
                "k_weights_by_lead": {
                    lead_val: {
                        "w_k": k_weights_by_lead[lead_val]["w_k"],
                        "dropped_moments": k_weights_by_lead[lead_val].get("dropped_moments", ()),
                        "jtest_stat": k_weights_by_lead[lead_val].get("jtest_stat"),
                        "jtest_df": k_weights_by_lead[lead_val].get("jtest_df"),
                        "jtest_pval": k_weights_by_lead[lead_val].get("jtest_pval"),
                    }
                    for lead_val in k_identified_leads
                },
                "k_init_by_lead": {
                    lead_val: k_weights_by_lead[lead_val]["k_init"]
                    for lead_val in k_identified_leads
                },
                "k_final_by_lead": {
                    lead_val: k_weights_by_lead[lead_val]["k_final"]
                    for lead_val in k_identified_leads
                },
            })
            return DidResult(
                estimates=tuple(k_estimate_rows),
                metadata=metadata,
                bootstrap_draws=k_draws,
            )

        # ===== Standard SA kmax=2 path (unchanged) =====
        estimates_by_lead, identified_leads, unidentified_leads = _identify_requested_sa_leads(
            frame,
            requested_leads=requested_leads,
            covariates=covariate_specs,
            thres=validated_thres,
        )
        draws = _compute_sa_bootstrap_draws_parallel(
            frame,
            leads=identified_leads,
            covariates=covariate_specs,
            n_boot=validated_n_boot,
            random_seed=validated_random_seed,
            thres=validated_thres,
            cluster_mode=str(metadata["cluster_mode"]),
            n_cores=validated_n_cores,
        ) if validated_parallel and validated_n_boot > 1 else _compute_sa_bootstrap_draws(
            frame,
            leads=identified_leads,
            covariates=covariate_specs,
            n_boot=validated_n_boot,
            random_seed=validated_random_seed,
            thres=validated_thres,
            cluster_mode=str(metadata["cluster_mode"]),
        )
        uncertainty = _summarize_bootstrap_draws(draws, level=validated_level)
        weights_by_lead: dict[int, dict[str, float | tuple[tuple[float, ...], ...] | None]] = {}
        estimate_rows_list: list[DidEstimateRow] = []

        for current_lead in identified_leads:
            lead_draws = tuple(draw for draw in draws if draw.lead == current_lead)
            ddid_row, weights = _compute_sa_double_did_row(
                lead=current_lead,
                component_estimates=estimates_by_lead[current_lead],
                draws=lead_draws,
                se_boot=validated_se_boot,
                level=validated_level,
            )
            weights_by_lead[current_lead] = weights
            if ddid_row is not None:
                estimate_rows_list.append(ddid_row)
            for estimator in ("DID", "sDID"):
                std_error, ci_lo, ci_hi = uncertainty[(current_lead, estimator)]
                if not validated_se_boot:
                    ci_lo, ci_hi = _normal_ci_bounds(
                        estimates_by_lead[current_lead][estimator],
                        std_error,
                        validated_level,
                    )
                estimate_rows_list.append(
                    DidEstimateRow(
                        estimator=f"SA-{estimator}",
                        lead=current_lead,
                        estimate=estimates_by_lead[current_lead][estimator],
                        std_error=std_error,
                        ci_lo=ci_lo,
                        ci_hi=ci_hi,
                        weight=weights["w_did"] if estimator == "DID" else weights["w_sdid"],
                    )
                )

        metadata.update(
            {
                "requested_leads": requested_leads,
                "identified_leads": identified_leads,
                "filtered_leads": unidentified_leads,
                "unidentified_leads": unidentified_leads,
                "requested_lead": requested_leads,
                "identified_lead": identified_leads,
                "filtered_lead": unidentified_leads,
                "unidentified_lead": unidentified_leads,
                "n_boot_requested": validated_n_boot,
                "n_boot_realized": validated_n_boot,
                "n_boot": validated_n_boot,
                "n_clusters": int(pd.Series(frame["cluster_source"]).nunique(dropna=True)),
                "datatype": metadata["data_type"],
                "clustvar": metadata["cluster_column"],
                "ci_method": "bootstrap" if validated_se_boot else "asymptotic",
                "se_boot": validated_se_boot,
                "level": validated_level,
                "random_seed": validated_random_seed,
                "covariates": covariate_labels,
                "thres": validated_thres,
                "weights_by_lead": {
                    current_lead: {
                        "w_did": weights_by_lead[current_lead]["w_did"],
                        "w_sdid": weights_by_lead[current_lead]["w_sdid"],
                    }
                    for current_lead in identified_leads
                },
                "W_by_lead": {
                    current_lead: weights_by_lead[current_lead]["W"]
                    for current_lead in identified_leads
                },
                "vcov_gmm_by_lead": {
                    current_lead: weights_by_lead[current_lead]["vcov_gmm"]
                    for current_lead in identified_leads
                },
                "w_did": weights_by_lead[identified_leads[0]]["w_did"] if len(identified_leads) == 1 else None,
                "w_sdid": weights_by_lead[identified_leads[0]]["w_sdid"] if len(identified_leads) == 1 else None,
                "double_did_available": all(
                    weights_by_lead[current_lead]["w_did"] is not None
                    and weights_by_lead[current_lead]["w_sdid"] is not None
                    for current_lead in identified_leads
                ),
                "double_did_available_leads": tuple(
                    current_lead
                    for current_lead in identified_leads
                    if weights_by_lead[current_lead]["w_did"] is not None
                    and weights_by_lead[current_lead]["w_sdid"] is not None
                ),
            }
        )
        return DidResult(
            estimates=tuple(estimate_rows_list),
            metadata=metadata,
            bootstrap_draws=draws,
        )

    frame, metadata = _prepare_frame(
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

    validated_kmax = _validate_kmax(kmax)

    if validated_kmax > 2:
        # ===== K-DID path (kmax > 2) =====
        if data_type == "rcs":
            raise DidValueError(
                ErrorCode.E008,
                "K-DID with kmax > 2 requires panel data (data_type='panel').",
                context={"kmax": kmax, "data_type": data_type},
            )
        k_estimates_by_lead, k_identified_leads, k_unidentified_leads = _identify_requested_k_leads(
            frame,
            requested_leads=requested_leads,
            kmax=validated_kmax,
            covariates=covariate_specs,
        )
        k_draws = _compute_k_bootstrap_draws_parallel(
            frame,
            leads=k_identified_leads,
            kmax=validated_kmax,
            covariates=covariate_specs,
            n_boot=validated_n_boot,
            random_seed=validated_random_seed,
            n_cores=validated_n_cores,
        ) if validated_parallel and validated_n_boot > 1 else _compute_k_bootstrap_draws(
            frame,
            leads=k_identified_leads,
            kmax=validated_kmax,
            covariates=covariate_specs,
            n_boot=validated_n_boot,
            random_seed=validated_random_seed,
        )
        k_weights_by_lead: dict[int, dict] = {}
        k_estimate_rows: list[DidEstimateRow] = []
        for current_lead in k_identified_leads:
            lead_draws = tuple(d for d in k_draws if d.lead == current_lead)
            kdid_row, weights = _compute_kdid_row(
                lead=current_lead,
                component_estimates=k_estimates_by_lead[current_lead],
                draws=lead_draws,
                se_boot=validated_se_boot,
                jtest=jtest,
                level=validated_level,
            )
            k_weights_by_lead[current_lead] = weights
            if kdid_row is not None:
                k_estimate_rows.append(kdid_row)
            # Add component estimate rows
            k_keys_sorted = sorted(k_estimates_by_lead[current_lead].keys())
            for k_comp in k_keys_sorted:
                tau_k = k_estimates_by_lead[current_lead][k_comp]
                estimator_label = {1: "DID", 2: "sDID"}.get(k_comp, f"kDID-{k_comp}")
                # Get SE from bootstrap draws
                comp_draws_list = [
                    d.components[k_comp - 1]
                    for d in lead_draws
                    if len(d.components) >= k_comp
                ]
                if len(comp_draws_list) >= 2:
                    comp_arr = np.asarray(comp_draws_list, dtype=float)
                    comp_se = float(np.std(comp_arr, ddof=1))
                    alpha_k = 1 - validated_level / 100
                    if validated_se_boot:
                        ci_lo = float(np.quantile(comp_arr, alpha_k / 2))
                        ci_hi = float(np.quantile(comp_arr, 1 - alpha_k / 2))
                    else:
                        ci_lo, ci_hi = _normal_ci_bounds(tau_k, comp_se, validated_level)
                else:
                    comp_se = ci_lo = ci_hi = None
                if comp_se is not None:
                    k_estimate_rows.append(DidEstimateRow(
                        estimator=estimator_label,
                        lead=current_lead,
                        estimate=tau_k,
                        std_error=comp_se,
                        ci_lo=ci_lo,
                        ci_hi=ci_hi,
                        weight=None,
                    ))

        metadata.update({
            "requested_leads": requested_leads,
            "identified_leads": k_identified_leads,
            "filtered_leads": k_unidentified_leads,
            "unidentified_leads": k_unidentified_leads,
            "requested_lead": requested_leads,
            "identified_lead": k_identified_leads,
            "filtered_lead": k_unidentified_leads,
            "unidentified_lead": k_unidentified_leads,
            "n_boot_requested": validated_n_boot,
            "n_boot_realized": validated_n_boot,
            "n_boot": validated_n_boot,
            "n_clusters": int(frame["cluster_id"].nunique()),
            "datatype": metadata.get("data_type", data_type),
            "clustvar": metadata.get("cluster_column", ""),
            "ci_method": "bootstrap" if validated_se_boot else "asymptotic",
            "se_boot": validated_se_boot,
            "level": validated_level,
            "random_seed": validated_random_seed,
            "covariates": covariate_labels,
            "kmax": validated_kmax,
            "jtest": jtest,
            "k_weights_by_lead": {
                lead_val: {
                    "w_k": k_weights_by_lead[lead_val]["w_k"],
                    "dropped_moments": k_weights_by_lead[lead_val].get("dropped_moments", ()),
                    "jtest_stat": k_weights_by_lead[lead_val].get("jtest_stat"),
                    "jtest_df": k_weights_by_lead[lead_val].get("jtest_df"),
                    "jtest_pval": k_weights_by_lead[lead_val].get("jtest_pval"),
                }
                for lead_val in k_identified_leads
            },
            "k_init_by_lead": {
                lead_val: k_weights_by_lead[lead_val]["k_init"]
                for lead_val in k_identified_leads
            },
            "k_final_by_lead": {
                lead_val: k_weights_by_lead[lead_val]["k_final"]
                for lead_val in k_identified_leads
            },
        })
        return DidResult(
            estimates=tuple(k_estimate_rows),
            metadata=metadata,
            bootstrap_draws=k_draws,
        )

    # ===== Standard K=2 path (unchanged) =====
    estimates_by_lead, identified_leads, unidentified_leads = _identify_requested_leads(
        frame,
        requested_leads=requested_leads,
        covariates=covariate_specs,
    )
    draws = _compute_bootstrap_draws_parallel(
        frame,
        leads=identified_leads,
        covariates=covariate_specs,
        n_boot=validated_n_boot,
        random_seed=validated_random_seed,
        n_cores=validated_n_cores,
    ) if validated_parallel and validated_n_boot > 1 else _compute_bootstrap_draws(
        frame,
        leads=identified_leads,
        covariates=covariate_specs,
        n_boot=validated_n_boot,
        random_seed=validated_random_seed,
    )
    uncertainty = _summarize_bootstrap_draws(draws, level=validated_level)
    weights_by_lead: dict[int, dict[str, float | tuple[tuple[float, ...], ...] | None]] = {}
    estimate_rows_list: list[DidEstimateRow] = []

    for current_lead in identified_leads:
        lead_draws = tuple(draw for draw in draws if draw.lead == current_lead)
        ddid_row, weights = _compute_double_did_row(
            lead=current_lead,
            component_estimates=estimates_by_lead[current_lead],
            draws=lead_draws,
            se_boot=validated_se_boot,
            level=validated_level,
        )
        weights_by_lead[current_lead] = weights
        if ddid_row is not None:
            estimate_rows_list.append(ddid_row)
        for estimator in ("DID", "sDID"):
            std_error, ci_lo, ci_hi = uncertainty[(current_lead, estimator)]
            if not validated_se_boot:
                ci_lo, ci_hi = _normal_ci_bounds(
                    estimates_by_lead[current_lead][estimator],
                    std_error,
                    validated_level,
                )
            estimate_rows_list.append(
                DidEstimateRow(
                    estimator=estimator,
                    lead=current_lead,
                    estimate=estimates_by_lead[current_lead][estimator],
                    std_error=std_error,
                    ci_lo=ci_lo,
                    ci_hi=ci_hi,
                    weight=weights["w_did"] if estimator == "DID" else weights["w_sdid"],
                )
            )

    metadata.update(
        {
            "requested_leads": requested_leads,
            "identified_leads": identified_leads,
            "filtered_leads": unidentified_leads,
            "unidentified_leads": unidentified_leads,
            "requested_lead": requested_leads,
            "identified_lead": identified_leads,
            "filtered_lead": unidentified_leads,
            "unidentified_lead": unidentified_leads,
            "n_boot_requested": validated_n_boot,
            "n_boot_realized": validated_n_boot,
            "n_boot": validated_n_boot,
            "n_clusters": int(frame["cluster_id"].nunique()),
            "datatype": metadata["data_type"],
            "clustvar": metadata["cluster_column"],
            "ci_method": "bootstrap" if validated_se_boot else "asymptotic",
            "se_boot": validated_se_boot,
            "level": validated_level,
            "random_seed": validated_random_seed,
            "covariates": covariate_labels,
            "weights_by_lead": {
                current_lead: {
                    "w_did": weights_by_lead[current_lead]["w_did"],
                    "w_sdid": weights_by_lead[current_lead]["w_sdid"],
                }
                for current_lead in identified_leads
            },
            "W_by_lead": {
                current_lead: weights_by_lead[current_lead]["W"]
                for current_lead in identified_leads
            },
            "vcov_gmm_by_lead": {
                current_lead: weights_by_lead[current_lead]["vcov_gmm"]
                for current_lead in identified_leads
            },
            "w_did": weights_by_lead[identified_leads[0]]["w_did"] if len(identified_leads) == 1 else None,
            "w_sdid": weights_by_lead[identified_leads[0]]["w_sdid"] if len(identified_leads) == 1 else None,
            "double_did_available": all(
                weights_by_lead[current_lead]["w_did"] is not None and weights_by_lead[current_lead]["w_sdid"] is not None
                for current_lead in identified_leads
            ),
            "double_did_available_leads": tuple(
                current_lead
                for current_lead in identified_leads
                if weights_by_lead[current_lead]["w_did"] is not None
                and weights_by_lead[current_lead]["w_sdid"] is not None
            ),
        }
    )

    return DidResult(
        estimates=tuple(estimate_rows_list),
        metadata=metadata,
        bootstrap_draws=draws,
    )


__all__ = ["did"]
