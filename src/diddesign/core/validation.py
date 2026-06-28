from __future__ import annotations

import math
import numbers
import re
from collections.abc import Iterable, Mapping
from typing import Any

from ..errors import ErrorCode, DidValueError, DidDataError, WarningCode, did_warn


DataContractError = DidDataError


def _is_bool_like(value: Any) -> bool:
    return isinstance(value, bool) or (
        type(value).__module__ == "numpy"
        and type(value).__name__ in {"bool", "bool_"}
    )


def materialize_rows(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    materialized = list(rows)
    if not materialized:
        raise DidValueError(
            ErrorCode.E003,
            "At least one observation is required.",
            context={"n_rows": 0},
        )
    return materialized


def require_column(
    rows: Iterable[Mapping[str, Any]],
    column: str | None,
    *,
    allow_missing: bool = False,
    field_name: str = "column",
) -> str:
    if not isinstance(column, str):
        raise DidValueError(
            ErrorCode.E001,
            f"{field_name} must be a column name string.",
            context={"field_name": field_name, "received_type": type(column).__name__},
        )
    if not column.strip():
        raise DidValueError(
            ErrorCode.E001,
            f"{field_name} must be a non-empty column name.",
            context={"field_name": field_name, "received_value": repr(column)},
        )

    for index, row in enumerate(rows):
        if column not in row or (not allow_missing and row[column] is None):
            raise DidValueError(
                ErrorCode.E001,
                f"Column '{column}' is required and cannot be missing in row {index}.",
                context={"column": column, "row_index": index, "field_name": field_name},
            )
    return column


def require_binary_indicator(
    rows: Iterable[Mapping[str, Any]],
    *,
    column: str,
    label: str,
) -> None:
    for index, row in enumerate(rows):
        value = row[column]
        if _is_bool_like(value):
            continue
        if isinstance(value, numbers.Real) and math.isfinite(float(value)) and value in {0, 1}:
            continue
        raise DidValueError(
            ErrorCode.E012,
            f"{label} indicator must be binary (0/1); found {value!r} in row {index}.",
            context={"column": column, "label": label, "row_index": index, "actual_value": repr(value)},
        )


def validate_design(design: str, data_type: str) -> None:
    if not isinstance(design, str):
        raise DidValueError(
            ErrorCode.E020,
            "design must be a string.",
            context={"received_type": type(design).__name__},
        )
    if design not in {"did", "sa"}:
        raise DidValueError(
            ErrorCode.E020,
            "design must be either 'did' or 'sa'.",
            context={"received_value": design, "allowed": ["did", "sa"]},
        )

    if not isinstance(data_type, str):
        raise DidValueError(
            ErrorCode.E020,
            "data_type must be a string.",
            context={"received_type": type(data_type).__name__},
        )
    if data_type not in {"panel", "rcs"}:
        raise DidValueError(
            ErrorCode.E020,
            "data_type must be either 'panel' or 'rcs'.",
            context={"received_value": data_type, "allowed": ["panel", "rcs"]},
        )

    if design == "sa" and data_type == "rcs":
        raise DidValueError(
            ErrorCode.E020,
            "design(sa) requires panel data and cannot use the rcs branch.",
            context={"design": design, "data_type": data_type},
        )


def validate_unique_panel_cells(
    rows: Iterable[Mapping[str, Any]],
    *,
    unit_id: str,
    time: str,
) -> None:
    seen: set[tuple[Any, Any]] = set()
    for row in rows:
        cell = (row[unit_id], row[time])
        if cell in seen:
            raise DidValueError(
                ErrorCode.E008,
                "Duplicate panel observations detected: duplicate unit-time observations are not allowed in panel data.",
                context={"unit_id_col": unit_id, "time_col": time, "duplicate_cell": cell},
            )
        seen.add(cell)


def validate_standard_did_panel_treatment_path(
    rows: Iterable[Mapping[str, Any]],
    *,
    unit_id: str,
    time: str,
    treatment: str,
    time_order: tuple[Any, ...],
) -> None:
    time_rank = {level: rank for rank, level in enumerate(time_order)}
    observed_by_unit: dict[Any, list[tuple[int, int]]] = {}
    first_treat_ranks: set[int] = set()
    has_treated_unit = False
    has_never_treated_control_unit = False

    for row in rows:
        treatment_value = row[treatment]
        treatment_flag = int(treatment_value) if not isinstance(treatment_value, bool) else int(treatment_value)
        observed_by_unit.setdefault(row[unit_id], []).append((time_rank[row[time]], treatment_flag))

    for uid, path in observed_by_unit.items():
        ordered_path = sorted(path)
        first_treat_rank: int | None = None
        treated_started = False

        for rank, treatment_flag in ordered_path:
            if treatment_flag == 1 and first_treat_rank is None:
                first_treat_rank = rank
            if treatment_flag == 0 and treated_started:
                raise DidValueError(
                    ErrorCode.E013,
                    "Treatment variable must be cumulative (absorbing) for the standard DID design.",
                    context={"unit_id_col": unit_id, "unit": uid, "treatment_col": treatment},
                )
            treated_started = treated_started or treatment_flag == 1

        if first_treat_rank is not None:
            has_treated_unit = True
            first_treat_ranks.add(first_treat_rank)
        else:
            has_never_treated_control_unit = True

    if len(first_treat_ranks) > 1:
        raise DidValueError(
            ErrorCode.E014,
            "Standard DID requires treated units to share a common treatment adoption time.",
            context={"n_distinct_adoption_times": len(first_treat_ranks), "adoption_ranks": sorted(first_treat_ranks)},
        )

    if not has_treated_unit or not has_never_treated_control_unit:
        raise DidValueError(
            ErrorCode.E003,
            "Standard DID requires at least one treated unit and one never-treated control unit.",
            context={"has_treated": has_treated_unit, "has_control": has_never_treated_control_unit},
        )


def validate_sa_panel_preconditions(
    rows: Iterable[Mapping[str, Any]],
    *,
    unit_id: str,
    time: str,
) -> tuple[Any, ...]:
    require_column(rows, unit_id)
    require_column(rows, time)
    validate_unique_panel_cells(rows, unit_id=unit_id, time=time)

    time_order, _ = resolve_time_order_metadata(rows, time=time)
    if len(time_order) < 3:
        raise DidValueError(
            ErrorCode.E015,
            "design(sa) requires a balanced panel with at least three distinct time periods.",
            context={"n_periods": len(time_order), "time_order": time_order, "minimum_required": 3},
        )

    expected_times = set(time_order)
    observed_by_unit: dict[Any, set[Any]] = {}
    for row in rows:
        observed_by_unit.setdefault(row[unit_id], set()).add(row[time])

    for uid, observed_times in observed_by_unit.items():
        if observed_times != expected_times:
            raise DidValueError(
                ErrorCode.E008,
                "design(sa) requires a balanced panel with one observation per unit-time cell.",
                context={"unit": uid, "expected_periods": len(expected_times), "observed_periods": len(observed_times)},
            )

    return time_order


def validate_sa_treatment_path(
    rows: Iterable[Mapping[str, Any]],
    *,
    unit_id: str,
    time: str,
    treatment: str,
    time_order: tuple[Any, ...],
) -> None:
    time_rank = {level: rank for rank, level in enumerate(time_order)}
    observed_by_unit: dict[Any, list[tuple[int, int]]] = {}
    has_treated = False
    has_control = False

    for row in rows:
        treatment_value = row[treatment]
        treatment_flag = int(treatment_value) if not isinstance(treatment_value, bool) else int(treatment_value)
        has_treated = has_treated or treatment_flag == 1
        has_control = has_control or treatment_flag == 0
        observed_by_unit.setdefault(row[unit_id], []).append((time_rank[row[time]], treatment_flag))

    if not has_treated or not has_control:
        raise DidValueError(
            ErrorCode.E003,
            "Treatment variable must contain both 0 and 1 values for the SA design.",
            context={"has_treated": has_treated, "has_control": has_control, "treatment_col": treatment},
        )

    for uid, path in observed_by_unit.items():
        treated_started = False
        for _, treatment_flag in sorted(path):
            if treatment_flag == 0 and treated_started:
                raise DidValueError(
                    ErrorCode.E013,
                    "Treatment variable must be cumulative (absorbing) for the SA design.",
                    context={"unit_id_col": unit_id, "unit": uid, "treatment_col": treatment},
                )
            treated_started = treated_started or treatment_flag == 1


def validate_rcs_post_indicator(
    rows: Iterable[Mapping[str, Any]],
    *,
    time: str,
    post: str,
    time_order: tuple[Any, ...],
) -> None:
    post_by_time: dict[Any, set[int]] = {}

    for row in rows:
        post_value = row[post]
        post_flag = int(post_value) if not isinstance(post_value, bool) else int(post_value)
        post_by_time.setdefault(row[time], set()).add(post_flag)

    ordered_post_flags: list[int] = []
    entered_post_period = False

    for level in time_order:
        flags = post_by_time.get(level, set())
        if len(flags) != 1:
            raise DidValueError(
                ErrorCode.E016,
                "post indicator must be uniquely determined by time for repeated cross-section data.",
                context={"time_level": level, "post_col": post, "conflicting_values": sorted(flags)},
            )
        post_flag = next(iter(flags))
        ordered_post_flags.append(post_flag)
        if post_flag == 1:
            entered_post_period = True
        elif entered_post_period:
            raise DidValueError(
                ErrorCode.E016,
                "post indicator must switch from 0 to 1 at most once and remain 1 for all later time periods in repeated cross-section data.",
                context={"time_level": level, "post_col": post, "time_order": time_order},
            )

    if 0 not in ordered_post_flags or 1 not in ordered_post_flags:
        raise DidValueError(
            ErrorCode.E016,
            "Repeated cross-section data requires at least one pre-treatment period and one post-treatment period.",
            context={"post_col": post, "observed_flags": ordered_post_flags},
        )


def resolve_time_order(rows: Iterable[Mapping[str, Any]], *, time: str) -> tuple[Any, ...]:
    time_order, _ = resolve_time_order_metadata(rows, time=time)
    return time_order


def resolve_time_order_metadata(
    rows: Iterable[Mapping[str, Any]],
    *,
    time: str,
) -> tuple[tuple[Any, ...], str]:
    observed_levels = tuple(dict.fromkeys(row[time] for row in rows))
    if not observed_levels:
        raise DidValueError(
            ErrorCode.E001,
            "At least one time value is required.",
            context={"time_col": time, "n_levels": 0},
        )

    if all(isinstance(level, str) for level in observed_levels):
        lexical_levels = tuple(sorted(observed_levels))
        if _has_numeric_suffix_order_mismatch(observed_levels, lexical_levels) or observed_levels != lexical_levels:
            raise DidValueError(
                ErrorCode.E019,
                f"Ambiguous string time ordering detected for {time}. "
                "Ambiguous string time order would reorder observed time labels lexicographically. "
                "Automatic encoding would reorder observed time labels lexicographically. "
                "Recode time to numeric or lexically ordered strings before estimation.",
                context={"time_col": time, "observed_order": observed_levels, "lexical_order": lexical_levels},
            )
        return lexical_levels, "string"

    if all(isinstance(level, numbers.Real) and not _is_bool_like(level) for level in observed_levels):
        if not all(math.isfinite(float(level)) for level in observed_levels):
            raise DidValueError(
                ErrorCode.E017,
                f"Time column '{time}' must contain only finite numeric values.",
                context={"time_col": time, "non_finite_values": [v for v in observed_levels if not math.isfinite(float(v))]},
            )
        return tuple(sorted(observed_levels)), "numeric"

    raise DidValueError(
        ErrorCode.E017,
        f"Time column '{time}' must use either all numeric or all string labels.",
        context={"time_col": time, "types_found": list({type(v).__name__ for v in observed_levels})},
    )


def _has_numeric_suffix_order_mismatch(
    observed_levels: tuple[str, ...],
    lexical_levels: tuple[str, ...],
) -> bool:
    if len(observed_levels) <= 1 or len(lexical_levels) <= 1:
        return False

    suffix_pattern = re.compile(r"^(.*?)(\d+)$")
    if not all(suffix_pattern.match(level) for level in lexical_levels):
        return False

    prefixes = {suffix_pattern.match(level).group(1) for level in lexical_levels}
    if len(prefixes) != 1:
        return False

    numeric_levels = tuple(
        level
        for _, level in sorted(
            (int(suffix_pattern.match(level).group(2)), level) for level in lexical_levels
        )
    )
    return lexical_levels != numeric_levels
