from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .validation import (
    DataContractError,
    DidDataError,
    materialize_rows,
    require_binary_indicator,
    require_column,
    resolve_time_order_metadata,
    validate_rcs_post_indicator,
    validate_sa_panel_preconditions,
    validate_design,
    validate_sa_treatment_path,
    validate_standard_did_panel_treatment_path,
    validate_unique_panel_cells,
)


def _require_distinct_role_columns(**roles: str | None) -> None:
    seen: dict[str, str] = {}
    for role, column in roles.items():
        if column is None:
            continue
        previous_role = seen.get(column)
        if previous_role is not None:
            raise DataContractError(
                f"{role} column must be distinct from {previous_role} column."
            )
        seen[column] = role


@dataclass(frozen=True)
class NormalizedDataContract:
    """Minimal normalized metadata for downstream diagnostics and estimators."""

    design: str
    data_type: str
    branch: str
    outcome: str
    treatment: str
    time: str
    unit_id: str | None
    post: str | None
    time_order: tuple[Any, ...]
    cluster_default: str | None
    validation_trace: tuple[str, ...]

    def as_metadata(self) -> dict[str, Any]:
        return {
            "design": self.design,
            "data_type": self.data_type,
            "branch": self.branch,
            "outcome": self.outcome,
            "treatment": self.treatment,
            "time": self.time,
            "unit_id": self.unit_id,
            "post": self.post,
            "time_order": self.time_order,
            "cluster_default": self.cluster_default,
            "validation_trace": self.validation_trace,
        }


def normalize_design_data(
    rows,
    *,
    outcome: str,
    treatment: str,
    time: str,
    unit_id: str | None = None,
    post: str | None = None,
    design: str = "did",
    data_type: str = "panel",
) -> NormalizedDataContract:
    materialized = materialize_rows(rows)
    validate_design(design, data_type)
    require_column(materialized, outcome, allow_missing=True, field_name="outcome")
    require_column(materialized, treatment, field_name="treatment")
    require_column(materialized, time, field_name="time")
    _require_distinct_role_columns(
        outcome=outcome,
        treatment=treatment,
        time=time,
        unit_id=unit_id if data_type == "panel" else None,
        post=post if data_type == "rcs" else None,
    )
    require_binary_indicator(materialized, column=treatment, label="treatment")

    validation_trace: list[str] = [
        "required:outcome",
        "required:treatment",
        "required:time",
        "binary:treatment",
    ]
    cluster_default: str | None = None
    if data_type == "panel":
        if unit_id is None:
            raise DataContractError("unit_id is required for panel data.")
        require_column(materialized, unit_id, field_name="unit_id")
        cluster_default = unit_id
        validation_trace.append("required:unit_id")
        if design == "sa":
            time_order = validate_sa_panel_preconditions(materialized, unit_id=unit_id, time=time)
            validate_sa_treatment_path(
                materialized,
                unit_id=unit_id,
                time=time,
                treatment=treatment,
                time_order=time_order,
            )
            _, time_label_kind = resolve_time_order_metadata(materialized, time=time)
            validation_trace.extend(
                ("balanced-panel", "unique:unit-time", f"time-order:{time_label_kind}", "absorbing:treatment")
            )
        else:
            validate_unique_panel_cells(materialized, unit_id=unit_id, time=time)
            time_order, time_label_kind = resolve_time_order_metadata(materialized, time=time)
            validate_standard_did_panel_treatment_path(
                materialized,
                unit_id=unit_id,
                time=time,
                treatment=treatment,
                time_order=time_order,
            )
            validation_trace.append("unique:unit-time")
            validation_trace.append(f"time-order:{time_label_kind}")
        branch = f"{design}-panel"
    else:
        if post is None:
            raise DataContractError("post is required for repeated cross-section data.")
        require_column(materialized, post, field_name="post")
        require_binary_indicator(materialized, column=post, label="post")
        validation_trace.extend(("required:post", "binary:post"))
        time_order, time_label_kind = resolve_time_order_metadata(materialized, time=time)
        validate_rcs_post_indicator(materialized, time=time, post=post, time_order=time_order)
        validation_trace.append(f"time-order:{time_label_kind}")
        branch = "did-rcs"
    return NormalizedDataContract(
        design=design,
        data_type=data_type,
        branch=branch,
        outcome=outcome,
        treatment=treatment,
        time=time,
        unit_id=unit_id,
        post=post,
        time_order=time_order,
        cluster_default=cluster_default,
        validation_trace=tuple(validation_trace),
    )


__all__ = [
    "DataContractError",
    "DidDataError",
    "NormalizedDataContract",
    "normalize_design_data",
]
