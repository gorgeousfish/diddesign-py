from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from operator import index as _index
from types import MappingProxyType
from typing import Any

import pandas as pd


_ESTIMATE_COLUMNS = ("estimator", "lead", "estimate", "std_error", "ci_lo", "ci_hi", "weight")
_BOOTSTRAP_COLUMNS = ("iteration", "lead", "did", "sdid")
_BOOTSTRAP_K_PREFIX = "component_"
_WEIGHT_COLUMNS = ("lead", "w_did", "w_sdid", "double_did_available")
_GMM_COLUMNS = (
    "lead",
    "w_did",
    "w_sdid",
    "double_did_available",
    "vcov_did",
    "vcov_sdid",
    "vcov_covariance",
    "W_did",
    "W_sdid",
    "W_covariance",
    "gmm_variance",
)
_SUPPORTED_ESTIMATOR_LABELS = frozenset(
    {
        "Double-DID",
        "K-DID",
        "DID",
        "sDID",
        "SA-Double-DID",
        "SA-DID",
        "SA-sDID",
        "SA-K-DID",
    }
)


def _is_bool_like(value: Any) -> bool:
    return isinstance(value, bool) or (
        type(value).__module__ == "numpy"
        and type(value).__name__ in {"bool", "bool_"}
    )


def _freeze_metadata_value(value: Any) -> Any:
    if _is_bool_like(value):
        return bool(value)
    if type(value).__module__ == "numpy" and type(value).__name__.startswith(
        ("int", "uint")
    ):
        return int(value)
    if type(value).__module__ == "numpy" and type(value).__name__.startswith("float"):
        return _require_finite_float(value, field_name="metadata value")
    if isinstance(value, float):
        return _require_finite_float(value, field_name="metadata value")
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


def _thaw_metadata_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _thaw_metadata_value(nested_value)
            for key, nested_value in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_thaw_metadata_value(item) for item in value)
    return value


def _require_finite_float(value: Any, *, field_name: str) -> float:
    if _is_bool_like(value):
        raise ValueError(f"{field_name} must be finite.")
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be finite.") from None
    if not math.isfinite(coerced):
        raise ValueError(f"{field_name} must be finite.")
    return coerced


def _require_optional_finite_float(value: Any, *, field_name: str) -> float | None:
    if value is None:
        return None
    return _require_finite_float(value, field_name=field_name)


def _require_unit_weight_sum(w_did: float | None, w_sdid: float | None) -> None:
    if w_did is None or w_sdid is None:
        return
    if not math.isclose(w_did + w_sdid, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("w_did and w_sdid must sum to 1 when present.")


def _require_bool(value: Any, *, field_name: str) -> bool:
    if _is_bool_like(value):
        return bool(value)
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a boolean.")
    return value


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


def _require_metadata_lead(value: Any, *, field_name: str) -> int:
    if isinstance(value, str):
        try:
            parsed = int(value, 10)
        except ValueError:
            raise ValueError(f"{field_name} lead must be an integer.") from None
        if str(parsed) != value:
            raise ValueError(f"{field_name} lead must be an integer.")
        if parsed < 0:
            raise ValueError(f"{field_name} lead must be non-negative.")
        return parsed
    return _require_nonnegative_int(value, field_name=f"{field_name} lead")


def _require_weight_metadata_lead(value: Any) -> int:
    return _require_metadata_lead(value, field_name="weights_by_lead")


def _require_estimator_label(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("estimator must be a string.")
    if value in _SUPPORTED_ESTIMATOR_LABELS:
        return value
    # Accept kDID-N pattern for K-DID component labels (k >= 3)
    if value.startswith("kDID-") and value[5:].isdigit() and int(value[5:]) >= 3:
        return value
    # Accept SA-kDID-N pattern for SA K-DID component labels (k >= 3)
    if value.startswith("SA-kDID-") and value[8:].isdigit() and int(value[8:]) >= 3:
        return value
    raise ValueError(f"estimator must be one of: {', '.join(sorted(_SUPPORTED_ESTIMATOR_LABELS))}.")


def _require_weight_metadata_mapping(value: Any, *, field_name: str) -> Mapping[Any, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping when present.")
    return value


def _sorted_weight_metadata_items(weights_by_lead: Mapping[Any, Any]) -> tuple[tuple[int, Any, Mapping[Any, Any]], ...]:
    rows: list[tuple[int, Any, Mapping[Any, Any]]] = []
    seen_leads: set[int] = set()
    for lead, weight_values in weights_by_lead.items():
        parsed_lead = _require_weight_metadata_lead(lead)
        if parsed_lead in seen_leads:
            raise ValueError("weights_by_lead contains duplicate lead entries after normalization.")
        seen_leads.add(parsed_lead)
        if not isinstance(weight_values, Mapping):
            raise TypeError("weights_by_lead values must be mappings.")
        rows.append((parsed_lead, lead, weight_values))
    return tuple(sorted(rows, key=lambda item: item[0]))


def _parsed_metadata_leads(mapping: Mapping[Any, Any], *, field_name: str) -> frozenset[int]:
    parsed_leads: set[int] = set()
    for lead in mapping:
        parsed_lead = _require_metadata_lead(lead, field_name=field_name)
        if parsed_lead in parsed_leads:
            raise ValueError(f"{field_name} contains duplicate lead entries after normalization.")
        parsed_leads.add(parsed_lead)
    return frozenset(parsed_leads)


def _metadata_lead_tuple(value: Any, *, field_name: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise TypeError(f"{field_name} must be a sequence of leads when present.")
    parsed_leads: list[int] = []
    seen_leads: set[int] = set()
    for lead in value:
        parsed_lead = _require_metadata_lead(lead, field_name=field_name)
        if parsed_lead in seen_leads:
            raise ValueError(f"{field_name} must not contain duplicate leads.")
        seen_leads.add(parsed_lead)
        parsed_leads.append(parsed_lead)
    return tuple(parsed_leads)


def _metadata_lead_set(value: tuple[int, ...] | None) -> set[int] | None:
    if value is None:
        return None
    return set(value)


def _validate_result_family_metadata(
    estimates: tuple["DidEstimateRow", ...],
    metadata: Mapping[str, Any],
) -> None:
    has_sa_estimates = any(row.estimator.startswith("SA-") for row in estimates)
    has_standard_estimates = any(not row.estimator.startswith("SA-") for row in estimates)

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
        is_sa_branch = branch.startswith("sa-")
        if is_sa_branch and design is None and not has_standard_estimates:
            raise ValueError("SA branch metadata requires design metadata to be 'sa'.")
        if design == "sa" and not is_sa_branch:
            raise ValueError("branch metadata must match estimator labels.")
        if design == "did" and is_sa_branch:
            raise ValueError("branch metadata must match estimator labels.")

    if not estimates:
        return

    if has_sa_estimates and has_standard_estimates:
        raise ValueError("estimates must not mix SA and non-SA estimator labels.")

    expected_design = "sa" if has_sa_estimates else "did"
    if design is not None:
        if design != expected_design:
            raise ValueError("design metadata must match estimator labels.")

    if branch is not None:
        is_sa_branch = branch.startswith("sa-")
        if has_sa_estimates and not is_sa_branch:
            raise ValueError("branch metadata must match estimator labels.")
        if has_standard_estimates and is_sa_branch:
            raise ValueError("branch metadata must match estimator labels.")
    if design is None and has_sa_estimates:
        raise ValueError("SA estimator rows require design metadata to be 'sa'.")


def _estimate_weight_metadata_field(estimator: str) -> str | None:
    if estimator in {"DID", "SA-DID"}:
        return "w_did"
    if estimator in {"sDID", "SA-sDID"}:
        return "w_sdid"
    return None


def _validate_estimate_weight_metadata_consistency(
    estimates: tuple["DidEstimateRow", ...],
    weight_values_by_lead: Mapping[int, Mapping[Any, Any]],
) -> None:
    estimate_values = {(row.estimator, row.lead): row.estimate for row in estimates}
    for row in estimates:
        if row.estimator.endswith("Double-DID"):
            if row.weight is not None:
                raise ValueError("Double-DID estimate rows must not carry component weights.")
            if row.lead not in weight_values_by_lead:
                continue
            weight_values = weight_values_by_lead[row.lead]
            w_did = _require_optional_finite_float(
                weight_values.get("w_did"),
                field_name="w_did",
            )
            w_sdid = _require_optional_finite_float(
                weight_values.get("w_sdid"),
                field_name="w_sdid",
            )
            if w_did is None or w_sdid is None:
                continue
            if row.estimator == "SA-Double-DID":
                component_labels = ("SA-DID", "SA-sDID")
            else:
                component_labels = ("DID", "sDID")
            component_values = (
                estimate_values.get((component_labels[0], row.lead)),
                estimate_values.get((component_labels[1], row.lead)),
            )
            if component_values[0] is None or component_values[1] is None:
                raise ValueError(
                    "Double-DID estimate rows with weights require DID and sDID component rows for the same lead."
                )
            expected_estimate = w_did * component_values[0] + w_sdid * component_values[1]
            if not math.isclose(
                row.estimate,
                expected_estimate,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    "Double-DID estimate rows must match weighted DID and sDID component estimates."
                )
            continue

        metadata_field = _estimate_weight_metadata_field(row.estimator)
        if metadata_field is None:
            continue

        weight_values = weight_values_by_lead.get(row.lead)
        if weight_values is None:
            if row.weight is not None:
                raise ValueError("component estimate row weights require weights_by_lead metadata.")
            continue

        expected_weight = _require_optional_finite_float(
            weight_values.get(metadata_field),
            field_name=metadata_field,
        )
        if expected_weight is None:
            if row.weight is not None:
                raise ValueError("component estimate row weights must match weights_by_lead.")
            continue
        if row.weight is None or not math.isclose(
            row.weight,
            expected_weight,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("component estimate row weights must match weights_by_lead.")
        double_did_label = "SA-Double-DID" if row.estimator.startswith("SA-") else "Double-DID"
        if (double_did_label, row.lead) not in estimate_values:
            raise ValueError("component estimate row weights require a matching Double-DID estimate row.")


def _validate_lead_matrix_metadata(
    matrix_by_lead: Mapping[Any, Any] | None,
    *,
    field_name: str,
    weights_leads: frozenset[int],
) -> None:
    if matrix_by_lead is None:
        return
    matrix_leads = _parsed_metadata_leads(matrix_by_lead, field_name=field_name)
    extra_leads = sorted(matrix_leads - weights_leads)
    if extra_leads:
        raise ValueError(f"{field_name} leads must also be present in weights_by_lead.")
    for matrix in matrix_by_lead.values():
        _matrix_sum(matrix, field_name=f"{field_name} matrix")


def _validate_result_surface_consistency(
    estimates: tuple["DidEstimateRow", ...],
    bootstrap_draws: tuple["DidBootstrapDraw", ...],
    metadata: Mapping[str, Any],
) -> None:
    _validate_result_family_metadata(estimates, metadata)

    estimate_keys: set[tuple[str, int]] = set()
    estimate_leads: set[int] = set()
    for row in estimates:
        key = (row.estimator, row.lead)
        if key in estimate_keys:
            raise ValueError("estimates must not contain duplicate estimator/lead rows.")
        estimate_keys.add(key)
        estimate_leads.add(row.lead)

    lead_metadata: dict[str, tuple[int, ...] | None] = {}
    double_did_available_leads = _metadata_lead_tuple(
        metadata.get("double_did_available_leads"),
        field_name="double_did_available_leads",
    )
    for canonical_field, alias_field in (
        ("requested_leads", "requested_lead"),
        ("identified_leads", "identified_lead"),
        ("filtered_leads", "filtered_lead"),
        ("unidentified_leads", "unidentified_lead"),
    ):
        canonical = _metadata_lead_tuple(metadata.get(canonical_field), field_name=canonical_field)
        alias = _metadata_lead_tuple(metadata.get(alias_field), field_name=alias_field)
        if canonical is not None and alias is not None and canonical != alias:
            raise ValueError(f"{alias_field} must match {canonical_field}.")
        lead_metadata[canonical_field] = canonical if canonical is not None else alias

    requested_leads = lead_metadata["requested_leads"]
    identified_leads = lead_metadata["identified_leads"]
    filtered_leads = lead_metadata["filtered_leads"]
    unidentified_leads = lead_metadata["unidentified_leads"]

    if not estimate_leads and any(
        field in metadata
        for field in (
            "requested_leads",
            "requested_lead",
            "identified_leads",
            "identified_lead",
            "filtered_leads",
            "filtered_lead",
            "unidentified_leads",
            "unidentified_lead",
            "double_did_available_leads",
        )
    ):
        raise ValueError("lead metadata requires estimate rows.")

    identified_set = _metadata_lead_set(identified_leads)
    if identified_set is not None and identified_set != estimate_leads:
        raise ValueError("identified_leads must match estimate leads.")

    requested_set = _metadata_lead_set(requested_leads)
    filtered_set = _metadata_lead_set(filtered_leads)
    unidentified_set = _metadata_lead_set(unidentified_leads)
    if filtered_set is not None and unidentified_set is not None and filtered_set != unidentified_set:
        raise ValueError("filtered_leads must match unidentified_leads.")
    if requested_set is not None and identified_set is not None:
        missing_requested = sorted(identified_set - requested_set)
        if missing_requested:
            raise ValueError("identified_leads must be requested leads.")
    if requested_set is not None and filtered_set is not None:
        extra_filtered = sorted(filtered_set - requested_set)
        if extra_filtered:
            raise ValueError("filtered_leads must be requested leads.")
    if requested_set is not None and identified_set is not None and filtered_set is not None:
        if identified_set & filtered_set:
            raise ValueError("identified_leads and filtered_leads must be disjoint.")
        if identified_set | filtered_set != requested_set:
            raise ValueError("requested_leads must equal identified_leads plus filtered_leads.")

    double_did_set = _metadata_lead_set(double_did_available_leads)
    double_did_estimate_leads = {
        row.lead for row in estimates if row.estimator.endswith("Double-DID")
    }
    if double_did_set is not None:
        if not double_did_set <= estimate_leads:
            raise ValueError("double_did_available_leads must be present in estimates.")
        if double_did_set != double_did_estimate_leads:
            raise ValueError("double_did_available_leads must match Double-DID estimate rows.")

    double_did_available = metadata.get("double_did_available")
    if double_did_available is not None:
        double_did_available = _require_bool(
            double_did_available,
            field_name="double_did_available",
        )
        all_estimate_leads_have_double_did = (
            bool(estimate_leads) and double_did_estimate_leads == estimate_leads
        )
        if double_did_available != all_estimate_leads_have_double_did:
            raise ValueError(
                "double_did_available must match whether every estimate lead has a Double-DID row."
            )
    weights_by_lead = _require_weight_metadata_mapping(
        metadata.get("weights_by_lead"),
        field_name="weights_by_lead",
    )
    W_by_lead = _require_weight_metadata_mapping(
        metadata.get("W_by_lead"),
        field_name="W_by_lead",
    )
    weight_values_by_lead: dict[int, Mapping[Any, Any]] = {}
    if weights_by_lead is not None:
        weight_values_by_lead = {
            lead: weight_values
            for lead, _original_lead, weight_values in _sorted_weight_metadata_items(weights_by_lead)
        }
    _validate_estimate_weight_metadata_consistency(estimates, weight_values_by_lead)
    claimed_double_did_leads: set[int] = set()
    claimed_double_did_leads.update(double_did_estimate_leads)
    if double_did_set is not None:
        claimed_double_did_leads.update(double_did_set)
    if double_did_available is True:
        claimed_double_did_leads.update(double_did_estimate_leads)
    for row in estimates:
        if not row.estimator.endswith("Double-DID"):
            continue
        weight_values = weight_values_by_lead.get(row.lead)
        if weight_values is None:
            continue
        w_did = _require_optional_finite_float(
            weight_values.get("w_did"),
            field_name="w_did",
        )
        w_sdid = _require_optional_finite_float(
            weight_values.get("w_sdid"),
            field_name="w_sdid",
        )
        if w_did is not None and w_sdid is not None:
            claimed_double_did_leads.add(row.lead)
    for lead in claimed_double_did_leads:
        if ("SA-Double-DID", lead) in estimate_keys:
            required_components = {("SA-DID", lead), ("SA-sDID", lead)}
        else:
            required_components = {("DID", lead), ("sDID", lead)}
        if not required_components.issubset(estimate_keys):
            raise ValueError(
                "Double-DID estimate rows require DID and sDID component rows for the same lead."
            )
        if lead not in weight_values_by_lead:
            raise ValueError(
                "Double-DID estimate rows require weights_by_lead entries for each Double-DID lead."
            )
        weight_values = weight_values_by_lead[lead]
        if weight_values.get("w_did") is None or weight_values.get("w_sdid") is None:
            raise ValueError(
                "Double-DID estimate rows require w_did and w_sdid weights for each Double-DID lead."
            )
        W = _lead_metadata_value(W_by_lead, lead, lead)
        if W is None or _gmm_variance_from_weight_matrix(W) is None:
            raise ValueError(
                "Double-DID estimate rows require W_by_lead entries with positive GMM variance for each Double-DID lead."
            )

    draw_keys: set[tuple[int, int]] = set()
    draw_leads_by_iteration: dict[int, set[int]] = {}
    for draw in bootstrap_draws:
        draw_key = (draw.iteration, draw.lead)
        if draw_key in draw_keys:
            raise ValueError("bootstrap_draws must not contain duplicate iteration/lead rows.")
        draw_keys.add(draw_key)
        draw_leads_by_iteration.setdefault(draw.iteration, set()).add(draw.lead)

    missing_draw_leads = sorted({draw.lead for draw in bootstrap_draws} - estimate_leads)
    if missing_draw_leads:
        raise ValueError("bootstrap_draws leads must be present in estimates.")
    for draw_leads in draw_leads_by_iteration.values():
        if draw_leads != estimate_leads:
            raise ValueError("bootstrap_draws must include every estimate lead for each iteration.")
    if draw_leads_by_iteration:
        iterations = tuple(sorted(draw_leads_by_iteration))
        iteration_count = len(iterations)
        expected_iterations = tuple(range(1, len(iterations) + 1))
        if iterations != expected_iterations:
            raise ValueError("bootstrap_draws iteration labels must be contiguous starting at 1.")
        n_boot_realized = metadata.get("n_boot_realized")
        if n_boot_realized is None:
            raise ValueError("n_boot_realized is required when bootstrap_draws are present.")
        if n_boot_realized is not None:
            parsed_n_boot_realized = _require_nonnegative_int(
                n_boot_realized,
                field_name="n_boot_realized",
            )
            if parsed_n_boot_realized != iteration_count:
                raise ValueError("n_boot_realized must match bootstrap iteration count.")
    elif metadata.get("n_boot_realized") is not None:
        iteration_count = 0
        parsed_n_boot_realized = _require_nonnegative_int(
            metadata["n_boot_realized"],
            field_name="n_boot_realized",
        )
        if parsed_n_boot_realized != 0:
            raise ValueError("n_boot_realized must match bootstrap iteration count.")
    else:
        iteration_count = 0
    parsed_n_boot = None
    parsed_n_boot_requested = None
    if metadata.get("n_boot") is not None:
        parsed_n_boot = _require_nonnegative_int(
            metadata["n_boot"],
            field_name="n_boot",
        )
    if metadata.get("n_boot_requested") is not None and metadata.get("n_boot_realized") is not None:
        parsed_n_boot_requested = _require_nonnegative_int(
            metadata["n_boot_requested"],
            field_name="n_boot_requested",
        )
        parsed_n_boot_realized = _require_nonnegative_int(
            metadata["n_boot_realized"],
            field_name="n_boot_realized",
        )
        if parsed_n_boot_requested < parsed_n_boot_realized:
            raise ValueError("n_boot_requested must be greater than or equal to n_boot_realized.")
    if metadata.get("n_boot_requested") is not None:
        parsed_n_boot_requested = _require_nonnegative_int(
            metadata["n_boot_requested"],
            field_name="n_boot_requested",
        )
        if parsed_n_boot is not None and parsed_n_boot != parsed_n_boot_requested:
            raise ValueError("n_boot must match n_boot_requested when both are present.")
    if metadata.get("n_boot_realized") is not None and parsed_n_boot is not None:
        parsed_n_boot_realized = _require_nonnegative_int(
            metadata["n_boot_realized"],
            field_name="n_boot_realized",
        )
        if parsed_n_boot < parsed_n_boot_realized:
            raise ValueError("n_boot must be greater than or equal to n_boot_realized.")
    if not draw_leads_by_iteration and metadata.get("n_boot_realized") is None:
        requested_bootstrap_count = max(
            parsed_n_boot or 0,
            parsed_n_boot_requested or 0,
        )
        if requested_bootstrap_count > 0:
            raise ValueError(
                "n_boot_realized is required when bootstrap metadata requests positive draws."
            )

    if weights_by_lead is None:
        return
    missing_weight_leads = sorted(
        _parsed_metadata_leads(weights_by_lead, field_name="weights_by_lead")
        - estimate_leads
    )
    if missing_weight_leads:
        raise ValueError("weights_by_lead leads must be present in estimates.")


def _validate_result_metadata(metadata: Mapping[str, Any]) -> None:
    weights_by_lead = _require_weight_metadata_mapping(metadata.get("weights_by_lead"), field_name="weights_by_lead")
    W_by_lead = _require_weight_metadata_mapping(metadata.get("W_by_lead"), field_name="W_by_lead")
    vcov_by_lead = _require_weight_metadata_mapping(metadata.get("vcov_gmm_by_lead"), field_name="vcov_gmm_by_lead")
    if weights_by_lead is None:
        if W_by_lead is not None:
            raise ValueError("W_by_lead requires weights_by_lead.")
        if vcov_by_lead is not None:
            raise ValueError("vcov_gmm_by_lead requires weights_by_lead.")
        return

    sorted_weight_items = _sorted_weight_metadata_items(weights_by_lead)
    weight_leads = frozenset(lead for lead, _original_lead, _weight_values in sorted_weight_items)
    for lead, _original_lead, weight_values in sorted_weight_items:
        DidWeightRow(
            lead=lead,
            w_did=weight_values.get("w_did"),
            w_sdid=weight_values.get("w_sdid"),
            double_did_available=(
                weight_values.get("w_did") is not None
                and weight_values.get("w_sdid") is not None
            ),
        )
    _validate_lead_matrix_metadata(W_by_lead, field_name="W_by_lead", weights_leads=weight_leads)
    _validate_lead_matrix_metadata(vcov_by_lead, field_name="vcov_gmm_by_lead", weights_leads=weight_leads)
    if vcov_by_lead is not None:
        for vcov in vcov_by_lead.values():
            _require_positive_semidefinite_matrix(
                vcov,
                field_name="vcov_gmm_by_lead matrix",
            )
    if W_by_lead is not None:
        for lead, original_lead, weight_values in sorted_weight_items:
            w_did = _require_optional_finite_float(
                weight_values.get("w_did"),
                field_name="w_did",
            )
            w_sdid = _require_optional_finite_float(
                weight_values.get("w_sdid"),
                field_name="w_sdid",
            )
            if w_did is None or w_sdid is None:
                if _lead_metadata_value(W_by_lead, original_lead, lead) is not None:
                    raise ValueError("W_by_lead entries require w_did and w_sdid weights.")
                continue
            W = _lead_metadata_value(W_by_lead, original_lead, lead)
            if W is not None and _gmm_variance_from_weight_matrix(W) is None:
                raise ValueError("W_by_lead entries must have positive GMM variance when weights are present.")
            normalized_weights = _normalized_gmm_weights_from_weight_matrix(W)
            if normalized_weights is not None and (
                not math.isclose(w_did, normalized_weights[0], rel_tol=1e-9, abs_tol=1e-9)
                or not math.isclose(w_sdid, normalized_weights[1], rel_tol=1e-9, abs_tol=1e-9)
            ):
                raise ValueError("weights_by_lead must match normalized W_by_lead row sums.")


def _lead_metadata_value(mapping: Mapping[Any, Any] | None, original_lead: Any, parsed_lead: int) -> Any:
    if mapping is None:
        return None
    if original_lead in mapping:
        return mapping[original_lead]
    if parsed_lead in mapping:
        return mapping[parsed_lead]
    string_lead = str(parsed_lead)
    if string_lead in mapping:
        return mapping[string_lead]
    return None


def _matrix_entry(
    matrix: Any,
    row: int,
    column: int,
    *,
    field_name: str,
) -> float | None:
    if matrix is None:
        return None
    try:
        row_count = len(matrix)
        column_counts = tuple(len(matrix[index]) for index in range(row_count))
    except TypeError:
        raise ValueError(f"{field_name} must be a 2x2 finite numeric matrix when present.") from None
    if row_count != 2 or column_counts != (2, 2):
        raise ValueError(f"{field_name} must be a 2x2 finite numeric matrix when present.")
    try:
        value = matrix[row][column]
    except (IndexError, KeyError, TypeError):
        raise ValueError(f"{field_name} must be a 2x2 finite numeric matrix when present.") from None
    if _is_bool_like(value):
        raise ValueError(f"{field_name} must be a 2x2 finite numeric matrix when present.")
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be a 2x2 finite numeric matrix when present.") from None
    if not math.isfinite(coerced):
        raise ValueError(f"{field_name} must be a 2x2 finite numeric matrix when present.")
    return coerced


def _matrix_sum(matrix: Any, *, field_name: str) -> float | None:
    if matrix is None:
        return None
    entries = (
        _matrix_entry(matrix, 0, 0, field_name=field_name),
        _matrix_entry(matrix, 0, 1, field_name=field_name),
        _matrix_entry(matrix, 1, 0, field_name=field_name),
        _matrix_entry(matrix, 1, 1, field_name=field_name),
    )
    if not math.isclose(entries[1], entries[2], rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{field_name} must be symmetric when present.")
    return float(sum(value for value in entries if value is not None))


def _require_positive_semidefinite_matrix(matrix: Any, *, field_name: str) -> None:
    if matrix is None:
        return
    top_left = _matrix_entry(matrix, 0, 0, field_name=field_name)
    covariance = _matrix_entry(matrix, 0, 1, field_name=field_name)
    covariance_symmetric = _matrix_entry(matrix, 1, 0, field_name=field_name)
    bottom_right = _matrix_entry(matrix, 1, 1, field_name=field_name)
    if not math.isclose(covariance, covariance_symmetric, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{field_name} must be symmetric when present.")
    determinant = top_left * bottom_right - covariance * covariance
    if top_left < 0 or bottom_right < 0 or determinant < -1e-9:
        raise ValueError(f"{field_name} must be positive semidefinite when present.")


def _gmm_variance_from_weight_matrix(matrix: Any) -> float | None:
    if matrix is None:
        return None
    w_did = _matrix_entry(matrix, 0, 0, field_name="W_by_lead matrix")
    w_cov = _matrix_entry(matrix, 0, 1, field_name="W_by_lead matrix")
    w_cov_symmetric = _matrix_entry(matrix, 1, 0, field_name="W_by_lead matrix")
    w_sdid = _matrix_entry(matrix, 1, 1, field_name="W_by_lead matrix")
    if not math.isclose(w_cov, w_cov_symmetric, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("W_by_lead matrix must be symmetric when present.")
    determinant = w_did * w_sdid - w_cov * w_cov
    if w_did <= 0 or w_sdid <= 0 or determinant <= 0:
        return None
    total = w_did + w_cov + w_cov_symmetric + w_sdid
    if total is None or total <= 0:
        return None
    return 1.0 / total


def _normalized_gmm_weights_from_weight_matrix(matrix: Any) -> tuple[float, float] | None:
    if matrix is None or _gmm_variance_from_weight_matrix(matrix) is None:
        return None
    w_did = _matrix_entry(matrix, 0, 0, field_name="W_by_lead matrix")
    w_cov = _matrix_entry(matrix, 0, 1, field_name="W_by_lead matrix")
    w_cov_symmetric = _matrix_entry(matrix, 1, 0, field_name="W_by_lead matrix")
    w_sdid = _matrix_entry(matrix, 1, 1, field_name="W_by_lead matrix")
    total = w_did + w_cov + w_cov_symmetric + w_sdid
    return ((w_did + w_cov) / total, (w_cov_symmetric + w_sdid) / total)


@dataclass(frozen=True)
class DidEstimateRow:
    """Stable serialized row for a single estimator/lead estimate."""

    estimator: str
    lead: int
    estimate: float
    std_error: float | None
    ci_lo: float | None
    ci_hi: float | None
    weight: float | None = None

    def __post_init__(self) -> None:
        estimator = _require_estimator_label(self.estimator)
        lead = _require_nonnegative_int(self.lead, field_name="lead")
        estimate = _require_finite_float(self.estimate, field_name="estimate")
        std_error = _require_optional_finite_float(self.std_error, field_name="std_error")
        if std_error is not None and std_error < 0:
            raise ValueError("std_error must be non-negative.")
        ci_lo = _require_optional_finite_float(self.ci_lo, field_name="ci_lo")
        ci_hi = _require_optional_finite_float(self.ci_hi, field_name="ci_hi")
        if (ci_lo is None) != (ci_hi is None):
            raise ValueError("ci_lo and ci_hi must be jointly present or jointly missing.")
        if std_error is None and ci_lo is not None:
            raise ValueError("ci_lo and ci_hi require std_error.")
        if ci_lo is not None and ci_hi is not None and ci_lo > ci_hi:
            raise ValueError("ci_lo must be less than or equal to ci_hi.")
        if std_error == 0 and ci_lo is not None and (
            not math.isclose(ci_lo, estimate, rel_tol=1e-9, abs_tol=1e-9)
            or not math.isclose(ci_hi, estimate, rel_tol=1e-9, abs_tol=1e-9)
        ):
            raise ValueError("ci_lo and ci_hi must equal estimate when std_error is zero.")
        weight = _require_optional_finite_float(self.weight, field_name="weight")
        if estimator in {"Double-DID", "SA-Double-DID"} and weight is not None:
            raise ValueError("Double-DID estimate rows must not carry component weights.")
        object.__setattr__(self, "estimator", estimator)
        object.__setattr__(self, "lead", lead)
        object.__setattr__(self, "estimate", estimate)
        object.__setattr__(self, "std_error", std_error)
        object.__setattr__(self, "ci_lo", ci_lo)
        object.__setattr__(self, "ci_hi", ci_hi)
        object.__setattr__(self, "weight", weight)

    def as_dict(self) -> dict[str, str | int | float | None]:
        return {
            "estimator": self.estimator,
            "lead": self.lead,
            "estimate": self.estimate,
            "std_error": self.std_error,
            "ci_lo": self.ci_lo,
            "ci_hi": self.ci_hi,
            "weight": self.weight,
        }


@dataclass(frozen=True)
class DidBootstrapDraw:
    """Per-bootstrap DID and sequential-DID component estimates."""

    iteration: int
    lead: int
    did: float
    sdid: float

    def __post_init__(self) -> None:
        iteration = _require_nonnegative_int(self.iteration, field_name="iteration")
        if iteration == 0:
            raise ValueError("iteration must be positive.")
        object.__setattr__(self, "iteration", iteration)
        object.__setattr__(self, "lead", _require_nonnegative_int(self.lead, field_name="lead"))
        object.__setattr__(self, "did", _require_finite_float(self.did, field_name="did"))
        object.__setattr__(self, "sdid", _require_finite_float(self.sdid, field_name="sdid"))

    def as_dict(self) -> dict[str, int | float]:
        return {
            "iteration": self.iteration,
            "lead": self.lead,
            "did": self.did,
            "sdid": self.sdid,
        }


@dataclass(frozen=True)
class DidBootstrapDrawK:
    """Per-bootstrap K-dimensional component estimates for K-DID."""

    iteration: int
    lead: int
    components: tuple[float, ...]

    def __post_init__(self) -> None:
        iteration = _require_nonnegative_int(self.iteration, field_name="iteration")
        if iteration == 0:
            raise ValueError("iteration must be positive.")
        object.__setattr__(self, "iteration", iteration)
        object.__setattr__(self, "lead", _require_nonnegative_int(self.lead, field_name="lead"))
        if not isinstance(self.components, tuple):
            object.__setattr__(self, "components", tuple(self.components))
        for i, c in enumerate(self.components):
            if not isinstance(c, (int, float)) or not math.isfinite(c):
                raise ValueError(f"components[{i}] must be finite.")
        if len(self.components) < 1:
            raise ValueError("components must have at least one element.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "lead": self.lead,
            "components": self.components,
        }


@dataclass(frozen=True)
class DidWeightRow:
    """Stable serialized row for per-lead Double-DID weights."""

    lead: int
    w_did: float | None
    w_sdid: float | None
    double_did_available: bool

    def __post_init__(self) -> None:
        w_did = _require_optional_finite_float(self.w_did, field_name="w_did")
        w_sdid = _require_optional_finite_float(self.w_sdid, field_name="w_sdid")
        double_did_available = _require_bool(
            self.double_did_available,
            field_name="double_did_available",
        )
        if (w_did is None) != (w_sdid is None):
            raise ValueError("w_did and w_sdid must be jointly present or jointly missing.")
        _require_unit_weight_sum(w_did, w_sdid)
        if double_did_available != (w_did is not None and w_sdid is not None):
            raise ValueError("double_did_available must match weight availability.")
        object.__setattr__(self, "lead", _require_nonnegative_int(self.lead, field_name="lead"))
        object.__setattr__(self, "w_did", w_did)
        object.__setattr__(self, "w_sdid", w_sdid)
        object.__setattr__(self, "double_did_available", double_did_available)

    def as_dict(self) -> dict[str, int | float | bool | None]:
        return {
            "lead": self.lead,
            "w_did": self.w_did,
            "w_sdid": self.w_sdid,
            "double_did_available": self.double_did_available,
        }


@dataclass(frozen=True)
class DidGmmRow:
    """Stable per-lead GMM weighting matrix row."""

    lead: int
    w_did: float | None
    w_sdid: float | None
    double_did_available: bool
    vcov_did: float | None
    vcov_sdid: float | None
    vcov_covariance: float | None
    W_did: float | None
    W_sdid: float | None
    W_covariance: float | None
    gmm_variance: float | None

    def __post_init__(self) -> None:
        w_did = _require_optional_finite_float(self.w_did, field_name="w_did")
        w_sdid = _require_optional_finite_float(self.w_sdid, field_name="w_sdid")
        gmm_variance = _require_optional_finite_float(self.gmm_variance, field_name="gmm_variance")
        vcov_did = _require_optional_finite_float(self.vcov_did, field_name="vcov_did")
        vcov_sdid = _require_optional_finite_float(self.vcov_sdid, field_name="vcov_sdid")
        vcov_covariance = _require_optional_finite_float(
            self.vcov_covariance,
            field_name="vcov_covariance",
        )
        W_did = _require_optional_finite_float(self.W_did, field_name="W_did")
        W_sdid = _require_optional_finite_float(self.W_sdid, field_name="W_sdid")
        W_covariance = _require_optional_finite_float(
            self.W_covariance,
            field_name="W_covariance",
        )
        double_did_available = _require_bool(
            self.double_did_available,
            field_name="double_did_available",
        )
        if gmm_variance is not None and gmm_variance <= 0:
            raise ValueError("gmm_variance must be positive when present.")
        weights_present = w_did is not None and w_sdid is not None
        if (w_did is None) != (w_sdid is None):
            raise ValueError("w_did and w_sdid must be jointly present or jointly missing.")
        _require_unit_weight_sum(w_did, w_sdid)
        vcov_entries = (vcov_did, vcov_sdid, vcov_covariance)
        if any(value is None for value in vcov_entries) and any(
            value is not None for value in vcov_entries
        ):
            raise ValueError("vcov GMM entries must be jointly present or jointly missing.")
        if all(value is not None for value in vcov_entries):
            vcov_determinant = vcov_did * vcov_sdid - vcov_covariance * vcov_covariance
            if vcov_did < 0 or vcov_sdid < 0 or vcov_determinant < -1e-9:
                raise ValueError("vcov GMM entries must be positive semidefinite.")
        W_entries = (W_did, W_sdid, W_covariance)
        if any(value is None for value in W_entries) and any(
            value is not None for value in W_entries
        ):
            raise ValueError("W GMM entries must be jointly present or jointly missing.")
        W_present = all(value is not None for value in W_entries)
        if gmm_variance is not None and not W_present:
            raise ValueError("gmm_variance requires W GMM entries.")
        if gmm_variance is None and W_present:
            raise ValueError("W GMM entries require gmm_variance.")
        if W_present:
            W_total = W_did + 2.0 * W_covariance + W_sdid
            W_determinant = W_did * W_sdid - W_covariance * W_covariance
            if W_did <= 0 or W_sdid <= 0 or W_determinant <= 0 or W_total <= 0:
                raise ValueError("W GMM entries must imply positive GMM variance.")
            expected_gmm_variance = 1.0 / W_total
            if not math.isclose(
                gmm_variance,
                expected_gmm_variance,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ValueError("gmm_variance must match W GMM entries.")
        if gmm_variance is not None and not weights_present:
            raise ValueError("gmm_variance requires w_did and w_sdid weights.")
        if double_did_available and (w_did is None or w_sdid is None or gmm_variance is None):
            raise ValueError("double_did_available GMM rows require weights and gmm_variance.")
        if double_did_available != (weights_present and gmm_variance is not None):
            raise ValueError(
                "double_did_available must match GMM weight and variance availability."
            )
        if W_present and weights_present:
            W_total = W_did + 2.0 * W_covariance + W_sdid
            expected_w_did = (W_did + W_covariance) / W_total
            expected_w_sdid = (W_covariance + W_sdid) / W_total
            if not math.isclose(w_did, expected_w_did, rel_tol=1e-9, abs_tol=1e-9) or not math.isclose(
                w_sdid,
                expected_w_sdid,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ValueError("weights must match normalized W GMM row sums.")
        object.__setattr__(self, "lead", _require_nonnegative_int(self.lead, field_name="lead"))
        object.__setattr__(self, "w_did", w_did)
        object.__setattr__(self, "w_sdid", w_sdid)
        object.__setattr__(self, "double_did_available", double_did_available)
        object.__setattr__(self, "vcov_did", vcov_did)
        object.__setattr__(self, "vcov_sdid", vcov_sdid)
        object.__setattr__(self, "vcov_covariance", vcov_covariance)
        object.__setattr__(self, "W_did", W_did)
        object.__setattr__(self, "W_sdid", W_sdid)
        object.__setattr__(self, "W_covariance", W_covariance)
        object.__setattr__(self, "gmm_variance", gmm_variance)

    def as_dict(self) -> dict[str, int | float | bool | None]:
        return {
            "lead": self.lead,
            "w_did": self.w_did,
            "w_sdid": self.w_sdid,
            "double_did_available": self.double_did_available,
            "vcov_did": self.vcov_did,
            "vcov_sdid": self.vcov_sdid,
            "vcov_covariance": self.vcov_covariance,
            "W_did": self.W_did,
            "W_sdid": self.W_sdid,
            "W_covariance": self.W_covariance,
            "gmm_variance": self.gmm_variance,
        }


@dataclass(frozen=True)
class DidResult:
    """Immutable result of DID, Double-DID, K-DID, or SA estimation.

    Returned by :func:`did`, this object stores the full estimation output:
    point estimates for each component estimator, bootstrap draws used to
    compute standard errors and GMM weights, and metadata recording the
    design configuration.

    The object is frozen at construction time; nested metadata dictionaries
    are recursively converted to immutable mappings so that downstream code
    cannot accidentally mutate the estimation record.

    Frame accessors provide the primary interface for reporting:

    - ``to_estimates_frame()`` — component and combined estimates as a
      pandas DataFrame with columns: estimator, lead, estimate, std_error,
      ci_lo, ci_hi, weight.
    - ``to_bootstrap_frame()`` — raw bootstrap draws (iterations × components)
      for custom inference or diagnostic inspection.
    - ``to_weights_frame()`` — per-lead GMM weights showing how the data
      allocate efficiency across DID and sDID.
    - ``to_gmm_frame()`` — full GMM calculation rows including the bootstrap
      covariance matrix and weight matrix entries.
    - ``to_k_weights_frame()`` — K-dimensional weight vectors for K-DID.
    - ``to_latex()`` — publication-ready LaTeX table string.
    """

    estimates: tuple[DidEstimateRow, ...]
    metadata: Mapping[str, Any]
    bootstrap_draws: tuple[DidBootstrapDraw | DidBootstrapDrawK, ...]

    def __post_init__(self) -> None:
        estimates = tuple(self.estimates)
        bootstrap_draws = tuple(self.bootstrap_draws)
        if any(not isinstance(row, DidEstimateRow) for row in estimates):
            raise TypeError("estimates must contain DidEstimateRow instances.")
        if any(not isinstance(draw, (DidBootstrapDraw, DidBootstrapDrawK)) for draw in bootstrap_draws):
            raise TypeError("bootstrap_draws must contain DidBootstrapDraw or DidBootstrapDrawK instances.")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")
        metadata = dict(self.metadata)
        # Skip strict 2D weight/Double-DID validation for K-DID results
        is_kdid = any(
            row.estimator in {"K-DID", "SA-K-DID"}
            or row.estimator.startswith("kDID-")
            or row.estimator.startswith("SA-kDID-")
            for row in estimates
        )
        if not is_kdid:
            _validate_result_metadata(metadata)
        object.__setattr__(self, "estimates", estimates)
        object.__setattr__(self, "bootstrap_draws", bootstrap_draws)
        object.__setattr__(self, "metadata", _freeze_metadata_value(metadata))
        if not is_kdid:
            _validate_result_surface_consistency(
                self.estimates,
                self.bootstrap_draws,
                self.metadata,
            )

    def estimate_rows(self) -> tuple[DidEstimateRow, ...]:
        return self.estimates

    def component_estimate_rows(self) -> tuple[DidEstimateRow, ...]:
        return tuple(row for row in self.estimates if "Double-DID" not in row.estimator)

    def weight_rows(self) -> tuple[DidWeightRow, ...]:
        weights_by_lead = _require_weight_metadata_mapping(self.metadata.get("weights_by_lead"), field_name="weights_by_lead")
        if weights_by_lead is None:
            return ()

        rows: list[DidWeightRow] = []
        for lead, _original_lead, weight_values in _sorted_weight_metadata_items(weights_by_lead):
            w_did = weight_values.get("w_did")
            w_sdid = weight_values.get("w_sdid")
            rows.append(
                DidWeightRow(
                    lead=int(lead),
                    w_did=w_did,
                    w_sdid=w_sdid,
                    double_did_available=w_did is not None and w_sdid is not None,
                )
            )
        return tuple(rows)

    def gmm_rows(self) -> tuple[DidGmmRow, ...]:
        weights_by_lead = _require_weight_metadata_mapping(self.metadata.get("weights_by_lead"), field_name="weights_by_lead")
        W_by_lead = _require_weight_metadata_mapping(self.metadata.get("W_by_lead"), field_name="W_by_lead")
        vcov_by_lead = _require_weight_metadata_mapping(self.metadata.get("vcov_gmm_by_lead"), field_name="vcov_gmm_by_lead")
        if weights_by_lead is None:
            return ()

        rows: list[DidGmmRow] = []
        for lead, original_lead, weight_values in _sorted_weight_metadata_items(weights_by_lead):
            W = _lead_metadata_value(W_by_lead, original_lead, lead)
            vcov = _lead_metadata_value(vcov_by_lead, original_lead, lead)
            w_did = _require_optional_finite_float(weight_values.get("w_did"), field_name="w_did")
            w_sdid = _require_optional_finite_float(weight_values.get("w_sdid"), field_name="w_sdid")
            gmm_variance = _gmm_variance_from_weight_matrix(W)
            rows.append(
                DidGmmRow(
                    lead=int(lead),
                    w_did=w_did,
                    w_sdid=w_sdid,
                    double_did_available=w_did is not None and w_sdid is not None and gmm_variance is not None,
                    vcov_did=_matrix_entry(vcov, 0, 0, field_name="vcov_gmm_by_lead matrix"),
                    vcov_sdid=_matrix_entry(vcov, 1, 1, field_name="vcov_gmm_by_lead matrix"),
                    vcov_covariance=_matrix_entry(vcov, 0, 1, field_name="vcov_gmm_by_lead matrix"),
                    W_did=_matrix_entry(W, 0, 0, field_name="W_by_lead matrix"),
                    W_sdid=_matrix_entry(W, 1, 1, field_name="W_by_lead matrix"),
                    W_covariance=_matrix_entry(W, 0, 1, field_name="W_by_lead matrix"),
                    gmm_variance=gmm_variance,
                )
            )
        return tuple(rows)

    def to_serialized_result(self) -> dict[str, Any]:
        """Return detached serialized estimate, draw, weight, GMM, and metadata records."""

        return {
            "estimates": tuple(row.as_dict() for row in self.estimates),
            "bootstrap_draws": tuple(draw.as_dict() for draw in self.bootstrap_draws),
            "weights": tuple(row.as_dict() for row in self.weight_rows()),
            "gmm": tuple(row.as_dict() for row in self.gmm_rows()),
            "metadata": _thaw_metadata_value(self.metadata),
        }

    def as_payload(self) -> dict[str, Any]:
        """Compatibility alias for :meth:`to_serialized_result`."""

        return self.to_serialized_result()

    def to_estimates_frame(self) -> pd.DataFrame:
        """Return detached estimator rows as a stable pandas DataFrame."""

        return pd.DataFrame.from_records(
            (row.as_dict() for row in self.estimates),
            columns=_ESTIMATE_COLUMNS,
        )

    def to_bootstrap_frame(self) -> pd.DataFrame:
        """Return detached bootstrap draw rows as a stable pandas DataFrame."""

        if any(isinstance(draw, DidBootstrapDrawK) for draw in self.bootstrap_draws):
            if any(isinstance(draw, DidBootstrapDraw) for draw in self.bootstrap_draws):
                raise TypeError("bootstrap_draws cannot mix DID and K-DID draw records.")
            max_components = max(
                (len(draw.components) for draw in self.bootstrap_draws),
                default=0,
            )
            component_columns = tuple(
                f"{_BOOTSTRAP_K_PREFIX}{component_index}"
                for component_index in range(1, max_components + 1)
            )
            columns = ("iteration", "lead", *component_columns)
            records: list[dict[str, int | float | None]] = []
            for draw in self.bootstrap_draws:
                record: dict[str, int | float | None] = {
                    "iteration": draw.iteration,
                    "lead": draw.lead,
                }
                for component_index, component_value in enumerate(draw.components, start=1):
                    record[f"{_BOOTSTRAP_K_PREFIX}{component_index}"] = component_value
                records.append(record)
            return pd.DataFrame.from_records(records, columns=columns)

        return pd.DataFrame.from_records(
            (draw.as_dict() for draw in self.bootstrap_draws),
            columns=_BOOTSTRAP_COLUMNS,
        )

    def to_weights_frame(self) -> pd.DataFrame:
        """Return detached per-lead GMM weight rows as a stable pandas DataFrame."""

        return pd.DataFrame.from_records(
            (row.as_dict() for row in self.weight_rows()),
            columns=_WEIGHT_COLUMNS,
        )

    def to_gmm_frame(self) -> pd.DataFrame:
        """Return detached per-lead GMM matrix rows as a stable DataFrame."""

        return pd.DataFrame.from_records(
            (row.as_dict() for row in self.gmm_rows()),
            columns=_GMM_COLUMNS,
        )

    def report(self, verbose: int = 1) -> None:
        """Print diagnostic report for this estimation result.
    
        Parameters
        ----------
        verbose : int, default 1
            0=silent, 1=summary, 2=detailed
        """
        from ..diagnostics_reporter import DiagnosticsReporter
        reporter = DiagnosticsReporter(self, verbose=verbose)
        reporter.print_report()
    
    def report_dict(self) -> dict[str, Any]:
        """Return diagnostic info as a dictionary."""
        from ..diagnostics_reporter import DiagnosticsReporter
        reporter = DiagnosticsReporter(self, verbose=2)
        return reporter.to_dict()
    
    def to_dataframe(self) -> "pd.DataFrame":
        """Convert estimates to a pandas DataFrame.

        Returns a DataFrame with columns: estimator, lead, estimate,
        std_error, ci_lo, ci_hi, weight.

        Returns
        -------
        pd.DataFrame

        Examples
        --------
        >>> result = did(data, ...)
        >>> df = result.to_dataframe()
        >>> df[df["estimator"] == "Double-DID"]
        """
        import pandas as pd
        rows = []
        for est in self.estimates:
            rows.append({
                "estimator": est.estimator,
                "lead": est.lead,
                "estimate": est.estimate,
                "std_error": est.std_error,
                "ci_lo": est.ci_lo,
                "ci_hi": est.ci_hi,
                "weight": est.weight,
            })
        return pd.DataFrame(rows)

    def to_latex(
        self,
        *,
        caption: str | None = None,
        label: str | None = None,
        decimal_places: int = 4,
        include_ci: bool = True,
        include_weight: bool = False,
        stars: bool = True,
    ) -> str:
        """Export estimates as a LaTeX table string.

        Produces a publication-ready LaTeX tabular environment suitable for
        direct inclusion in academic papers.

        Parameters
        ----------
        caption : str, optional
            Table caption. If None, no caption is added.
        label : str, optional
            LaTeX label for cross-referencing.
        decimal_places : int, optional
            Number of decimal places for numerical values. Default 4.
        include_ci : bool, optional
            Whether to include confidence interval columns. Default True.
        include_weight : bool, optional
            Whether to include GMM weight column. Default False.
        stars : bool, optional
            Whether to add significance stars (\* p<0.10, \*\* p<0.05, \*\*\* p<0.01).
            Uses normal approximation: abs(estimate/se) > z_crit. Default True.

        Returns
        -------
        str
            Complete LaTeX table string (with \\begin{table}...\\end{table}).

        Examples
        --------
        >>> result = did(data, ...)
        >>> print(result.to_latex(caption="Double DID Estimates"))
        >>> # Write to file
        >>> with open("table1.tex", "w") as f:
        ...     f.write(result.to_latex())
        """
        fmt = f"{{:.{decimal_places}f}}"

        def _fmt(value: float | None) -> str:
            if value is None:
                return ""
            return fmt.format(value)

        def _stars(estimate: float, std_error: float | None) -> str:
            if not stars or std_error is None or std_error == 0:
                return ""
            z = abs(estimate / std_error)
            if z > 2.576:
                return "$^{***}$"
            elif z > 1.96:
                return "$^{**}$"
            elif z > 1.645:
                return "$^{*}$"
            return ""

        # Build column spec
        col_headers = ["Estimator", "Lead", "Estimate", "Std. Error"]
        col_align = "llrr"
        if include_ci:
            col_headers += ["CI Low", "CI High"]
            col_align += "rr"
        if include_weight:
            col_headers += ["Weight"]
            col_align += "r"

        lines: list[str] = []
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        if caption is not None:
            lines.append(r"\caption{" + caption + "}")
        if label is not None:
            lines.append(r"\label{" + label + "}")
        lines.append(r"\begin{tabular}{" + col_align + "}")
        lines.append(r"\hline\hline")
        lines.append(" & ".join(col_headers) + r" \\")
        lines.append(r"\hline")

        for row in self.estimates:
            est_str = _fmt(row.estimate) + _stars(row.estimate, row.std_error)
            cells = [
                row.estimator,
                str(row.lead),
                est_str,
                _fmt(row.std_error),
            ]
            if include_ci:
                cells.append(_fmt(row.ci_lo))
                cells.append(_fmt(row.ci_hi))
            if include_weight:
                cells.append(_fmt(row.weight))
            lines.append(" & ".join(cells) + r" \\")

        lines.append(r"\hline\hline")
        if stars:
            lines.append(
                r"\multicolumn{"
                + str(len(col_headers))
                + r"}{l}{\footnotesize Note: $^{*}$p$<$0.10, $^{**}$p$<$0.05, $^{***}$p$<$0.01} \\"
            )
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        return "\n".join(lines)

    def to_polars(self) -> "pl.DataFrame":
        """Convert estimates to a polars DataFrame.

        Requires polars to be installed.

        Returns
        -------
        polars.DataFrame

        Raises
        ------
        ImportError
            If polars is not installed.
        """
        try:
            import polars as pl
        except ImportError:
            raise ImportError(
                "polars is required for to_polars(). Install with: pip install polars"
            )
        return pl.from_pandas(self.to_dataframe())

    def __repr__(self) -> str:
        """Human-readable summary for REPL/Notebook display."""
        lines = [f"DidResult(design='{self.metadata.get('design', '?')}', "
                 f"data_type='{self.metadata.get('data_type', '?')}', "
                 f"n_boot={self.metadata.get('n_boot_realized', '?')})"]
        lines.append(f"  Estimates ({len(self.estimates)} rows):")
        for est in self.estimates:
            star = ""
            if est.std_error and est.std_error > 0:
                z = abs(est.estimate / est.std_error)
                if z > 2.576:
                    star = "***"
                elif z > 1.96:
                    star = "**"
                elif z > 1.645:
                    star = "*"
            se_str = f"  SE={est.std_error:.4f}" if est.std_error else ""
            ci_str = f"  CI=[{est.ci_lo:.4f}, {est.ci_hi:.4f}]" if est.ci_lo is not None else ""
            lines.append(f"    {est.estimator:15s} lead={est.lead}: "
                         f"{est.estimate:.6f}{star}{se_str}{ci_str}")
        w_rows = self.weight_rows()
        if w_rows:
            w = w_rows[0]
            if w.double_did_available:
                lines.append(f"  Weights: w_DID={w.w_did:.4f}, w_sDID={w.w_sdid:.4f}")
        return "\n".join(lines)

    def to_k_weights_frame(self) -> pd.DataFrame:
        """Return K-dimensional GMM weight rows for K-DID/SA-K-DID results.

        For results with kmax > 2, the per-lead weight vector has K components.
        This method extracts weights from the ``k_weights_by_lead`` metadata
        and returns a DataFrame with columns: lead, k, weight, plus J-test info.

        Returns an empty DataFrame if no K-DID weight metadata is present.
        """
        columns = ("lead", "k", "weight", "jtest_stat", "jtest_df", "jtest_pval", "dropped")
        k_weights = self.metadata.get("k_weights_by_lead")
        if k_weights is None:
            return pd.DataFrame(columns=columns)

        records: list[dict[str, Any]] = []
        for lead, info in sorted(
            k_weights.items(),
            key=lambda x: _require_weight_metadata_lead(x[0]),
        ):
            parsed_lead = _require_weight_metadata_lead(lead)
            w_k = info.get("w_k", ())
            jtest_stat = info.get("jtest_stat")
            jtest_df = info.get("jtest_df")
            jtest_pval = info.get("jtest_pval")
            dropped = info.get("dropped_moments", ())
            for idx, w in enumerate(w_k, start=1):
                records.append({
                    "lead": parsed_lead,
                    "k": idx,
                    "weight": float(w),
                    "jtest_stat": float(jtest_stat) if jtest_stat is not None else None,
                    "jtest_df": int(jtest_df) if jtest_df is not None else None,
                    "jtest_pval": float(jtest_pval) if jtest_pval is not None else None,
                    "dropped": idx in (dropped or ()),
                })
        return pd.DataFrame.from_records(records, columns=columns)


DidGmmAuditRow = DidGmmRow

__all__ = [
    "DidBootstrapDraw",
    "DidEstimateRow",
    "DidGmmAuditRow",
    "DidGmmRow",
    "DidResult",
    "DidWeightRow",
]
