"""Repeated-cross-section DID workflow used by the manuscript examples.

Run from the repository root:

    python diddesign-py/examples/malesky_rcs_workflow.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "diddesign-stata" / "examples" / "data" / "malesky2014.dta"
sys.path.insert(0, str(REPO_ROOT / "diddesign-py" / "src"))

from diddesign import did, did_check, fit  # noqa: E402


def _float(value: object) -> float:
    return float(value)


def _optional_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _fit_rows(rows_frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in rows_frame.itertuples(index=False):
        rows.append(
            {
                "source": str(row.source),
                "estimator": None if pd.isna(row.estimator) else str(row.estimator),
                "lag": _optional_float(row.lag),
                "lead": _optional_float(row.lead),
                "time_to_treat": int(row.time_to_treat),
                "estimate": _float(row.estimate),
                "std_error": _float(row.std_error),
                "ci90_lb": _float(row.ci90_lb),
                "ci90_ub": _float(row.ci90_ub),
            }
        )
    return rows


def main() -> None:
    data = pd.read_stata(DATA_PATH)
    result = did(
        data,
        outcome="pro4",
        treatment="treatment",
        time="year",
        post="post_treat",
        data_type="rcs",
        covariates=["lnarea"],
        n_boot=20,
        random_seed=1234,
    )
    check_result = did_check(
        data=data,
        outcome="pro4",
        treatment="treatment",
        time="year",
        post="post_treat",
        data_type="rcs",
        covariates=["lnarea"],
        n_boot=20,
        random_seed=1234,
        lag=1,
    )

    estimates = result.to_estimates_frame()
    weights = result.to_weights_frame()
    diagnostics = check_result.to_summary_frame()
    fit_rows = fit(result, check_fit=check_result, as_frame=True)
    double_did = estimates.loc[estimates["estimator"] == "Double-DID"].iloc[0]
    component_rows = estimates.loc[estimates["estimator"].isin(["DID", "sDID"])]
    component_estimates = {
        str(row.estimator): _float(row.estimate)
        for row in component_rows.itertuples(index=False)
    }
    weight_row = weights.iloc[0]
    gmm_weights = {
        "DID": _float(weight_row["w_did"]),
        "sDID": _float(weight_row["w_sdid"]),
    }
    recomposed = (
        gmm_weights["DID"] * component_estimates["DID"]
        + gmm_weights["sDID"] * component_estimates["sDID"]
    )
    diagnostic = diagnostics.iloc[0]

    print(
        json.dumps(
            {
                "rows": int(len(data)),
                "years": sorted(int(year) for year in data["year"].dropna().unique()),
                "double_did_estimate": float(double_did["estimate"]),
                "estimate_rows": int(len(estimates)),
                "weight_rows": int(len(weights)),
                "diagnostic_rows": int(len(diagnostics)),
                "fit_row_count": int(len(fit_rows)),
                "component_estimates": component_estimates,
                "gmm_weights": gmm_weights,
                "recomposition_abs_error": abs(recomposed - _float(double_did["estimate"])),
                "diagnostic_row": {
                    "lag": int(diagnostic["lag"]),
                    "estimate_raw": _float(diagnostic["estimate_raw"]),
                    "std_error_raw": _float(diagnostic["std_error_raw"]),
                    "eqci95_lb_std": _float(diagnostic["eqci95_lb_std"]),
                    "eqci95_ub_std": _float(diagnostic["eqci95_ub_std"]),
                },
                "fit_rows": _fit_rows(fit_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
