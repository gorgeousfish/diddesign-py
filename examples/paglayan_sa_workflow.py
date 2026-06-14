"""Staggered-adoption DID workflow used by the manuscript examples.

Run from the repository root:

    python diddesign-py/examples/paglayan_sa_workflow.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "diddesign-stata" / "examples" / "data" / "paglayan2019.dta"
sys.path.insert(0, str(REPO_ROOT / "diddesign-py" / "src"))

from diddesign import did, fit  # noqa: E402


def _float(value: object) -> float:
    return float(value)


def _lead_key(value: object) -> str:
    return str(int(value))


def _fit_rows(rows_frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in rows_frame.itertuples(index=False):
        rows.append(
            {
                "source": str(row.source),
                "estimator": str(row.estimator),
                "lead": int(row.lead),
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
    data["log_expenditure"] = np.log(data["pupil_expenditure"] + 1.0)
    result = did(
        data,
        outcome="log_expenditure",
        treatment="treatment",
        time="year",
        unit_id="state",
        design="sa",
        lead=(0, 1, 2),
        thres=1,
        n_boot=4,
        random_seed=1234,
    )

    estimates = result.to_estimates_frame()
    weights = result.to_weights_frame()
    fit_rows = fit(result, as_frame=True)
    double_did_rows = estimates.loc[estimates["estimator"] == "SA-Double-DID"]
    component_rows = estimates.loc[estimates["estimator"].isin(["SA-DID", "SA-sDID"])]
    component_estimates: dict[str, dict[str, float]] = {}
    for row in component_rows.itertuples(index=False):
        component_estimates.setdefault(_lead_key(row.lead), {})[str(row.estimator)] = _float(
            row.estimate
        )
    gmm_weights = {
        _lead_key(row.lead): {"SA-DID": _float(row.w_did), "SA-sDID": _float(row.w_sdid)}
        for row in weights.itertuples(index=False)
    }
    double_did_estimates = {
        _lead_key(row.lead): _float(row.estimate)
        for row in double_did_rows.itertuples(index=False)
    }
    recomposition_errors = []
    for lead, lead_components in component_estimates.items():
        recomposed = (
            gmm_weights[lead]["SA-DID"] * lead_components["SA-DID"]
            + gmm_weights[lead]["SA-sDID"] * lead_components["SA-sDID"]
        )
        recomposition_errors.append(abs(recomposed - double_did_estimates[lead]))

    print(
        json.dumps(
            {
                "rows": int(len(data)),
                "states": int(data["state"].nunique()),
                "requested_leads": [0, 1, 2],
                "double_did_estimates": double_did_estimates,
                "estimate_rows": int(len(estimates)),
                "weight_rows": int(len(weights)),
                "fit_row_count": int(len(fit_rows)),
                "component_estimates": component_estimates,
                "gmm_weights": gmm_weights,
                "max_recomposition_abs_error": max(recomposition_errors),
                "fit_rows": _fit_rows(fit_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
