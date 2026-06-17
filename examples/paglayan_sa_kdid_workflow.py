"""SA-K-DID (kmax=3) workflow on Paglayan 2019 staggered-adoption data.

Demonstrates the K-DID generalization in staggered-adoption design,
which uses k-th order differencing (k=1,2,...,K) with GMM-optimal
weights and over-identification J-test.

Run from the repository root:

    python diddesign-py/examples/paglayan_sa_kdid_workflow.py
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

from diddesign import did  # noqa: E402


def _float(value: object) -> float:
    return float(value)


def _lead_key(value: object) -> str:
    return str(int(value))


def main() -> None:
    data = pd.read_stata(DATA_PATH)
    data["log_expenditure"] = np.log(data["pupil_expenditure"] + 1.0)

    # ---- SA-K-DID: kmax=3, jtest=True ----
    result = did(
        data,
        outcome="log_expenditure",
        treatment="treatment",
        time="year",
        unit_id="state",
        design="sa",
        kmax=3,
        jtest=True,
        lead=(0, 1, 2),
        thres=1,
        n_boot=10,
        random_seed=2024,
    )

    estimates = result.to_estimates_frame()
    k_weights_frame = result.to_k_weights_frame()

    # ---- Component estimates (SA-DID, SA-sDID, SA-kDID-3) ----
    component_labels = {"SA-DID", "SA-sDID", "SA-kDID-3"}
    component_rows = estimates.loc[estimates["estimator"].isin(component_labels)]
    component_estimates: dict[str, dict[str, float]] = {}
    for row in component_rows.itertuples(index=False):
        component_estimates.setdefault(_lead_key(row.lead), {})[str(row.estimator)] = _float(
            row.estimate
        )

    # ---- GMM combined estimate (SA-K-DID) ----
    sa_kdid_rows = estimates.loc[estimates["estimator"] == "SA-K-DID"]
    sa_kdid_estimates: dict[str, float] = {
        _lead_key(row.lead): _float(row.estimate)
        for row in sa_kdid_rows.itertuples(index=False)
    }

    # ---- K-dimensional weight vector per lead ----
    k_weights_by_lead: dict[str, dict[str, object]] = {}
    for lead_val in sorted(k_weights_frame["lead"].unique()):
        sub = k_weights_frame.loc[k_weights_frame["lead"] == lead_val]
        weights_vec = [_float(w) for w in sub["weight"]]
        lead_info: dict[str, object] = {"w_k": weights_vec}
        # J-test info (same for all k within a lead)
        jtest_stat = sub["jtest_stat"].iloc[0]
        jtest_pval = sub["jtest_pval"].iloc[0]
        jtest_df = sub["jtest_df"].iloc[0]
        if jtest_stat is not None and not (isinstance(jtest_stat, float) and np.isnan(jtest_stat)):
            lead_info["jtest_stat"] = _float(jtest_stat)
            lead_info["jtest_pval"] = _float(jtest_pval)
            lead_info["jtest_df"] = int(jtest_df)
        k_weights_by_lead[_lead_key(lead_val)] = lead_info

    # ---- Recomposition consistency: |SA-K-DID - Σ w_k·component_k| < ε ----
    recomposition_errors: list[float] = []
    for lead_str, lead_components in component_estimates.items():
        if lead_str not in sa_kdid_estimates or lead_str not in k_weights_by_lead:
            continue
        w_k = k_weights_by_lead[lead_str]["w_k"]
        # Components ordered: k=1 (SA-DID), k=2 (SA-sDID), k=3 (SA-kDID-3)
        ordered_labels = ["SA-DID", "SA-sDID", "SA-kDID-3"]
        recomposed = sum(
            w * lead_components.get(lab, 0.0)
            for w, lab in zip(w_k, ordered_labels)
        )
        recomposition_errors.append(abs(recomposed - sa_kdid_estimates[lead_str]))

    print(
        json.dumps(
            {
                "rows": int(len(data)),
                "states": int(data["state"].nunique()),
                "requested_leads": [0, 1, 2],
                "kmax": 3,
                "jtest": True,
                "sa_kdid_estimates": sa_kdid_estimates,
                "component_estimates": component_estimates,
                "k_weights_by_lead": k_weights_by_lead,
                "max_recomposition_abs_error": max(recomposition_errors)
                if recomposition_errors
                else None,
                "estimate_rows": int(len(estimates)),
                "k_weight_rows": int(len(k_weights_frame)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
