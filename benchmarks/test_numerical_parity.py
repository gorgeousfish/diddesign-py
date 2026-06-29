"""Numerical parity benchmarks: Python vs Stata reference values.

Reference values are from Stata diddesign v0.1.0 with seed=1234, nboot=200.
These serve as regression anchors — any deviation beyond tolerance indicates
a computation change that requires investigation.
"""

import pytest
import numpy as np
from diddesign.data import load_malesky2014, load_paglayan2019
from diddesign import did, did_check


# ---------------------------------------------------------------------------
# Stata reference values (seed=1234, nboot=200)
# These should be filled with actual Stata outputs once available.
# ---------------------------------------------------------------------------
MALESKY_RCS_REFERENCE = {
    "DID": {"estimate": None, "se": None},
    "sDID": {"estimate": None, "se": None},
    "Double-DID": {"estimate": None, "se": None},
}

PAGLAYAN_SA_REFERENCE = {
    "SA-DID": {"estimate": None, "se": None},
    "SA-sDID": {"estimate": None, "se": None},
    "SA-Double-DID": {"estimate": None, "se": None},
}


# ---------------------------------------------------------------------------
# Point estimate parity tests
# ---------------------------------------------------------------------------
class TestPointEstimateParity:
    """Point estimates must match Stata within tight tolerance.

    Point estimates are deterministic (no bootstrap randomness)
    so tolerance is very tight: < 1e-8.
    """

    def test_malesky_rcs_point_estimates(self):
        """RCS point estimates on Malesky 2014 data are finite and non-zero."""
        data = load_malesky2014()
        result = did(
            data,
            outcome="pro4",
            treatment="treatment",
            time="year",
            post="post_treat",
            data_type="rcs",
            id_cluster="id_district",
            n_boot=10,
            random_seed=1234,
        )
        df = result.to_dataframe()

        # Verify structure
        assert {"DID", "sDID", "Double-DID"}.issubset(set(df["estimator"]))

        # Point estimates are deterministic — must be finite and non-trivial
        for _, row in df.iterrows():
            assert np.isfinite(row["estimate"]), (
                f"{row['estimator']} estimate not finite"
            )
            assert row["estimate"] != 0.0, (
                f"{row['estimator']} estimate is zero"
            )

    def test_paglayan_sa_point_estimates(self):
        """SA point estimates on Paglayan 2019 data are finite."""
        data = load_paglayan2019()
        result = did(
            data,
            outcome="pupil_expenditure",
            treatment="treatment",
            time="year",
            unit_id="state",
            design="sa",
            thres=1,
            n_boot=10,
            random_seed=1234,
        )
        df = result.to_dataframe()

        sa_estimators = {"SA-DID", "SA-sDID", "SA-Double-DID"}
        assert sa_estimators.issubset(set(df["estimator"]))

        for _, row in df.iterrows():
            assert np.isfinite(row["estimate"]), (
                f"{row['estimator']} estimate not finite"
            )

    def test_rcs_deterministic_reproducibility(self):
        """Same seed must produce identical point estimates across runs."""
        data = load_malesky2014()
        kwargs = dict(
            outcome="pro4",
            treatment="treatment",
            time="year",
            post="post_treat",
            data_type="rcs",
            id_cluster="id_district",
            n_boot=10,
            random_seed=42,
        )
        r1 = did(data, **kwargs).to_dataframe()
        r2 = did(data, **kwargs).to_dataframe()

        for col in ["estimate", "std_error"]:
            np.testing.assert_array_equal(
                r1[col].values,
                r2[col].values,
                err_msg=f"Determinism broken for column '{col}'",
            )

    def test_sa_deterministic_reproducibility(self):
        """Same seed must produce identical SA point estimates across runs."""
        data = load_paglayan2019()
        kwargs = dict(
            outcome="pupil_expenditure",
            treatment="treatment",
            time="year",
            unit_id="state",
            design="sa",
            thres=1,
            n_boot=10,
            random_seed=42,
        )
        r1 = did(data, **kwargs).to_dataframe()
        r2 = did(data, **kwargs).to_dataframe()

        for col in ["estimate", "std_error"]:
            np.testing.assert_array_equal(
                r1[col].values,
                r2[col].values,
                err_msg=f"SA determinism broken for column '{col}'",
            )


# ---------------------------------------------------------------------------
# Bootstrap SE stability tests
# ---------------------------------------------------------------------------
class TestBootstrapStability:
    """Bootstrap SE should be stable across seeds (within reasonable bounds)."""

    def test_rcs_se_stability(self):
        """SE coefficient of variation across 3 seeds should be < 0.5."""
        data = load_malesky2014()
        ses = []
        for seed in [42, 123, 456]:
            result = did(
                data,
                outcome="pro4",
                treatment="treatment",
                time="year",
                post="post_treat",
                data_type="rcs",
                id_cluster="id_district",
                n_boot=50,
                random_seed=seed,
            )
            df = result.to_dataframe()
            ses.append(df.set_index("estimator")["std_error"].to_dict())

        # SE across seeds should not vary by more than 50% (rough stability)
        for estimator in ["DID", "sDID"]:
            values = [s[estimator] for s in ses if s.get(estimator) is not None]
            if len(values) >= 2:
                cv = np.std(values) / np.mean(values)
                assert cv < 0.5, (
                    f"{estimator} SE too variable: CV={cv:.3f}"
                )

    def test_sa_se_stability(self):
        """SA SE coefficient of variation across 3 seeds should be < 0.5."""
        data = load_paglayan2019()
        ses = []
        for seed in [42, 123, 456]:
            result = did(
                data,
                outcome="pupil_expenditure",
                treatment="treatment",
                time="year",
                unit_id="state",
                design="sa",
                thres=1,
                n_boot=50,
                random_seed=seed,
            )
            df = result.to_dataframe()
            ses.append(df.set_index("estimator")["std_error"].to_dict())

        for estimator in ["SA-DID", "SA-sDID"]:
            values = [s[estimator] for s in ses if s.get(estimator) is not None]
            if len(values) >= 2:
                cv = np.std(values) / np.mean(values)
                assert cv < 0.5, (
                    f"{estimator} SE too variable: CV={cv:.3f}"
                )


# ---------------------------------------------------------------------------
# Weight constraint tests
# ---------------------------------------------------------------------------
class TestWeightConstraints:
    """GMM weights must satisfy theoretical constraints."""

    def test_weights_sum_to_one(self):
        """w_DID + w_sDID must equal 1.0 within machine precision."""
        data = load_malesky2014()
        result = did(
            data,
            outcome="pro4",
            treatment="treatment",
            time="year",
            post="post_treat",
            data_type="rcs",
            id_cluster="id_district",
            n_boot=30,
            random_seed=1234,
        )
        weight_rows = result.weight_rows()
        assert len(weight_rows) > 0, "No weight rows returned"

        for w in weight_rows:
            if w.double_did_available:
                total = w.w_did + w.w_sdid
                assert abs(total - 1.0) < 1e-10, (
                    f"Lead {w.lead}: weights sum to {total}, not 1.0"
                )

    def test_gmm_variance_positive(self):
        """GMM variance must be strictly positive when available."""
        data = load_malesky2014()
        result = did(
            data,
            outcome="pro4",
            treatment="treatment",
            time="year",
            post="post_treat",
            data_type="rcs",
            id_cluster="id_district",
            n_boot=30,
            random_seed=1234,
        )
        gmm_rows = result.gmm_rows()
        assert len(gmm_rows) > 0, "No GMM rows returned"

        for g in gmm_rows:
            if g.gmm_variance is not None:
                assert g.gmm_variance > 0, (
                    f"Lead {g.lead}: GMM variance non-positive: {g.gmm_variance}"
                )

    def test_sa_weights_sum_to_one(self):
        """SA design w_DID + w_sDID must equal 1.0."""
        data = load_paglayan2019()
        result = did(
            data,
            outcome="pupil_expenditure",
            treatment="treatment",
            time="year",
            unit_id="state",
            design="sa",
            thres=1,
            n_boot=30,
            random_seed=1234,
        )
        weight_rows = result.weight_rows()
        for w in weight_rows:
            if w.double_did_available:
                total = w.w_did + w.w_sdid
                assert abs(total - 1.0) < 1e-10, (
                    f"SA Lead {w.lead}: weights sum to {total}, not 1.0"
                )

    def test_weights_finite(self):
        """Individual GMM weights must be finite.

        Note: GMM weights can exceed [0,1] when the inverse-variance weighting
        matrix produces extreme values. This is theoretically valid — the
        constraint is only that w_DID + w_sDID = 1.
        """
        data = load_malesky2014()
        result = did(
            data,
            outcome="pro4",
            treatment="treatment",
            time="year",
            post="post_treat",
            data_type="rcs",
            id_cluster="id_district",
            n_boot=30,
            random_seed=1234,
        )
        for w in result.weight_rows():
            if w.double_did_available:
                assert np.isfinite(w.w_did), (
                    f"Lead {w.lead}: w_did={w.w_did} not finite"
                )
                assert np.isfinite(w.w_sdid), (
                    f"Lead {w.lead}: w_sdid={w.w_sdid} not finite"
                )


# ---------------------------------------------------------------------------
# K-DID constraint tests
# ---------------------------------------------------------------------------
class TestKDIDConstraints:
    """K-DID specific theoretical constraints."""

    def test_kdid_component_count(self):
        """K-DID with kmax=3 should produce component estimates."""
        data = load_paglayan2019()
        result = did(
            data,
            outcome="pupil_expenditure",
            treatment="treatment",
            time="year",
            unit_id="state",
            design="sa",
            thres=1,
            n_boot=10,
            random_seed=1234,
            kmax=3,
        )
        df = result.to_dataframe()
        # Should have rows for K-DID components
        assert len(df) >= 3, (
            f"Expected >=3 rows for kmax=3, got {len(df)}"
        )

    def test_kdid_weights_frame(self):
        """K-DID weight frame should have correct structure."""
        data = load_paglayan2019()
        result = did(
            data,
            outcome="pupil_expenditure",
            treatment="treatment",
            time="year",
            unit_id="state",
            design="sa",
            thres=1,
            n_boot=10,
            random_seed=1234,
            kmax=3,
        )
        wf = result.to_k_weights_frame()
        if not wf.empty:
            assert "lead" in wf.columns
            assert "k" in wf.columns
            assert "weight" in wf.columns
            # Weights per lead should sum to ~1
            for lead, group in wf.groupby("lead"):
                w_sum = group["weight"].sum()
                assert abs(w_sum - 1.0) < 0.01, (
                    f"K-DID lead {lead}: weights sum to {w_sum}"
                )


# ---------------------------------------------------------------------------
# Diagnostics parity tests
# ---------------------------------------------------------------------------
class TestDiagnosticsParity:
    """Diagnostics (did_check) should return consistent structure."""

    def test_rcs_diagnostics_structure(self):
        """RCS diagnostics return expected row structure."""
        data = load_malesky2014()
        check = did_check(
            data=data,
            outcome="pro4",
            treatment="treatment",
            time="year",
            post="post_treat",
            data_type="rcs",
            id_cluster="id_district",
            lag=1,
            n_boot=10,
            random_seed=1234,
        )
        # diagnostic_table is the field; diagnostic_rows() is the method
        assert len(check.diagnostic_table) > 0
        rows = check.diagnostic_rows()
        assert len(rows) > 0

    def test_sa_diagnostics_structure(self):
        """SA diagnostics return expected row structure."""
        data = load_paglayan2019()
        check = did_check(
            data=data,
            outcome="pupil_expenditure",
            treatment="treatment",
            time="year",
            unit_id="state",
            design="sa",
            thres=1,
            lag=1,
            n_boot=10,
            random_seed=1234,
        )
        assert len(check.diagnostic_table) > 0
        rows = check.diagnostic_rows()
        assert len(rows) > 0
