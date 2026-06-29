"""Performance benchmarks for diddesign.

Measures execution time for key operations to detect regressions.
Run with: pytest benchmarks/test_performance.py -v -s --tb=short
"""

import time

import pytest
from diddesign.data import load_malesky2014, load_paglayan2019
from diddesign import did, did_check


# ---------------------------------------------------------------------------
# Estimation timing benchmarks
# ---------------------------------------------------------------------------
class TestEstimationPerformance:
    """Timing benchmarks for estimation functions."""

    def test_rcs_estimation_time(self):
        """RCS estimation with n_boot=50 should complete in < 30s."""
        data = load_malesky2014()
        start = time.perf_counter()
        result = did(
            data,
            outcome="pro4",
            treatment="treatment",
            time="year",
            post="post_treat",
            data_type="rcs",
            id_cluster="id_district",
            n_boot=50,
            random_seed=42,
        )
        elapsed = time.perf_counter() - start
        assert elapsed < 30.0, f"RCS estimation took {elapsed:.1f}s (limit: 30s)"
        assert result.estimates, "No estimates returned"
        print(f"\n  RCS n_boot=50: {elapsed:.2f}s")

    def test_sa_estimation_time(self):
        """SA estimation with n_boot=50 should complete in < 60s."""
        data = load_paglayan2019()
        start = time.perf_counter()
        result = did(
            data,
            outcome="pupil_expenditure",
            treatment="treatment",
            time="year",
            unit_id="state",
            design="sa",
            thres=1,
            n_boot=50,
            random_seed=42,
        )
        elapsed = time.perf_counter() - start
        assert elapsed < 60.0, f"SA estimation took {elapsed:.1f}s (limit: 60s)"
        assert result.estimates, "No estimates returned"
        print(f"\n  SA n_boot=50: {elapsed:.2f}s")

    def test_diagnostics_time(self):
        """Diagnostics with n_boot=30 should complete in < 20s."""
        data = load_malesky2014()
        start = time.perf_counter()
        check = did_check(
            data=data,
            outcome="pro4",
            treatment="treatment",
            time="year",
            post="post_treat",
            data_type="rcs",
            id_cluster="id_district",
            lag=1,
            n_boot=30,
            random_seed=42,
        )
        elapsed = time.perf_counter() - start
        assert elapsed < 20.0, f"Diagnostics took {elapsed:.1f}s (limit: 20s)"
        assert check.diagnostic_rows, "No diagnostic rows returned"
        print(f"\n  Diagnostics n_boot=30: {elapsed:.2f}s")

    def test_kdid_estimation_time(self):
        """K-DID (kmax=3) estimation should complete in < 90s."""
        data = load_paglayan2019()
        start = time.perf_counter()
        result = did(
            data,
            outcome="pupil_expenditure",
            treatment="treatment",
            time="year",
            unit_id="state",
            design="sa",
            thres=1,
            n_boot=30,
            random_seed=42,
            kmax=3,
        )
        elapsed = time.perf_counter() - start
        assert elapsed < 90.0, f"K-DID estimation took {elapsed:.1f}s (limit: 90s)"
        assert result.estimates, "No estimates returned"
        print(f"\n  K-DID kmax=3 n_boot=30: {elapsed:.2f}s")


# ---------------------------------------------------------------------------
# Scaling benchmarks
# ---------------------------------------------------------------------------
class TestScaling:
    """Verify that computation scales reasonably with n_boot."""

    def test_linear_scaling_nboot(self):
        """Doubling n_boot should roughly double time (within 3x)."""
        data = load_malesky2014()

        start = time.perf_counter()
        did(
            data,
            outcome="pro4",
            treatment="treatment",
            time="year",
            post="post_treat",
            data_type="rcs",
            id_cluster="id_district",
            n_boot=20,
            random_seed=42,
        )
        t1 = time.perf_counter() - start

        start = time.perf_counter()
        did(
            data,
            outcome="pro4",
            treatment="treatment",
            time="year",
            post="post_treat",
            data_type="rcs",
            id_cluster="id_district",
            n_boot=40,
            random_seed=42,
        )
        t2 = time.perf_counter() - start

        ratio = t2 / t1 if t1 > 0 else float("inf")
        assert ratio < 3.0, f"Scaling ratio {ratio:.2f} > 3.0 (expected ~2.0)"
        print(f"\n  Scaling: n_boot 20→40, time ratio = {ratio:.2f}")

    def test_sa_scaling_nboot(self):
        """SA doubling n_boot should roughly double time (within 3x)."""
        data = load_paglayan2019()

        start = time.perf_counter()
        did(
            data,
            outcome="pupil_expenditure",
            treatment="treatment",
            time="year",
            unit_id="state",
            design="sa",
            thres=1,
            n_boot=20,
            random_seed=42,
        )
        t1 = time.perf_counter() - start

        start = time.perf_counter()
        did(
            data,
            outcome="pupil_expenditure",
            treatment="treatment",
            time="year",
            unit_id="state",
            design="sa",
            thres=1,
            n_boot=40,
            random_seed=42,
        )
        t2 = time.perf_counter() - start

        ratio = t2 / t1 if t1 > 0 else float("inf")
        assert ratio < 3.0, f"SA scaling ratio {ratio:.2f} > 3.0 (expected ~2.0)"
        print(f"\n  SA Scaling: n_boot 20→40, time ratio = {ratio:.2f}")


# ---------------------------------------------------------------------------
# Result object performance
# ---------------------------------------------------------------------------
class TestResultObjectPerformance:
    """Verify that result object methods are fast."""

    def test_to_dataframe_time(self):
        """Converting result to DataFrame should be < 0.1s."""
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
            random_seed=42,
        )

        start = time.perf_counter()
        for _ in range(100):
            _ = result.to_dataframe()
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, f"100x to_dataframe() took {elapsed:.2f}s (limit: 1s)"
        print(f"\n  100x to_dataframe(): {elapsed * 1000:.1f}ms")

    def test_serialization_time(self):
        """Serialization round-trip should be < 0.5s."""
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
            random_seed=42,
        )

        start = time.perf_counter()
        for _ in range(50):
            _ = result.to_serialized_result()
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, f"50x serialization took {elapsed:.2f}s (limit: 1s)"
        print(f"\n  50x to_serialized_result(): {elapsed * 1000:.1f}ms")
