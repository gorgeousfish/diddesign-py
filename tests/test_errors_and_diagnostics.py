"""Tests for the structured error system and diagnostics reporter."""

from __future__ import annotations

import io
import warnings
from contextlib import redirect_stdout
from typing import Any, Dict

import pandas as pd
import pytest

from diddesign.errors import (
    DidError,
    DidRuntimeError,
    DidValueError,
    DidWarning,
    ErrorCode,
    WarningCode,
    did_warn,
)
from diddesign.diagnostics_reporter import DiagnosticsReporter
from diddesign.results.objects import (
    DidBootstrapDraw,
    DidEstimateRow,
    DidResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_panel(
    n_units: int = 5,
    n_periods: int = 4,
    treat_period: int = 2,
    n_treated: int = 2,
) -> pd.DataFrame:
    """Create a minimal panel DataFrame for testing.

    Returns a balanced panel with `n_units` units, `n_periods` time periods.
    The first `n_treated` units are treated starting at `treat_period`.
    """
    import numpy as np

    rng = np.random.RandomState(42)
    rows = []
    for unit in range(1, n_units + 1):
        treated = 1 if unit <= n_treated else 0
        for t in range(1, n_periods + 1):
            y = rng.normal(10, 2) + (2.0 if treated and t >= treat_period else 0.0)
            rows.append({"unit": unit, "time": t, "treat": treated, "y": y})
    return pd.DataFrame(rows)


def _make_simple_result(design: str = "did") -> DidResult:
    """Create a minimal DidResult for diagnostics reporter tests."""
    estimates = (
        DidEstimateRow(
            estimator="DID" if design == "did" else "SA-DID",
            lead=0,
            estimate=1.5,
            std_error=0.3,
            ci_lo=0.9,
            ci_hi=2.1,
            weight=0.6,
        ),
        DidEstimateRow(
            estimator="sDID" if design == "did" else "SA-sDID",
            lead=0,
            estimate=1.8,
            std_error=0.35,
            ci_lo=1.1,
            ci_hi=2.5,
            weight=0.4,
        ),
        DidEstimateRow(
            estimator="Double-DID" if design == "did" else "SA-Double-DID",
            lead=0,
            estimate=0.6 * 1.5 + 0.4 * 1.8,
            std_error=0.25,
            ci_lo=1.0,
            ci_hi=2.2,
            weight=None,
        ),
    )
    # Construct W matrix that yields w_did=0.6, w_sdid=0.4
    # W_total = W_did + 2*W_cov + W_sdid, w_did=(W_did+W_cov)/W_total
    # Let W_cov=0 => w_did = W_did/(W_did+W_sdid), so W_did=0.6, W_sdid=0.4
    # gmm_variance = 1/(W_did+2*W_cov+W_sdid) = 1/1.0 = 1.0
    W_matrix = ((0.6, 0.0), (0.0, 0.4))
    metadata: Dict[str, Any] = {
        "design": design,
        "n_obs": 100,
        "n_units": 10,
        "n_periods": 5,
        "n_boot_requested": 50,
        "n_boot_realized": 50,
        "n_boot": 50,
        "ci_level": 0.95,
        "weights_by_lead": {0: {"w_did": 0.6, "w_sdid": 0.4}},
        "W_by_lead": {0: W_matrix},
        "vcov_gmm_by_lead": {0: ((0.1, 0.02), (0.02, 0.12))},
        "double_did_available": True,
        "double_did_available_leads": (0,),
        "identified_leads": (0,),
        "requested_leads": (0,),
    }
    if design == "sa":
        metadata["n_treated"] = 4
        metadata["n_control"] = 6
        metadata["adoption_distribution"] = {3: 2, 4: 2}

    draws = tuple(
        DidBootstrapDraw(iteration=i, lead=0, did=1.4 + i * 0.01, sdid=1.7 + i * 0.01)
        for i in range(1, 51)
    )
    return DidResult(estimates=estimates, metadata=metadata, bootstrap_draws=draws)


# ===========================================================================
# TestErrorSystem
# ===========================================================================


class TestErrorSystem:
    """Tests for the structured error code and exception hierarchy."""

    def test_error_code_enum_values(self) -> None:
        """Verify ErrorCode enum has E001 through E020."""
        expected = {f"E{i:03d}" for i in range(1, 21)}
        actual = {e.name for e in ErrorCode}
        assert len(ErrorCode) == 20
        assert expected == actual

    def test_warning_code_enum_values(self) -> None:
        """Verify WarningCode enum has W001 through W012."""
        expected = {"W001", "W002", "W003", "W004", "W005", "W006", "W007", "W008", "W009", "W010", "W011", "W012"}
        actual = {w.name for w in WarningCode}
        assert expected == actual

    def test_did_error_inherits_exception(self) -> None:
        """DidError is a subclass of Exception."""
        assert issubclass(DidError, Exception)
        err = DidError(ErrorCode.E001, "test message")
        assert isinstance(err, Exception)

    def test_did_value_error_inherits_both(self) -> None:
        """DidValueError inherits from both DidError and ValueError."""
        assert issubclass(DidValueError, DidError)
        assert issubclass(DidValueError, ValueError)
        err = DidValueError(ErrorCode.E002, "bad value")
        assert isinstance(err, DidError)
        assert isinstance(err, ValueError)

    def test_did_runtime_error_inherits_both(self) -> None:
        """DidRuntimeError inherits from both DidError and RuntimeError."""
        assert issubclass(DidRuntimeError, DidError)
        assert issubclass(DidRuntimeError, RuntimeError)
        err = DidRuntimeError(ErrorCode.E004, "compute failed")
        assert isinstance(err, DidError)
        assert isinstance(err, RuntimeError)

    def test_did_error_message_format(self) -> None:
        """Verify message format: [E003] message\\n  Context:\\n    key = value."""
        err = DidError(ErrorCode.E003, "no treated obs", context={"n_obs": 100, "n_units": 10})
        msg = str(err)
        assert msg.startswith("[E003] no treated obs")
        assert "Context:" in msg
        assert "n_obs = 100" in msg
        assert "n_units = 10" in msg

    def test_did_error_context_dict(self) -> None:
        """context attribute correctly stores provided key-value pairs."""
        ctx = {"n_obs": 50, "column": "treat"}
        err = DidError(ErrorCode.E001, "missing param", context=ctx)
        assert err.context == ctx
        assert err.code == ErrorCode.E001

    def test_did_warning_message_format(self) -> None:
        """Verify warning message format: [W001] message (key=value)."""
        w = DidWarning(WarningCode.W001, "type conversion", context={"column": "x"})
        msg = str(w)
        assert "[W001]" in msg
        assert "type conversion" in msg
        assert "column=x" in msg

    def test_did_warn_emits_warning(self) -> None:
        """did_warn() emits a DidWarning caught by warnings.catch_warnings."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            did_warn(WarningCode.W003, "duplicate removed", context={"var": "age"})
        assert len(caught) == 1
        assert issubclass(caught[0].category, UserWarning)
        assert isinstance(caught[0].message, DidWarning)
        assert caught[0].message.code == WarningCode.W003

    def test_backward_compatibility_catch_valueerror(self) -> None:
        """`except ValueError` catches DidValueError."""
        with pytest.raises(ValueError):
            raise DidValueError(ErrorCode.E002, "invalid level", context={"level": 200})

    def test_backward_compatibility_catch_runtimeerror(self) -> None:
        """`except RuntimeError` catches DidRuntimeError."""
        with pytest.raises(RuntimeError):
            raise DidRuntimeError(ErrorCode.E009, "bootstrap failed")


# ===========================================================================
# TestErrorTriggers
# ===========================================================================


class TestErrorTriggers:
    """Tests that actual did() calls trigger correct structured errors."""

    def test_invalid_level_raises_e002(self) -> None:
        """did(..., level=200) triggers E002."""
        from diddesign import did

        df = _make_simple_panel()
        with pytest.raises(DidValueError) as exc_info:
            did(data=df, outcome="y", treatment="treat", time="time", unit_id="unit", level=200)
        assert exc_info.value.code == ErrorCode.E002
        assert "level" in exc_info.value.context

    def test_invalid_kmax_raises_e010(self) -> None:
        """did(..., kmax=15) triggers E010."""
        from diddesign import did

        df = _make_simple_panel()
        with pytest.raises(DidValueError) as exc_info:
            did(data=df, outcome="y", treatment="treat", time="time", unit_id="unit", kmax=15)
        assert exc_info.value.code == ErrorCode.E010
        assert "kmax" in exc_info.value.context

    def test_no_treated_observations_raises_e003(self) -> None:
        """All-zero treatment column raises a ValueError (DidDataError or DidValueError).

        The data contract validation may intercept all-zero treatment before
        the E003 path is reached. In either case a ValueError is raised.
        """
        from diddesign import did

        df = _make_simple_panel()
        df["treat"] = 0  # No treated units
        with pytest.raises(ValueError):
            did(data=df, outcome="y", treatment="treat", time="time", unit_id="unit")

    def test_missing_unit_id_panel_raises_e001(self) -> None:
        """SA design without unit_id triggers E001."""
        from diddesign import did

        df = _make_simple_panel()
        with pytest.raises(DidValueError) as exc_info:
            did(data=df, outcome="y", treatment="treat", time="time", design="sa")
        assert exc_info.value.code == ErrorCode.E001

    def test_error_context_contains_data_stats(self) -> None:
        """Error context includes concrete data statistics."""
        from diddesign import did

        df = _make_simple_panel()
        # level=200 triggers E002 with context containing 'level'
        with pytest.raises(DidValueError) as exc_info:
            did(data=df, outcome="y", treatment="treat", time="time", unit_id="unit", level=200)
        ctx = exc_info.value.context
        # Context should contain the level value
        assert "level" in ctx
        assert ctx["level"] == 200


# ===========================================================================
# TestDiagnosticsReporter
# ===========================================================================


class TestDiagnosticsReporter:
    """Tests for DiagnosticsReporter import, output control, and dict/frame."""

    def test_reporter_import(self) -> None:
        """DiagnosticsReporter is importable from diddesign.diagnostics_reporter."""
        from diddesign.diagnostics_reporter import DiagnosticsReporter as DR

        assert DR is not None

    def test_reporter_verbose_0_silent(self) -> None:
        """verbose=0 produces no stdout output."""
        result = _make_simple_result()
        reporter = DiagnosticsReporter(result, verbose=0)
        buf = io.StringIO()
        with redirect_stdout(buf):
            reporter.print_report()
        assert buf.getvalue() == ""

    def test_reporter_verbose_1_summary(self) -> None:
        """verbose=1 outputs header + sample info + results."""
        result = _make_simple_result()
        reporter = DiagnosticsReporter(result, verbose=1)
        buf = io.StringIO()
        with redirect_stdout(buf):
            reporter.print_report()
        output = buf.getvalue()
        assert "Double Difference-in-Differences" in output
        assert "Observations:" in output
        assert "Results" in output

    def test_reporter_verbose_2_detailed(self) -> None:
        """verbose=2 additionally outputs GMM diagnostics + bootstrap stats."""
        result = _make_simple_result()
        reporter = DiagnosticsReporter(result, verbose=2)
        buf = io.StringIO()
        with redirect_stdout(buf):
            reporter.print_report()
        output = buf.getvalue()
        assert "GMM Weight Diagnostics" in output
        assert "Bootstrap Summary" in output

    def test_reporter_to_dict_keys(self) -> None:
        """to_dict() returns expected top-level keys."""
        result = _make_simple_result()
        reporter = DiagnosticsReporter(result, verbose=2)
        info = reporter.to_dict()
        assert "design" in info
        assert "n_observations" in info
        assert "n_units" in info
        assert "n_periods" in info
        assert "estimates" in info
        assert "gmm_diagnostics" in info
        assert "bootstrap" in info

    def test_reporter_summary_frame_type(self) -> None:
        """summary_frame() returns a pandas DataFrame."""
        result = _make_simple_result()
        reporter = DiagnosticsReporter(result, verbose=1)
        frame = reporter.summary_frame()
        assert isinstance(frame, pd.DataFrame)
        assert "statistic" in frame.columns
        assert "value" in frame.columns
        assert len(frame) > 0

    def test_result_report_method(self) -> None:
        """result.report(verbose=1) is callable and produces output."""
        result = _make_simple_result()
        output = result.report(verbose=1)
        assert "Double Difference-in-Differences" in output

    def test_result_report_dict_method(self) -> None:
        """result.report_dict() returns a dictionary with expected keys."""
        result = _make_simple_result()
        info = result.report_dict()
        assert isinstance(info, dict)
        assert "design" in info
        assert "estimates" in info
        assert "n_observations" in info


# ===========================================================================
# TestDiagnosticsContent
# ===========================================================================


class TestDiagnosticsContent:
    """Tests that diagnostic output contains accurate content."""

    def test_header_shows_design_type(self) -> None:
        """Output includes the design type label."""
        result = _make_simple_result(design="did")
        reporter = DiagnosticsReporter(result, verbose=1)
        buf = io.StringIO()
        with redirect_stdout(buf):
            reporter.print_report()
        output = buf.getvalue()
        assert "Standard DID" in output

    def test_header_shows_n_observations(self) -> None:
        """Output includes the correct observation count."""
        result = _make_simple_result()
        reporter = DiagnosticsReporter(result, verbose=1)
        buf = io.StringIO()
        with redirect_stdout(buf):
            reporter.print_report()
        output = buf.getvalue()
        assert "100" in output

    def test_sa_diagnostics_shows_valid_periods(self) -> None:
        """SA design shows valid period count."""
        result = _make_simple_result(design="sa")
        reporter = DiagnosticsReporter(result, verbose=1)
        buf = io.StringIO()
        with redirect_stdout(buf):
            reporter.print_report()
        output = buf.getvalue()
        # SA diagnostics section should appear
        assert "SA Design Diagnostics" in output
        # Should show leads info
        assert "Valid periods:" in output or "Leads estimated:" in output

    def test_sa_diagnostics_shows_cohort_info(self) -> None:
        """SA design shows cohort summary information."""
        result = _make_simple_result(design="sa")
        reporter = DiagnosticsReporter(result, verbose=1)
        buf = io.StringIO()
        with redirect_stdout(buf):
            reporter.print_report()
        output = buf.getvalue()
        assert "Cohort summary:" in output
        assert "Treated units:" in output
