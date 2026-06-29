import math
from statistics import NormalDist

import numpy as np
import pytest


_Z_90 = NormalDist().inv_cdf(0.95)


def _eqci_bounds(estimate_std: float, std_error_std: float) -> tuple[float, float]:
    radius = max(
        abs(estimate_std - _Z_90 * std_error_std),
        abs(estimate_std + _Z_90 * std_error_std),
    )
    return (-radius, radius)


def _small_panel_check_rows():
    return [
        {"unit": "a", "time": 0, "treat": 0, "outcome": 10.0},
        {"unit": "a", "time": 1, "treat": 0, "outcome": 11.0},
        {"unit": "a", "time": 2, "treat": 1, "outcome": 14.0},
        {"unit": "b", "time": 0, "treat": 0, "outcome": 12.0},
        {"unit": "b", "time": 1, "treat": 0, "outcome": 13.0},
        {"unit": "b", "time": 2, "treat": 1, "outcome": 16.0},
        {"unit": "c", "time": 0, "treat": 0, "outcome": 8.0},
        {"unit": "c", "time": 1, "treat": 0, "outcome": 9.0},
        {"unit": "c", "time": 2, "treat": 0, "outcome": 10.0},
        {"unit": "d", "time": 0, "treat": 0, "outcome": 9.0},
        {"unit": "d", "time": 1, "treat": 0, "outcome": 10.0},
        {"unit": "d", "time": 2, "treat": 0, "outcome": 11.0},
    ]


def _small_rcs_check_rows():
    return [
        {"cluster": "north", "time": 0, "treat_group": 0, "post": 0, "outcome": 1.0},
        {"cluster": "north", "time": 0, "treat_group": 1, "post": 0, "outcome": 2.0},
        {"cluster": "south", "time": 0, "treat_group": 0, "post": 0, "outcome": 1.2},
        {"cluster": "south", "time": 0, "treat_group": 1, "post": 0, "outcome": 2.2},
        {"cluster": "north", "time": 1, "treat_group": 0, "post": 0, "outcome": 2.0},
        {"cluster": "north", "time": 1, "treat_group": 1, "post": 0, "outcome": 3.0},
        {"cluster": "south", "time": 1, "treat_group": 0, "post": 0, "outcome": 2.2},
        {"cluster": "south", "time": 1, "treat_group": 1, "post": 0, "outcome": 3.2},
        {"cluster": "north", "time": 2, "treat_group": 0, "post": 1, "outcome": 3.0},
        {"cluster": "north", "time": 2, "treat_group": 1, "post": 1, "outcome": 5.5},
        {"cluster": "south", "time": 2, "treat_group": 0, "post": 1, "outcome": 3.2},
        {"cluster": "south", "time": 2, "treat_group": 1, "post": 1, "outcome": 5.7},
    ]


def test_did_check_summary_keeps_raw_headline_and_standardized_eqci():
    from diddesign.diagnostics import DidCheckDiagnosticRow, did_check

    eqci95_lb_std, eqci95_ub_std = _eqci_bounds(estimate_std=0.25, std_error_std=0.1)

    result = did_check(
        diagnostic_rows=[
            DidCheckDiagnosticRow(
                lag=1,
                estimate_std=0.25,
                std_error_std=0.1,
                estimate_raw=1.5,
                std_error_raw=0.4,
                eqci95_lb_std=eqci95_lb_std,
                eqci95_ub_std=eqci95_ub_std,
            )
        ],
        trends_rows=[],
        metadata={"branch": "did-panel"},
    )

    assert result.metadata["branch"] == "did-panel"
    assert result.summary_rows() == (
        {
            "lag": 1,
            "estimate_raw": 1.5,
            "std_error_raw": 0.4,
            "eqci95_lb_std": eqci95_lb_std,
            "eqci95_ub_std": eqci95_ub_std,
        },
    )


def test_did_check_freezes_nested_metadata_and_keeps_payload_rows_detached():
    from diddesign.diagnostics import DidCheckDiagnosticRow, did_check

    eqci95_lb_std, eqci95_ub_std = _eqci_bounds(estimate_std=0.25, std_error_std=0.1)
    source_matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    source_metadata = {
        "branch": "did-panel",
        "numpy_flag": np.bool_(True),
        "numpy_count": np.int64(4),
        "numpy_scale": np.float64(1.25),
        "audit": {
            "lags": [1],
            "matrix": source_matrix,
            "source": {"name": "manual-check", "enabled": np.bool_(False)},
        },
    }

    result = did_check(
        diagnostic_rows=[
            DidCheckDiagnosticRow(
                lag=1,
                estimate_std=0.25,
                std_error_std=0.1,
                estimate_raw=1.5,
                std_error_raw=0.4,
                eqci95_lb_std=eqci95_lb_std,
                eqci95_ub_std=eqci95_ub_std,
            )
        ],
        trends_rows=[],
        metadata=source_metadata,
    )

    source_metadata["audit"]["lags"].append(2)
    source_metadata["audit"]["source"]["name"] = "mutated"
    source_matrix[0, 0] = 99.0

    assert result.metadata["audit"]["lags"] == (1,)
    assert result.metadata["audit"]["matrix"] == ((1.0, 2.0), (3.0, 4.0))
    assert result.metadata["audit"]["source"]["name"] == "manual-check"
    assert result.metadata["numpy_flag"] is True
    assert result.metadata["numpy_count"] == 4
    assert isinstance(result.metadata["numpy_count"], int)
    assert result.metadata["numpy_scale"] == pytest.approx(1.25)
    assert isinstance(result.metadata["numpy_scale"], float)
    assert result.metadata["audit"]["source"]["enabled"] is False
    with pytest.raises(TypeError):
        result.metadata["audit"]["source"]["name"] = "other"

    summary_rows = result.summary_rows()
    summary_rows[0]["estimate_raw"] = -999.0

    assert result.summary_rows()[0]["estimate_raw"] == pytest.approx(1.5)


def test_did_check_rejects_non_finite_nested_metadata_after_numpy_array_freeze():
    from diddesign.diagnostics import DidCheckDiagnosticRow, did_check

    eqci95_lb_std, eqci95_ub_std = _eqci_bounds(estimate_std=0.25, std_error_std=0.1)
    diagnostic_rows = [
        DidCheckDiagnosticRow(
            lag=1,
            estimate_std=0.25,
            std_error_std=0.1,
            estimate_raw=1.5,
            std_error_raw=0.4,
            eqci95_lb_std=eqci95_lb_std,
            eqci95_ub_std=eqci95_ub_std,
        )
    ]

    with pytest.raises(ValueError, match="metadata value must be finite\\."):
        did_check(
            diagnostic_rows=diagnostic_rows,
            trends_rows=[],
            metadata={"audit": {"matrix": np.array([[1.0, np.inf]])}},
        )

    with pytest.raises(ValueError, match="metadata value must be finite\\."):
        did_check(
            diagnostic_rows=diagnostic_rows,
            trends_rows=[],
            metadata={"audit": {"values": [1.0, float("nan")]}},
        )


def test_did_check_result_direct_constructor_freezes_metadata_and_rejects_bad_rows():
    from diddesign.diagnostics import DidCheckDiagnosticRow, DidCheckPatternRow, DidCheckResult, DidCheckTrendRow

    eqci95_lb_std, eqci95_ub_std = _eqci_bounds(estimate_std=0.25, std_error_std=0.1)
    diagnostic_row = DidCheckDiagnosticRow(
        lag=1,
        estimate_std=0.25,
        std_error_std=0.1,
        estimate_raw=1.5,
        std_error_raw=0.4,
        eqci95_lb_std=eqci95_lb_std,
        eqci95_ub_std=eqci95_ub_std,
    )
    trend_row = DidCheckTrendRow(
        time_to_treat=-1,
        group="Control",
        outcome_mean=2.0,
        outcome_sd=0.3,
        n_obs=12,
    )
    source_metadata = {"audit": {"lags": [1]}}

    result = DidCheckResult(
        diagnostic_table=(diagnostic_row,),
        trends_table=(trend_row,),
        metadata=source_metadata,
    )
    source_metadata["audit"]["lags"].append(2)

    assert result.metadata["audit"]["lags"] == (1,)
    with pytest.raises(TypeError):
        result.metadata["audit"]["lags"] = (3,)

    with pytest.raises(TypeError, match="diagnostic_table must contain DidCheckDiagnosticRow instances\\."):
        DidCheckResult(
            diagnostic_table=({"lag": 1},),
            trends_table=(trend_row,),
            metadata={},
        )

    with pytest.raises(TypeError, match="trends_table must contain DidCheckTrendRow instances\\."):
        DidCheckResult(
            diagnostic_table=(diagnostic_row,),
            trends_table=({"time_to_treat": -1},),
            metadata={},
        )

    with pytest.raises(TypeError, match="pattern_table must contain DidCheckPatternRow instances\\."):
        DidCheckResult(
            diagnostic_table=(diagnostic_row,),
            trends_table=(),
            pattern_table=({"id_time": 1},),
            metadata={},
        )

    with pytest.raises(ValueError, match="DidCheckResult cannot mix trend and pattern plotting rows\\."):
        DidCheckResult(
            diagnostic_table=(diagnostic_row,),
            trends_table=(trend_row,),
            pattern_table=(
                DidCheckPatternRow(
                    id_time=1,
                    unit_order=1,
                    status="treated",
                ),
            ),
            metadata={},
        )

    with pytest.raises(
        ValueError,
        match="pattern_table requires design metadata to be 'sa'\\.",
    ):
        DidCheckResult(
            diagnostic_table=(diagnostic_row,),
            trends_table=(),
            pattern_table=(
                DidCheckPatternRow(
                    id_time=1,
                    unit_order=1,
                    status="treated",
                ),
            ),
            metadata={"design": "did"},
        )

    with pytest.raises(
        ValueError,
        match="pattern_table requires design metadata to be 'sa'\\.",
    ):
        DidCheckResult(
            diagnostic_table=(diagnostic_row,),
            trends_table=(),
            pattern_table=(
                DidCheckPatternRow(
                    id_time=1,
                    unit_order=1,
                    status="treated",
                ),
            ),
            metadata={},
        )


def test_did_check_result_rejects_duplicate_public_payload_coordinates():
    from diddesign.diagnostics import DidCheckDiagnosticRow, DidCheckPatternRow, DidCheckResult, DidCheckTrendRow

    eqci95_lb_std, eqci95_ub_std = _eqci_bounds(estimate_std=0.25, std_error_std=0.1)
    diagnostic_row = DidCheckDiagnosticRow(
        lag=1,
        estimate_std=0.25,
        std_error_std=0.1,
        estimate_raw=1.5,
        std_error_raw=0.4,
        eqci95_lb_std=eqci95_lb_std,
        eqci95_ub_std=eqci95_ub_std,
    )
    trend_row = DidCheckTrendRow(
        time_to_treat=-1,
        group="Control",
        outcome_mean=2.0,
        outcome_sd=0.3,
        n_obs=12,
    )
    pattern_row = DidCheckPatternRow(id_time=1, unit_order=1, status="treated")

    with pytest.raises(
        ValueError,
        match="diagnostic_table must not contain duplicate lag rows\\.",
    ):
        DidCheckResult(
            diagnostic_table=(diagnostic_row, diagnostic_row),
            trends_table=(trend_row,),
            metadata={},
        )

    with pytest.raises(
        ValueError,
        match="trends_table must not contain duplicate time/group rows\\.",
    ):
        DidCheckResult(
            diagnostic_table=(diagnostic_row,),
            trends_table=(trend_row, trend_row),
            metadata={},
        )

    with pytest.raises(
        ValueError,
        match="pattern_table must not contain duplicate time/unit rows\\.",
    ):
        DidCheckResult(
            diagnostic_table=(diagnostic_row,),
            trends_table=(),
            pattern_table=(pattern_row, pattern_row),
            metadata={"design": "sa"},
        )


def test_did_check_result_rejects_incoherent_family_metadata():
    from diddesign.diagnostics import DidCheckDiagnosticRow, DidCheckPatternRow, DidCheckResult, DidCheckTrendRow

    eqci95_lb_std, eqci95_ub_std = _eqci_bounds(estimate_std=0.25, std_error_std=0.1)
    diagnostic_row = DidCheckDiagnosticRow(
        lag=1,
        estimate_std=0.25,
        std_error_std=0.1,
        estimate_raw=1.5,
        std_error_raw=0.4,
        eqci95_lb_std=eqci95_lb_std,
        eqci95_ub_std=eqci95_ub_std,
    )
    trend_row = DidCheckTrendRow(
        time_to_treat=-1,
        group="Control",
        outcome_mean=2.0,
        outcome_sd=0.3,
        n_obs=12,
    )
    pattern_row = DidCheckPatternRow(id_time=1, unit_order=1, status="treated")

    result = DidCheckResult(
        diagnostic_table=(diagnostic_row,),
        trends_table=(),
        pattern_table=(pattern_row,),
        metadata={"design": "sa", "branch": "sa-panel-check"},
    )

    assert result.metadata["design"] == "sa"
    assert result.metadata["branch"] == "sa-panel-check"

    with pytest.raises(TypeError, match="design metadata must be a string when present\\."):
        DidCheckResult(
            diagnostic_table=(diagnostic_row,),
            trends_table=(),
            metadata={"design": 1},
        )

    with pytest.raises(ValueError, match="design metadata must be 'did' or 'sa' when present\\."):
        DidCheckResult(
            diagnostic_table=(diagnostic_row,),
            trends_table=(),
            metadata={"design": "event-study"},
        )

    with pytest.raises(TypeError, match="branch metadata must be a string when present\\."):
        DidCheckResult(
            diagnostic_table=(diagnostic_row,),
            trends_table=(),
            metadata={"branch": 1},
        )

    with pytest.raises(
        ValueError,
        match="SA diagnostic branch metadata requires design metadata to be 'sa'\\.",
    ):
        DidCheckResult(
            diagnostic_table=(diagnostic_row,),
            trends_table=(trend_row,),
            metadata={"branch": "sa-panel-check"},
        )

    with pytest.raises(ValueError, match="branch metadata must match diagnostic design\\."):
        DidCheckResult(
            diagnostic_table=(diagnostic_row,),
            trends_table=(),
            pattern_table=(pattern_row,),
            metadata={"design": "sa", "branch": "did-panel-check"},
        )

    with pytest.raises(ValueError, match="branch metadata must match diagnostic design\\."):
        DidCheckResult(
            diagnostic_table=(diagnostic_row,),
            trends_table=(trend_row,),
            metadata={"design": "did", "branch": "sa-panel-check"},
        )

    with pytest.raises(ValueError, match="SA diagnostic metadata requires pattern_table\\."):
        DidCheckResult(
            diagnostic_table=(diagnostic_row,),
            trends_table=(trend_row,),
            metadata={"design": "sa", "branch": "sa-panel-check"},
        )

    with pytest.raises(ValueError, match="SA diagnostic metadata requires pattern_table\\."):
        DidCheckResult(
            diagnostic_table=(diagnostic_row,),
            trends_table=(),
            metadata={"design": "sa", "branch": "sa-panel-check"},
        )


def test_did_check_result_rejects_incoherent_public_lag_metadata():
    from diddesign.diagnostics import DidCheckDiagnosticRow, DidCheckResult

    eqci95_lb_std, eqci95_ub_std = _eqci_bounds(estimate_std=0.25, std_error_std=0.1)
    lag_one = DidCheckDiagnosticRow(
        lag=1,
        estimate_std=0.25,
        std_error_std=0.1,
        estimate_raw=1.5,
        std_error_raw=0.4,
        eqci95_lb_std=eqci95_lb_std,
        eqci95_ub_std=eqci95_ub_std,
    )
    lag_two = DidCheckDiagnosticRow(
        lag=2,
        estimate_std=None,
        std_error_std=None,
        estimate_raw=0.8,
        std_error_raw=0.3,
        eqci95_lb_std=None,
        eqci95_ub_std=None,
    )

    result = DidCheckResult(
        diagnostic_table=(lag_one, lag_two),
        trends_table=(),
        metadata={
            "requested_lags": [1, 2, 3],
            "identified_lags": [1, 2],
            "unidentified_lags": [3],
            "raw_only_lags": [2],
        },
    )

    assert result.metadata["requested_lags"] == (1, 2, 3)
    assert result.metadata["identified_lags"] == (1, 2)
    assert result.metadata["unidentified_lags"] == (3,)
    assert result.metadata["raw_only_lags"] == (2,)

    with pytest.raises(ValueError, match="metadata\\['requested_lags'\\] must include every diagnostic lag\\."):
        DidCheckResult(
            diagnostic_table=(lag_one, lag_two),
            trends_table=(),
            metadata={"requested_lags": [1]},
        )

    with pytest.raises(ValueError, match="metadata\\['identified_lags'\\] must match diagnostic_table lag rows\\."):
        DidCheckResult(
            diagnostic_table=(lag_one, lag_two),
            trends_table=(),
            metadata={"identified_lags": [1]},
        )

    with pytest.raises(ValueError, match="metadata\\['unidentified_lags'\\] requires metadata\\['requested_lags'\\]\\."):
        DidCheckResult(
            diagnostic_table=(lag_one,),
            trends_table=(),
            metadata={"unidentified_lags": [2]},
        )

    with pytest.raises(ValueError, match="metadata\\['unidentified_lags'\\] must not include diagnostic lag rows\\."):
        DidCheckResult(
            diagnostic_table=(lag_one,),
            trends_table=(),
            metadata={"requested_lags": [1, 2], "unidentified_lags": [1]},
        )

    with pytest.raises(ValueError, match="metadata\\['raw_only_lags'\\] must be a subset of diagnostic lag rows\\."):
        DidCheckResult(
            diagnostic_table=(lag_one,),
            trends_table=(),
            metadata={"raw_only_lags": [2]},
        )

    with pytest.raises(
        ValueError,
        match="metadata\\['raw_only_lags'\\] must match diagnostic rows with missing standardized estimates\\.",
    ):
        DidCheckResult(
            diagnostic_table=(lag_one, lag_two),
            trends_table=(),
            metadata={"raw_only_lags": []},
        )

    with pytest.raises(
        ValueError,
        match="metadata\\['raw_only_lags'\\] is required when diagnostic rows have missing standardized estimates\\.",
    ):
        DidCheckResult(
            diagnostic_table=(lag_one, lag_two),
            trends_table=(),
            metadata={},
        )

    with pytest.raises(
        ValueError,
        match="metadata\\['raw_only_lags'\\] must match diagnostic rows with missing standardized estimates\\.",
    ):
        DidCheckResult(
            diagnostic_table=(lag_one,),
            trends_table=(),
            metadata={"raw_only_lags": [1]},
        )

    with pytest.raises(
        ValueError,
        match="metadata\\['requested_lags'\\] must not contain duplicate lag values\\.",
    ):
        DidCheckResult(
            diagnostic_table=(lag_one,),
            trends_table=(),
            metadata={"requested_lags": [1, 1], "identified_lags": [1]},
        )

    with pytest.raises(
        ValueError,
        match="metadata\\['requested_lags'\\] must equal identified_lags plus unidentified_lags\\.",
    ):
        DidCheckResult(
            diagnostic_table=(lag_one,),
            trends_table=(),
            metadata={"requested_lags": [1, 2, 3], "identified_lags": [1], "unidentified_lags": [2]},
        )

    with pytest.raises(ValueError, match="lag metadata requires diagnostic_table rows\\."):
        DidCheckResult(
            diagnostic_table=(),
            trends_table=(),
            metadata={"requested_lags": [1], "identified_lags": [], "unidentified_lags": [1]},
        )


def test_did_check_factory_infers_public_raw_only_lag_metadata():
    from diddesign.diagnostics import DidCheckDiagnosticRow, did_check

    result = did_check(
        diagnostic_rows=(
            DidCheckDiagnosticRow(
                lag=1,
                estimate_std=None,
                std_error_std=None,
                estimate_raw=0.8,
                std_error_raw=0.3,
                eqci95_lb_std=None,
                eqci95_ub_std=None,
            ),
        ),
        trends_rows=(),
    )

    assert result.metadata["raw_only_lags"] == (1,)
    assert result.named_plot_payloads()["placebo"] == ()


def test_did_check_factory_rejects_invalid_diagnostic_rows_before_raw_only_inference():
    from diddesign.diagnostics import did_check

    with pytest.raises(TypeError, match="diagnostic_table must contain DidCheckDiagnosticRow instances\\."):
        did_check(
            diagnostic_rows=({"lag": 1},),
            trends_rows=(),
        )


def test_did_check_result_rejects_incoherent_public_bootstrap_metadata():
    from diddesign.diagnostics import DidCheckDiagnosticRow, DidCheckResult

    eqci95_lb_std, eqci95_ub_std = _eqci_bounds(estimate_std=0.25, std_error_std=0.1)
    diagnostic_row = DidCheckDiagnosticRow(
        lag=1,
        estimate_std=0.25,
        std_error_std=0.1,
        estimate_raw=1.5,
        std_error_raw=0.4,
        eqci95_lb_std=eqci95_lb_std,
        eqci95_ub_std=eqci95_ub_std,
    )

    result = DidCheckResult(
        diagnostic_table=(diagnostic_row,),
        trends_table=(),
        metadata={"n_boot": np.int64(5), "n_boot_requested": np.int64(5), "n_boot_realized": np.int64(4)},
    )

    assert result.metadata["n_boot"] == 5
    assert isinstance(result.metadata["n_boot"], int)
    assert result.metadata["n_boot_requested"] == 5
    assert isinstance(result.metadata["n_boot_requested"], int)
    assert result.metadata["n_boot_realized"] == 4
    assert isinstance(result.metadata["n_boot_realized"], int)

    with pytest.raises(ValueError, match="n_boot_requested must be greater than or equal to n_boot_realized\\."):
        DidCheckResult(
            diagnostic_table=(diagnostic_row,),
            trends_table=(),
            metadata={"n_boot_requested": 1, "n_boot_realized": 2},
        )

    with pytest.raises(ValueError, match="n_boot_realized must be non-negative\\."):
        DidCheckResult(
            diagnostic_table=(diagnostic_row,),
            trends_table=(),
            metadata={"n_boot_realized": -1},
        )

    with pytest.raises(ValueError, match="n_boot_requested must be an integer\\."):
        DidCheckResult(
            diagnostic_table=(diagnostic_row,),
            trends_table=(),
            metadata={"n_boot_requested": True},
        )

    with pytest.raises(ValueError, match="n_boot must match n_boot_requested when both are present\\."):
        DidCheckResult(
            diagnostic_table=(diagnostic_row,),
            trends_table=(),
            metadata={"n_boot": 6, "n_boot_requested": 5, "n_boot_realized": 5},
        )

    with pytest.raises(ValueError, match="n_boot must be greater than or equal to n_boot_realized\\."):
        DidCheckResult(
            diagnostic_table=(diagnostic_row,),
            trends_table=(),
            metadata={"n_boot": 1, "n_boot_realized": 2},
        )

    with pytest.raises(
        ValueError,
        match="n_boot_realized is required when bootstrap metadata requests positive draws\\.",
    ):
        DidCheckResult(
            diagnostic_table=(diagnostic_row,),
            trends_table=(),
            metadata={"n_boot_requested": 5},
        )

    with pytest.raises(
        ValueError,
        match="n_boot_realized is required when bootstrap metadata requests positive draws\\.",
    ):
        DidCheckResult(
            diagnostic_table=(diagnostic_row,),
            trends_table=(),
            metadata={"n_boot": 5},
        )

    with pytest.raises(ValueError, match="n_boot must be an integer\\."):
        DidCheckResult(
            diagnostic_table=(diagnostic_row,),
            trends_table=(),
            metadata={"n_boot": False},
        )


def test_did_check_result_rejects_non_mapping_metadata_at_public_boundary():
    from diddesign.diagnostics import DidCheckDiagnosticRow, DidCheckResult, did_check

    eqci95_lb_std, eqci95_ub_std = _eqci_bounds(estimate_std=0.25, std_error_std=0.1)
    diagnostic_row = DidCheckDiagnosticRow(
        lag=1,
        estimate_std=0.25,
        std_error_std=0.1,
        estimate_raw=1.5,
        std_error_raw=0.4,
        eqci95_lb_std=eqci95_lb_std,
        eqci95_ub_std=eqci95_ub_std,
    )

    with pytest.raises(TypeError, match="metadata must be a mapping\\."):
        DidCheckResult(
            diagnostic_table=(diagnostic_row,),
            trends_table=(),
            metadata=(("branch", "did-panel-check"),),
        )

    with pytest.raises(TypeError, match="metadata must be a mapping\\."):
        did_check(
            diagnostic_rows=(diagnostic_row,),
            trends_rows=(),
            metadata=(("branch", "did-panel-check"),),
        )


def test_did_check_exposes_serialized_diagnostic_rows_with_dual_channels():
    from diddesign.diagnostics import DidCheckDiagnosticRow, did_check

    eqci95_lb_std, eqci95_ub_std = _eqci_bounds(estimate_std=0.25, std_error_std=0.1)

    result = did_check(
        diagnostic_rows=[
            DidCheckDiagnosticRow(
                lag=1,
                estimate_std=0.25,
                std_error_std=0.1,
                estimate_raw=1.5,
                std_error_raw=0.4,
                eqci95_lb_std=eqci95_lb_std,
                eqci95_ub_std=eqci95_ub_std,
            )
        ],
        trends_rows=[],
    )

    assert result.diagnostic_rows() == (
        {
            "lag": 1,
            "estimate_std": 0.25,
            "std_error_std": 0.1,
            "estimate_raw": 1.5,
            "std_error_raw": 0.4,
            "eqci95_lb_std": eqci95_lb_std,
            "eqci95_ub_std": eqci95_ub_std,
        },
    )


def test_did_check_result_payload_and_frames_are_detached_truth_surfaces():
    from diddesign.diagnostics import DidCheckDiagnosticRow, DidCheckTrendRow, did_check

    eqci95_lb_std, eqci95_ub_std = _eqci_bounds(estimate_std=0.25, std_error_std=0.1)
    result = did_check(
        diagnostic_rows=[
            DidCheckDiagnosticRow(
                lag=1,
                estimate_std=0.25,
                std_error_std=0.1,
                estimate_raw=1.5,
                std_error_raw=0.4,
                eqci95_lb_std=eqci95_lb_std,
                eqci95_ub_std=eqci95_ub_std,
            ),
            DidCheckDiagnosticRow(
                lag=2,
                estimate_std=None,
                std_error_std=None,
                estimate_raw=0.8,
                std_error_raw=0.3,
                eqci95_lb_std=None,
                eqci95_ub_std=None,
            ),
        ],
        trends_rows=[
            DidCheckTrendRow(
                time_to_treat=-1,
                group="Control",
                outcome_mean=2.0,
                outcome_sd=0.3,
                n_obs=12,
            )
        ],
        metadata={"branch": "did-panel-check", "audit": {"lags": [1, 2]}, "raw_only_lags": [2]},
    )

    payload = result.to_serialized_result()
    diagnostics_frame = result.to_diagnostics_frame()
    summary_frame = result.to_summary_frame()
    placebo_frame = result.to_placebo_frame()
    trends_frame = result.to_trends_frame()
    pattern_frame = result.to_pattern_frame()

    assert tuple(payload) == ("diagnostics", "summary", "plots", "metadata")
    assert payload["diagnostics"] == result.diagnostic_rows()
    assert payload["summary"] == result.summary_rows()
    assert payload["plots"] == result.named_plot_rows()
    assert payload["metadata"]["audit"]["lags"] == (1, 2)
    assert result.as_payload() == payload
    assert list(diagnostics_frame.columns) == [
        "lag",
        "estimate_std",
        "std_error_std",
        "estimate_raw",
        "std_error_raw",
        "eqci95_lb_std",
        "eqci95_ub_std",
    ]
    assert list(summary_frame.columns) == [
        "lag",
        "estimate_raw",
        "std_error_raw",
        "eqci95_lb_std",
        "eqci95_ub_std",
    ]
    assert list(placebo_frame.columns) == [
        "lag",
        "time_to_treat",
        "estimate_std",
        "std_error_std",
        "eqci95_lb_std",
        "eqci95_ub_std",
    ]
    assert list(trends_frame.columns) == [
        "time_to_treat",
        "group",
        "outcome_mean",
        "outcome_sd",
        "ci90_lb",
        "ci90_ub",
        "n_obs",
    ]
    assert list(pattern_frame.columns) == ["id_time", "unit_order", "status"]
    assert pattern_frame.empty

    payload["metadata"]["audit"]["lags"] = (99,)
    diagnostics_frame.loc[0, "estimate_raw"] = -999.0
    summary_frame.loc[0, "estimate_raw"] = -999.0
    placebo_frame.loc[0, "estimate_std"] = -999.0
    trends_frame.loc[0, "outcome_mean"] = -999.0

    assert result.metadata["audit"]["lags"] == (1, 2)
    assert result.diagnostic_rows()[0]["estimate_raw"] == pytest.approx(1.5)
    assert result.summary_rows()[0]["estimate_raw"] == pytest.approx(1.5)
    assert result.named_plot_payloads()["placebo"][0]["estimate_std"] == pytest.approx(0.25)
    assert result.named_plot_payloads()["trends"][0]["outcome_mean"] == pytest.approx(2.0)


def test_did_check_rejects_boolean_diagnostic_numeric_fields():
    from diddesign.diagnostics import DidCheckDiagnosticRow

    with pytest.raises(ValueError, match="numeric diagnostic fields must be finite numbers, not booleans\\."):
        DidCheckDiagnosticRow.from_mapping(
            {
                "lag": 1,
                "estimate_std": True,
                "std_error_std": 0.1,
                "estimate_raw": 1.5,
                "std_error_raw": 0.4,
            }
        )

    with pytest.raises(ValueError, match="numeric diagnostic fields must be finite numbers, not booleans\\."):
        DidCheckDiagnosticRow.from_mapping(
            {
                "lag": 1,
                "estimate_raw": False,
                "std_error_raw": 0.4,
            }
        )

    with pytest.raises(ValueError, match="numeric diagnostic fields must be finite numbers, not booleans\\."):
        DidCheckDiagnosticRow.from_mapping(
            {
                "lag": 1,
                "estimate_std": np.bool_(True),
                "std_error_std": 0.1,
                "estimate_raw": 1.5,
                "std_error_raw": 0.4,
            }
        )


def test_did_check_rejects_boolean_plot_payload_numeric_fields():
    from diddesign.diagnostics import DidCheckPatternRow, DidCheckTrendRow

    with pytest.raises(ValueError, match="time_to_treat must be finite\\."):
        DidCheckTrendRow(
            time_to_treat=True,
            group="Control",
            outcome_mean=2.0,
            outcome_sd=0.3,
        )

    with pytest.raises(ValueError, match="outcome_mean must be finite\\."):
        DidCheckTrendRow(
            time_to_treat=-1,
            group="Control",
            outcome_mean=False,
            outcome_sd=0.3,
        )

    with pytest.raises(ValueError, match="id_time must be finite\\."):
        DidCheckPatternRow(id_time=True, unit_order=1, status="treated")


def test_did_check_accepts_r_style_option_and_is_panel_surface_for_data_first_panel_runs():
    from diddesign.diagnostics import did_check

    data_rows = [
        {"unit": "a", "time": 0, "treat": 0, "outcome": 10.0},
        {"unit": "a", "time": 1, "treat": 0, "outcome": 11.0},
        {"unit": "a", "time": 2, "treat": 1, "outcome": 14.0},
        {"unit": "b", "time": 0, "treat": 0, "outcome": 12.0},
        {"unit": "b", "time": 1, "treat": 0, "outcome": 13.0},
        {"unit": "b", "time": 2, "treat": 1, "outcome": 16.0},
        {"unit": "c", "time": 0, "treat": 0, "outcome": 8.0},
        {"unit": "c", "time": 1, "treat": 0, "outcome": 9.0},
        {"unit": "c", "time": 2, "treat": 0, "outcome": 10.0},
        {"unit": "d", "time": 0, "treat": 0, "outcome": 9.0},
        {"unit": "d", "time": 1, "treat": 0, "outcome": 10.0},
        {"unit": "d", "time": 2, "treat": 0, "outcome": 11.0},
    ]

    result = did_check(
        data=data_rows,
        formula="outcome ~ treat",
        time="time",
        unit_id="unit",
        is_panel=True,
        option={"lag": 1, "n_boot": 5},
        random_seed=7,
    )

    assert result.metadata["data_type"] == "panel"
    assert result.metadata["cluster_column"] == "unit"
    assert result.metadata["n_boot_requested"] == 5
    assert tuple(row["lag"] for row in result.summary_rows()) == (1,)


def test_data_first_did_check_records_identified_and_unidentified_lag_metadata():
    from diddesign.diagnostics import did_check

    result = did_check(
        data=_small_panel_check_rows(),
        outcome="outcome",
        treatment="treat",
        time="time",
        unit_id="unit",
        lag=(1, 2),
        n_boot=5,
        random_seed=7,
    )

    assert tuple(row["lag"] for row in result.summary_rows()) == (1,)
    assert result.metadata["requested_lags"] == (1, 2)
    assert result.metadata["identified_lags"] == (1,)
    assert result.metadata["unidentified_lags"] == (2,)
    assert result.metadata["raw_only_lags"] == ()


def test_data_first_did_check_identifies_saturated_placebo_regressions():
    from diddesign.diagnostics import did_check

    data_rows = []
    for unit, treated, outcomes in (
        ("a", 1, (1.0, 3.0, 6.0)),
        ("b", 1, (1.2, 3.2, 6.2)),
        ("c", 0, (1.0, 2.0, 3.0)),
        ("d", 0, (1.2, 2.2, 3.2)),
    ):
        for time, outcome in enumerate(outcomes):
            data_rows.append(
                {
                    "unit": unit,
                    "time": time,
                    "treat": int(treated and time == 2),
                    "outcome": outcome,
                }
            )

    result = did_check(
        data=data_rows,
        outcome="outcome",
        treatment="treat",
        time="time",
        unit_id="unit",
        lag=1,
        n_boot=4,
        random_seed=1,
    )

    summary_rows = result.summary_rows()
    assert result.metadata["identified_lags"] == (1,)
    assert summary_rows[0]["lag"] == 1
    assert summary_rows[0]["estimate_raw"] == pytest.approx(1.0)


def test_data_first_did_check_keeps_raw_placebo_when_control_sd_is_zero():
    from diddesign.diagnostics import did_check

    data_rows = []
    for unit, treated, outcomes in (
        ("a", 1, (1.0, 4.0, 8.0)),
        ("b", 1, (2.0, 5.0, 9.0)),
        ("c", 0, (5.0, 6.0, 7.0)),
        ("d", 0, (5.0, 6.0, 7.0)),
    ):
        for time, outcome in enumerate(outcomes):
            data_rows.append(
                {
                    "unit": unit,
                    "time": time,
                    "treat": int(treated and time == 2),
                    "outcome": outcome,
                }
            )

    result = did_check(
        data=data_rows,
        outcome="outcome",
        treatment="treat",
        time="time",
        unit_id="unit",
        lag=1,
        n_boot=10,
        random_seed=1,
    )

    diagnostic_row = result.diagnostic_rows()[0]
    assert result.metadata["identified_lags"] == (1,)
    assert result.metadata["raw_only_lags"] == (1,)
    assert diagnostic_row["estimate_raw"] == pytest.approx(2.0)
    assert diagnostic_row["std_error_raw"] == pytest.approx(0.0, abs=1e-12)
    assert diagnostic_row["estimate_std"] is None
    assert diagnostic_row["std_error_std"] is None
    assert diagnostic_row["eqci95_lb_std"] is None
    assert diagnostic_row["eqci95_ub_std"] is None
    assert result.summary_rows() == (
        {
            "lag": 1,
            "estimate_raw": pytest.approx(2.0),
            "std_error_raw": pytest.approx(0.0, abs=1e-12),
            "eqci95_lb_std": None,
            "eqci95_ub_std": None,
        },
    )
    assert result.named_plot_payloads()["placebo"] == ()


def test_rcs_did_check_is_invariant_to_lexically_ordered_string_time_relabeling():
    from diddesign.diagnostics import did_check

    rows = _small_rcs_check_rows()
    relabeled_rows = []
    time_relabel = {
        current_time: f"period_{index:02d}"
        for index, current_time in enumerate(sorted({row["time"] for row in rows}))
    }
    for row in rows:
        relabeled_row = dict(row)
        relabeled_row["time"] = time_relabel[row["time"]]
        relabeled_rows.append(relabeled_row)

    base = did_check(
        data=rows,
        outcome="outcome",
        treatment="treat_group",
        time="time",
        post="post",
        data_type="rcs",
        id_cluster="cluster",
        lag=1,
        n_boot=5,
        random_seed=77,
    )
    relabeled = did_check(
        data=relabeled_rows,
        outcome="outcome",
        treatment="treat_group",
        time="time",
        post="post",
        data_type="rcs",
        id_cluster="cluster",
        lag=1,
        n_boot=5,
        random_seed=77,
    )

    assert relabeled.metadata["time_order"] == (
        "period_00",
        "period_01",
        "period_02",
    )
    assert "time-order:string" in relabeled.metadata["validation_trace"]
    assert base.metadata["identified_lags"] == relabeled.metadata["identified_lags"]
    assert base.metadata["unidentified_lags"] == relabeled.metadata["unidentified_lags"]
    assert base.metadata["raw_only_lags"] == relabeled.metadata["raw_only_lags"]
    assert base.summary_rows() == pytest.approx(relabeled.summary_rows())
    assert base.diagnostic_rows() == pytest.approx(relabeled.diagnostic_rows())
    assert base.named_plot_payloads()["placebo"] == pytest.approx(
        relabeled.named_plot_payloads()["placebo"]
    )
    assert base.named_plot_payloads()["trends"] == pytest.approx(
        relabeled.named_plot_payloads()["trends"]
    )


@pytest.mark.parametrize("reserved_key", ["outcome", "branch", "design", "covariates", "n_boot_realized"])
def test_data_first_did_check_rejects_metadata_that_overrides_generated_contract_fields(reserved_key):
    from diddesign.diagnostics import did_check

    with pytest.raises(
        ValueError,
        match=rf"metadata cannot override data-driven did_check fields: {reserved_key}\.",
    ):
        did_check(
            data=_small_panel_check_rows(),
            outcome="outcome",
            treatment="treat",
            time="time",
            unit_id="unit",
            covariates=() if reserved_key == "covariates" else None,
            lag=1,
            n_boot=5,
            random_seed=7,
            metadata={reserved_key: "poisoned"},
        )


def test_data_first_did_check_preserves_custom_metadata_without_overriding_generated_contract():
    from diddesign.diagnostics import did_check

    result = did_check(
        data=_small_panel_check_rows(),
        outcome="outcome",
        treatment="treat",
        time="time",
        unit_id="unit",
        lag=1,
        n_boot=5,
        random_seed=7,
        metadata={"audit_note": "ok"},
    )

    assert result.metadata["outcome"] == "outcome"
    assert result.metadata["branch"] == "did-panel-check"
    assert result.metadata["design"] == "did"
    assert result.metadata["n_boot_realized"] == 5
    assert result.metadata["audit_note"] == "ok"


def test_data_first_did_check_rejects_rank_deficient_placebo_regressions():
    from diddesign.diagnostics import did_check

    data_rows = []
    for unit, treated, outcomes in (
        ("a", 1, (10.0, 11.0, 14.0)),
        ("b", 1, (12.0, 13.0, 16.0)),
        ("c", 0, (8.0, 9.0, 10.0)),
        ("d", 0, (9.0, 10.0, 11.0)),
    ):
        for time, outcome in enumerate(outcomes):
            data_rows.append(
                {
                    "unit": unit,
                    "time": time,
                    "treat": treated if time == 2 else 0,
                    "outcome": outcome,
                    "x": 1 if treated and time == 1 else 0,
                }
            )

    with pytest.raises(ValueError, match=r"No identifiable lag\(\) values remain for placebo tests\."):
        did_check(
            data=data_rows,
            outcome="outcome",
            treatment="treat",
            time="time",
            unit_id="unit",
            covariates=("x",),
            lag=1,
            n_boot=5,
            random_seed=7,
        )


def test_data_first_did_check_rejects_placebo_lags_without_two_raw_bootstrap_draws(monkeypatch):
    from diddesign import diagnostics

    data_rows = [
        {"unit": "a", "time": 0, "treat": 0, "outcome": 10.0},
        {"unit": "a", "time": 1, "treat": 0, "outcome": 11.0},
        {"unit": "a", "time": 2, "treat": 1, "outcome": 14.0},
        {"unit": "b", "time": 0, "treat": 0, "outcome": 12.0},
        {"unit": "b", "time": 1, "treat": 0, "outcome": 13.0},
        {"unit": "b", "time": 2, "treat": 1, "outcome": 16.0},
        {"unit": "c", "time": 0, "treat": 0, "outcome": 8.0},
        {"unit": "c", "time": 1, "treat": 0, "outcome": 9.0},
        {"unit": "c", "time": 2, "treat": 0, "outcome": 10.0},
        {"unit": "d", "time": 0, "treat": 0, "outcome": 9.0},
        {"unit": "d", "time": 1, "treat": 0, "outcome": 10.0},
        {"unit": "d", "time": 2, "treat": 0, "outcome": 11.0},
    ]

    def no_valid_raw_draws(frame, *, lags, covariates, n_boot, random_seed):
        return (
            {lag: [None for _ in range(n_boot)] for lag in lags},
            {lag: [None for _ in range(n_boot)] for lag in lags},
        )

    monkeypatch.setattr(diagnostics, "_compute_bootstrap_draws", no_valid_raw_draws)

    with pytest.raises(
        ValueError,
        match=r"No identifiable lag\(\) values remain for placebo tests: fewer than two valid raw bootstrap draws are available for every requested lag\.",
    ):
        diagnostics.did_check(
            data=data_rows,
            outcome="outcome",
            treatment="treat",
            time="time",
            unit_id="unit",
            lag=1,
            n_boot=5,
            random_seed=7,
        )


def test_did_check_rejects_non_mapping_option_at_public_boundary():
    from diddesign.diagnostics import did_check

    with pytest.raises(TypeError, match="option must be a mapping when provided\\."):
        did_check(
            data=[],
            outcome="outcome",
            treatment="treat",
            time="time",
            unit_id="unit",
            option=["lag", 1],
        )


def test_did_check_accepts_numpy_boolean_is_panel_scalar_at_public_boundary():
    from diddesign.diagnostics import did_check

    data_rows = [
        {"unit": "a", "time": 0, "treat": 0, "outcome": 10.0},
        {"unit": "a", "time": 1, "treat": 0, "outcome": 11.0},
        {"unit": "a", "time": 2, "treat": 1, "outcome": 14.0},
        {"unit": "b", "time": 0, "treat": 0, "outcome": 12.0},
        {"unit": "b", "time": 1, "treat": 0, "outcome": 13.0},
        {"unit": "b", "time": 2, "treat": 1, "outcome": 16.0},
        {"unit": "c", "time": 0, "treat": 0, "outcome": 8.0},
        {"unit": "c", "time": 1, "treat": 0, "outcome": 9.0},
        {"unit": "c", "time": 2, "treat": 0, "outcome": 10.0},
        {"unit": "d", "time": 0, "treat": 0, "outcome": 9.0},
        {"unit": "d", "time": 1, "treat": 0, "outcome": 10.0},
        {"unit": "d", "time": 2, "treat": 0, "outcome": 11.0},
    ]

    result = did_check(
        data=data_rows,
        formula="outcome ~ treat",
        time="time",
        unit_id="unit",
        is_panel=np.bool_(True),
        option={"lag": 1, "n_boot": 5},
        random_seed=7,
        metadata={"manual": np.bool_(True)},
    )

    assert result.metadata["data_type"] == "panel"
    assert result.metadata["manual"] is True


def test_did_check_rejects_non_boolean_is_panel_values_at_public_boundary():
    from diddesign.diagnostics import did_check

    for bad_value in (1, 0, "true"):
        with pytest.raises(TypeError, match="is_panel must be a boolean when provided\\."):
            did_check(
                data=[],
                outcome="outcome",
                treatment="treat",
                time="time",
                unit_id="unit",
                is_panel=bad_value,
            )


def test_did_check_rejects_invalid_id_cluster_values_at_public_boundary():
    from diddesign.diagnostics import did_check

    with pytest.raises(TypeError, match="id_cluster must be a column name string when provided\\."):
        did_check(
            data=[],
            outcome="outcome",
            treatment="treat",
            time="time",
            unit_id="unit",
            id_cluster=1,
        )

    for bad_value in ("", "   "):
        with pytest.raises(ValueError, match="id_cluster must be a non-empty column name\\."):
            did_check(
                data=[],
                outcome="outcome",
                treatment="treat",
                time="time",
                unit_id="unit",
                option={"id_cluster": bad_value},
            )


@pytest.mark.parametrize("id_cluster", ["outcome", "treat", "time"])
def test_did_check_rejects_id_cluster_that_reuses_estimand_role_columns(id_cluster):
    from diddesign.diagnostics import did_check

    rows = [
        {"unit": "a", "time": 0, "treat": 0, "outcome": 1.0},
        {"unit": "a", "time": 1, "treat": 0, "outcome": 1.1},
        {"unit": "a", "time": 2, "treat": 1, "outcome": 1.5},
        {"unit": "b", "time": 0, "treat": 0, "outcome": 0.8},
        {"unit": "b", "time": 1, "treat": 0, "outcome": 0.9},
        {"unit": "b", "time": 2, "treat": 1, "outcome": 1.4},
        {"unit": "c", "time": 0, "treat": 0, "outcome": 0.7},
        {"unit": "c", "time": 1, "treat": 0, "outcome": 0.8},
        {"unit": "c", "time": 2, "treat": 0, "outcome": 0.9},
        {"unit": "d", "time": 0, "treat": 0, "outcome": 1.2},
        {"unit": "d", "time": 1, "treat": 0, "outcome": 1.3},
        {"unit": "d", "time": 2, "treat": 0, "outcome": 1.4},
    ]

    with pytest.raises(
        ValueError,
        match="id_cluster must not reuse outcome, treatment, time, or post columns\\.",
    ):
        did_check(
            data=rows,
            outcome="outcome",
            treatment="treat",
            time="time",
            unit_id="unit",
            id_cluster=id_cluster,
            lag=1,
            n_boot=2,
        )


def test_did_check_allows_explicit_id_cluster_matching_panel_unit_identifier():
    from diddesign.diagnostics import did_check

    rows = [
        {"unit": "a", "time": 0, "treat": 0, "outcome": 1.0},
        {"unit": "a", "time": 1, "treat": 0, "outcome": 1.1},
        {"unit": "a", "time": 2, "treat": 1, "outcome": 1.5},
        {"unit": "b", "time": 0, "treat": 0, "outcome": 0.8},
        {"unit": "b", "time": 1, "treat": 0, "outcome": 0.9},
        {"unit": "b", "time": 2, "treat": 1, "outcome": 1.4},
        {"unit": "c", "time": 0, "treat": 0, "outcome": 0.7},
        {"unit": "c", "time": 1, "treat": 0, "outcome": 0.8},
        {"unit": "c", "time": 2, "treat": 0, "outcome": 0.9},
        {"unit": "d", "time": 0, "treat": 0, "outcome": 1.2},
        {"unit": "d", "time": 1, "treat": 0, "outcome": 1.3},
        {"unit": "d", "time": 2, "treat": 0, "outcome": 1.4},
    ]

    result = did_check(
        data=rows,
        outcome="outcome",
        treatment="treat",
        time="time",
        unit_id="unit",
        id_cluster="unit",
        lag=1,
        n_boot=4,
        random_seed=20260518,
    )

    assert result.metadata["cluster_column"] == "unit"
    assert result.metadata["cluster_mode"] == "explicit"


def test_did_check_rejects_missing_explicit_cluster_values_before_bootstrap_sampling():
    from diddesign.diagnostics import did_check

    rows = _small_panel_check_rows()
    rows[1]["cluster"] = np.nan
    for row in rows:
        row.setdefault("cluster", row["unit"])

    with pytest.raises(
        ValueError,
        match="id_cluster must not contain missing values; found missing value in row 1\\.",
    ):
        did_check(
            data=rows,
            outcome="outcome",
            treatment="treat",
            time="time",
            unit_id="unit",
            id_cluster="cluster",
            lag=1,
            n_boot=4,
            random_seed=20260518,
        )


def test_data_driven_did_check_rejects_missing_model_columns_before_pandas_projection():
    from diddesign.diagnostics import did_check

    rows = [
        {"unit": "a", "time": 0, "treat": 0, "outcome": 1.0},
        {"unit": "a", "time": 1, "treat": 0, "outcome": 1.1},
        {"unit": "a", "time": 2, "treat": 1, "outcome": 1.5},
        {"unit": "b", "time": 0, "treat": 0, "outcome": 0.8},
        {"unit": "b", "time": 1, "treat": 0, "outcome": 0.9},
        {"unit": "b", "time": 2, "treat": 0, "outcome": 1.0},
    ]

    with pytest.raises(ValueError, match="Column 'missing_cluster' is required"):
        did_check(
            data=rows,
            outcome="outcome",
            treatment="treat",
            time="time",
            unit_id="unit",
            id_cluster="missing_cluster",
            lag=1,
            n_boot=2,
        )

    with pytest.raises(ValueError, match="Column 'missing_covariate' is required"):
        did_check(
            data=rows,
            outcome="outcome",
            treatment="treat",
            time="time",
            unit_id="unit",
            covariates=("missing_covariate",),
            lag=1,
            n_boot=2,
        )


@pytest.mark.parametrize(
    ("kwargs", "exception", "error"),
    [
        ({"outcome": 1}, TypeError, "outcome must be a column name string\\."),
        ({"treatment": 1}, TypeError, "treatment must be a column name string\\."),
        ({"time": 1}, TypeError, "time must be a column name string\\."),
        ({"unit_id": 1}, TypeError, "unit_id must be a column name string when provided\\."),
        ({"post": 1, "data_type": "rcs"}, TypeError, "post must be a column name string when provided\\."),
        ({"outcome": ""}, ValueError, "outcome must be a non-empty column name\\."),
        ({"treatment": "   "}, ValueError, "treatment must be a non-empty column name\\."),
        ({"time": ""}, ValueError, "time must be a non-empty column name\\."),
        ({"unit_id": "   "}, ValueError, "unit_id must be a non-empty column name\\."),
        ({"post": "", "data_type": "rcs"}, ValueError, "post must be a non-empty column name\\."),
    ],
)
def test_did_check_rejects_invalid_column_name_values_at_public_boundary(kwargs, exception, error):
    from diddesign.diagnostics import did_check

    call_kwargs = {
        "data": [],
        "outcome": "outcome",
        "treatment": "treat",
        "time": "time",
        "unit_id": "unit",
    }
    if kwargs.get("data_type") == "rcs":
        call_kwargs = {
            "data": [],
            "outcome": "outcome",
            "treatment": "treat_group",
            "time": "time",
            "post": "post",
            "data_type": "rcs",
        }
    call_kwargs.update(kwargs)

    with pytest.raises(exception, match=error):
        did_check(**call_kwargs)


@pytest.mark.parametrize(
    ("kwargs", "exception", "error"),
    [
        ({"design": 1}, TypeError, "design must be a string\\."),
        ({"data_type": 1}, TypeError, "data_type must be a string\\."),
        ({"design": "event"}, ValueError, "design must be 'did' or 'sa'\\."),
        ({"data_type": "cross"}, ValueError, "data_type must be 'panel' or 'rcs'\\."),
    ],
)
def test_did_check_rejects_invalid_design_and_data_type_values_at_public_boundary(kwargs, exception, error):
    from diddesign.diagnostics import did_check

    call_kwargs = {
        "data": [],
        "outcome": "outcome",
        "treatment": "treat",
        "time": "time",
        "unit_id": "unit",
    }
    call_kwargs.update(kwargs)

    with pytest.raises(exception, match=error):
        did_check(**call_kwargs)


def test_plot_check_uses_named_payloads_and_omits_raw_only_rows():
    from diddesign.diagnostics import DidCheckDiagnosticRow, DidCheckTrendRow, did_check
    from diddesign.plotting import check

    eqci95_lb_std, eqci95_ub_std = _eqci_bounds(estimate_std=0.25, std_error_std=0.1)

    result = did_check(
        diagnostic_rows=[
            DidCheckDiagnosticRow(
                lag=1,
                estimate_std=0.25,
                std_error_std=0.1,
                estimate_raw=1.5,
                std_error_raw=0.4,
                eqci95_lb_std=eqci95_lb_std,
                eqci95_ub_std=eqci95_ub_std,
            ),
            DidCheckDiagnosticRow(
                lag=2,
                estimate_std=None,
                std_error_std=None,
                estimate_raw=0.5,
                std_error_raw=0.2,
                eqci95_lb_std=None,
                eqci95_ub_std=None,
            ),
        ],
        trends_rows=[
            DidCheckTrendRow(
                time_to_treat=-1,
                group="Control",
                outcome_mean=2.0,
                outcome_sd=0.3,
                n_obs=12,
            )
        ],
        metadata={"raw_only_lags": [2]},
    )

    payloads = check(result)

    assert tuple(payloads) == ("placebo", "trends")
    assert result.diagnostic_table[1].estimate_raw == 0.5
    assert payloads["placebo"] == (
        {
            "lag": 1,
            "time_to_treat": -1,
            "estimate_std": 0.25,
            "std_error_std": 0.1,
            "eqci95_lb_std": eqci95_lb_std,
            "eqci95_ub_std": eqci95_ub_std,
        },
    )


def test_plot_check_named_payloads_can_return_detached_data_frames():
    from diddesign.diagnostics import DidCheckDiagnosticRow, DidCheckTrendRow, did_check
    from diddesign.plotting import check

    eqci95_lb_std, eqci95_ub_std = _eqci_bounds(estimate_std=0.25, std_error_std=0.1)
    result = did_check(
        diagnostic_rows=[
            DidCheckDiagnosticRow(
                lag=1,
                estimate_std=0.25,
                std_error_std=0.1,
                estimate_raw=1.5,
                std_error_raw=0.4,
                eqci95_lb_std=eqci95_lb_std,
                eqci95_ub_std=eqci95_ub_std,
            ),
        ],
        trends_rows=[
            DidCheckTrendRow(
                time_to_treat=-1,
                group="Control",
                outcome_mean=2.0,
                outcome_sd=0.3,
                n_obs=12,
            ),
        ],
    )

    payloads = check(result)
    frames = check(result, as_frame=True)

    assert tuple(frames) == ("placebo", "trends")
    assert frames["placebo"]["lag"].tolist() == [1]
    assert frames["trends"]["group"].tolist() == ["Control"]
    assert frames["placebo"].loc[0, "estimate_std"] == pytest.approx(
        payloads["placebo"][0]["estimate_std"]
    )
    assert frames["placebo"].equals(result.to_placebo_frame())
    assert frames["trends"].equals(result.to_trends_frame())

    frames["placebo"].loc[0, "estimate_std"] = -999.0

    assert check(result)["placebo"][0]["estimate_std"] == pytest.approx(
        payloads["placebo"][0]["estimate_std"]
    )


def test_plot_check_empty_frames_keep_stable_public_columns():
    from diddesign.diagnostics import did_check
    from diddesign.plotting import check

    result = did_check(diagnostic_rows=(), trends_rows=())

    frames = check(result, as_frame=True)

    assert tuple(frames) == ("placebo", "trends")
    assert list(frames["placebo"].columns) == [
        "lag",
        "time_to_treat",
        "estimate_std",
        "std_error_std",
        "eqci95_lb_std",
        "eqci95_ub_std",
    ]
    assert list(frames["trends"].columns) == [
        "time_to_treat",
        "group",
        "outcome_mean",
        "outcome_sd",
        "ci90_lb",
        "ci90_ub",
        "n_obs",
    ]
    assert frames["placebo"].empty
    assert frames["trends"].empty


def test_plot_check_derives_trend_ci_from_outcome_sd():
    from diddesign.diagnostics import DidCheckTrendRow, did_check
    from diddesign.plotting import check

    result = did_check(
        diagnostic_rows=[],
        trends_rows=[
            DidCheckTrendRow(
                time_to_treat=0,
                group="Treated",
                outcome_mean=3.2,
                outcome_sd=0.25,
                n_obs=20,
            )
        ],
    )

    trend_row = check(result)["trends"][0]
    z_90 = NormalDist().inv_cdf(0.95)

    assert trend_row["group"] == "Treated"
    assert trend_row["time_to_treat"] == 0
    assert trend_row["outcome_sd"] == 0.25
    assert trend_row["n_obs"] == 20
    assert math.isclose(trend_row["ci90_lb"], 3.2 - z_90 * 0.25)
    assert math.isclose(trend_row["ci90_ub"], 3.2 + z_90 * 0.25)


def test_trend_rows_normalize_direct_group_labels_before_plot_payloads():
    from diddesign.diagnostics import DidCheckTrendRow, did_check
    from diddesign.plotting import check

    result = did_check(
        diagnostic_rows=[],
        trends_rows=[
            DidCheckTrendRow(
                time_to_treat=-1,
                group=" 0 ",
                outcome_mean=2.0,
                outcome_sd=0.3,
                n_obs=12,
            ),
            DidCheckTrendRow(
                time_to_treat=0,
                group=1.0,
                outcome_mean=3.2,
                outcome_sd=0.25,
                n_obs=20,
            ),
        ],
    )

    assert [row["group"] for row in check(result)["trends"]] == ["Control", "Treated"]

    assert DidCheckTrendRow(
        time_to_treat=1,
        group="control",
        outcome_mean=2.0,
        outcome_sd=0.3,
        n_obs=12,
    ).group == "Control"
    assert DidCheckTrendRow.from_mapping(
        {
            "time_to_treat": 1,
            "group": "treated",
            "outcome_mean": 3.2,
            "outcome_sd": 0.25,
            "n_obs": 20,
        }
    ).group == "Treated"

    with pytest.raises(ValueError, match="group is required\\."):
        DidCheckTrendRow(
            time_to_treat=0,
            group=" ",
            outcome_mean=3.2,
            outcome_sd=0.25,
            n_obs=20,
        )

    with pytest.raises(ValueError, match="group must be 'Control', 'Treated', 0, or 1\\."):
        DidCheckTrendRow(
            time_to_treat=0,
            group="Placebo",
            outcome_mean=3.2,
            outcome_sd=0.25,
            n_obs=20,
        )

    with pytest.raises(ValueError, match="group must be 'Control', 'Treated', 0, or 1\\."):
        DidCheckTrendRow.from_mapping(
            {
                "time_to_treat": 0,
                "group": "comparison",
                "outcome_mean": 3.2,
                "outcome_sd": 0.25,
                "n_obs": 20,
            }
        )


def test_did_check_rejects_eqci_bounds_inconsistent_with_standardized_inputs():
    from diddesign.diagnostics import DidCheckDiagnosticRow

    with pytest.raises(ValueError, match="Standardized estimate and std_error require EqCI95 bounds\\."):
        DidCheckDiagnosticRow(
            lag=1,
            estimate_std=0.25,
            std_error_std=0.1,
            estimate_raw=1.5,
            std_error_raw=0.4,
            eqci95_lb_std=None,
            eqci95_ub_std=None,
        )

    with pytest.raises(ValueError, match="EqCI95 bounds must match the standardized estimate and std_error."):
        DidCheckDiagnosticRow(
            lag=1,
            estimate_std=0.25,
            std_error_std=0.1,
            estimate_raw=1.5,
            std_error_raw=0.4,
            eqci95_lb_std=-0.30,
            eqci95_ub_std=0.30,
        )


def test_did_check_allows_zero_lag_diagnostic_rows_and_maps_them_to_time_zero():
    from diddesign.diagnostics import DidCheckDiagnosticRow, did_check
    from diddesign.plotting import check

    eqci95_lb_std, eqci95_ub_std = _eqci_bounds(estimate_std=0.25, std_error_std=0.1)

    row = DidCheckDiagnosticRow(
        lag=0,
        estimate_std=0.25,
        std_error_std=0.1,
        estimate_raw=1.5,
        std_error_raw=0.4,
        eqci95_lb_std=eqci95_lb_std,
        eqci95_ub_std=eqci95_ub_std,
    )

    result = did_check(diagnostic_rows=[row], trends_rows=[])

    assert result.summary_rows() == (
        {
            "lag": 0,
            "estimate_raw": 1.5,
            "std_error_raw": 0.4,
            "eqci95_lb_std": eqci95_lb_std,
            "eqci95_ub_std": eqci95_ub_std,
        },
    )
    assert check(result)["placebo"] == (
        {
            "lag": 0,
            "time_to_treat": 0,
            "estimate_std": 0.25,
            "std_error_std": 0.1,
            "eqci95_lb_std": eqci95_lb_std,
            "eqci95_ub_std": eqci95_ub_std,
        },
    )


def test_did_check_rejects_negative_placebo_lag():
    from diddesign.diagnostics import DidCheckDiagnosticRow

    eqci95_lb_std, eqci95_ub_std = _eqci_bounds(estimate_std=0.25, std_error_std=0.1)

    with pytest.raises(ValueError, match="lag must be a non-negative diagnostic lag."):
        DidCheckDiagnosticRow(
            lag=-1,
            estimate_std=0.25,
            std_error_std=0.1,
            estimate_raw=1.5,
            std_error_raw=0.4,
            eqci95_lb_std=eqci95_lb_std,
            eqci95_ub_std=eqci95_ub_std,
        )


def test_diagnostic_rows_reject_non_finite_public_payload_values():
    from diddesign.diagnostics import DidCheckDiagnosticRow

    eqci95_lb_std, eqci95_ub_std = _eqci_bounds(estimate_std=0.25, std_error_std=0.1)

    with pytest.raises(ValueError, match="estimate_raw must be finite\\."):
        DidCheckDiagnosticRow(
            lag=1,
            estimate_std=0.25,
            std_error_std=0.1,
            estimate_raw=float("nan"),
            std_error_raw=0.4,
            eqci95_lb_std=eqci95_lb_std,
            eqci95_ub_std=eqci95_ub_std,
        )

    with pytest.raises(ValueError, match="estimate_raw must be finite\\."):
        DidCheckDiagnosticRow(
            lag=1,
            estimate_std=0.25,
            std_error_std=0.1,
            estimate_raw=np.bool_(True),
            std_error_raw=0.4,
            eqci95_lb_std=eqci95_lb_std,
            eqci95_ub_std=eqci95_ub_std,
        )

    with pytest.raises(ValueError, match="std_error_raw must be non-negative\\."):
        DidCheckDiagnosticRow(
            lag=1,
            estimate_std=0.25,
            std_error_std=0.1,
            estimate_raw=1.5,
            std_error_raw=-0.4,
            eqci95_lb_std=eqci95_lb_std,
            eqci95_ub_std=eqci95_ub_std,
        )

    with pytest.raises(ValueError, match="std_error_std must be non-negative\\."):
        DidCheckDiagnosticRow(
            lag=1,
            estimate_std=0.25,
            std_error_std=-0.1,
            estimate_raw=1.5,
            std_error_raw=0.4,
            eqci95_lb_std=eqci95_lb_std,
            eqci95_ub_std=eqci95_ub_std,
        )

    with pytest.raises(ValueError, match="eqci95_ub_std must be finite\\."):
        DidCheckDiagnosticRow(
            lag=1,
            estimate_std=0.25,
            std_error_std=0.1,
            estimate_raw=1.5,
            std_error_raw=0.4,
            eqci95_lb_std=eqci95_lb_std,
            eqci95_ub_std=float("inf"),
        )


def test_did_check_from_mapping_rejects_non_integer_lag_values():
    from diddesign.diagnostics import DidCheckDiagnosticRow

    eqci95_lb_std, eqci95_ub_std = _eqci_bounds(estimate_std=0.25, std_error_std=0.1)

    with pytest.raises(ValueError, match="lag must be an integer"):
        DidCheckDiagnosticRow.from_mapping(
            {
                "lag": 1.5,
                "estimate_std": 0.25,
                "std_error_std": 0.1,
                "estimate_raw": 1.5,
                "std_error_raw": 0.4,
                "eqci95_lb_std": eqci95_lb_std,
                "eqci95_ub_std": eqci95_ub_std,
            }
        )


def test_diagnostic_payload_from_mapping_rejects_unparseable_numeric_fields_with_field_errors():
    from diddesign.diagnostics import DidCheckDiagnosticRow, DidCheckPatternRow, DidCheckTrendRow

    with pytest.raises(ValueError, match="estimate_std must be finite\\."):
        DidCheckDiagnosticRow.from_mapping(
            {
                "lag": 1,
                "estimate_std": "not-a-number",
                "std_error_std": 0.1,
                "estimate_raw": 1.5,
                "std_error_raw": 0.4,
                "eqci95_lb_std": -0.41448536269514724,
                "eqci95_ub_std": 0.41448536269514724,
            }
        )

    with pytest.raises(ValueError, match="outcome_mean must be finite\\."):
        DidCheckTrendRow.from_mapping(
            {
                "time_to_treat": 0,
                "group": "Control",
                "outcome_mean": "not-a-number",
                "outcome_sd": 0.3,
                "n_obs": 12,
            }
        )

    with pytest.raises(ValueError, match="id_time must be finite\\."):
        DidCheckPatternRow.from_mapping(
            {
                "id_time": "not-a-number",
                "unit_order": 1,
                "status": "treated",
            }
        )


def test_diagnostic_payload_rows_reject_float_integer_identity_at_direct_boundary():
    from diddesign.diagnostics import DidCheckDiagnosticRow, DidCheckPatternRow, DidCheckTrendRow

    eqci95_lb_std, eqci95_ub_std = _eqci_bounds(estimate_std=0.25, std_error_std=0.1)

    with pytest.raises(ValueError, match="lag must be an integer\\."):
        DidCheckDiagnosticRow(
            lag=1.0,
            estimate_std=0.25,
            std_error_std=0.1,
            estimate_raw=1.5,
            std_error_raw=0.4,
            eqci95_lb_std=eqci95_lb_std,
            eqci95_ub_std=eqci95_ub_std,
        )

    with pytest.raises(ValueError, match="n_obs must be an integer\\."):
        DidCheckTrendRow(
            time_to_treat=0,
            group="Control",
            outcome_mean=2.0,
            outcome_sd=0.3,
            n_obs=12.0,
        )

    with pytest.raises(ValueError, match="unit_order must be an integer\\."):
        DidCheckPatternRow(id_time=1, unit_order=2.0, status="treated")


def test_plot_check_rejects_negative_outcome_sd():
    from diddesign.diagnostics import DidCheckTrendRow

    with pytest.raises(ValueError, match="outcome_sd must be non-negative."):
        DidCheckTrendRow(
            time_to_treat=0,
            group="Treated",
            outcome_mean=3.2,
            outcome_sd=-0.25,
            n_obs=20,
        )


def test_trend_rows_reject_zero_observation_count():
    from diddesign.diagnostics import DidCheckTrendRow

    with pytest.raises(ValueError, match="n_obs must be positive\\."):
        DidCheckTrendRow(
            time_to_treat=0,
            group="Treated",
            outcome_mean=3.2,
            outcome_sd=0.25,
            n_obs=0,
        )

    with pytest.raises(ValueError, match="n_obs must be positive\\."):
        DidCheckTrendRow.from_mapping(
            {
                "time_to_treat": 0,
                "group": "Control",
                "outcome_mean": 2.0,
                "outcome_sd": 0.3,
                "n_obs": 0,
            }
        )


def test_trend_rows_reject_non_finite_public_payload_values():
    from diddesign.diagnostics import DidCheckTrendRow

    with pytest.raises(ValueError, match="outcome_mean must be finite\\."):
        DidCheckTrendRow(
            time_to_treat=0,
            group="Treated",
            outcome_mean=float("nan"),
            outcome_sd=0.25,
            n_obs=20,
        )

    with pytest.raises(ValueError, match="time_to_treat must be finite\\."):
        DidCheckTrendRow(
            time_to_treat=float("inf"),
            group="Treated",
            outcome_mean=3.2,
            outcome_sd=0.25,
            n_obs=20,
        )

    with pytest.raises(ValueError, match="n_obs must be an integer\\."):
        DidCheckTrendRow(
            time_to_treat=0,
            group="Treated",
            outcome_mean=3.2,
            outcome_sd=0.25,
            n_obs=20.5,
        )


def test_trend_rows_reject_non_finite_and_non_binary_numeric_group_labels():
    from diddesign.diagnostics import DidCheckTrendRow

    with pytest.raises(ValueError, match="group must be finite when numeric\\."):
        DidCheckTrendRow(
            time_to_treat=0,
            group=np.nan,
            outcome_mean=3.2,
            outcome_sd=0.25,
            n_obs=20,
        )

    with pytest.raises(ValueError, match="numeric group labels must be 0/1\\."):
        DidCheckTrendRow.from_mapping(
            {
                "time_to_treat": 0,
                "Gi": 2,
                "outcome_mean": 3.2,
                "outcome_sd": 0.25,
                "n_obs": 20,
            }
        )


def test_trend_row_from_mapping_rejects_non_integer_n_obs():
    from diddesign.diagnostics import DidCheckTrendRow

    with pytest.raises(ValueError, match="n_obs must be an integer"):
        DidCheckTrendRow.from_mapping(
            {
                "time_to_treat": -1,
                "group": "Control",
                "outcome_mean": 2.0,
                "outcome_sd": 0.3,
                "n_obs": 12.5,
            }
        )


def test_diagnostic_payload_from_mapping_factories_reject_non_mapping_inputs():
    from diddesign.diagnostics import DidCheckDiagnosticRow, DidCheckPatternRow, DidCheckTrendRow

    with pytest.raises(TypeError, match="mapping must be a mapping\\."):
        DidCheckDiagnosticRow.from_mapping([("lag", 1)])

    with pytest.raises(TypeError, match="mapping must be a mapping\\."):
        DidCheckTrendRow.from_mapping([("time_to_treat", -1)])

    with pytest.raises(TypeError, match="mapping must be a mapping\\."):
        DidCheckPatternRow.from_mapping([("time", 1)])


def test_pattern_row_from_mapping_rejects_non_integer_unit_order():
    from diddesign.diagnostics import DidCheckPatternRow

    with pytest.raises(ValueError, match="unit_order must be an integer"):
        DidCheckPatternRow.from_mapping(
            {
                "id_time": 1,
                "unit_order": 2.25,
                "status": "treated",
            }
        )


def test_pattern_rows_normalize_treatment_coded_status_labels():
    from diddesign.diagnostics import DidCheckPatternRow

    control_row = DidCheckPatternRow.from_mapping(
        {
            "time": 1,
            "unit_order": 1,
            "treatment": 0.0,
        }
    )
    treated_row = DidCheckPatternRow(
        id_time=2,
        unit_order=2,
        status=1.0,
    )

    assert control_row.plot_row() == {
        "id_time": 1,
        "unit_order": 1,
        "status": "control",
    }
    assert treated_row.plot_row() == {
        "id_time": 2,
        "unit_order": 2,
        "status": "treated",
    }

    with pytest.raises(ValueError, match="status must be either 'control' or 'treated'\\."):
        DidCheckPatternRow.from_mapping(
            {
                "time": 1,
                "unit_order": 1,
                "treatment": 2,
            }
        )


def test_pattern_rows_reject_non_finite_public_payload_values():
    from diddesign.diagnostics import DidCheckPatternRow

    with pytest.raises(ValueError, match="id_time must be finite\\."):
        DidCheckPatternRow(id_time=float("nan"), unit_order=1, status="treated")

    with pytest.raises(ValueError, match="unit_order must be positive\\."):
        DidCheckPatternRow(id_time=1, unit_order=0, status="treated")


def test_data_driven_did_check_rejects_bootstrap_sizes_below_two():
    from diddesign.diagnostics import did_check

    rows = [
        {"id": 1, "year": 1, "treat": 0, "outcome": 10.0},
        {"id": 1, "year": 2, "treat": 0, "outcome": 11.0},
        {"id": 1, "year": 3, "treat": 1, "outcome": 15.0},
        {"id": 2, "year": 1, "treat": 0, "outcome": 12.0},
        {"id": 2, "year": 2, "treat": 0, "outcome": 13.0},
        {"id": 2, "year": 3, "treat": 1, "outcome": 17.0},
        {"id": 3, "year": 1, "treat": 0, "outcome": 7.0},
        {"id": 3, "year": 2, "treat": 0, "outcome": 8.0},
        {"id": 3, "year": 3, "treat": 0, "outcome": 9.0},
        {"id": 4, "year": 1, "treat": 0, "outcome": 8.0},
        {"id": 4, "year": 2, "treat": 0, "outcome": 9.0},
        {"id": 4, "year": 3, "treat": 0, "outcome": 10.0},
    ]

    with pytest.raises(ValueError, match="n_boot must be an integer greater than or equal to 2."):
        did_check(
            data=rows,
            outcome="outcome",
            treatment="treat",
            time="year",
            unit_id="id",
            lag=1,
            n_boot=1,
        )


@pytest.mark.parametrize("bad_n_boot", [True, False, 4.0, "4", None])
def test_data_driven_did_check_rejects_non_integer_bootstrap_sizes_at_public_boundary(bad_n_boot):
    from diddesign.diagnostics import did_check

    rows = [
        {"id": 1, "year": 1, "treat": 0, "outcome": 10.0},
        {"id": 1, "year": 2, "treat": 0, "outcome": 11.0},
        {"id": 1, "year": 3, "treat": 1, "outcome": 15.0},
        {"id": 2, "year": 1, "treat": 0, "outcome": 12.0},
        {"id": 2, "year": 2, "treat": 0, "outcome": 13.0},
        {"id": 2, "year": 3, "treat": 1, "outcome": 17.0},
        {"id": 3, "year": 1, "treat": 0, "outcome": 7.0},
        {"id": 3, "year": 2, "treat": 0, "outcome": 8.0},
        {"id": 3, "year": 3, "treat": 0, "outcome": 9.0},
        {"id": 4, "year": 1, "treat": 0, "outcome": 8.0},
        {"id": 4, "year": 2, "treat": 0, "outcome": 9.0},
        {"id": 4, "year": 3, "treat": 0, "outcome": 10.0},
    ]

    with pytest.raises(ValueError, match="n_boot must be an integer greater than or equal to 2\\."):
        did_check(
            data=rows,
            outcome="outcome",
            treatment="treat",
            time="year",
            unit_id="id",
            lag=1,
            n_boot=bad_n_boot,
        )


@pytest.mark.parametrize(
    ("mutator", "error"),
    [
        (lambda rows: rows[1].update({"outcome": float("inf")}), "outcome.*finite numeric"),
        (lambda rows: rows[1].update({"x": float("inf")}), "covariate 'x'.*finite numeric"),
    ],
)
def test_data_driven_did_check_rejects_non_finite_observed_numeric_inputs_before_regression(mutator, error):
    from diddesign.diagnostics import did_check

    rows = [
        {"id": 1, "year": 1, "treat": 0, "outcome": 10.0, "x": 0.0},
        {"id": 1, "year": 2, "treat": 0, "outcome": 11.0, "x": 1.0},
        {"id": 1, "year": 3, "treat": 1, "outcome": 15.0, "x": 2.0},
        {"id": 2, "year": 1, "treat": 0, "outcome": 12.0, "x": 0.5},
        {"id": 2, "year": 2, "treat": 0, "outcome": 13.0, "x": 1.5},
        {"id": 2, "year": 3, "treat": 1, "outcome": 17.0, "x": 2.5},
        {"id": 3, "year": 1, "treat": 0, "outcome": 7.0, "x": 0.25},
        {"id": 3, "year": 2, "treat": 0, "outcome": 8.0, "x": 1.25},
        {"id": 3, "year": 3, "treat": 0, "outcome": 9.0, "x": 2.25},
        {"id": 4, "year": 1, "treat": 0, "outcome": 8.0, "x": 0.75},
        {"id": 4, "year": 2, "treat": 0, "outcome": 9.0, "x": 1.75},
        {"id": 4, "year": 3, "treat": 0, "outcome": 10.0, "x": 2.75},
    ]
    mutator(rows)

    with pytest.raises(ValueError, match=error):
        did_check(
            data=rows,
            outcome="outcome",
            treatment="treat",
            time="year",
            unit_id="id",
            covariates=("x",),
            lag=1,
            n_boot=4,
            random_seed=123,
        )


def test_data_driven_did_check_rejects_duplicate_lag_sequence_at_public_boundary():
    from diddesign.diagnostics import did_check

    with pytest.raises(ValueError, match="lag sequence must not contain duplicate values\\."):
        did_check(
            data=(),
            outcome="outcome",
            treatment="treat",
            time="year",
            unit_id="id",
            lag=(1, 1),
            n_boot=4,
            random_seed=123,
        )


@pytest.mark.parametrize("bad_seed", [True, False, np.bool_(True), np.bool_(False), 1.0, "1", -1, 2**32])
def test_data_driven_did_check_rejects_invalid_random_seed_values_before_sampling(bad_seed):
    from diddesign.diagnostics import did_check

    rows = [
        {"id": 1, "year": 1, "treat": 0, "outcome": 10.0},
        {"id": 1, "year": 2, "treat": 0, "outcome": 11.0},
        {"id": 1, "year": 3, "treat": 1, "outcome": 15.0},
        {"id": 2, "year": 1, "treat": 0, "outcome": 12.0},
        {"id": 2, "year": 2, "treat": 0, "outcome": 13.0},
        {"id": 2, "year": 3, "treat": 1, "outcome": 17.0},
        {"id": 3, "year": 1, "treat": 0, "outcome": 7.0},
        {"id": 3, "year": 2, "treat": 0, "outcome": 8.0},
        {"id": 3, "year": 3, "treat": 0, "outcome": 9.0},
        {"id": 4, "year": 1, "treat": 0, "outcome": 8.0},
        {"id": 4, "year": 2, "treat": 0, "outcome": 9.0},
        {"id": 4, "year": 3, "treat": 0, "outcome": 10.0},
    ]

    with pytest.raises(ValueError, match="random_seed"):
        did_check(
            data=rows,
            outcome="outcome",
            treatment="treat",
            time="year",
            unit_id="id",
            lag=1,
            n_boot=4,
            random_seed=bad_seed,
        )


def test_data_driven_did_check_normalizes_numpy_integer_random_seed_metadata():
    from diddesign.diagnostics import did_check

    rows = [
        {"id": 1, "year": 1, "treat": 0, "outcome": 10.0},
        {"id": 1, "year": 2, "treat": 0, "outcome": 11.0},
        {"id": 1, "year": 3, "treat": 1, "outcome": 15.0},
        {"id": 2, "year": 1, "treat": 0, "outcome": 12.0},
        {"id": 2, "year": 2, "treat": 0, "outcome": 13.0},
        {"id": 2, "year": 3, "treat": 1, "outcome": 17.0},
        {"id": 3, "year": 1, "treat": 0, "outcome": 7.0},
        {"id": 3, "year": 2, "treat": 0, "outcome": 8.0},
        {"id": 3, "year": 3, "treat": 0, "outcome": 9.0},
        {"id": 4, "year": 1, "treat": 0, "outcome": 8.0},
        {"id": 4, "year": 2, "treat": 0, "outcome": 9.0},
        {"id": 4, "year": 3, "treat": 0, "outcome": 10.0},
    ]

    result = did_check(
        data=rows,
        outcome="outcome",
        treatment="treat",
        time="year",
        unit_id="id",
        lag=1,
        n_boot=4,
        random_seed=np.int64(7),
    )

    assert result.metadata["random_seed"] == 7
    assert isinstance(result.metadata["random_seed"], int)


def test_data_driven_did_check_normalizes_numpy_integer_lag_and_bootstrap_metadata():
    from diddesign.diagnostics import did_check

    rows = [
        {"id": 1, "year": 1, "treat": 0, "outcome": 10.0},
        {"id": 1, "year": 2, "treat": 0, "outcome": 11.0},
        {"id": 1, "year": 3, "treat": 1, "outcome": 15.0},
        {"id": 2, "year": 1, "treat": 0, "outcome": 12.0},
        {"id": 2, "year": 2, "treat": 0, "outcome": 13.0},
        {"id": 2, "year": 3, "treat": 1, "outcome": 17.0},
        {"id": 3, "year": 1, "treat": 0, "outcome": 7.0},
        {"id": 3, "year": 2, "treat": 0, "outcome": 8.0},
        {"id": 3, "year": 3, "treat": 0, "outcome": 9.0},
        {"id": 4, "year": 1, "treat": 0, "outcome": 8.0},
        {"id": 4, "year": 2, "treat": 0, "outcome": 9.0},
        {"id": 4, "year": 3, "treat": 0, "outcome": 10.0},
    ]

    result = did_check(
        data=rows,
        outcome="outcome",
        treatment="treat",
        time="year",
        unit_id="id",
        lag=np.int64(1),
        n_boot=np.int64(4),
        random_seed=7,
    )

    assert result.metadata["requested_lags"] == (1,)
    assert all(isinstance(lag, int) for lag in result.metadata["requested_lags"])
    assert result.metadata["n_boot_requested"] == 4
    assert isinstance(result.metadata["n_boot_requested"], int)
    assert result.metadata["n_boot_realized"] == 4
    assert isinstance(result.metadata["n_boot_realized"], int)


def test_data_driven_did_check_rejects_malformed_covariate_terms_at_public_boundary():
    from diddesign.diagnostics import did_check

    rows = [
        {"id": 1, "year": 1, "treat": 0, "outcome": 10.0, "x": 1.0},
        {"id": 1, "year": 2, "treat": 0, "outcome": 11.0, "x": 1.1},
        {"id": 1, "year": 3, "treat": 1, "outcome": 15.0, "x": 1.2},
        {"id": 2, "year": 1, "treat": 0, "outcome": 12.0, "x": 0.9},
        {"id": 2, "year": 2, "treat": 0, "outcome": 13.0, "x": 1.0},
        {"id": 2, "year": 3, "treat": 1, "outcome": 17.0, "x": 1.1},
        {"id": 3, "year": 1, "treat": 0, "outcome": 7.0, "x": 0.7},
        {"id": 3, "year": 2, "treat": 0, "outcome": 8.0, "x": 0.8},
        {"id": 3, "year": 3, "treat": 0, "outcome": 9.0, "x": 0.9},
        {"id": 4, "year": 1, "treat": 0, "outcome": 8.0, "x": 0.8},
        {"id": 4, "year": 2, "treat": 0, "outcome": 9.0, "x": 0.9},
        {"id": 4, "year": 3, "treat": 0, "outcome": 10.0, "x": 1.0},
    ]

    with pytest.raises(ValueError, match="covariates cannot contain blank entries\\."):
        did_check(
            data=rows,
            outcome="outcome",
            treatment="treat",
            time="year",
            unit_id="id",
            covariates=[""],
            n_boot=4,
        )

    with pytest.raises(ValueError, match="factor\\(\\) covariate terms must name a column\\."):
        did_check(
            data=rows,
            outcome="outcome",
            treatment="treat",
            time="year",
            unit_id="id",
            covariates=["factor()"],
            n_boot=4,
        )

    for bad_covariate in ("factor(x + z)", "factor(x))"):
        with pytest.raises(
            ValueError,
            match=r"covariates must contain only column names or factor\(column\) terms\.",
        ):
            did_check(
                data=rows,
                outcome="outcome",
                treatment="treat",
                time="year",
                unit_id="id",
                covariates=[bad_covariate],
                n_boot=4,
            )

    with pytest.raises(TypeError, match=r"covariates must be a sequence of column names or factor\(\.\.\.\) terms\."):
        did_check(
            data=rows,
            outcome="outcome",
            treatment="treat",
            time="year",
            unit_id="id",
            covariates=[1],
            n_boot=4,
        )

    with pytest.raises(TypeError, match=r"covariates must be a sequence of column names or factor\(\.\.\.\) terms\."):
        did_check(
            data=rows,
            outcome="outcome",
            treatment="treat",
            time="year",
            unit_id="id",
            covariates="x",
            n_boot=4,
        )


def test_data_driven_did_check_rejects_duplicate_covariates_before_fitting_placebo_design():
    from diddesign.diagnostics import did_check

    rows = [
        {"id": 1, "year": 1, "treat": 0, "outcome": 10.0, "x": 1.0},
        {"id": 1, "year": 2, "treat": 0, "outcome": 11.0, "x": 1.1},
        {"id": 1, "year": 3, "treat": 1, "outcome": 15.0, "x": 1.2},
        {"id": 2, "year": 1, "treat": 0, "outcome": 12.0, "x": 0.9},
        {"id": 2, "year": 2, "treat": 0, "outcome": 13.0, "x": 1.0},
        {"id": 2, "year": 3, "treat": 1, "outcome": 17.0, "x": 1.1},
        {"id": 3, "year": 1, "treat": 0, "outcome": 7.0, "x": 0.7},
        {"id": 3, "year": 2, "treat": 0, "outcome": 8.0, "x": 0.8},
        {"id": 3, "year": 3, "treat": 0, "outcome": 9.0, "x": 0.9},
        {"id": 4, "year": 1, "treat": 0, "outcome": 8.0, "x": 0.8},
        {"id": 4, "year": 2, "treat": 0, "outcome": 9.0, "x": 0.9},
        {"id": 4, "year": 3, "treat": 0, "outcome": 10.0, "x": 1.0},
    ]

    for covariates in (("x", "x"), ("x", "factor(x)")):
        with pytest.raises(ValueError, match="covariates must not contain duplicate column names\\."):
            did_check(
                data=rows,
                outcome="outcome",
                treatment="treat",
                time="year",
                unit_id="id",
                covariates=covariates,
                n_boot=4,
                random_seed=123,
            )


@pytest.mark.parametrize("covariates", [("outcome",), ("treat",), ("year",), ("id",)])
def test_data_driven_did_check_rejects_covariates_that_reuse_design_role_columns(covariates):
    from diddesign.diagnostics import did_check

    rows = [
        {"id": 1, "year": 1, "treat": 0, "outcome": 10.0, "x": 1.0},
        {"id": 1, "year": 2, "treat": 0, "outcome": 11.0, "x": 1.1},
        {"id": 1, "year": 3, "treat": 1, "outcome": 15.0, "x": 1.2},
        {"id": 2, "year": 1, "treat": 0, "outcome": 12.0, "x": 0.9},
        {"id": 2, "year": 2, "treat": 0, "outcome": 13.0, "x": 1.0},
        {"id": 2, "year": 3, "treat": 1, "outcome": 17.0, "x": 1.1},
        {"id": 3, "year": 1, "treat": 0, "outcome": 7.0, "x": 0.7},
        {"id": 3, "year": 2, "treat": 0, "outcome": 8.0, "x": 0.8},
        {"id": 3, "year": 3, "treat": 0, "outcome": 9.0, "x": 0.9},
        {"id": 4, "year": 1, "treat": 0, "outcome": 8.0, "x": 0.8},
        {"id": 4, "year": 2, "treat": 0, "outcome": 9.0, "x": 0.9},
        {"id": 4, "year": 3, "treat": 0, "outcome": 10.0, "x": 1.0},
    ]

    with pytest.raises(
        ValueError,
        match="covariates must not reuse outcome, treatment, time, unit_id, or post columns\\.",
    ):
        did_check(
            data=rows,
            outcome="outcome",
            treatment="treat",
            time="year",
            unit_id="id",
            covariates=covariates,
            n_boot=4,
            random_seed=123,
        )
