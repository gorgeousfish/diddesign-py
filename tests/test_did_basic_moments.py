from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest


def _panel_rows() -> list[dict[str, object]]:
    return [
        {"unit": "a", "time": 0, "treat": 0, "outcome": 10.0},
        {"unit": "a", "time": 1, "treat": 0, "outcome": 11.0},
        {"unit": "a", "time": 2, "treat": 1, "outcome": 15.0},
        {"unit": "b", "time": 0, "treat": 0, "outcome": 12.0},
        {"unit": "b", "time": 1, "treat": 0, "outcome": 13.0},
        {"unit": "b", "time": 2, "treat": 1, "outcome": 15.0},
        {"unit": "c", "time": 0, "treat": 0, "outcome": 8.0},
        {"unit": "c", "time": 1, "treat": 0, "outcome": 9.0},
        {"unit": "c", "time": 2, "treat": 0, "outcome": 9.5},
        {"unit": "d", "time": 0, "treat": 0, "outcome": 9.0},
        {"unit": "d", "time": 1, "treat": 0, "outcome": 10.0},
        {"unit": "d", "time": 2, "treat": 0, "outcome": 11.5},
    ]


def _panel_rows_with_degenerate_bootstrap_vcov() -> list[dict[str, object]]:
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


def _panel_rows_with_dynamic_leads() -> list[dict[str, object]]:
    return [
        {"unit": "a", "time": 0, "treat": 0, "outcome": 10.0},
        {"unit": "a", "time": 1, "treat": 0, "outcome": 11.0},
        {"unit": "a", "time": 2, "treat": 1, "outcome": 14.0},
        {"unit": "a", "time": 3, "treat": 1, "outcome": 16.0},
        {"unit": "a", "time": 4, "treat": 1, "outcome": 19.0},
        {"unit": "b", "time": 0, "treat": 0, "outcome": 12.0},
        {"unit": "b", "time": 1, "treat": 0, "outcome": 14.0},
        {"unit": "b", "time": 2, "treat": 1, "outcome": 18.0},
        {"unit": "b", "time": 3, "treat": 1, "outcome": 20.0},
        {"unit": "b", "time": 4, "treat": 1, "outcome": 24.0},
        {"unit": "c", "time": 0, "treat": 0, "outcome": 8.0},
        {"unit": "c", "time": 1, "treat": 0, "outcome": 9.0},
        {"unit": "c", "time": 2, "treat": 0, "outcome": 10.0},
        {"unit": "c", "time": 3, "treat": 0, "outcome": 10.5},
        {"unit": "c", "time": 4, "treat": 0, "outcome": 12.5},
        {"unit": "d", "time": 0, "treat": 0, "outcome": 9.0},
        {"unit": "d", "time": 1, "treat": 0, "outcome": 10.0},
        {"unit": "d", "time": 2, "treat": 0, "outcome": 11.0},
        {"unit": "d", "time": 3, "treat": 0, "outcome": 13.0},
        {"unit": "d", "time": 4, "treat": 0, "outcome": 13.5},
    ]


def _panel_rows_with_identifiable_covariate() -> list[dict[str, object]]:
    return [
        {"unit": "a", "time": 0, "treat": 0, "outcome": 10.0, "x": 0.0},
        {"unit": "a", "time": 1, "treat": 0, "outcome": 11.0, "x": 0.2},
        {"unit": "a", "time": 2, "treat": 1, "outcome": 14.0, "x": 0.4},
        {"unit": "a", "time": 3, "treat": 1, "outcome": 16.0, "x": 0.6},
        {"unit": "b", "time": 0, "treat": 0, "outcome": 12.0, "x": 1.0},
        {"unit": "b", "time": 1, "treat": 0, "outcome": 14.0, "x": 1.2},
        {"unit": "b", "time": 2, "treat": 1, "outcome": 18.0, "x": 1.4},
        {"unit": "b", "time": 3, "treat": 1, "outcome": 20.0, "x": 1.6},
        {"unit": "c", "time": 0, "treat": 0, "outcome": 8.0, "x": 2.0},
        {"unit": "c", "time": 1, "treat": 0, "outcome": 9.0, "x": 2.2},
        {"unit": "c", "time": 2, "treat": 0, "outcome": 10.0, "x": 2.4},
        {"unit": "c", "time": 3, "treat": 0, "outcome": 10.5, "x": 2.6},
        {"unit": "d", "time": 0, "treat": 0, "outcome": 9.0, "x": 3.0},
        {"unit": "d", "time": 1, "treat": 0, "outcome": 10.0, "x": 3.2},
        {"unit": "d", "time": 2, "treat": 0, "outcome": 11.0, "x": 3.4},
        {"unit": "d", "time": 3, "treat": 0, "outcome": 13.0, "x": 3.6},
        {"unit": "e", "time": 0, "treat": 0, "outcome": 7.0, "x": 4.0},
        {"unit": "e", "time": 1, "treat": 0, "outcome": 8.5, "x": 4.2},
        {"unit": "e", "time": 2, "treat": 0, "outcome": 9.5, "x": 4.4},
        {"unit": "e", "time": 3, "treat": 0, "outcome": 10.5, "x": 4.6},
        {"unit": "f", "time": 0, "treat": 0, "outcome": 11.0, "x": 5.0},
        {"unit": "f", "time": 1, "treat": 0, "outcome": 12.0, "x": 5.2},
        {"unit": "f", "time": 2, "treat": 0, "outcome": 12.5, "x": 5.4},
        {"unit": "f", "time": 3, "treat": 0, "outcome": 13.2, "x": 5.6},
    ]


def _panel_rows_with_non_singular_weights() -> list[dict[str, object]]:
    return [
        {"unit": "a", "time": 0, "treat": 0, "outcome": 10.0},
        {"unit": "a", "time": 1, "treat": 0, "outcome": 11.0},
        {"unit": "a", "time": 2, "treat": 1, "outcome": 14.0},
        {"unit": "b", "time": 0, "treat": 0, "outcome": 13.0},
        {"unit": "b", "time": 1, "treat": 0, "outcome": 15.0},
        {"unit": "b", "time": 2, "treat": 1, "outcome": 19.0},
        {"unit": "c", "time": 0, "treat": 0, "outcome": 8.0},
        {"unit": "c", "time": 1, "treat": 0, "outcome": 8.5},
        {"unit": "c", "time": 2, "treat": 0, "outcome": 9.5},
        {"unit": "d", "time": 0, "treat": 0, "outcome": 9.0},
        {"unit": "d", "time": 1, "treat": 0, "outcome": 10.5},
        {"unit": "d", "time": 2, "treat": 0, "outcome": 11.0},
    ]


def _panel_rows_where_unit_and_group_lags_diverge() -> list[dict[str, object]]:
    return [
        {"unit": "a", "time": 0, "treat": 0, "outcome": 0.0},
        {"unit": "a", "time": 1, "treat": 0, "outcome": 10.0},
        {"unit": "a", "time": 2, "treat": 1, "outcome": 30.0},
        {"unit": "b", "time": 0, "treat": 0, "outcome": 100.0},
        {"unit": "b", "time": 1, "treat": 0, "outcome": 105.0},
        {"unit": "b", "time": 2, "treat": 1, "outcome": 120.0},
        {"unit": "c", "time": 0, "treat": 0, "outcome": 0.0},
        {"unit": "c", "time": 1, "treat": 0, "outcome": 4.0},
        {"unit": "c", "time": 2, "treat": 0, "outcome": 8.0},
        {"unit": "d", "time": 0, "treat": 0, "outcome": 20.0},
        {"unit": "d", "time": 1, "treat": 0, "outcome": 21.0},
        {"unit": "d", "time": 2, "treat": 0, "outcome": 23.0},
    ]


def _panel_rows_with_missing_outcome() -> list[dict[str, object]]:
    return [
        {"unit": "a", "time": 0, "treat": 0, "outcome": 10.0},
        {"unit": "a", "time": 1, "treat": 0, "outcome": 11.0},
        {"unit": "a", "time": 2, "treat": 1, "outcome": None},
        {"unit": "b", "time": 0, "treat": 0, "outcome": 13.0},
        {"unit": "b", "time": 1, "treat": 0, "outcome": 15.0},
        {"unit": "b", "time": 2, "treat": 1, "outcome": 19.0},
        {"unit": "c", "time": 0, "treat": 0, "outcome": 8.0},
        {"unit": "c", "time": 1, "treat": 0, "outcome": 8.5},
        {"unit": "c", "time": 2, "treat": 0, "outcome": 9.5},
        {"unit": "d", "time": 0, "treat": 0, "outcome": 9.0},
        {"unit": "d", "time": 1, "treat": 0, "outcome": 10.5},
        {"unit": "d", "time": 2, "treat": 0, "outcome": 11.0},
    ]


def _rcs_rows() -> list[dict[str, object]]:
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


def _rcs_rows_with_factor_covariate() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    cluster_specs = (
        ("north", "urban", 0.0),
        ("south", "rural", 0.3),
        ("east", "urban", -0.2),
        ("west", "rural", 0.5),
    )
    for current_time, post in ((0, 0), (1, 0), (2, 1)):
        for cluster, region, baseline in cluster_specs:
            for treat_group in (0, 1):
                region_effect = 0.4 if region == "urban" else -0.1
                treatment_effect = 1.25 if treat_group and post else 0.0
                rows.append(
                    {
                        "cluster": cluster,
                        "region": region,
                        "time": current_time,
                        "treat_group": treat_group,
                        "post": post,
                        "outcome": 1.0
                        + baseline
                        + current_time
                        + 0.5 * treat_group
                        + region_effect
                        + treatment_effect,
                    }
                )
    return rows


def _rcs_rows_with_saturated_cell_means() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    cell_means = {
        (0, 0): 1.0,
        (1, 0): 2.0,
        (0, 1): 2.0,
        (1, 1): 4.0,
        (0, 2): 3.0,
        (1, 2): 7.0,
    }
    for (treat_group, time), outcome in cell_means.items():
        for index in range(2):
            rows.append(
                {
                    "cluster": f"{treat_group}-{time}-{index}",
                    "time": time,
                    "treat_group": treat_group,
                    "post": int(time == 2),
                    "outcome": outcome + index * 0.1,
                }
            )
    return rows


def test_panel_did_emits_double_did_when_unit_bootstrap_weights_are_identifiable():
    from diddesign.estimators import did

    result = did(
        _panel_rows_with_non_singular_weights(),
        outcome="outcome",
        treatment="treat",
        time="time",
        unit_id="unit",
        n_boot=6,
        random_seed=1234,
    )

    rows = result.estimate_rows()

    assert [row.estimator for row in rows] == ["Double-DID", "DID", "sDID"]
    assert [row.lead for row in rows] == [0, 0, 0]
    assert rows[0].estimate == pytest.approx(rows[1].weight * rows[1].estimate + rows[2].weight * rows[2].estimate)
    assert rows[1].estimate == pytest.approx(2.75)
    assert rows[2].estimate == pytest.approx(2.25)
    assert rows[1].weight + rows[2].weight == pytest.approx(1.0)
    assert result.metadata["data_type"] == "panel"
    assert result.metadata["requested_leads"] == (0,)
    assert result.metadata["identified_leads"] == (0,)
    assert result.metadata["cluster_column"] == "unit"
    assert result.metadata["cluster_mode"] == "unit"
    assert result.metadata["double_did_available"] is True
    assert result.metadata["n_boot_requested"] == 6
    assert result.metadata["n_boot_realized"] == 6


def test_panel_sdid_transformed_outcome_uses_group_specific_lagged_means():
    from diddesign.estimators.did import _parse_covariates, _prepare_frame

    frame, _ = _prepare_frame(
        _panel_rows_where_unit_and_group_lags_diverge(),
        outcome="outcome",
        treatment="treat",
        time="time",
        unit_id="unit",
        post=None,
        covariates=_parse_covariates(None),
        data_type="panel",
        id_cluster=None,
    )

    by_unit_time = frame.set_index(["unit", "time"])[["Ymean", "outcome_delta"]]

    assert by_unit_time.loc[("a", 1), "Ymean"] == pytest.approx(50.0)
    assert by_unit_time.loc[("b", 1), "Ymean"] == pytest.approx(50.0)
    assert by_unit_time.loc[("a", 1), "outcome_delta"] == pytest.approx(-40.0)
    assert by_unit_time.loc[("b", 1), "outcome_delta"] == pytest.approx(55.0)


def test_panel_bootstrap_relabels_duplicate_unit_copies_before_group_lagged_means(monkeypatch):
    import importlib

    did_module = importlib.import_module("diddesign.estimators.did")
    frame, _ = did_module._prepare_frame(
        _panel_rows_where_unit_and_group_lags_diverge(),
        outcome="outcome",
        treatment="treat",
        time="time",
        unit_id="unit",
        post=None,
        covariates=did_module._parse_covariates(None),
        data_type="panel",
        id_cluster=None,
    )

    class DuplicateUnitRng:
        def randint(self, low, high=None, size=None):
            assert low == 0
            assert high == 4
            assert size == 4
            return np.asarray([0, 0, 2, 2], dtype=int)

    inspected_boot_frames = []

    def inspect_bootstrap_frame(boot_frame, *, lead, covariates):
        assert lead == 0
        assert covariates == ()
        if not inspected_boot_frames:
            inspected_boot_frames.append(boot_frame.copy())
        return {"DID": 1.0, "sDID": 2.0}

    monkeypatch.setattr(did_module.np.random, "RandomState", lambda seed: DuplicateUnitRng())
    monkeypatch.setattr(did_module, "_compute_component_estimates", inspect_bootstrap_frame)

    draws = did_module._compute_bootstrap_draws(
        frame,
        leads=(0,),
        covariates=(),
        n_boot=2,
        random_seed=20260519,
    )

    assert [(draw.iteration, draw.did, draw.sdid) for draw in draws] == [
        (1, 1.0, 2.0),
        (2, 1.0, 2.0),
    ]
    boot_frame = inspected_boot_frames[0]
    assert len(boot_frame) == 12
    assert set(boot_frame["_delta_unit_id"]) == {
        (1, "a"),
        (2, "a"),
        (3, "c"),
        (4, "c"),
    }

    by_copy_time = boot_frame.set_index(["_delta_unit_id", "time"])["outcome_delta"]
    assert by_copy_time[((1, "a"), 1)] == pytest.approx(10.0)
    assert by_copy_time[((2, "a"), 1)] == pytest.approx(10.0)
    assert by_copy_time[((1, "a"), 2)] == pytest.approx(20.0)
    assert by_copy_time[((2, "a"), 2)] == pytest.approx(20.0)
    assert by_copy_time[((3, "c"), 1)] == pytest.approx(4.0)
    assert by_copy_time[((4, "c"), 1)] == pytest.approx(4.0)


def test_panel_explicit_cluster_bootstrap_relabels_units_inside_repeated_blocks(monkeypatch):
    import importlib

    did_module = importlib.import_module("diddesign.estimators.did")
    rows = []
    for row in _panel_rows_where_unit_and_group_lags_diverge():
        cluster = "g1" if row["unit"] in {"a", "b"} else "g2"
        rows.append({**row, "cluster": cluster})

    frame, metadata = did_module._prepare_frame(
        rows,
        outcome="outcome",
        treatment="treat",
        time="time",
        unit_id="unit",
        post=None,
        covariates=did_module._parse_covariates(None),
        data_type="panel",
        id_cluster="cluster",
    )

    class DuplicateClusterRng:
        def randint(self, low, high=None, size=None):
            assert low == 0
            assert high == 2
            assert size == 2
            return np.asarray([0, 0], dtype=int)

    inspected_boot_frames = []

    def inspect_bootstrap_frame(boot_frame, *, lead, covariates):
        assert lead == 0
        assert covariates == ()
        if not inspected_boot_frames:
            inspected_boot_frames.append(boot_frame.copy())
        return {"DID": 1.0, "sDID": 2.0}

    monkeypatch.setattr(did_module.np.random, "RandomState", lambda seed: DuplicateClusterRng())
    monkeypatch.setattr(did_module, "_compute_component_estimates", inspect_bootstrap_frame)

    draws = did_module._compute_bootstrap_draws(
        frame,
        leads=(0,),
        covariates=(),
        n_boot=2,
        random_seed=20260520,
    )

    assert metadata["cluster_mode"] == "explicit"
    assert [(draw.iteration, draw.did, draw.sdid) for draw in draws] == [
        (1, 1.0, 2.0),
        (2, 1.0, 2.0),
    ]
    boot_frame = inspected_boot_frames[0]
    assert len(boot_frame) == 12
    assert set(boot_frame["_delta_unit_id"]) == {
        (1, "a"),
        (1, "b"),
        (2, "a"),
        (2, "b"),
    }
    assert not boot_frame.duplicated(["_delta_unit_id", "time"]).any()

    by_copy_time = boot_frame.set_index(["_delta_unit_id", "time"])["outcome_delta"]
    assert by_copy_time[((1, "a"), 1)] == pytest.approx(-40.0)
    assert by_copy_time[((2, "a"), 1)] == pytest.approx(-40.0)
    assert by_copy_time[((1, "b"), 1)] == pytest.approx(55.0)
    assert by_copy_time[((2, "b"), 1)] == pytest.approx(55.0)
    assert by_copy_time[((1, "a"), 2)] == pytest.approx(-27.5)
    assert by_copy_time[((2, "a"), 2)] == pytest.approx(-27.5)
    assert by_copy_time[((1, "b"), 2)] == pytest.approx(62.5)
    assert by_copy_time[((2, "b"), 2)] == pytest.approx(62.5)


def test_panel_bootstrap_discards_whole_iteration_when_any_requested_lead_fails(monkeypatch):
    import importlib

    did_module = importlib.import_module("diddesign.estimators.did")
    frame, _ = did_module._prepare_frame(
        _panel_rows_with_dynamic_leads(),
        outcome="outcome",
        treatment="treat",
        time="time",
        unit_id="unit",
        post=None,
        covariates=did_module._parse_covariates(None),
        data_type="panel",
        id_cluster=None,
    )

    class StableRng:
        def randint(self, low, high=None, size=None):
            assert low == 0
            assert high == 4
            assert size == 4
            return np.asarray([0, 1, 2, 3], dtype=int)

    current_attempt = 0
    inspected_calls = []

    def sometimes_unidentified_lead(boot_frame, *, lead, covariates):
        nonlocal current_attempt
        assert covariates == ()
        if lead == 0:
            current_attempt += 1
        inspected_calls.append((current_attempt, lead))
        if current_attempt == 1 and lead == 1:
            raise ValueError("lead 1 is unidentified in this bootstrap iteration")
        return {
            "DID": 100.0 + 10.0 * current_attempt + lead,
            "sDID": 200.0 + 10.0 * current_attempt + lead,
        }

    monkeypatch.setattr(did_module.np.random, "RandomState", lambda seed: StableRng())
    monkeypatch.setattr(did_module, "_compute_component_estimates", sometimes_unidentified_lead)

    draws = did_module._compute_bootstrap_draws(
        frame,
        leads=(0, 1),
        covariates=(),
        n_boot=2,
        random_seed=20260520,
    )

    assert inspected_calls == [(1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]
    assert [(draw.iteration, draw.lead, draw.did, draw.sdid) for draw in draws] == [
        (1, 0, 120.0, 220.0),
        (1, 1, 121.0, 221.0),
        (2, 0, 130.0, 230.0),
        (2, 1, 131.0, 231.0),
    ]


def test_panel_did_identifies_saturated_two_by_two_moment_regressions():
    from diddesign.estimators import did

    rows = [
        {"unit": "a", "time": 0, "treat": 0, "outcome": 1.0},
        {"unit": "a", "time": 1, "treat": 0, "outcome": 2.0},
        {"unit": "a", "time": 2, "treat": 1, "outcome": 5.0},
        {"unit": "b", "time": 0, "treat": 0, "outcome": 1.0},
        {"unit": "b", "time": 1, "treat": 0, "outcome": 2.0},
        {"unit": "b", "time": 2, "treat": 0, "outcome": 3.0},
    ]

    result = did(
        rows,
        outcome="outcome",
        treatment="treat",
        time="time",
        unit_id="unit",
        n_boot=4,
        random_seed=7,
    )

    rows_by_estimator = {row.estimator: row for row in result.estimate_rows()}
    assert rows_by_estimator["DID"].estimate == pytest.approx(2.0)
    assert rows_by_estimator["sDID"].estimate == pytest.approx(2.0)
    assert result.metadata["identified_leads"] == (0,)


def test_panel_did_retains_missing_outcome_rows_for_model_listwise_deletion():
    from diddesign.estimators import did

    result = did(
        _panel_rows_with_missing_outcome(),
        outcome="outcome",
        treatment="treat",
        time="time",
        unit_id="unit",
        n_boot=6,
        random_seed=1234,
    )

    rows = result.estimate_rows()
    rows_by_estimator = {row.estimator: row for row in rows}

    assert rows_by_estimator["DID"].estimate == pytest.approx(5.25)
    assert rows_by_estimator["sDID"].estimate == pytest.approx(4.75)
    if "Double-DID" in rows_by_estimator:
        assert rows_by_estimator["Double-DID"].estimate == pytest.approx(
            rows_by_estimator["DID"].weight * rows_by_estimator["DID"].estimate
            + rows_by_estimator["sDID"].weight * rows_by_estimator["sDID"].estimate
        )
    assert result.metadata["time_order"] == (0, 1, 2)
    assert result.metadata["n_boot_realized"] == 6


def test_rcs_did_uses_observation_fallback_cluster_when_no_cluster_is_supplied():
    from diddesign.estimators import did

    result = did(
        _rcs_rows(),
        outcome="outcome",
        treatment="treat_group",
        time="time",
        post="post",
        data_type="rcs",
        n_boot=5,
        random_seed=99,
    )

    rows = result.estimate_rows()

    assert [row.estimator for row in rows] == ["Double-DID", "DID", "sDID"]
    assert rows[0].estimate == pytest.approx(rows[1].weight * rows[1].estimate + rows[2].weight * rows[2].estimate)
    assert rows[1].estimate == pytest.approx(1.5)
    assert rows[2].estimate == pytest.approx(1.5)
    assert rows[1].weight + rows[2].weight == pytest.approx(1.0)
    assert result.metadata["data_type"] == "rcs"
    assert result.metadata["cluster_column"] == "_observation"
    assert result.metadata["cluster_mode"] == "observation"
    assert result.metadata["n_boot_requested"] == 5
    assert result.metadata["n_boot_realized"] == 5


def test_rcs_sdid_uses_group_specific_lagged_means_for_transformed_outcomes():
    from diddesign.estimators import did

    result = did(
        _rcs_rows_with_saturated_cell_means(),
        outcome="outcome",
        treatment="treat_group",
        time="time",
        post="post",
        data_type="rcs",
        n_boot=4,
        random_seed=1,
    )

    rows_by_estimator = {row.estimator: row for row in result.estimate_rows()}

    assert rows_by_estimator["DID"].estimate == pytest.approx(2.0)
    assert rows_by_estimator["sDID"].estimate == pytest.approx(1.0)
    assert result.metadata["data_type"] == "rcs"


def test_rcs_explicit_cluster_bootstrap_keeps_group_mean_deltas_without_unit_relabeling(monkeypatch):
    import importlib

    did_module = importlib.import_module("diddesign.estimators.did")
    rows = []
    cell_values = {
        "c1": {
            (0, 0): 0.0,
            (1, 0): 10.0,
            (0, 1): 2.0,
            (1, 1): 14.0,
            (0, 2): 6.0,
            (1, 2): 25.0,
        },
        "c2": {
            (0, 0): 100.0,
            (1, 0): 110.0,
            (0, 1): 104.0,
            (1, 1): 118.0,
            (0, 2): 111.0,
            (1, 2): 135.0,
        },
        "c3": {
            (0, 0): 300.0,
            (1, 0): 310.0,
            (0, 1): 306.0,
            (1, 1): 320.0,
            (0, 2): 315.0,
            (1, 2): 345.0,
        },
    }
    for cluster, values in cell_values.items():
        for (treat_group, time), outcome in values.items():
            rows.append(
                {
                    "cluster": cluster,
                    "time": time,
                    "treat_group": treat_group,
                    "post": int(time == 2),
                    "outcome": outcome,
                }
            )

    frame, metadata = did_module._prepare_frame(
        rows,
        outcome="outcome",
        treatment="treat_group",
        time="time",
        unit_id=None,
        post="post",
        covariates=did_module._parse_covariates(None),
        data_type="rcs",
        id_cluster="cluster",
    )

    class DuplicateClusterRng:
        def randint(self, low, high=None, size=None):
            assert low == 0
            assert high == 3
            assert size == 3
            return np.asarray([0, 0, 1], dtype=int)

    inspected_boot_frames = []

    def inspect_bootstrap_frame(boot_frame, *, lead, covariates):
        assert lead == 0
        assert covariates == ()
        if not inspected_boot_frames:
            inspected_boot_frames.append(boot_frame.copy())
        return {"DID": 1.0, "sDID": 2.0}

    monkeypatch.setattr(did_module.np.random, "RandomState", lambda seed: DuplicateClusterRng())
    monkeypatch.setattr(did_module, "_compute_component_estimates", inspect_bootstrap_frame)

    draws = did_module._compute_bootstrap_draws(
        frame,
        leads=(0,),
        covariates=(),
        n_boot=2,
        random_seed=20260520,
    )

    assert metadata["cluster_mode"] == "explicit"
    assert metadata["data_type"] == "rcs"
    assert [(draw.iteration, draw.did, draw.sdid) for draw in draws] == [
        (1, 1.0, 2.0),
        (2, 1.0, 2.0),
    ]
    boot_frame = inspected_boot_frames[0]
    assert "_delta_unit_id" not in boot_frame
    assert len(boot_frame) == 18
    # String cluster_ids are factorized to integers: c1->0, c2->1, c3->2
    assert boot_frame["cluster_id"].tolist().count(0) == 12
    assert boot_frame["cluster_id"].tolist().count(1) == 6

    gi0_time1 = boot_frame.loc[
        (boot_frame["Gi"] == 0.0)
        & (boot_frame["time"] == 1)
        & (boot_frame["cluster_id"] == 0),
        ["Ymean", "outcome_delta"],
    ]
    assert gi0_time1["Ymean"].tolist() == pytest.approx([100.0 / 3.0, 100.0 / 3.0])
    assert gi0_time1["outcome_delta"].tolist() == pytest.approx([2.0 - 100.0 / 3.0] * 2)

    gi1_time1 = boot_frame.loc[
        (boot_frame["Gi"] == 1.0)
        & (boot_frame["time"] == 1)
        & (boot_frame["cluster_id"] == 1),
        ["Ymean", "outcome_delta"],
    ]
    assert gi1_time1["Ymean"].tolist() == pytest.approx([130.0 / 3.0])
    assert gi1_time1["outcome_delta"].tolist() == pytest.approx([118.0 - 130.0 / 3.0])


def test_panel_did_accepts_formula_string_entrypoint_without_changing_estimates():
    from diddesign.estimators import did

    manual = did(
        _panel_rows_with_identifiable_covariate(),
        outcome="outcome",
        treatment="treat",
        time="time",
        unit_id="unit",
        covariates=("x",),
        n_boot=6,
        random_seed=1234,
    )
    via_formula = did(
        _panel_rows_with_identifiable_covariate(),
        formula="outcome ~ treat | x",
        time="time",
        unit_id="unit",
        n_boot=6,
        random_seed=1234,
    )

    assert [(row.estimator, row.lead) for row in via_formula.estimate_rows()] == [
        (row.estimator, row.lead) for row in manual.estimate_rows()
    ]
    assert [
        (row.estimate, row.std_error, row.ci_lo, row.ci_hi, row.weight)
        for row in via_formula.estimate_rows()
    ] == pytest.approx(
        [
            (row.estimate, row.std_error, row.ci_lo, row.ci_hi, row.weight)
            for row in manual.estimate_rows()
        ]
    )
    assert via_formula.metadata["data_type"] == "panel"
    assert via_formula.metadata["cluster_column"] == "unit"


def test_panel_did_accepts_valid_formula_spec_object_without_changing_estimates():
    from diddesign import did_formula
    from diddesign.estimators import did

    manual = did(
        _panel_rows_with_identifiable_covariate(),
        outcome="outcome",
        treatment="treat",
        time="time",
        unit_id="unit",
        covariates=("x",),
        n_boot=6,
        random_seed=1234,
    )
    via_formula_spec = did(
        _panel_rows_with_identifiable_covariate(),
        formula=did_formula("outcome ~ treat | x", is_panel=True),
        time="time",
        unit_id="unit",
        n_boot=6,
        random_seed=1234,
    )

    assert [(row.estimator, row.lead) for row in via_formula_spec.estimate_rows()] == [
        (row.estimator, row.lead) for row in manual.estimate_rows()
    ]
    assert [
        (row.estimate, row.std_error, row.ci_lo, row.ci_hi, row.weight)
        for row in via_formula_spec.estimate_rows()
    ] == pytest.approx(
        [
            (row.estimate, row.std_error, row.ci_lo, row.ci_hi, row.weight)
            for row in manual.estimate_rows()
        ]
    )


def test_rcs_did_accepts_formula_string_entrypoint_and_infers_post_column():
    from diddesign.estimators import did

    manual = did(
        _rcs_rows(),
        outcome="outcome",
        treatment="treat_group",
        time="time",
        post="post",
        data_type="rcs",
        n_boot=5,
        random_seed=99,
    )
    via_formula = did(
        _rcs_rows(),
        formula="outcome ~ treat_group + post",
        time="time",
        data_type="rcs",
        n_boot=5,
        random_seed=99,
    )

    assert [(row.estimator, row.lead) for row in via_formula.estimate_rows()] == [
        (row.estimator, row.lead) for row in manual.estimate_rows()
    ]
    assert [
        (row.estimate, row.std_error, row.ci_lo, row.ci_hi, row.weight)
        for row in via_formula.estimate_rows()
    ] == pytest.approx(
        [
            (row.estimate, row.std_error, row.ci_lo, row.ci_hi, row.weight)
            for row in manual.estimate_rows()
        ]
    )
    assert via_formula.metadata["data_type"] == "rcs"
    assert via_formula.metadata["cluster_column"] == "_observation"


def test_rcs_did_is_invariant_to_lexically_ordered_string_time_relabeling():
    from diddesign.estimators import did

    rows = _rcs_rows()
    relabeled_rows = []
    time_relabel = {
        current_time: f"period_{index:02d}"
        for index, current_time in enumerate(sorted({row["time"] for row in rows}))
    }
    for row in rows:
        relabeled_row = dict(row)
        relabeled_row["time"] = time_relabel[row["time"]]
        relabeled_rows.append(relabeled_row)

    base = did(
        rows,
        outcome="outcome",
        treatment="treat_group",
        time="time",
        post="post",
        data_type="rcs",
        id_cluster="cluster",
        n_boot=5,
        random_seed=99,
    )
    relabeled = did(
        relabeled_rows,
        outcome="outcome",
        treatment="treat_group",
        time="time",
        post="post",
        data_type="rcs",
        id_cluster="cluster",
        n_boot=5,
        random_seed=99,
    )

    assert relabeled.metadata["time_order"] == (
        "period_00",
        "period_01",
        "period_02",
    )
    assert "time-order:string" in relabeled.metadata["validation_trace"]
    assert base.metadata["identified_leads"] == relabeled.metadata["identified_leads"]
    assert base.metadata["filtered_leads"] == relabeled.metadata["filtered_leads"]
    assert base.metadata["double_did_available_leads"] == relabeled.metadata["double_did_available_leads"]

    for base_row, relabeled_row in zip(
        base.estimate_rows(),
        relabeled.estimate_rows(),
        strict=True,
    ):
        assert (base_row.estimator, base_row.lead) == (
            relabeled_row.estimator,
            relabeled_row.lead,
        )
        assert relabeled_row.estimate == pytest.approx(base_row.estimate)
        if base_row.weight is None:
            assert relabeled_row.weight is None
        else:
            assert relabeled_row.weight == pytest.approx(base_row.weight)

    pd.testing.assert_frame_equal(
        base.to_bootstrap_frame(),
        relabeled.to_bootstrap_frame(),
    )


def test_rcs_did_factor_covariates_preserve_formula_and_cluster_contracts():
    from diddesign.estimators import did

    rows = _rcs_rows_with_factor_covariate()
    manual = did(
        rows,
        outcome="outcome",
        treatment="treat_group",
        time="time",
        post="post",
        data_type="rcs",
        covariates=("factor(region)",),
        id_cluster="cluster",
        n_boot=6,
        random_seed=20260518,
    )
    via_formula = did(
        rows,
        formula="outcome ~ treat_group + post | factor(region)",
        time="time",
        data_type="rcs",
        id_cluster="cluster",
        n_boot=6,
        random_seed=20260518,
    )

    assert [(row.estimator, row.lead) for row in via_formula.estimate_rows()] == [
        (row.estimator, row.lead) for row in manual.estimate_rows()
    ]
    assert [
        (row.estimate, row.std_error, row.ci_lo, row.ci_hi, row.weight)
        for row in via_formula.estimate_rows()
    ] == pytest.approx(
        [
            (row.estimate, row.std_error, row.ci_lo, row.ci_hi, row.weight)
            for row in manual.estimate_rows()
        ]
    )
    assert via_formula.metadata["data_type"] == "rcs"
    assert via_formula.metadata["covariates"] == ("factor(region)",)
    assert via_formula.metadata["cluster_column"] == "cluster"
    assert via_formula.metadata["cluster_mode"] == "explicit"
    assert via_formula.metadata["n_clusters"] == 4
    assert via_formula.metadata["n_boot_realized"] == 6
    assert via_formula.to_bootstrap_frame()["iteration"].nunique() == 6


def test_did_rejects_empty_lead_sequence_at_public_boundary():
    from diddesign.estimators import did

    with pytest.raises(ValueError, match="lead must contain at least one non-negative lead\\."):
        did(
            _panel_rows_with_non_singular_weights(),
            outcome="outcome",
            treatment="treat",
            time="time",
            unit_id="unit",
            lead=[],
            n_boot=4,
            random_seed=1234,
        )


def test_did_rejects_duplicate_lead_sequence_at_public_boundary():
    from diddesign.estimators import did

    with pytest.raises(ValueError, match="lead sequence must not contain duplicate values\\."):
        did(
            _panel_rows_with_non_singular_weights(),
            outcome="outcome",
            treatment="treat",
            time="time",
            unit_id="unit",
            lead=(0, 0),
            n_boot=4,
            random_seed=1234,
        )


def test_rcs_did_accepts_r_style_option_and_is_panel_surface_without_changing_estimates():
    from diddesign.estimators import did

    manual = did(
        _rcs_rows(),
        outcome="outcome",
        treatment="treat_group",
        time="time",
        post="post",
        data_type="rcs",
        n_boot=5,
        id_cluster="cluster",
        random_seed=99,
    )
    via_option = did(
        _rcs_rows(),
        formula="outcome ~ treat_group + post",
        time="time",
        is_panel=False,
        option={"n_boot": 5, "id_cluster": "cluster"},
        random_seed=99,
    )

    assert [(row.estimator, row.lead) for row in via_option.estimate_rows()] == [
        (row.estimator, row.lead) for row in manual.estimate_rows()
    ]
    assert [
        (row.estimate, row.std_error, row.ci_lo, row.ci_hi, row.weight)
        for row in via_option.estimate_rows()
    ] == pytest.approx(
        [
            (row.estimate, row.std_error, row.ci_lo, row.ci_hi, row.weight)
            for row in manual.estimate_rows()
        ]
    )
    assert via_option.metadata["data_type"] == "rcs"
    assert via_option.metadata["cluster_column"] == "cluster"
    assert via_option.metadata["n_boot_requested"] == 5


def test_did_accepts_numpy_boolean_is_panel_scalar_at_public_boundary():
    from diddesign.estimators import did

    result = did(
        _rcs_rows(),
        formula="outcome ~ treat_group + post",
        time="time",
        is_panel=np.bool_(False),
        option={"n_boot": 5, "id_cluster": "cluster"},
        random_seed=99,
    )

    assert result.metadata["data_type"] == "rcs"
    assert result.metadata["cluster_column"] == "cluster"


def test_did_rejects_non_boolean_is_panel_values_at_public_boundary():
    from diddesign.estimators import did

    for bad_value in (1, 0, "false"):
        with pytest.raises(ValueError, match="is_panel must be a boolean when provided\\."):
            did(
                _panel_rows(),
                outcome="outcome",
                treatment="treat",
                time="time",
                unit_id="unit",
                is_panel=bad_value,
            )


def test_did_rejects_invalid_id_cluster_values_at_public_boundary():
    from diddesign.estimators import did

    with pytest.raises(ValueError, match="id_cluster must be a column name string when provided\\."):
        did(
            _panel_rows(),
            outcome="outcome",
            treatment="treat",
            time="time",
            unit_id="unit",
            id_cluster=1,
        )

    for bad_value in ("", "   "):
        with pytest.raises(ValueError, match="id_cluster must be a non-empty column name\\."):
            did(
                _panel_rows(),
                outcome="outcome",
                treatment="treat",
                time="time",
                unit_id="unit",
                option={"id_cluster": bad_value},
            )


@pytest.mark.parametrize("id_cluster", ["outcome", "treat", "time"])
def test_did_rejects_id_cluster_that_reuses_estimand_role_columns(id_cluster):
    from diddesign.estimators import did

    with pytest.raises(
        ValueError,
        match="id_cluster must not reuse outcome, treatment, time, or post columns\\.",
    ):
        did(
            _panel_rows(),
            outcome="outcome",
            treatment="treat",
            time="time",
            unit_id="unit",
            id_cluster=id_cluster,
            n_boot=2,
        )


def test_did_allows_explicit_id_cluster_matching_panel_unit_identifier():
    from diddesign.estimators import did

    result = did(
        _panel_rows(),
        outcome="outcome",
        treatment="treat",
        time="time",
        unit_id="unit",
        id_cluster="unit",
        n_boot=4,
        random_seed=20260518,
    )

    assert result.metadata["cluster_column"] == "unit"
    assert result.metadata["cluster_mode"] == "explicit"


def test_did_rejects_missing_explicit_cluster_values_before_bootstrap_sampling():
    from diddesign.estimators import did

    rows = [dict(row, cluster=row["unit"]) for row in _panel_rows()]
    rows[1]["cluster"] = np.nan

    with pytest.raises(
        ValueError,
        match="id_cluster must not contain missing values; found missing value in row 1\\.",
    ):
        did(
            rows,
            outcome="outcome",
            treatment="treat",
            time="time",
            unit_id="unit",
            id_cluster="cluster",
            n_boot=4,
            random_seed=20260518,
        )


def test_rcs_did_rejects_missing_explicit_cluster_values_before_bootstrap_sampling():
    from diddesign.estimators import did

    rows = _rcs_rows()
    rows[1]["cluster"] = pd.NA

    with pytest.raises(
        ValueError,
        match="id_cluster must not contain missing values; found missing value in row 1\\.",
    ):
        did(
            rows,
            outcome="outcome",
            treatment="treat_group",
            time="time",
            post="post",
            data_type="rcs",
            id_cluster="cluster",
            n_boot=4,
            random_seed=20260518,
        )


def test_did_rejects_missing_model_columns_before_pandas_projection():
    from diddesign.estimators import did

    with pytest.raises(ValueError, match="Column 'missing_cluster' is required"):
        did(
            _panel_rows(),
            outcome="outcome",
            treatment="treat",
            time="time",
            unit_id="unit",
            id_cluster="missing_cluster",
            n_boot=2,
        )

    with pytest.raises(ValueError, match="Column 'missing_covariate' is required"):
        did(
            _panel_rows(),
            outcome="outcome",
            treatment="treat",
            time="time",
            unit_id="unit",
            covariates=("missing_covariate",),
            n_boot=2,
        )


@pytest.mark.parametrize(
    ("kwargs", "exception", "error"),
    [
        ({"outcome": 1}, ValueError, "outcome must be a column name string\\."),
        ({"treatment": 1}, ValueError, "treatment must be a column name string\\."),
        ({"time": 1}, ValueError, "time must be a column name string\\."),
        ({"unit_id": 1}, ValueError, "unit_id must be a column name string when provided\\."),
        ({"post": 1, "data_type": "rcs"}, ValueError, "post must be a column name string when provided\\."),
        ({"outcome": ""}, ValueError, "outcome must be a non-empty column name\\."),
        ({"treatment": "   "}, ValueError, "treatment must be a non-empty column name\\."),
        ({"time": ""}, ValueError, "time must be a non-empty column name\\."),
        ({"unit_id": "   "}, ValueError, "unit_id must be a non-empty column name\\."),
        ({"post": "", "data_type": "rcs"}, ValueError, "post must be a non-empty column name\\."),
    ],
)
def test_did_rejects_invalid_column_name_values_at_public_boundary(kwargs, exception, error):
    from diddesign.estimators import did

    call_kwargs = {
        "outcome": "outcome",
        "treatment": "treat",
        "time": "time",
        "unit_id": "unit",
        "n_boot": 4,
    }
    rows = _panel_rows()
    if kwargs.get("data_type") == "rcs":
        rows = _rcs_rows()
        call_kwargs = {
            "outcome": "outcome",
            "treatment": "treat_group",
            "time": "time",
            "post": "post",
            "data_type": "rcs",
            "n_boot": 4,
        }
    call_kwargs.update(kwargs)

    with pytest.raises(exception, match=error):
        did(rows, **call_kwargs)


@pytest.mark.parametrize(
    ("kwargs", "exception", "error"),
    [
        ({"design": 1}, ValueError, "design must be a string\\."),
        ({"data_type": 1}, ValueError, "data_type must be a string\\."),
        ({"design": "event"}, ValueError, "design must be 'did' or 'sa'\\."),
        ({"data_type": "cross"}, ValueError, "data_type must be 'panel' or 'rcs'\\."),
    ],
)
def test_did_rejects_invalid_design_and_data_type_values_at_public_boundary(kwargs, exception, error):
    from diddesign.estimators import did

    call_kwargs = {
        "outcome": "outcome",
        "treatment": "treat",
        "time": "time",
        "unit_id": "unit",
        "n_boot": 4,
    }
    call_kwargs.update(kwargs)

    with pytest.raises(exception, match=error):
        did(_panel_rows(), **call_kwargs)


def test_did_rejects_non_mapping_option_at_public_boundary():
    from diddesign.estimators import did

    with pytest.raises(ValueError, match="option must be a mapping when provided\\."):
        did(
            _panel_rows(),
            outcome="outcome",
            treatment="treat",
            time="time",
            unit_id="unit",
            option=["n_boot", 4],
        )


def test_did_rejects_mixing_formula_with_manual_surface_fields():
    from diddesign.estimators import did

    with pytest.raises(
        ValueError,
        match="formula cannot be combined with outcome, treatment, post, or covariates",
    ):
        did(
            _panel_rows(),
            formula="outcome ~ treat",
            outcome="outcome",
            time="time",
            unit_id="unit",
            n_boot=2,
        )


def test_did_rejects_standard_panel_inputs_without_a_never_treated_control_unit():
    from diddesign.estimators import did

    rows = [
        {"unit": "a", "time": 0, "treat": 0, "outcome": 1.0},
        {"unit": "a", "time": 1, "treat": 1, "outcome": 1.3},
        {"unit": "a", "time": 2, "treat": 1, "outcome": 1.5},
        {"unit": "b", "time": 0, "treat": 0, "outcome": 0.9},
        {"unit": "b", "time": 1, "treat": 1, "outcome": 1.2},
        {"unit": "b", "time": 2, "treat": 1, "outcome": 1.4},
    ]

    with pytest.raises(ValueError, match="never-treated control unit"):
        did(
            rows,
            outcome="outcome",
            treatment="treat",
            time="time",
            unit_id="unit",
            n_boot=4,
        )


def test_did_supports_dynamic_leads_and_emits_three_row_bundle_per_identified_lead():
    from diddesign.estimators import did

    result = did(
        _panel_rows_with_dynamic_leads(),
        outcome="outcome",
        treatment="treat",
        time="time",
        unit_id="unit",
        lead=(0, 1, 2),
        n_boot=12,
        random_seed=123,
    )

    rows = result.estimate_rows()
    expected_pairs = [
        ("DID", 0),
        ("sDID", 0),
        ("Double-DID", 1),
        ("DID", 1),
        ("sDID", 1),
        ("Double-DID", 2),
        ("DID", 2),
        ("sDID", 2),
    ]

    assert [(row.estimator, row.lead) for row in rows] == expected_pairs
    assert result.metadata["requested_leads"] == (0, 1, 2)
    assert result.metadata["identified_leads"] == (0, 1, 2)
    assert result.metadata["filtered_leads"] == ()
    assert result.metadata["unidentified_leads"] == ()
    assert result.metadata["double_did_available"] is False
    assert result.metadata["double_did_available_leads"] == (1, 2)
    assert result.metadata["weights_by_lead"][0] == {"w_did": None, "w_sdid": None}

    rows_by_key = {(row.estimator, row.lead): row for row in rows}
    assert rows_by_key[("DID", 0)].weight is None
    assert rows_by_key[("sDID", 0)].weight is None
    for current_lead in (1, 2):
        ddid_row = rows_by_key[("Double-DID", current_lead)]
        did_row = rows_by_key[("DID", current_lead)]
        sdid_row = rows_by_key[("sDID", current_lead)]
        assert ddid_row.weight is None
        assert ddid_row.estimate == pytest.approx(did_row.weight * did_row.estimate + sdid_row.weight * sdid_row.estimate)
        assert did_row.weight + sdid_row.weight == pytest.approx(1.0)


def test_did_public_surface_keeps_multi_lead_bootstrap_weights_aligned(monkeypatch):
    import importlib

    from diddesign import DidBootstrapDraw, did

    did_module = importlib.import_module("diddesign.estimators.did")

    def deterministic_bootstrap_draws(frame, *, leads, covariates, n_boot, random_seed):
        assert leads == (0, 1, 2)
        assert covariates == ()
        assert n_boot == 4
        assert random_seed == 20260520
        offsets = ((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0))
        draws = []
        for iteration, (did_offset, sdid_offset) in enumerate(offsets, start=1):
            for current_lead in leads:
                draws.append(
                    DidBootstrapDraw(
                        iteration=iteration,
                        lead=current_lead,
                        did=10.0 * current_lead + did_offset,
                        sdid=10.0 * current_lead + sdid_offset,
                    )
                )
        return tuple(draws)

    monkeypatch.setattr(did_module, "_compute_bootstrap_draws", deterministic_bootstrap_draws)

    result = did(
        _panel_rows_with_dynamic_leads(),
        outcome="outcome",
        treatment="treat",
        time="time",
        unit_id="unit",
        lead=(0, 1, 2),
        n_boot=4,
        random_seed=20260520,
    )

    rows_by_key = {(row.estimator, row.lead): row for row in result.estimate_rows()}

    assert result.metadata["identified_leads"] == (0, 1, 2)
    assert result.metadata["double_did_available"] is True
    assert result.metadata["double_did_available_leads"] == (0, 1, 2)
    assert result.metadata["n_boot_realized"] == 4
    for current_lead in (0, 1, 2):
        weights = result.metadata["weights_by_lead"][current_lead]
        assert weights["w_did"] == pytest.approx(0.5)
        assert weights["w_sdid"] == pytest.approx(0.5)
        assert rows_by_key[("Double-DID", current_lead)].estimate == pytest.approx(
            0.5 * rows_by_key[("DID", current_lead)].estimate
            + 0.5 * rows_by_key[("sDID", current_lead)].estimate
        )

    bootstrap_frame = result.to_bootstrap_frame()
    assert bootstrap_frame.groupby("iteration")["lead"].apply(tuple).to_dict() == {
        1: (0, 1, 2),
        2: (0, 1, 2),
        3: (0, 1, 2),
        4: (0, 1, 2),
    }
    assert result.to_weights_frame()["lead"].tolist() == [0, 1, 2]
    assert result.to_gmm_frame()["double_did_available"].tolist() == [True, True, True]


def test_did_defaults_to_asymptotic_inference_without_changing_point_estimates_or_weights():
    from diddesign.estimators import did

    z_975 = NormalDist().inv_cdf(0.975)
    default_result = did(
        _panel_rows_with_non_singular_weights(),
        outcome="outcome",
        treatment="treat",
        time="time",
        unit_id="unit",
        n_boot=20,
        random_seed=1234,
    )
    bootstrap_result = did(
        _panel_rows_with_non_singular_weights(),
        outcome="outcome",
        treatment="treat",
        time="time",
        unit_id="unit",
        n_boot=20,
        random_seed=1234,
        se_boot=np.bool_(True),
    )

    asymptotic_rows = default_result.estimate_rows()
    bootstrap_rows = bootstrap_result.estimate_rows()

    assert default_result.metadata["ci_method"] == "asymptotic"
    assert default_result.metadata["se_boot"] is False
    assert bootstrap_result.metadata["ci_method"] == "bootstrap"
    assert bootstrap_result.metadata["se_boot"] is True
    assert [row.estimate for row in asymptotic_rows] == pytest.approx(
        [row.estimate for row in bootstrap_rows]
    )
    assert [row.weight for row in asymptotic_rows[1:]] == pytest.approx(
        [row.weight for row in bootstrap_rows[1:]]
    )
    assert asymptotic_rows[1].std_error == pytest.approx(bootstrap_rows[1].std_error)
    assert asymptotic_rows[2].std_error == pytest.approx(bootstrap_rows[2].std_error)

    lead_draws = np.asarray(
        [(draw.did, draw.sdid) for draw in default_result.bootstrap_draws],
        dtype=float,
    )
    vcov = np.cov(lead_draws.T, ddof=1)
    inverse_vcov = np.linalg.pinv(vcov)
    expected_ddid_std_error = math.sqrt(
        1.0 / float((inverse_vcov @ np.ones(2, dtype=float)).sum())
    )

    assert asymptotic_rows[0].std_error == pytest.approx(expected_ddid_std_error)
    for row in asymptotic_rows:
        assert row.std_error is not None
        assert row.ci_lo == pytest.approx(row.estimate - z_975 * row.std_error)
        assert row.ci_hi == pytest.approx(row.estimate + z_975 * row.std_error)


def test_did_rejects_duplicate_covariates_before_fitting_the_design_matrix():
    from diddesign.estimators import did

    for covariates in (["x", "x"], ["x", "factor(x)"]):
        with pytest.raises(ValueError, match="covariates must not contain duplicate column names\\."):
            did(
                _panel_rows_with_identifiable_covariate(),
                outcome="outcome",
                treatment="treat",
                time="time",
                unit_id="unit",
                covariates=covariates,
                n_boot=4,
                random_seed=1,
            )


@pytest.mark.parametrize("covariates", [("outcome",), ("treat",), ("time",), ("unit",)])
def test_did_rejects_covariates_that_reuse_design_role_columns(covariates):
    from diddesign.estimators import did

    with pytest.raises(
        ValueError,
        match="covariates must not reuse outcome, treatment, time, unit_id, or post columns\\.",
    ):
        did(
            _panel_rows_with_identifiable_covariate(),
            outcome="outcome",
            treatment="treat",
            time="time",
            unit_id="unit",
            covariates=covariates,
            n_boot=4,
            random_seed=1,
        )


def test_panel_sdid_reports_group_lagged_mean_identification_failure_with_missing_preperiod_group_mean():
    from diddesign.estimators import did

    rows = [
        {"unit": "a", "time": 0, "treat": 0, "outcome": float("nan")},
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

    with pytest.raises(
        ValueError,
        match="No identifiable lead values remain for did\\(\\)\\. Last failure: Insufficient variation to estimate outcome_delta\\.",
    ):
        did(
            rows,
            outcome="outcome",
            treatment="treat",
            time="time",
            unit_id="unit",
            n_boot=4,
            random_seed=20260503,
        )


def test_did_rejects_non_finite_outcome_values_before_estimating_moments():
    from diddesign.estimators import did

    rows = _panel_rows_with_non_singular_weights()
    rows[0]["outcome"] = float("inf")

    with pytest.raises(ValueError, match="outcome.*finite"):
        did(
            rows,
            outcome="outcome",
            treatment="treat",
            time="time",
            unit_id="unit",
            n_boot=4,
            random_seed=20260503,
        )


def test_did_rejects_non_finite_numeric_covariates_before_estimating_moments():
    from diddesign.estimators import did

    rows = _panel_rows_with_identifiable_covariate()
    rows[0]["x"] = float("inf")

    with pytest.raises(ValueError, match="covariate 'x'.*finite"):
        did(
            rows,
            outcome="outcome",
            treatment="treat",
            time="time",
            unit_id="unit",
            covariates=("x",),
            n_boot=4,
            random_seed=20260503,
        )


@pytest.mark.parametrize("bad_covariate", ["factor(x + z)", "factor(x))"])
def test_did_rejects_unsupported_covariate_terms_at_public_boundary(bad_covariate):
    from diddesign.estimators import did

    with pytest.raises(
        ValueError,
        match=r"covariates must contain only column names or factor\(column\) terms\.",
    ):
        did(
            _panel_rows_with_identifiable_covariate(),
            outcome="outcome",
            treatment="treat",
            time="time",
            unit_id="unit",
            covariates=[bad_covariate],
            n_boot=4,
            random_seed=20260517,
        )


def test_did_rejects_bare_string_covariates_at_public_boundary():
    from diddesign.estimators import did

    with pytest.raises(ValueError, match=r"covariates must be a sequence of column names or factor\(\.\.\.\) terms\."):
        did(
            _panel_rows_with_identifiable_covariate(),
            outcome="outcome",
            treatment="treat",
            time="time",
            unit_id="unit",
            covariates="x",
            n_boot=4,
            random_seed=20260517,
        )


def test_did_marks_double_did_unavailable_when_bootstrap_vcov_is_numerically_degenerate():
    from diddesign.estimators import did

    result = did(
        _panel_rows_with_degenerate_bootstrap_vcov(),
        outcome="outcome",
        treatment="treat",
        time="time",
        unit_id="unit",
        n_boot=6,
        random_seed=1234,
    )

    payload = result.as_payload()

    assert [(row["estimator"], row["lead"]) for row in payload["estimates"]] == [
        ("DID", 0),
        ("sDID", 0),
    ]
    assert payload["metadata"]["double_did_available"] is False
    assert payload["metadata"]["double_did_available_leads"] == ()
    assert payload["metadata"]["weights_by_lead"][0] == {"w_did": None, "w_sdid": None}
    assert payload["metadata"]["W_by_lead"][0] is None
    vcov = np.asarray(payload["metadata"]["vcov_gmm_by_lead"][0], dtype=float)
    assert np.linalg.norm(vcov, ord=np.inf) < 1e-12
    assert result.weight_rows()[0].double_did_available is False


def test_gmm_weight_helpers_reject_nonzero_rank_deficient_bootstrap_vcov():
    from diddesign.estimators.did import _compute_double_did_row, _compute_sa_double_did_row
    from diddesign.results import DidBootstrapDraw

    draws = (
        DidBootstrapDraw(iteration=1, lead=0, did=1.0, sdid=2.0),
        DidBootstrapDraw(iteration=2, lead=0, did=2.0, sdid=4.0),
        DidBootstrapDraw(iteration=3, lead=0, did=3.0, sdid=6.0),
    )
    component_estimates = {"DID": 2.0, "sDID": 4.0}

    for compute_row in (_compute_double_did_row, _compute_sa_double_did_row):
        row, weights = compute_row(
            lead=0,
            component_estimates=component_estimates,
            draws=draws,
            se_boot=True,
        )

        assert row is None
        assert weights["w_did"] is None
        assert weights["w_sdid"] is None
        assert weights["W"] is None
        assert weights["vcov_gmm"] == ((1.0, 2.0), (2.0, 4.0))


def test_gmm_weight_helpers_allow_identifiable_negative_optimal_weights(monkeypatch):
    import importlib

    from diddesign.results import DidBootstrapDraw

    did_module = importlib.import_module("diddesign.estimators.did")
    draws = (
        DidBootstrapDraw(iteration=1, lead=0, did=1.0, sdid=1.0),
        DidBootstrapDraw(iteration=2, lead=0, did=2.0, sdid=4.0),
        DidBootstrapDraw(iteration=3, lead=0, did=4.0, sdid=5.0),
    )
    component_estimates = {"DID": 10.0, "sDID": 20.0}
    inverse_vcov = did_module.np.asarray([[1.0, -2.0], [-2.0, 5.0]], dtype=float)

    monkeypatch.setattr(did_module.np.linalg, "inv", lambda vcov: inverse_vcov)

    for compute_row, estimator in (
        (did_module._compute_double_did_row, "Double-DID"),
        (did_module._compute_sa_double_did_row, "SA-Double-DID"),
    ):
        row, weights = compute_row(
            lead=0,
            component_estimates=component_estimates,
            draws=draws,
            se_boot=False,
        )

        assert row is not None
        assert row.estimator == estimator
        assert weights["w_did"] == pytest.approx(-0.5)
        assert weights["w_sdid"] == pytest.approx(1.5)
        assert row.estimate == pytest.approx(25.0)
        assert row.std_error == pytest.approx(math.sqrt(0.5))
        assert row.ci_lo is not None
        assert row.ci_hi is not None


def test_did_public_surface_preserves_negative_gmm_weights_and_w_matrix(monkeypatch):
    import importlib

    from diddesign import DidBootstrapDraw, did

    did_module = importlib.import_module("diddesign.estimators.did")

    def deterministic_bootstrap_draws(frame, *, leads, covariates, n_boot, random_seed):
        assert leads == (0,)
        assert covariates == ()
        assert n_boot == 3
        assert random_seed == 20260520
        return (
            DidBootstrapDraw(iteration=1, lead=0, did=1.0, sdid=1.0),
            DidBootstrapDraw(iteration=2, lead=0, did=2.0, sdid=4.0),
            DidBootstrapDraw(iteration=3, lead=0, did=4.0, sdid=5.0),
        )

    inverse_vcov = did_module.np.asarray([[1.0, -2.0], [-2.0, 5.0]], dtype=float)
    monkeypatch.setattr(did_module, "_compute_bootstrap_draws", deterministic_bootstrap_draws)
    monkeypatch.setattr(did_module.np.linalg, "inv", lambda vcov: inverse_vcov)

    result = did(
        _panel_rows_with_non_singular_weights(),
        outcome="outcome",
        treatment="treat",
        time="time",
        unit_id="unit",
        n_boot=3,
        random_seed=20260520,
    )

    rows_by_estimator = {row.estimator: row for row in result.estimate_rows()}
    weight_row = result.weight_rows()[0]
    gmm_row = result.gmm_rows()[0]
    payload = result.as_payload()

    assert rows_by_estimator["DID"].weight == pytest.approx(-0.5)
    assert rows_by_estimator["sDID"].weight == pytest.approx(1.5)
    assert rows_by_estimator["Double-DID"].estimate == pytest.approx(
        -0.5 * rows_by_estimator["DID"].estimate
        + 1.5 * rows_by_estimator["sDID"].estimate
    )
    assert rows_by_estimator["Double-DID"].std_error == pytest.approx(math.sqrt(0.5))
    assert result.metadata["weights_by_lead"][0]["w_did"] == pytest.approx(-0.5)
    assert result.metadata["weights_by_lead"][0]["w_sdid"] == pytest.approx(1.5)
    assert result.metadata["W_by_lead"][0] == ((1.0, -2.0), (-2.0, 5.0))
    assert weight_row.w_did == pytest.approx(-0.5)
    assert weight_row.w_sdid == pytest.approx(1.5)
    assert gmm_row.W_did == pytest.approx(1.0)
    assert gmm_row.W_sdid == pytest.approx(5.0)
    assert gmm_row.W_covariance == pytest.approx(-2.0)
    assert gmm_row.gmm_variance == pytest.approx(0.5)
    assert payload["gmm"][0]["w_did"] == pytest.approx(-0.5)
    assert payload["gmm"][0]["w_sdid"] == pytest.approx(1.5)


def test_gmm_weight_helpers_reject_invalid_inverse_weight_matrices(monkeypatch):
    import importlib

    from diddesign.results import DidBootstrapDraw

    did_module = importlib.import_module("diddesign.estimators.did")
    draws = (
        DidBootstrapDraw(iteration=1, lead=0, did=1.0, sdid=2.0),
        DidBootstrapDraw(iteration=2, lead=0, did=2.0, sdid=2.5),
        DidBootstrapDraw(iteration=3, lead=0, did=3.0, sdid=5.0),
    )
    component_estimates = {"DID": 2.0, "sDID": 3.0}

    monkeypatch.setattr(
        did_module.np.linalg,
        "inv",
        lambda vcov: did_module.np.asarray([[did_module.np.inf, 0.0], [0.0, 1.0]], dtype=float),
    )
    for compute_row in (did_module._compute_double_did_row, did_module._compute_sa_double_did_row):
        row, weights = compute_row(
            lead=0,
            component_estimates=component_estimates,
            draws=draws,
            se_boot=False,
        )

        assert row is None
        assert weights["w_did"] is None
        assert weights["w_sdid"] is None
        assert weights["W"] is None
        assert weights["vcov_gmm"] is not None

    monkeypatch.setattr(
        did_module.np.linalg,
        "inv",
        lambda vcov: did_module.np.asarray([[1.0, -2.0], [-2.0, 1.0]], dtype=float),
    )
    for compute_row in (did_module._compute_double_did_row, did_module._compute_sa_double_did_row):
        row, weights = compute_row(
            lead=0,
            component_estimates=component_estimates,
            draws=draws,
            se_boot=False,
        )

        assert row is None
        assert weights["w_did"] is None
        assert weights["w_sdid"] is None
        assert weights["W"] is None
        assert weights["vcov_gmm"] is not None

    monkeypatch.setattr(
        did_module.np.linalg,
        "inv",
        lambda vcov: did_module.np.asarray([[2.0, 0.0], [1.0, 2.0]], dtype=float),
    )
    for compute_row in (did_module._compute_double_did_row, did_module._compute_sa_double_did_row):
        row, weights = compute_row(
            lead=0,
            component_estimates=component_estimates,
            draws=draws,
            se_boot=False,
        )

        assert row is None
        assert weights["w_did"] is None
        assert weights["w_sdid"] is None
        assert weights["W"] is None
        assert weights["vcov_gmm"] is not None

    monkeypatch.setattr(
        did_module.np.linalg,
        "inv",
        lambda vcov: did_module.np.asarray([[2.0, 3.0], [3.0, 2.0]], dtype=float),
    )
    for compute_row in (did_module._compute_double_did_row, did_module._compute_sa_double_did_row):
        row, weights = compute_row(
            lead=0,
            component_estimates=component_estimates,
            draws=draws,
            se_boot=False,
        )

        assert row is None
        assert weights["w_did"] is None
        assert weights["w_sdid"] is None
        assert weights["W"] is None
        assert weights["vcov_gmm"] is not None

    monkeypatch.setattr(
        did_module.np.linalg,
        "inv",
        lambda vcov: did_module.np.asarray(
            [[2.0e-14, -1.0e-14], [-1.0e-14, 2.0e-14]],
            dtype=float,
        ),
    )
    for compute_row in (did_module._compute_double_did_row, did_module._compute_sa_double_did_row):
        row, weights = compute_row(
            lead=0,
            component_estimates=component_estimates,
            draws=draws,
            se_boot=False,
        )

        assert row is None
        assert weights["w_did"] is None
        assert weights["w_sdid"] is None
        assert weights["W"] is None
        assert weights["vcov_gmm"] is not None


def test_gmm_weight_helpers_reject_bootstrap_draws_from_other_leads():
    from diddesign.estimators.did import _compute_double_did_row, _compute_sa_double_did_row
    from diddesign.results import DidBootstrapDraw

    draws = (
        DidBootstrapDraw(iteration=1, lead=0, did=1.0, sdid=2.0),
        DidBootstrapDraw(iteration=2, lead=1, did=2.0, sdid=3.0),
    )
    component_estimates = {"DID": 1.5, "sDID": 2.5}

    with pytest.raises(ValueError, match="Double-DID bootstrap draws must match the requested lead\\."):
        _compute_double_did_row(
            lead=0,
            component_estimates=component_estimates,
            draws=draws,
            se_boot=True,
        )

    with pytest.raises(ValueError, match="SA-Double-DID bootstrap draws must match the requested lead\\."):
        _compute_sa_double_did_row(
            lead=0,
            component_estimates=component_estimates,
            draws=draws,
            se_boot=True,
        )


def test_did_rejects_non_integer_leads_instead_of_coercing_them():
    from diddesign.estimators import did

    with pytest.raises(ValueError, match="lead"):
        did(
            _panel_rows(),
            outcome="outcome",
            treatment="treat",
            time="time",
            unit_id="unit",
            lead=0.5,
            n_boot=4,
        )


def test_did_rejects_bootstrap_sizes_below_two():
    from diddesign.estimators import did

    with pytest.raises(ValueError, match="n_boot"):
        did(
            _panel_rows(),
            outcome="outcome",
            treatment="treat",
            time="time",
            unit_id="unit",
            n_boot=1,
        )


def test_did_rejects_float_bootstrap_sizes_even_when_the_value_is_whole():
    from diddesign.estimators import did

    with pytest.raises(ValueError, match="n_boot"):
        did(
            _panel_rows(),
            outcome="outcome",
            treatment="treat",
            time="time",
            unit_id="unit",
            n_boot=4.0,
        )


@pytest.mark.parametrize("bad_seed", [True, False, np.bool_(True), np.bool_(False), 1.0, "1", -1, 2**32])
def test_did_rejects_invalid_random_seed_values_before_bootstrap_sampling(bad_seed):
    from diddesign.estimators import did

    with pytest.raises(ValueError, match="random_seed"):
        did(
            _panel_rows(),
            outcome="outcome",
            treatment="treat",
            time="time",
            unit_id="unit",
            n_boot=4,
            random_seed=bad_seed,
        )


def test_did_normalizes_numpy_integer_random_seed_in_public_metadata():
    from diddesign.estimators import did

    result = did(
        _panel_rows(),
        outcome="outcome",
        treatment="treat",
        time="time",
        unit_id="unit",
        n_boot=4,
        random_seed=np.int64(7),
    )

    assert result.metadata["random_seed"] == 7
    assert isinstance(result.metadata["random_seed"], int)


def test_did_normalizes_numpy_integer_bootstrap_and_lead_public_metadata():
    from diddesign.estimators import did

    result = did(
        _panel_rows_with_dynamic_leads(),
        outcome="outcome",
        treatment="treat",
        time="time",
        unit_id="unit",
        lead=(np.int64(0), np.int64(1)),
        n_boot=np.int64(4),
        random_seed=7,
    )

    assert result.metadata["requested_lead"] == (0, 1)
    assert all(isinstance(lead, int) for lead in result.metadata["requested_lead"])
    assert result.metadata["n_boot_requested"] == 4
    assert isinstance(result.metadata["n_boot_requested"], int)
    assert result.metadata["n_boot_realized"] == 4
    assert isinstance(result.metadata["n_boot_realized"], int)


def test_did_rejects_rcs_post_indicator_that_returns_to_pre_treatment():
    from diddesign.core.data_contracts import DataContractError
    from diddesign.estimators import did

    rows = [
        {"cluster": "north", "time": 0, "treat_group": 0, "post": 0, "outcome": 1.0},
        {"cluster": "north", "time": 0, "treat_group": 1, "post": 0, "outcome": 2.0},
        {"cluster": "south", "time": 0, "treat_group": 0, "post": 0, "outcome": 1.1},
        {"cluster": "south", "time": 0, "treat_group": 1, "post": 0, "outcome": 2.1},
        {"cluster": "north", "time": 1, "treat_group": 0, "post": 0, "outcome": 1.3},
        {"cluster": "north", "time": 1, "treat_group": 1, "post": 0, "outcome": 2.3},
        {"cluster": "south", "time": 1, "treat_group": 0, "post": 0, "outcome": 1.4},
        {"cluster": "south", "time": 1, "treat_group": 1, "post": 0, "outcome": 2.4},
        {"cluster": "north", "time": 2, "treat_group": 0, "post": 1, "outcome": 1.5},
        {"cluster": "north", "time": 2, "treat_group": 1, "post": 1, "outcome": 3.0},
        {"cluster": "south", "time": 2, "treat_group": 0, "post": 1, "outcome": 1.6},
        {"cluster": "south", "time": 2, "treat_group": 1, "post": 1, "outcome": 3.1},
        {"cluster": "north", "time": 3, "treat_group": 0, "post": 0, "outcome": 1.7},
        {"cluster": "north", "time": 3, "treat_group": 1, "post": 0, "outcome": 3.3},
        {"cluster": "south", "time": 3, "treat_group": 0, "post": 0, "outcome": 1.8},
        {"cluster": "south", "time": 3, "treat_group": 1, "post": 0, "outcome": 3.4},
    ]

    with pytest.raises(DataContractError, match="post.*once.*1"):
        did(
            rows,
            outcome="outcome",
            treatment="treat_group",
            time="time",
            post="post",
            data_type="rcs",
            id_cluster="cluster",
            n_boot=4,
            random_seed=1,
        )


def test_panel_seeded_bootstrap_is_stable_to_input_row_order():
    from diddesign.estimators import did

    rows = _panel_rows_with_non_singular_weights()
    shuffled = [rows[index] for index in (8, 9, 10, 11, 4, 5, 0, 1, 2, 6, 7, 3)]

    base = did(
        rows,
        outcome="outcome",
        treatment="treat",
        time="time",
        unit_id="unit",
        n_boot=8,
        random_seed=20260608,
    )
    reordered = did(
        shuffled,
        outcome="outcome",
        treatment="treat",
        time="time",
        unit_id="unit",
        n_boot=8,
        random_seed=20260608,
    )

    pd.testing.assert_frame_equal(base.to_estimates_frame(), reordered.to_estimates_frame())
    pd.testing.assert_frame_equal(base.to_bootstrap_frame(), reordered.to_bootstrap_frame())
    pd.testing.assert_frame_equal(base.to_weights_frame(), reordered.to_weights_frame())
    pd.testing.assert_frame_equal(base.to_gmm_frame(), reordered.to_gmm_frame())


def test_sa_seeded_bootstrap_with_explicit_clusters_is_stable_to_input_row_order():
    from diddesign.estimators import did

    rows = []
    for unit, adoption, cluster, baseline in (
        ("a", 2, "cluster-b", 1.0),
        ("b", 3, "cluster-a", 1.4),
        ("c", None, "cluster-c", 0.8),
        ("d", None, "cluster-d", 1.2),
    ):
        for current_time in range(5):
            treated = int(adoption is not None and current_time >= adoption)
            rows.append(
                {
                    "unit": unit,
                    "cluster": cluster,
                    "time": current_time,
                    "treat": treated,
                    "outcome": baseline + 0.25 * current_time + 0.05 * current_time**2 + 1.5 * treated,
                }
            )
    shuffled = [rows[index] for index in (9, 8, 7, 6, 5, 15, 16, 17, 18, 19, 0, 1, 2, 3, 4, 14, 13, 12, 11, 10)]

    base = did(
        rows,
        outcome="outcome",
        treatment="treat",
        time="time",
        unit_id="unit",
        id_cluster="cluster",
        design="sa",
        lead=(0, 1),
        thres=1,
        n_boot=8,
        random_seed=20260608,
    )
    reordered = did(
        shuffled,
        outcome="outcome",
        treatment="treat",
        time="time",
        unit_id="unit",
        id_cluster="cluster",
        design="sa",
        lead=(0, 1),
        thres=1,
        n_boot=8,
        random_seed=20260608,
    )

    pd.testing.assert_frame_equal(base.to_estimates_frame(), reordered.to_estimates_frame())
    pd.testing.assert_frame_equal(base.to_bootstrap_frame(), reordered.to_bootstrap_frame())
    pd.testing.assert_frame_equal(base.to_weights_frame(), reordered.to_weights_frame())
    pd.testing.assert_frame_equal(base.to_gmm_frame(), reordered.to_gmm_frame())
