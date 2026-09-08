"""Independent numerical and time-availability acceptance tests for monthly OLS."""

import ast
import copy
import json
import math
from pathlib import Path

import pytest

from model_risk_lab.forecast import example_request, run_experiment


def month(index):
    year, offset = divmod(2020 * 12 + index, 12)
    return f"{year:04d}-{offset + 1:02d}"


def request_for(
    targets, observations, *, development=7, n_splits=1, validation=2, horizon=1, min_train=3
):
    ids = sorted(observations)
    return {
        "schema_version": 1,
        "title": "Invented monthly test",
        "rows": [
            {
                "period": month(index),
                "target": target,
                "features": {key: observations[key][index] for key in ids},
            }
            for index, target in enumerate(targets)
        ],
        "features": [
            {"id": key, "name": key.upper(), "lag": horizon, "release_delay": 0} for key in ids
        ],
        "horizon": horizon,
        "development_end": month(development - 1),
        "n_splits": n_splits,
        "validation_months": validation,
        "min_train": min_train,
        "target": {"name": "Test target", "unit": "index points", "transformation": "none"},
    }


def by_id(result):
    return {candidate["id"]: candidate for candidate in result["candidates"]}


def test_example_has_reproducible_180_months_fifteen_candidates_two_baselines():
    request = example_request()
    assert request == example_request()
    assert len(request["rows"]) == 180
    result = run_experiment(request, reveal_holdout=True)
    candidates = result["candidates"]
    assert len(candidates) == 17
    assert sum(candidate["role"] == "candidate" for candidate in candidates) == 15
    assert sum(candidate["role"] == "baseline" for candidate in candidates) == 2
    assert all(candidate["status"] == "ok" for candidate in candidates)
    assert result["selected_on_development"] == "ols-f1-f2"
    assert {candidate["development"]["n"] for candidate in candidates} == {36}
    assert {candidate["holdout"]["n"] for candidate in candidates} == {24}
    assert len(result["folds"]) == 51
    assert result["coverage"] == {
        "raw": 180,
        "usable": 177,
        "excluded": 3,
        "development": {"raw": 156, "usable": 153, "excluded": 3, "validation_usable": 36},
        "holdout": {"raw": 24, "usable": 24, "excluded": 0},
    }
    json.dumps(result, allow_nan=False)


def test_hand_calculated_ols_and_fold_metrics_in_original_units():
    # Training x=[0,1,2,3], y=[1,3,2,6]: xbar=1.5, ybar=3,
    # slope=7/5, intercept=.9, SSE=4.2, SST=14, R2=.7, adjusted R2=.55.
    request = request_for([99, 1, 3, 2, 6, 5, 8, 10, 12, 14], {"f1": list(range(10))})
    result = run_experiment(request, reveal_holdout=True)
    fold = next(fold for fold in result["folds"] if fold["model_id"] == "ols-f1")
    fitted = fold["fit"]
    assert fitted["train_periods"] == [month(index) for index in range(1, 5)]
    assert fitted["intercept"] == pytest.approx(0.9)
    assert fitted["coefficients"]["f1"] == pytest.approx(1.4)
    assert fitted["feature_means"]["f1"] == 1.5
    assert fitted["feature_scales"]["f1"] == pytest.approx(math.sqrt(1.25))
    assert fitted["r2"] == pytest.approx(0.7)
    assert fitted["adjusted_r2"] == pytest.approx(0.55)
    assert fitted["condition_number"] == pytest.approx(1)
    assert fitted["vif"]["f1"] == pytest.approx(1)
    # Forecasts 6.5,7.9 against 5,8: residuals +1.5,-.1.
    score = by_id(result)["ols-f1"]["development"]
    assert score == pytest.approx({"n": 2, "mae": 0.8, "rmse": math.sqrt(1.13), "bias": 0.7})
    predictions = [
        row["predictions"]["ols-f1"]
        for row in result["predictions"]
        if row["split"] == "development"
    ]
    assert predictions == pytest.approx([6.5, 7.9])


def test_exact_single_and_dual_factor_coefficients_are_not_standardized_units():
    x1 = [index * index / 10 for index in range(24)]
    x2 = [(index % 4) - 1 for index in range(24)]
    targets = [7] + [7 + 2 * x1[index - 1] - 3 * x2[index - 1] for index in range(1, 24)]
    result = run_experiment(
        request_for(targets, {"f1": x1, "f2": x2}, development=20, n_splits=2, validation=3)
    )
    candidate = by_id(result)["ols-f1-f2"]
    assert candidate["fit"]["intercept"] == pytest.approx(7)
    assert candidate["fit"]["coefficients"] == pytest.approx({"f1": 2, "f2": -3})
    assert candidate["fit"]["r2"] == pytest.approx(1)
    assert candidate["development"]["mae"] == pytest.approx(0, abs=1e-11)
    assert result["selected_on_development"] == "ols-f1-f2"
    single_targets = [3] + [3 + 2 * value for value in x1[:-1]]
    single = by_id(run_experiment(request_for(single_targets, {"f1": x1}, development=20)))[
        "ols-f1"
    ]
    assert single["fit"]["intercept"] == pytest.approx(3)
    assert single["fit"]["coefficients"] == pytest.approx({"f1": 2})


def test_baselines_are_scored_on_the_same_months_with_independent_hand_oracle():
    request = request_for([99, 1, 3, 2, 6, 5, 8, 10, 12, 14], {"f1": list(range(10))})
    result = run_experiment(request, reveal_holdout=True)
    models = by_id(result)
    # Fold training mean=3. Forecasts [3,3] vs [5,8] => errors [-2,-5].
    assert models["baseline-mean"]["development"] == pytest.approx(
        {"n": 2, "mae": 3.5, "rmse": math.sqrt(14.5), "bias": -3.5}
    )
    # Persistence [6,5] vs [5,8] => errors [+1,-3].
    assert models["baseline-persistence"]["development"] == pytest.approx(
        {"n": 2, "mae": 2, "rmse": math.sqrt(5), "bias": -1}
    )
    held = [row for row in result["predictions"] if row["split"] == "holdout"]
    assert [row["predictions"]["baseline-persistence"] for row in held] == [8, 10, 12]
    assert "updating historical baseline" in result["protocol"]["persistence_rule"]
    # Final mean uses the six available development targets, never holdout targets.
    assert [row["predictions"]["baseline-mean"] for row in held] == pytest.approx([25 / 6] * 3)


def test_calendar_purge_and_final_cutoff_respect_horizon():
    request = example_request()
    request["horizon"] = 3
    for feature in request["features"]:
        feature["lag"] = 3 + feature["release_delay"]
    result = run_experiment(request)
    assert result["protocol"]["final_training_cutoff"] == "2022-10"
    for fold in result["folds"]:
        month_year, month_no = map(int, fold["validation_start"].split("-"))
        cutoff_year, cutoff_no = map(int, fold["training_cutoff"].split("-"))
        assert (month_year * 12 + month_no) - (cutoff_year * 12 + cutoff_no) == 3
        assert max(fold["train_periods"]) == fold["training_cutoff"]
        assert max(fold["train_periods"]) < min(fold["validation_periods"])
    for candidate in result["candidates"]:
        assert candidate["fit"]["train_end"] == "2022-10"
    changed = copy.deepcopy(request)
    for row in changed["rows"]:
        if "2022-11" <= row["period"] <= "2022-12":
            row["target"] += 10000
    after = run_experiment(changed)
    assert [candidate["fit"] for candidate in result["candidates"]] == [
        candidate["fit"] for candidate in after["candidates"]
    ]


def test_training_scaler_does_not_see_validation_or_later_observations():
    request = example_request()
    before = run_experiment(request)
    first = min(fold["validation_start"] for fold in before["folds"])
    changed = copy.deepcopy(request)
    for index, row in enumerate(changed["rows"]):
        if row["period"] >= first:
            row["target"] += 1000
            row["features"] = {key: value + 2000 + index for key, value in row["features"].items()}
    after = run_experiment(changed)
    prior = [fold["fit"] for fold in before["folds"] if fold["fold"] == 1]
    later = [fold["fit"] for fold in after["folds"] if fold["fold"] == 1]
    assert prior == later
    before_first = next(row for row in before["predictions"] if row["period"] == first)
    after_first = next(row for row in after["predictions"] if row["period"] == first)
    assert before_first["predictions"] == after_first["predictions"]
    assert before_first["actual"] != after_first["actual"]


@pytest.mark.parametrize("change", ["targets", "features", "missing_features", "extend_holdout"])
def test_holdout_changes_cannot_change_selection_coefficients_or_cv_denominators(change):
    request = example_request()
    before = run_experiment(request, reveal_holdout=True)
    changed = copy.deepcopy(request)
    for index, row in enumerate(changed["rows"]):
        if row["period"] > changed["development_end"]:
            if change == "targets":
                row["target"] += 1_000_000 + index
            elif change == "features":
                row["features"] = {
                    key: value * -3000 + index for key, value in row["features"].items()
                }
            elif change == "missing_features":
                row["features"] = dict.fromkeys(row["features"])
    if change == "extend_holdout":
        changed["rows"].append(
            {
                "period": "2025-01",
                "target": 900,
                "features": {f"f{index}": 100 * index for index in range(1, 6)},
            }
        )
    after = run_experiment(changed, reveal_holdout=True)
    assert before["selected_on_development"] == after["selected_on_development"]
    assert before["selection_fingerprint"] == after["selection_fingerprint"]
    assert before["data_fingerprint"] != after["data_fingerprint"]
    assert before["folds"] == after["folds"]
    assert before["coverage"]["development"] == after["coverage"]["development"]
    for left, right in zip(before["candidates"], after["candidates"]):
        assert left["status"] == right["status"]
        assert left["development"] == right["development"]
        assert left["fit"] == right["fit"]


def test_holdout_unrevealed_returns_no_holdout_scores_or_predictions():
    request = example_request()
    hidden = run_experiment(request)
    shown = run_experiment(request, reveal_holdout=True)
    assert hidden["stage"] == "development"
    assert shown["stage"] == "holdout"
    assert {row["split"] for row in hidden["predictions"]} == {"development"}
    assert all(
        candidate["holdout"] is None and candidate["holdout_reason"] is None
        for candidate in hidden["candidates"]
    )
    assert hidden["selection_fingerprint"] == shown["selection_fingerprint"]
    assert hidden["data_fingerprint"] == shown["data_fingerprint"]
    assert hidden["selected_on_development"] == shown["selected_on_development"]
    assert hidden["folds"] == shown["folds"]
    assert [row for row in shown["predictions"] if row["split"] == "development"] == hidden[
        "predictions"
    ]


def test_missing_feature_excludes_lagged_month_for_every_candidate_and_baseline():
    request = example_request()
    request["rows"][130]["features"]["f1"] = None
    result = run_experiment(request)
    assert len(result["input_rows"]) == 180
    raw_missing = result["input_rows"][130]
    lagged_missing = result["input_rows"][131]
    assert raw_missing["features"]["f1"] is None
    assert raw_missing["status"] == "usable"
    assert lagged_missing["status"] == "excluded"
    assert lagged_missing["reasons"] == [
        {"code": "missing_lagged_feature", "feature": "f1", "source_period": raw_missing["period"]}
    ]
    assert {candidate["development"]["n"] for candidate in result["candidates"]} == {35}
    assert lagged_missing["period"] not in {row["period"] for row in result["predictions"]}
    assert result["coverage"]["excluded"] == 4
    assert result["input_rows"][0]["reasons"][0]["code"] == "warmup"


def test_pooled_rmse_uses_all_rows_not_an_average_of_fold_rmse():
    request = example_request()
    for index in (122, 124, 126):
        request["rows"][index]["features"]["f1"] = None
    result = run_experiment(request)
    model = by_id(result)["baseline-persistence"]
    residuals = [row["predictions"][model["id"]] - row["actual"] for row in result["predictions"]]
    expected = math.sqrt(sum(value * value for value in residuals) / len(residuals))
    assert model["development"]["n"] == 33
    assert model["development"]["rmse"] == pytest.approx(expected)
    folds = [fold for fold in result["folds"] if fold["model_id"] == model["id"]]
    assert [fold["development"]["n"] for fold in folds] == [9, 12, 12]
    naive = sum(fold["development"]["rmse"] for fold in folds) / 3
    assert abs(expected - naive) > 1e-3


def test_rank_deficient_and_constant_candidates_are_retained_with_fold_failures():
    request = example_request()
    request["features"][2].update(lag=1, release_delay=0)
    for row in request["rows"]:
        row["features"]["f3"] = 2 * row["features"]["f1"]
        row["features"]["f4"] = 5
    result = run_experiment(request)
    candidates = by_id(result)
    assert len(candidates) == 17
    assert candidates["ols-f3"]["status"] == "ok"
    assert candidates["ols-f1-f3"]["status"] == "failed"
    assert "rank_deficient" in candidates["ols-f1-f3"]["reason"]
    assert candidates["ols-f4"]["status"] == "failed"
    assert "constant_feature" in candidates["ols-f4"]["reason"]
    assert candidates["ols-f1-f4"]["development"] is None
    assert all(
        fold["status"] == "failed" for fold in result["folds"] if fold["model_id"] == "ols-f1-f3"
    )
    assert all("ols-f1-f3" not in row["predictions"] for row in result["predictions"])
    assert result["selected_on_development"] not in {"ols-f1-f3", "ols-f4", "ols-f1-f4"}


def test_full_rank_but_ill_conditioned_pair_is_not_ranked_as_success():
    request = example_request()
    request["features"][2].update(lag=1, release_delay=0)
    for index, row in enumerate(request["rows"]):
        row["features"]["f3"] = row["features"]["f1"] + 1e-9 * math.sin(index)
    result = run_experiment(request)
    candidate = by_id(result)["ols-f1-f3"]
    assert candidate["status"] == "failed"
    assert "ill_conditioned" in candidate["reason"]
    assert candidate["development"] is None


def test_initial_fold_failure_is_retained_even_when_final_fit_succeeds():
    request = request_for(
        list(range(11)), {"f1": list(range(11))}, development=8, min_train=6, validation=3
    )
    result = run_experiment(request)
    candidate = by_id(result)["ols-f1"]
    assert candidate["status"] == "failed"
    assert "fold 1: insufficient_training_sample" in candidate["reason"]
    assert candidate["fit"]["n"] == 7
    assert candidate["development"] is None
    assert result["selected_on_development"] is None
    assert result["folds"][0]["fit"] is None


def test_all_ols_failures_do_not_hide_baseline_results_or_select_a_baseline():
    request = request_for(list(range(12)), {"f1": [2] * 12}, development=9)
    result = run_experiment(request)
    candidates = by_id(result)
    assert result["selected_on_development"] is None
    assert result["protocol"]["selection_reason"] == "all_ols_candidates_failed"
    assert candidates["baseline-mean"]["status"] == "ok"
    assert candidates["baseline-persistence"]["status"] == "ok"
    assert all(
        set(row["predictions"]) == {"baseline-mean", "baseline-persistence"}
        for row in result["predictions"]
    )


def test_exact_ties_use_stable_ids_and_order_does_not_change_any_result():
    targets = [3] + [3 + 2 * index for index in range(19)]
    request = request_for(targets, {"f1": list(range(20)), "f2": list(range(20))}, development=16)
    result = run_experiment(request, reveal_holdout=True)
    assert result["selected_on_development"] == "ols-f1"
    reordered = copy.deepcopy(request)
    reordered["rows"].reverse()
    reordered["features"].reverse()
    assert run_experiment(reordered, reveal_holdout=True) == result
    assert request["rows"][0]["period"] == "2020-01"


def test_constant_training_target_has_explicit_undefined_r2():
    request = request_for([5] * 12, {"f1": list(range(12))}, development=9)
    result = run_experiment(request, reveal_holdout=True)
    for candidate in result["candidates"]:
        assert candidate["fit"]["r2"] is None
        assert candidate["fit"]["adjusted_r2"] is None
        assert candidate["development"]["mae"] == pytest.approx(0, abs=1e-12)
    json.dumps(result, allow_nan=False)


def test_finite_extreme_extrapolation_does_not_overflow_squared_error_aggregation():
    observations = [((index % 5) - 2) * 1e-145 for index in range(24)]
    targets = [3] + [3 + 2e10 * ((index % 5) - 2) for index in range(23)]
    request = request_for(targets, {"f1": observations}, development=18)
    before = run_experiment(request)
    for row in request["rows"][18:]:
        row["features"]["f1"] = 1e12
    after = run_experiment(request, reveal_holdout=True)
    model = by_id(after)["ols-f1"]
    assert model["status"] == "ok"
    assert model["holdout"]["rmse"] > 1e160
    assert math.isfinite(model["holdout"]["rmse"])
    assert before["selected_on_development"] == after["selected_on_development"]
    assert before["selection_fingerprint"] == after["selection_fingerprint"]
    assert by_id(before)["ols-f1"]["fit"] == model["fit"]
    json.dumps(after, allow_nan=False)


def test_empty_common_holdout_is_explicit_and_does_not_cancel_development_selection():
    request = request_for(list(range(15)), {"f1": list(range(15))}, development=10)
    before = run_experiment(request)
    # The raw last-development feature first becomes relevant in the holdout.
    for row in request["rows"][9:]:
        row["features"]["f1"] = None
    after = run_experiment(request, reveal_holdout=True)
    assert after["coverage"]["holdout"] == {"raw": 5, "usable": 0, "excluded": 5}
    assert before["selected_on_development"] == after["selected_on_development"]
    assert all(candidate["status"] == "ok" for candidate in after["candidates"])
    assert all(
        candidate["holdout"] is None and candidate["holdout_reason"] == "no_common_holdout_rows"
        for candidate in after["candidates"]
    )
    assert not any(row["split"] == "holdout" for row in after["predictions"])


@pytest.mark.parametrize("horizon,lag,delay", [(2, 1, 0), (3, 3, 1), (1, 2, 2)])
def test_unavailable_release_lags_are_rejected(horizon, lag, delay):
    request = example_request()
    request["horizon"] = horizon
    request["features"][0].update(lag=lag, release_delay=delay)
    with pytest.raises(ValueError, match="lag must be at least horizon"):
        run_experiment(request)


@pytest.mark.parametrize("bad", [None, True, "3", math.nan, math.inf, -math.inf, 1e13])
def test_missing_or_invalid_targets_are_never_filled_or_silently_dropped(bad):
    request = example_request()
    request["rows"][2]["target"] = bad
    with pytest.raises(ValueError, match="target at"):
        run_experiment(request)


@pytest.mark.parametrize(
    "case",
    [
        "duplicate",
        "gap",
        "bad_month",
        "zero_year",
        "no_holdout",
        "missing_feature_key",
        "unknown_feature",
        "duplicate_id",
        "string_feature",
        "invalid_schema",
        "too_long",
    ],
)
def test_invalid_input_contract_is_rejected_before_any_fit(case):
    request = example_request()
    if case == "duplicate":
        request["rows"][1] = copy.deepcopy(request["rows"][0])
    elif case == "gap":
        request["rows"].pop(4)
    elif case == "bad_month":
        request["rows"][0]["period"] = "2010-1"
    elif case == "zero_year":
        request["rows"][0]["period"] = "0000-01"
    elif case == "no_holdout":
        request["development_end"] = request["rows"][-1]["period"]
    elif case == "missing_feature_key":
        del request["rows"][0]["features"]["f1"]
    elif case == "unknown_feature":
        request["features"][0]["id"] = "free-form/path"
    elif case == "duplicate_id":
        request["features"][0]["id"] = "f2"
    elif case == "string_feature":
        request["rows"][0]["features"]["f1"] = "12"
    elif case == "invalid_schema":
        request["schema_version"] = True
    elif case == "too_long":
        request["rows"] *= 14
    with pytest.raises(ValueError):
        run_experiment(request)


def test_input_request_is_not_mutated_and_python_310_syntax_is_supported():
    request = example_request()
    before = copy.deepcopy(request)
    run_experiment(request, reveal_holdout=True)
    assert request == before
    source = Path(__file__).parents[1] / "src/model_risk_lab/forecast.py"
    ast.parse(source.read_text(), feature_version=(3, 10))
