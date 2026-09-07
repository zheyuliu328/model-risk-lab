import numpy as np
import pandas as pd
import pytest

from model_risk_lab.credit import (CORE, build_design, check_availability, development_splits,
                                   evaluate, fit_candidate, r_squared, rate_metrics,
                                   smoothed_logit, synthetic_quarters)


def test_reproducibility_and_count_contract():
    raw = synthetic_quarters()
    pd.testing.assert_frame_equal(raw, synthetic_quarters())
    assert len(raw) == 148
    assert raw.defaults.between(0, raw.borrowers).all()
    assert len(build_design(raw).features) == 144


@pytest.mark.parametrize("case", ["missing", "duplicate", "unsorted", "gapped", "bad_count", "fractional", "no_borrowers"])
def test_reject_invalid_raw(case):
    raw = synthetic_quarters()
    if case == "missing":
        raw.loc[12, "growth"] = np.nan
    elif case == "duplicate":
        raw.index = [0] + list(range(147))
    elif case == "unsorted":
        raw = raw.iloc[::-1]
    elif case == "gapped":
        raw = raw.drop(index=12)
    elif case == "bad_count":
        raw.loc[12, "defaults"] = 10001
    elif case == "fractional":
        raw["defaults"] = raw.defaults.astype(float)
        raw.loc[12, "defaults"] = 1.5
    else:
        raw.loc[12, "borrowers"] = 0
    with pytest.raises(ValueError):
        build_design(raw)


def test_current_future_perturbation_cannot_change_available_features():
    raw = synthetic_quarters()
    before = build_design(raw)
    raw.loc[75:, ["growth", "unemployment", "rate"]] += 100
    raw.loc[75:, "defaults"] = 9999
    after = build_design(raw)
    pd.testing.assert_frame_equal(before.features.loc[:75], after.features.loc[:75])


def test_lookahead_is_rejected_despite_time_split():
    d = build_design(synthetic_quarters())
    sources = d.source_quarters.copy()
    sources.loc[35, "growth_mean4"] = 35
    with pytest.raises(ValueError, match="unavailable"):
        check_availability(d.features, sources)


def test_folds_are_forward_and_scaler_fits_training_only():
    d = build_design(synthetic_quarters())
    for train, test in development_splits(d):
        assert train.max() < test.min() and test.max() < 108
        model = fit_candidate("ridge_1", d.features.iloc[train], d.target_logit.iloc[train])
        np.testing.assert_allclose(model[0].mean_, d.features.iloc[train][CORE].mean())
        before = model[0].mean_.copy()
        model.predict(d.features.iloc[test][CORE] + 1000)
        np.testing.assert_array_equal(before, model[0].mean_)


def test_holdout_is_not_used_to_select_model():
    raw = synthetic_quarters()
    baseline = evaluate(raw=raw)
    raw.loc[112:, ["growth", "unemployment", "rate"]] += 30
    raw.loc[112:, "defaults"] = 9000
    changed = evaluate(raw=raw)
    assert baseline["selected_on_development"] == changed["selected_on_development"]
    assert baseline["cv"] == changed["cv"]
    assert baseline["folds"] == changed["folds"]
    name = baseline["selected_on_development"]
    # Features available at the first holdout prediction origin are also unchanged.
    assert baseline["predictions"][name][0] == changed["predictions"][name][0]


def test_nested_ols_training_sse_cannot_increase():
    d = build_design(synthetic_quarters())
    x, y = d.features.iloc[:48], d.target_logit.iloc[:48]
    simple = fit_candidate("ols", x, y)
    complex_model = fit_candidate("complex_ols", x, y)
    sse = np.sum((y-simple.predict(x[CORE]))**2)
    complex_sse = np.sum((y-complex_model.predict(x))**2)
    assert complex_sse <= sse + 1e-10


def test_zero_full_default_and_r2_edges_are_explicit():
    assert np.isfinite(smoothed_logit([0, 100], [100, 100])).all()
    assert r_squared([1, 1], [1, 1])["value"] is None
    assert r_squared([1], [2])["value"] is None
    assert r_squared([1, 2, 3], [10, 10, 10])["value"] < 0
    assert rate_metrics([.01, .02], [.02, .03])["bias_bp"] == pytest.approx(100)


def test_invalid_model_is_excluded_and_all_predictions_share_holdout():
    result = evaluate()
    invalid = result["invalid_protocol"]
    assert invalid["availability_check_rejected"]
    assert not invalid["eligible_for_selection"]
    assert invalid["case"] not in result["cv"]
    for name, values in result["predictions"].items():
        assert len(values) == 36
        if name != "quarter":
            assert np.isfinite(values).all()
            assert np.all((np.array(values) >= 0) & (np.array(values) <= 1))


def test_caller_supplied_data_does_not_inherit_generator_provenance():
    raw = synthetic_quarters(seed=123)
    raw.index += 100
    result = evaluate(raw=raw)
    assert result["seed"] is None
    assert result["data_kind"] == "caller_supplied_unverified_origin"
    assert result["protocol"]["regime_shift_quarter"] is None
    assert result["protocol"]["holdout_quarters"] == [212, 247]
    assert result["protocol"]["final_training_quarters"] == [140, 211]
