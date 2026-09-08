"""Small, auditable monthly OLS experiments using only NumPy and the standard library.

Inputs are raw observations dated by observation month.  Each feature is shifted
by its declared lag before any sample selection, scaling, or fit.  This module
does not train on the holdout, infer publication dates, transform the target,
fill missing observations, or certify that external forecasts are independent.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

MAX_ROWS = 2400
MAX_ABS_VALUE = 1e12
MAX_CONDITION = 1e8
_PERIOD = re.compile(r"[0-9]{4}-(0[1-9]|1[0-2])\Z")
_FEATURE_ID = re.compile(r"f[1-5]\Z")


def _month(value: Any, label: str) -> int:
    if not isinstance(value, str) or not _PERIOD.fullmatch(value):
        raise ValueError(f"{label} must be a YYYY-MM calendar month")
    year, month = (int(part) for part in value.split("-"))
    if year < 1:
        raise ValueError(f"{label} must have a year from 0001 to 9999")
    return year * 12 + month - 1


def _period(month: int) -> str:
    year, offset = divmod(month, 12)
    return f"{year:04d}-{offset + 1:02d}"


def _integer(value: Any, label: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{label} must be an integer from {low} to {high}")
    return value


def _text(value: Any, label: str, limit: int = 200) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{label} must be nonempty text of at most {limit} characters")
    return value


def _number(value: Any, label: str, missing: bool = False) -> float | None:
    if value is None and missing:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number" + (" or null" if missing else ""))
    try:
        numeric = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{label} must be finite and bounded by {MAX_ABS_VALUE:g}") from None
    if not math.isfinite(numeric) or abs(numeric) > MAX_ABS_VALUE:
        raise ValueError(f"{label} must be finite and bounded by {MAX_ABS_VALUE:g}")
    return numeric


def _keys(value: Any, allowed: set[str], label: str, required: set[str]) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    if set(value) - allowed:
        raise ValueError(f"{label} has unrecognized fields")
    if required - set(value):
        raise ValueError(
            f"{label} is missing required fields: {', '.join(sorted(required - set(value)))}"
        )


def _normalise(request: Any) -> dict:
    required = {
        "schema_version",
        "title",
        "rows",
        "features",
        "horizon",
        "development_end",
        "target",
    }
    allowed = required | {"n_splits", "validation_months", "min_train", "source_note"}
    _keys(request, allowed, "request", required)
    _integer(request["schema_version"], "schema_version", 1, 1)
    horizon = _integer(request["horizon"], "horizon", 1, 12)
    features = request["features"]
    if not isinstance(features, list) or not 1 <= len(features) <= 5:
        raise ValueError("features must contain from one to five declared features")
    normal_features = []
    for feature in features:
        _keys(feature, {"id", "name", "lag", "release_delay"}, "feature", {"id", "name", "lag"})
        feature_id = feature["id"]
        if not isinstance(feature_id, str) or not _FEATURE_ID.fullmatch(feature_id):
            raise ValueError("feature id must be one of f1, f2, f3, f4, f5")
        delay = _integer(feature.get("release_delay", 0), "release_delay", 0, MAX_ROWS)
        lag = _integer(feature["lag"], "lag", 1, MAX_ROWS)
        if lag < horizon + delay:
            raise ValueError(f"{feature_id}: lag must be at least horizon + release_delay")
        normal_features.append(
            {
                "id": feature_id,
                "name": _text(feature["name"], "feature name"),
                "lag": lag,
                "release_delay": delay,
            }
        )
    normal_features.sort(key=lambda item: item["id"])
    feature_ids = [item["id"] for item in normal_features]
    if len(set(feature_ids)) != len(feature_ids):
        raise ValueError("feature ids must be unique")
    target = request["target"]
    _keys(target, {"name", "unit", "transformation"}, "target", {"name", "unit", "transformation"})
    target = {key: _text(target[key], f"target {key}") for key in sorted(target)}
    rows = request["rows"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_ROWS:
        raise ValueError(f"rows must contain from 1 to {MAX_ROWS} monthly observations")
    normal_rows = []
    for row in rows:
        _keys(row, {"period", "target", "features"}, "row", {"period", "target", "features"})
        _month(row["period"], "row period")
        _keys(row["features"], set(feature_ids), "row features", set(feature_ids))
        normal_rows.append(
            {
                "period": row["period"],
                "target": _number(row["target"], f"target at {row['period']}"),
                "features": {
                    key: _number(row["features"][key], f"{key} at {row['period']}", missing=True)
                    for key in feature_ids
                },
            }
        )
    normal_rows.sort(key=lambda item: item["period"])
    months = [_month(row["period"], "row period") for row in normal_rows]
    if len(set(months)) != len(months):
        raise ValueError("monthly periods must be unique; duplicate rows are not merged")
    if months != list(range(months[0], months[-1] + 1)):
        raise ValueError(
            "monthly periods must be continuous; missing calendar months are not filled"
        )
    development_end = _month(request["development_end"], "development_end")
    if not months[0] <= development_end < months[-1]:
        raise ValueError(
            "development_end must be present and leave at least one explicit holdout month"
        )
    n_splits = _integer(request.get("n_splits", 3), "n_splits", 1, 24)
    validation_months = _integer(
        request.get("validation_months", 12), "validation_months", 1, MAX_ROWS
    )
    if n_splits * validation_months > development_end - months[0] + 1:
        raise ValueError(
            "development period is shorter than the requested calendar validation windows"
        )
    source_note = request.get("source_note", "")
    if not isinstance(source_note, str) or len(source_note) > 4000:
        raise ValueError("source_note must be text of at most 4000 characters")
    return {
        "schema_version": 1,
        "title": _text(request["title"], "title"),
        "rows": normal_rows,
        "features": normal_features,
        "horizon": horizon,
        "development_end": request["development_end"],
        "n_splits": n_splits,
        "validation_months": validation_months,
        "min_train": _integer(request.get("min_train", 36), "min_train", 3, MAX_ROWS),
        "target": target,
        "source_note": source_note,
    }


def _fingerprint(value: dict) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict | None:
    if len(actual) == 0:
        return None
    residual = predicted - actual
    absolute = np.abs(residual)
    largest = float(np.max(absolute))
    # Normalize before squaring/summing: an extreme but finite extrapolation
    # need not overflow merely because a naive squared-error sum would.
    rmse = largest * float(np.sqrt(np.mean((residual / largest) ** 2))) if largest else 0.0
    return {
        "n": int(len(actual)),
        "mae": float(np.sum(absolute / len(actual))),
        "rmse": rmse,
        "bias": float(np.sum(residual / len(actual))),
    }


@dataclass
class _Fit:
    public: dict
    coefficients: np.ndarray | None = None
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None


def _r2(
    actual: np.ndarray, predicted: np.ndarray, predictors: int
) -> tuple[float | None, float | None]:
    centered = actual - np.mean(actual)
    sst = float(centered @ centered)
    if sst == 0.0:
        return None, None
    residual = actual - predicted
    value = 1.0 - float(residual @ residual) / sst
    adjusted = 1.0 - (1.0 - value) * (len(actual) - 1) / (len(actual) - predictors - 1)
    return value, adjusted


def _fit(
    model: dict,
    x: np.ndarray,
    y: np.ndarray,
    train: list[int],
    periods: list[str],
    feature_ids: list[str],
    min_train: int,
) -> tuple[_Fit | None, str | None]:
    selected_features = model["features"]
    n = len(train)
    if n < min_train or n <= len(selected_features) + 1:
        return (
            None,
            f"insufficient_training_sample: n={n}, minimum={max(min_train, len(selected_features) + 2)}",
        )
    actual = y[train]
    public = {
        "n": n,
        "train_start": periods[train[0]],
        "train_end": periods[train[-1]],
        "train_periods": [periods[index] for index in train],
        "intercept": None,
        "coefficients": {},
        "feature_means": {},
        "feature_scales": {},
        "target_mean": float(np.mean(actual)),
        "r2": None,
        "adjusted_r2": None,
        "condition_number": None,
        "vif": {},
    }
    if model["id"] == "baseline-persistence":
        public["method"] = (
            "No fitted parameters; use y[t-horizon] at each target month's prediction origin."
        )
        return _Fit(public), None
    if model["id"] == "baseline-mean":
        public["intercept"] = public["target_mean"]
        public["r2"], public["adjusted_r2"] = _r2(actual, np.full(n, public["target_mean"]), 0)
        public["condition_number"] = 1.0
        return _Fit(public), None
    columns = [feature_ids.index(key) for key in selected_features]
    features = x[np.ix_(train, columns)]
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0, ddof=0)
    if np.any(np.ptp(features, axis=0) == 0) or np.any(scale == 0):
        return None, "constant_feature: at least one feature has zero training variation"
    standardised = (features - mean) / scale
    design = np.column_stack((np.ones(n), standardised))
    try:
        singular = np.linalg.svd(design, compute_uv=False)
        rank = int(
            np.count_nonzero(singular > singular[0] * max(design.shape) * np.finfo(float).eps)
        )
        if rank < design.shape[1]:
            return None, "rank_deficient: training design columns are linearly dependent"
        condition = float(singular[0] / singular[-1])
        if not math.isfinite(condition) or condition > MAX_CONDITION:
            return None, f"ill_conditioned: standardised design condition exceeds {MAX_CONDITION:g}"
        coefficients = np.linalg.lstsq(design, actual, rcond=None)[0]
        prediction = design @ coefficients
        raw_coefficients = coefficients[1:] / scale
        intercept = float(coefficients[0] - mean @ raw_coefficients)
        gram = standardised.T @ standardised / n
        vif = np.diag(np.linalg.inv(gram))
        r2, adjusted_r2 = _r2(actual, prediction, len(columns))
    except np.linalg.LinAlgError:
        return None, "numerical_failure: linear algebra did not converge"
    checks = [*coefficients, *raw_coefficients, intercept, *vif]
    if not all(math.isfinite(float(value)) for value in checks):
        return None, "numerical_failure: nonfinite fitted parameters"
    if any(value is not None and not math.isfinite(value) for value in (r2, adjusted_r2)):
        return None, "numerical_failure: nonfinite training diagnostics"
    public.update(
        {
            "intercept": intercept,
            "coefficients": dict(zip(selected_features, map(float, raw_coefficients))),
            "feature_means": dict(zip(selected_features, map(float, mean))),
            "feature_scales": dict(zip(selected_features, map(float, scale))),
            "r2": r2,
            "adjusted_r2": adjusted_r2,
            "condition_number": condition,
            "vif": dict(zip(selected_features, (max(1.0, float(value)) for value in vif))),
        }
    )
    return _Fit(public, coefficients, mean, scale), None


def _predict(
    model: dict,
    fitted: _Fit,
    x: np.ndarray,
    y: np.ndarray,
    indexes: list[int],
    feature_ids: list[str],
    horizon: int,
) -> np.ndarray:
    if model["id"] == "baseline-mean":
        return np.full(len(indexes), fitted.public["intercept"], dtype=float)
    if model["id"] == "baseline-persistence":
        return y[[index - horizon for index in indexes]].copy()
    columns = [feature_ids.index(key) for key in model["features"]]
    features = x[np.ix_(indexes, columns)]
    scaled = (features - fitted.mean) / fitted.scale
    return np.column_stack((np.ones(len(indexes)), scaled)) @ fitted.coefficients


def _finite_predictions(actual: np.ndarray, predicted: np.ndarray) -> bool:
    with np.errstate(over="ignore", invalid="ignore"):
        residual = predicted - actual
        return bool(np.all(np.isfinite(predicted)) and np.all(np.isfinite(residual)))


def run_experiment(request: dict, reveal_holdout: bool = False) -> dict:
    """Evaluate all single/pair OLS models without allowing holdout reselection.

    Invalid schemas, unavailable declared lags, missing targets, and broken
    monthly calendars raise ValueError.  Missing feature values exclude the
    corresponding *lagged* rows for every candidate and both baselines.  Model
    failures remain in the returned candidates and fold diagnostics.
    """
    if not isinstance(reveal_holdout, bool):
        raise ValueError("reveal_holdout must be a boolean")
    normal = _normalise(request)
    raw = normal["rows"]
    features = normal["features"]
    feature_ids = [item["id"] for item in features]
    periods = [row["period"] for row in raw]
    months = [_month(period, "period") for period in periods]
    end = _month(normal["development_end"], "development_end")
    horizon = normal["horizon"]
    final_cutoff = end + 1 - horizon
    first_validation = end + 1 - normal["n_splits"] * normal["validation_months"]
    x = np.full((len(raw), len(features)), np.nan, dtype=float)
    y = np.array([row["target"] for row in raw], dtype=float)
    input_rows = []
    usable = []
    for index, row in enumerate(raw):
        reasons, lagged, source_periods = [], {}, {}
        for column, feature in enumerate(features):
            key = feature["id"]
            source = index - feature["lag"]
            source_periods[key] = _period(months[index] - feature["lag"])
            if source < 0:
                lagged[key] = None
                reasons.append(
                    {"code": "warmup", "feature": key, "source_period": source_periods[key]}
                )
            else:
                value = raw[source]["features"][key]
                lagged[key] = value
                if value is None:
                    reasons.append(
                        {
                            "code": "missing_lagged_feature",
                            "feature": key,
                            "source_period": source_periods[key],
                        }
                    )
                else:
                    x[index, column] = value
        ok = not reasons
        usable.append(ok)
        input_rows.append(
            {
                **row,
                "lagged_features": lagged,
                "feature_source_periods": source_periods,
                "status": "usable" if ok else "excluded",
                "reasons": reasons,
                "split": "development" if months[index] <= end else "holdout",
                "eligible_for_cv": ok and first_validation <= months[index] <= end,
                "eligible_for_final_fit": ok and months[index] <= final_cutoff,
            }
        )
    fold_windows = []
    for fold in range(normal["n_splits"]):
        start = first_validation + fold * normal["validation_months"]
        stop = start + normal["validation_months"] - 1
        train = [
            index
            for index, month in enumerate(months)
            if usable[index] and month <= start - horizon
        ]
        validation = [
            index for index, month in enumerate(months) if usable[index] and start <= month <= stop
        ]
        fold_windows.append(
            {
                "fold": fold + 1,
                "start": start,
                "stop": stop,
                "train": train,
                "validation": validation,
            }
        )
    final_train = [
        index for index, month in enumerate(months) if usable[index] and month <= final_cutoff
    ]
    holdout = [index for index, month in enumerate(months) if usable[index] and month > end]
    validation_indexes = [index for window in fold_windows for index in window["validation"]]
    models = []
    for count in (1, 2):
        for subset in itertools.combinations(feature_ids, count):
            names = [feature["name"] for feature in features if feature["id"] in subset]
            models.append(
                {
                    "id": "ols-" + "-".join(subset),
                    "name": "OLS: " + " + ".join(names),
                    "role": "candidate",
                    "features": list(subset),
                }
            )
    models.extend(
        [
            {"id": "baseline-mean", "name": "Training mean", "role": "baseline", "features": []},
            {
                "id": "baseline-persistence",
                "name": "Lagged actual (persistence)",
                "role": "baseline",
                "features": [],
            },
        ]
    )
    candidates, folds = [], []
    cv_predictions, holdout_predictions = {}, {}
    for model in models:
        failures, values = [], []
        for window in fold_windows:
            train, validation = window["train"], window["validation"]
            fitted, reason = _fit(model, x, y, train, periods, feature_ids, normal["min_train"])
            predicted, metrics = None, None
            if not validation:
                reason = (
                    "insufficient_validation_sample: no common usable rows in this calendar fold"
                )
            if reason is None:
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    predicted = _predict(model, fitted, x, y, validation, feature_ids, horizon)
                if not _finite_predictions(y[validation], predicted):
                    reason = "numerical_failure: validation predictions or residuals are nonfinite"
                else:
                    metrics = _metrics(y[validation], predicted)
                    values.extend(map(float, predicted))
            if reason:
                failures.append(f"fold {window['fold']}: {reason}")
            folds.append(
                {
                    "model_id": model["id"],
                    "fold": window["fold"],
                    "status": "failed" if reason else "ok",
                    "reason": reason,
                    "training_cutoff": _period(window["start"] - horizon),
                    "train_periods": [periods[index] for index in train],
                    "validation_start": _period(window["start"]),
                    "validation_end": _period(window["stop"]),
                    "validation_periods": [periods[index] for index in validation],
                    "development": metrics,
                    "fit": fitted.public if fitted else None,
                }
            )
        final_fit, final_reason = _fit(
            model, x, y, final_train, periods, feature_ids, normal["min_train"]
        )
        if final_reason:
            failures.append(f"final fit: {final_reason}")
        development, holdout_metrics, holdout_reason = None, None, None
        if not failures:
            cv_predictions[model["id"]] = values
            development = _metrics(y[validation_indexes], np.array(values))
            if reveal_holdout and holdout:
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    prediction = _predict(model, final_fit, x, y, holdout, feature_ids, horizon)
                if _finite_predictions(y[holdout], prediction):
                    holdout_predictions[model["id"]] = list(map(float, prediction))
                    holdout_metrics = _metrics(y[holdout], prediction)
                else:
                    # Holdout evaluation cannot retroactively invalidate a development model.
                    holdout_reason = (
                        "numerical_failure: holdout predictions or residuals are nonfinite"
                    )
            elif reveal_holdout:
                holdout_reason = "no_common_holdout_rows"
        candidates.append(
            {
                **model,
                "status": "failed" if failures else "ok",
                "reason": "; ".join(failures) if failures else None,
                "development": development,
                "holdout": holdout_metrics,
                "holdout_reason": holdout_reason,
                "fit": final_fit.public if final_fit else None,
            }
        )
    ranking = sorted(
        (
            candidate
            for candidate in candidates
            if candidate["role"] == "candidate" and candidate["status"] == "ok"
        ),
        key=lambda candidate: (candidate["development"]["mae"], candidate["id"]),
    )
    selected = ranking[0]["id"] if ranking else None
    predictions = []
    offset = 0
    for window in fold_windows:
        for index in window["validation"]:
            predictions.append(
                {
                    "period": periods[index],
                    "actual": float(y[index]),
                    "split": "development",
                    "fold": window["fold"],
                    "predictions": {key: values[offset] for key, values in cv_predictions.items()},
                }
            )
            offset += 1
    if reveal_holdout:
        for offset, index in enumerate(holdout):
            predictions.append(
                {
                    "period": periods[index],
                    "actual": float(y[index]),
                    "split": "holdout",
                    "fold": None,
                    "predictions": {
                        key: values[offset] for key, values in holdout_predictions.items()
                    },
                }
            )
    development_indexes = [index for index, month in enumerate(months) if month <= end]
    holdout_indexes = [index for index, month in enumerate(months) if month > end]

    def count(indexes: list[int]) -> dict:
        valid = sum(usable[index] for index in indexes)
        return {"raw": len(indexes), "usable": valid, "excluded": len(indexes) - valid}

    selection_input = {key: value for key, value in normal.items() if key not in {"rows", "title"}}
    selection_input["rows"] = [raw[index] for index in development_indexes]
    selection_input["algorithm"] = "monthly-ols-v1"
    result = {
        "schema_version": 1,
        "title": normal["title"],
        "stage": "holdout" if reveal_holdout else "development",
        "selected_on_development": selected,
        "selection_fingerprint": _fingerprint(selection_input),
        "data_fingerprint": _fingerprint(normal),
        "protocol": {
            "algorithm": "monthly-ols-v1",
            "target": normal["target"],
            "source_note": normal["source_note"],
            "horizon": horizon,
            "feature_lags": features,
            "development_end": normal["development_end"],
            "holdout_start": _period(end + 1),
            "final_training_cutoff": _period(final_cutoff),
            "n_splits": normal["n_splits"],
            "validation_months": normal["validation_months"],
            "min_train": normal["min_train"],
            "first_validation_period": _period(first_validation),
            "ranking": [candidate["id"] for candidate in ranking],
            "selection_rule": "Minimum pooled development MAE among successful OLS candidates, breaking exact ties by stable model id; baselines are comparisons only.",
            "selection_reason": "selected_on_development_only"
            if selected
            else "all_ols_candidates_failed",
            "training_rule": "Use all common usable target periods <= first validation/holdout month - horizon. Refit only between development folds; keep final coefficients fixed throughout holdout.",
            "scaling_rule": "Feature centering and population standard deviation are fitted only on each training window. Report coefficients and intercept in original input units.",
            "availability_rule": "For target month t, use feature observation t-lag with lag >= horizon + release_delay. Target y[t-horizon] is assumed available at each prediction origin.",
            "sample_rule": "All declared lagged features must be present even for single-feature candidates and baselines. The development and holdout common samples are constructed separately; missing observations are never filled.",
            "persistence_rule": "Persistence uses observed y[t-horizon] at each rolling prediction origin, including earlier validation/holdout actuals once available. This is an updating historical baseline, not a single-origin multi-step forecast.",
            "metrics_rule": "Equal weight per common target month; pooled RMSE is sqrt(mean squared error), not the mean of fold RMSEs. Bias = prediction - actual.",
            "diagnostics_rule": "R2 is undefined for a constant training target. Condition number uses the intercept plus standardized design; VIF uses standardized feature correlations. No p-values or significance claims.",
            "holdout_revealed": reveal_holdout,
            "limits": {
                "max_rows": MAX_ROWS,
                "max_features": 5,
                "max_abs_value": MAX_ABS_VALUE,
                "max_condition_number": MAX_CONDITION,
            },
            "disclosures": [
                "This tool screens linear candidates; it does not provide model approval, causal inference, or financial validation.",
                "Target transformation is a caller declaration: the supplied values are scored as-is, with no automatic transform or inverse transform.",
                "Publication delays and target availability are caller declarations, not independently verified evidence.",
                "Development tuning may be optimistic. The holdout must not be reused to select features, parameters, or a new winning candidate.",
                "A model failing any development fold or the final training fit is retained but excluded from pooled development comparison and selection.",
            ],
        },
        "coverage": {
            **count(list(range(len(raw)))),
            "development": {
                **count(development_indexes),
                "validation_usable": len(validation_indexes),
            },
            "holdout": count(holdout_indexes),
        },
        "candidates": candidates,
        "folds": folds,
        "input_rows": input_rows,
        "predictions": predictions,
    }
    # Reject accidental NumPy scalars or NaN/Infinity before crossing the API boundary.
    json.dumps(result, allow_nan=False)
    return result


def example_request() -> dict:
    """Return 180 invented months, five lagged features, and a 24-month holdout."""
    rng = np.random.default_rng(20260909)
    values = np.zeros((180, 5), dtype=float)
    innovations = rng.normal(size=values.shape)
    for index in range(180):
        previous = values[index - 1] if index else np.zeros(5)
        values[index] = 0.55 * previous + innovations[index]
    lags = [1, 2, 3, 1, 2]
    target = 25 + rng.normal(scale=0.6, size=180)
    for index in range(180):
        if index >= 1:
            target[index] += 2.5 * values[index - 1, 0]
        if index >= 2:
            target[index] -= 1.5 * values[index - 2, 1]
    start = _month("2010-01", "example start")
    return {
        "schema_version": 1,
        "title": "Synthetic monthly OLS candidate screen",
        "rows": [
            {
                "period": _period(start + index),
                "target": float(target[index]),
                "features": {f"f{column + 1}": float(values[index, column]) for column in range(5)},
            }
            for index in range(180)
        ],
        "features": [
            {
                "id": f"f{column + 1}",
                "name": name,
                "lag": lags[column],
                "release_delay": lags[column] - 1,
            }
            for column, name in enumerate(["Growth", "Demand", "Costs", "Activity", "Capacity"])
        ],
        "horizon": 1,
        "development_end": "2022-12",
        "n_splits": 3,
        "validation_months": 12,
        "min_train": 36,
        "target": {"name": "Synthetic demand", "unit": "index points", "transformation": "none"},
        "source_note": "Entirely synthetic data generated locally with a fixed NumPy random seed; no client or external model data.",
    }
