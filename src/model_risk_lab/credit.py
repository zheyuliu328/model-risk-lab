"""Synthetic quarterly default-rate prediction, with observable information at t-1.

The mechanism and evaluation plan are fixed in docs/CREDIT_METHOD.md.
This is logit-transformed rate regression, not an IFRS 9 calculation or a binomial GLM.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CORE = ["odr_logit_lag1", "unemployment_lag1", "growth_lag2", "rate_lag1", "growth_mean4"]
NOISE = [f"noise_{i:02d}_lag1" for i in range(32)]
CANDIDATES = ("ols", "ridge_0.1", "ridge_1", "ridge_10", "complex_ols")
BASELINES = ("last_quarter", "train_mean")


def synthetic_quarters(seed: int = 20260907, n: int = 148) -> pd.DataFrame:
    """Invented mechanism and parameters; no data are read or downloaded."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 8:
        raise ValueError("n must be an integer >= 8")
    rng = np.random.default_rng(seed)
    macro = np.zeros((n, 3))
    z = np.full(n, -4.5)
    for t in range(1, n):
        macro[t] = np.array([0.65, 0.75, 0.55]) * macro[t-1] + rng.normal(0, 0.6, 3)
        g2 = macro[max(0, t-2), 0]
        z[t] = (-4.5 + 0.55 * (z[t-1] + 4.5) + 0.30 * macro[t-1, 1]
                - 0.40 * g2 + 0.15 * macro[t-1, 2] + 0.20 * rng.normal()
                + (0.60 if t >= 124 else 0.0))
    df = pd.DataFrame(macro, columns=["growth", "unemployment", "rate"])
    df.index.name = "quarter"
    df["borrowers"] = 10000
    df["defaults"] = rng.binomial(10000, expit(z))
    # Drawn after the target process, so noise dimensions cannot alter the target RNG stream.
    for i in range(32):
        df[f"noise_{i:02d}"] = rng.normal(size=n)
    return df


def validate_raw(raw: pd.DataFrame) -> None:
    required = ["growth", "unemployment", "rate", "defaults", "borrowers"]
    required += [f"noise_{i:02d}" for i in range(32)]
    if not set(required).issubset(raw.columns) or len(raw) < 8:
        raise ValueError("missing required inputs or insufficient quarters")
    if not raw.index.is_unique or not raw.index.is_monotonic_increasing:
        raise ValueError("quarters must be unique and ordered")
    idx = raw.index.to_numpy()
    if not np.issubdtype(idx.dtype, np.integer) or not np.all(np.diff(idx) == 1):
        raise ValueError("quarters must be consecutive integer periods")
    values = raw[required].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("missing/nonfinite raw inputs are not silently dropped")
    d, n = raw.defaults.to_numpy(), raw.borrowers.to_numpy()
    if np.any(n <= 0) or np.any(d < 0) or np.any(d > n):
        raise ValueError("counts must satisfy 0 <= defaults <= borrowers and borrowers > 0")
    if np.any(d != np.floor(d)) or np.any(n != np.floor(n)):
        raise ValueError("borrower/default counts must be integers")


def smoothed_logit(defaults, borrowers):
    return logit((np.asarray(defaults) + 0.5) / (np.asarray(borrowers) + 1.0))


@dataclass
class Design:
    features: pd.DataFrame
    target_logit: pd.Series
    observed_rate: pd.Series
    last_rate: pd.Series
    source_quarters: pd.DataFrame


def check_availability(features: pd.DataFrame, sources: pd.DataFrame) -> None:
    if not features.index.equals(sources.index) or not features.columns.equals(sources.columns):
        raise ValueError("feature/source metadata must match exactly")
    if not np.isfinite(features.to_numpy()).all() or not np.isfinite(sources.to_numpy()).all():
        raise ValueError("features and provenance must be finite")
    if not np.all(sources.to_numpy() < features.index.to_numpy()[:, None]):
        raise ValueError("feature contains information unavailable at the prediction origin")


def build_design(raw: pd.DataFrame) -> Design:
    validate_raw(raw)
    y = pd.Series(smoothed_logit(raw.defaults, raw.borrowers), index=raw.index)
    rate = raw.defaults / raw.borrowers
    x = pd.DataFrame({
        "odr_logit_lag1": y.shift(1), "unemployment_lag1": raw.unemployment.shift(1),
        "growth_lag2": raw.growth.shift(2), "rate_lag1": raw.rate.shift(1),
        "growth_mean4": raw.growth.shift(1).rolling(4).mean(),
        **{f"noise_{i:02d}_lag1": raw[f"noise_{i:02d}"].shift(1) for i in range(32)},
    }).iloc[4:]
    sources = pd.DataFrame({name: x.index - (2 if name == "growth_lag2" else 1)
                            for name in x.columns}, index=x.index)
    check_availability(x, sources)
    return Design(x, y.loc[x.index], rate.loc[x.index], rate.shift(1).loc[x.index], sources)


def features_for(name: str) -> list[str]:
    return CORE + NOISE if name == "complex_ols" else CORE


def fit_candidate(name: str, x: pd.DataFrame, y: pd.Series):
    if name not in CANDIDATES:
        raise ValueError("unknown candidate")
    estimator = Ridge(alpha=float(name.split("_")[1])) if name.startswith("ridge_") else LinearRegression()
    model = make_pipeline(StandardScaler(), estimator)
    model.fit(x[features_for(name)], y)
    return model


def rate_metrics(actual, predicted) -> dict[str, float]:
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if actual.shape != predicted.shape or not actual.size:
        raise ValueError("metrics require equal nonempty shapes")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("metrics require finite observations and predictions")
    residual = predicted - actual
    return {"mae_bp": float(np.mean(abs(residual)) * 10000),
            "rmse_bp": float(np.sqrt(np.mean(residual**2)) * 10000),
            "bias_bp": float(np.mean(residual) * 10000)}


def r_squared(actual, predicted) -> dict:
    a, p = np.asarray(actual), np.asarray(predicted)
    if a.shape != p.shape or not np.isfinite(a).all() or not np.isfinite(p).all():
        raise ValueError("R2 requires matching finite data")
    total = float(np.sum((a - a.mean())**2)) if len(a) else 0.0
    if len(a) < 2 or total == 0:
        return {"value": None, "status": "undefined_constant_or_short_target"}
    return {"value": float(1 - np.sum((a-p)**2) / total), "status": "defined"}


def development_splits(design: Design):
    if len(design.features) != 144:
        raise ValueError("the fixed experiment requires exactly 144 usable quarters")
    return list(TimeSeriesSplit(n_splits=5, test_size=12, max_train_size=72).split(design.features.iloc[:108]))


def evaluate(seed: int = 20260907, raw: pd.DataFrame | None = None) -> dict:
    """Tune using development folds only; then reveal one locked holdout evaluation."""
    input_kind = "synthetic" if raw is None else "caller_supplied_unverified_origin"
    raw = synthetic_quarters(seed) if raw is None else raw
    d = build_design(raw)
    folds, records, pooled = development_splits(d), [], {}
    names = BASELINES + CANDIDATES
    for name in names:
        pooled[name] = {"actual": [], "prediction": [], "actual_logit": [], "prediction_logit": []}
        for fold, (train, test) in enumerate(folds, 1):
            xtrain, xtest = d.features.iloc[train], d.features.iloc[test]
            ytrain = d.target_logit.iloc[train]
            if name in BASELINES:
                pred = (d.last_rate.iloc[test].to_numpy() if name == "last_quarter"
                        else np.full(len(test), d.observed_rate.iloc[train].mean()))
                plogit = logit(np.clip(pred, 1e-10, 1-1e-10))
                train_r2, scaler_mean = None, None
            else:
                model = fit_candidate(name, xtrain, ytrain)
                plogit = model.predict(xtest[features_for(name)])
                pred = expit(plogit)
                train_r2 = r_squared(ytrain, model.predict(xtrain[features_for(name)]))
                scaler_mean = model[0].mean_.tolist()
            actual = d.observed_rate.iloc[test].to_numpy()
            records.append({"model": name, "fold": fold,
                            "train_start": int(xtrain.index[0]), "train_end": int(xtrain.index[-1]),
                            "test_start": int(xtest.index[0]), "test_end": int(xtest.index[-1]),
                            "n_train": len(train), "n_test": len(test),
                            **rate_metrics(actual, pred), "train_r2_logit": train_r2,
                            "scaler_mean": scaler_mean})
            for key, val in [("actual", actual), ("prediction", pred),
                             ("actual_logit", d.target_logit.iloc[test]), ("prediction_logit", plogit)]:
                pooled[name][key].extend(np.asarray(val).tolist())
    cv = {name: {**rate_metrics(p["actual"], p["prediction"]),
                 "r2_logit": r_squared(p["actual_logit"], p["prediction_logit"])}
          for name, p in pooled.items()}
    selected = min(CANDIDATES, key=lambda name: cv[name]["mae_bp"])
    # Candidate is fixed before any holdout target is scored; same final training window for all.
    train, test = np.arange(36, 108), np.arange(108, 144)
    holdout, predictions = {}, {"quarter": d.features.index[test].tolist(),
                               "actual": d.observed_rate.iloc[test].tolist()}
    for name in names:
        if name in BASELINES:
            pred = (d.last_rate.iloc[test].to_numpy() if name == "last_quarter"
                    else np.full(len(test), d.observed_rate.iloc[train].mean()))
        else:
            model = fit_candidate(name, d.features.iloc[train], d.target_logit.iloc[train])
            pred = expit(model.predict(d.features.iloc[test][features_for(name)]))
        predictions[name] = pred.tolist()
        holdout[name] = rate_metrics(d.observed_rate.iloc[test], pred)
    # Deliberate invalid protocol: the current target itself is included as a feature.
    # Never part of ranking; this proves time splitting alone does not establish availability.
    invalid = []
    bad = d.features[CORE].copy()
    bad["current_target"] = d.target_logit
    bad_sources = d.source_quarters[CORE].copy()
    bad_sources["current_target"] = bad.index
    try:
        check_availability(bad, bad_sources)
        availability_rejected = False
    except ValueError:
        availability_rejected = True
    for train, test in folds:
        leaky = make_pipeline(StandardScaler(), LinearRegression()).fit(bad.iloc[train], d.target_logit.iloc[train])
        invalid.append(rate_metrics(d.observed_rate.iloc[test], expit(leaky.predict(bad.iloc[test]))))
    return {"seed": seed if input_kind == "synthetic" else None,
            "data_kind": input_kind, "selected_on_development": selected,
            "protocol": {"usable_quarters": 144, "development": 108, "holdout": 36,
                         "final_training_quarters": [int(d.features.index[36]), int(d.features.index[107])],
                         "holdout_quarters": [int(d.features.index[108]), int(d.features.index[143])],
                         "information": "one-step prediction, observed history updated each quarter; coefficients fixed within test blocks",
                         "regime_shift_quarter": 124 if input_kind == "synthetic" else None},
            "cv": cv, "folds": records, "holdout": holdout, "predictions": predictions,
            "invalid_protocol": {"case": "contemporaneous_target_feature",
                                 "eligible_for_selection": False,
                                 "availability_check_rejected": availability_rejected,
                                 "mean_fold_mae_bp": float(np.mean([m["mae_bp"] for m in invalid]))}}
