"""Fixed baselines and WTI-only logistic regression fitted on supplied rows.

This module cannot choose training dates. The replay caller must first select
historical rows and labels available at the current forecast cutoff.
"""

import math
from numbers import Integral, Real
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from oil_audit.features import WTI_FEATURE_COLUMNS


MODEL_NAMES = ("majority", "momentum", "logistic")
LOGISTIC_PARAMETERS = {
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 2000,
    "random_state": 0,
}


def _feature_vector(row: dict) -> list[float]:
    """Allow only the six WTI fields; retain explicit nulls for imputation."""
    features = row["features"]
    result = []
    for column in WTI_FEATURE_COLUMNS:
        if column not in features:
            raise ValueError(f"Missing required WTI feature column: {column}")
        value = features[column]
        if value is None:
            result.append(float("nan"))
        elif isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"WTI feature {column} must be finite numeric data or null")
        else:
            result.append(float(value))
    return result


def _training_label(row: dict) -> int:
    value = row["label"]
    if isinstance(value, bool) or not isinstance(value, Integral) or value not in (0, 1):
        raise ValueError("Training labels must be integers 0 or 1")
    return int(value)


def fit_predict_models(training_rows: list[dict], forecast_row: dict) -> dict:
    """Fit anew on training rows, then predict one untouched forecast row.

    Majority ties and logistic probabilities of exactly 0.5 predict up.
    Momentum uses the sign of the unscaled five-business-day dollar change;
    a zero change predicts not-up and a missing change falls back to majority.
    Momentum is a direction rule and has no estimated probability.

    Median imputation and standardization are fitted only on training rows.
    A column missing throughout training stays in place and is imputed as zero
    by SimpleImputer(keep_empty_features=True). No missing indicators are added.
    Single-class training sets and convergence failures raise explicit errors.
    """
    if not training_rows:
        raise ValueError("Cannot fit models without training rows")
    labels = [_training_label(row) for row in training_rows]
    if len(set(labels)) != 2:
        raise ValueError("Logistic regression requires both training label classes")
    train = np.asarray([_feature_vector(row) for row in training_rows], dtype=float)
    forecast = np.asarray([_feature_vector(forecast_row)], dtype=float)
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(**LOGISTIC_PARAMETERS)),
    ])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(train, labels)
    except ConvergenceWarning as exc:
        raise RuntimeError("Logistic regression did not converge; evaluation stopped") from exc
    probability = float(model.predict_proba(forecast)[0, 1])
    if not math.isfinite(probability):
        raise RuntimeError("Logistic regression produced a non-finite probability")
    prevalence = sum(labels) / len(labels)
    majority = int(prevalence >= 0.5)
    momentum = forecast_row["features"]["wti_change_5_business_days_usd"]
    imputer = model.named_steps["imputer"]
    scaler = model.named_steps["scaler"]
    classifier = model.named_steps["classifier"]
    return {
        "majority": {
            "prediction": majority,
            "probability_up": prevalence,
        },
        "momentum": {
            "prediction": majority if momentum is None else int(momentum > 0),
            "probability_up": None,
            "used_majority_fallback": momentum is None,
        },
        "logistic": {
            "prediction": int(probability >= 0.5),
            "probability_up": probability,
            "diagnostics": {
                "feature_columns": list(WTI_FEATURE_COLUMNS),
                "imputer_statistics": dict(zip(WTI_FEATURE_COLUMNS, imputer.statistics_.tolist())),
                "scaler_mean": dict(zip(WTI_FEATURE_COLUMNS, scaler.mean_.tolist())),
                "scaler_scale": dict(zip(WTI_FEATURE_COLUMNS, scaler.scale_.tolist())),
                "coef": dict(zip(WTI_FEATURE_COLUMNS, classifier.coef_[0].tolist())),
                "intercept": float(classifier.intercept_[0]),
                "iterations": int(classifier.n_iter_[0]),
            },
        },
    }
