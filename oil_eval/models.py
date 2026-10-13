"""Fixed baselines and logistic regressions fitted on supplied rows.

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

from oil_audit.features import FEATURE_COLUMNS, WTI_FEATURE_COLUMNS
from .inventory_features import INVENTORY_MODEL_COLUMNS


MODEL_NAMES = ("majority", "momentum", "logistic")
COMPARISON_MODEL_NAMES = MODEL_NAMES + ("logistic_wti_brent",)
INVENTORY_MODEL_NAMES = MODEL_NAMES + ("logistic_wti_inventory",)
LOGISTIC_PARAMETERS = {
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 2000,
    "random_state": 0,
}


def _feature_vector(row: dict, feature_columns: tuple = WTI_FEATURE_COLUMNS) -> list[float]:
    """Select a fixed feature allowlist; retain explicit nulls for imputation."""
    if feature_columns not in (WTI_FEATURE_COLUMNS, FEATURE_COLUMNS, INVENTORY_MODEL_COLUMNS):
        raise ValueError("Unsupported logistic feature columns")
    feature_set = {WTI_FEATURE_COLUMNS: "WTI", FEATURE_COLUMNS: "WTI/Brent",
                   INVENTORY_MODEL_COLUMNS: "WTI/inventory"}[feature_columns]
    features = row["features"]
    result = []
    for column in feature_columns:
        if column not in features:
            raise ValueError(f"Missing required {feature_set} feature column: {column}")
        value = features[column]
        if value is None:
            result.append(float("nan"))
        elif isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{feature_set} feature {column} must be finite numeric data or null")
        else:
            result.append(float(value))
    return result


def _training_label(row: dict) -> int:
    value = row["label"]
    if isinstance(value, bool) or not isinstance(value, Integral) or value not in (0, 1):
        raise ValueError("Training labels must be integers 0 or 1")
    return int(value)


def _fit_predict_logistic(training_rows: list[dict], forecast_row: dict,
                          labels: list[int], feature_columns: tuple) -> dict:
    """Use identical fitting settings for every predefined feature set."""
    train = np.asarray([_feature_vector(row, feature_columns) for row in training_rows], dtype=float)
    forecast = np.asarray([_feature_vector(forecast_row, feature_columns)], dtype=float)
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
    imputer = model.named_steps["imputer"]
    scaler = model.named_steps["scaler"]
    classifier = model.named_steps["classifier"]
    return {
        "prediction": int(probability >= 0.5),
        "probability_up": probability,
        "diagnostics": {
            "feature_columns": list(feature_columns),
            "imputer_statistics": dict(zip(feature_columns, imputer.statistics_.tolist())),
            "scaler_mean": dict(zip(feature_columns, scaler.mean_.tolist())),
            "scaler_scale": dict(zip(feature_columns, scaler.scale_.tolist())),
            "coef": dict(zip(feature_columns, classifier.coef_[0].tolist())),
            "intercept": float(classifier.intercept_[0]),
            "iterations": int(classifier.n_iter_[0]),
        },
    }


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
    logistic = _fit_predict_logistic(training_rows, forecast_row, labels, WTI_FEATURE_COLUMNS)
    prevalence = sum(labels) / len(labels)
    majority = int(prevalence >= 0.5)
    momentum = forecast_row["features"]["wti_change_5_business_days_usd"]
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
        "logistic": logistic,
    }


def fit_predict_comparison_models(training_rows: list[dict], forecast_row: dict) -> dict:
    """Fit WTI and WTI/Brent models on exactly the same supplied rows and labels.

    The caller must select a common eligible cohort before calling. Both
    regressions receive every supplied row in the same order; this function
    neither selects rows nor changes labels. Brent lookback nulls remain in
    place and are imputed from this shared training set. Only the combined
    regression uses Brent and the aligned Brent/WTI spread.
    """
    result = fit_predict_models(training_rows, forecast_row)
    labels = [_training_label(row) for row in training_rows]
    result["logistic_wti_brent"] = _fit_predict_logistic(
        training_rows, forecast_row, labels, FEATURE_COLUMNS)
    return result


def fit_predict_inventory_models(training_rows: list[dict], forecast_row: dict) -> dict:
    """Add five inventory inputs without removing any supplied WTI examples.

    Both regressions use the same rows and active labels in the supplied order.
    Missing inventory inputs remain null until the training-only imputer; an
    entirely missing training column uses zero. Missingness is recorded in
    diagnostics, never appended as additional model inputs. Original WTI and
    baseline payloads retain their existing shape and numerical values.
    """
    result = fit_predict_models(training_rows, forecast_row)
    labels = [_training_label(row) for row in training_rows]
    inventory = _fit_predict_logistic(training_rows, forecast_row, labels, INVENTORY_MODEL_COLUMNS)
    inventory["diagnostics"].update({
        "training_missing_counts": {
            column: sum(row["features"][column] is None for row in training_rows)
            for column in INVENTORY_MODEL_COLUMNS
        },
        "forecast_missing_columns": [
            column for column in INVENTORY_MODEL_COLUMNS if forecast_row["features"][column] is None
        ],
    })
    result["logistic_wti_inventory"] = inventory
    return result
