"""Fixed baselines and logistic regressions fitted on supplied rows.

This module cannot choose training dates. The replay caller must first select
historical rows and labels available at the current forecast cutoff.
"""

import math
from numbers import Integral, Real
import warnings

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
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
INVENTORY_REGULARIZATION_MODEL_NAMES = INVENTORY_MODEL_NAMES + ("logistic_wti_inventory_c01",)
INVENTORY_WINDOW_MODEL_NAMES = INVENTORY_REGULARIZATION_MODEL_NAMES + ("logistic_wti_inventory_c01_rolling",)
INVENTORY_TREE_MODEL_NAMES = INVENTORY_REGULARIZATION_MODEL_NAMES + ("tree_wti_inventory",)
LOGISTIC_PARAMETERS = {
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 2000,
    "random_state": 0,
}
INVENTORY_REGULARIZATION_PARAMETERS = {**LOGISTIC_PARAMETERS, "C": 0.1}
TREE_PARAMETERS = {
    "loss": "log_loss", "learning_rate": 0.05, "n_estimators": 50,
    "subsample": 1.0, "criterion": "friedman_mse", "min_samples_split": 2,
    "min_samples_leaf": 30, "min_weight_fraction_leaf": 0.0, "max_depth": 2,
    "min_impurity_decrease": 0.0, "init": None, "random_state": 0,
    "max_features": None, "verbose": 0, "max_leaf_nodes": None,
    "warm_start": False, "validation_fraction": 0.1, "n_iter_no_change": None,
    "tol": 0.0001, "ccp_alpha": 0.0,
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
                          labels: list[int], feature_columns: tuple, *, parameters=None) -> dict:
    """Fit a predefined feature set with the fixed control or candidate settings."""
    train = np.asarray([_feature_vector(row, feature_columns) for row in training_rows], dtype=float)
    forecast = np.asarray([_feature_vector(forecast_row, feature_columns)], dtype=float)
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(**(LOGISTIC_PARAMETERS if parameters is None else parameters))),
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


def _fit_predict_inventory(training_rows, forecast_row, labels, *, parameters=None):
    """Keep the inventory diagnostics identical for control and candidate fits."""
    inventory = _fit_predict_logistic(training_rows, forecast_row, labels,
                                     INVENTORY_MODEL_COLUMNS, parameters=parameters)
    inventory["diagnostics"].update({
        "training_missing_counts": {
            column: sum(row["features"][column] is None for row in training_rows)
            for column in INVENTORY_MODEL_COLUMNS
        },
        "forecast_missing_columns": [
            column for column in INVENTORY_MODEL_COLUMNS if forecast_row["features"][column] is None
        ],
    })
    return inventory


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
    result["logistic_wti_inventory"] = _fit_predict_inventory(training_rows, forecast_row, labels)
    return result


def fit_predict_inventory_regularization_models(training_rows: list[dict], forecast_row: dict) -> dict:
    """Add the single C=0.1 inventory candidate; reproduce every C=1 control.

    There is no parameter selection, threshold change, or cohort filtering.
    The candidate uses the same eleven inputs and training-only preprocessing
    as the original inventory model, with the same missingness diagnostics.
    """
    result = fit_predict_inventory_models(training_rows, forecast_row)
    labels = [_training_label(row) for row in training_rows]
    result["logistic_wti_inventory_c01"] = _fit_predict_inventory(
        training_rows, forecast_row, labels, parameters=INVENTORY_REGULARIZATION_PARAMETERS)
    return result


def fit_predict_inventory_window_models(expanding_rows: list[dict], rolling_rows: list[dict],
                                       forecast_row: dict) -> dict:
    """Reproduce expanding controls and fit C=0.1 on the supplied rolling rows.

    Replay selects the causal rolling subset and enforces the minimum history.
    The candidate uses the same eleven inputs and parameters as its expanding
    control, with imputation and scaling fitted only on that rolling subset.
    Missing inventory values never remove a supplied row.
    """
    if not rolling_rows:
        raise ValueError("Cannot fit rolling model without training rows")
    labels = [_training_label(row) for row in rolling_rows]
    if len(set(labels)) != 2:
        raise ValueError("Rolling logistic regression requires both training label classes")
    result = fit_predict_inventory_regularization_models(expanding_rows, forecast_row)
    result["logistic_wti_inventory_c01_rolling"] = _fit_predict_inventory(
        rolling_rows, forecast_row, labels, parameters=INVENTORY_REGULARIZATION_PARAMETERS)
    return result


def _fit_predict_inventory_tree(training_rows, forecast_row, labels):
    """Fit the fixed tree candidate after training-only median imputation."""
    columns = INVENTORY_MODEL_COLUMNS
    train = np.asarray([_feature_vector(row, columns) for row in training_rows], dtype=float)
    forecast = np.asarray([_feature_vector(forecast_row, columns)], dtype=float)
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("classifier", GradientBoostingClassifier(**TREE_PARAMETERS)),
    ])
    model.fit(train, labels)
    probability = float(model.predict_proba(forecast)[0, 1])
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise RuntimeError("Gradient boosting produced an invalid probability")
    imputer = model.named_steps["imputer"]
    classifier = model.named_steps["classifier"]
    estimators = classifier.estimators_.ravel()
    depths = [int(estimator.get_depth()) for estimator in estimators]
    return {
        "prediction": int(probability >= 0.5),
        "probability_up": probability,
        "diagnostics": {
            "feature_columns": list(columns),
            "imputer_statistics": dict(zip(columns, imputer.statistics_.tolist())),
            "training_missing_counts": {
                column: sum(row["features"][column] is None for row in training_rows)
                for column in columns
            },
            "forecast_missing_columns": [column for column in columns if forecast_row["features"][column] is None],
            "n_estimators_actual": int(classifier.n_estimators_),
            "max_depth_actual": max(depths),
            "feature_importances": dict(zip(columns, classifier.feature_importances_.tolist())),
            "estimator_depths": depths,
            "estimator_leaf_counts": [int(estimator.get_n_leaves()) for estimator in estimators],
            "estimator_leaf_sample_counts": [
                estimator.tree_.n_node_samples[estimator.tree_.children_left == -1].tolist()
                for estimator in estimators
            ],
        },
    }


def fit_predict_inventory_tree_models(training_rows: list[dict], forecast_row: dict) -> dict:
    """Keep all five expanding controls and add one fixed small tree model.

    The tree sees the same ordered eleven-input examples and active labels as
    the expanding inventory regressions. Missing columns use training medians
    or zero when wholly missing; no scaling, extra indicators, sample weights,
    calibration, validation split or early stopping is used.
    """
    result = fit_predict_inventory_regularization_models(training_rows, forecast_row)
    labels = [_training_label(row) for row in training_rows]
    result["tree_wti_inventory"] = _fit_predict_inventory_tree(training_rows, forecast_row, labels)
    return result
