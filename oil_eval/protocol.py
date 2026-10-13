"""Fixed development protocol; there is intentionally no holdout-run switch."""

from datetime import date

from oil_audit.features import WTI_FEATURE_COLUMNS


TRAIN_START = date(2011, 4, 7)
DEVELOPMENT_START = date(2017, 1, 1)
HOLDOUT_START = date(2023, 1, 1)
SCORING_AS_OF = date(2022, 12, 31)
MIN_TRAINING_ROWS = 150


def protocol_config() -> dict:
    return {
        "protocol_id": "wti-development-v1",
        "initial_training_start": TRAIN_START.isoformat(),
        "development_start": DEVELOPMENT_START.isoformat(),
        "development_scoring_as_of": SCORING_AS_OF.isoformat(),
        "holdout_start": HOLDOUT_START.isoformat(),
        "holdout_status": "closed: no fitting, prediction, scoring or tuning",
        "feature_columns": list(WTI_FEATURE_COLUMNS),
        "fit_schedule": "expand and refit at every development forecast",
        "training_labels": "active as of issuance under the preceding-calendar-day convention",
        "scoring_labels": "first jointly available pair, available before holdout start",
        "boundary_rule": "forecast and target dates must both precede holdout start",
        "minimum_training_rows": MIN_TRAINING_ROWS,
        "required_training_classes": [0, 1],
        "majority": {"probability": "training up-class frequency", "tie_prediction": 1},
        "momentum": {"feature": "wti_change_5_business_days_usd", "rule": "positive means up",
                     "zero_prediction": 0, "missing_fallback": "majority prediction",
                     "probability": None},
        "logistic": {"imputer": "training median; all-missing training column uses zero",
                     "scaler": "training StandardScaler", "penalty": "l2", "C": 1.0,
                     "solver": "lbfgs", "max_iter": 2000, "random_state": 0,
                     "class_weight": None, "threshold": 0.5, "tie_prediction": 1},
        "tuning": "none: fixed parameters and thresholds before first results",
        "primary_comparison": "paired development accuracy; probability quality also reported",
        "uncertainty": "descriptive yearly results; no independent-sample significance claim",
    }


def forecast_role(row: dict) -> str:
    """Partition by dates without accessing feature values or outcome labels."""
    forecast = date.fromisoformat(row["forecast_date"])
    target = date.fromisoformat(row["target_date"])
    if forecast >= HOLDOUT_START:
        return "holdout"
    if forecast < TRAIN_START:
        return "before_training_start"
    if target <= forecast:
        raise ValueError("Target must follow forecast date")
    if target >= HOLDOUT_START:
        return "crosses_holdout_boundary"
    if forecast < DEVELOPMENT_START:
        return "initial_training"
    return "development"
