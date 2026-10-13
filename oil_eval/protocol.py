"""Fixed development protocol; there is intentionally no holdout-run switch."""

from datetime import date

from oil_audit.features import FEATURE_COLUMNS, WTI_FEATURE_COLUMNS
from .inventory_features import INVENTORY_MODEL_COLUMNS


TRAIN_START = date(2011, 4, 7)
DEVELOPMENT_START = date(2017, 1, 1)
HOLDOUT_START = date(2023, 1, 1)
SCORING_AS_OF = date(2022, 12, 31)
MIN_TRAINING_ROWS = 150
EXPERIMENTS = ("wti", "brent-comparison", "inventory-comparison")


def protocol_config(experiment: str = "wti") -> dict:
    if experiment not in EXPERIMENTS:
        raise ValueError("Unknown development experiment")
    config = {
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
    if experiment == "brent-comparison":
        config.update({
            "protocol_id": "wti-brent-development-v1",
            "feature_columns": list(FEATURE_COLUMNS),
            "compared_feature_sets": {"logistic": list(WTI_FEATURE_COLUMNS),
                                      "logistic_wti_brent": list(FEATURE_COLUMNS)},
            "training_cohort": "same ordered, historically labelled WTI-and-Brent-eligible rows for all models",
            "forecast_cohort": "exact forecast IDs, dates, targets and scoring states from the verified WTI reference",
            "ineligible_comparison_forecast": "fail before fitting; never silently drop reference dates",
            "reference_reproduction": "when WTI training inputs match, require exact reference model outputs",
            "primary_comparison": "paired WTI-plus-Brent minus WTI-only development accuracy; probability quality also reported",
            "tuning": "none: keep the WTI experiment's parameters, thresholds and dates fixed",
        })
    if experiment == "inventory-comparison":
        config.update({
            "protocol_id": "wti-inventory-development-v1",
            "feature_columns": list(INVENTORY_MODEL_COLUMNS),
            "compared_feature_sets": {"logistic": list(WTI_FEATURE_COLUMNS),
                                      "logistic_wti_inventory": list(INVENTORY_MODEL_COLUMNS)},
            "inventory_sources": "audited EIA commercial-crude and Cushing issue archives; qualified archive replay",
            "inventory_weekly_change": "published within-issue difference, million barrels",
            "inventory_four_week_change": "current stock minus latest available exact observation 28 days earlier; same definition regime",
            "inventory_report_age": "calendar days since actual latest eligible release; shared between both series",
            "inventory_cutoff": "release date strictly before each historical forecast; freeze features at that issuance",
            "inventory_missing": "retain nulls and explicit reasons; fit median imputation only on training rows; no missing-indicator inputs",
            "unusable_inventory_issue": "all four changes null; known release metadata still supplies report age; no older-issue or PDF fallback",
            "training_cohort": "exact original WTI-eligible rows and active labels; missing inventory never removes rows",
            "forecast_cohort": "exact forecast IDs, dates, targets and scoring states from the verified WTI reference",
            "reference_reproduction": "require all original WTI training hashes and model payloads to reproduce exactly",
            "primary_comparison": "paired WTI-plus-inventory minus WTI-only development accuracy; constants and probability quality also reported",
            "tuning": "none: keep the WTI experiment's parameters, threshold and dates fixed",
        })
    return config


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
