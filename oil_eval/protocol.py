"""Fixed development protocol; there is intentionally no holdout-run switch."""

from datetime import date

from oil_audit.features import FEATURE_COLUMNS, WTI_FEATURE_COLUMNS
from .inventory_features import INVENTORY_MODEL_COLUMNS
from .window import ROLLING_WINDOW_YEARS


TRAIN_START = date(2011, 4, 7)
DEVELOPMENT_START = date(2017, 1, 1)
HOLDOUT_START = date(2023, 1, 1)
SCORING_AS_OF = date(2022, 12, 31)
MIN_TRAINING_ROWS = 150
EXPERIMENTS = ("wti", "brent-comparison", "inventory-comparison", "inventory-regularization",
               "inventory-window-comparison", "inventory-tree-comparison", "inventory-availability-delay")


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
    if experiment in ("inventory-comparison", "inventory-regularization", "inventory-window-comparison",
                      "inventory-tree-comparison", "inventory-availability-delay"):
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
    if experiment in ("inventory-regularization", "inventory-window-comparison", "inventory-tree-comparison",
                      "inventory-availability-delay"):
        config.update({
            "protocol_id": "wti-inventory-regularization-development-v1",
            "compared_feature_sets": {
                **config["compared_feature_sets"],
                "logistic_wti_inventory_c01": list(INVENTORY_MODEL_COLUMNS),
            },
            "logistic_variants": {
                "logistic": dict(config["logistic"]),
                "logistic_wti_inventory": dict(config["logistic"]),
                "logistic_wti_inventory_c01": {**config["logistic"], "C": 0.1},
            },
            "candidate_selection": "retrospective single predefined C=0.1 candidate after observing earlier development results",
            "reference_reproduction": "require exact selected authenticated WTI and C=1 inventory cohorts, training, inventory provenance and model payloads; native runtime fingerprints must match before fitting",
            "primary_comparison": "paired development Brier score and log loss for C=0.1 minus C=1 inventory; accuracy and yearly stability also reported",
            "success_criteria": "report aggregate Brier/log-loss differences against C=1 and both simple probability benchmarks; report each yearly difference descriptively, without a formal success threshold or significance claim",
            "tuning": "one fixed regularization candidate; no grid, feature, threshold, date or holdout changes",
        })
    if experiment == "inventory-window-comparison":
        config.update({
            "protocol_id": "wti-inventory-window-development-v1",
            "compared_feature_sets": {
                **config["compared_feature_sets"],
                "logistic_wti_inventory_c01_rolling": list(INVENTORY_MODEL_COLUMNS),
            },
            "logistic_variants": {
                **config["logistic_variants"],
                "logistic_wti_inventory_c01_rolling": dict(config["logistic_variants"]["logistic_wti_inventory_c01"]),
            },
            "fit_schedule": "refit expanding controls and one fixed rolling-window candidate at every development forecast",
            "rolling_window_years": ROLLING_WINDOW_YEARS,
            "rolling_window_boundary": "training forecast date at or after issuance minus four calendar years and strictly before issuance; clamp February 29 to February 28 if needed",
            "rolling_window_selection": "filter the ordered expanding cohort only after causal eligibility; retain active labels, frozen features and missing values",
            "training_cohort": "controls retain the exact expanding cohort; only the rolling C=0.1 candidate uses its fixed four-calendar-year subset, with minimum 150 and both classes unchanged",
            "candidate_selection": "retrospective single predefined four-year window after earlier development results; duration chosen by capacity-only audit without model predictions",
            "capacity_audit": "three years supplies 147-155 eligible rows and misses minimum 150 on 119 of 310 dates; four years supplies 197-207 rows and both classes on all 310 dates",
            "reference_reproduction": "require exact authenticated WTI, C=1 inventory and expanding C=0.1 cohorts, training, inventory provenance and model payloads; native runtime fingerprints must match before fitting",
            "primary_comparison": "paired development Brier score and log loss for rolling minus expanding C=0.1 inventory; accuracy and yearly stability also reported",
            "success_criteria": "report aggregate Brier/log-loss differences against expanding C=0.1 and both simple probability benchmarks; report each yearly difference descriptively, without a formal success threshold or significance claim",
            "tuning": "one fixed four-calendar-year window; no duration grid, parameter, feature, threshold, date or holdout changes",
        })
    if experiment == "inventory-tree-comparison":
        config.update({
            "protocol_id": "wti-inventory-tree-development-v1",
            "compared_feature_sets": {
                **config["compared_feature_sets"],
                "tree_wti_inventory": list(INVENTORY_MODEL_COLUMNS),
            },
            "tree": {
                "classifier": "GradientBoostingClassifier",
                "imputer": "training median; all-missing training column uses zero",
                "scaler": None, "missing_indicators": False, "class_weight": None,
                "sample_weight": None, "calibration": None, "threshold": 0.5, "tie_prediction": 1,
                "parameters": {
                    "loss": "log_loss", "learning_rate": 0.05, "n_estimators": 50,
                    "subsample": 1.0, "criterion": "friedman_mse", "min_samples_split": 2,
                    "min_samples_leaf": 30, "min_weight_fraction_leaf": 0.0, "max_depth": 2,
                    "min_impurity_decrease": 0.0, "init": None, "random_state": 0,
                    "max_features": None, "verbose": 0, "max_leaf_nodes": None,
                    "warm_start": False, "validation_fraction": 0.1, "n_iter_no_change": None,
                    "tol": 0.0001, "ccp_alpha": 0.0,
                },
                "early_stopping": "disabled; no validation subset is created",
                "feature_importance": "normalized impurity decrease recorded as a fitted-model diagnostic, without causal interpretation",
            },
            "candidate_selection": "retrospective single predefined small gradient-boosting candidate after earlier development results",
            "reference_reproduction": "require exact authenticated WTI, C=1 inventory and expanding C=0.1 cohorts, training, inventory provenance and model payloads; native runtime fingerprints must match before fitting",
            "primary_comparison": "paired development Brier score and log loss for tree minus expanding C=0.1 inventory; accuracy and yearly stability also reported",
            "success_criteria": "report aggregate Brier/log-loss differences against expanding C=0.1 and both simple probability benchmarks; report each yearly difference descriptively, without a formal success threshold or significance claim",
            "tuning": "one fixed small tree candidate; no grid, calibration, parameter, feature, threshold, date or holdout changes",
        })
    if experiment == "inventory-availability-delay":
        config.update({
            "protocol_id": "wti-inventory-availability-delay-development-v1",
            "compared_feature_sets": {
                **config["compared_feature_sets"],
                "logistic_wti_inventory_c01_delayed": list(INVENTORY_MODEL_COLUMNS),
            },
            "logistic_variants": {
                **config["logistic_variants"],
                "logistic_wti_inventory_c01_delayed": dict(config["logistic_variants"]["logistic_wti_inventory_c01"]),
            },
            "additional_information_delay_business_days": 1,
            "delayed_information_clock": "P is the previous US business day before original forecast D; WTI archive cutoff is P minus one calendar day; inventory original earliest eligible issuance must be no later than P",
            "delayed_training_labels": "select every active label state, including revisions and withdrawals, as of P; never use a later state or frozen scoring labels for fitting",
            "delayed_features": "reconstruct and freeze all eleven inputs at each original issuance using the delayed source clock; never shift forecast dates or reuse a previous weekly feature row",
            "delayed_feature_ages": "measure WTI observation age and inventory release age at actual forecast D; preserve existing WTI lookbacks anchored at the latest usable delayed observation",
            "training_cohort": "controls retain exact original expanding rows; delayed C=0.1 uses its own frozen delayed-feature eligibility and labels active at P, with all exclusions and label changes recorded; minimum 150 and both classes unchanged",
            "forecast_cohort": "exact original forecast IDs, dates, targets and frozen scoring states; fail rather than drop any development forecast",
            "source_admission": "authenticate original source bytes; gate ALFRED observation and vintage dates before value decoding; mask future interval ends; require baseline WTI reconstruction to match every original row before delayed fitting",
            "candidate_selection": "retrospective single predefined one-additional-business-day availability sensitivity check after earlier development results",
            "reference_reproduction": "require exact authenticated WTI, C=1 inventory and expanding C=0.1 controls on their original information clock; native runtime fingerprints must match before fitting",
            "primary_comparison": "paired development Brier score and log loss for delayed minus original expanding C=0.1 inventory; accuracy and yearly stability also reported",
            "success_criteria": "report aggregate and yearly differences descriptively against the original information convention and simple probability benchmarks, without a formal success threshold or significance claim",
            "tuning": "one fixed availability delay; no lag grid, feature, parameter, threshold, forecast, target or holdout changes",
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
