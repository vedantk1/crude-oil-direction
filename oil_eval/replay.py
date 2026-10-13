"""Expanding historical fits and paired scoring for development only."""

from collections import Counter, defaultdict
from datetime import date

from threadpoolctl import threadpool_limits

from oil_audit.dataset import json_bytes, sha256
from oil_audit.labels import select_label
from oil_audit.training import training_rows
from .metrics import classification_metrics
from .models import MODEL_NAMES, fit_predict_models
from .protocol import (HOLDOUT_START, MIN_TRAINING_ROWS, SCORING_AS_OF,
                       forecast_role, protocol_config)


def replay_development(feature_rows, label_events, scoring_labels, *,
                       predictor=fit_predict_models):
    """Predict first, then attach development outcomes for scoring.

    Training eligibility follows each issuance's label state. The final
    holdout is rejected at this layer even if a caller bypasses the loader.
    """
    boundary = HOLDOUT_START.isoformat()
    if any(row["forecast_date"] >= boundary for row in feature_rows):
        raise ValueError("Holdout features cannot enter development replay")
    if any(event["forecast_date"] >= boundary or event["available_from"] >= boundary
           for event in label_events + scoring_labels):
        raise ValueError("Holdout label information cannot enter development replay")
    rows = sorted(feature_rows, key=lambda row: row["forecast_date"])
    if len({row["row_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate forecast IDs")
    if len({row["forecast_date"] for row in rows}) != len(rows):
        raise ValueError("Duplicate forecast dates")
    histories = defaultdict(list)
    for event in label_events:
        histories[event["row_id"]].append(event)
    for history in histories.values():
        history.sort(key=lambda event: event["available_from"])
    evaluation_by_id = {row["row_id"]: row for row in scoring_labels}
    features_by_id = {row["row_id"]: row for row in rows}
    excluded = Counter()
    predictions = []
    with threadpool_limits(limits=1):
        for row in rows:
            role = forecast_role(row)
            if role != "development":
                excluded[role] += 1
                continue
            if not row["eligibility"]["wti"]:
                excluded["ineligible_wti_inputs"] += 1
                continue
            issuance = date.fromisoformat(row["forecast_date"])
            past = [candidate for candidate in rows
                    if candidate["forecast_date"] < row["forecast_date"]]
            train = training_rows(past, label_events, issuance, SCORING_AS_OF, "wti")
            if len(train) < MIN_TRAINING_ROWS:
                raise ValueError("Insufficient eligible training history at " + row["forecast_date"])
            if {example["label"] for example in train} != {0, 1}:
                raise ValueError("Both training classes required at " + row["forecast_date"])
            selected_events = [select_label(histories[example["row_id"]], issuance) for example in train]
            if any(event["target_date"] >= row["forecast_date"]
                   or event["forecast_date"] >= row["forecast_date"]
                   or event["available_from"] > row["forecast_date"] for event in selected_events):
                raise ValueError("Training-label maturity violation")
            if any(features_by_id[example["row_id"]]["target_date"] >= row["forecast_date"]
                   for example in train):
                raise ValueError("Training target overlaps the forecast origin")
            models = predictor(train, row)
            if set(models) != set(MODEL_NAMES):
                raise ValueError("All declared models must predict the same forecast")
            outcome = evaluation_by_id.get(row["row_id"])
            if outcome is not None and (
                outcome["target_date"] != row["target_date"]
                or outcome["available_from"] <= row["target_date"]
                or outcome["label_status"] != "available"
            ):
                raise ValueError("Invalid development scoring outcome")
            predictions.append({
                "row_id": row["row_id"], "forecast_date": row["forecast_date"],
                "target_date": row["target_date"],
                "training": {
                    "count": len(train), "up_count": sum(example["label"] for example in train),
                    "first_forecast_date": train[0]["forecast_date"],
                    "last_forecast_date": train[-1]["forecast_date"],
                    "latest_target_date": max(event["target_date"] for event in selected_events),
                    "latest_label_available_from": max(event["available_from"] for event in selected_events),
                    "input_sha256": sha256(json_bytes(train)),
                },
                "models": models,
                "scoring_status": "available" if outcome else "unavailable_by_development_cutoff",
                "label": outcome["label"] if outcome else None,
                "label_available_from": outcome["available_from"] if outcome else None,
            })
    if not predictions:
        raise ValueError("No eligible development forecasts")
    return predictions, summarize_predictions(predictions, dict(excluded))


def _metrics(rows):
    scored = [row for row in rows if row["scoring_status"] == "available"]
    labels = [row["label"] for row in scored]
    return {name: classification_metrics(
        labels, [row["models"][name]["prediction"] for row in scored],
        None if name == "momentum" else [row["models"][name]["probability_up"] for row in scored]
    ) for name in MODEL_NAMES}


def summarize_predictions(predictions, excluded):
    scored = [row for row in predictions if row["scoring_status"] == "available"]
    if any(row["forecast_date"] >= HOLDOUT_START.isoformat()
           or row["target_date"] >= HOLDOUT_START.isoformat() for row in predictions):
        raise ValueError("Holdout forecasts or targets cannot be scored")
    years = sorted({row["forecast_date"][:4] for row in predictions})
    paired = {}
    for baseline in ("majority", "momentum"):
        counts = Counter()
        for row in scored:
            logistic_correct = row["models"]["logistic"]["prediction"] == row["label"]
            baseline_correct = row["models"][baseline]["prediction"] == row["label"]
            key = ("both_correct" if logistic_correct and baseline_correct else
                   "both_wrong" if not logistic_correct and not baseline_correct else
                   "logistic_only_correct" if logistic_correct else "baseline_only_correct")
            counts[key] += 1
        paired[baseline] = {
            key: counts[key] for key in
            ("both_correct", "both_wrong", "logistic_only_correct", "baseline_only_correct")}
        paired[baseline]["accuracy_difference_logistic_minus_baseline"] = (
            (counts["logistic_only_correct"] - counts["baseline_only_correct"]) / len(scored)
            if scored else None)
    return {
        "protocol": protocol_config(),
        "forecast_date_range": [predictions[0]["forecast_date"], predictions[-1]["forecast_date"]],
        "prediction_count": len(predictions), "scored_count": len(scored),
        "unscored_count": len(predictions) - len(scored),
        "excluded_pre_holdout_rows": excluded,
        "scored_direction_counts": {"up": sum(row["label"] for row in scored),
                                     "not_up": sum(row["label"] == 0 for row in scored)},
        "training_count_range": [min(row["training"]["count"] for row in predictions),
                                 max(row["training"]["count"] for row in predictions)],
        "momentum_missing_fallback_count": sum(
            row["models"]["momentum"]["used_majority_fallback"] for row in predictions),
        "maximum_logistic_iterations": max(
            row["models"]["logistic"]["diagnostics"]["iterations"] for row in predictions),
        "metrics": _metrics(predictions),
        "yearly": {year: {
            "prediction_count": sum(row["forecast_date"].startswith(year) for row in predictions),
            "metrics": _metrics([row for row in predictions if row["forecast_date"].startswith(year)]),
        } for year in years},
        "paired_accuracy_comparisons": paired,
        "holdout_evaluated": False,
    }
