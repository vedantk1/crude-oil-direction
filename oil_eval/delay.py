"""Dedicated replay of one delayed information clock against exact controls."""

from collections import defaultdict
from copy import deepcopy
from datetime import date
import math

from threadpoolctl import threadpool_limits

from oil_audit.calendar import next_business_day
from oil_audit.dataset import json_bytes, sha256
from oil_audit.features import WTI_FEATURE_COLUMNS
from oil_audit.labels import select_label
from .benchmarks import summarize_benchmarks
from .delay_data import _validate_rows, delay_information_dates
from .inventory_features import INVENTORY_FEATURE_COLUMNS, INVENTORY_MODEL_COLUMNS
from .metrics import classification_metrics
from .models import (INVENTORY_REGULARIZATION_MODEL_NAMES, INVENTORY_REGULARIZATION_PARAMETERS,
                     _fit_predict_inventory, _training_label)
from .protocol import HOLDOUT_START, MIN_TRAINING_ROWS, forecast_role, protocol_config
from .reference import (validate_inventory_reference_schedule, validate_reference_schedule,
                        verify_regularization_reference_reproduction)
from .replay import replay_development


DELAYED_MODEL_NAME = "logistic_wti_inventory_c01_delayed"
DELAY_MODEL_NAMES = INVENTORY_REGULARIZATION_MODEL_NAMES + (DELAYED_MODEL_NAME,)
_METRICS = ("accuracy", "balanced_accuracy", "roc_auc", "brier_score", "log_loss")


def fit_delayed_candidate(training_rows, forecast_row):
    """Reuse the fixed eleven-input C=0.1 fit with no new model settings."""
    labels = [_training_label(row) for row in training_rows]
    if not training_rows or set(labels) != {0, 1}:
        raise ValueError("Delayed logistic regression requires both training label classes")
    return _fit_predict_inventory(training_rows, forecast_row, labels,
                                   parameters=INVENTORY_REGULARIZATION_PARAMETERS)


def _training_view(rows, histories, forecast_date, knowledge_date):
    selected, events, excluded = [], [], {"wti_ineligible": [], "label_unavailable": []}
    for row in rows:
        if row["forecast_date"] >= forecast_date:
            continue
        if not row["eligibility"]["wti"]:
            excluded["wti_ineligible"].append(row["row_id"])
            continue
        event = select_label(histories[row["row_id"]], knowledge_date)
        if event is None or event["label_status"] != "available":
            excluded["label_unavailable"].append(row["row_id"])
            continue
        if (event["forecast_date"] != row["forecast_date"] or event["target_date"] != row["target_date"]
                or event["target_date"] >= knowledge_date.isoformat()
                or event["available_from"] > knowledge_date.isoformat()):
            raise ValueError("Delayed training label maturity or identity violation")
        selected.append({"row_id": row["row_id"], "forecast_date": row["forecast_date"],
                         "features": {name: row["features"][name] for name in INVENTORY_MODEL_COLUMNS},
                         "label": event["label"]})
        events.append(event)
    return selected, events, excluded


def _validate_delayed_rows(original_rows, delayed_rows):
    _validate_rows(original_rows)
    _validate_rows(delayed_rows)
    original = {row["row_id"]: row for row in original_rows}
    if set(original) != {row["row_id"] for row in delayed_rows}:
        raise ValueError("Delayed features must retain every original raw forecast row")
    for row in delayed_rows:
        control = original[row["row_id"]]
        if any(row[key] != control[key] for key in ("forecast_date", "target_date")):
            raise ValueError("Delayed features cannot change forecast origins or targets")
        knowledge, cutoff = delay_information_dates(date.fromisoformat(row["forecast_date"]))
        if any(not isinstance(row.get(key), dict) for key in (
                "provenance", "inventory_provenance", "availability_delay_provenance")):
            raise ValueError("Delayed features require source and availability provenance dictionaries")
        provenance = row["availability_delay_provenance"]
        if (row.get("information_cutoff_date") != cutoff.isoformat()
                or provenance.get("label_knowledge_date") != knowledge.isoformat()
                or provenance.get("information_cutoff_date") != cutoff.isoformat()
                or provenance.get("feature_age_reference_date") != row["forecast_date"]):
            raise ValueError("Delayed feature provenance differs from the fixed information clock")
        if set(row["features"]) != set(INVENTORY_MODEL_COLUMNS):
            raise ValueError("Delayed features must contain exactly the eleven declared inputs")
        if type(row["eligibility"].get("wti")) is not bool or row["eligibility"]["wti"] != (
                row["features"]["wti_latest_price"] is not None):
            raise ValueError("Delayed WTI eligibility differs from its available price")
        for value in row["features"].values():
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                raise ValueError("Delayed features must be finite numeric values or explicit nulls")


def _plan_delayed_fits(original_rows, delayed_rows, label_events, reference_rows):
    histories = defaultdict(list)
    for event in label_events:
        histories[event["row_id"]].append(event)
    for history in histories.values():
        history.sort(key=lambda event: event["available_from"])
    original_rows = sorted(original_rows, key=lambda row: row["forecast_date"])
    delayed_rows = sorted(delayed_rows, key=lambda row: row["forecast_date"])
    delayed = {row["row_id"]: row for row in delayed_rows}
    plans = []
    for reference in reference_rows:
        row = delayed[reference["row_id"]]
        if forecast_role(row) != "development" or not row["eligibility"]["wti"]:
            raise ValueError("Delayed view cannot predict every original development forecast")
        issuance = date.fromisoformat(row["forecast_date"])
        knowledge, cutoff = delay_information_dates(issuance)
        train, selected_events, excluded = _training_view(delayed_rows, histories, row["forecast_date"], knowledge)
        baseline, _, _ = _training_view(original_rows, histories, row["forecast_date"], issuance)
        if sha256(json_bytes(baseline)) != reference["training"]["inventory_input_sha256"]:
            raise ValueError("Original training inputs differ from the authenticated expanding control")
        if len(train) < MIN_TRAINING_ROWS:
            raise ValueError("Insufficient delayed training history at " + row["forecast_date"])
        if {_training_label(example) for example in train} != {0, 1}:
            raise ValueError("Both delayed training classes required at " + row["forecast_date"])
        baseline_by_id = {example["row_id"]: example for example in baseline}
        train_by_id = {example["row_id"]: example for example in train}
        projection = [{**example, "features": {name: example["features"][name] for name in WTI_FEATURE_COLUMNS}}
                      for example in train]
        selected_states = [{key: event[key] for key in (
            "row_id", "forecast_date", "target_date", "available_from", "label_status", "label",
            "origin_vintage_start", "target_vintage_start")} for event in selected_events]
        metadata = {
            "count": len(train), "up_count": sum(example["label"] for example in train),
            "first_forecast_date": train[0]["forecast_date"], "last_forecast_date": train[-1]["forecast_date"],
            "label_knowledge_date": knowledge.isoformat(), "information_cutoff_date": cutoff.isoformat(),
            "latest_target_date": max(event["target_date"] for event in selected_events),
            "latest_original_label_available_from": max(event["available_from"] for event in selected_events),
            "latest_delayed_label_available_from": max(next_business_day(date.fromisoformat(
                event["available_from"])).isoformat() for event in selected_events),
            "input_sha256": sha256(json_bytes(projection)), "inventory_input_sha256": sha256(json_bytes(train)),
            "selected_label_states_sha256": sha256(json_bytes(selected_states)),
            "selected_row_ids": [example["row_id"] for example in train],
            "selected_label_states": selected_states,
            "original_training_count": len(baseline),
            "removed_original_training_ids": [example["row_id"] for example in baseline if example["row_id"] not in train_by_id],
            "added_vs_original_training_ids": [example["row_id"] for example in train if example["row_id"] not in baseline_by_id],
            "changed_active_labels": [{"row_id": example["row_id"],
                                       "original_label": baseline_by_id[example["row_id"]]["label"],
                                       "delayed_label": example["label"]}
                                      for example in train if example["row_id"] in baseline_by_id
                                      and example["label"] != baseline_by_id[example["row_id"]]["label"]],
            "excluded_past_rows": excluded,
            "inventory_missing_counts": {name: sum(example["features"][name] is None for example in train)
                                         for name in INVENTORY_FEATURE_COLUMNS},
        }
        plans.append((row, train, metadata))
    return plans


def replay_delayed_development(original_rows, delayed_rows, label_events, scoring_labels, *,
                               reference_rows, inventory_reference_rows, regularization_reference_rows,
                               control_predictor=None, candidate_predictor=None):
    """Preflight every delayed fit, reproduce controls, then score fixed dates."""
    boundary = HOLDOUT_START.isoformat()
    if any(row["forecast_date"] >= boundary for row in original_rows + delayed_rows):
        raise ValueError("Holdout features cannot enter availability-delay replay")
    if any(event["forecast_date"] >= boundary or event["available_from"] >= boundary
           for event in label_events + scoring_labels):
        raise ValueError("Holdout label information cannot enter availability-delay replay")
    if any(rows is None for rows in (reference_rows, inventory_reference_rows, regularization_reference_rows)):
        raise ValueError("Availability delay requires all three authenticated control references")
    _validate_delayed_rows(original_rows, delayed_rows)
    validate_reference_schedule(original_rows, scoring_labels, reference_rows, require_brent=False)
    validate_inventory_reference_schedule(original_rows, scoring_labels, inventory_reference_rows)
    validate_inventory_reference_schedule(original_rows, scoring_labels, regularization_reference_rows)
    plans = _plan_delayed_fits(original_rows, delayed_rows, label_events, regularization_reference_rows)
    controls, summary = replay_development(
        original_rows, label_events, scoring_labels, experiment="inventory-regularization",
        reference_rows=reference_rows, inventory_reference_rows=inventory_reference_rows,
        predictor=control_predictor)
    verify_regularization_reference_reproduction(controls, regularization_reference_rows)
    predictor = candidate_predictor or fit_delayed_candidate
    predictions = []
    with threadpool_limits(limits=1):
        for control, (forecast, train, metadata) in zip(controls, plans):
            if control["row_id"] != forecast["row_id"]:
                raise ValueError("Delayed fit schedule differs from the original controls")
            output = predictor(train, forecast)
            row = deepcopy(control)
            row["models"][DELAYED_MODEL_NAME] = output
            row["delayed_training"] = metadata
            row["delayed_information"] = {
                "features": deepcopy(forecast["features"]),
                "provenance": deepcopy(forecast.get("provenance")),
                "inventory_provenance": deepcopy(forecast["inventory_provenance"]),
                "availability_delay_provenance": deepcopy(forecast["availability_delay_provenance"]),
            }
            predictions.append(row)
    summary["protocol"] = protocol_config("inventory-availability-delay")
    summary["regularization_reference_reproduction"] = verify_regularization_reference_reproduction(
        predictions, regularization_reference_rows)
    scored = [row for row in predictions if row["scoring_status"] == "available"]

    def metrics(rows):
        return classification_metrics([row["label"] for row in rows],
                                      [row["models"][DELAYED_MODEL_NAME]["prediction"] for row in rows],
                                      [row["models"][DELAYED_MODEL_NAME]["probability_up"] for row in rows])

    summary["metrics"][DELAYED_MODEL_NAME] = metrics(scored)
    for year, group in summary["yearly"].items():
        group["metrics"][DELAYED_MODEL_NAME] = metrics([row for row in scored if row["forecast_date"].startswith(year)])
    comparators = tuple(name for name in DELAY_MODEL_NAMES if name.startswith("logistic"))
    summary["constant_benchmarks"] = summarize_benchmarks(predictions, comparators)
    summary["maximum_iterations_by_logistic_model"][DELAYED_MODEL_NAME] = max(
        row["models"][DELAYED_MODEL_NAME]["diagnostics"]["iterations"] for row in predictions)
    summary["delayed_training_count_range"] = [min(row["delayed_training"]["count"] for row in predictions),
                                               max(row["delayed_training"]["count"] for row in predictions)]
    summary["delayed_training_changes"] = {key: sum(len(row["delayed_training"][key]) for row in predictions)
                                           for key in ("removed_original_training_ids", "added_vs_original_training_ids",
                                                       "changed_active_labels")}
    summary["paired_availability_delay"] = _paired_delay(scored, summary["metrics"], summary["yearly"])
    return predictions, summary


def _paired_delay(scored, metrics, yearly):
    counts = {key: 0 for key in ("both_correct", "both_wrong", "delayed_only_correct", "original_only_correct")}
    original = "logistic_wti_inventory_c01"
    for row in scored:
        delayed_correct = row["models"][DELAYED_MODEL_NAME]["prediction"] == row["label"]
        original_correct = row["models"][original]["prediction"] == row["label"]
        key = ("both_correct" if delayed_correct and original_correct else
               "both_wrong" if not delayed_correct and not original_correct else
               "delayed_only_correct" if delayed_correct else "original_only_correct")
        counts[key] += 1

    def differences(values):
        return {name: (values[DELAYED_MODEL_NAME][name] - values[original][name]
                       if values[DELAYED_MODEL_NAME][name] is not None and values[original][name] is not None else None)
                for name in _METRICS}

    return {**counts, "metric_differences_delayed_minus_original": differences(metrics),
            "yearly_metric_differences_delayed_minus_original": {
                year: differences(group["metrics"]) for year, group in yearly.items()}}
