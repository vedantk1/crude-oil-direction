"""Authenticate and lock a prior development run without opening the holdout."""

from datetime import date
from pathlib import Path
import re

from oil_audit.dataset import json_bytes, sha256, workspace_path
from .data import _decode, _gate_date
from .inventory_features import INVENTORY_FEATURE_COLUMNS
from .models import INVENTORY_MODEL_NAMES, INVENTORY_REGULARIZATION_MODEL_NAMES, MODEL_NAMES
from .protocol import HOLDOUT_START, forecast_role, protocol_config


COHORT_FIELDS = ("row_id", "forecast_date", "target_date", "scoring_status",
                 "label", "label_available_from")
_LABEL_AVAILABILITY = re.compile(rb'"label_available_from"\s*:\s*"(\d{4}-\d{2}-\d{2})"')


def _load_run_reference(report_path, source, runtime, *, experiment="wti",
                        model_names=MODEL_NAMES, identity_fields=None, report_fields=None,
                        native_runtime=None):
    """Share byte authentication while retaining each fixed experiment contract."""
    expected_protocol = protocol_config(experiment)
    inventory = experiment in ("inventory-comparison", "inventory-regularization")
    report_bytes = report_path.read_bytes()
    report = _decode(report_bytes)
    directory = workspace_path(report["evaluation_directory"])
    manifest_bytes = (directory / "manifest.json").read_bytes()
    manifest = _decode(manifest_bytes)
    identity = {key: value for key, value in manifest.items() if key not in ("run_id", "artifacts")}
    if sha256(json_bytes(identity))[:20] != manifest.get("run_id"):
        raise ValueError("Reference run identity does not match its manifest")
    if (manifest["source"] != source or manifest["protocol"] != expected_protocol
            or manifest["runtime_versions"] != runtime or manifest["numeric_threads"] != 1):
        raise ValueError("Reference dataset, protocol or runtime differs from the fixed comparison")
    if any(manifest.get(key) != value for key, value in (identity_fields or {}).items()):
        raise ValueError("Inventory reference, audit or frozen features differ from the saved comparison")
    if native_runtime is not None and manifest.get("native_runtime_sha256") != native_runtime:
        raise ValueError("Reference native runtime differs or is unrecorded; establish compatible controls")
    artifact_names = {"predictions.jsonl", "summary.json"}
    if inventory:
        artifact_names.add("inventory_features.jsonl")
    if set(manifest["artifacts"]) != artifact_names:
        raise ValueError("Unexpected reference artifacts")
    if inventory and manifest["artifacts"]["inventory_features.jsonl"] != manifest["inventory_features"]:
        raise ValueError("Inventory feature artifact differs from its frozen feature identity")
    payloads = {}
    for name, expected in manifest["artifacts"].items():
        payload = (directory / name).read_bytes()
        if sha256(payload) != expected["sha256"] or len(payload) != expected["bytes"]:
            raise ValueError("Reference artifact checksum mismatch: " + name)
        payloads[name] = payload
    summary = _decode(payloads["summary.json"])
    expected_report = {"run_id": manifest["run_id"], "evaluation_directory": str(directory),
                       "dataset_id": source["dataset_id"], "runtime_versions": runtime, **summary,
                       **(report_fields or {})}
    if report != expected_report or summary["protocol"] != expected_protocol or summary["holdout_evaluated"] is not False:
        raise ValueError("Reference report does not match the closed-holdout run")
    rows = []
    for line in payloads["predictions.jsonl"].splitlines():
        if _gate_date(line, "forecast_date") >= HOLDOUT_START:
            raise ValueError("Reference contains a holdout forecast")
        if inventory and any(date.fromisoformat(value.decode("ascii")) >= HOLDOUT_START
                             for value in _LABEL_AVAILABILITY.findall(line)):
            raise ValueError("Inventory reference contains a holdout outcome availability")
        row = _decode(line)
        if forecast_role(row) != "development":
            raise ValueError("Reference forecast is outside the development window")
        if inventory:
            if row.get("scoring_status") == "available":
                if (type(row.get("label")) is not int or row["label"] not in (0, 1)
                        or not isinstance(row.get("label_available_from"), str)
                        or not date.fromisoformat(row["target_date"]) < date.fromisoformat(
                            row["label_available_from"]) < HOLDOUT_START):
                    raise ValueError("Invalid inventory reference scoring outcome")
            elif (row.get("scoring_status") != "unavailable_by_development_cutoff"
                  or row.get("label") is not None or row.get("label_available_from") is not None):
                raise ValueError("Invalid inventory reference unscored state")
        if set(row["models"]) != set(model_names):
            family = "WTI/inventory" if inventory else "WTI"
            raise ValueError("Reference does not contain the original " + family + " model family")
        rows.append(row)
    if (len(rows) != summary["prediction_count"]
            or len({row["row_id"] for row in rows}) != len(rows)
            or [row["forecast_date"] for row in rows] != sorted({row["forecast_date"] for row in rows})
            or sum(row["scoring_status"] == "available" for row in rows) != summary["scored_count"]):
        raise ValueError("Reference forecast counts or ordering differ from its report")
    if inventory:
        frozen_rows = []
        for line in payloads["inventory_features.jsonl"].splitlines():
            if _gate_date(line, "forecast_date") >= HOLDOUT_START:
                raise ValueError("Inventory reference contains a holdout feature")
            row = _decode(line)
            if forecast_role(row) not in ("initial_training", "development"):
                raise ValueError("Inventory reference feature is outside the admitted history")
            if set(row["features"]) != set(INVENTORY_FEATURE_COLUMNS):
                raise ValueError("Inventory reference feature columns differ from the fixed model")
            frozen_rows.append(row)
        if (len({row["row_id"] for row in frozen_rows}) != len(frozen_rows)
                or [row["forecast_date"] for row in frozen_rows] != sorted(
                    {row["forecast_date"] for row in frozen_rows})):
            raise ValueError("Inventory reference feature counts or ordering differ")
        frozen_by_id = {row["row_id"]: row for row in frozen_rows}
        for row in rows:
            frozen = frozen_by_id.get(row["row_id"])
            if (frozen is None or any(frozen[key] != row[key] for key in
                                     ("forecast_date", "target_date"))
                    or {"features": frozen["features"], "provenance": frozen["provenance"]}
                    != row["inventory"]):
                raise ValueError("Inventory reference forecasts differ from their frozen features")
    reference = {"run_id": manifest["run_id"], "evaluation_directory": str(directory),
                 "report_sha256": sha256(report_bytes),
                 "manifest_sha256": sha256(manifest_bytes), "artifacts": manifest["artifacts"]}
    return reference, rows


def load_reference(report_path: Path, source: dict, runtime: dict, *, native_runtime=None):
    """Authenticate the original WTI report, manifest and every artifact."""
    return _load_run_reference(report_path, source, runtime, native_runtime=native_runtime)


def load_inventory_comparison_reference(report_path: Path, source: dict, runtime: dict,
                                       wti_reference: dict, inventory_reference: dict,
                                       inventory_features: dict, *, native_runtime=None):
    """Lock C=1 controls to the same audit and all reconstructed frozen rows."""
    return _load_run_reference(
        report_path, source, runtime, experiment="inventory-comparison",
        model_names=INVENTORY_MODEL_NAMES,
        identity_fields={"reference": wti_reference, "inventory_reference": inventory_reference,
                         "inventory_features": inventory_features},
        native_runtime=native_runtime,
        report_fields={"reference_run_id": wti_reference["run_id"],
                       "inventory_audit_id": inventory_reference["audit_id"]})


def load_inventory_regularization_reference(report_path: Path, source: dict, runtime: dict,
                                           wti_reference: dict, inventory_reference: dict,
                                           inventory_features: dict, inventory_comparison_reference: dict,
                                           *, native_runtime=None):
    """Authenticate expanding C=0.1 controls and their complete reference chain."""
    return _load_run_reference(
        report_path, source, runtime, experiment="inventory-regularization",
        model_names=INVENTORY_REGULARIZATION_MODEL_NAMES,
        identity_fields={"reference": wti_reference, "inventory_reference": inventory_reference,
                         "inventory_features": inventory_features,
                         "inventory_comparison_reference": inventory_comparison_reference},
        native_runtime=native_runtime,
        report_fields={"reference_run_id": wti_reference["run_id"],
                       "inventory_audit_id": inventory_reference["audit_id"],
                       "inventory_reference_run_id": inventory_comparison_reference["run_id"]})


def validate_reference_schedule(feature_rows, scoring_labels, reference_rows, *, require_brent=True):
    """Lock the entire comparison cohort before the first model fit."""
    labels = {row["row_id"]: row for row in scoring_labels}
    expected = []
    for row in sorted(feature_rows, key=lambda item: item["forecast_date"]):
        if forecast_role(row) != "development" or not row["eligibility"]["wti"]:
            continue
        if require_brent and not row["eligibility"].get("wti_brent", False):
            raise ValueError("Reference development date has no eligible Brent inputs: " + row["forecast_date"])
        outcome = labels.get(row["row_id"])
        expected.append({
            "row_id": row["row_id"], "forecast_date": row["forecast_date"], "target_date": row["target_date"],
            "scoring_status": "available" if outcome else "unavailable_by_development_cutoff",
            "label": outcome["label"] if outcome else None,
            "label_available_from": outcome["available_from"] if outcome else None,
        })
    if expected != [{key: row[key] for key in COHORT_FIELDS} for row in reference_rows]:
        raise ValueError("Comparison forecast dates, targets or scoring labels differ from the WTI reference")


def verify_reference_reproduction(predictions, reference_rows) -> dict:
    """Require unchanged WTI predictions whenever the actual fit inputs match."""
    if [{key: row[key] for key in COHORT_FIELDS} for row in predictions] != [
            {key: row[key] for key in COHORT_FIELDS} for row in reference_rows]:
        raise ValueError("Saved comparison cohort differs from the WTI reference")
    identical_inputs = identical_outputs = 0
    for row, reference in zip(predictions, reference_rows):
        inputs_match = row["training"]["input_sha256"] == reference["training"]["input_sha256"]
        outputs_match = all(row["models"][name] == reference["models"][name] for name in MODEL_NAMES)
        identical_inputs += inputs_match
        identical_outputs += outputs_match
        if inputs_match and not outputs_match:
            raise ValueError("WTI model outputs changed despite identical historical training inputs")
    return {"forecast_cohort_identical": True,
            "identical_wti_training_input_count": identical_inputs,
            "identical_reference_model_output_count": identical_outputs,
            "all_wti_training_inputs_identical": identical_inputs == len(predictions),
            "all_reference_model_outputs_identical": identical_outputs == len(predictions)}


def validate_inventory_reference_schedule(feature_rows, scoring_labels, inventory_reference_rows):
    """Check dates, outcomes and raw forecast inputs before any model fits."""
    validate_reference_schedule(feature_rows, scoring_labels, inventory_reference_rows,
                                require_brent=False)
    by_id = {row["row_id"]: row for row in feature_rows}
    for reference in inventory_reference_rows:
        row = by_id[reference["row_id"]]
        inventory = {"features": {column: row["features"][column]
                                  for column in INVENTORY_FEATURE_COLUMNS},
                     "provenance": row.get("inventory_provenance")}
        if inventory != reference["inventory"]:
            raise ValueError("Inventory forecast features or provenance differ from the C=1 reference")


def verify_inventory_reference_reproduction(predictions, inventory_reference_rows) -> dict:
    """Require every C=1 training detail, frozen input and control output exactly."""
    if [{key: row[key] for key in COHORT_FIELDS} for row in predictions] != [
            {key: row[key] for key in COHORT_FIELDS} for row in inventory_reference_rows]:
        raise ValueError("Saved comparison cohort differs from the inventory reference")
    for row, reference in zip(predictions, inventory_reference_rows):
        if row["training"] != reference["training"]:
            raise ValueError("Inventory training inputs or cohort details differ from the C=1 reference")
        if row["inventory"] != reference["inventory"]:
            raise ValueError("Inventory forecast inputs or provenance differ from the C=1 reference")
        if any(row["models"][name] != reference["models"][name] for name in INVENTORY_MODEL_NAMES):
            raise ValueError("Original inventory comparison model outputs differ from the C=1 reference")
    return {"forecast_cohort_identical": True,
            "identical_inventory_training_input_count": len(predictions),
            "identical_inventory_forecast_input_count": len(predictions),
            "identical_control_model_output_count": len(predictions),
            "all_inventory_training_inputs_identical": True,
            "all_inventory_forecast_inputs_identical": True,
            "all_control_model_outputs_identical": True}


def verify_regularization_reference_reproduction(predictions, regularization_reference_rows) -> dict:
    """Require all expanding control outputs, including the selected C=0.1 fit."""
    result = verify_inventory_reference_reproduction(predictions, regularization_reference_rows)
    for row, reference in zip(predictions, regularization_reference_rows):
        if row["models"]["logistic_wti_inventory_c01"] != reference["models"]["logistic_wti_inventory_c01"]:
            raise ValueError("Expanding C=0.1 model outputs differ from the regularization reference")
    return {**result, "identical_expanding_c01_model_output_count": len(predictions),
            "all_expanding_c01_model_outputs_identical": True}
