"""Authenticate and lock a prior development run without opening the holdout."""

from pathlib import Path

from oil_audit.dataset import json_bytes, sha256, workspace_path
from .data import _decode, _gate_date
from .models import MODEL_NAMES
from .protocol import HOLDOUT_START, forecast_role, protocol_config


COHORT_FIELDS = ("row_id", "forecast_date", "target_date", "scoring_status",
                 "label", "label_available_from")


def load_reference(report_path: Path, source: dict, runtime: dict):
    """Check the old report, manifest and every artifact before retaining rows."""
    report_bytes = report_path.read_bytes()
    report = _decode(report_bytes)
    directory = workspace_path(report["evaluation_directory"])
    manifest_bytes = (directory / "manifest.json").read_bytes()
    manifest = _decode(manifest_bytes)
    identity = {key: value for key, value in manifest.items() if key not in ("run_id", "artifacts")}
    if sha256(json_bytes(identity))[:20] != manifest.get("run_id"):
        raise ValueError("Reference run identity does not match its manifest")
    if (manifest["source"] != source or manifest["protocol"] != protocol_config()
            or manifest["runtime_versions"] != runtime or manifest["numeric_threads"] != 1):
        raise ValueError("Reference dataset, protocol or runtime differs from the fixed WTI comparison")
    if set(manifest["artifacts"]) != {"predictions.jsonl", "summary.json"}:
        raise ValueError("Unexpected reference artifacts")
    payloads = {}
    for name, expected in manifest["artifacts"].items():
        payload = (directory / name).read_bytes()
        if sha256(payload) != expected["sha256"] or len(payload) != expected["bytes"]:
            raise ValueError("Reference artifact checksum mismatch: " + name)
        payloads[name] = payload
    summary = _decode(payloads["summary.json"])
    expected_report = {"run_id": manifest["run_id"], "evaluation_directory": str(directory),
                       "dataset_id": source["dataset_id"], "runtime_versions": runtime, **summary}
    if report != expected_report or summary["protocol"] != protocol_config() or summary["holdout_evaluated"] is not False:
        raise ValueError("Reference report does not match the closed-holdout run")
    rows = []
    for line in payloads["predictions.jsonl"].splitlines():
        if _gate_date(line, "forecast_date") >= HOLDOUT_START:
            raise ValueError("Reference contains a holdout forecast")
        row = _decode(line)
        if forecast_role(row) != "development":
            raise ValueError("Reference forecast is outside the development window")
        if set(row["models"]) != set(MODEL_NAMES):
            raise ValueError("Reference does not contain the original WTI model family")
        rows.append(row)
    if (len(rows) != summary["prediction_count"]
            or len({row["row_id"] for row in rows}) != len(rows)
            or [row["forecast_date"] for row in rows] != sorted({row["forecast_date"] for row in rows})
            or sum(row["scoring_status"] == "available" for row in rows) != summary["scored_count"]):
        raise ValueError("Reference forecast counts or ordering differ from its report")
    reference = {"run_id": manifest["run_id"], "evaluation_directory": str(directory),
                 "report_sha256": sha256(report_bytes),
                 "manifest_sha256": sha256(manifest_bytes), "artifacts": manifest["artifacts"]}
    return reference, rows


def validate_reference_schedule(feature_rows, scoring_labels, reference_rows):
    """Lock the entire comparison cohort before the first model fit."""
    labels = {row["row_id"]: row for row in scoring_labels}
    expected = []
    for row in sorted(feature_rows, key=lambda item: item["forecast_date"]):
        if forecast_role(row) != "development" or not row["eligibility"]["wti"]:
            continue
        if not row["eligibility"].get("wti_brent", False):
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
