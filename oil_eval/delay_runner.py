"""Run the predefined development-only availability sensitivity check."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform

from oil_audit.dataset import json_bytes, jsonl_bytes, sha256, workspace_path
from .data import load_development_data
from .delay import replay_delayed_development
from .delay_data import (build_delayed_feature_rows, load_delay_wti_source,
                         reconstruct_baseline_wti_rows)
from .inventory_features import (INVENTORY_FEATURE_COLUMNS, attach_inventory_features,
                                 load_inventory_reference)
from .protocol import forecast_role, protocol_config
from .reference import (load_inventory_comparison_reference,
                        load_inventory_regularization_reference, load_reference)
from .runner import _native_runtime_sha256


def run(dataset_dir, vintage_dir, output_root, report, *, reference_report,
        inventory_audit_report, inventory_reference_report, regularization_reference_report):
    paths = [dataset_dir, vintage_dir, output_root, report, reference_report,
             inventory_audit_report, inventory_reference_report, regularization_reference_report]
    for path in paths:
        workspace_path(str(path))
        if not path.resolve().is_relative_to(Path.cwd().resolve()):
            raise ValueError("Availability-delay paths must stay in the workspace")
    if report.exists() or report.resolve().is_relative_to(output_root.resolve()):
        raise ValueError("Choose a new report outside immutable evaluation artifacts")
    if any(report.resolve().is_relative_to(root.resolve()) for root in (
            Path("data/datasets"), Path("data/evaluations"), Path("data/audits/inventories"))):
        raise ValueError("Delay report cannot be placed inside historical artifact roots")
    for parent in (report.parent, *report.parents, output_root, *output_root.parents):
        if (parent / "manifest.json").exists():
            raise ValueError("Availability-delay outputs cannot overwrite an immutable artifact")
    source, features, events, labels = load_development_data(dataset_dir)
    features = [row for row in features if forecast_role(row) in ("initial_training", "development")]
    source_metadata, wti = load_delay_wti_source(vintage_dir, source["source_manifest_sha256"])
    _, reconstruction = reconstruct_baseline_wti_rows(features, wti)
    runtime = {name: version(name) for name in
               ("scikit-learn", "numpy", "scipy", "joblib", "threadpoolctl")}
    runtime["python"] = platform.python_version()
    native = _native_runtime_sha256()
    reference, reference_rows = load_reference(reference_report, source, runtime, native_runtime=native)
    inventory_reference, issues, records = load_inventory_reference(inventory_audit_report)
    if inventory_reference["reference_run"] != reference:
        raise ValueError("Inventory audit and delay controls must share their authenticated WTI reference")
    original_rows, inventory_summary = attach_inventory_features(features, issues, records)
    inventory_payload = jsonl_bytes([{
        **{key: row[key] for key in ("row_id", "forecast_date", "target_date")},
        "features": {column: row["features"][column] for column in INVENTORY_FEATURE_COLUMNS},
        "provenance": row["inventory_provenance"],
    } for row in original_rows])
    inventory_identity = {"sha256": sha256(inventory_payload), "bytes": len(inventory_payload)}
    control_reference, inventory_reference_rows = load_inventory_comparison_reference(
        inventory_reference_report, source, runtime, reference, inventory_reference,
        inventory_identity, native_runtime=native)
    selected_reference, regularization_reference_rows = load_inventory_regularization_reference(
        regularization_reference_report, source, runtime, reference, inventory_reference,
        inventory_identity, control_reference, native_runtime=native)
    delayed_rows, delay_summary = build_delayed_feature_rows(original_rows, wti, issues, records)
    delayed_payload = jsonl_bytes(delayed_rows)
    code = {str(path): sha256(path.read_bytes()) for directory in ("oil_audit", "oil_eval")
            for path in sorted(Path(directory).glob("*.py"))}
    identity = {
        "source": source, "protocol": protocol_config("inventory-availability-delay"),
        "source_code_sha256": code, "runtime_versions": runtime, "native_runtime_sha256": native,
        "requirements_sha256": sha256(Path("requirements.txt").read_bytes()), "numeric_threads": 1,
        "reference": reference, "inventory_reference": inventory_reference,
        "inventory_comparison_reference": control_reference,
        "inventory_regularization_reference": selected_reference,
        "inventory_features": inventory_identity, "bounded_wti_source": source_metadata,
        "baseline_wti_reconstruction": reconstruction,
        "delayed_features": {"sha256": sha256(delayed_payload), "bytes": len(delayed_payload)},
    }
    run_id = sha256(json_bytes(identity))[:20]
    destination = output_root / run_id
    predictions, summary = replay_delayed_development(
        original_rows, delayed_rows, events, labels, reference_rows=reference_rows,
        inventory_reference_rows=inventory_reference_rows,
        regularization_reference_rows=regularization_reference_rows)
    summary.update(inventory_feature_summary=inventory_summary,
                   delayed_feature_summary=delay_summary, baseline_wti_reconstruction=reconstruction)
    outputs = {"predictions.jsonl": jsonl_bytes(predictions), "summary.json": json_bytes(summary),
               "inventory_features.jsonl": inventory_payload, "delayed_features.jsonl": delayed_payload}
    manifest = {"run_id": run_id, **identity, "artifacts": {
        name: {"sha256": sha256(payload), "bytes": len(payload)} for name, payload in outputs.items()}}
    outputs["manifest.json"] = json_bytes(manifest)
    for name, payload in outputs.items():
        path = destination / name
        if path.exists() and path.read_bytes() != payload:
            raise ValueError("Refusing to replace different delay artifacts: " + str(path))
    destination.mkdir(parents=True, exist_ok=True)
    for name, payload in outputs.items():
        path = destination / name
        if not path.exists():
            path.write_bytes(payload)
    result = {"run_id": run_id, "evaluation_directory": str(destination),
              "dataset_id": source["dataset_id"], "runtime_versions": runtime,
              "reference_run_id": reference["run_id"],
              "inventory_audit_id": inventory_reference["audit_id"],
              "inventory_reference_run_id": control_reference["run_id"],
              "regularization_reference_run_id": selected_reference["run_id"], **summary}
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("xb") as stream:
        stream.write(json_bytes(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset-dir", "vintage-dir", "output-root", "report", "reference-report",
                 "inventory-audit-report", "inventory-reference-report", "regularization-reference-report"):
        parser.add_argument("--" + name, type=workspace_path, required=True)
    args = parser.parse_args()
    result = run(args.dataset_dir, args.vintage_dir, args.output_root, args.report,
                 reference_report=args.reference_report, inventory_audit_report=args.inventory_audit_report,
                 inventory_reference_report=args.inventory_reference_report,
                 regularization_reference_report=args.regularization_reference_report)
    print(json.dumps({key: result[key] for key in ("run_id", "evaluation_directory", "metrics",
                      "prediction_count", "scored_count", "unscored_count", "holdout_evaluated")}, indent=2))


if __name__ == "__main__":
    main()
