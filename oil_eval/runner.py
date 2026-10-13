"""Reproducible development-only evaluation artifacts and CLI."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform

from oil_audit.dataset import json_bytes, jsonl_bytes, sha256, workspace_path
from .data import load_development_data
from .protocol import EXPERIMENTS, protocol_config
from .reference import load_reference
from .replay import replay_development


WTI_REPORT = Path("reports/development_evaluation.json")
BRENT_REPORT = Path("reports/brent_comparison.json")


def run(dataset_dir: Path, output_root: Path = Path("data/evaluations"),
        report: Path | None = None, *, experiment: str = "wti",
        reference_report: Path = WTI_REPORT) -> dict:
    protocol = protocol_config(experiment)
    comparison = experiment == "brent-comparison"
    report = report if report is not None else (BRENT_REPORT if comparison else WTI_REPORT)
    source, features, events, labels = (
        load_development_data(dataset_dir, feature_set="wti_brent") if comparison
        else load_development_data(dataset_dir))
    source_code = {str(path): sha256(path.read_bytes())
                   for directory in ("oil_eval", "oil_audit")
                   for path in sorted(Path(directory).glob("*.py"))}
    runtime = {name: version(name) for name in
               ("scikit-learn", "numpy", "scipy", "joblib", "threadpoolctl")}
    runtime["python"] = platform.python_version()
    identity = {"source": source, "protocol": protocol,
                "source_code_sha256": source_code, "runtime_versions": runtime,
                "requirements_sha256": sha256(Path("requirements.txt").read_bytes()),
                "numeric_threads": 1}
    reference_rows = None
    if comparison:
        reference, reference_rows = load_reference(reference_report, source, runtime)
        reference_directory = Path(reference["evaluation_directory"])
        protected = [reference_report, WTI_REPORT] + [reference_directory / name for name in
                                                     ("manifest.json", "summary.json", "predictions.jsonl")]
        if report.resolve() in {path.resolve() for path in protected}:
            raise ValueError("Comparison report cannot overwrite the WTI reference")
        identity["reference"] = reference
    run_id = sha256(json_bytes(identity))[:20]
    destination = output_root / run_id
    if report.resolve() in {(destination / name).resolve() for name in
                            ("manifest.json", "summary.json", "predictions.jsonl")}:
        raise ValueError("Report cannot overwrite an immutable evaluation artifact")
    if comparison:
        predictions, summary = replay_development(
            features, events, labels, experiment=experiment, reference_rows=reference_rows)
    else:
        predictions, summary = replay_development(features, events, labels)
    outputs = {"predictions.jsonl": jsonl_bytes(predictions), "summary.json": json_bytes(summary)}
    manifest = {"run_id": run_id, **identity,
                "artifacts": {name: {"sha256": sha256(payload), "bytes": len(payload)}
                              for name, payload in outputs.items()}}
    outputs["manifest.json"] = json_bytes(manifest)
    destination.mkdir(parents=True, exist_ok=True)
    for name, payload in outputs.items():
        path = destination / name
        if path.exists() and path.read_bytes() != payload:
            raise ValueError("Refusing to replace a different evaluation artifact: " + str(path))
        if not path.exists():
            path.write_bytes(payload)
    result = {"run_id": run_id, "evaluation_directory": str(destination),
              "dataset_id": source["dataset_id"], "runtime_versions": runtime, **summary}
    if comparison:
        result["reference_run_id"] = reference["run_id"]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_bytes(json_bytes(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=workspace_path, required=True)
    parser.add_argument("--output-root", type=workspace_path, default=Path("data/evaluations"))
    parser.add_argument("--report", type=workspace_path,
                        help="defaults to a separate report for each experiment")
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="wti")
    parser.add_argument("--reference-report", type=workspace_path, default=WTI_REPORT,
                        help="authenticated WTI development report for brent-comparison")
    args = parser.parse_args()
    result = run(args.dataset_dir, args.output_root, args.report,
                 experiment=args.experiment, reference_report=args.reference_report)
    print(json.dumps({key: result[key] for key in (
        "run_id", "evaluation_directory", "prediction_count", "scored_count", "unscored_count",
        "metrics", "holdout_evaluated")}, indent=2))


if __name__ == "__main__":
    main()
