"""Reproducible development-only evaluation artifacts and CLI."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform

from oil_audit.dataset import json_bytes, jsonl_bytes, sha256, workspace_path
from .data import load_development_data
from .protocol import protocol_config
from .replay import replay_development


def run(dataset_dir: Path, output_root: Path = Path("data/evaluations"),
        report: Path = Path("reports/development_evaluation.json")) -> dict:
    source, features, events, labels = load_development_data(dataset_dir)
    source_code = {str(path): sha256(path.read_bytes())
                   for directory in ("oil_eval", "oil_audit")
                   for path in sorted(Path(directory).glob("*.py"))}
    runtime = {name: version(name) for name in
               ("scikit-learn", "numpy", "scipy", "joblib", "threadpoolctl")}
    runtime["python"] = platform.python_version()
    identity = {"source": source, "protocol": protocol_config(),
                "source_code_sha256": source_code, "runtime_versions": runtime,
                "requirements_sha256": sha256(Path("requirements.txt").read_bytes()),
                "numeric_threads": 1}
    run_id = sha256(json_bytes(identity))[:20]
    predictions, summary = replay_development(features, events, labels)
    destination = output_root / run_id
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
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_bytes(json_bytes(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=workspace_path, required=True)
    parser.add_argument("--output-root", type=workspace_path, default=Path("data/evaluations"))
    parser.add_argument("--report", type=workspace_path, default=Path("reports/development_evaluation.json"))
    args = parser.parse_args()
    result = run(args.dataset_dir, args.output_root, args.report)
    print(json.dumps({key: result[key] for key in (
        "run_id", "evaluation_directory", "prediction_count", "scored_count", "unscored_count",
        "metrics", "holdout_evaluated")}, indent=2))


if __name__ == "__main__":
    main()
