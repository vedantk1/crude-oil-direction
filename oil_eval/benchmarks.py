"""Fixed constant benchmarks, including a retrospective development-only CLI.

The CLI authenticates existing model-run artifacts and never fits a model or
opens the source dataset. Historical reports and artifacts remain unchanged.
"""

import argparse
from collections import Counter
from datetime import date
from pathlib import Path
import platform
import json

from oil_audit.dataset import json_bytes, jsonl_bytes, sha256, workspace_path
from .data import _decode, _gate_date, _validate_identity
from .metrics import classification_metrics
from .models import COMPARISON_MODEL_NAMES, MODEL_NAMES
from .protocol import HOLDOUT_START, forecast_role, protocol_config
from .reference import COHORT_FIELDS, verify_reference_reproduction


BENCHMARK_NAMES = ("always_up", "always_not_up", "constant_half")
WTI_REPORT = Path("reports/development_evaluation.json")
BRENT_REPORT = Path("reports/brent_comparison.json")
BENCHMARK_REPORT = Path("reports/development_benchmarks.json")
METRIC_NAMES = ("accuracy", "balanced_accuracy", "roc_auc", "brier_score", "log_loss")
# Only code used by this diagnostic's scoring, authentication, fixed feature
# contracts and calendar/serialization helpers belongs in its run identity.
BENCHMARK_DEPENDENCIES = (
    "oil_eval/benchmarks.py", "oil_eval/metrics.py", "oil_eval/data.py",
    "oil_eval/models.py", "oil_eval/protocol.py", "oil_eval/reference.py",
    "oil_audit/dataset.py", "oil_audit/calendar.py", "oil_audit/features.py",
)


def benchmark_policy() -> dict:
    return {
        "benchmark_id": "fixed-development-constants-v1",
        "selection": "all three constants fixed independently of observed outcomes; no best-class selection",
        "label_zero": "not up, including unchanged-price ties",
        "always_up": {"prediction": 1, "probability_up": None},
        "always_not_up": {"prediction": 0, "probability_up": None},
        "constant_half": {"prediction": 1, "probability_up": 0.5,
                          "threshold": 0.5, "tie_prediction": 1},
        "scoring": "same first-available outcomes and development dates as the saved models",
        "interpretation": "retrospective development diagnostic; not a newly selected model or holdout result",
    }


def constant_predictions() -> dict:
    """Return fresh outputs without consulting any data or class frequencies."""
    return {
        "always_up": {"prediction": 1, "probability_up": None},
        "always_not_up": {"prediction": 0, "probability_up": None},
        "constant_half": {"prediction": 1, "probability_up": 0.5},
    }


def _validate_scoring_rows(rows):
    for row in rows:
        forecast, target = _validate_identity(row)
        if forecast_role(row) != "development":
            raise ValueError("Benchmarks require development-only forecast and target dates")
        status = row.get("scoring_status")
        if status == "available":
            if type(row.get("label")) is not int or row["label"] not in (0, 1):
                raise ValueError("Scored benchmark labels must be binary integers")
            try:
                available = date.fromisoformat(row["label_available_from"])
            except (TypeError, ValueError) as exc:
                raise ValueError("Invalid benchmark label availability") from exc
            if not target < available < HOLDOUT_START:
                raise ValueError("Benchmark outcome crosses the development availability boundary")
        elif status == "unavailable_by_development_cutoff":
            if row.get("label") is not None or row.get("label_available_from") is not None:
                raise ValueError("Unscored benchmark outcomes must remain null")
        else:
            raise ValueError("Unknown benchmark scoring status")


def _model_metrics(scored, name):
    probabilities = [row["models"][name]["probability_up"] for row in scored]
    if probabilities and any(value is None for value in probabilities) and not all(
            value is None for value in probabilities):
        raise ValueError("Saved model has incomplete probability estimates")
    return classification_metrics(
        [row["label"] for row in scored],
        [row["models"][name]["prediction"] for row in scored],
        probabilities if probabilities and probabilities[0] is not None else None,
    )


def _group_summary(rows, model_names):
    scored = [row for row in rows if row["scoring_status"] == "available"]
    labels = [row["label"] for row in scored]
    constants = constant_predictions()
    metrics = {name: classification_metrics(
        labels, [output["prediction"]] * len(scored),
        None if output["probability_up"] is None else [output["probability_up"]] * len(scored),
    ) for name, output in constants.items()}
    model_metrics = {name: _model_metrics(scored, name) for name in model_names}
    paired = {}
    for name in model_names:
        paired[name] = {}
        for benchmark, output in constants.items():
            counts = Counter()
            for row in scored:
                model_correct = row["models"][name]["prediction"] == row["label"]
                benchmark_correct = output["prediction"] == row["label"]
                category = ("both_correct" if model_correct and benchmark_correct else
                            "both_wrong" if not model_correct and not benchmark_correct else
                            "model_only_correct" if model_correct else "benchmark_only_correct")
                counts[category] += 1
            paired[name][benchmark] = {
                **{key: counts[key] for key in
                   ("both_correct", "both_wrong", "model_only_correct", "benchmark_only_correct")},
                "metric_differences_model_minus_benchmark": {
                    metric: (model_metrics[name][metric] - metrics[benchmark][metric]
                             if model_metrics[name][metric] is not None and metrics[benchmark][metric] is not None
                             else None) for metric in METRIC_NAMES},
            }
    return {
        "prediction_count": len(rows), "scored_count": len(scored),
        "unscored_count": len(rows) - len(scored),
        "scored_direction_counts": {"up": sum(labels), "not_up": len(labels) - sum(labels)},
        "metrics": metrics, "comparator_metrics": model_metrics,
        "paired_comparisons": paired,
    }


def summarize_benchmarks(predictions, model_names=("logistic",)) -> dict:
    """Score fixed constants on exactly the supplied development outcomes."""
    _validate_scoring_rows(predictions)
    if (len({row["row_id"] for row in predictions}) != len(predictions)
            or [row["forecast_date"] for row in predictions] != sorted(
                {row["forecast_date"] for row in predictions})):
        raise ValueError("Benchmark forecasts must have unique chronological dates and identities")
    years = sorted({row["forecast_date"][:4] for row in predictions})
    return {
        "policy": benchmark_policy(), **_group_summary(predictions, model_names),
        "yearly": {year: _group_summary(
            [row for row in predictions if row["forecast_date"].startswith(year)], model_names)
            for year in years},
        "holdout_evaluated": False,
    }


def _load_saved_run(report_path: Path, experiment: str):
    """Authenticate a saved run using only its report, manifest and artifacts."""
    report_bytes = report_path.read_bytes()
    report = _decode(report_bytes)
    directory = workspace_path(report["evaluation_directory"])
    manifest_bytes = (directory / "manifest.json").read_bytes()
    manifest = _decode(manifest_bytes)
    identity = {key: value for key, value in manifest.items() if key not in ("run_id", "artifacts")}
    if sha256(json_bytes(identity))[:20] != manifest.get("run_id"):
        raise ValueError("Saved benchmark source run identity mismatch")
    if manifest["protocol"] != protocol_config(experiment) or manifest["numeric_threads"] != 1:
        raise ValueError("Saved benchmark source protocol differs from the fixed development run")
    if set(manifest["artifacts"]) != {"predictions.jsonl", "summary.json"}:
        raise ValueError("Unexpected saved benchmark source artifacts")
    payloads = {}
    for name, expected in manifest["artifacts"].items():
        payload = (directory / name).read_bytes()
        if sha256(payload) != expected["sha256"] or len(payload) != expected["bytes"]:
            raise ValueError("Saved benchmark source artifact checksum mismatch: " + name)
        payloads[name] = payload
    summary = _decode(payloads["summary.json"])
    expected_report = {
        "run_id": manifest["run_id"], "evaluation_directory": str(directory),
        "dataset_id": manifest["source"]["dataset_id"],
        "runtime_versions": manifest["runtime_versions"], **summary,
    }
    if experiment == "brent-comparison":
        expected_report["reference_run_id"] = manifest["reference"]["run_id"]
    if (report != expected_report or summary["protocol"] != protocol_config(experiment)
            or summary["holdout_evaluated"] is not False):
        raise ValueError("Saved benchmark report differs from its closed-holdout artifacts")
    names = COMPARISON_MODEL_NAMES if experiment == "brent-comparison" else MODEL_NAMES
    rows = []
    for line in payloads["predictions.jsonl"].splitlines():
        if _gate_date(line, "forecast_date") >= HOLDOUT_START:
            raise ValueError("Saved benchmark source contains a holdout forecast")
        row = _decode(line)
        if set(row["models"]) != set(names):
            raise ValueError("Saved benchmark source model family differs")
        rows.append(row)
    if not rows:
        raise ValueError("Saved benchmark source contains no forecasts")
    checked = summarize_benchmarks(rows, names)
    if any(summary[key] != checked[key] for key in
           ("prediction_count", "scored_count", "unscored_count", "scored_direction_counts")):
        raise ValueError("Saved benchmark source counts differ from its summary")
    if summary["metrics"] != checked["comparator_metrics"]:
        raise ValueError("Saved benchmark source model metrics differ from predictions")
    if set(summary["yearly"]) != set(checked["yearly"]) or any(
        summary["yearly"][year]["prediction_count"] != group["prediction_count"]
        or summary["yearly"][year]["metrics"] != group["comparator_metrics"]
        for year, group in checked["yearly"].items()
    ):
        raise ValueError("Saved benchmark source yearly metrics differ from predictions")
    provenance = {
        "run_id": manifest["run_id"], "evaluation_directory": str(directory),
        "report_sha256": sha256(report_bytes), "manifest_sha256": sha256(manifest_bytes),
        "artifacts": manifest["artifacts"],
    }
    return provenance, manifest, rows


def run(wti_report: Path = WTI_REPORT, brent_report: Path = BRENT_REPORT,
        output_root: Path = Path("data/evaluations"), report: Path = BENCHMARK_REPORT) -> dict:
    """Write supplemental diagnostics while preserving both original runs."""
    paths = [workspace_path(str(path)) for path in (wti_report, brent_report, output_root, report)]
    wti_report, brent_report, output_root, report = paths
    wti_reference, wti_manifest, wti_rows = _load_saved_run(wti_report, "wti")
    brent_reference, brent_manifest, brent_rows = _load_saved_run(brent_report, "brent-comparison")
    if (wti_manifest["source"] != brent_manifest["source"]
            or wti_manifest["runtime_versions"] != brent_manifest["runtime_versions"]
            or brent_manifest["reference"] != wti_reference):
        raise ValueError("Saved benchmark runs must share the source, runtime and authenticated WTI reference")
    reproduction = verify_reference_reproduction(brent_rows, wti_rows)
    protected = [wti_report, brent_report, WTI_REPORT, BRENT_REPORT]
    for reference in (wti_reference, brent_reference):
        protected += [Path(reference["evaluation_directory"]) / name
                      for name in ("manifest.json", "summary.json", "predictions.jsonl")]
    if report.resolve() in {path.resolve() for path in protected}:
        raise ValueError("Benchmark report cannot overwrite an original report or artifact")
    comparison_rows = [{
        **{key: wti[key] for key in COHORT_FIELDS},
        "models": {"logistic": wti["models"]["logistic"],
                   "logistic_wti_brent": brent["models"]["logistic_wti_brent"]},
    } for wti, brent in zip(wti_rows, brent_rows)]
    summary = summarize_benchmarks(comparison_rows, ("logistic", "logistic_wti_brent"))
    summary.update({
        "analysis_kind": "supplemental retrospective development diagnostics; no model fitting",
        "forecast_date_range": [comparison_rows[0]["forecast_date"], comparison_rows[-1]["forecast_date"]],
        "reference_reproduction": reproduction,
    })
    identity = {
        "source_runs": {"wti": {"report_path": str(wti_report), **wti_reference},
                        "brent": {"report_path": str(brent_report), **brent_reference}},
        "policy": benchmark_policy(), "python_version": platform.python_version(),
        "source_code_sha256": {name: sha256(Path(name).read_bytes())
                               for name in BENCHMARK_DEPENDENCIES},
    }
    run_id = sha256(json_bytes(identity))[:20]
    destination = output_root / run_id
    compact = [{
        **{key: row[key] for key in COHORT_FIELDS}, "benchmarks": constant_predictions(),
        "comparators": {name: {key: output[key] for key in ("prediction", "probability_up")}
                        for name, output in row["models"].items()},
    } for row in comparison_rows]
    outputs = {"benchmarks.jsonl": jsonl_bytes(compact), "summary.json": json_bytes(summary)}
    manifest = {"run_id": run_id, **identity,
                "artifacts": {name: {"sha256": sha256(payload), "bytes": len(payload)}
                              for name, payload in outputs.items()}}
    outputs["manifest.json"] = json_bytes(manifest)
    if report.resolve() in {(destination / name).resolve() for name in outputs}:
        raise ValueError("Benchmark report cannot overwrite its immutable artifacts")
    for name, payload in outputs.items():
        path = destination / name
        if path.exists() and path.read_bytes() != payload:
            raise ValueError("Refusing to replace a different benchmark artifact: " + str(path))
    destination.mkdir(parents=True, exist_ok=True)
    for name, payload in outputs.items():
        path = destination / name
        if not path.exists():
            path.write_bytes(payload)
    result = {"run_id": run_id, "benchmark_directory": str(destination),
              "source_runs": identity["source_runs"], **summary}
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_bytes(json_bytes(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wti-report", type=workspace_path, default=WTI_REPORT)
    parser.add_argument("--brent-report", type=workspace_path, default=BRENT_REPORT)
    parser.add_argument("--output-root", type=workspace_path, default=Path("data/evaluations"))
    parser.add_argument("--report", type=workspace_path, default=BENCHMARK_REPORT)
    args = parser.parse_args()
    result = run(args.wti_report, args.brent_report, args.output_root, args.report)
    print(json.dumps({key: result[key] for key in (
        "run_id", "benchmark_directory", "prediction_count", "scored_count", "unscored_count",
        "metrics", "holdout_evaluated")}, indent=2))


if __name__ == "__main__":
    main()
