"""Fixed-constant scoring and authenticated retrospective artifact checks."""

from copy import deepcopy
from datetime import date
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from oil_audit.calendar import add_business_days, next_business_day
from oil_audit.dataset import json_bytes, jsonl_bytes, sha256
from oil_eval.benchmarks import (
    BENCHMARK_NAMES, benchmark_policy, constant_predictions,
    run, summarize_benchmarks,
)
from oil_eval.data import _decode
from oil_eval.models import COMPARISON_MODEL_NAMES, MODEL_NAMES
from oil_eval.protocol import protocol_config


def sample_rows(combined=False):
    rows = []
    for index, (day, label, probability, combined_probability) in enumerate((
        ("2017-01-05", 1, 0.6, 0.7), ("2017-01-19", 0, 0.7, 0.3),
        ("2018-01-04", 1, 0.3, 0.4), ("2018-01-18", 1, 0.8, 0.9),
        ("2018-01-25", None, 0.3, 0.7),
    )):
        target = add_business_days(date.fromisoformat(day), 5)
        models = {
            "majority": {"prediction": 1, "probability_up": 0.55},
            "momentum": {"prediction": index % 2, "probability_up": None,
                         "used_majority_fallback": False},
            "logistic": {"prediction": int(probability >= 0.5), "probability_up": probability,
                         "diagnostics": {"iterations": 3}},
        }
        if combined:
            models["logistic_wti_brent"] = {
                "prediction": int(combined_probability >= 0.5), "probability_up": combined_probability,
                "diagnostics": {"iterations": 4},
            }
        rows.append({
            "row_id": "wti-direction-" + day.replace("-", ""),
            "forecast_date": day, "target_date": target.isoformat(),
            "scoring_status": "available" if label is not None else "unavailable_by_development_cutoff",
            "label": label,
            "label_available_from": next_business_day(target).isoformat() if label is not None else None,
            "models": models, "training": {"input_sha256": str(index) * 64},
        })
    return rows


class ConstantBenchmarkTests(unittest.TestCase):
    def test_fixed_outputs_do_not_choose_the_majority_or_invent_probabilities(self):
        self.assertEqual(tuple(constant_predictions()), BENCHMARK_NAMES)
        self.assertEqual(constant_predictions(), {
            "always_up": {"prediction": 1, "probability_up": None},
            "always_not_up": {"prediction": 0, "probability_up": None},
            "constant_half": {"prediction": 1, "probability_up": 0.5},
        })
        changed = constant_predictions()
        changed["always_up"]["prediction"] = 0
        self.assertEqual(constant_predictions()["always_up"]["prediction"], 1)
        self.assertIn("including unchanged-price ties", benchmark_policy()["label_zero"])

    def test_known_imbalanced_metrics_years_and_pairing(self):
        rows = sample_rows(combined=True)
        original = deepcopy(rows)
        result = summarize_benchmarks(rows, ("logistic", "logistic_wti_brent"))
        self.assertEqual((result["prediction_count"], result["scored_count"], result["unscored_count"]),
                         (5, 4, 1))
        self.assertEqual(result["scored_direction_counts"], {"up": 3, "not_up": 1})
        up = result["metrics"]["always_up"]
        down = result["metrics"]["always_not_up"]
        half = result["metrics"]["constant_half"]
        self.assertEqual(up["accuracy"], 0.75)
        self.assertEqual(down["accuracy"], 0.25)
        self.assertEqual(up["balanced_accuracy"], 0.5)
        self.assertEqual(down["balanced_accuracy"], 0.5)
        self.assertEqual(up["confusion_counts"], {"tn": 0, "fp": 1, "fn": 0, "tp": 3})
        for direction_only in (up, down):
            self.assertIsNone(direction_only["brier_score"])
            self.assertIsNone(direction_only["log_loss"])
            self.assertIsNone(direction_only["roc_auc"])
        self.assertEqual(half["accuracy"], up["accuracy"])
        self.assertEqual(half["brier_score"], 0.25)
        self.assertAlmostEqual(half["log_loss"], math.log(2))
        self.assertEqual(half["roc_auc"], 0.5)
        self.assertEqual(result["yearly"]["2017"]["metrics"]["always_up"]["accuracy"], 0.5)
        self.assertEqual(result["yearly"]["2018"]["metrics"]["always_up"]["accuracy"], 1)
        paired = result["paired_comparisons"]["logistic"]["always_up"]
        self.assertEqual({key: paired[key] for key in
                          ("both_correct", "both_wrong", "model_only_correct", "benchmark_only_correct")},
                         {"both_correct": 2, "both_wrong": 1, "model_only_correct": 0,
                          "benchmark_only_correct": 1})
        self.assertEqual(paired["metric_differences_model_minus_benchmark"]["accuracy"], -0.25)
        self.assertEqual(rows, original)
        self.assertFalse(result["holdout_evaluated"])

    def test_empty_and_single_class_metrics_remain_explicit(self):
        result = summarize_benchmarks([])
        self.assertEqual(result["scored_count"], 0)
        self.assertIsNone(result["metrics"]["constant_half"]["brier_score"])
        one = summarize_benchmarks(sample_rows()[:1])
        self.assertIsNone(one["metrics"]["constant_half"]["roc_auc"])
        self.assertIsNone(one["metrics"]["always_up"]["balanced_accuracy"])
        self.assertEqual(one["metrics"]["constant_half"]["brier_score"], 0.25)

    def test_invalid_scoring_or_holdout_inputs_are_rejected(self):
        mutations = (
            lambda row: row.update(label=True),
            lambda row: row.update(label=2),
            lambda row: row.update(label_available_from="2023-01-03"),
            lambda row: row.update(label_available_from=row["target_date"]),
            lambda row: row.update(scoring_status="unknown"),
            lambda row: row.update(scoring_status="unavailable_by_development_cutoff"),
            lambda row: row.update(target_date="2023-01-03"),
        )
        for mutation in mutations:
            rows = sample_rows()
            mutation(rows[0])
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                summarize_benchmarks(rows)
        for rows in (list(reversed(sample_rows())), sample_rows() + sample_rows()[:1]):
            with self.assertRaisesRegex(ValueError, "unique chronological"):
                summarize_benchmarks(rows)


class SavedBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".test-benchmarks-", dir=".")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).relative_to(Path.cwd())
        self.source = {"dataset_id": "synthetic", "manifest_sha256": "a" * 64}
        self.runtime = {"python": "synthetic", "scikit-learn": "synthetic"}
        self.wti_report = self.root / "wti-report.json"
        self.brent_report = self.root / "brent-report.json"
        self.output_root = self.root / "supplemental"
        self.report = self.root / "benchmarks-report.json"
        self.wti_reference = self.save("wti")
        self.save("brent-comparison", reference=self.wti_reference)

    def save(self, experiment, rows=None, reference=None, identity_changes=None,
             summary_changes=None, payload_rows=None):
        combined = experiment == "brent-comparison"
        rows = sample_rows(combined) if rows is None else rows
        names = COMPARISON_MODEL_NAMES if combined else MODEL_NAMES
        checked = summarize_benchmarks(rows, names)
        summary = {
            "protocol": protocol_config(experiment), "holdout_evaluated": False,
            **{key: checked[key] for key in ("prediction_count", "scored_count", "unscored_count",
                                           "scored_direction_counts")},
            "metrics": checked["comparator_metrics"],
            "yearly": {year: {"prediction_count": group["prediction_count"],
                              "metrics": group["comparator_metrics"]}
                       for year, group in checked["yearly"].items()},
        }
        summary.update(summary_changes or {})
        directory = self.root / experiment
        directory.mkdir(exist_ok=True)
        identity = {"source": self.source, "protocol": protocol_config(experiment),
                    "runtime_versions": self.runtime, "numeric_threads": 1,
                    "source_code_sha256": {"oil_eval/models.py": "b" * 64},
                    "requirements_sha256": "c" * 64}
        if combined:
            identity["reference"] = reference
        identity.update(identity_changes or {})
        payload = (jsonl_bytes(rows) if payload_rows is None else
                   b"".join((row if isinstance(row, bytes) else json.dumps(row).encode()) + b"\n"
                            for row in payload_rows))
        payloads = {"predictions.jsonl": payload, "summary.json": json_bytes(summary)}
        manifest = {"run_id": sha256(json_bytes(identity))[:20], **identity,
                    "artifacts": {name: {"sha256": sha256(value), "bytes": len(value)}
                                  for name, value in payloads.items()}}
        payloads["manifest.json"] = json_bytes(manifest)
        for name, value in payloads.items():
            (directory / name).write_bytes(value)
        report = {"run_id": manifest["run_id"], "evaluation_directory": str(directory),
                  "dataset_id": self.source["dataset_id"], "runtime_versions": identity["runtime_versions"],
                  **summary}
        if combined:
            report["reference_run_id"] = identity["reference"]["run_id"]
        report_path = self.brent_report if combined else self.wti_report
        report_path.write_bytes(json_bytes(report))
        return {"run_id": manifest["run_id"], "evaluation_directory": str(directory),
                "report_sha256": sha256(report_path.read_bytes()),
                "manifest_sha256": sha256(payloads["manifest.json"]),
                "artifacts": manifest["artifacts"]}

    def build(self, report=None):
        return run(self.wti_report, self.brent_report, self.output_root,
                   self.report if report is None else report)

    def test_reproducible_supplement_preserves_sources_without_loading_data_or_fitting(self):
        source_files = [self.wti_report, self.brent_report]
        source_files += [self.root / experiment / name
                         for experiment in ("wti", "brent-comparison")
                         for name in ("predictions.jsonl", "summary.json", "manifest.json")]
        before = {path: path.read_bytes() for path in source_files}
        with patch("oil_eval.data.load_development_data", side_effect=AssertionError("dataset accessed")), \
             patch("oil_eval.models.Pipeline.fit", side_effect=AssertionError("model fitted")):
            first = self.build()
            second = self.build()
        self.assertEqual(first, second)
        self.assertEqual(first["prediction_count"], 5)
        self.assertEqual(first["scored_count"], 4)
        self.assertTrue(first["reference_reproduction"]["forecast_cohort_identical"])
        self.assertEqual(before, {path: path.read_bytes() for path in source_files})
        directory = Path(first["benchmark_directory"])
        manifest = json.loads((directory / "manifest.json").read_text())
        self.assertEqual(set(manifest["artifacts"]), {"benchmarks.jsonl", "summary.json"})
        for name, expected in manifest["artifacts"].items():
            payload = (directory / name).read_bytes()
            self.assertEqual((sha256(payload), len(payload)), (expected["sha256"], expected["bytes"]))
        rows = [json.loads(line) for line in (directory / "benchmarks.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 5)
        self.assertIsNone(rows[-1]["label"])
        self.assertTrue(all(row["benchmarks"] == constant_predictions() for row in rows))

    def test_changed_artifact_report_and_identity_are_rejected(self):
        path = self.root / "wti" / "predictions.jsonl"
        path.write_bytes(path.read_bytes() + b"corrupted\n")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.build()
        self.restore_sources()
        report = json.loads(self.wti_report.read_text())
        report["scored_count"] += 1
        self.wti_report.write_bytes(json_bytes(report))
        with self.assertRaisesRegex(ValueError, "report differs"):
            self.build()
        self.restore_sources()
        path = self.root / "wti" / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["requirements_sha256"] = "d" * 64
        path.write_bytes(json_bytes(manifest))
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.build()

    def restore_sources(self):
        self.wti_reference = self.save("wti")
        self.save("brent-comparison", reference=self.wti_reference)

    def test_source_runtime_and_reference_mismatches_are_rejected(self):
        for change in (
            {"source": {**self.source, "manifest_sha256": "f" * 64}},
            {"runtime_versions": {**self.runtime, "python": "different"}},
            {"reference": {**self.wti_reference, "report_sha256": "f" * 64}},
        ):
            self.save("brent-comparison", reference=self.wti_reference, identity_changes=change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "share the source, runtime"):
                self.build()

    def test_protocol_and_closed_holdout_checks(self):
        self.save("wti", identity_changes={"protocol": {**protocol_config(), "minimum_training_rows": 1}})
        with self.assertRaisesRegex(ValueError, "protocol differs"):
            self.build()
        self.save("wti", summary_changes={"holdout_evaluated": True})
        with self.assertRaisesRegex(ValueError, "closed-holdout"):
            self.build()

    def test_holdout_poison_is_rejected_before_json_decoding(self):
        poison = b'{"forecast_date":"2023-01-03", heldout_poison_cannot_decode}'
        self.save("wti", payload_rows=sample_rows() + [poison])

        def guarded_decode(payload):
            self.assertNotIn(b"heldout_poison", payload)
            return _decode(payload)

        with patch("oil_eval.benchmarks._decode", side_effect=guarded_decode):
            with self.assertRaisesRegex(ValueError, "contains a holdout forecast"):
                self.build()

    def test_recomputed_saved_metrics_reject_consistently_rehashed_bad_summary(self):
        checked = summarize_benchmarks(sample_rows(), MODEL_NAMES)
        wrong = deepcopy(checked["comparator_metrics"])
        wrong["logistic"]["accuracy"] = 1.0
        self.save("wti", summary_changes={"metrics": wrong})
        with self.assertRaisesRegex(ValueError, "model metrics differ"):
            self.build()
        yearly = {year: {"prediction_count": group["prediction_count"],
                         "metrics": deepcopy(group["comparator_metrics"])}
                  for year, group in checked["yearly"].items()}
        yearly["2017"]["metrics"]["logistic"]["accuracy"] = 0.123
        self.save("wti", summary_changes={"yearly": yearly})
        with self.assertRaisesRegex(ValueError, "yearly metrics differ"):
            self.build()

    def test_exact_scoring_cohort_is_required(self):
        for change in ("label", "label_available_from", "removed_date"):
            rows = sample_rows(combined=True)
            if change == "label":
                rows[0]["label"] = 0
            elif change == "label_available_from":
                rows[0]["label_available_from"] = "2017-01-17"
            else:
                rows = rows[:-1]
            self.save("brent-comparison", rows=rows, reference=self.wti_reference)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "cohort differs"):
                self.build()

    def test_original_and_supplemental_artifacts_cannot_be_overwritten(self):
        protected = [self.wti_report, self.brent_report]
        protected += [self.root / experiment / name for experiment in ("wti", "brent-comparison")
                      for name in ("manifest.json", "summary.json", "predictions.jsonl")]
        for path in protected:
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "cannot overwrite an original"):
                self.build(path)
        result = self.build()
        artifact = Path(result["benchmark_directory"]) / "summary.json"
        with self.assertRaisesRegex(ValueError, "cannot overwrite its immutable"):
            self.build(artifact)
        artifact.write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "Refusing to replace"):
            self.build()

    def test_cli_cannot_open_dataset_or_holdout(self):
        for flag in ("--holdout", "--dataset-dir"):
            result = subprocess.run([sys.executable, "-m", "oil_eval.benchmarks", flag],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("unrecognized arguments", result.stderr)


if __name__ == "__main__":
    unittest.main()
