"""Synthetic reference authentication and exact development-cohort safeguards."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from oil_audit.dataset import json_bytes, sha256
from oil_eval.data import _decode
from oil_eval.models import MODEL_NAMES
from oil_eval.protocol import protocol_config
from oil_eval.reference import (
    load_reference, validate_reference_schedule, verify_reference_reproduction,
)


def reference_rows():
    rows = []
    for index, (forecast, target) in enumerate((
        ("2017-01-05", "2017-01-12"), ("2017-01-19", "2017-01-26"),
    )):
        rows.append({
            "row_id": "wti-direction-" + forecast.replace("-", ""),
            "forecast_date": forecast, "target_date": target,
            "scoring_status": "available" if index == 0 else "unavailable_by_development_cutoff",
            "label": 1 if index == 0 else None,
            "label_available_from": "2017-01-13" if index == 0 else None,
            "training": {"input_sha256": str(index) * 64},
            "models": {
                "majority": {"prediction": 1, "probability_up": 0.5},
                "momentum": {"prediction": 0, "probability_up": None,
                             "used_majority_fallback": False},
                "logistic": {"prediction": 1, "probability_up": 0.6,
                             "diagnostics": {"iterations": 3}},
            },
        })
    return rows


def source_rows(rows):
    features, scoring = [], []
    for row in rows:
        identity = {key: row[key] for key in ("row_id", "forecast_date", "target_date")}
        features.append({**identity, "eligibility": {"wti": True, "wti_brent": True}})
        if row["scoring_status"] == "available":
            scoring.append({**identity, "label": row["label"],
                            "available_from": row["label_available_from"]})
    return features, scoring


class ReferenceAuthenticationTests(unittest.TestCase):
    def setUp(self):
        # The saved report intentionally requires a workspace-relative path.
        self.temp = tempfile.TemporaryDirectory(prefix=".test-reference-", dir=".")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).relative_to(Path.cwd())
        self.directory = self.root / "reference"
        self.directory.mkdir()
        self.report_path = self.root / "report.json"
        self.source = {"dataset_id": "synthetic-dataset", "manifest_sha256": "a" * 64}
        self.runtime = {"python": "synthetic", "scikit-learn": "synthetic"}

    def save(self, rows=None, identity_changes=None, summary_changes=None):
        rows = reference_rows() if rows is None else rows
        summary = {
            "protocol": protocol_config(), "holdout_evaluated": False,
            "prediction_count": len(rows),
            "scored_count": sum(isinstance(row, dict) and row["scoring_status"] == "available"
                                for row in rows),
        }
        summary.update(summary_changes or {})
        payloads = {
            "predictions.jsonl": b"".join(
                (row if isinstance(row, bytes) else json.dumps(row).encode()) + b"\n"
                for row in rows),
            "summary.json": json_bytes(summary),
        }
        identity = {
            "source": self.source, "protocol": protocol_config(),
            "runtime_versions": self.runtime, "numeric_threads": 1,
            "source_code_sha256": {"oil_eval/models.py": "b" * 64},
            "requirements_sha256": "c" * 64,
        }
        identity.update(identity_changes or {})
        manifest = {
            **identity, "run_id": sha256(json_bytes(identity))[:20],
            "artifacts": {name: {"sha256": sha256(payload), "bytes": len(payload)}
                          for name, payload in payloads.items()},
        }
        for name, payload in payloads.items():
            (self.directory / name).write_bytes(payload)
        (self.directory / "manifest.json").write_bytes(json_bytes(manifest))
        report = {
            "run_id": manifest["run_id"], "evaluation_directory": str(self.directory),
            "dataset_id": self.source["dataset_id"], "runtime_versions": self.runtime,
            **summary,
        }
        self.report_path.write_bytes(json_bytes(report))
        return manifest, report

    def load(self):
        return load_reference(self.report_path, self.source, self.runtime)

    def test_authenticates_manifest_report_and_both_artifacts(self):
        manifest, _ = self.save()
        identity, rows = self.load()
        self.assertEqual(rows, reference_rows())
        self.assertEqual(identity, {
            "run_id": manifest["run_id"],
            "evaluation_directory": str(self.directory),
            "report_sha256": sha256(self.report_path.read_bytes()),
            "manifest_sha256": sha256((self.directory / "manifest.json").read_bytes()),
            "artifacts": manifest["artifacts"],
        })

    def test_manifest_identity_corruption_is_rejected(self):
        manifest, _ = self.save()
        manifest["source_code_sha256"]["oil_eval/models.py"] = "d" * 64
        (self.directory / "manifest.json").write_bytes(json_bytes(manifest))
        with self.assertRaisesRegex(ValueError, "identity does not match"):
            self.load()

    def test_dataset_protocol_runtime_and_thread_mismatches_are_rejected(self):
        changes = (
            {"source": {**self.source, "dataset_id": "other-dataset"}},
            {"protocol": {**protocol_config(), "minimum_training_rows": 1}},
            {"runtime_versions": {**self.runtime, "python": "different"}},
            {"numeric_threads": 2},
        )
        for change in changes:
            with self.subTest(change=change):
                self.save(identity_changes=change)
                with self.assertRaisesRegex(ValueError, "dataset, protocol or runtime differs"):
                    self.load()

    def test_payload_corruption_is_rejected_before_json_decoding(self):
        for artifact in ("predictions.jsonl", "summary.json"):
            with self.subTest(artifact=artifact):
                self.save()
                (self.directory / artifact).write_bytes(b"corrupted, not JSON")
                with self.assertRaisesRegex(ValueError, "artifact checksum mismatch"):
                    self.load()

    def test_artifact_size_or_manifest_artifact_set_mismatch_is_rejected(self):
        manifest, _ = self.save()
        manifest["artifacts"]["predictions.jsonl"]["bytes"] += 1
        (self.directory / "manifest.json").write_bytes(json_bytes(manifest))
        with self.assertRaisesRegex(ValueError, "artifact checksum mismatch"):
            self.load()
        manifest, _ = self.save()
        manifest["artifacts"]["unexpected.json"] = {"sha256": "e" * 64, "bytes": 0}
        (self.directory / "manifest.json").write_bytes(json_bytes(manifest))
        with self.assertRaisesRegex(ValueError, "Unexpected reference artifacts"):
            self.load()

    def test_changed_report_and_open_holdout_summary_are_rejected(self):
        _, report = self.save()
        report["scored_count"] += 1
        self.report_path.write_bytes(json_bytes(report))
        with self.assertRaisesRegex(ValueError, "report does not match"):
            self.load()
        for change in ({"holdout_evaluated": True},
                       {"protocol": {**protocol_config(), "holdout_start": "2024-01-01"}}):
            with self.subTest(change=change):
                self.save(summary_changes=change)
                with self.assertRaisesRegex(ValueError, "report does not match"):
                    self.load()

    def test_holdout_poison_is_rejected_before_decoding_it(self):
        poison = b'{"forecast_date":"2023-01-03", heldout_poison_cannot_decode}'
        self.save(rows=[reference_rows()[0], poison])

        def guarded_decode(payload):
            self.assertNotIn(b"heldout_poison", payload)
            return _decode(payload)

        with patch("oil_eval.reference._decode", side_effect=guarded_decode):
            with self.assertRaisesRegex(ValueError, "contains a holdout forecast"):
                self.load()

    def test_non_development_reference_dates_and_extra_models_are_rejected(self):
        for change in (
            {"forecast_date": "2016-12-22", "target_date": "2016-12-30"},
            {"forecast_date": "2022-12-29", "target_date": "2023-01-06"},
        ):
            rows = reference_rows()
            rows[0].update(change)
            self.save(rows=rows)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "outside the development window"):
                self.load()
        rows = reference_rows()
        rows[0]["models"]["logistic_wti_brent"] = {"prediction": 1}
        self.save(rows=rows)
        with self.assertRaisesRegex(ValueError, "original WTI model family"):
            self.load()

    def test_count_order_and_duplicate_identity_mismatches_are_rejected(self):
        duplicate = reference_rows()
        duplicate[1]["row_id"] = duplicate[0]["row_id"]
        variants = (
            (reference_rows(), {"prediction_count": 3}),
            (reference_rows(), {"scored_count": 2}),
            (list(reversed(reference_rows())), {}),
            (duplicate, {}),
        )
        for rows, changes in variants:
            with self.subTest(changes=changes, rows=rows):
                self.save(rows=rows, summary_changes=changes)
                with self.assertRaisesRegex(ValueError, "counts or ordering differ"):
                    self.load()


class ReferenceScheduleTests(unittest.TestCase):
    def test_exact_development_cohort_accepts_unsorted_features_and_other_roles(self):
        rows = reference_rows()
        features, scoring = source_rows(rows)
        other = deepcopy(features[0])
        other.update(row_id="initial-training", forecast_date="2011-04-07", target_date="2011-04-14")
        validate_reference_schedule(list(reversed(features)) + [other], scoring, rows)

    def test_changed_dates_targets_ids_or_scoring_states_are_rejected(self):
        rows = reference_rows()
        for field, value in (
            ("row_id", "different-id"), ("forecast_date", "2017-01-06"),
            ("target_date", "2017-01-13"), ("label", 0),
            ("label_available_from", "2017-01-17"),
            ("scoring_status", "unavailable_by_development_cutoff"),
        ):
            features, scoring = source_rows(rows)
            changed = deepcopy(rows)
            changed[0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "dates, targets or scoring labels differ"):
                validate_reference_schedule(features, scoring, changed)
        features, scoring = source_rows(rows)
        with self.assertRaisesRegex(ValueError, "dates, targets or scoring labels differ"):
            validate_reference_schedule(features, [], rows)
        with self.assertRaisesRegex(ValueError, "dates, targets or scoring labels differ"):
            validate_reference_schedule(features[:-1], scoring, rows)

    def test_brent_ineligible_reference_date_fails_instead_of_dropping_it(self):
        rows = reference_rows()
        features, scoring = source_rows(rows)
        features[0]["eligibility"]["wti_brent"] = False
        with self.assertRaisesRegex(ValueError, "no eligible Brent inputs"):
            validate_reference_schedule(features, scoring, rows)


class ReferenceReproductionTests(unittest.TestCase):
    def comparison(self):
        rows = reference_rows()
        comparison = deepcopy(rows)
        for row in comparison:
            row["models"]["logistic_wti_brent"] = {"prediction": 0, "probability_up": 0.4}
        return comparison, rows

    def test_identical_original_models_and_training_are_reported_exactly(self):
        comparison, rows = self.comparison()
        self.assertEqual(verify_reference_reproduction(comparison, rows), {
            "forecast_cohort_identical": True,
            "identical_wti_training_input_count": 2,
            "identical_reference_model_output_count": 2,
            "all_wti_training_inputs_identical": True,
            "all_reference_model_outputs_identical": True,
        })

    def test_unchanged_training_with_changed_any_original_model_fails(self):
        for model in MODEL_NAMES:
            comparison, rows = self.comparison()
            comparison[0]["models"][model]["prediction"] ^= 1
            with self.subTest(model=model), self.assertRaisesRegex(ValueError, "despite identical historical training inputs"):
                verify_reference_reproduction(comparison, rows)
        comparison, rows = self.comparison()
        comparison[0]["models"]["logistic"]["diagnostics"]["iterations"] += 1
        with self.assertRaisesRegex(ValueError, "despite identical historical training inputs"):
            verify_reference_reproduction(comparison, rows)

    def test_changed_training_hash_cannot_claim_identical_fit_inputs(self):
        comparison, rows = self.comparison()
        comparison[0]["training"]["input_sha256"] = "f" * 64
        result = verify_reference_reproduction(comparison, rows)
        self.assertEqual(result["identical_wti_training_input_count"], 1)
        self.assertFalse(result["all_wti_training_inputs_identical"])
        self.assertTrue(result["all_reference_model_outputs_identical"])
        comparison[0]["models"]["logistic"]["probability_up"] = 0.7
        result = verify_reference_reproduction(comparison, rows)
        self.assertEqual(result["identical_reference_model_output_count"], 1)
        self.assertFalse(result["all_reference_model_outputs_identical"])

    def test_changed_cohort_fails_before_reproduction_claim(self):
        comparison, rows = self.comparison()
        for changed in (comparison[:-1], list(reversed(comparison))):
            with self.assertRaisesRegex(ValueError, "cohort differs"):
                verify_reference_reproduction(changed, rows)
        comparison[0]["label"] = 0
        with self.assertRaisesRegex(ValueError, "cohort differs"):
            verify_reference_reproduction(comparison, rows)


if __name__ == "__main__":
    unittest.main()
