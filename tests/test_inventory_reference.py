"""Synthetic authentication of saved inventory comparison references."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from oil_audit.dataset import json_bytes, sha256
from oil_eval.data import _decode
from oil_eval.inventory_features import INVENTORY_FEATURE_COLUMNS
from oil_eval.protocol import protocol_config
from tests.test_eval_reference import reference_rows


def inventory_reference_rows():
    rows = reference_rows()
    for index, row in enumerate(rows):
        row["training"]["inventory_input_sha256"] = str(index + 2) * 64
        row["models"]["logistic_wti_inventory"] = {
            "prediction": 0, "probability_up": 0.4, "diagnostics": {"iterations": 1}}
        row["inventory"] = {
            "features": dict(zip(INVENTORY_FEATURE_COLUMNS, [1, None, 2, 3, 2])),
            "provenance": {"status": "available", "information_cutoff_date": "2017-01-04" if index == 0 else "2017-01-18"},
        }
    return rows


def frozen_rows():
    rows = [{"row_id": "wti-direction-20110407", "forecast_date": "2011-04-07", "target_date": "2011-04-14",
             "features": dict.fromkeys(INVENTORY_FEATURE_COLUMNS), "provenance": {"status": "no_earlier_release"}}]
    for row in inventory_reference_rows():
        rows.append({**{key: row[key] for key in ("row_id", "forecast_date", "target_date")}, **row["inventory"]})
    return rows


class InventoryComparisonReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".test-inventory-reference-", dir=".")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).relative_to(Path.cwd())
        self.directory = self.root / "reference"
        self.directory.mkdir()
        self.report_path = self.root / "inventory.json"
        self.source = {"dataset_id": "synthetic", "manifest_sha256": "a" * 64}
        self.runtime = {"python": "synthetic", "scikit-learn": "synthetic"}
        self.wti_reference = {"run_id": "wti-original", "evaluation_directory": str(self.root / "wti"),
                              "report_sha256": "b" * 64}
        self.inventory_reference = {"audit_id": "audit-original", "audit_directory": str(self.root / "audit"),
                                    "reference_run": self.wti_reference, "manifest_sha256": "c" * 64}

    def save(self, *, rows=None, feature_rows=None, identity_changes=None, summary_changes=None,
             experiment="inventory-comparison"):
        rows = inventory_reference_rows() if rows is None else rows
        feature_rows = frozen_rows() if feature_rows is None else feature_rows
        summary = {"protocol": protocol_config(experiment), "holdout_evaluated": False,
                   "prediction_count": len(rows),
                   "scored_count": sum(isinstance(row, dict) and row["scoring_status"] == "available" for row in rows)}
        summary.update(summary_changes or {})
        serialize = lambda values: b"".join((value if isinstance(value, bytes) else json.dumps(value).encode()) + b"\n" for value in values)
        payloads = {"summary.json": json_bytes(summary), "predictions.jsonl": serialize(rows),
                    "inventory_features.jsonl": serialize(feature_rows)}
        self.inventory_features = {"sha256": sha256(payloads["inventory_features.jsonl"]),
                                   "bytes": len(payloads["inventory_features.jsonl"])}
        identity = {"source": self.source, "protocol": protocol_config(experiment),
                    "runtime_versions": self.runtime, "numeric_threads": 1,
                    "source_code_sha256": {"oil_eval/models.py": "d" * 64}, "requirements_sha256": "e" * 64,
                    "reference": self.wti_reference, "inventory_reference": self.inventory_reference,
                    "inventory_features": self.inventory_features}
        identity.update(identity_changes or {})
        manifest = {**identity, "run_id": sha256(json_bytes(identity))[:20],
                    "artifacts": {name: {"sha256": sha256(payload), "bytes": len(payload)}
                                  for name, payload in payloads.items()}}
        for name, payload in payloads.items():
            (self.directory / name).write_bytes(payload)
        (self.directory / "manifest.json").write_bytes(json_bytes(manifest))
        report = {"run_id": manifest["run_id"], "evaluation_directory": str(self.directory),
                  "dataset_id": self.source["dataset_id"], "runtime_versions": self.runtime,
                  "reference_run_id": self.wti_reference["run_id"],
                  "inventory_audit_id": self.inventory_reference["audit_id"], **summary}
        if experiment == "inventory-regularization":
            report["inventory_reference_run_id"] = identity["inventory_comparison_reference"]["run_id"]
        self.report_path.write_bytes(json_bytes(report))
        return manifest, report

    def load(self, **options):
        from oil_eval.reference import load_inventory_comparison_reference
        return load_inventory_comparison_reference(self.report_path, self.source, self.runtime,
                                                   self.wti_reference, self.inventory_reference, self.inventory_features,
                                                   **options)

    def test_strict_inventory_reference_native_fingerprint_must_be_present_and_equal(self):
        expected = {"scipy": {"scipy/example.so": "a" * 64}}
        other = {"scipy": {"scipy/example.so": "b" * 64}}
        for recorded in (None, other, expected):
            self.save(identity_changes={} if recorded is None else {"native_runtime_sha256": recorded})
            with self.subTest(recorded=recorded):
                if recorded == expected:
                    _, rows = self.load(native_runtime=expected)
                    self.assertEqual(rows, inventory_reference_rows())
                else:
                    with self.assertRaisesRegex(ValueError, "native runtime"):
                        self.load(native_runtime=expected)
                # Explicitly preserve inspection of historical references without a native gate.
                self.assertEqual(self.load()[1], inventory_reference_rows())

    def test_strict_wti_reference_native_fingerprint_must_be_present_and_equal(self):
        from oil_eval.reference import load_reference
        from tests.test_eval_reference import ReferenceAuthenticationTests
        fixture = ReferenceAuthenticationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        expected = {"scipy": {"scipy/example.so": "a" * 64}}
        other = {"scipy": {"scipy/example.so": "b" * 64}}
        for recorded in (None, other, expected):
            fixture.save(identity_changes={} if recorded is None else {"native_runtime_sha256": recorded})
            with self.subTest(recorded=recorded):
                if recorded == expected:
                    _, rows = load_reference(fixture.report_path, fixture.source, fixture.runtime, native_runtime=expected)
                    self.assertEqual(rows, reference_rows())
                else:
                    with self.assertRaisesRegex(ValueError, "native runtime"):
                        load_reference(fixture.report_path, fixture.source, fixture.runtime, native_runtime=expected)
                self.assertEqual(fixture.load()[1], reference_rows())

    def test_authenticates_report_manifest_all_artifacts_and_frozen_inputs(self):
        manifest, _ = self.save()
        with patch("oil_eval.data.load_development_data", side_effect=AssertionError("raw price dataset read")), \
             patch("oil_eval.models.Pipeline.fit", side_effect=AssertionError("model fit")):
            reference, rows = self.load()
        self.assertEqual(rows, inventory_reference_rows())
        self.assertEqual(reference["run_id"], manifest["run_id"])
        self.assertEqual(reference["manifest_sha256"], sha256((self.directory / "manifest.json").read_bytes()))
        self.assertEqual(reference["report_sha256"], sha256(self.report_path.read_bytes()))
        self.assertEqual(reference["artifacts"], manifest["artifacts"])

    def test_changed_source_runtime_protocol_threads_and_reference_identities_fail(self):
        changes = [
            {"source": {**self.source, "dataset_id": "changed"}},
            {"runtime_versions": {**self.runtime, "python": "changed"}},
            {"protocol": {**protocol_config("inventory-comparison"), "minimum_training_rows": 1}},
            {"numeric_threads": 2},
            {"reference": {**self.wti_reference, "report_sha256": "f" * 64}},
            {"inventory_reference": {**self.inventory_reference, "manifest_sha256": "f" * 64}},
        ]
        for change in changes:
            with self.subTest(change=change):
                self.save(identity_changes=change)
                with self.assertRaises(ValueError):
                    self.load()

    def test_changed_historical_code_hash_without_reanchoring_fails(self):
        manifest, _ = self.save()
        manifest["source_code_sha256"]["oil_eval/models.py"] = "f" * 64
        (self.directory / "manifest.json").write_bytes(json_bytes(manifest))
        with self.assertRaisesRegex(ValueError, "identity"):
            self.load()

    def test_changed_frozen_feature_bytes_or_size_fail_even_if_other_inputs_match(self):
        self.save()
        for field, value in (("sha256", "f" * 64), ("bytes", self.inventory_features["bytes"] + 1)):
            original = self.inventory_features
            self.inventory_features = {**original, field: value}
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.load()
            self.inventory_features = original

    def test_each_corrupt_artifact_fails_before_json_decoding(self):
        for name in ("predictions.jsonl", "summary.json", "inventory_features.jsonl"):
            self.save()
            (self.directory / name).write_bytes(b"broken JSON")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "checksum"):
                self.load()

    def test_frozen_artifact_metadata_must_match_the_manifest_feature_identity(self):
        manifest, _ = self.save()
        manifest["artifacts"]["inventory_features.jsonl"]["bytes"] += 1
        (self.directory / "manifest.json").write_bytes(json_bytes(manifest))
        with self.assertRaisesRegex(ValueError, "feature artifact"):
            self.load()

    def test_prediction_inventory_values_and_provenance_must_match_frozen_artifact(self):
        for field in ("features", "provenance"):
            rows = inventory_reference_rows()
            if field == "features":
                rows[-1]["inventory"][field][INVENTORY_FEATURE_COLUMNS[0]] = 999
            else:
                rows[-1]["inventory"][field]["status"] = "changed"
            self.save(rows=rows)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "frozen features"):
                self.load()

    def test_unexpected_artifacts_fail_without_reading_the_unexpected_path(self):
        manifest, _ = self.save()
        manifest["artifacts"]["../outside.json"] = {"sha256": "f" * 64, "bytes": 0}
        (self.directory / "manifest.json").write_bytes(json_bytes(manifest))
        with self.assertRaises(ValueError):
            self.load()

    def test_changed_report_or_open_holdout_summary_fails(self):
        _, report = self.save()
        report["scored_count"] += 1
        self.report_path.write_bytes(json_bytes(report))
        with self.assertRaises(ValueError):
            self.load()
        self.save(summary_changes={"holdout_evaluated": True})
        with self.assertRaises(ValueError):
            self.load()

    def test_holdout_poison_in_predictions_or_features_is_never_decoded(self):
        poison = b'{"forecast_date":"2023-01-03", heldout_poison_cannot_decode}'
        for changed in ({"rows": [inventory_reference_rows()[0], poison]},
                        {"feature_rows": frozen_rows() + [poison]}):
            self.save(**changed)

            def guarded_decode(payload):
                self.assertNotIn(b"heldout_poison", payload)
                return _decode(payload)

            with self.subTest(artifact=next(iter(changed))), patch("oil_eval.reference._decode", side_effect=guarded_decode):
                with self.assertRaises(ValueError):
                    self.load()

    def test_future_label_availability_poison_is_rejected_before_decoding(self):
        poison = (b'{"forecast_date":"2017-01-19", "label_available_from":"2023-01-03", '
                  b'future_outcome_poison_cannot_decode}')
        self.save(rows=[inventory_reference_rows()[0], poison])

        def guarded_decode(payload):
            self.assertNotIn(b"future_outcome_poison", payload)
            return _decode(payload)

        with patch("oil_eval.reference._decode", side_effect=guarded_decode):
            with self.assertRaisesRegex(ValueError, "holdout outcome availability"):
                self.load()

    def test_available_scoring_state_requires_binary_integer_and_mature_outcome(self):
        changes = [{"label": label} for label in (None, True, 1.0, 2, "1")]
        changes += [{"label_available_from": value} for value in
                    (None, "2017-01-12", "2017-01-05", "2023-01-03")]
        for change in changes:
            rows = inventory_reference_rows()
            rows[0].update(change)
            self.save(rows=rows)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.load()

    def test_unscored_state_keeps_both_outcome_fields_null_and_status_known(self):
        for change in ({"label": 0}, {"label_available_from": "2017-01-27"},
                       {"scoring_status": "unknown"}, {"scoring_status": "available"}):
            rows = inventory_reference_rows()
            rows[-1].update(change)
            self.save(rows=rows)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.load()

    def test_initial_forecasts_boundary_targets_and_wrong_model_family_fail(self):
        for mutation in (
            lambda row: row.update(forecast_date="2016-12-22", target_date="2016-12-30"),
            lambda row: row.update(forecast_date="2022-12-29", target_date="2023-01-06"),
            lambda row: row["models"].pop("logistic_wti_inventory"),
            lambda row: row["models"].update(logistic_wti_inventory_c01={"prediction": 1}),
        ):
            rows = inventory_reference_rows()
            mutation(rows[0])
            self.save(rows=rows)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.load()

    def test_feature_dates_order_duplicates_and_missing_columns_fail(self):
        for mutation in (
            lambda rows: rows[0].update(forecast_date="2022-12-29", target_date="2023-01-06"),
            lambda rows: rows.reverse(),
            lambda rows: rows.append(deepcopy(rows[-1])),
            lambda rows: rows[-1]["features"].pop(INVENTORY_FEATURE_COLUMNS[-1]),
        ):
            features = frozen_rows()
            mutation(features)
            self.save(feature_rows=features)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.load()

    def test_prediction_counts_order_and_duplicate_ids_fail(self):
        duplicate = inventory_reference_rows()
        duplicate[-1]["row_id"] = duplicate[0]["row_id"]
        for rows, change in ((inventory_reference_rows(), {"prediction_count": 3}),
                             (inventory_reference_rows(), {"scored_count": 2}),
                             (list(reversed(inventory_reference_rows())), {}), (duplicate, {})):
            self.save(rows=rows, summary_changes=change)
            with self.subTest(change=change, first=rows[0]["row_id"]), self.assertRaises(ValueError):
                self.load()


def regularization_reference_rows():
    rows = inventory_reference_rows()
    for row in rows:
        row["models"]["logistic_wti_inventory_c01"] = {
            "prediction": 1, "probability_up": 0.55, "diagnostics": {"iterations": 2}}
    return rows


class InventoryRegularizationReferenceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = InventoryComparisonReferenceTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.comparison_reference = {"run_id": "saved-c1", "evaluation_directory": str(self.fixture.root / "c1"),
                                     "manifest_sha256": "f" * 64}
        self.native = {"scipy": {"scipy/example.so": "a" * 64}}

    def save(self, *, rows=None, identity_changes=None, **options):
        identity = {"inventory_comparison_reference": self.comparison_reference,
                    "native_runtime_sha256": self.native}
        identity.update(identity_changes or {})
        return self.fixture.save(rows=regularization_reference_rows() if rows is None else rows,
                                 identity_changes=identity, experiment="inventory-regularization", **options)

    def load(self):
        from oil_eval.reference import load_inventory_regularization_reference
        fixture = self.fixture
        return load_inventory_regularization_reference(
            fixture.report_path, fixture.source, fixture.runtime, fixture.wti_reference,
            fixture.inventory_reference, fixture.inventory_features, self.comparison_reference,
            native_runtime=self.native)

    def test_authenticates_five_models_and_three_artifacts_without_fitting(self):
        manifest, _ = self.save()
        with patch("oil_eval.models.Pipeline.fit", side_effect=AssertionError("model fit")), \
             patch("oil_eval.data.load_development_data", side_effect=AssertionError("raw dataset read")):
            reference, rows = self.load()
        self.assertEqual(rows, regularization_reference_rows())
        self.assertTrue(all(len(row["models"]) == 5 for row in rows))
        self.assertEqual(reference["artifacts"], manifest["artifacts"])
        self.assertEqual(len(reference["artifacts"]), 3)
        self.assertEqual(reference["manifest_sha256"], sha256((self.fixture.directory / "manifest.json").read_bytes()))

    def test_reference_chain_runtime_and_native_fingerprints_must_match(self):
        fixture = self.fixture
        changes = (
            {"source": {**fixture.source, "dataset_id": "different"}},
            {"runtime_versions": {**fixture.runtime, "python": "different"}},
            {"reference": {**fixture.wti_reference, "run_id": "different"}},
            {"inventory_reference": {**fixture.inventory_reference, "audit_id": "different"}},
            {"inventory_comparison_reference": {**self.comparison_reference, "manifest_sha256": "0" * 64}},
            {"native_runtime_sha256": None},
            {"native_runtime_sha256": {"scipy": {"scipy/example.so": "b" * 64}}},
        )
        for change in changes:
            self.save(identity_changes=change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.load()

    def test_all_artifact_corruption_and_changed_frozen_input_identity_fail(self):
        for name in ("predictions.jsonl", "summary.json", "inventory_features.jsonl"):
            self.save()
            (self.fixture.directory / name).write_bytes(b"corrupt")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "checksum"):
                self.load()
        self.save()
        self.fixture.inventory_features = {**self.fixture.inventory_features, "sha256": "b" * 64}
        with self.assertRaises(ValueError):
            self.load()

    def test_holdout_features_predictions_and_future_outcomes_are_never_decoded(self):
        holdout = b'{"forecast_date":"2023-01-03", forbidden_poison_cannot_decode}'
        outcome = b'{"forecast_date":"2017-01-19", "label_available_from":"2023-01-03", forbidden_poison_cannot_decode}'
        for options in ({"rows": [regularization_reference_rows()[0], holdout]},
                        {"feature_rows": frozen_rows() + [holdout]},
                        {"rows": [regularization_reference_rows()[0], outcome]}):
            self.save(**options)

            def guarded_decode(payload):
                self.assertNotIn(b"forbidden_poison", payload)
                return _decode(payload)

            with self.subTest(options=list(options)), patch("oil_eval.reference._decode", side_effect=guarded_decode):
                with self.assertRaises(ValueError):
                    self.load()

    def test_wrong_model_family_or_boundary_forecast_cannot_be_used_as_control(self):
        for mutate in (
            lambda row: row["models"].pop("logistic_wti_inventory_c01"),
            lambda row: row["models"].update(unexpected_rolling_model={"prediction": 1}),
            lambda row: row.update(forecast_date="2022-12-29", target_date="2023-01-06"),
        ):
            rows = regularization_reference_rows()
            mutate(rows[-1])
            self.save(rows=rows)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                self.load()

    def test_saved_prediction_inventory_and_scoring_state_remain_authenticated(self):
        for mutate in (
            lambda row: row["inventory"]["features"].update({INVENTORY_FEATURE_COLUMNS[0]: 999}),
            lambda row: row["inventory"]["provenance"].update(status="different"),
            lambda row: row.update(label=1),
            lambda row: row.update(label_available_from="2017-01-27"),
        ):
            rows = regularization_reference_rows()
            mutate(rows[-1])
            self.save(rows=rows)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                self.load()

    def test_report_changes_and_undeclared_artifacts_fail(self):
        manifest, report = self.save()
        report["inventory_reference_run_id"] = "different-c1"
        self.fixture.report_path.write_bytes(json_bytes(report))
        with self.assertRaises(ValueError):
            self.load()
        manifest, _ = self.save()
        manifest["artifacts"]["../outside.json"] = {"sha256": "a" * 64, "bytes": 0}
        (self.fixture.directory / "manifest.json").write_bytes(json_bytes(manifest))
        with self.assertRaisesRegex(ValueError, "Unexpected reference artifacts"):
            self.load()


if __name__ == "__main__":
    unittest.main()
