"""Availability-delay orchestration with synthetic source and replay doubles."""

from contextlib import contextmanager, ExitStack
from copy import deepcopy
from datetime import date
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from oil_audit.dataset import jsonl_bytes, sha256
from oil_eval.delay_runner import run
from oil_eval.inventory_features import INVENTORY_FEATURE_COLUMNS
from tests.test_eval_replay import feature_row


@contextmanager
def delay_case(root):
    rows = [feature_row(date(2011, 4, 7)), feature_row(date(2017, 1, 5)), feature_row(date(2022, 12, 29))]
    admitted = rows[:2]
    augmented = deepcopy(admitted)
    for row in augmented:
        row["features"].update(dict.fromkeys(INVENTORY_FEATURE_COLUMNS))
        row["inventory_provenance"] = {"status": "synthetic_missing"}
    delayed = deepcopy(augmented)
    for row in delayed:
        row["availability_delay_provenance"] = {"synthetic_delay": True}
    source = {"dataset_id": "synthetic-dataset", "source_manifest_sha256": "a" * 64}
    wti = object()
    reference = {"run_id": "synthetic-wti", "evaluation_directory": str(root / "wti")}
    audit = {"audit_id": "synthetic-audit", "audit_directory": str(root / "audit"), "reference_run": reference}
    inventory = {"run_id": "synthetic-inventory", "evaluation_directory": str(root / "inventory")}
    regularization = {"run_id": "synthetic-regularization", "evaluation_directory": str(root / "regularization")}
    reconstruction = {"row_count": 2, "all_wti_features_and_eligibility_identical": True,
                      "original_wti_projection_sha256": "b" * 64, "reconstructed_wti_projection_sha256": "b" * 64}
    case = {
        "rows": rows, "admitted": admitted, "augmented": augmented, "delayed": delayed,
        "source": source, "wti": wti, "source_metadata": {"synthetic_bounded_source": True},
        "reference": reference, "audit": audit, "inventory": inventory, "regularization": regularization,
        "reference_rows": [{"synthetic_wti_reference": True}],
        "inventory_rows": [{"synthetic_inventory_reference": True}],
        "regularization_rows": [{"synthetic_regularization_reference": True}],
        "issues": [{"synthetic_issue": True}], "records": [{"synthetic_record": True}],
        "events": [{"synthetic_event": True}], "labels": [{"synthetic_label": True}],
        "reconstruction": reconstruction,
        "native": {"scipy": {"synthetic.so": "c" * 64}},
        "predictions": [{"synthetic_prediction": True}],
        "inventory_summary": {"row_count": 2}, "delay_summary": {"row_count": 2, "extra_business_days": 1},
        "summary": {"holdout_evaluated": False, "prediction_count": 1, "scored_count": 1,
                    "unscored_count": 0, "metrics": {"synthetic": {"count": 1}}},
    }
    values = {
        "load_development_data": (source, rows, case["events"], case["labels"]),
        "load_delay_wti_source": (case["source_metadata"], wti),
        "reconstruct_baseline_wti_rows": (admitted, reconstruction),
        "load_reference": (reference, case["reference_rows"]),
        "load_inventory_reference": (audit, case["issues"], case["records"]),
        "attach_inventory_features": (augmented, case["inventory_summary"]),
        "load_inventory_comparison_reference": (inventory, case["inventory_rows"]),
        "load_inventory_regularization_reference": (regularization, case["regularization_rows"]),
        "build_delayed_feature_rows": (delayed, case["delay_summary"]),
        "replay_delayed_development": (case["predictions"], case["summary"]),
        "_native_runtime_sha256": case["native"],
    }
    case["order"], case["mocks"] = [], {}
    with ExitStack() as stack:
        for name, result in values.items():
            def invoke(*args, _name=name, _result=result, **kwargs):
                case["order"].append(_name)
                return _result
            case["mocks"][name] = stack.enter_context(patch("oil_eval.delay_runner." + name, side_effect=invoke))
        yield case


def execute(root, *, report=None, output_root=None):
    return run(root / "dataset", root / "vintages", output_root or root / "outputs",
               report or root / "reports/delay.json", reference_report=root / "reports/wti.json",
               inventory_audit_report=root / "reports/audit.json",
               inventory_reference_report=root / "reports/inventory.json",
               regularization_reference_report=root / "reports/regularization.json")


class AvailabilityDelayRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".test-delay-runner-", dir=".")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).relative_to(Path.cwd())

    def test_reconstructs_baseline_before_authenticated_references_and_persists_closed_payloads(self):
        with delay_case(self.root) as case:
            original = deepcopy(case["rows"])
            result = execute(self.root)
            mocks = case["mocks"]
            mocks["load_delay_wti_source"].assert_called_once_with(self.root / "vintages", "a" * 64)
            mocks["reconstruct_baseline_wti_rows"].assert_called_once_with(case["admitted"], case["wti"])
            self.assertLess(case["order"].index("reconstruct_baseline_wti_rows"), case["order"].index("load_reference"))
            for name in ("load_reference", "load_inventory_comparison_reference", "load_inventory_regularization_reference"):
                self.assertLess(case["order"].index(name), case["order"].index("replay_delayed_development"))
                self.assertEqual(mocks[name].call_args.kwargs, {"native_runtime": case["native"]})
            mocks["attach_inventory_features"].assert_called_once_with(case["admitted"], case["issues"], case["records"])
            mocks["build_delayed_feature_rows"].assert_called_once_with(case["augmented"], case["wti"], case["issues"], case["records"])
            mocks["replay_delayed_development"].assert_called_once_with(
                case["augmented"], case["delayed"], case["events"], case["labels"],
                reference_rows=case["reference_rows"], inventory_reference_rows=case["inventory_rows"],
                regularization_reference_rows=case["regularization_rows"])
            frozen_payload = jsonl_bytes([{
                **{key: row[key] for key in ("row_id", "forecast_date", "target_date")},
                "features": {column: row["features"][column] for column in INVENTORY_FEATURE_COLUMNS},
                "provenance": row["inventory_provenance"],
            } for row in case["augmented"]])
            feature_identity = {"sha256": sha256(frozen_payload), "bytes": len(frozen_payload)}
            self.assertEqual(mocks["load_inventory_comparison_reference"].call_args.args[3:],
                             (case["reference"], case["audit"], feature_identity))
            self.assertEqual(mocks["load_inventory_regularization_reference"].call_args.args[3:],
                             (case["reference"], case["audit"], feature_identity, case["inventory"]))
            directory = Path(result["evaluation_directory"])
            manifest = json.loads((directory / "manifest.json").read_bytes())
            self.assertEqual(set(manifest["artifacts"]), {"predictions.jsonl", "summary.json", "inventory_features.jsonl", "delayed_features.jsonl"})
            for name, identity in manifest["artifacts"].items():
                payload = (directory / name).read_bytes()
                self.assertEqual(identity, {"sha256": sha256(payload), "bytes": len(payload)})
            self.assertEqual((directory / "inventory_features.jsonl").read_bytes(), frozen_payload)
            self.assertEqual((directory / "delayed_features.jsonl").read_bytes(), jsonl_bytes(case["delayed"]))
            self.assertEqual(manifest["baseline_wti_reconstruction"], case["reconstruction"])
            self.assertEqual(manifest["bounded_wti_source"], case["source_metadata"])
            self.assertEqual(manifest["native_runtime_sha256"], case["native"])
            self.assertEqual(result, json.loads((self.root / "reports/delay.json").read_bytes()))
            self.assertFalse(result["holdout_evaluated"])
            self.assertFalse(json.loads((directory / "summary.json").read_bytes())["holdout_evaluated"])
            self.assertEqual(result["regularization_reference_run_id"], case["regularization"]["run_id"])
            self.assertEqual(case["rows"], original)

    def test_source_reconstruction_reference_and_feature_failures_stop_before_any_fit_or_output(self):
        for name in ("load_delay_wti_source", "reconstruct_baseline_wti_rows", "load_reference",
                     "load_inventory_reference", "load_inventory_comparison_reference",
                     "load_inventory_regularization_reference", "build_delayed_feature_rows"):
            with self.subTest(stage=name), delay_case(self.root) as case:
                case["mocks"][name].side_effect = ValueError("Synthetic authentication mismatch")
                with self.assertRaisesRegex(ValueError, "authentication mismatch"):
                    execute(self.root)
                case["mocks"]["replay_delayed_development"].assert_not_called()
                self.assertFalse((self.root / "outputs").exists())
                self.assertFalse((self.root / "reports/delay.json").exists())
                if name in ("load_delay_wti_source", "reconstruct_baseline_wti_rows"):
                    case["mocks"]["load_reference"].assert_not_called()

    def test_audit_wti_identity_mismatch_stops_before_inventory_build_and_fits(self):
        with delay_case(self.root) as case:
            case["audit"]["reference_run"] = {"run_id": "different-wti"}
            with self.assertRaisesRegex(ValueError, "share their authenticated WTI reference"):
                execute(self.root)
            for name in ("attach_inventory_features", "load_inventory_comparison_reference",
                         "load_inventory_regularization_reference", "build_delayed_feature_rows",
                         "replay_delayed_development"):
                case["mocks"][name].assert_not_called()
            self.assertFalse((self.root / "outputs").exists())

    def test_existing_reports_and_manifest_ancestors_are_preserved_before_source_load(self):
        retained = self.root / "reports/retained.json"
        retained.parent.mkdir()
        retained.write_bytes(b"retained report")
        old = self.root / "old-run"
        old.mkdir()
        (old / "manifest.json").write_bytes(b"retained manifest")
        variants = [(retained, None), (self.root / "outputs/new-report.json", None),
                    (old / "summary.json", None), (old / "nested/report.json", None),
                    (self.root / "reports/new-report.json", old / "nested")]
        for report, output in variants:
            with self.subTest(report=report, output=output), delay_case(self.root) as case:
                with self.assertRaises(ValueError):
                    execute(self.root, report=report, output_root=output)
                case["mocks"]["load_development_data"].assert_not_called()
                case["mocks"]["replay_delayed_development"].assert_not_called()
        self.assertEqual(retained.read_bytes(), b"retained report")
        self.assertEqual((old / "manifest.json").read_bytes(), b"retained manifest")

    def test_new_report_cannot_be_inserted_into_a_different_historical_artifact_root(self):
        for report in (Path("data/evaluations/delay-guard-regression.json"),
                       Path("data/audits/inventories/delay-guard-regression.json"),
                       Path("data/datasets/delay-guard-regression.json")):
            with self.subTest(report=report), delay_case(self.root) as case:
                # A missing protection must never write outside the temporary fixture.
                case["mocks"]["replay_delayed_development"].side_effect = AssertionError("Protected report reached fitting")
                with self.assertRaisesRegex(ValueError, "immutable|historical|artifact"):
                    execute(self.root, report=report)
                case["mocks"]["replay_delayed_development"].assert_not_called()

    def test_identical_artifacts_are_reused_and_corruption_never_restores_earlier_missing_files(self):
        with delay_case(self.root):
            first = execute(self.root)
            directory = Path(first["evaluation_directory"])
            before = {path.name: path.read_bytes() for path in directory.iterdir()}
            second = execute(self.root, report=self.root / "reports/second.json")
            self.assertEqual(second["run_id"], first["run_id"])
            self.assertEqual({path.name: path.read_bytes() for path in directory.iterdir()}, before)
            original_report = (self.root / "reports/delay.json").read_bytes()
            (directory / "predictions.jsonl").unlink()
            (directory / "delayed_features.jsonl").write_bytes(b"corrupt retained artifact")
            partial = {path.name: path.read_bytes() for path in directory.iterdir()}
            with self.assertRaisesRegex(ValueError, "Refusing to replace"):
                execute(self.root, report=self.root / "reports/rejected.json")
            self.assertEqual({path.name: path.read_bytes() for path in directory.iterdir()}, partial)
            self.assertEqual((self.root / "reports/delay.json").read_bytes(), original_report)
            self.assertFalse((self.root / "reports/rejected.json").exists())


if __name__ == "__main__":
    unittest.main()
