import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from oil_eval.runner import run


class RunnerTests(unittest.TestCase):
    def test_inventory_runner_freezes_only_admitted_rows_and_saves_null_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = {"run_id": "original-wti", "evaluation_directory": "data/synthetic-reference"}
            inventory_reference = {"audit_id": "inventory-source", "audit_directory": "data/synthetic-audit",
                                   "reference_run": reference}
            admitted = {"row_id": "one", "forecast_date": "2017-01-05", "target_date": "2017-01-12"}
            crossing = {"row_id": "crossing", "forecast_date": "2022-12-29", "target_date": "2023-01-06"}
            from oil_eval.inventory_features import INVENTORY_FEATURE_COLUMNS
            augmented = {**admitted, "features": dict.fromkeys(INVENTORY_FEATURE_COLUMNS),
                         "inventory_provenance": {"status": "no_earlier_release"}}
            with patch("oil_eval.runner.load_development_data", return_value=({"dataset_id": "synthetic"}, [admitted, crossing], [], [])) as loader, \
                 patch("oil_eval.runner.load_reference", return_value=(reference, [])), \
                 patch("oil_eval.runner.load_inventory_reference", return_value=(inventory_reference, [], [])), \
                 patch("oil_eval.runner.attach_inventory_features", return_value=([augmented], {"row_count": 1})) as attach, \
                 patch("oil_eval.runner.replay_development", return_value=([], {"holdout_evaluated": False})) as replay:
                result = run(root, root / "runs", root / "inventory.json", experiment="inventory-comparison")
                loader.assert_called_once_with(root)
                attach.assert_called_once_with([admitted], [], [])
                replay.assert_called_once_with([augmented, crossing], [], [],
                                               experiment="inventory-comparison", reference_rows=[])
                self.assertEqual(result["inventory_audit_id"], inventory_reference["audit_id"])
                destination = Path(result["evaluation_directory"])
                saved = json.loads((destination / "inventory_features.jsonl").read_text())
                self.assertEqual(saved["features"], dict.fromkeys(INVENTORY_FEATURE_COLUMNS))
                manifest = json.loads((destination / "manifest.json").read_text())
                self.assertEqual(manifest["inventory_reference"], inventory_reference)
                self.assertIn("inventory_features.jsonl", manifest["artifacts"])

    def test_inventory_reference_mismatch_stops_before_feature_build_and_replay(self):
        with patch("oil_eval.runner.load_development_data", return_value=({}, [], [], [])), \
             patch("oil_eval.runner.load_reference", return_value=({"run_id": "original", "evaluation_directory": "data/reference"}, [])), \
             patch("oil_eval.runner.load_inventory_reference", return_value=({"reference_run": {"run_id": "other"}}, [], [])), \
             patch("oil_eval.runner.attach_inventory_features") as attach, \
             patch("oil_eval.runner.replay_development") as replay:
            with self.assertRaisesRegex(ValueError, "do not share"):
                run(Path("data/synthetic"), experiment="inventory-comparison")
            attach.assert_not_called()
            replay.assert_not_called()

    def test_inventory_report_cannot_overwrite_historical_or_old_immutable_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = {"run_id": "original-wti", "evaluation_directory": "data/reference"}
            inventory_reference = {"audit_id": "inventory", "audit_directory": "data/inventory-audit",
                                   "reference_run": reference}
            destinations = [Path("reports/brent_comparison.json"), Path("reports/development_benchmarks.json"),
                            Path("reports/inventory_audit.json"), Path("data/inventory-audit/records.jsonl"),
                            root / "runs" / "old-run" / "inventory_features.jsonl"]
            with patch("oil_eval.runner.load_development_data", return_value=({}, [], [], [])), \
                 patch("oil_eval.runner.load_reference", return_value=(reference, [])), \
                 patch("oil_eval.runner.load_inventory_reference", return_value=(inventory_reference, [], [])), \
                 patch("oil_eval.runner.attach_inventory_features", return_value=([], {})), \
                 patch("oil_eval.runner.replay_development") as replay:
                for report in destinations:
                    with self.subTest(report=str(report)), self.assertRaisesRegex(ValueError, "overwrite"):
                        run(root, root / "runs", report, experiment="inventory-comparison")
                replay.assert_not_called()

    def test_comparison_records_reference_and_routes_only_combined_development_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = {"run_id": "original-wti", "evaluation_directory": "data/synthetic-reference",
                         "report_sha256": "b" * 64}
            source = {"dataset_id": "synthetic"}
            summary = {"holdout_evaluated": False}
            with patch("oil_eval.runner.load_development_data", return_value=(source, [], [], [])) as loader, \
                 patch("oil_eval.runner.load_reference", return_value=(reference, [{"reference": True}])) as reference_loader, \
                 patch("oil_eval.runner.replay_development", return_value=([], summary)) as replay:
                result = run(root, root / "runs", root / "comparison.json", experiment="brent-comparison")
                loader.assert_called_once_with(root, feature_set="wti_brent")
                reference_loader.assert_called_once()
                replay.assert_called_once_with([], [], [], experiment="brent-comparison",
                                               reference_rows=[{"reference": True}])
                self.assertEqual(result["reference_run_id"], reference["run_id"])
                manifest = json.loads((Path(result["evaluation_directory"]) / "manifest.json").read_text())
                self.assertEqual(manifest["reference"], reference)
                self.assertEqual(manifest["protocol"]["protocol_id"], "wti-brent-development-v1")

    def test_comparison_cannot_overwrite_reference_before_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_report = root / "original.json"
            reference_directory = root / "reference"
            reference = {"run_id": "original-wti", "evaluation_directory": str(reference_directory)}
            destinations = [reference_report, Path("reports/development_evaluation.json")] + [
                reference_directory / name for name in ("manifest.json", "summary.json", "predictions.jsonl")]
            with patch("oil_eval.runner.load_development_data", return_value=({"dataset_id": "synthetic"}, [], [], [])), \
                 patch("oil_eval.runner.load_reference", return_value=(reference, [])), \
                 patch("oil_eval.runner.replay_development") as replay:
                for report in destinations:
                    with self.subTest(report=report.name), self.assertRaisesRegex(ValueError, "cannot overwrite"):
                        run(root, root / "runs", report, experiment="brent-comparison",
                            reference_report=reference_report)
                replay.assert_not_called()

    def test_comparison_authentication_failure_stops_before_replay(self):
        with patch("oil_eval.runner.load_development_data", return_value=({}, [], [], [])), \
             patch("oil_eval.runner.load_reference", side_effect=ValueError("Reference artifact checksum mismatch")), \
             patch("oil_eval.runner.replay_development") as replay:
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                run(Path("data/synthetic"), experiment="brent-comparison")
            replay.assert_not_called()

    def test_identical_run_reuses_artifacts_and_rejects_modified_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = {"dataset_id": "synthetic", "manifest_sha256": "a" * 64}
            summary = {"holdout_evaluated": False, "prediction_count": 1, "scored_count": 0,
                       "unscored_count": 1, "metrics": {}}
            with patch("oil_eval.runner.load_development_data", return_value=(source, [], [], [])), \
                 patch("oil_eval.runner.replay_development", return_value=([{"forecast_date": "2017-01-05"}], summary)):
                first = run(root, root / "runs", root / "report.json")
                second = run(root, root / "runs", root / "report.json")
                self.assertEqual(first, second)
                destination = Path(first["evaluation_directory"])
                manifest = json.loads((destination / "manifest.json").read_text())
                self.assertEqual(manifest["numeric_threads"], 1)
                self.assertEqual(manifest["protocol"]["holdout_start"], "2023-01-01")
                for name in ("manifest.json", "summary.json", "predictions.jsonl"):
                    with self.subTest(name=name), self.assertRaisesRegex(ValueError, "immutable evaluation artifact"):
                        run(root, root / "runs", destination / name)
                (destination / "predictions.jsonl").write_text("modified\n")
                with self.assertRaisesRegex(ValueError, "Refusing to replace"):
                    run(root, root / "runs", root / "report.json")

    def test_cli_has_no_holdout_execution_option(self):
        result = subprocess.run([
            sys.executable, "-m", "oil_eval", "--dataset-dir", "data/unused", "--holdout",
        ], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("unrecognized arguments: --holdout", result.stderr)


if __name__ == "__main__":
    unittest.main()
