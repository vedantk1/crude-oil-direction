from contextlib import contextmanager, ExitStack
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from oil_eval.runner import run


@contextmanager
def regularization_case(root, *, window=False):
    """Stub file-backed dependencies; never read actual sources or fit models."""
    from oil_eval.inventory_features import INVENTORY_FEATURE_COLUMNS
    source = {"dataset_id": "synthetic"}
    wti_reference = {"run_id": "original-wti", "evaluation_directory": str(root / "wti-reference")}
    audit = {"audit_id": "original-audit", "audit_directory": str(root / "audit"), "reference_run": wti_reference}
    inventory_reference = {"run_id": "original-inventory", "evaluation_directory": str(root / "inventory-reference")}
    admitted = {"row_id": "one", "forecast_date": "2017-01-05", "target_date": "2017-01-12"}
    crossing = {"row_id": "crossing", "forecast_date": "2022-12-29", "target_date": "2023-01-06"}
    augmented = {**admitted, "features": dict.fromkeys(INVENTORY_FEATURE_COLUMNS),
                 "inventory_provenance": {"status": "no_earlier_release"}}
    values = {"source": source, "wti_reference": wti_reference, "audit": audit,
              "inventory_reference": inventory_reference, "admitted": admitted, "crossing": crossing,
              "augmented": augmented, "wti_rows": [{"original_wti": True}],
              "inventory_rows": [{"original_inventory": True}],
              "native_runtime": {"scipy": {"scipy/example.so": "a" * 64}}}
    returns = {
        "load_development_data": (source, [admitted, crossing], [], []),
        "load_reference": (wti_reference, values["wti_rows"]),
        "load_inventory_reference": (audit, [], []),
        "load_inventory_comparison_reference": (inventory_reference, values["inventory_rows"]),
        "attach_inventory_features": ([augmented], {"row_count": 1}),
        "replay_development": ([], {"holdout_evaluated": False}),
        "_native_runtime_sha256": values["native_runtime"],
    }
    if window:
        values["regularization_reference"] = {
            "run_id": "original-regularization", "evaluation_directory": str(root / "regularization-reference")}
        values["regularization_rows"] = [{"original_regularization": True}]
        returns["load_inventory_regularization_reference"] = (
            values["regularization_reference"], values["regularization_rows"])
    with ExitStack() as stack:
        values["mocks"] = {name: stack.enter_context(patch("oil_eval.runner." + name, return_value=value))
                           for name, value in returns.items()}
        yield values


class RunnerTests(unittest.TestCase):
    def test_tree_comparison_routes_shared_expanding_controls_native_build_and_cli_reference(self):
        from oil_audit.dataset import jsonl_bytes, sha256
        from oil_eval.inventory_features import INVENTORY_FEATURE_COLUMNS
        from oil_eval.runner import INVENTORY_TREE_REPORT, main
        self.assertEqual(INVENTORY_TREE_REPORT, Path("reports/inventory_tree_comparison.json"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "regularization-original.json"
            with regularization_case(root, window=True) as case, \
                 patch("oil_eval.runner.INVENTORY_TREE_REPORT", root / "tree.json"):
                result = run(root, root / "runs", experiment="inventory-tree-comparison",
                             regularization_reference_report=original)
                mocks = case["mocks"]
                mocks["load_development_data"].assert_called_once_with(root)
                mocks["attach_inventory_features"].assert_called_once_with([case["admitted"]], [], [])
                payload = jsonl_bytes([{**case["admitted"], "features": dict.fromkeys(INVENTORY_FEATURE_COLUMNS),
                                        "provenance": case["augmented"]["inventory_provenance"]}])
                feature_identity = {"sha256": sha256(payload), "bytes": len(payload)}
                args = mocks["load_inventory_regularization_reference"].call_args.args
                self.assertEqual(args[:2], (original, case["source"]))
                self.assertEqual(args[3:], (case["wti_reference"], case["audit"], feature_identity,
                                           case["inventory_reference"]))
                for name in ("load_reference", "load_inventory_comparison_reference", "load_inventory_regularization_reference"):
                    self.assertEqual(mocks[name].call_args.kwargs, {"native_runtime": case["native_runtime"]})
                mocks["replay_development"].assert_called_once_with(
                    [case["augmented"], case["crossing"]], [], [], experiment="inventory-tree-comparison",
                    reference_rows=case["wti_rows"], inventory_reference_rows=case["inventory_rows"],
                    regularization_reference_rows=case["regularization_rows"])
                self.assertEqual(result["regularization_reference_run_id"], "original-regularization")
                self.assertEqual(json.loads((root / "tree.json").read_bytes()), result)
                destination = Path(result["evaluation_directory"])
                manifest = json.loads((destination / "manifest.json").read_bytes())
                self.assertEqual(manifest["inventory_regularization_reference"], case["regularization_reference"])
                self.assertEqual(manifest["inventory_comparison_reference"], case["inventory_reference"])
                self.assertEqual(manifest["inventory_features"], feature_identity)
                self.assertEqual((destination / "inventory_features.jsonl").read_bytes(), payload)
        output = {"run_id": "synthetic", "evaluation_directory": "data/synthetic", "prediction_count": 0,
                  "scored_count": 0, "unscored_count": 0, "metrics": {}, "holdout_evaluated": False}
        for flags, expected in (([], Path("reports/inventory_regularization.json")),
                                (["--regularization-reference-report", "data/custom-c01.json"], Path("data/custom-c01.json"))):
            argv = ["oil_eval", "--dataset-dir", "data/synthetic", "--experiment", "inventory-tree-comparison", *flags]
            with self.subTest(flags=flags), patch.object(sys, "argv", argv), \
                 patch("oil_eval.runner.run", return_value=output) as execute, patch("builtins.print"):
                main()
            self.assertEqual(execute.call_args.kwargs["experiment"], "inventory-tree-comparison")
            self.assertEqual(execute.call_args.kwargs["regularization_reference_report"], expected)

    def test_tree_comparison_missing_reference_stops_before_replay_or_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("load_reference", "load_inventory_reference", "load_inventory_comparison_reference",
                         "load_inventory_regularization_reference"):
                with self.subTest(reference=name), regularization_case(root, window=True) as case:
                    case["mocks"][name].side_effect = FileNotFoundError("Missing synthetic reference")
                    with self.assertRaisesRegex(FileNotFoundError, "Missing synthetic reference"):
                        run(root, root / "runs", root / "tree.json", experiment="inventory-tree-comparison")
                    case["mocks"]["replay_development"].assert_not_called()
                    self.assertFalse((root / "runs").exists())
                    self.assertFalse((root / "tree.json").exists())

    def test_tree_comparison_protects_historical_reports_controls_and_old_tree_artifacts(self):
        historical_reports = (
            "brent_comparison.json", "dataset_summary.json", "development_benchmarks.json",
            "development_evaluation.json", "inventory_audit.json", "inventory_comparison.json",
            "inventory_regularization.json", "inventory_window_comparison.json", "price_audit.json",
            "regularization_runtime_check.json", "source_reconciliation.json", "vintage_audit.json",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original_report = root / "regularization-original.json"
            old_tree = root / "old-tree-output"
            old_tree.mkdir()
            names = ("manifest.json", "summary.json", "predictions.jsonl", "inventory_features.jsonl")
            for name in names:
                (old_tree / name).write_bytes(b"immutable synthetic tree artifact\n")
            before = {path: path.read_bytes() for path in old_tree.iterdir()}
            with regularization_case(root, window=True) as case:
                # Stop before any write if a historical-path guard regresses.
                case["mocks"]["replay_development"].side_effect = AssertionError("Protected destination reached replay")
                destinations = [Path("reports") / name for name in historical_reports]
                destinations += [original_report, root / "audit" / "records.jsonl"]
                destinations += [root / subdirectory / name for subdirectory in
                                 ("regularization-reference", "inventory-reference", "runs/older-tree", "old-tree-output")
                                 for name in names]
                for report in destinations:
                    with self.subTest(report=str(report)), self.assertRaisesRegex(ValueError, "overwrite"):
                        run(root, root / "runs", report, experiment="inventory-tree-comparison",
                            regularization_reference_report=original_report)
                case["mocks"]["replay_development"].assert_not_called()
            self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_window_comparison_routes_all_three_control_references_and_shared_frozen_features(self):
        from oil_audit.dataset import jsonl_bytes, sha256
        from oil_eval.inventory_features import INVENTORY_FEATURE_COLUMNS
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with regularization_case(root, window=True) as case:
                report = root / "regularization-original.json"
                result = run(root, root / "runs", root / "window.json", experiment="inventory-window-comparison",
                             regularization_reference_report=report)
                mocks = case["mocks"]
                mocks["load_development_data"].assert_called_once_with(root)
                mocks["attach_inventory_features"].assert_called_once_with([case["admitted"]], [], [])
                stored = {**case["admitted"], "features": dict.fromkeys(INVENTORY_FEATURE_COLUMNS),
                          "provenance": case["augmented"]["inventory_provenance"]}
                payload = jsonl_bytes([stored])
                feature_identity = {"sha256": sha256(payload), "bytes": len(payload)}
                args = mocks["load_inventory_regularization_reference"].call_args.args
                self.assertEqual(args[0:2], (report, case["source"]))
                self.assertEqual(args[3:], (case["wti_reference"], case["audit"], feature_identity,
                                           case["inventory_reference"]))
                for name in ("load_reference", "load_inventory_comparison_reference", "load_inventory_regularization_reference"):
                    self.assertEqual(mocks[name].call_args.kwargs, {"native_runtime": case["native_runtime"]})
                mocks["replay_development"].assert_called_once_with(
                    [case["augmented"], case["crossing"]], [], [], experiment="inventory-window-comparison",
                    reference_rows=case["wti_rows"], inventory_reference_rows=case["inventory_rows"],
                    regularization_reference_rows=case["regularization_rows"])
                self.assertEqual(result["regularization_reference_run_id"], "original-regularization")
                self.assertEqual(result["inventory_reference_run_id"], "original-inventory")
                self.assertEqual(result["reference_run_id"], "original-wti")
                destination = Path(result["evaluation_directory"])
                manifest = json.loads((destination / "manifest.json").read_bytes())
                self.assertEqual(manifest["inventory_regularization_reference"], case["regularization_reference"])
                self.assertEqual(manifest["inventory_comparison_reference"], case["inventory_reference"])
                self.assertEqual(manifest["inventory_features"], feature_identity)
                self.assertEqual((destination / "inventory_features.jsonl").read_bytes(), payload)

    def test_window_comparison_each_authentication_failure_prevents_replay_and_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for failure in ("load_reference", "load_inventory_reference", "load_inventory_comparison_reference",
                            "load_inventory_regularization_reference"):
                with self.subTest(failure=failure), regularization_case(root, window=True) as case:
                    case["mocks"][failure].side_effect = ValueError("Synthetic reference authentication failure")
                    with self.assertRaisesRegex(ValueError, "authentication failure"):
                        run(root, root / "runs", root / "window.json", experiment="inventory-window-comparison")
                    case["mocks"]["replay_development"].assert_not_called()
                    self.assertFalse((root / "runs").exists())
                    self.assertFalse((root / "window.json").exists())

    def test_window_comparison_protects_regularization_report_all_artifacts_and_prior_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original_report = root / "regularization-original.json"
            with regularization_case(root, window=True) as case:
                case["mocks"]["replay_development"].side_effect = AssertionError("Protected destination reached replay")
                destinations = [original_report, Path("reports/inventory_regularization.json"),
                                Path("reports/inventory_comparison.json"), Path("reports/development_evaluation.json"),
                                Path("reports/inventory_audit.json"), root / "audit" / "records.jsonl",
                                root / "runs" / "older" / "predictions.jsonl"]
                destinations += [root / subdirectory / name
                                 for subdirectory in ("regularization-reference", "inventory-reference")
                                 for name in ("manifest.json", "summary.json", "predictions.jsonl", "inventory_features.jsonl")]
                for report in destinations:
                    with self.subTest(report=str(report)), self.assertRaisesRegex(ValueError, "overwrite"):
                        run(root, root / "runs", report, experiment="inventory-window-comparison",
                            regularization_reference_report=original_report)
                case["mocks"]["replay_development"].assert_not_called()

    def test_window_cli_routes_default_and_explicit_regularization_reference(self):
        from oil_eval.runner import main
        output = {"run_id": "synthetic", "evaluation_directory": "data/synthetic", "prediction_count": 0,
                  "scored_count": 0, "unscored_count": 0, "metrics": {}, "holdout_evaluated": False}
        for arguments, expected in (([], Path("reports/inventory_regularization.json")),
                                    (["--regularization-reference-report", "data/custom-c01.json"], Path("data/custom-c01.json"))):
            argv = ["oil_eval", "--dataset-dir", "data/synthetic", "--experiment", "inventory-window-comparison", *arguments]
            with self.subTest(arguments=arguments), patch.object(sys, "argv", argv), \
                 patch("oil_eval.runner.run", return_value=output) as execute, patch("builtins.print"):
                main()
            self.assertEqual(execute.call_args.kwargs["experiment"], "inventory-window-comparison")
            self.assertEqual(execute.call_args.kwargs["regularization_reference_report"], expected)

    def test_native_build_change_gets_new_run_identity_without_replacing_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_file = root / "source-manifest.json"
            source_file.write_bytes(b'{"source":"synthetic immutable input"}\n')
            source = {"dataset_id": "synthetic", "manifest_sha256": "c" * 64}
            native_a = {"scipy": {"scipy/example.so": "a" * 64}}
            native_b = {"scipy": {"scipy/example.so": "b" * 64}}
            with patch("oil_eval.runner.load_development_data", return_value=(source, [], [], [])), \
                 patch("oil_eval.runner.version", return_value="same-version"), \
                 patch("oil_eval.runner.platform.python_version", return_value="same-python"), \
                 patch("oil_eval.runner._native_runtime_sha256", side_effect=[native_a, native_b, native_a]), \
                 patch("oil_eval.runner.replay_development", return_value=([], {"holdout_evaluated": False})):
                first = run(root, root / "runs", root / "original.json")
                first_directory = Path(first["evaluation_directory"])
                preserved = {path: path.read_bytes() for path in
                             [source_file, root / "original.json", *first_directory.iterdir()]}
                second = run(root, root / "runs", root / "different-build.json")
                repeated = run(root, root / "runs", root / "repeated-build.json")
            self.assertNotEqual(first["run_id"], second["run_id"])
            self.assertEqual(first["run_id"], repeated["run_id"])
            first_manifest = json.loads((first_directory / "manifest.json").read_bytes())
            second_manifest = json.loads((Path(second["evaluation_directory"]) / "manifest.json").read_bytes())
            self.assertEqual(first_manifest["native_runtime_sha256"], native_a)
            self.assertEqual(second_manifest["native_runtime_sha256"], native_b)
            self.assertEqual({key: value for key, value in first_manifest.items()
                              if key not in ("native_runtime_sha256", "run_id")},
                             {key: value for key, value in second_manifest.items()
                              if key not in ("native_runtime_sha256", "run_id")})
            self.assertEqual(preserved, {path: path.read_bytes() for path in preserved})

    def test_regularization_authenticates_both_references_and_freezes_only_admitted_rows(self):
        from oil_audit.dataset import jsonl_bytes, sha256
        from oil_eval.inventory_features import INVENTORY_FEATURE_COLUMNS
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with regularization_case(root) as case:
                wti_report, inventory_report, audit_report = (root / "wti.json", root / "inventory-original.json", root / "audit.json")
                result = run(root, root / "runs", root / "regularization.json", experiment="inventory-regularization",
                             reference_report=wti_report, inventory_reference_report=inventory_report,
                             inventory_audit_report=audit_report)
                mocks = case["mocks"]
                mocks["load_development_data"].assert_called_once_with(root)
                mocks["load_inventory_reference"].assert_called_once_with(audit_report)
                self.assertEqual(mocks["load_reference"].call_args.kwargs,
                                 {"native_runtime": case["native_runtime"]})
                mocks["attach_inventory_features"].assert_called_once_with([case["admitted"]], [], [])
                stored = {**case["admitted"], "features": dict.fromkeys(INVENTORY_FEATURE_COLUMNS),
                          "provenance": case["augmented"]["inventory_provenance"]}
                payload = jsonl_bytes([stored])
                args = mocks["load_inventory_comparison_reference"].call_args.args
                self.assertEqual(args[0:2], (inventory_report, case["source"]))
                self.assertEqual(args[3:], (case["wti_reference"], case["audit"], {"sha256": sha256(payload), "bytes": len(payload)}))
                self.assertEqual(mocks["load_inventory_comparison_reference"].call_args.kwargs,
                                 {"native_runtime": case["native_runtime"]})
                mocks["replay_development"].assert_called_once_with(
                    [case["augmented"], case["crossing"]], [], [], experiment="inventory-regularization",
                    reference_rows=case["wti_rows"], inventory_reference_rows=case["inventory_rows"])
                self.assertEqual(result["inventory_reference_run_id"], "original-inventory")
                self.assertEqual(result["reference_run_id"], "original-wti")
                self.assertEqual(result["inventory_audit_id"], "original-audit")
                destination = Path(result["evaluation_directory"])
                self.assertEqual((destination / "inventory_features.jsonl").read_bytes(), payload)
                manifest = json.loads((destination / "manifest.json").read_bytes())
                self.assertEqual(manifest["inventory_comparison_reference"], case["inventory_reference"])
                self.assertEqual(manifest["inventory_reference"], case["audit"])
                self.assertEqual(manifest["inventory_features"], {"sha256": sha256(payload), "bytes": len(payload)})

    def test_regularization_reference_or_audit_authentication_failure_stops_before_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for failure in ("load_reference", "load_inventory_reference", "load_inventory_comparison_reference"):
                with self.subTest(failure=failure), regularization_case(root) as case:
                    case["mocks"][failure].side_effect = ValueError("Synthetic authentication checksum failure")
                    with self.assertRaisesRegex(ValueError, "authentication checksum"):
                        run(root, root / "runs", root / "regularization.json", experiment="inventory-regularization")
                    case["mocks"]["replay_development"].assert_not_called()
                    self.assertFalse((root / "regularization.json").exists())
                    self.assertFalse((root / "runs").exists())

    def test_regularization_audit_wti_mismatch_stops_before_feature_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with regularization_case(root) as case:
                case["audit"]["reference_run"] = {"run_id": "different-wti"}
                with self.assertRaisesRegex(ValueError, "do not share"):
                    run(root, root / "runs", root / "regularization.json", experiment="inventory-regularization")
                case["mocks"]["attach_inventory_features"].assert_not_called()
                case["mocks"]["load_inventory_comparison_reference"].assert_not_called()
                case["mocks"]["replay_development"].assert_not_called()

    def test_regularization_protects_both_original_reports_audit_and_every_inventory_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original_report = root / "inventory-original.json"
            with regularization_case(root) as case:
                # If a protection regresses, stop before the runner can write a tracked report.
                case["mocks"]["replay_development"].side_effect = AssertionError("Protected destination reached replay")
                destinations = [original_report, Path("reports/inventory_comparison.json"),
                                Path("reports/development_evaluation.json"), Path("reports/brent_comparison.json"),
                                Path("reports/development_benchmarks.json"), Path("reports/inventory_audit.json"),
                                root / "audit" / "records.jsonl", root / "runs" / "older" / "inventory_features.jsonl"]
                destinations += [root / "inventory-reference" / name for name in
                                 ("manifest.json", "summary.json", "predictions.jsonl", "inventory_features.jsonl")]
                for report in destinations:
                    with self.subTest(report=str(report)), self.assertRaisesRegex(ValueError, "overwrite"):
                        run(root, root / "runs", report, experiment="inventory-regularization",
                            inventory_reference_report=original_report)
                case["mocks"]["replay_development"].assert_not_called()

    def test_regularization_cli_routes_default_and_explicit_inventory_reference(self):
        from oil_eval.runner import main
        output = {"run_id": "synthetic", "evaluation_directory": "data/synthetic", "prediction_count": 0,
                  "scored_count": 0, "unscored_count": 0, "metrics": {}, "holdout_evaluated": False}
        for arguments, expected in (([], Path("reports/inventory_comparison.json")),
                                    (["--inventory-reference-report", "data/custom-original.json"], Path("data/custom-original.json"))):
            argv = ["oil_eval", "--dataset-dir", "data/synthetic", "--experiment", "inventory-regularization", *arguments]
            with self.subTest(arguments=arguments), patch.object(sys, "argv", argv), \
                 patch("oil_eval.runner.run", return_value=output) as execute, patch("builtins.print"):
                main()
            self.assertEqual(execute.call_args.kwargs["experiment"], "inventory-regularization")
            self.assertEqual(execute.call_args.kwargs["inventory_reference_report"], expected)

    def test_regularization_cli_has_no_parameter_sweep_option(self):
        result = subprocess.run([sys.executable, "-m", "oil_eval", "--dataset-dir", "data/unused",
                                 "--experiment", "inventory-regularization", "--C", "10"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("unrecognized arguments: --C 10", result.stderr)

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
