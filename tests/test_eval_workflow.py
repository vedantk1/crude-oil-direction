"""Fresh-root orchestration tests use synthetic captures and mocked stage outputs."""

from contextlib import contextmanager
from datetime import date
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from oil_audit.dataset import json_bytes
from oil_eval.workflow import capture_and_run, main, run


@contextmanager
def stages(root):
    vintage, inventory = root / "vintages", root / "inventory"
    vintage.mkdir()
    inventory.mkdir()
    (vintage / "manifest.json").write_bytes(json_bytes([{"path": str(vintage / "synthetic-test-input")}]))
    (inventory / "manifest.json").write_bytes(json_bytes({"synthetic": False}))

    def save(report, result):
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_bytes(json_bytes(result))
        return result

    def build(data_dir, output_root, *, report, maximum_as_of):
        destination = output_root / "discovered-dataset"
        destination.mkdir(parents=True)
        (destination / "manifest.json").write_bytes(json_bytes({"dataset_id": "discovered-dataset"}))
        return save(report, {"dataset_directory": str(destination)})

    def evaluate(dataset_dir, output_root, report, *, experiment="wti", **options):
        return save(report, {"run_id": "new-" + experiment, "evaluation_directory": str(output_root / experiment),
                             "dataset_id": "discovered-dataset", "holdout_evaluated": False})

    def audit(snapshot, output_root, report, **options):
        return save(report, {"audit_id": "new-audit", "audit_directory": str(output_root / "new-audit")})

    def benchmarks(wti_report, brent_report, output_root, report):
        return save(report, {"run_id": "new-benchmarks", "benchmark_directory": str(output_root / "new-benchmarks")})

    with patch("oil_eval.workflow.build", side_effect=build) as builder, \
         patch("oil_eval.workflow.evaluate", side_effect=evaluate) as evaluator, \
         patch("oil_eval.workflow.audit_inventory", side_effect=audit) as auditor, \
         patch("oil_eval.workflow.run_benchmarks", side_effect=benchmarks) as benchmarker:
        yield vintage, inventory, builder, evaluator, auditor, benchmarker


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".test-workflow-", dir=".")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).relative_to(Path.cwd())

    def test_discovers_new_ids_and_routes_all_references_to_separate_fresh_reports(self):
        with stages(self.root) as (vintage, inventory, builder, evaluator, auditor, benchmarker):
            output = self.root / "new-run"
            before = {path: path.read_bytes() for path in (vintage / "manifest.json", inventory / "manifest.json")}
            result = run(vintage, inventory, output, brent=True, benchmarks=True, window=True, tree=True)
            reports = output / "reports"
            builder.assert_called_once_with(vintage, output / "datasets", report=reports / "dataset.json",
                                            maximum_as_of=date(2022, 12, 31))
            self.assertEqual(result["dataset_directory"], str(output / "datasets/discovered-dataset"))
            self.assertEqual([call.kwargs.get("experiment", "wti") for call in evaluator.call_args_list],
                             ["wti", "inventory-comparison", "inventory-regularization", "brent-comparison",
                              "inventory-window-comparison", "inventory-tree-comparison"])
            for call in evaluator.call_args_list:
                self.assertEqual(call.args[:2], (Path(result["dataset_directory"]), output / "evaluations"))
                if call.kwargs.get("experiment", "wti") != "wti":
                    self.assertEqual(call.kwargs["reference_report"], reports / "wti.json")
                if call.kwargs.get("experiment", "").startswith("inventory-"):
                    self.assertEqual(call.kwargs["inventory_audit_report"], reports / "inventory_audit.json")
                    self.assertEqual(call.kwargs["inventory_reference_report"], reports / "inventory_comparison.json")
                    self.assertEqual(call.kwargs["regularization_reference_report"], reports / "inventory_regularization.json")
            self.assertEqual(auditor.call_args.kwargs["reference_report"], reports / "wti.json")
            benchmarker.assert_called_once_with(reports / "wti.json", reports / "brent_comparison.json",
                                                output / "evaluations", reports / "benchmarks.json")
            self.assertEqual(result, json.loads((output / "workflow.json").read_bytes()))
            self.assertEqual(result["status"], "complete")
            self.assertFalse(result["holdout_evaluated"])
            self.assertEqual(len(result["reports"]), 9)
            self.assertEqual(result["stages"]["dataset"]["dataset_id"], "discovered-dataset")
            self.assertEqual({path: path.read_bytes() for path in before}, before)
            self.assertFalse((output / "workflow.tmp").exists())

    def test_failure_preserves_completed_stages_and_retry_requires_a_new_root(self):
        with stages(self.root) as (vintage, inventory, builder, evaluator, auditor, _):
            auditor.side_effect = ValueError("Invalid source evidence")
            output = self.root / "failed-run"
            with self.assertRaisesRegex(ValueError, "Invalid source evidence"):
                run(vintage, inventory, output)
            checkpoint = json.loads((output / "workflow.json").read_bytes())
            self.assertEqual(checkpoint["status"], "failed")
            self.assertEqual(checkpoint["error_type"], "ValueError")
            self.assertEqual(checkpoint["active_stage"], "inventory_audit")
            self.assertEqual(set(checkpoint["stages"]), {"dataset", "wti"})
            original = (output / "reports/wti.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "must be new"):
                run(vintage, inventory, output)
            self.assertEqual((output / "reports/wti.json").read_bytes(), original)
            self.assertEqual(builder.call_count, 1)
            self.assertEqual(evaluator.call_count, 1)

    def test_output_and_report_guards_fail_before_builder(self):
        with stages(self.root) as (vintage, inventory, builder, *_):
            existing = self.root / "existing"
            existing.mkdir()
            (existing / "manifest.json").write_text("retained")
            for output, report in (
                (existing, None), (existing / "nested", None),
                (Path("data/evaluations/workflow-test"), None),
                (Path("reports/workflow-test"), None),
                (self.root / "fresh-a", self.root / "fresh-a/evaluations/reports"),
                (self.root / "fresh-b", Path("reports/fresh-workflow-test")),
                (self.root / "fresh-c", self.root / "fresh-c"),
                (self.root / "fresh-d", existing),
            ):
                with self.subTest(output=output, report=report), self.assertRaises(ValueError):
                    run(vintage, inventory, output, report_dir=report)
            builder.assert_not_called()
            self.assertEqual((existing / "manifest.json").read_text(), "retained")

    def test_optional_evidence_passes_through_and_synthetic_provenance_cannot_be_mislabelled(self):
        with stages(self.root) as (vintage, inventory, builder, _, auditor, _):
            evidence = {name: self.root / (name + ".json") for name in
                        ("evidence_manifest", "probe_manifest", "quarantine_manifest")}
            for path in evidence.values():
                path.write_text("{}")
            run(vintage, inventory, self.root / "with-evidence", **evidence)
            for key, path in evidence.items():
                self.assertEqual(auditor.call_args.kwargs[key], path)
            (vintage / "manifest.json").write_bytes(json_bytes([{"path": str(vintage / "fixture"), "synthetic": True}]))
            with self.assertRaisesRegex(ValueError, "consistently"):
                run(vintage, inventory, self.root / "mixed")
            (inventory / "manifest.json").write_bytes(json_bytes({"synthetic": True}))
            with self.assertRaisesRegex(ValueError, "kind differs"):
                run(vintage, inventory, self.root / "wrong-kind", synthetic=False)
            result = run(vintage, inventory, self.root / "detected")
            self.assertEqual(result["kind"], "synthetic_demo")
            self.assertEqual(builder.call_count, 2)

    def test_manifest_cannot_redirect_source_reads_outside_its_declared_capture(self):
        with stages(self.root) as (vintage, inventory, builder, *_):
            outside = self.root / "unrelated-source"
            outside.write_text("must remain outside the capture")
            (vintage / "manifest.json").write_bytes(json_bytes([{"path": str(outside)}]))
            with self.assertRaisesRegex(ValueError, "outside its supplied capture"):
                run(vintage, inventory, self.root / "redirected")
            builder.assert_not_called()
            self.assertFalse((self.root / "redirected").exists())

    def test_download_is_explicit_discovers_source_ids_and_uses_capture_only_subprocess(self):
        output = self.root / "capture"

        def download(command, *, check):
            self.assertTrue(check)
            self.assertEqual(command[1:5], ["-m", "oil_audit.vintages", "--download", "--capture-only"])
            root = Path(command[command.index("--data-dir") + 1]) / "new-capture-id"
            root.mkdir(parents=True)
            (root / "manifest.json").write_text("[]")

        def inventory(output_root, *, max_workers):
            self.assertEqual(max_workers, 2)
            root = output_root / "other-capture-id"
            root.mkdir(parents=True)
            path = root / "manifest.json"
            path.write_text("{}")
            return path

        with patch("oil_eval.workflow.subprocess.run", side_effect=download) as downloader, \
             patch("oil_eval.workflow.capture_inventories", side_effect=inventory), \
             patch("oil_eval.workflow.run", return_value={"status": "complete"}) as replay:
            capture_and_run(output, max_workers=2, brent=True)
            replay.assert_called_once_with(output / "sources/vintages/new-capture-id",
                                            output / "sources/inventories/other-capture-id", output / "replay", brent=True)
            state = json.loads((output / "capture_workflow.json").read_bytes())
            self.assertEqual(state["status"], "complete")
            self.assertFalse(state["holdout_evaluated"])
            with self.assertRaisesRegex(ValueError, "must be new"):
                capture_and_run(output)
            self.assertEqual(downloader.call_count, 1)

    def test_cli_never_downloads_implicitly_or_accepts_ambiguous_sources(self):
        with patch("oil_eval.workflow.run") as replay, patch("oil_eval.workflow.capture_and_run") as download:
            for argv in (
                ["workflow", "--run-root", "data/new"],
                ["workflow", "--run-root", "data/new", "--download", "--vintage-dir", "data/source"],
            ):
                with patch("sys.argv", argv), patch("sys.stderr"), self.assertRaises(SystemExit):
                    main()
            replay.assert_not_called()
            download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
