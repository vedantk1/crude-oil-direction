"""Offline synthetic end-to-end checks; never read historical project data."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from oil_audit.inventory import load_snapshot
from oil_audit.vintages import form_metadata, verify_manifest, zip_rows
from oil_eval.demo import generate_sources, run


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".test-demo-", dir=".")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).relative_to(Path.cwd())

    def test_generated_sources_are_repeatable_typed_and_explicitly_invented(self):
        first = generate_sources(self.root / "first")
        second = generate_sources(self.root / "second")
        for source in (first, second):
            self.assertTrue(source["synthetic"])
            verify_manifest(Path(source["vintage_dir"]))
            metadata, issues, records, failures, _ = load_snapshot(Path(source["inventory_snapshot"]))
            self.assertEqual(len(issues), len(records))
            self.assertFalse(failures)
            self.assertEqual({issue["release_date"][:4] for issue in issues}, {str(year) for year in range(2011, 2023)})
            self.assertIn("2011_07_27", metadata["additional_unindexed_issue_ids"])
            for series in ("DCOILWTICO", "DCOILBRENTEU"):
                vintage_dir = Path(source["vintage_dir"])
                listed, _ = form_metadata(series, vintage_dir)
                self.assertEqual(listed[-1], "2023-01-04")
                rows = zip_rows(vintage_dir / (series + "-realtime-part1.zip"))
                self.assertEqual(sum(row[series] == "FUTURE_VALUE_MUST_NOT_BE_DECODED" for row in rows), 2)
                self.assertTrue(all(row.get("synthetic") is True
                                    for row in json.loads((vintage_dir / "manifest.json").read_bytes())))
        for path in (self.root / "first").rglob("*"):
            if path.is_file() and path.name not in ("manifest.json", "synthetic_sources.json"):
                self.assertEqual(path.read_bytes(), (self.root / "second" / path.relative_to(self.root / "first")).read_bytes())
        with self.assertRaisesRegex(ValueError, "must be new"):
            generate_sources(self.root / "first")

    def test_real_offline_pipeline_authenticates_new_reference_chain_and_keeps_holdout_closed(self):
        output = self.root / "end-to-end"
        with patch("urllib.request.urlopen", side_effect=AssertionError("Demo attempted network access")), \
             patch("oil_audit.inventory_sources.urlopen", side_effect=AssertionError("Demo attempted network access")):
            result = run(output, tree=True)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["kind"], "synthetic_demo")
        self.assertFalse(result["holdout_evaluated"])
        reports = {name: json.loads(Path(path).read_bytes()) for name, path in result["reports"].items()}
        self.assertEqual(set(reports), {"dataset", "wti", "inventory_audit", "inventory_comparison",
                                       "inventory_regularization", "brent_comparison", "benchmarks", "inventory_tree_comparison"})
        self.assertEqual(reports["dataset"]["config"]["maximum_supported_as_of_date"], "2022-12-31")
        for name in ("wti", "inventory_comparison", "inventory_regularization", "brent_comparison", "inventory_tree_comparison"):
            report = reports[name]
            self.assertFalse(report["holdout_evaluated"])
            self.assertEqual(report["prediction_count"], 7)
            self.assertGreaterEqual(report["training_count_range"][0], 150)
            self.assertEqual(set(report["yearly"]), {str(year) for year in range(2017, 2023)})
            predictions = [json.loads(line) for line in (Path(report["evaluation_directory"]) / "predictions.jsonl").read_bytes().splitlines()]
            self.assertTrue(all(row["forecast_date"] < "2023-01-01" and row["target_date"] < "2023-01-01" for row in predictions))
            if name != "wti":
                self.assertEqual(report["reference_run_id"], reports["wti"]["run_id"])
        for name in ("inventory_regularization", "inventory_tree_comparison"):
            self.assertEqual(reports[name]["inventory_reference_run_id"], reports["inventory_comparison"]["run_id"])
            self.assertTrue(reports[name]["inventory_reference_reproduction"]["all_control_model_outputs_identical"])
        self.assertEqual(reports["inventory_tree_comparison"]["regularization_reference_run_id"], reports["inventory_regularization"]["run_id"])
        self.assertTrue(reports["inventory_tree_comparison"]["regularization_reference_reproduction"]["all_expanding_c01_model_outputs_identical"])
        dataset = Path(result["dataset_directory"])
        for name in ("features.jsonl", "label_events.jsonl", "evaluation_labels.jsonl"):
            self.assertNotIn(b"FUTURE_VALUE", (dataset / name).read_bytes())
        old_manifest = (output / "replay/workflow.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "must be new"):
            run(output)
        self.assertEqual((output / "replay/workflow.json").read_bytes(), old_manifest)


if __name__ == "__main__":
    unittest.main()
