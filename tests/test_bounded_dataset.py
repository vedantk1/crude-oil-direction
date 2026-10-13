from datetime import date
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from oil_audit.dataset import build, read_dataset
from oil_audit import vintages


class BoundedBuildTests(unittest.TestCase):
    def sources(self, root, capture="2023-01-02T00:00:00+00:00"):
        (root / "manifest.json").write_text(json.dumps([
            {"path": series + "-realtime-part1.zip", "retrieved_at_utc": capture}
            for series in vintages.SERIES]))

    def records(self, series, _, maximum):
        self.assertEqual(maximum, date(2022, 12, 31))
        return [{"period_start_date": "2022-12-28", "realtime_start_date": "2022-12-29",
                 "realtime_end_date": "", series: "70"}]

    def test_supported_end_is_capture_covered_bound_even_if_no_issuance_on_that_day(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.sources(root)
            with patch("oil_audit.dataset.vintages.verify_manifest"), \
                 patch("oil_audit.dataset.vintages.form_metadata", return_value=(
                     ["2022-12-29", "2023-01-03"], [])), \
                 patch("oil_audit.dataset.vintages.records", side_effect=AssertionError("Unbounded decode")), \
                 patch("oil_audit.bounded_vintages.bounded_records", side_effect=self.records):
                result = build(root, root / "outputs", report=root / "report.json",
                               maximum_as_of=date(2022, 12, 31))
            manifest, data = read_dataset(Path(result["dataset_directory"]))
            self.assertEqual(manifest["config"]["maximum_supported_as_of_date"], "2022-12-31")
            self.assertEqual([row["forecast_date"] for row in data["features.jsonl"]], ["2022-12-30"])

    def test_capture_bound_failure_precedes_any_source_value_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.sources(root, "2022-12-31T22:59:59+00:00")
            with patch("oil_audit.dataset.vintages.verify_manifest"), \
                 patch("oil_audit.dataset.vintages.records", side_effect=AssertionError("Unbounded decode")), \
                 patch("oil_audit.bounded_vintages.bounded_records", side_effect=AssertionError("Bounded decode")):
                with self.assertRaisesRegex(ValueError, "exceeds captured history"):
                    build(root, root / "outputs", maximum_as_of=date(2022, 12, 31))

    def test_capture_only_verifies_bytes_without_running_numeric_audit(self):
        with patch("sys.argv", ["vintages", "--download", "--capture-only"]), \
             patch.object(vintages, "ROOT", Path("data/test-capture")), \
             patch.object(vintages, "download") as capture, \
             patch.object(vintages, "verify_manifest") as verify, \
             patch.object(vintages, "audit", side_effect=AssertionError("Numeric audit")), \
             patch("builtins.print"):
            vintages.main()
            capture.assert_called_once()
            verify.assert_called_once()

    def test_capture_only_requires_explicit_download(self):
        with patch("sys.argv", ["vintages", "--capture-only"]), patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                vintages.main()


if __name__ == "__main__":
    unittest.main()
